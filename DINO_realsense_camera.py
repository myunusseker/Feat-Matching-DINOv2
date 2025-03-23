import copy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import rospy
import rostopic
import actionlib
import yk_msgs.msg
from sensor_msgs.msg import Image as ROSImage
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped

from PIL import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
from numpy.linalg import norm

import torch
import torchvision.transforms as transforms

RGB_IMAGE_TOPIC = '/yk_builder/camera/color/image_raw'
DEPTH_IMAGE_TOPIC = '/yk_builder/camera/aligned_depth_to_color/image_raw'
RGB_INFO_TOPIC = '/yk_builder/camera/color/camera_info'

T_cam_eef = np.array(
    [[-0.01722443, -0.99952439, -0.02557959, -0.06725155],
     [ 0.9998478,  -0.01714775, -0.00321424, -0.01221117],
     [ 0.00277408, -0.02563106,  0.99966762,  0.07440511],
     [ 0.,          0.,          0.,          1.        ]]
)
def get_transformation_matrix_from_pose(pose):
    T = np.eye(4)
    rotation = R.from_quat(pose['orientation']).as_matrix()
    T[:3, :3] = rotation
    T[:3, 3] = pose['position']
    return T

def get_pose_from_transformation_matrix(T):
    """
    Convert a 4x4 transformation matrix into a pose dictionary with position and quaternion orientation.
    """
    position = T[:3, 3]  # Extract translation (x, y, z)
    rotation = R.from_matrix(T[:3, :3])  # Extract rotation matrix
    orientation = rotation.as_quat()  # Convert rotation to quaternion

    return {"position": position, "orientation": orientation}

def deproject_pixel(depth_data:np.ndarray, pixel_x:int, pixel_y:int, K:np.ndarray):
    K = np.reshape(K, (3, 3))
    pixel_data = [pixel_y, pixel_x]
    depth = depth_data[pixel_x, pixel_y]

    point_3d = np.linalg.inv(K).dot(np.r_[pixel_data, 1.0])

    return depth*point_3d

def pose_callback(msg):
    """Callback function to update tool0 pose."""
    global robot_pose
    robot_pose = msg.pose  # Store the pose from PoseStamped


class Dinov2Matcher:

    def __init__(self, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14", smaller_edge_size=448,
                 patch_size=14, device="cuda", source_image_path='water_bottle.jpeg', source_patch=(2, 16)):
        self.repo_name = repo_name
        self.model_name = model_name
        self.smaller_edge_size = smaller_edge_size
        self.patch_size = patch_size
        self.device = device    

        self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)
        self.model.eval()
        

        self.transform = transforms.Compose([
            transforms.Resize(size=smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC,
                              antialias=True),
            transforms.ToTensor(),
        ])

        self.ref_img_name = source_image_path
        self.ref_patch = source_patch


        # Prepare reference image
        self.ref_img = cv2.cvtColor(cv2.imread(self.ref_img_name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # Extract features for the reference image
        self.ref_img_tensor, self.ref_grid, self.ref_scale = self.prepare_image(self.ref_img)
        self.ref_features = self.extract_features(self.ref_img_tensor).reshape(*self.ref_grid, -1)
        self.ref_norm = norm(self.ref_features[self.ref_patch])

    def prepare_image(self, rgb_image_numpy):
        image = Image.fromarray(rgb_image_numpy)
        image_tensor = self.transform(image)
        resize_scale = image.width / image_tensor.shape[2]

        height, width = image_tensor.shape[1:]  # C x H x W
        cropped_width, cropped_height = width - width % self.patch_size, height - height % self.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        grid_size = (cropped_height // self.patch_size, cropped_width // self.patch_size)
        return image_tensor, grid_size, resize_scale

    def extract_features(self, image_tensor):
        with torch.inference_mode():
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
        return tokens.cpu().numpy()

    def calculate_heatmap(self, target_img):
        # Extract features for the target image
        target_img_tensor, target_grid, target_scale = self.prepare_image(target_img)
        target_features = self.extract_features(target_img_tensor).reshape(*target_grid, -1)

        # Calculate Heatmap Scores
        target_norms = norm(target_features, axis=2)
        heatmap_scores = np.tensordot(self.ref_features[self.ref_patch], target_features, axes=([0], [2])) / (self.ref_norm * target_norms)
        heatmap_scores = np.nan_to_num(heatmap_scores)

        # Normalize the heatmap scores
        unnorm_heatmap = heatmap_scores.copy()
        heatmap_scores = (heatmap_scores - heatmap_scores.min()) #/ (heatmap_scores.max() - heatmap_scores.min())

        # Convert heatmap to image using OpenCV
        heatmap_img = cv2.applyColorMap((heatmap_scores * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        heatmap_img = cv2.resize(heatmap_img, (target_img.shape[1], target_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        unnorm_heatmap = cv2.resize(unnorm_heatmap, (target_img.shape[1], target_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        target_heatmap_img = cv2.addWeighted(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB), 0.4, heatmap_img, 0.6, 0)

        return target_heatmap_img, unnorm_heatmap


def find_highest_heatmap_point(heatmap_img):
    i, j = np.unravel_index(np.argmax(heatmap_img), heatmap_img.shape)
    return j, i
    # Convert heatmap image to grayscale in order to find the whitest point
    gray_heatmap = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2GRAY)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray_heatmap)
    max_loc_flatten = heatmap_img.shape[0] * max_loc[0] + max_loc[1]
    return max_loc, max_val, max_loc_flatten

def get_client(action_name, action_msg):
    # Creates the SimpleActionClient, passing the type of the action to the constructor.
    client = actionlib.SimpleActionClient(action_name, action_msg)

    # Waits until the action server has started up and started listening for goals.
    client.wait_for_server()

    return client

def move(pose, velocity=0.07, acceleration=0.07):

    goal = yk_msgs.msg.GoToPoseGoal(
        pose=pose,
        max_velocity_scaling_factor=velocity,
        max_acceleration_scaling_factor=acceleration
    )

    move_client.send_goal(goal)
    move_client.wait_for_result()

    return move_client.get_result()

def get_robot_pose():
    robot_pose_ros = rospy.wait_for_message("/yk_builder/tool0_pose", rostopic.get_topic_class("/yk_builder/tool0_pose")[0],1)
    return robot_pose_ros.pose

def get_camera_image(image_topic, desired_encoding="passthrough"):
    image_ros = rospy.wait_for_message(image_topic, rostopic.get_topic_class(image_topic)[0],1)
    return CvBridge().imgmsg_to_cv2(image_ros, desired_encoding=desired_encoding)

def get_point_from_image(depth_image, point_x, point_y, K):
    point = deproject_pixel(depth_image, point_x, point_y, K) / 1000.
    point = np.array([*point, 1])
    robot_pose = get_robot_pose()
    robot_pose_dict = {
        'position': np.array([robot_pose.position.x, robot_pose.position.y, robot_pose.position.z]),
        'orientation': np.array([robot_pose.orientation.x, robot_pose.orientation.y, robot_pose.orientation.z, robot_pose.orientation.w]),
    }
    T_eef_robot = get_transformation_matrix_from_pose(robot_pose_dict)
    point = T_eef_robot @ T_cam_eef @ point
    return point[:3], (T_eef_robot @ T_cam_eef)[:3, 3]

def main(camera_info):
    # Init Dinov2Matcher
    dm = Dinov2Matcher(repo_name='facebookresearch/dinov2', model_name='dinov2_vitb14', patch_size=14, source_image_path='source_lid.jpg', source_patch=(14, 27))
    try:
        # for i in range(100):
        while True:
            if rospy.is_shutdown():  # Check if ROS is shutting down
                break

            rgb_image_ros = rospy.wait_for_message(
                RGB_IMAGE_TOPIC,
                rostopic.get_topic_class(RGB_IMAGE_TOPIC)[0],
                1
            )
            color_buffer = get_camera_image(RGB_IMAGE_TOPIC, desired_encoding="bgr8")
            depth_buffer = get_camera_image(DEPTH_IMAGE_TOPIC)

            target_img = cv2.cvtColor(color_buffer, cv2.COLOR_BGR2RGB)
            target_heatmap_img, heatmap_scores = dm.calculate_heatmap(target_img)
            vis_img = cv2.cvtColor(target_heatmap_img, cv2.COLOR_BGR2RGB)
            max_loc = find_highest_heatmap_point(heatmap_scores)
            plt.imshow(vis_img)
            plt.show()
            print(max_loc[1], max_loc[0])
            point_3d, camera_3d = get_point_from_image(depth_buffer, max_loc[1], max_loc[0], camera_info.K)
            print("point", point_3d)
            print("camera", camera_3d)
            translation = np.array([0, 0, 0])

            print(heatmap_scores[max_loc[1], max_loc[0]])

            if heatmap_scores[max_loc[1], max_loc[0]] > 0.56:
                translation = np.array([point_3d[0] - camera_3d[0], point_3d[1] - camera_3d[1], point_3d[2] - camera_3d[2]])
                translation[2] += 0.235

            if np.linalg.norm(translation) < 0.01:
                translation=np.zeros(3)

            print("translation", translation)
            target_pose = copy.deepcopy(robot_pose)
            target_pose.position.x += translation[0]
            target_pose.position.y += translation[1]
            target_pose.position.z += translation[2]
            if target_pose.position.z < 0.45:
                target_pose.position.z = 0.45
            print(target_pose)
            #move(target_pose)
            #exit(0)


    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting gracefully...")

if __name__ == "__main__":
    try:
        rospy.init_node("image_listener", anonymous=True)
        msg_type = rostopic.get_topic_class(RGB_INFO_TOPIC)[0]
        camera_info = rospy.wait_for_message(RGB_INFO_TOPIC, msg_type, 1)
        rospy.Subscriber("/yk_builder/tool0_pose", PoseStamped, pose_callback)
        print("Initializing robot move service")
        move_client = get_client('/yk_builder/yk_go_to_pose', yk_msgs.msg.GoToPoseAction)
        base_pose = Pose(
            position=Point(0.0717, -0.3005, 0.5420),
            orientation=Quaternion(1, 0, 0, 0)
        )
        move(base_pose)
        main(camera_info)  # Now properly handles Ctrl+C
    except rospy.ROSInterruptException:
        print("\nGracefully exiting on ROS shutdown.")
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting...")

