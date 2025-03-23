import sys
import numpy as np
import pickle
from numpy.linalg import norm
import open3d as o3d
import matplotlib
matplotlib.use('TkAgg')
import time

import rospy
import actionlib
import sys
import yk_msgs.msg
import robotiq_mm_ros.msg
import cv2
import numpy as np
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose, Point, Quaternion
from geometry_msgs.msg import PoseStamped

from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

bridge = CvBridge()
camera_image = None
robot_pose = None

def image_callback(msg):
        global camera_image
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            # Convert to NumPy array
            camera_image = np.array(cv_image, dtype=np.uint8)
        except CvBridgeError as e:
            print("CvBridge Error:", e)

def pose_callback(msg):
    """Callback function to update tool0 pose."""
    global robot_pose
    robot_pose = msg.pose  # Store the pose from PoseStamped


class Dinov2Matcher:

    def __init__(self, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14", smaller_edge_size=448,
                 patch_size=14, device="cuda", ref_img_name='water_bottle.jpeg', ref_patch=(2, 16)):
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

        self.ref_img_name = ref_img_name
        self.ref_patch = ref_patch

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
        heatmap_scores = (heatmap_scores - heatmap_scores.min()) / (heatmap_scores.max() - heatmap_scores.min())

        # Convert heatmap to image using OpenCV
        heatmap_img = cv2.applyColorMap((heatmap_scores * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        heatmap_img = cv2.resize(heatmap_img, (target_img.shape[1], target_img.shape[0]))
        unnorm_heatmap = cv2.resize(unnorm_heatmap, (target_img.shape[1], target_img.shape[0]))
        target_heatmap_img = cv2.addWeighted(cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR), 0.4, heatmap_img, 0.6, 0)

        return heatmap_scores, target_heatmap_img, heatmap_img, unnorm_heatmap


def find_highest_heatmap_point(heatmap_img):
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

def main():
    # Init Dinov2Matcher
    # dm = Dinov2Matcher(repo_name='facebookresearch/dinov2', model_name='dinov2_vitb14', patch_size=14, ref_img_name='source_lid.jpg', ref_patch=(14, 27))
    dm = Dinov2Matcher(repo_name='facebookresearch/dinov2', model_name='dinov2_vitb14', patch_size=14, ref_img_name='source_lid.jpg', ref_patch=(27, 17))
    try:
        # for i in range(100):
        while True:
            if rospy.is_shutdown():  # Check if ROS is shutting down
                break

            # Load target image
            if camera_image is None:
                print("Waiting for camera image...")
                rospy.sleep(0.1)  # Small sleep to avoid busy looping
                continue
            
            color_buffer = camera_image.copy()
            target_img = cv2.cvtColor(color_buffer, cv2.COLOR_BGR2RGB)
            heatmap_scores, target_heatmap_img, heatmap_img, unnorm_heatmap = dm.calculate_heatmap(target_img)
            max_loc, max_val, _ = find_highest_heatmap_point(heatmap_img)

            camera_center = np.array([unnorm_heatmap.shape[0] / 2, unnorm_heatmap.shape[1] / 2])
            hole_pixel_position = np.array([max_loc[1], max_loc[0]])
            translation = np.array([0, 0])

            if unnorm_heatmap[max_loc[1], max_loc[0]] > 0.5:
                translation = (hole_pixel_position - camera_center) / -24.0 / 100

            if np.linalg.norm(translation) < 0.01:
                translation=np.zeros(2)

            print(translation)
            target_pose = copy.deepcopy(robot_pose)
            target_pose.position.x += translation[0]
            target_pose.position.y += translation[1]
            print(target_pose)
            move(target_pose)

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting gracefully...")

if __name__ == "__main__":
    try:
        rospy.init_node("image_listener", anonymous=True)
        rospy.Subscriber("/yk_builder/camera/color/image_raw", ROSImage, image_callback)
        rospy.Subscriber("/yk_builder/tool0_pose", PoseStamped, pose_callback)
        print("Initializing robot move service")
        move_client = get_client('/yk_builder/yk_go_to_pose', yk_msgs.msg.GoToPoseAction)
        base_pose = Pose(
            position=Point(0.0717, -0.3005, 0.5420),
            orientation=Quaternion(1, 0, 0, 0)
        )
        move(base_pose)
        main()  # Now properly handles Ctrl+C
    except rospy.ROSInterruptException:
        print("\nGracefully exiting on ROS shutdown.")
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting...")

