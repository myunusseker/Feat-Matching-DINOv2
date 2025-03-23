import rospy
import rostopic
import numpy as np
from cv_bridge import CvBridge
import cv2
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

RGB_IMAGE_TOPIC = '/yk_builder/camera/color/image_raw'
DEPTH_IMAGE_TOPIC = '/yk_builder/camera/aligned_depth_to_color/image_raw'
RGB_INFO_TOPIC = '/yk_builder/camera/color/camera_info'
PIXEL_X = 400
PIXEL_Y = 838
T_cam_eef = np.array(
    [[-0.01722443, -0.99952439, -0.02557959, -0.06725155],
     [ 0.9998478,  -0.01714775, -0.00321424, -0.01221117],
     [ 0.00277408, -0.02563106,  0.99966762,  0.07440511],
     [ 0.,          0.,          0.,          1.        ]]
)
T_eef_robot = np.array(
    [[1, 0, 0, 0.0717019],
     [ 0, -1, 0, -0.3005],
     [ 0, 0, -1,  0.541999],
     [ 0., 0., 0., 1.]]
)


def deproject_pixel(depth_data: np.ndarray, pixel_x: int, pixel_y: int, K: np.ndarray):
    pixel_data = [pixel_y, pixel_x]
    depth = depth_data[pixel_x, pixel_y]

    point_3d = np.linalg.inv(K).dot(np.r_[pixel_data, 1.0])

    return depth * point_3d


def visualize_img(topic: str):
    msg_type = rostopic.get_topic_class(topic)[0]
    depth_image_ros = rospy.wait_for_message(topic, msg_type, 1)
    depth_data = CvBridge().imgmsg_to_cv2(depth_image_ros, desired_encoding="passthrough")
    plt.imshow(depth_data, cmap='gray')
    plt.show()


def main():
    rospy.init_node("shobhit_testing", anonymous=True)
    msg_type = rostopic.get_topic_class(RGB_INFO_TOPIC)[0]
    camera_info = rospy.wait_for_message(RGB_INFO_TOPIC, msg_type, 1)
    visualize_img(RGB_IMAGE_TOPIC)

    while True:
        point_3d_array = []

        for i in range(5):
            msg_type = rostopic.get_topic_class(DEPTH_IMAGE_TOPIC)[0]
            depth_image_ros = rospy.wait_for_message(DEPTH_IMAGE_TOPIC, msg_type, 1)
            depth_data = CvBridge().imgmsg_to_cv2(depth_image_ros, desired_encoding="passthrough")

            K = np.resize(camera_info.K, (3, 3))

            point_3d = deproject_pixel(depth_data, PIXEL_X, PIXEL_Y, K)
            if point_3d[2] != 0 and point_3d[2] != 65536:
                point_3d_array.append(point_3d)
            else:
                i = i - 1

        final_point = np.zeros(3)
        final_point[0] = np.sum([each[0] for each in point_3d_array]) / 5
        final_point[1] = np.sum([each[1] for each in point_3d_array]) / 5
        final_point[2] = np.sum([each[2] for each in point_3d_array]) / 5
        final_point /= 1000
        final_point = np.array([*final_point, 1])
        print(f"Point in 3D: {final_point}")
        print(f"Point wrt EEF: {T_cam_eef @ final_point}")
        print(f"Point wrt Robot Base: {T_eef_robot @ T_cam_eef @ final_point}")


if __name__ == '__main__':
    main()
