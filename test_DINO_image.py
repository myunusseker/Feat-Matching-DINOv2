import pyrealsense2 as rs
import sys
import numpy as np
import pickle
from numpy.linalg import norm
import open3d as o3d
import matplotlib
import time

matplotlib.use('TkAgg')
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches


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

        # Show the reference image and reference patch
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(self.ref_img_tensor.squeeze().permute(1, 2, 0))
        rect = patches.Rectangle(
            (ref_patch[1] * self.patch_size, ref_patch[0] * self.patch_size), self.patch_size, self.patch_size,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        plt.title('Reference Image with Reference Patch')
        plt.show()

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
        if self.model_name == "dino_vitb8":
            return tokens.cpu().numpy()[1:]
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
        heatmap_scores = (heatmap_scores - heatmap_scores.min()) / (heatmap_scores.max() - heatmap_scores.min())

        # Convert heatmap to image using OpenCV
        heatmap_img = cv2.applyColorMap((heatmap_scores * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        heatmap_img = cv2.resize(heatmap_img, (target_img.shape[1], target_img.shape[0]))
        target_heatmap_img = cv2.addWeighted(cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR), 0.4, heatmap_img, 0.6, 0)

        return heatmap_scores, target_heatmap_img, heatmap_img


def find_highest_heatmap_point(heatmap_img):
    # Convert heatmap image to grayscale in order to find the whitest point
    gray_heatmap = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2GRAY)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray_heatmap)
    max_loc_flatten = heatmap_img.shape[0] * max_loc[0] + max_loc[1]
    return max_loc, max_val, max_loc_flatten


def set_camera_view(vis, target_point):
    view_control = vis.get_view_control()

    # Adjust the zoom level (smaller value zooms in more)
    view_control.set_zoom(0.8)


def main(color_buffer):
    # Init Dinov2Matcher
    # dm = Dinov2Matcher(repo_name='facebookresearch/dinov2', model_name='dinov2_vitb14', patch_size=14, ref_img_name='microwave2.png', ref_patch=(15, 24))
    dm = Dinov2Matcher(repo_name='facebookresearch/dinov2', model_name='dinov2_vitb14', patch_size=14, ref_img_name='water_bottle.jpeg', ref_patch=(2, 16))
    # dm = Dinov2Matcher(repo_name='facebookresearch/dinov2', model_name='dinov2_vitb14', patch_size=14, ref_img_name='water_bottle2.jpg', ref_patch=(18, 15))
    # dm = Dinov2Matcher(repo_name='facebookresearch/dinov2', model_name='dinov2_vitb14', patch_size=14, ref_img_name='cord.jpg', ref_patch=(15, 26))
    # dm = Dinov2Matcher(repo_name='facebookresearch/dinov2', model_name='dinov2_vitb14', patch_size=14, ref_img_name='cord.jpg', ref_patch=(35, 26))


    # Load target image
    target_img = cv2.cvtColor(color_buffer, cv2.COLOR_BGR2RGB)
    heatmap_scores, target_heatmap_img, heatmap_img = dm.calculate_heatmap(target_img)

    mask1 = heatmap_img[:, :, 0] < 40
    mask2 = heatmap_img[:, :, 1] < 250
    mask3 = heatmap_img[:, :, 2] < 256
    mask = mask1 & mask2 & mask3

    filtered_heatmap_img = heatmap_img.copy()
    filtered_heatmap_img[mask] = [0, 0, 0]

    filtered_target_heatmap_img = target_heatmap_img.copy()
    filtered_target_heatmap_img[mask] = [0, 0, 0]

    filtered_color_buffer = color_buffer.copy()
    filtered_color_buffer[mask] = [0, 0, 0]

    images_up = np.hstack((color_buffer, target_heatmap_img))
    images_down = np.hstack((filtered_color_buffer, filtered_target_heatmap_img))
    images = np.vstack((images_up, images_down))

    scale_factor = 0.45  # Adjust the scale factor as needed
    images = cv2.resize(images, (0, 0), fx=scale_factor, fy=scale_factor)
    print(images.shape)
    plt.imshow(images)
    plt.show()

    return filtered_color_buffer

if __name__ == "__main__":

    image = cv2.imread('water_target.jpeg')

    plt.imshow(image)
    plt.show()

    main(image)
