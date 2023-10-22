# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import os
import tensorflow as tf
import cv2


def get_image_paths(root_dir):
    image_paths = []
    for scene_folder in os.listdir(root_dir):
        scene_folder_path = os.path.join(root_dir, scene_folder)

        for numbered_folder in os.listdir(scene_folder_path):
            numbered_folder_path = os.path.join(scene_folder_path, numbered_folder)
            image_paths.append(numbered_folder_path)

    return image_paths

def is_corrupt(image):
    return image is None or image.shape[0] == 0 or image.shape[1] == 0

image_root = "../alignment_generator/NEW_PROCESSED_DATA"
image_paths = get_image_paths(image_root)

image_names = ['drivable.png', 'lanes.png', 'vehicles.png']  # Replace with actual image file paths

# Load the images
for path in image_paths:
  images = []
  for image_name in image_names:
      image = cv2.imread(os.path.join(path, image_name))
      if is_corrupt(image):
          print(f"Corrupt image: {image_name} in folder: {path}")
          break
      images.append(image)

  if len(images) == len(image_names):
    image1, image2, image3 = images

    # overlap the lanes to light green
    height, width, channel = images[0].shape
    #print(height, width, channel)
    combined_image = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
    for i in range(height):
        for j in range(width):
        #print(lanes[i,j]) if lane is white, change it to light green
            if images[1][i,j,0] == 255 and images[1][i,j,1] == 255 and images[1][i,j,2] == 255:
                combined_image[i,j,0] = 204
                combined_image[i,j,1] = 255
                combined_image[i,j,2] = 153
    # overlap the vehicles to light red
    for i in range(height):
        for j in range(width): # if vehicles is white, change it the light red
            if images[2][i,j,0] == 255 and images[2][i,j,1] == 255 and images[2][i,j,2] == 255:
                combined_image[i,j,0] = 255
                combined_image[i,j,1] = 102
                combined_image[i,j,2] = 102

    # covert the rgb image to be bgr and save
    # Convert the RGB image to grayscale
    gray_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2GRAY)
    combined_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)

    # save rbg imag and gray image into corresponding folder
    cv2.imwrite(os.path.join(path, 'combined_rgb.jpg'), combined_image)
    cv2.imwrite(os.path.join(path, 'combined_gray.jpg'), gray_image)
    print('combined images are saved into ' + path)
