import os
import cv2
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from mask_utils import *

obj_dir = r"C:\Users\kaleem\Learn\Wuerth\thesis\data\data_temp\datasets_temp\test_set\rotated_data\1271810_new"
bgr_dir = r"C:\Users\kaleem\Learn\Wuerth\thesis\data\data_temp\datasets_temp\test_set\data_gen\sd_v20_background_images_new"
results_dir = r"C:\Users\kaleem\Learn\Wuerth\thesis\data\data_temp\datasets_temp\test_set\test\sd_v20_super_impose_shadow_copy"

from ultralytics import SAM
weights = r"C:\Users\kaleem\Learn\Wuerth\thesis\data\data_temp\datasets_temp\test_set\weights\mobile_sam.pt"
model = SAM(weights)

def shadow_super_impose(objects_dir, bgr_dir, results_dir):
    obj_imgs = glob.glob(os.path.join(objects_dir, "*.jpg"))
    bgr_imgs = glob.glob(os.path.join(bgr_dir, "*.png"))
    
    dir_name = obj_imgs[0].split("\\")[-2]
    os.makedirs(os.path.join(results_dir, dir_name), exist_ok=True)

    for obj_image_path in obj_imgs:
        obj_img_name = obj_image_path.split("\\")[-1].split(".")[0]
        obj_image = cv2.imread(obj_image_path)

        obj_img = cv2.imread(obj_image_path)
        mask = get_mask(obj_image_path, model)

        for bgr_img_path in bgr_imgs:
            bgr_img_name = bgr_img_path.split("\\")[-1].split(".")[0]
            alpha_channel_with_shadow = create_alpha_channel_with_shadow(obj_img, mask)
            rotated_image = rotate_image(alpha_channel_with_shadow)
            result_image = paste_image(cv2.imread(bgr_img_path), rotated_image)

            save_name = os.path.join(os.path.join(results_dir, dir_name), obj_img_name + "_" + bgr_img_name  + ".png")
            print(save_name)
            cv2.imwrite(save_name, result_image)
            



shadow_super_impose(obj_dir, bgr_dir, results_dir)