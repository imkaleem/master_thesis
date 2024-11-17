import os
import glob
import json
import shutil
import numpy as np
from utils import *

products_imgs_dir = r"C:\Users\kaleem\Learn\Wuerth\thesis\data\data_temp\datasets_temp\test_set\classes\unique_new"
save_dir = r"C:\Users\kaleem\Learn\Wuerth\thesis\data\data_temp\datasets_temp\test_set\rotated_data_new"
def augment_data_rotation(products_imgs_dir, save_dir, degrees = 36):
    if os.path.isdir(products_imgs_dir):
        
        # Augmentation
        for folder in os.listdir(products_imgs_dir):
            folder_path = os.path.join(products_imgs_dir, folder)
            print(folder_path)
            save_path = os.path.join(save_dir, folder)
            for degree in np.arange(0, 361, degrees):
                print(os.path.exists(save_dir))
                save_rotated_images(folder_path, degree, save_path)
        
    else:
        print("No images found in the directory!")
        exit(0)
        
augment_data_rotation(products_imgs_dir, save_dir)