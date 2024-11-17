import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

def make_square(image_path, output_path, fill_color=(255, 255, 255)):
    # Load the image
    image = Image.open(image_path)
    
    # Get original dimensions
    width, height = image.size
    
    # Determine the size of the new square image
    max_side = max(width, height)
    
    # Create a new image with a white background
    new_image = Image.new('RGB', (max_side, max_side), fill_color)
    
    # Calculate the position to paste the original image onto the new image
    paste_position = ((max_side - width) // 2, (max_side - height) // 2)
    
    # Paste the original image onto the new square image
    new_image.paste(image, paste_position)
    
    # Save the new image
    new_image.save(output_path)
    print(f"Image saved as square image at: {output_path}")
    
# Make the image square
input_image_path = image_paths[0]
output_image_path = r"C:\Users\kaleem\Learn\Wuerth\thesis\data\data_temp\datasets_temp\test_set\square.jpg"
make_square(input_image_path, output_image_path)


#Make the set of images square
dir = r"C:\Users\kaleem\Learn\Wuerth\thesis\data\data_temp\datasets_temp\test_set\classes\classes"
image_paths = [os.path.join(path, name) for path, subdirs, files in os.walk(dir) for name in
                          files if os.path.splitext(name)[-1].lower() in ('.jpg', '.jpeg', '.png', '.webp', '.bmp')]
for image_path in image_paths:
    make_square(image_path, image_path)