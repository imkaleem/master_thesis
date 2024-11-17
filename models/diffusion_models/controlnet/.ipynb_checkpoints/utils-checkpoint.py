import numpy as np
from PIL import Image
def save_images(prompt, images, count):
    is_saved = False
    paths = []
    if isinstance(images, Image.Image):
        images.save(prompt + str(count + 1) + ".png")
        is_saved = True
        paths = prompt + str(count + 1) + ".png"
        
    elif isinstance(images, list):
        for img in images:
            count += 1
            img.save(prompt + str(count + 1) + ".png")
            paths.append([prompt + str(count + 1) + ".png"])
        is_saved = True
    
    if is_saved:
        print("The image(s) is / are saved successfully on the following path (s)... \n", paths)