import os
import numpy as np
from PIL import Image
import cv2

import matplotlib.pyplot as plt
import torch

from cv_utils import *
from loss_utils import get_loss

def invert_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a threshold to create a binary mask
    _, mask = cv2.threshold(gray, 251, 255, cv2.THRESH_BINARY)
    
    # Invert the binary mask
    mask = cv2.bitwise_not(mask)
    inverted_image = cv2.bitwise_and(image, image, mask=mask)
    return inverted_image




def invert_image_green(image, color = 1, intensity = 255):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a threshold to create a binary mask
    _, mask = cv2.threshold(gray, 251, 255, cv2.THRESH_BINARY_INV)
    
    # Create a green color mask
    green_mask = np.zeros_like(image)
    green_mask[:, :, color] = intensity
    
    # Keep the object area unchanged
    result_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Combine the green mask with the inverted mask for the non-object area
    result_image += cv2.bitwise_and(green_mask, green_mask, mask=~mask)
    
    return result_image

def find_object_mask(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to convert image to binary
    _, binary_image = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_contour = max(contours, key=cv2.contourArea)
    
    # Create an empty mask
    mask = np.zeros_like(gray_image)
    
    # Draw contours on the mask
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    
    return mask

def remove_isolated_regions(mask):
    # Find connected components in the mask
    num_labels, labels = cv2.connectedComponents(mask)

    # Iterate through each label (excluding the background label)
    for label in range(1, num_labels):
        # Create a binary mask for the current connected component
        component_mask = np.uint8(labels == label)

        # If the component mask contains only a small number of pixels,
        # it's likely an isolated region and should be removed
        if np.sum(component_mask) < 30:
            # Set the pixels of the isolated region to 0 (background)
            mask[component_mask > 0] = 0

    return mask

def remove_isolated_regions(mask):
    # Find connected components in the mask
    num_labels, labels = cv2.connectedComponents(mask)

    # Iterate through each label (excluding the background label)
    for label in range(1, num_labels):
        # Create a binary mask for the current connected component
        component_mask = np.uint8(labels == label)

        # If the component mask contains only a small number of pixels,
        # it's likely an isolated region and should be removed
        if np.sum(component_mask) < 30:
            # Set the pixels of the isolated region to 0 (background)
            mask[component_mask > 0] = 0

    return mask

def exclude_green_pixels(image, mask):
    # Find the indices where the pixel value is [0, 150, 0]
    green_indices = np.where(np.all(image == [0, 150, 0], axis=-1))
    #print(green_indices)

    # Set the mask value to 0 at those indices
    mask[green_indices] = 0

    return mask


def remove_border_noise(mask, n):
    # Convert the mask to binary
    mask_binary = np.where(mask > 0, 1, 0).astype(np.uint8)

    # Create a structuring element for erosion
    kernel = np.ones((n, n), np.uint8)

    # Perform morphological erosion
    mask_eroded = cv2.erode(mask_binary, kernel, iterations=1)

    # Convert the eroded mask back to the original data type
    mask_eroded = np.where(mask_eroded > 0, 255, 0).astype(np.uint8)

    return mask_eroded

def remove_noise(mask, itrs = 4):
    mask = mask.astype(bool)
    
    for itr in range(1, itrs):
        print(f"................................................Iteration {itr}.................................................")
        if itr > 2:
            itr = 1
        count = 0
        total = np.sum(mask.astype(bool))
        print(f"Total number of pixels in object are {total}.")
        
        for x, y in zip(np.where(mask == True)[0], np.where(mask == True)[1]):                           # Iteration 1 with n = 1, Iteration 2 with n = 2, Iteration 3 with n = 3
            top_bottom = np.sum(mask[x - itr: x, y]) + np.sum(mask[x + 1: x + itr + 1, y])
            left_right = np.sum(mask[x, y - itr : y]) + np.sum(mask[x, y + 1: y + itr + 1])
            if top_bottom == 0 or left_right == 0:             # noise
                mask[x, y] = 0
                #print(top_bottom, left_right)
            #print(x, y)
            #if count % 4000 == 0:
            #    pass
                #print(top_bottom, left_right)
                #print(f"Total = {count}, Percentage = {round((count / total) * 100, 3)}")
            count += 1
    return mask

def convert_binary_to_3c_mask(binary_mask):
    # Ensure binary_mask is in the range [0, 255]
    binary_mask = np.clip(binary_mask, 0, 255)
    
    # Convert the binary mask into a black and white mask
    bw_mask = binary_mask / 255

    # Invert the mask if needed (optional)
    bw_mask = 1 - bw_mask
    bw_mask = bw_mask * 255

    # Convert the 2D mask to a 3-channel mask
    bw_mask_3channel = np.repeat(bw_mask[:, :, np.newaxis], 3, axis=2)

    return bw_mask_3channel

def smooth_mask(mask, kernel_size=5, iterations=1):
    # Apply erosion followed by dilation to remove noise
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    smoothed_mask = cv2.erode(mask, kernel, iterations=iterations)
    smoothed_mask = cv2.dilate(smoothed_mask, kernel, iterations=iterations)
    
    # Apply Gaussian blur to further smooth the mask
    smoothed_mask = cv2.GaussianBlur(smoothed_mask, (kernel_size, kernel_size), 0)
    
    return smoothed_mask

def remove_border_noise(mask, n):
    # Convert the mask to binary
    mask_binary = np.where(mask > 0, 1, 0).astype(np.uint8)

    # Create a structuring element for erosion
    kernel = np.ones((n, n), np.uint8)

    # Perform morphological erosion
    mask_eroded = cv2.erode(mask_binary, kernel, iterations=1)

    # Convert the eroded mask back to the original data type
    mask_eroded = np.where(mask_eroded > 0, 255, 0).astype(np.uint8)

    return mask_eroded

def remove_isolated_dots(mask, kernel_size=3):
    # Define a kernel for erosion
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Erode the mask to remove isolated dots
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    
    return eroded_mask

def canny_edge(image):
        
    image = np.array(image)

    low_threshold = 50
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    return canny_image

def generate_mask3c(image_path, itr = 4):
    print(image_path)
    image = cv2.imread(image_path)    
    img= invert_image(image)
    green = invert_image_green(image, 1, 150)
    #plt.imshow(green)
    #angle, mask = find_object_mask(green)
    mask = find_object_mask(img)
    #new_mask = exclude_green_pixels(green, mask)
    mask_i = remove_isolated_regions(mask)
    mask_ii = exclude_green_pixels(green, mask_i)
    
    #mask_clean = remove_noise(mask_ii, itr)
    mask_clean = remove_border_noise(mask_ii.astype(np.uint8) * 255 , 2)
    mask_clean = remove_isolated_dots(mask_clean.astype(np.uint8))
    mask_clean = mask_clean.astype(bool)
    #plt.imshow(mask_clean)
    
    mask_copy = mask_clean.astype(np.uint8) * 255
    mask_3c = convert_binary_to_3c_mask(mask_clean.astype("uint8") * 255)       #new_mask
    mask_3c = mask_3c.astype("uint8")
    plt.imshow(mask_3c)
    #print(mask_copy.dtype)
    
    cv2.imwrite(image_path.replace(image_path[-4], "_mask."), mask_3c)
    return mask_3c, mask_i, mask_ii, mask_clean, mask, green


def get_mask(obj_img_path, model):
    img = cv2.imread(obj_img_path)
    h, w = img.shape[:2]
    bbox = [0, 0, w, h]
    masks = model.predict(obj_img_path, bboxes = bbox)
    return masks[0].masks.data[0]

def rotate_image(image):
    # Generate a random angle between 0 and 360 degrees
    random_angle = np.random.uniform(0, 360)

    # Get the center of the image
    rows, cols = image.shape[:2]
    center = (cols // 2, rows // 2)

    # Generate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, random_angle, 1.0)

    # Apply the rotation to the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    return rotated_image

def find_object_angle(mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Get the bounding rotated rectangle
    rotated_rect = cv2.minAreaRect(contours[0])
    
    # Extract the angle from the rotated rectangle
    angle = rotated_rect[-1]
    
    return angle

def create_alpha_channel(image_path, mask):
    image = image_path
    # Read the image with a transparent background (PNG or GIF)
    #image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # If the image already has an alpha channel, return it
    if image.shape[2] == 4:
        return image

    # Create an alpha channel based on a mask
    alpha_channel = np.where(mask, 0, 255).astype(np.uint8)

    # Apply morphological operations to remove noise from the borders
    kernel = np.ones((3, 3), np.uint8)
    alpha_channel = cv2.erode(alpha_channel, kernel, iterations=1)
    alpha_channel = cv2.dilate(alpha_channel, kernel, iterations=1)
    
    # Apply opening operation to remove small white regions (noise)
    alpha_channel = cv2.morphologyEx(alpha_channel, cv2.MORPH_OPEN, kernel)

    # Add the alpha channel to the original image
    image_with_alpha = np.dstack([image, alpha_channel])

    # Crop the image based on the alpha channel
    alpha_channel_binary = (alpha_channel != 0).astype(np.uint8)
    contours, _ = cv2.findContours(alpha_channel_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the bounding box of the contours
    bounding_box = cv2.boundingRect(contours[0])

    # Extract the region of interest (ROI) based on the bounding box
    cropped_image_with_alpha = image_with_alpha[bounding_box[1]:bounding_box[1]+bounding_box[3],
                                                 bounding_box[0]:bounding_box[0]+bounding_box[2]]
    return image_with_alpha


def remove_border_noise(mask, n):
    # Convert the mask to binary
    mask_binary = np.where(mask > 0, 1, 0).astype(np.uint8)

    # Create a structuring element for erosion
    kernel = np.ones((n, n), np.uint8)

    # Perform morphological erosion
    mask_eroded = cv2.erode(mask_binary, kernel, iterations=1)

    # Convert the eroded mask back to the original data type
    mask_eroded = np.where(mask_eroded > 0, 255, 0).astype(np.uint8)

    return mask_eroded

def create_shadow(mask, angle, shift_distance=0):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Get the contour of the object
    contour = contours[0]
    
    # Create an empty mask for the shadow
    shadow_mask = np.zeros_like(mask, dtype=np.uint8)
    
    # Calculate the shift vector based on the angle
    shift_x = int(shift_distance * np.cos(np.radians(angle)))
    shift_y = int(shift_distance * np.sin(np.radians(angle)))
    
    # Shift each point of the contour
    shifted_contour = contour.copy()  # Make a copy to avoid modifying the original contour
    for point in shifted_contour:
        point[0][0] += shift_x
        point[0][1] += shift_y
    
    # Draw the shifted contour on the shadow mask
    cv2.drawContours(shadow_mask, [shifted_contour], -1, (255), thickness=cv2.FILLED)
    
    return shadow_mask

def subtract_masks(original_mask, shadow_mask):
    # Perform bitwise XOR operation between the original and shadow masks
    remaining_mask = cv2.bitwise_xor(original_mask, shadow_mask)
    return remaining_mask

def show_difference(shadow_mask, mask):
    # Convert masks to boolean
    shadow_bool = shadow_mask.astype(bool)
    mask_bool = mask.astype(bool)

    # Find the area in shadow_mask that does not exist in mask
    difference = shadow_bool ^ mask_bool

    # Convert the resulting boolean mask back to uint8 for visualization
    difference_uint8 = difference.astype(np.uint8) * 255

    return difference_uint8


def show_difference(shadow_mask, mask):
    # Convert masks to boolean arrays
    shadow_mask_bool = shadow_mask.astype(bool)
    mask_bool = mask.astype(bool)

    # Find the area that exists in shadow_mask but not in mask
    difference_bool = np.logical_and(shadow_mask_bool, np.logical_not(mask_bool))

    # Convert the boolean array back to uint8
    difference_uint8 = difference_bool.astype(np.uint8) * 255

    return difference_uint8

def blur_difference_image(difference_image, blur_radius=5):
    # Apply Gaussian blur to the difference image
    blurred_image = cv2.GaussianBlur(difference_image, (blur_radius, blur_radius), 0)
    
    return blurred_image


def add_shadow_to_image(original_image, difference_image):
    # Create a mask where the difference exists
    shadow_mask = (difference_image > 0).astype(np.uint8)

    # Convert shadow mask to 3 channels
    shadow_mask_3channels = np.repeat(shadow_mask[:, :, np.newaxis], 3, axis=2)

    # Create a copy of the original image
    image_with_shadow = original_image.copy()

    # Overlay the shadow region onto the original image
    image_with_shadow[shadow_mask_3channels == 1] = 127  # Set shadowed region to black
    image_with_shadow[shadow_mask_3channels == 1] = 30  # Set shadowed region to black

    return image_with_shadow
def create_alpha_channel_with_shadow(image, mask, distance = 10):
    mask_eroded = remove_border_noise(~mask.detach().numpy(), 3)
    angle = find_object_angle(~mask_eroded)
    shadow_mask = create_shadow(mask_eroded, angle, distance)
    mask_difference = show_difference(shadow_mask, mask_eroded)
    blurred_difference_mask = blur_difference_image(mask_difference, blur_radius=5)
    #plt.imshow(blurred_difference_mask)
    plt.imshow(mask_eroded)
    #plt.imshow(mask_difference)
    image_with_shadow = add_shadow_to_image(image, blurred_difference_mask)
    #plt.imshow(image_with_shadow)
    full_mask = blurred_difference_mask.astype(bool) | mask_eroded.astype(bool)
    plt.imshow(full_mask)
    cropped_image_squarely = create_alpha_channel(image_with_shadow, ~full_mask)
    return cropped_image_squarely

def paste_image(background, foreground, position=(0, 0)):
    """
    Paste a smaller image (foreground) onto a larger image (background) at the specified position.
    
    Parameters:
        - background (numpy.ndarray): The larger image.
        - foreground (numpy.ndarray): The smaller image to be pasted.
        - position (tuple): The position (x, y) where the top-left corner of the smaller image will be placed.
        
    Returns:
        - numpy.ndarray: The resulting image with the smaller image pasted onto the larger one.
    """
    h, w = foreground.shape[:2]
    background = cv2.resize(background, (h + 400, w + 400))
    
    center_x = np.random.randint(0, 1024)
    center_y = np.random.randint(0, 1024)
    
    # Ensure the position is within the bounds of the background image
    y, x = (center_y, center_x)
    y = max(0, min(y, background.shape[0] - h))
    x = max(0, min(x, background.shape[1] - w))
    
    # Create a region of interest (ROI) for the smaller image on the background
    roi = background[y:y + h, x:x + w]
    
    # Create a mask for the foreground image
    mask = (foreground[:, :, 3] / 255.0).reshape((h, w, 1))
    
    # Blend the images using the alpha channel of the foreground
    blended = roi * (1 - mask) + foreground[:, :, :3] * mask
    
    # Update the background with the blended region
    background[y:y + h, x:x + w] = blended
    
    return background