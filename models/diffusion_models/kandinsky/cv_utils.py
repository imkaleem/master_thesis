import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_percentage_white_pixels(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to binarize the image (assuming white is foreground)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Count the number of white pixels
    num_white_pixels = cv2.countNonZero(binary)

    # Calculate the total number of pixels in the image
    total_pixels = image.shape[0] * image.shape[1]

    # Calculate the percentage of white pixels
    percentage_white_pixels = (num_white_pixels / total_pixels) * 100

    return percentage_white_pixels

def invert_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a threshold to create a binary mask
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    
    # Invert the binary mask
    mask = cv2.bitwise_not(mask)
    inverted_image = cv2.bitwise_and(image, image, mask=mask)
    return inverted_image

def remove_noise(image):
    raw_image = image.copy()
    top_side_average = np.array([int(np.average(raw_image[:, 0, channel])) for channel in range(3)])
    raw_image[:, 0, :3] = top_side_average
    
    right_side_average = np.array([int(np.average(raw_image[raw_image.shape[0] - 1, :, channel])) for channel in range(3)])
    raw_image[raw_image.shape[0] - 1, :, :3] = right_side_average
    
    lower_side_average = np.array([int(np.average(raw_image[:, raw_image.shape[1] - 1, channel])) for channel in range(3)])
    raw_image[:, raw_image.shape[1] - 1, :3] = lower_side_average
    
    left_side_average = np.array([int(np.average(raw_image[0, :, channel])) for channel in range(3)])
    raw_image[0, :, :3] = left_side_average
    return raw_image

def draw_rotated_rectangle(image, contour, rect_color=(0, 255, 0), font=5):
    # Find the minimum area rectangle enclosing the contour
    rect = cv2.minAreaRect(contour)
    
    # Convert the rectangle parameters to integer
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Draw the rotated rectangle on the image
    result_image = cv2.drawContours(image.copy(), [box], 0, rect_color, 2)
    
    return result_image


def find_contours(image):
    raw_image = remove_noise(image)

    grey = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2GRAY)
    _, binary_image = cv2.threshold(grey, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours

def draw_contours(image, contours, color=(0, 255, 0), thickness=2):
    """
    Draw contours on the input image.
    
    Parameters:
        image: Input image.
        contours: List of contours obtained from cv2.findContours().
        color: Contour color. Default is green (0, 255, 0).
        thickness: Contour thickness. Default is 2.
        
    Returns:
        Image with contours drawn on it.
    """
    
    # Make a copy of the input image to avoid modifying the original image
    image_with_contours = image.copy()
    
    # Draw contours on the image
    cv2.drawContours(image_with_contours, contours, -1, color, thickness)
    plt.imshow(image_with_contours)

def draw_rectangle(image, rect_color = (0, 255, 0), font = 5):
    contours = find_contours(image)
    biggest_contour = max(contours, key = cv2.contourArea)
    
    image_copy = image.copy()
    result_image = draw_rotated_rectangle(image_copy, biggest_contour, rect_color, font)
    plt.imshow(result_image)

def calculate_width_to_height_ratio(w, h):
    # Calculate the width-to-height ratio
    ratio = w / h
    return ratio

def draw_convex_hull(image, rect_color = (0, 255, 0), font = 5):
    contours = find_contours(image)
    biggest_contour = max(contours, key = cv2.contourArea)

    image_copy = image.copy()
    hull = cv2.convexHull(biggest_contour)
    cv2.polylines(image_copy, [hull], True, (0,0,255), 2)
    #plt.imshow(image_copy)
    return hull

def draw_height_width_of_hull(hull):
    rotated_hull, center = rotate_to_horizontal(hull)
    #print(rotated_hull)
    width, height, vertices = find_length_and_width(rotated_hull)
    plot_hull_with_dimensions(rotated_hull, width, height, vertices)
    return rotated_hull, width, height, center

def draw_cropped_hull(image, rect_color = (0, 255, 0), font = 5):
    hull = draw_convex_hull(image, rect_color, font)
    #print(hull)
    rotated_hull, width, height, center = draw_height_width_of_hull(hull)
    return rotated_hull, width, height, center



def get_rotated_rectangle_dimensions(image, contour):

    temp_img = image.copy()
    
    # Find minimum area bounding rectangle
    rect = cv2.minAreaRect(contour)

    # Get dimensions of the rectangle
    width = rect[1][0]
    height = rect[1][1]

    # Get center of the rectangle
    center = rect[0]

    # Draw rotated rectangle
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(temp_img, [box], 0, (0, 255, 0), 2)
    plt.imshow(temp_img)

    return width, height, center


def get_rotated_rectangle_dimensions(image, contour):

    temp_img = image.copy()
    
    # Find minimum area bounding rectangle
    rect = cv2.minAreaRect(contour)

    # Get dimensions of the rectangle
    width = rect[1][0]
    height = rect[1][1]

    orig_width = rect[1][0]
    orig_height = rect[1][1]

    # Swap dimensions if necessary so that longer side is called length and shorter side is called width
    if width > height:
        length, width = width, height
    else:
        length, width = height, width

    # Get center of the rectangle
    center = rect[0]

    # Get angle
    angle = rect[2]

    # Draw rotated rectangle
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(temp_img, [box], 0, (0, 255, 0), 2)
    plt.imshow(temp_img)

    return length, width, orig_width, orig_height, center, angle

def draw_ellipse_mask(image, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness=-1):
    # Create a blank mask image
    mask = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))

    # Draw the ellipse on the mask
    mask = cv2.ellipse(mask, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness).astype(np.bool_)

    return mask

def select_contour_with_allowed_area(contours: np.ndarray, contour_area_max: int, contour_area_min: int = 0) -> list:
    """
    Returns the contour with biggest area among those having an area withing the boundaries area_max and area_min
    """

    # print("max area: ", contour_area_max)
    # print("min area: ", contour_area_min)

    contours_indexes = np.arange(0, stop=len(contours))
    
    print(f"contours_indexes = {contours_indexes}")

    # Calculate the area of each contour
    contours_areas = np.array([cv2.contourArea(c) for c in contours])
    print(contours_areas[0])
    print(f"contours_areas = {contours_areas}")

    allowed_areas_index = ((contours_areas < contour_area_max) & (contours_areas > contour_area_min))
    
    print(f"allowed_areas_index = {allowed_areas_index}")

    allowed_contours_indexes = contours_indexes[allowed_areas_index]
    
    print(f"allowed_contours_indexes = {allowed_contours_indexes}")

    selected_contour_max_area_index = np.argmax(contours_areas[allowed_areas_index])
    
    print(f"selected_contour_max_area_index = {selected_contour_max_area_index}")

    contours_max_area_index = allowed_contours_indexes[selected_contour_max_area_index]
    
    print(f"contours_max_area_index = {contours_max_area_index}")
    
    print(f"Return = {contours[contours_max_area_index]}")

    return contours[contours_max_area_index]


def find_hull_center(hull):
    # Compute the moments of the convex hull
    M = cv2.moments(hull)
    
    # Calculate the centroid (center) of the convex hull
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    return cx, cy

def find_hull_corners(hull):
    # Get extreme points (corners) of the convex hull
    leftmost = tuple(hull[hull[:, :, 0].argmin()][0])
    rightmost = tuple(hull[hull[:, :, 0].argmax()][0])
    topmost = tuple(hull[hull[:, :, 1].argmin()][0])
    return leftmost, rightmost, topmost

def calculate_angle(center, corner):
    # Calculate the angle between the line joining center and corner with horizontal axis
    angle_rad = np.arctan2(corner[1] - center[1], corner[0] - center[0])
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def find_hull_corners(hull):
    # Get extreme points (corners) of the convex hull
    leftmost = tuple(hull[hull[:, :, 0].argmin()][0])
    rightmost = tuple(hull[hull[:, :, 0].argmax()][0])
    topmost = tuple(hull[hull[:, :, 1].argmin()][0])
    return leftmost, rightmost, topmost

def rotate_to_horizontal(hull):
    # Get the orientation angle of the hull
    angle = np.arctan2(hull[:, 0, 1].max() - hull[:, 0, 1].min(), hull[:, 0, 0].max() - hull[:, 0, 0].min()) * 180 / np.pi

    # Rotate the hull to align with the longest side along the x-axis
    center = (np.mean(hull[:, :, 0]), np.mean(hull[:, :, 1]))  # Center of the hull
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rotated_hull = cv2.transform(hull.reshape(-1, 1, 2), rotation_matrix)

    return rotated_hull.reshape(-1, 2), center

def plot_hull_and_center(hull, center):
    plt.figure()
    plt.plot(hull[:, 0], hull[:, 1], c='blue')  # Plot convex hull
    plt.scatter(center[0], center[1], c='red', marker='o')  # Plot center of convex hull
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Convex Hull and Its Center')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()

def find_length_and_width(rotated_hull):
    # Get the coordinates of the hull vertices
    vertices = rotated_hull.reshape(-1, 2)

    # Find the indices of the vertices with maximum distances in both horizontal and vertical directions
    top_index = np.argmax(vertices[:, 1])
    bottom_index = np.argmin(vertices[:, 1])
    right_index = np.argmax(vertices[:, 0])
    left_index = np.argmin(vertices[:, 0])

    # Calculate the distances between the vertices
    width = np.linalg.norm(vertices[left_index] - vertices[right_index])
    height = np.linalg.norm(vertices[top_index] - vertices[bottom_index])

    return width, height, vertices

def plot_hull_with_dimensions(rotated_hull, width, height, vertices):
    # Plot the convex hull
    plt.plot(rotated_hull[:, 0], rotated_hull[:, 1], c='blue')

    # Find the vertices with maximum distances in both horizontal and vertical directions
    top_index = np.argmax(vertices[:, 1])
    bottom_index = np.argmin(vertices[:, 1])
    right_index = np.argmax(vertices[:, 0])
    left_index = np.argmin(vertices[:, 0])

    # Plot lines from top to bottom and from right to left, meeting at the vertices with maximum distances
    plt.plot([vertices[top_index][0], vertices[bottom_index][0]], [vertices[top_index][1], vertices[bottom_index][1]], c='red', linestyle='--', label=f'Height: {height:.2f}')
    plt.plot([vertices[right_index][0], vertices[left_index][0]], [vertices[right_index][1], vertices[left_index][1]], c='green', linestyle='--', label=f'Width: {width:.2f}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Convex Hull with Dimensions')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.legend()
    plt.show()


def find_object_dimensions(mask):
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours = find_contours(mask)

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(largest_contour)

    # Plot the ellipse on the original image
    original_image = mask.copy()
    cv2.ellipse(original_image, ellipse, (0, 255, 0), 2)
    plt.imshow(original_image)

    # Extract major and minor axes lengths
    (center, axes, angle) = ellipse
    major_axis_length = max(axes)
    minor_axis_length = min(axes)

    return major_axis_length, minor_axis_length



def find_object_mask(original_image, original_mask, generated_image, threshold = 40):
    if calculate_percentage_white_pixels(generated_image) > 60:
        generated_image = invert_image(generated_image)
    # Convert images to grayscale
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    generated_gray = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors for both images
    kp1, des1 = orb.detectAndCompute(original_gray, None)
    kp2, des2 = orb.detectAndCompute(generated_gray, None)

    # Initialize Brute-Force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Define a threshold to consider a match as valid

    # List to store valid matches
    valid_matches = []

    # Filter valid matches based on threshold
    for match in matches:
        if match.distance < threshold:          # Adjust the threshold
            valid_matches.append(match)

    # If enough valid matches found, object is present
    if len(valid_matches) > 30:         # Adjust the count
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in valid_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in valid_matches]).reshape(-1, 1, 2)

        # Calculate homography
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp mask of object from original image to generated image
        mask_warped = cv2.warpPerspective(original_mask.astype(np.uint8), H, (generated_image.shape[1], generated_image.shape[0]))

        return mask_warped
    else:
        print("Object not found in generated image")
        return None


def find_object_location(original_image, generated_image, threshold = 80):
    # Convert images to grayscale
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    generated_gray = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors for both images
    kp1, des1 = orb.detectAndCompute(original_gray, None)
    kp2, des2 = orb.detectAndCompute(generated_gray, None)

    # Initialize Brute-Force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Define a threshold to consider a match as valid

    # List to store valid matches
    valid_matches = []

    # Filter valid matches based on threshold
    for match in matches:
        if match.distance < threshold:
            valid_matches.append(match)

    # If enough valid matches found, object is present
    if len(valid_matches) > 10:  # Adjust threshold as needed
        # Draw matches
        img_matches = cv2.drawMatches(original_image, kp1, generated_image, kp2, valid_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Display matches
        plt.imshow(img_matches)
    else:
        print("Object not found in generated image")


def object_present_in_generated(masked_object, masked_object_mask, generated_image):
    # Convert the masked object to grayscale
    masked_gray = cv2.cvtColor(masked_object, cv2.COLOR_BGR2GRAY)

    # Convert the generated image to grayscale
    generated_gray = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)

    # Perform template matching with the masked object and its mask
    result = cv2.matchTemplate(generated_gray, masked_gray, cv2.TM_CCOEFF_NORMED, masked_object_mask)

    # Find the location of the maximum matching score
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Check if the maximum matching score is above the threshold
    object_present = max_val >= 0

    #print("min_value:", min_val)
    #print("max_value:", max_val)

    return object_present


def orig_image_mask(image):
    percentage_white = calculate_percentage_white_pixels(image)
    if percentage_white > 50:
        image = invert_image(image)
        
    contours = find_contours(image)
    biggest_contour = max(contours, key = cv2.contourArea)
    l, w, orig_width, orig_height, center, angle = get_rotated_rectangle_dimensions(image, biggest_contour)
    center_coordinates = (int(center[0]), int(center[1]))
  
    axesLength = (int(orig_width / 2) + 15, int(orig_height / 2) + 15)
      
    startAngle = 0
    endAngle = 360
       
    # Red color in BGR
    color = (0, 255, 0)
       
    # Line thickness of 5 px 
    thickness = 5
       
    # Using cv2.ellipse() method 
    # Draw a ellipse with red line borders of thickness of 5 px
    image = cv2.ellipse(image, center_coordinates, axesLength,
               angle, startAngle, endAngle, color, thickness)
    
    mask = draw_ellipse_mask(image, center_coordinates, axesLength, angle, startAngle, endAngle, (225, 225, 225), thickness=-1)
    plt.imshow(mask)
    return mask, l, w

def draw_ellipse_mask(image, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness=-1):
    # Create a blank mask image
    mask = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))

    # Draw the ellipse on the mask
    mask = cv2.ellipse(mask, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness).astype(np.bool_)

    return mask

def generate_rect_mask(image_shape, center, width, height, angle):
    # Create blank mask
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    # Draw rotated rectangle on mask
    rect = ((center[0], center[1]), (width, height), angle)
    box = cv2.boxPoints(rect).astype(np.int0)
    cv2.drawContours(mask, [box], 0, (255), -1)
    
    return mask

def find_obj_rect(image):
    percentage_white = calculate_percentage_white_pixels(image)
    if percentage_white > 50:
        image = invert_image(image)
        
    contours = find_contours(image)
    biggest_contour = max(contours, key = cv2.contourArea)
    l, w, orig_width, orig_height, center, angle = get_rotated_rectangle_dimensions(image, biggest_contour)
    return (l, w, orig_width, orig_height, center, angle)

def object_present_in_generated(generated_image, masked_object, masked_object_mask = None):
    # Convert the masked object to grayscale
    masked_gray = cv2.cvtColor(masked_object, cv2.COLOR_BGR2GRAY)

    # Convert the generated image to grayscale
    generated_gray = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)

    # Perform template matching with the masked object and its mask
    if masked_object_mask is not None:
        result = cv2.matchTemplate(generated_gray, masked_gray, cv2.TM_CCOEFF_NORMED, masked_object_mask)
    else:
        result = cv2.matchTemplate(generated_gray, masked_gray, cv2.TM_CCOEFF_NORMED)

    # Find the location of the maximum matching score
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Check if the maximum matching score is above the threshold
    object_present = max_val >= 0.1

    #print("min_value:", min_val)
    #print("max_value:", max_val)

    return object_present


def find_object_location(original_image, generated_image, threshold = 80):

    if object_present_in_generated(generated_image, original_image):
        
        # Convert images to grayscale
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        generated_gray = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)
    
        # Initialize ORB detector
        orb = cv2.ORB_create()
    
        # Find keypoints and descriptors for both images
        kp1, des1 = orb.detectAndCompute(original_gray, None)
        kp2, des2 = orb.detectAndCompute(generated_gray, None)
    
        # Initialize Brute-Force matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
        # Match descriptors
        matches = bf.match(des1, des2)
    
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
    
        # Define a threshold to consider a match as valid
    
        # List to store valid matches
        valid_matches = []
    
        # Filter valid matches based on threshold
        for match in matches:
            if match.distance < threshold:
                valid_matches.append(match)
    
        # If enough valid matches found, object is present
        if len(valid_matches) > 10:  # Adjust threshold as needed
            # Draw matches
            img_matches = cv2.drawMatches(original_image, kp1, generated_image, kp2, valid_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
            # Display matches
            plt.imshow(img_matches)
            return True
        else:
            print("Object not found in generated image")
            return False
    else:
        print("object not found in generated image")
        return False

def length_to_width_ratio(image):
    percentage_white = calculate_percentage_white_pixels(image)
    if percentage_white > 50:
        image = invert_image(image)
        
    contours = find_contours(image)
    biggest_contour = max(contours, key = cv2.contourArea)
    l, w, orig_width, orig_height, center, angle = get_rotated_rectangle_dimensions(image, biggest_contour)
    ratio = w / l
    return ratio

def crop_img(image, crop = False):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to convert image to binary
    _, binary_image = cv2.threshold(gray_image, 40, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the contour with the maximum area
    max_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray_image)
    cv2.drawContours(mask, [max_contour], -1, (255), cv2.FILLED)
    
    # Get the rotated rectangle enclosing the contour
    rotated_rect = cv2.minAreaRect(max_contour)
    
    # Get the rotation angle of the rectangle
    angle = rotated_rect[-1]
    
    # Rotate the image to align the object horizontally
    if angle <= -1 or angle >= 1:
        #print("True, angle = ", angle)
        rotation_matrix = cv2.getRotationMatrix2D(rotated_rect[0], angle - 90, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    else:
        rotated_image = image.copy()
    
    # Get the coordinates of the rectangle vertices
    vertices = cv2.boxPoints(rotated_rect)
    vertices = np.int0(vertices)
    
    # Draw the rectangle on the image
    #cv2.drawContours(rotated_image, [vertices], -1, (0, 255, 0), 2)

    if crop:
        #print("True")
        # Calculate the bounding box of the rotated rectangle
        x, y, w, h = cv2.boundingRect(vertices)
        # Crop the rotated rectangle from the original image
        cropped_image = rotated_image[y:y+h, x:x+w]
        
        return cropped_image, mask[y:y+h, x:x+w]
    return rotated_image


def crop_img(image, crop = False):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to convert image to binary
    _, binary_image = cv2.threshold(gray_image, 40, 255, cv2.THRESH_BINARY)
    
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the contour with the maximum area
    max_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray_image)
    cv2.drawContours(mask, [max_contour], -1, (255), cv2.FILLED)
    
    # Get the rotated rectangle enclosing the contour
    rotated_rect = cv2.minAreaRect(max_contour)

    max_side_obj_length = max(rotated_rect[1])
    max_side_img_length = max(gray_image.shape)
    if max_side_obj_length > max_side_img_length:
        print("True")
        padding_size = int(max_side_obj_length - max_side_img_length) + 1
        # Add padding
        image = np.pad(image, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)), mode='constant')
    
    # Get the rotation angle of the rectangle
    angle = rotated_rect[-1]
    
    # Rotate the image to align the object horizontally
    if angle <= -1 or angle >= 1:
        #print("True, angle = ", angle)
        rotation_matrix = cv2.getRotationMatrix2D(rotated_rect[0], angle - 90, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    else:
        rotated_image = image.copy()
    
    # Get the coordinates of the rectangle vertices
    vertices = cv2.boxPoints(rotated_rect)
    vertices = np.int0(vertices)
    
    # Draw the rectangle on the image
    #cv2.drawContours(rotated_image, [vertices], -1, (0, 255, 0), 2)

    if crop:
        #print("True")
        # Calculate the bounding box of the rotated rectangle
        x, y, w, h = cv2.boundingRect(vertices)
        # Crop the rotated rectangle from the original image
        cropped_image = rotated_image[y:y+h, x:x+w]
        
        return cropped_image, mask[y:y+h, x:x+w]
    return rotated_image

def find_orig_img_mask(image):
    if calculate_percentage_white_pixels(image) > 60:
        image = invert_image(image)
    template = remove_noise(image)
    straight_template = crop_img(template)
    return crop_img(straight_template, crop = True)


def plot_rotated_rectangle_over_mask(image, mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the contour with the maximum area
    max_contour = max(contours, key=cv2.contourArea)
    
    # Get the rotated rectangle enclosing the contour
    rotated_rect = cv2.minAreaRect(max_contour)
    
    # Draw the rotated rectangle on a black image
    rect_image = np.zeros_like(mask)
    box = cv2.boxPoints(rotated_rect)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, (255), thickness=3)
    plt.imshow(image)

def get_rotated_rectangle_dimensions(mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the contour with the maximum area
    max_contour = max(contours, key=cv2.contourArea)
    
    # Get the rotated rectangle enclosing the contour
    rotated_rect = cv2.minAreaRect(max_contour)
    
    # Draw the rotated rectangle on a black image
    rect_image = np.zeros_like(mask)
    box = cv2.boxPoints(rotated_rect)
    box = np.int0(box)
    cv2.drawContours(mask, [box], 0, (255), thickness=3)
    #plt.imshow(mask)
    
    (cx, cy), (width, length), angle = rotated_rect
    
    # Swap width and length if length is greater than width
    if length < width:
        length, width = width, length
    
    return length, width

def resize_template(template, img):
    # Get the dimensions of the template and the input image
    template_height, template_width, _ = template.shape
    img_height, img_width, _ = img.shape
    
    # Check if the template dimensions are larger than the input image
    if template_height > img_height or template_width > img_width:
        # Reduce the dimensions of the template by half
        new_height = template_height // 2
        new_width = template_width // 2
        resized_template = cv2.resize(template, (new_width, new_height))
    else:
        resized_template = template
    
    return resized_template