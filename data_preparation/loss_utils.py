from cv_utils import *
def get_loss(image, image_gen, conf = 0.03, threshold = 40):
    loss = 0
    orig_img_mask, l, w = orig_image_mask(image)
    gen_img_mask = find_object_mask(image, orig_img_mask, image_gen, threshold = 40)
    
    if gen_img_mask is not None:
        image_length, image_width = find_object_dimensions(mask.astype("uint8"))
        image_gen_length, image_gen_width = find_object_dimensions(gen_img_mask.astype("uint8"))
        image_ratio = calculate_width_to_height_ratio(image_width, image_length)
        image_gen_ratio = calculate_width_to_height_ratio(image_gen_width, image_gen_length)
        print(image_ratio, image_gen_ratio)
        if (image_ratio <= (image_gen_ratio + conf)) and (image_ratio >= (image_gen_ratio - conf)):
            print("True I")
            return loss
        else:
            print("True II")
            loss += abs(image_gen_ratio - image_ratio)
            return loss
    else:
        print("True III")
        return 1