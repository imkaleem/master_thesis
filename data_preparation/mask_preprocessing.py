from mask_utils import *
image_path = r"C:\Users\kaleem\Learn\Wuerth\thesis\data\data_temp\datasets_temp\test_set\classes\32098732\32098732.jpg"
mask, i, ii, clean, orig_mask, green = generate_mask3c(image_path, 5)