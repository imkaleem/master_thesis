from PIL import Image

def superimpose_images(background_path, foreground_path, output_path):
    # Open both images
    background = Image.open(background_path).convert("RGBA")
    background = background.resize((800, 800))
    foreground = Image.open(foreground_path).convert("RGBA")
    foreground = foreground.resize((800, 800))

    #print(background.size, foreground.size)
    
    # Create a mask where white becomes transparent
    datas = foreground.getdata()
    
    new_data = []
    for item in datas:
        # Change all white (also shades of whites)
        # pixels to transparent
        if item[0] > 230 and item[1] > 230 and item[2] > 230:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    
    foreground.putdata(new_data)
    
    # Composite the images using the mask
    combined = Image.alpha_composite(background, foreground)

    import matplotlib.pyplot as plt
    plt.imshow(combined)
    
    # Save the final image
    combined.save(output_path, format="PNG")
    
def superimpose_images_dir(bg_imgs, foreground_path, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through background images
    for bg_img_path in bg_imgs:
        output_path = os.path.join(output_folder, os.path.basename(bg_img_path))
        superimpose_images(bg_img_path, foreground_path, output_path)

# Paths to the images
background_path = r"C:\Users\kaleem\Learn\Wuerth\thesis\data\data_temp\datasets_temp\test_set\data_gen\controlnet_prompt\150535\146_150535.jpg"
 
foreground_path = r"C:\Users\kaleem\Learn\Wuerth\thesis\data\data_temp\datasets_temp\test_set\classes\unique\150535\150535_new.jpg"

output_path = r"C:\Users\kaleem\Learn\Wuerth\thesis\data\data_temp\datasets_temp\test_set\generated\150535_gen.jpg"

# Call the function
superimpose_images(background_path, foreground_path, output_path)