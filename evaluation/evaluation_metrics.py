import os
import torch
from torchvision import transforms
from PIL import Image
from pytorch_fid import fid_score

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy.linalg import sqrtm




def load_images_from_directory(directory, image_size=(299, 299)):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust for your image formats
            image_path = os.path.join(directory, filename)
            img = load_img(image_path, target_size=image_size)
            img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
            images.append(img_array)
    return np.array(images)


def calculate_fid_single_image(true_image_path, fake_image_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Calculate Fréchet Inception Distance (FID) between two individual images.

    Parameters:
    - true_image_path (str): File path of the true image.
    - fake_image_path (str): File path of the generated (fake) image.
    - device (str): Device to use ('cuda' or 'cpu').

    Returns:
    - fid_score (float): The calculated FID score.
    """

    # Load images and preprocess
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    true_image = Image.open(true_image_path).convert('RGB')
    fake_image = Image.open(fake_image_path).convert('RGB')

    true_tensor = transform(true_image).unsqueeze(0).to(device)
    fake_tensor = transform(fake_image).unsqueeze(0).to(device)

    # Calculate FID score
    fid_value = fid_score.calculate_fid(true_tensor, fake_tensor, device=torch.device(device))

    return fid_value


def calculate_fid(real_images, generated_images):
    # Load InceptionV3 model pre-trained on ImageNet
    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    # Preprocess images and get their feature representations
    real_features = inception_model.predict(preprocess_input(real_images.astype('float32')))
    gen_features = inception_model.predict(preprocess_input(generated_images.astype('float32')))

    # Calculate mean and covariance of features
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)

    # Calculate FID score
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_score = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid_score



def calculate_ssim(real_images, generated_images):
    ssim_scores = []

    # Loop through each pair of images
    for real, generated in zip(real_images, generated_images):
        # Ensure images are in the right format (uint8)
        real_uint8 = tf.convert_to_tensor(real * 255, dtype=tf.uint8)
        generated_uint8 = tf.convert_to_tensor(generated * 255, dtype=tf.uint8)

        # Calculate SSIM
        score = tf.image.ssim(real_uint8, generated_uint8, max_val=255)
        ssim_scores.append(score.numpy())  # Convert the tensor to a numpy array

    # Return the average SSIM score
    return np.mean(ssim_scores)


# Load real and generated images
real_images_directory = r"C:\Users\kaleem\Learn\Wuerth\thesis\data\data_temp\datasets_temp\test_set\generated\super_impose\150535_new"
generated_images_directory = r"C:\Users\kaleem\Learn\Wuerth\thesis\data\data_temp\datasets_temp\test_set\generated\super_impose\150535"

real_images = load_images_from_directory(real_images_directory)
generated_images = load_images_from_directory(generated_images_directory)

fid_score = calculate_fid(real_images, generated_images)
print("FID Score:", fid_score)

ssim_score = calculate_ssim(real_images, generated_images)
print("Average SSIM Score:", ssim_score)