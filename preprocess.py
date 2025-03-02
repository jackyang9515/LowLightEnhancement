import os
import cv2
import numpy as np
import torch
from torchvision import datasets
from tqdm import tqdm
import matplotlib.pyplot as plt

"""
This file preprocesses the data in colorization/training_small and colorization/validation_small
Each file gets split into Y and UV tensors and gets saved to preprocessed_data/<<dataset-type>>/Y 
and preprocessed_data/<<dataset-type>>/UV 
"""




def verify_single_image(rgb_image, save_path=None):
    """
    Process a single image through YUV conversion and back to verify correctness.
    """
    # Convert to YUV
    yuv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)
    
    # Split channels
    Y, U, V = cv2.split(yuv)
    
    # Create tensors with correct normalization
    Y_tensor = torch.from_numpy(Y).unsqueeze(0).float() / 100.0  # Normalize Y to [0, 1]
    UV_tensor = torch.from_numpy(np.stack((U, V), axis=0)).float() / 128.0  # Normalize UV to [-1, 1]
    
    # Convert back for verification
    Y_recover = (Y_tensor.squeeze().numpy() * 100.0).astype(np.uint8)
    UV_recover = (UV_tensor.numpy() * 128.0).astype(np.uint8)
    
    # Reconstruct YUV image
    yuv_reconstructed = cv2.merge([Y_recover, UV_recover[0], UV_recover[1]])
    
    # Convert back to RGB
    rgb_reconstructed = cv2.cvtColor(yuv_reconstructed, cv2.COLOR_YUV2RGB)
    
    if save_path:
        # Plot original and reconstructed side by side
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_image)
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(rgb_reconstructed)
        plt.title('Reconstructed')
        plt.axis('off')
        
        plt.savefig(save_path)
        plt.close()
    
    return rgb_reconstructed

def preprocess_and_save(data_path, y_output_path, uv_output_path):
    """
    Preprocess images and save YUV tensors
    This function recombines the tensors Y and UV for first 5 images as a sanity check.
    """
    dataset = datasets.ImageFolder(data_path)
    os.makedirs("verification", exist_ok=True)

    for idx, (rgb, _) in tqdm(enumerate(dataset), total=len(dataset)):
        # Convert PIL image to numpy array
        rgb_np = np.array(rgb)
        
        # Resize to ensure consistent size (no need to do this)
        # rgb_np = cv2.resize(rgb_np, (256, 256))
        
        # Verify conversion process (save first few images for inspection)
        if idx < 5:
            verify_single_image(rgb_np, f"verification/verify_{idx}.png")
        
        # Convert RGB to YUV
        yuv = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2YUV)
        Y, U, V = cv2.split(yuv)
        
        # Create tensors with proper normalization
        Y_tensor = torch.from_numpy(Y).unsqueeze(0).float() / 100.0  # Y normalized to [0, 1]
        UV_tensor = torch.from_numpy(np.stack((U, V), axis=0)).float()
        UV_tensor = (UV_tensor - 128.0) / 128.0 # UV normalized to [-1, 1]

        # Save tensors
        torch.save(Y_tensor, os.path.join(y_output_path, f"Y_{idx}.pt"))
        torch.save(UV_tensor, os.path.join(uv_output_path, f"UV_{idx}.pt"))

if __name__ == "__main__":
    # Your existing directory setup code here
    train_path = "colorization/training_small/"
    train_y_path, train_uv_path = "preprocessed_data/training/Y/", "preprocessed_data/training/UV/"
    os.makedirs(train_y_path, exist_ok=True)
    os.makedirs(train_uv_path, exist_ok=True)

    valid_path = "colorization/validation_small/"
    valid_y_path = "preprocessed_data/validation/Y/"
    valid_uv_path = "preprocessed_data/validation/UV/"
    os.makedirs(valid_y_path, exist_ok=True)
    os.makedirs(valid_uv_path, exist_ok=True)
    
    preprocess_and_save(train_path, train_y_path, train_uv_path)
    preprocess_and_save(valid_path, valid_y_path, valid_uv_path)