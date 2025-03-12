import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

"""
This file preprocesses the data in LOLdataset/train450 and LOLdataset/valid35
- Low-light images are converted to YUV format and saved as tensors
- Target/high-light images are kept as RGB and saved as tensors
"""

def verify_yuv_conversion(img, save_path=None):
    """
    Verify the YUV conversion and back to RGB
    """
    # Convert to YUV
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    Y, U, V = cv2.split(yuv)
    
    # Create tensors (with normalization)
    Y_tensor = torch.from_numpy(Y).unsqueeze(0).float() / 255.0  # Normalize to 0-1
    UV_tensor = torch.from_numpy(np.stack((U, V), axis=0)).float() / 255.0  # Normalize to 0-1
    
    # Convert back for verification
    Y_recover = (Y_tensor.squeeze().numpy() * 255.0).astype(np.uint8)
    U_recover = (UV_tensor[0].numpy() * 255.0).astype(np.uint8)
    V_recover = (UV_tensor[1].numpy() * 255.0).astype(np.uint8)
    
    yuv_reconstructed = cv2.merge([Y_recover, U_recover, V_recover])
    rgb_reconstructed = cv2.cvtColor(yuv_reconstructed, cv2.COLOR_YUV2RGB)
    
    if save_path:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title(f'Original RGB')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(rgb_reconstructed)
        plt.title(f'YUV→RGB')
        plt.axis('off')
        
        # Display the Y channel (grayscale)
        plt.subplot(1, 3, 3)
        plt.imshow(Y_recover, cmap='gray')
        plt.title(f'Y Channel')
        plt.axis('off')
        
        plt.savefig(f"{save_path}.png")
        plt.close()

def preprocess_and_save(dataset_path, output_path_low_yuv, output_path_high_rgb):
    """
    Preprocess images:
    - Low-light images → YUV format
    - High-light images → RGB format
    """
    os.makedirs(output_path_low_yuv, exist_ok=True)
    os.makedirs(output_path_high_rgb, exist_ok=True)
    
    high_dir = os.path.join(dataset_path, "high")
    low_dir = os.path.join(dataset_path, "low")
    print(high_dir)
    
    image_files = sorted(os.listdir(high_dir))
    os.makedirs("verification", exist_ok=True)
    
    for idx, filename in tqdm(enumerate(image_files), total=len(image_files)):
        high_img_path = os.path.join(high_dir, filename)
        low_img_path = os.path.join(low_dir, filename)
        print(high_img_path, )
        
        if not os.path.exists(low_img_path):
            print(f"Warning: Missing low-light image for {filename}, skipping...")
            continue
        
        # Read images and convert from BGR to RGB
        high_img = cv2.imread(high_img_path)
        low_img = cv2.imread(low_img_path)
        
        high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)
        low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
        
        # Verify a few images
        if idx < 5:
            verify_yuv_conversion(low_img, f"verification/verify_yuv_{idx}")
        
        # Process low-light image to YUV
        low_yuv = cv2.cvtColor(low_img, cv2.COLOR_RGB2YUV)
        Y, U, V = cv2.split(low_yuv)
        
        # Create normalized tensors
        Y_tensor = torch.from_numpy(Y).unsqueeze(0).float() / 255.0  # Normalize to 0-1
        UV_tensor = torch.from_numpy(np.stack((U, V), axis=0)).float() / 255.0  # Normalize to 0-1
        
        # Process high-light image to RGB tensor
        high_tensor = torch.from_numpy(high_img.transpose(2, 0, 1)).float() / 255.0  # Normalize to 0-1
        
        # Save tensors
        torch.save(Y_tensor, os.path.join(output_path_low_yuv, f"low_Y_{idx}.pt"))
        torch.save(UV_tensor, os.path.join(output_path_low_yuv, f"low_UV_{idx}.pt"))
        torch.save(high_tensor, os.path.join(output_path_high_rgb, f"high_RGB_{idx}.pt"))

if __name__ == "__main__":
    # Directory setup
    train_path = "LOLdataset/train450/"
    train_output_low_yuv = "preprocessed_data/training/low_yuv/"
    train_output_high_rgb = "preprocessed_data/training/high_rgb/"
    
    valid_path = "LOLdataset/valid35/"
    valid_output_low_yuv = "preprocessed_data/validation/low_yuv/"
    valid_output_high_rgb = "preprocessed_data/validation/high_rgb/"
    
    # Create output directories
    for dir_path in [train_output_low_yuv, train_output_high_rgb,
                    valid_output_low_yuv, valid_output_high_rgb]:
        os.makedirs(dir_path, exist_ok=True)
    
    print("Processing training images...")
    preprocess_and_save(train_path, train_output_low_yuv, train_output_high_rgb)
    
    print("Processing validation images...")
    preprocess_and_save(valid_path, valid_output_low_yuv, valid_output_high_rgb)
    
    print("Preprocessing complete!")