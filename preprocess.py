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



def verify_image_pair(high_img, low_img, save_path=None):
    """
    Process an image pair through YUV conversion and back to verify correctness.
    """
    for img, label in zip([high_img, low_img], ["high", "low"]):
        yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        Y, U, V = cv2.split(yuv)
        
        Y_tensor = torch.from_numpy(Y).unsqueeze(0).float() / 100.0
        UV_tensor = torch.from_numpy(np.stack((U, V), axis=0)).float() / 128.0
        
        Y_recover = (Y_tensor.squeeze().numpy() * 100.0).astype(np.uint8)
        UV_recover = (UV_tensor.numpy() * 128.0).astype(np.uint8)
        
        yuv_reconstructed = cv2.merge([Y_recover, UV_recover[0], UV_recover[1]])
        rgb_reconstructed = cv2.cvtColor(yuv_reconstructed, cv2.COLOR_YUV2RGB)
        
        if save_path:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title(f'Original ({label})')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(rgb_reconstructed)
            plt.title(f'Reconstructed ({label})')
            plt.axis('off')
            
            plt.savefig(f"{save_path}_{label}.png")
            plt.close()


def preprocess_and_save(dataset_path, y_output_path, uv_output_path):
    os.makedirs(y_output_path, exist_ok=True)
    os.makedirs(uv_output_path, exist_ok=True)
    
    high_dir = os.path.join(dataset_path, "high")
    low_dir = os.path.join(dataset_path, "low")
    
    image_files = sorted(os.listdir(high_dir))
    os.makedirs("verification", exist_ok=True)
    
    for idx, filename in tqdm(enumerate(image_files), total=len(image_files)):
        high_img_path = os.path.join(high_dir, filename)
        low_img_path = os.path.join(low_dir, filename)
        
        if not os.path.exists(low_img_path):
            print(f"Warning: Missing low-light image for {filename}, skipping...")
            continue
        
        high_img = cv2.imread(high_img_path)
        low_img = cv2.imread(low_img_path)
        
        high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)
        low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
        
        if idx < 5:
            verify_image_pair(high_img, low_img, f"verification/verify_{idx}")
        
        for img, label in zip([high_img, low_img], ["high", "low"]):
            yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            Y, U, V = cv2.split(yuv)
            
            Y_tensor = torch.from_numpy(Y).unsqueeze(0).float() / 100.0
            UV_tensor = torch.from_numpy(np.stack((U, V), axis=0)).float()
            UV_tensor = (UV_tensor - 128.0) / 128.0
            
            torch.save(Y_tensor, os.path.join(y_output_path, f"{label}_Y_{idx}.pt"))
            torch.save(UV_tensor, os.path.join(uv_output_path, f"{label}_UV_{idx}.pt"))

if __name__ == "__main__":
    # Your existing directory setup code here
    train_path = "LOLdataset/train450/low/"
    train_y_path, train_uv_path = "preprocessed_data/training/Y/", "preprocessed_data/training/UV/"
    os.makedirs(train_y_path, exist_ok=True)
    os.makedirs(train_uv_path, exist_ok=True)

    valid_path = "LOLdataset/valid35/low/"
    valid_y_path = "preprocessed_data/validation/Y/"
    valid_uv_path = "preprocessed_data/validation/UV/"
    os.makedirs(valid_y_path, exist_ok=True)
    os.makedirs(valid_uv_path, exist_ok=True)
    
    preprocess_and_save(train_path, train_y_path, train_uv_path)
    preprocess_and_save(valid_path, valid_y_path, valid_uv_path)