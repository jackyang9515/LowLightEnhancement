import os
import cv2
import numpy as np
import torch
from torchvision import datasets
from tqdm import tqdm
import matplotlib.pyplot as plt

"""
This file preprocesses the data in colorization/training_small and colorization/validation_small
Each file gets split into L and AB tensors and gets saved to preprocessed_data/<<dataset-type>>/L 
and preprocessed_data/<<dataset-type>>/AB 
"""




def verify_single_image(rgb_image, save_path=None):
    """
    Process a single image through LAB conversion and back to verify correctness.
    """
    # Convert to LAB
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    
    # Split channels
    L, A, B = cv2.split(lab)
    
    # Create tensors with correct normalization
    L_tensor = torch.from_numpy(L).unsqueeze(0).float() / 100.0  # Normalize L to [0, 1]
    AB_tensor = torch.from_numpy(np.stack((A, B), axis=0)).float() / 128.0  # Normalize AB to [-1, 1]
    
    # Convert back for verification
    L_recover = (L_tensor.squeeze().numpy() * 100.0).astype(np.uint8)
    AB_recover = (AB_tensor.numpy() * 128.0).astype(np.uint8)
    
    # Reconstruct LAB image
    lab_reconstructed = cv2.merge([L_recover, AB_recover[0], AB_recover[1]])
    
    # Convert back to RGB
    rgb_reconstructed = cv2.cvtColor(lab_reconstructed, cv2.COLOR_LAB2RGB)
    
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

def preprocess_and_save(data_path, l_output_path, ab_output_path):
    """
    Preprocess images and save LAB tensors
    This function recombines the tensors L and AB for first 5 images as a sanity check.
    """
    dataset = datasets.ImageFolder(data_path)
    os.makedirs("verification", exist_ok=True)

    for idx, (rgb, _) in tqdm(enumerate(dataset), total=len(dataset)):
        # Convert PIL image to numpy array
        rgb_np = np.array(rgb)
        
        # Resize to ensure consistent size
        rgb_np = cv2.resize(rgb_np, (256, 256))
        
        # Verify conversion process (save first few images for inspection)
        if idx < 5:
            verify_single_image(rgb_np, f"verification/verify_{idx}.png")
        
        # Convert RGB to LAB
        lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)
        
        # Create tensors with proper normalization
        L_tensor = torch.from_numpy(L).unsqueeze(0).float() / 100.0  # L normalized to [0, 1]
        AB_tensor = torch.from_numpy(np.stack((A, B), axis=0)).float()   
        AB_tensor = (AB_tensor - 128.0) / 128.0 # AB normalized to [-1, 1]

        # Save tensors
        torch.save(L_tensor, os.path.join(l_output_path, f"L_{idx}.pt"))
        torch.save(AB_tensor, os.path.join(ab_output_path, f"AB_{idx}.pt"))

if __name__ == "__main__":
    # Your existing directory setup code here
    train_path = "colorization/training_small/"
    train_l_path, train_ab_path = "preprocessed_data/training/L/", "preprocessed_data/training/AB/"
    os.makedirs(train_l_path, exist_ok=True)
    os.makedirs(train_ab_path, exist_ok=True)

    valid_path = "colorization/validation_small/"
    valid_l_path = "preprocessed_data/validation/L/"
    valid_ab_path = "preprocessed_data/validation/AB/"
    os.makedirs(valid_l_path, exist_ok=True)
    os.makedirs(valid_ab_path, exist_ok=True)
    
    preprocess_and_save(train_path, train_l_path, train_ab_path)
    preprocess_and_save(valid_path, valid_l_path, valid_ab_path)