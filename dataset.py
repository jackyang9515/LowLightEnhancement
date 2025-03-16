import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
import torch.nn.functional as F

class YUVRGBPairedDataset(Dataset):
    """Dataset for loading pre-converted YUV and RGB tensors with crop and rotation support"""
    def __init__(self, low_yuv_dir, high_rgb_dir, crop_size=None, training=True):
        super(YUVRGBPairedDataset, self).__init__()
        self.low_yuv_dir = low_yuv_dir
        self.high_rgb_dir = high_rgb_dir
        self.crop_size = crop_size
        self.training = training
        
        self.low_y_files = sorted([f for f in os.listdir(low_yuv_dir) if f.startswith('low_Y_')])
        
        self.indices = [int(f.split('_')[-1].split('.')[0]) for f in self.low_y_files]
    
    def __len__(self):
        return len(self.low_y_files)
    
    def random_crop(self, tensors, crop_size):
        """Apply the same random crop to all tensors"""
        h, w = tensors[0].shape[-2:] 
        
        # Generate random crop coordinates
        if h < crop_size or w < crop_size:
            # If image is smaller than crop size, resize it
            new_h = max(h, crop_size)
            new_w = max(w, crop_size)
            resized_tensors = []
            for tensor in tensors:
                if tensor.dim() == 3:  # 3D tensor [C, H, W]
                    resized_tensors.append(F.interpolate(tensor.unsqueeze(0), size=(new_h, new_w), 
                                                         mode='bilinear', align_corners=False).squeeze(0))
                elif tensor.dim() == 4:  # 4D tensor [B, C, H, W]
                    resized_tensors.append(F.interpolate(tensor, size=(new_h, new_w), 
                                                        mode='bilinear', align_corners=False))
            tensors = resized_tensors
            h, w = new_h, new_w
        
        # Get random crop coordinates
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        
        # Apply crop to all tensors
        cropped_tensors = []
        for tensor in tensors:
            cropped = tensor[..., top:top+crop_size, left:left+crop_size]
            cropped_tensors.append(cropped)
            
        return cropped_tensors
    
    def random_rotate(self, tensors):
        """Apply the same random rotation to all tensors"""
        # Choose a random rotation angle from 0, 90, 180, 270 degrees
        k = random.randint(0, 3)  # 0=0°, 1=90°, 2=180°, 3=270°
        
        if k > 0:  # Only rotate if k > 0
            rotated_tensors = []
            for tensor in tensors:
                # torch.rot90 rotates in the plane specified by dims
                rotated = torch.rot90(tensor, k, dims=[-2, -1])
                rotated_tensors.append(rotated)
            return rotated_tensors
        return tensors
    
    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        
        low_y_path = os.path.join(self.low_yuv_dir, f"low_Y_{sample_idx}.pt")
        low_y_tensor = torch.load(low_y_path)
        
        low_uv_path = os.path.join(self.low_yuv_dir, f"low_UV_{sample_idx}.pt")
        low_uv_tensor = torch.load(low_uv_path)
        
        high_rgb_path = os.path.join(self.high_rgb_dir, f"high_RGB_{sample_idx}.pt")
        high_rgb_tensor = torch.load(high_rgb_path)
        
        # Apply transformations only during training
        if self.training:
            # Apply random crop if crop_size is specified
            if self.crop_size:
                low_y_tensor, low_uv_tensor, high_rgb_tensor = self.random_crop(
                    [low_y_tensor, low_uv_tensor, high_rgb_tensor], 
                    self.crop_size
                )
                
            # Apply random rotation
            low_y_tensor, low_uv_tensor, high_rgb_tensor = self.random_rotate(
                [low_y_tensor, low_uv_tensor, high_rgb_tensor]
            )
        
        return low_y_tensor, low_uv_tensor, high_rgb_tensor

def create_dataloaders(train_low_yuv, train_high_rgb, test_low_yuv, test_high_rgb, crop_size=256, batch_size=1):
    train_loader = None
    test_loader = None
    
    if train_low_yuv and train_high_rgb:
        train_dataset = YUVRGBPairedDataset(
            train_low_yuv, 
            train_high_rgb, 
            crop_size=crop_size, 
            training=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    if test_low_yuv and test_high_rgb:
        test_dataset = YUVRGBPairedDataset(
            test_low_yuv, 
            test_high_rgb, 
            training=False
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    return train_loader, test_loader