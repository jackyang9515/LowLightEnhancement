import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
from model import LYT, Discriminator
from losses import CombinedLoss
from torchvision.utils import save_image

class YUVRGBPairedDataset(Dataset):
    """Dataset for loading pre-converted YUV and RGB tensors"""
    def __init__(self, low_yuv_dir, high_rgb_dir):
        super(YUVRGBPairedDataset, self).__init__()
        self.low_yuv_dir = low_yuv_dir
        self.high_rgb_dir = high_rgb_dir
        
        # Get all Y component files for low-light images
        self.low_y_files = sorted([f for f in os.listdir(low_yuv_dir) if f.startswith('low_Y_')])
        
        # Get indices from filenames
        self.indices = [int(f.split('_')[-1].split('.')[0]) for f in self.low_y_files]
    
    def __len__(self):
        return len(self.low_y_files)
    
    def __getitem__(self, idx):
        # Get the actual index for this sample
        sample_idx = self.indices[idx]
        
        # Load the Y component of low-light image
        low_y_path = os.path.join(self.low_yuv_dir, f"low_Y_{sample_idx}.pt")
        low_y_tensor = torch.load(low_y_path)
        
        # Load the UV components of low-light image
        low_uv_path = os.path.join(self.low_yuv_dir, f"low_UV_{sample_idx}.pt")
        low_uv_tensor = torch.load(low_uv_path)
        
        # Load the high-light RGB image
        high_rgb_path = os.path.join(self.high_rgb_dir, f"high_RGB_{sample_idx}.pt")
        high_rgb_tensor = torch.load(high_rgb_path)
        
        return low_y_tensor, low_uv_tensor, high_rgb_tensor

def calculate_psnr(img1, img2, max_pixel_value=1.0):
    # Add a small epsilon to avoid division by zero
    mse = nn.functional.mse_loss(img1, img2)
    if mse < 1e-10:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel_value / (torch.sqrt(mse) + 1e-10))
    return psnr.item()

def main():
    # Hyperparameters
    learning_rate = 2e-4  # Lowered for stability
    num_epochs = 500
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data paths
    train_low_yuv = "preprocessed_data/training/low_yuv/"
    train_high_rgb = "preprocessed_data/training/high_rgb/"
    val_low_yuv = "preprocessed_data/validation/low_yuv/"
    val_high_rgb = "preprocessed_data/validation/high_rgb/"
    
    # Create datasets and dataloaders
    train_dataset = YUVRGBPairedDataset(train_low_yuv, train_high_rgb)
    val_dataset = YUVRGBPairedDataset(val_low_yuv, val_high_rgb)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=7,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize models
    generator = LYT().to(device)
    discriminator = Discriminator(input_nc=6).to(device)
    
    # Loss functions
    content_criterion = CombinedLoss(device)
    adversarial_criterion = nn.BCEWithLogitsLoss()
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    # Schedulers
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=num_epochs)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=num_epochs)
    
    # Results directory
    os.makedirs('results', exist_ok=True)
    
    # Training loop
    best_psnr = 0
    
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        # Progress bar
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                            desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (y, uv, target) in progress_bar:
            y, uv, target = y.to(device), uv.to(device), target.to(device)
            batch_size = y.size(0)
            
            # ----------------------
            # Train Discriminator
            # ----------------------
            optimizer_D.zero_grad()
            
            # Generate fake images
            with torch.no_grad():
                fake_images = generator(y, uv)
            
            # Prepare discriminator inputs
            y_expanded = y.repeat(1, 3, 1, 1)  # Expand Y to 3 channels
            real_pairs = torch.cat([y_expanded, target], dim=1)
            fake_pairs = torch.cat([y_expanded, fake_images.detach()], dim=1)
            
            # Create labels (with smoothing)
            real_label = torch.ones(batch_size, 1, 16, 16).to(device) * 0.9
            fake_label = torch.zeros(batch_size, 1, 16, 16).to(device) * 0.1
            
            # Forward pass
            real_output = discriminator(real_pairs)
            fake_output = discriminator(fake_pairs)
            
            # Adjust label size if needed
            if real_output.shape != real_label.shape:
                real_label = torch.ones_like(real_output) * 0.9
                fake_label = torch.zeros_like(fake_output) * 0.1
            
            # Calculate loss
            d_loss_real = adversarial_criterion(real_output, real_label)
            d_loss_fake = adversarial_criterion(fake_output, fake_label)
            d_loss = (d_loss_real + d_loss_fake) * 0.5
            
            # Backpropagation
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_D.step()
            
            # ----------------------
            # Train Generator
            # ----------------------
            optimizer_G.zero_grad()
            
            # Generate fake images
            fake_images = generator(y, uv)
            
            # Prepare discriminator input for generator training
            fake_pairs = torch.cat([y_expanded, fake_images], dim=1)
            
            # Forward pass
            fake_output = discriminator(fake_pairs)
            
            # Calculate losses
            g_loss_adv = adversarial_criterion(fake_output, real_label)
            g_loss_content = content_criterion(target, fake_images)
            
            # Combine losses with small weight for adversarial component
            lambda_adv = 0.0005  # Even smaller for initial stability
            g_loss = g_loss_content + lambda_adv * g_loss_adv
            
            # Backpropagation
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()
            
            # Update running losses
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'G_loss': g_loss.item(),
                'D_loss': d_loss.item()
            })
        
        # Calculate average losses
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        
        # Validation
        generator.eval()
        val_psnr = 0
        val_count = 0
        
        with torch.no_grad():
            for y, uv, target in val_loader:
                y, uv, target = y.to(device), uv.to(device), target.to(device)
                
                # Generate enhanced image
                output = generator(y, uv)
                
                # Calculate PSNR
                psnr = calculate_psnr(output, target)
                val_psnr += psnr
                val_count += 1
        
        avg_psnr = val_psnr / val_count
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs} - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}, PSNR: {avg_psnr:.4f}")
        
        # Save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(generator.state_dict(), 'results/best_generator.pth')
            torch.save(discriminator.state_dict(), 'results/best_discriminator.pth')
            print(f"Saved model with PSNR: {best_psnr:.4f}")
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
        
        # Save sample outputs
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                for i, (y, uv, target) in enumerate(val_loader):
                    if i >= 5:  # Just save a few samples
                        break
                    
                    y, uv, target = y.to(device), uv.to(device), target.to(device)
                    output = generator(y, uv)
                    
                    # Save images
                    save_image(output, f'results/epoch_{epoch+1}_sample_{i}_output.png')
                    save_image(target, f'results/epoch_{epoch+1}_sample_{i}_target.png')
                    
                    # Display Y channel (if desired)
                    y_display = y.repeat(1, 3, 1, 1)  # Make grayscale Y into 3 channels
                    save_image(y_display, f'results/epoch_{epoch+1}_sample_{i}_input.png')

if __name__ == "__main__":
    main()