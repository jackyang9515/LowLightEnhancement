import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
from model import LYT, Discriminator
from losses import CombinedLoss
from torchvision.utils import save_image
from torchmetrics.functional import structural_similarity_index_measure
from dataset import create_dataloaders

def calculate_psnr(img1, img2, max_pixel_value=1.0):
    img1_gray = img1.mean(dim=1, keepdim=True)
    img2_gray = img2.mean(dim=1, keepdim=True)
    
    mean_restored = img1_gray.mean()
    mean_target = img2_gray.mean()
    img1_normalized = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)

    mse = nn.functional.mse_loss(img1_normalized, img2, reduction='mean')
    if mse < 1e-10:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2, max_pixel_value=1.0, gt_mean=True):
    """
    Calculate SSIM (Structural Similarity Index) between two images.

    Args:
        img1 (torch.Tensor): First image (BxCxHxW)
        img2 (torch.Tensor): Second image (BxCxHxW)
        max_pixel_value (float): The maximum possible pixel value of the images. Default is 1.0.

    Returns:
        float: The SSIM value.
    """
    if gt_mean:
        img1_gray = img1.mean(axis=1, keepdim=True)
        img2_gray = img2.mean(axis=1, keepdim=True)
        
        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)

    ssim_val = structural_similarity_index_measure(img1, img2, data_range=max_pixel_value)
    return ssim_val.item()

def main():
    # Hyperparameters
    learning_rate = 2e-4 
    num_epochs = 200
    batch_size = 1 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data paths
    train_low_yuv = "preprocessed_data/training/low_yuv/"
    train_high_rgb = "preprocessed_data/training/high_rgb/"
    val_low_yuv = "preprocessed_data/validation/low_yuv/"
    val_high_rgb = "preprocessed_data/validation/high_rgb/"
        
    train_loader, val_loader = create_dataloaders(train_low_yuv, train_high_rgb, val_low_yuv, val_high_rgb, crop_size=256, batch_size=1)
    
    generator = LYT().to(device)
    discriminator = Discriminator(input_nc=6).to(device)
    
    content_criterion = CombinedLoss(device)
    adversarial_criterion = nn.BCEWithLogitsLoss()
    
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    

    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='max', factor=0.5, patience=10)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='max', factor=0.5, patience=10)
    
    # Two-phase training approach
    use_gan = False  # Start with content losses only
    gan_start_epoch = 25  # Start GAN training after this epoch
    
    os.makedirs('results', exist_ok=True)
    
    best_psnr = 0
    
    for epoch in range(num_epochs):
        if epoch >= gan_start_epoch and not use_gan:
            use_gan = True
            print("Enabling GAN training...")
        
        # ----------- TRAINING PHASE -----------
        generator.train()
        discriminator.train()
        epoch_g_loss = 0
        epoch_d_loss = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                           desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (y, uv, target) in progress_bar:
            y, uv, target = y.to(device), uv.to(device), target.to(device)
            batch_size = y.size(0)
            
            # ----------------------
            # Train Discriminator if GAN is enabled
            # ----------------------
            if use_gan:
                optimizer_D.zero_grad()

                with torch.no_grad():
                    fake_images = generator(y, uv)
                
                # Prepare discriminator inputs
                y_expanded = y.repeat(1, 3, 1, 1)  # Expand Y to 3 channels
                real_pairs = torch.cat([y_expanded, target], dim=1)
                fake_pairs = torch.cat([y_expanded, fake_images.detach()], dim=1)
                
                # Create labels (with smoothing)
                real_label = torch.ones(batch_size, 1, 16, 16).to(device) * 0.9
                fake_label = torch.zeros(batch_size, 1, 16, 16).to(device) * 0.1

                real_output = discriminator(real_pairs)
                fake_output = discriminator(fake_pairs)
                
                # Adjust label size if needed
                if real_output.shape != real_label.shape:
                    real_label = torch.ones_like(real_output) * 0.9
                    fake_label = torch.zeros_like(fake_output) * 0.1

                d_loss_real = adversarial_criterion(real_output, real_label)
                d_loss_fake = adversarial_criterion(fake_output, fake_label)
                d_loss = (d_loss_real + d_loss_fake) * 0.5
                

                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                optimizer_D.step()
                
                epoch_d_loss += d_loss.item()
            else:
                d_loss = torch.tensor(0.0)  # Placeholder when GAN is disabled
                
            # ----------------------
            # Train Generator
            # ----------------------
            optimizer_G.zero_grad()
            
            fake_images = generator(y, uv)
            
            g_loss_content = content_criterion(target, fake_images)
            
            if use_gan:
                y_expanded = y.repeat(1, 3, 1, 1)
                fake_pairs = torch.cat([y_expanded, fake_images], dim=1)
                fake_output = discriminator(fake_pairs)
                real_label = torch.ones_like(fake_output) * 0.9
                g_loss_adv = adversarial_criterion(fake_output, real_label)
                
                # Combine losses with small weight for adversarial component
                lambda_adv = 0.001  # Small weight for GAN loss
                g_loss = g_loss_content + lambda_adv * g_loss_adv
            else:
                g_loss = g_loss_content
            
            g_loss.backward()
            optimizer_G.step()
            
            epoch_g_loss += g_loss.item()
            
            progress_bar.set_postfix({
                'G_loss': g_loss.item(),
                'D_loss': d_loss.item() if use_gan else 0.0
            })
        
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader) if use_gan else 0.0
        
        # ----------- VALIDATION PHASE -----------
        generator.eval()
        val_psnr = 0
        val_ssim = 0
        
        with torch.no_grad():
            for y, uv, target in val_loader:
                y, uv, target = y.to(device), uv.to(device), target.to(device)
                
                # Generate enhanced image
                output = generator(y, uv)
                
                # Calculate PSNR
                psnr = calculate_psnr(output, target)
                ssim = calculate_ssim(output, target)
                val_psnr += psnr
                val_ssim += ssim
        
        avg_psnr = val_psnr / len(val_loader)
        avg_ssim = val_ssim / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.6f}")
        
        # Save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(generator.state_dict(), 'models/best_generator.pth')
            if use_gan:
                torch.save(discriminator.state_dict(), 'models/best_discriminator.pth')
            print(f"Saved model with PSNR: {best_psnr:.4f}")
        else:
            early_stop_counter += 1

        scheduler_G.step(avg_psnr)
        if use_gan:
            scheduler_D.step(avg_psnr)
        
        # Save sample outputs
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                for i, (y, uv, target) in enumerate(val_loader):
                    if i >= 5:
                        break
                    
                    y, uv, target = y.to(device), uv.to(device), target.to(device)
                    output = generator(y, uv)
                    
                    # Save images
                    save_image(output, f'results/epoch_{epoch+1}_sample_{i}_output.png')
                    save_image(target, f'results/epoch_{epoch+1}_sample_{i}_target.png')

                    y_display = y.repeat(1, 3, 1, 1)  
                    save_image(y_display, f'results/epoch_{epoch+1}_sample_{i}_input.png')

if __name__ == "__main__":
    main()