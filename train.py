import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from model import LYT, Discriminator
from losses import CombinedLoss, color_loss, psnr_loss, multiscale_ssim_loss, illumnation_smoothness_loss
from preprocess import preprocess_and_save
import os

def validate(model, dataloader, device):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    total_color = 0
    total_ill_smooth_loss = 0
    with torch.no_grad():
        for y, uv, high in dataloader:
            y, uv, high = y.to(device), uv.to(device), high.to(device)
            output = model(y, uv)

            # Calculate PSNR
            psnr = psnr_loss(high, output)
            total_psnr += psnr

            # Calculate SSIM
            ssim = multiscale_ssim_loss(high, output)
            total_ssim += ssim

            # Calculate color loss
            color = color_loss(high, output)
            total_color += color

            # Calculate illumination smoothness loss
            ill_smooth_loss = illumnation_smoothness_loss(output)
            total_ill_smooth_loss += ill_smooth_loss

    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    avg_color = total_color / len(dataloader)
    avg_ill_smooth_loss = total_ill_smooth_loss / len(dataloader)

    return avg_psnr, avg_ssim, avg_color, avg_ill_smooth_loss

class PairedDataset(Dataset):
    def __init__(self, y_dir, uv_dir, high_dir):
        self.y_dir = y_dir
        self.uv_dir = uv_dir
        self.high_dir = high_dir
        self.y_images = sorted([f for f in os.listdir(y_dir) if os.path.isfile(os.path.join(y_dir, f))])
        self.uv_images = sorted([f for f in os.listdir(uv_dir) if os.path.isfile(os.path.join(uv_dir, f))])
        self.high_images = sorted([f for f in os.listdir(high_dir) if os.path.isfile(os.path.join(high_dir, f))])

    def __len__(self):
        return len(self.high_images)

    def __getitem__(self, idx):
        y_image_path = os.path.join(self.y_dir, self.y_dir[idx])
        y_image = torch.load(y_image_path)

        uv_image_path = os.path.join(self.uv_dir, self.uv_dir[idx])
        uv_image = torch.load(uv_image_path)

        high_image_path = os.path.join(self.high_dir, self.high_images[idx])
        high_image = Image.open(high_image_path).convert('RGB')
        high_image = self.transform(high_image)

        return y_image, uv_image, high_image

def main():
    # Hyperparameters
    train_low_path = "LOLdataset/train450/low/"
    train_high_path = "LOLdataset/train450/high/"
    train_y_path, train_uv_path = "preprocessed_data/training/Y/", "preprocessed_data/training/UV/"
    os.makedirs(train_y_path, exist_ok=True)
    os.makedirs(train_uv_path, exist_ok=True)

    valid_low_path = "LOLdataset/valid35/low/"
    valid_high_path = "LOLdataset/valid35/high/"
    valid_y_path = "preprocessed_data/validation/Y/"
    valid_uv_path = "preprocessed_data/validation/UV/"
    os.makedirs(valid_y_path, exist_ok=True)
    os.makedirs(valid_uv_path, exist_ok=True)

    preprocess_and_save(train_low_path, train_y_path, train_uv_path)
    preprocess_and_save(valid_low_path, valid_y_path, valid_uv_path)

    train_dataset = PairedDataset(train_y_path, train_uv_path, train_high_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)

    valid_dataset = PairedDataset(valid_y_path, valid_uv_path, valid_high_path)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=4)

    learning_rate = 2e-4
    num_epochs = 1500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'LR: {learning_rate}; Epochs: {num_epochs}')

    # Model, loss, optimizer, and scheduler
    generator = LYT().to(device)
    discriminator = Discriminator(input_nc=3).to(device)

    criterion_g = CombinedLoss(device)
    criterion_d = nn.BCEWithLogitsLoss()

    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

    scheduler_g = CosineAnnealingLR(optimizer_g, T_max=num_epochs)
    scheduler_d = CosineAnnealingLR(optimizer_d, T_max=num_epochs)

    scaler = torch.cuda.amp.GradScaler()

    best_psnr = 0
    epochs = []
    psnr_losses = []
    ssim_losses = []
    color_losses = []
    ill_smooth_losses = []

    print('Training started.')
    for epoch in range(num_epochs):

        generator.train()
        discriminator.train()
        train_loss_g = 0.0
        train_loss_d = 0.0

        for batch_idx, batch in enumerate(train_loader):
            y_inputs, uv_inputs, targets = batch
            y_inputs, uv_inputs, targets = y_inputs.to(device), uv_inputs.to(device), targets.to(device)

            # =========================
            # 1) TRAIN DISCRIMINATOR
            # =========================
            optimizer_d.zero_grad()

            # 1a) Real images
            real_labels = torch.ones((targets.size(0), 1, 30, 30), device=device)   # shape matches D output
            fake_labels = torch.zeros((targets.size(0), 1, 30, 30), device=device)
            
            # Discriminator output on real images
            real_preds = discriminator(targets)
            real_loss = criterion_d(real_preds, real_labels)

            # 1b) Fake images
            with torch.no_grad():
                fake_images = generator(y_inputs, uv_inputs)  # G in eval mode for D’s perspective
            fake_preds = discriminator(fake_images)
            fake_loss = criterion_d(fake_preds, fake_labels)

            # Total D loss
            d_loss = (real_loss + fake_loss) * 0.5

            # Backprop D
            scaler.scale(d_loss).backward()
            scaler.step(optimizer_d)
            scaler.update()

            train_loss_d += d_loss.item()

            # ======================
            # 2) TRAIN GENERATOR
            # ======================
            optimizer_g.zero_grad()

            # Re‐generate fake images *without* no_grad(), so it accumulates gradients
            fake_images = generator(y_inputs, uv_inputs)

            # 2a) Adversarial loss: we want disc(fake) → real_labels
            adv_preds = discriminator(fake_images)
            g_adv_loss = criterion_d(adv_preds, real_labels)
            
            # 2b) Combine with your existing "content" or "perceptual" losses
            # CombinedLoss or any other custom losses you want
            g_content_loss = criterion_g(fake_images, targets)
            
            # Suppose you blend them, e.g. total_g_loss = content_loss + lambda_adv * adv_loss
            # You can tune lambda_adv to balance how strongly you want the adversarial push
            lambda_adv = 0.001
            g_loss = g_content_loss + lambda_adv * g_adv_loss

            # Backprop G
            scaler.scale(g_loss).backward()
            scaler.step(optimizer_g)
            scaler.update()

            train_loss_g += g_loss.item()

        avg_psnr, avg_ssim, avg_color, avg_ill_smooth_loss = validate(generator, valid_loader, device)
        
        epochs.append(epoch + 1)
        psnr_losses.append(avg_psnr)
        ssim_losses.append(avg_ssim)
        color_losses.append(avg_color)
        ill_smooth_losses.append(avg_ill_smooth_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, PSNR: {avg_psnr:.6f}, SSIM: {avg_ssim:.6f}')
        scheduler_g.step()
        scheduler_d.step()

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(generator.state_dict(), 'best_generator.pth')
            print(f'Saving generator with PSNR: {best_psnr:.6f}')

    psnr_losses = torch.stack(psnr_losses, dim=0)
    ssim_losses = torch.stack(ssim_losses, dim=0)
    color_losses = torch.stack(color_losses, dim=0)
    ill_smooth_losses = torch.stack(ill_smooth_losses, dim=0)

    plt.plot(epochs, psnr_losses, label='PSNR')
    plt.plot(epochs, ssim_losses, label='SSIM')
    plt.plot(epochs, color_losses, label='Color')
    plt.plot(epochs, ill_smooth_losses, label='Illumination Smoothness')

    plt.title('Losses of Validation Set Throughout generator Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
