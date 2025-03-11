import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from model import LYT
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
    model = LYT().to(device)
    criterion = CombinedLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_psnr = 0
    epochs = []
    psnr_losses = []
    ssim_losses = []
    color_losses = []
    ill_smooth_losses = []

    print('Training started.')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            y_inputs, uv_inputs, targets = batch
            y_inputs, uv_inputs, targets = y_inputs.to(device), uv_inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(y_inputs, uv_inputs)
            loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_psnr, avg_ssim, avg_color, avg_ill_smooth_loss = validate(model, valid_loader, device)
        
        epochs.append(epoch + 1)
        psnr_losses.append(avg_psnr)
        ssim_losses.append(avg_ssim)
        color_losses.append(avg_color)
        ill_smooth_losses.append(avg_ill_smooth_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, PSNR: {avg_psnr:.6f}, SSIM: {avg_ssim:.6f}')
        scheduler.step()

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saving model with PSNR: {best_psnr:.6f}')

    psnr_losses = torch.stack(psnr_losses, dim=0)
    ssim_losses = torch.stack(ssim_losses, dim=0)
    color_losses = torch.stack(color_losses, dim=0)
    ill_smooth_losses = torch.stack(ill_smooth_losses, dim=0)

    plt.plot(epochs, psnr_losses, label='PSNR')
    plt.plot(epochs, ssim_losses, label='SSIM')
    plt.plot(epochs, color_losses, label='Color')
    plt.plot(epochs, ill_smooth_losses, label='Illumination Smoothness')

    plt.title('Losses of Validation Set Throughout Model Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
