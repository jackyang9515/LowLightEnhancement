import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ms_ssim
import torchvision.transforms as T

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=True).features[:16]  # Until block3_conv3
        self.loss_model = vgg.to(device).eval()
        for param in self.loss_model.parameters():
            param.requires_grad = False

    def forward(self, y_true, y_pred):
        y_true, y_pred = y_true.to(next(self.loss_model.parameters()).device), y_pred.to(next(self.loss_model.parameters()).device)
        return F.mse_loss(self.loss_model(y_true), self.loss_model(y_pred))


def color_loss(y_true, y_pred):
    return torch.mean(torch.abs(torch.mean(y_true, dim=[1, 2, 3]) - torch.mean(y_pred, dim=[1, 2, 3])))

def psnr_loss(y_true, y_pred):
    mse = F.mse_loss(y_true, y_pred)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return 40.0 - torch.mean(psnr)

def smooth_l1_loss(y_true, y_pred):
    return F.smooth_l1_loss(y_true, y_pred)

def multiscale_ssim_loss(y_true, y_pred, max_val=1.0, power_factors=[0.5, 0.5]):
    return 1.0 - ms_ssim(y_true, y_pred, data_range=max_val, size_average=True)

def gaussian_kernel(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

def histogram_loss(y_true, y_pred, bins=256, sigma=0.01):
    
    bin_edges = torch.linspace(0.0, 1.0, bins, device=y_true.device)

    y_true_hist = torch.sum(gaussian_kernel(y_true.unsqueeze(-1), bin_edges, sigma), dim=0)
    y_pred_hist = torch.sum(gaussian_kernel(y_pred.unsqueeze(-1), bin_edges, sigma), dim=0)
    
    y_true_hist /= y_true_hist.sum()
    y_pred_hist /= y_pred_hist.sum()

    hist_distance = torch.mean(torch.abs(y_true_hist - y_pred_hist))
    return hist_distance

def exposure_loss(y_pred):
    pixel_means = torch.mean(y_true, dim=1, keepdim=True)
    pooling = torch.nn.AvgPool2d(16, 16) 
    mean = pooling(pixel_means)
    return torch.mean(torch.square(mean - 0.6))

def illumnation_smoothness_loss(y_pred):
    batches = y_pred.size()[0]
    channels = y_pred.size()[1]
    height = y_pred.size()[2]
    width = y_pred.size()[3]

    height_count = (width - 1) * channels
    width_count = width * (channels - 1)

    height_variance = torch.sum(torch.square((y_pred[:, :, 1:, :] - y_pred[:, :, :height - 1, :])))
    width_variance = torch.sum(torch.square((x[:, :, :, 1:] - x[:, :, :, :width - 1])))

    return 2 * (height_variance / height_count + width_variance / width_count) / batches

def spatial_consistency_loss(y_true, y_pred):
  left_kernel = torch.tensor([[[[0, 0, 0]], [[-1, 1, 0]], [[0, 0, 0]]]])
  right_kernel = torch.tensor([[[[0, 0, 0]], [[0, 1, -1]], [[0, 0, 0]]]])
  up_kernel = torch.tensor([[[[0, -1, 0]], [[0, 1, 0]], [[0, 0, 0]]]])
  down_kernel = torch.tensor([[[[0, 0, 0]], [[0, 1, 0]], [[0, -1, 0]]]])
  
  original_mean = torch.mean(y_true, dim=1, keepdim=True)
  enhanced_mean = torch.mean(y_pred, dim=1, keepdim=True)

  pooling = torch.nn.AvgPool2d(4, 4)

  original_avg_pool = pooling(original_mean)
  enhanced_avg_pool = pooling(enhanced_mean)
  
  k_original_left = F.conv2d(original_avg_pool, left_kernel, stride=(1, 1, 1, 1), padding=1)
  k_original_right = F.conv2d(original_avg_pool, right_kernel, strides=(1, 1, 1, 1), padding=1)
  k_original_up = F.conv2d(original_avg_pool, up_kernel, strides=(1, 1, 1, 1), padding=1)
  k_original_down = F.conv2d(original_avg_pool, down_kernel, strides(1, 1, 1, 1), padding=1)

  k_enhanced_left = F.conv2d(enhanced_avg_pool, left_kernel, strides=(1, 1, 1, 1), padding=1)
  k_enhanced_right = F.conv2d(enhanced_avg_pool, right_kernel, strides=(1, 1, 1, 1), padding=1)
  k_enhanced_up = F.conv2d(enhanced_avg_pool, up_kernel, strides=(1, 1, 1, 1), padding=1)
  k_enhanced_down = F.conv2d(enhanced_avg_pool, down_kernel, strides=(1, 1, 1, 1), padding=1)

  k_left = torch.square(k_original_left - k_enhanced_left)
  k_right = torch.square(k_original_right - k_enhanced_right)
  k_up = torch.square(k_original_up - k_enhanced_up)
  k_down = torch.square(k_original_down - k_enhanced_down)

  return k_left + k_right + k_up + k_down

class CombinedLoss(nn.Module):
    def __init__(self, device):
        super(CombinedLoss, self).__init__()
        self.perceptual_loss_model = VGGPerceptualLoss(device)
        self.alpha1 = 1.00
        self.alpha2 = 0.06
        self.alpha3 = 0.05
        self.alpha4 = 0.5
        self.alpha5 = 0.0083
        self.alpha6 = 0.25
        self.alpha7 = 0.1

    def forward(self, y_true, y_pred):
        smooth_l1_l = smooth_l1_loss(y_true, y_pred)
        ms_ssim_l = multiscale_ssim_loss(y_true, y_pred)
        perc_l = self.perceptual_loss_model(y_true, y_pred)
        hist_l = histogram_loss(y_true, y_pred)
        psnr_l = psnr_loss(y_true, y_pred)
        color_l = color_loss(y_true, y_pred)
        exposure_l = exposure_loss(y_pred)
        ill_smooth_l = illumnation_smoothness_loss(y_pred)
        spatial_const_l = spatial_consistency_loss(y_true, y_pred)

        total_loss = (self.alpha1 * smooth_l1_l + self.alpha2 * perc_l + 
                      self.alpha3 * hist_l + self.alpha5 * psnr_l + 
                      self.alpha6 * color_l + self.alpha4 * ms_ssim_l +
                      self.alpha7 * exposure_l + self.alpha7 * ill_smooth_l +
                      self.alpha7 * spatial_const_l)

        return torch.mean(total_loss)
