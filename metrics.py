import torch
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3
from PIL import Image


def calculate_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    sigma = 1.5
    gauss = torch.Tensor([
        np.exp(-(x - window_size//2)**2 / (2 * sigma**2)) 
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()
    
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(img1.size(1), 1, window_size, window_size).contiguous()
    window = window.to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class InceptionV3FeatureExtractor:
    
    def __init__(self, device='cpu'):
        self.device = device
        self.model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.model.eval()
        
        self.model.fc = torch.nn.Identity()
    
    def extract_features(self, images):
        if images.shape[-1] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        images = (images - 0.5) * 2 
        
        with torch.no_grad():
            features = self.model(images)
        
        return features


def evaluate_reconstruction(real_images, reconstructed_images, masks=None):
    real = (real_images + 1) / 2
    recon = (reconstructed_images + 1) / 2
    
    metrics = {}
    
    metrics['psnr_full'] = calculate_psnr(real, recon).item()
    metrics['ssim_full'] = calculate_ssim(real, recon).item()
    metrics['mae_full'] = calculate_mae(real, recon).item()
    metrics['mse_full'] = calculate_mse(real, recon).item()
    
    if masks is not None:
        occluded_mask = (1 - masks)
        
        real_occluded = real * occluded_mask
        recon_occluded = recon * occluded_mask
        
        metrics['psnr_occluded'] = calculate_psnr(real_occluded, recon_occluded).item()
        metrics['ssim_occluded'] = calculate_ssim(real_occluded, recon_occluded).item()
    
    return metrics


if __name__ == "__main__":
    print("Testing metrics...")
    
    real = torch.rand(4, 3, 128, 128)
    recon = real + torch.randn(4, 3, 128, 128) * 0.1 
    masks = torch.ones(4, 3, 128, 128)
    masks[:, :, 40:80, 40:80] = 0 
    
    metrics = evaluate_reconstruction(real * 2 - 1, recon * 2 - 1, masks)
    
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    print("\nMetrics test passed!")