"""
Loss Functions for CT-to-PET Diffusion Model.

This module implements:
1. WeightedPETLoss: Emphasizes high SUV (tumor) regions
2. FocalFrequencyLoss: Frequency domain alignment
3. CombinedDiffusionLoss: Combines all losses for training

Reference: Designed for medical image synthesis where tumor regions are critical.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class WeightedPETLoss(nn.Module):
    """
    Weighted L1 Loss for PET image reconstruction.
    
    PET images are sparse - most pixels are background (low SUV values),
    while tumors appear as bright hotspots (high SUV values).
    Standard MSE/L1 loss causes the model to generate blurry or blank images.
    
    This loss assigns higher weights to high-intensity regions (potential tumors).
    
    Args:
        threshold: Normalized SUV threshold for "high intensity" (default: 0.3)
        high_weight: Weight multiplier for high SUV regions (default: 10.0)
        low_weight: Weight for background regions (default: 1.0)
        loss_type: 'l1' or 'l2' (default: 'l1')
    """
    def __init__(
        self,
        threshold: float = 0.3,
        high_weight: float = 10.0,
        low_weight: float = 1.0,
        loss_type: str = 'l1',
    ):
        super().__init__()
        self.threshold = threshold
        self.high_weight = high_weight
        self.low_weight = low_weight
        self.loss_type = loss_type
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute weighted loss.
        
        Args:
            pred: Predicted PET image, shape (B, C, H, W)
            target: Ground truth PET image, shape (B, C, H, W)
            mask: Optional external mask for tumor regions, shape (B, 1, H, W)
        
        Returns:
            Weighted loss scalar
        """
        # Compute pixel-wise error
        if self.loss_type == 'l1':
            error = torch.abs(pred - target)
        else:  # l2
            error = (pred - target) ** 2
        
        # Create weight map based on target intensity
        if mask is not None:
            # Use provided tumor mask
            weight_map = torch.where(
                mask > 0.5,
                torch.full_like(mask, self.high_weight),
                torch.full_like(mask, self.low_weight)
            )
        else:
            # Auto-generate weight map from target intensity
            weight_map = torch.where(
                target > self.threshold,
                torch.full_like(target, self.high_weight),
                torch.full_like(target, self.low_weight)
            )
        
        # Apply weights and compute mean
        weighted_error = error * weight_map
        loss = weighted_error.mean()
        
        return loss


class GradientLoss(nn.Module):
    """
    Gradient/Edge preservation loss.
    
    Encourages the model to preserve edges and structural boundaries,
    which is important for anatomical accuracy.
    """
    def __init__(self):
        super().__init__()
        # Sobel filters for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient loss between pred and target.
        
        Args:
            pred: Predicted image, shape (B, C, H, W)
            target: Ground truth image, shape (B, C, H, W)
        
        Returns:
            Gradient loss scalar
        """
        # Handle multi-channel by processing each channel
        B, C, H, W = pred.shape
        
        loss = 0.0
        for c in range(C):
            pred_c = pred[:, c:c+1, :, :]
            target_c = target[:, c:c+1, :, :]
            
            # Compute gradients
            pred_gx = F.conv2d(pred_c, self.sobel_x, padding=1)
            pred_gy = F.conv2d(pred_c, self.sobel_y, padding=1)
            target_gx = F.conv2d(target_c, self.sobel_x, padding=1)
            target_gy = F.conv2d(target_c, self.sobel_y, padding=1)
            
            # L1 difference of gradients
            loss += torch.abs(pred_gx - target_gx).mean()
            loss += torch.abs(pred_gy - target_gy).mean()
        
        return loss / C


class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss for texture alignment.
    
    Compares images in the frequency domain to ensure texture consistency.
    This helps reduce blur and artifacts common in medical image synthesis.
    
    Reference: "Focal Frequency Loss for Image Reconstruction and Synthesis" (ICCV 2021)
    
    Args:
        alpha: Focal weight exponent (default: 1.0)
        log_matrix: Whether to use log scale for frequency (default: False)
    """
    def __init__(self, alpha: float = 1.0, log_matrix: bool = False):
        super().__init__()
        self.alpha = alpha
        self.log_matrix = log_matrix
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute focal frequency loss.
        
        Args:
            pred: Predicted image, shape (B, C, H, W)
            target: Ground truth image, shape (B, C, H, W)
        
        Returns:
            Focal frequency loss scalar
        """
        # Compute 2D FFT
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        
        # Compute magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        if self.log_matrix:
            pred_mag = torch.log1p(pred_mag)
            target_mag = torch.log1p(target_mag)
        
        # Compute frequency difference
        freq_diff = torch.abs(pred_mag - target_mag)
        
        # Focal weighting: emphasize frequencies with larger errors
        focal_weight = freq_diff ** self.alpha
        
        # Weighted loss
        loss = (focal_weight * freq_diff).mean()
        
        return loss


class CombinedDiffusionLoss(nn.Module):
    """
    Combined loss function for CT-to-PET Diffusion training.
    
    Combines:
    1. Noise prediction loss (standard diffusion loss)
    2. WeightedPETLoss (tumor emphasis)
    3. GradientLoss (edge preservation)
    4. FocalFrequencyLoss (texture consistency)
    
    Args:
        noise_weight: Weight for noise prediction loss (default: 1.0)
        pet_weight: Weight for WeightedPETLoss (default: 0.5)
        gradient_weight: Weight for gradient loss (default: 0.1)
        frequency_weight: Weight for focal frequency loss (default: 0.1)
        pet_threshold: Threshold for tumor region detection (default: 0.3)
        pet_high_weight: Weight multiplier for tumor regions (default: 10.0)
    """
    def __init__(
        self,
        noise_weight: float = 1.0,
        pet_weight: float = 0.5,
        gradient_weight: float = 0.1,
        frequency_weight: float = 0.1,
        pet_threshold: float = 0.3,
        pet_high_weight: float = 10.0,
    ):
        super().__init__()
        
        self.noise_weight = noise_weight
        self.pet_weight = pet_weight
        self.gradient_weight = gradient_weight
        self.frequency_weight = frequency_weight
        
        # Sub-losses
        self.noise_loss = nn.MSELoss()
        self.pet_loss = WeightedPETLoss(
            threshold=pet_threshold,
            high_weight=pet_high_weight,
            loss_type='l1'
        )
        self.gradient_loss = GradientLoss()
        self.frequency_loss = FocalFrequencyLoss()
    
    def forward(
        self,
        pred_noise: torch.Tensor,
        target_noise: torch.Tensor,
        pred_x0: Optional[torch.Tensor] = None,
        target_x0: Optional[torch.Tensor] = None,
        tumor_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Args:
            pred_noise: Predicted noise, shape (B, C, H, W)
            target_noise: Target noise (ground truth), shape (B, C, H, W)
            pred_x0: Reconstructed clean image (optional), shape (B, C, H, W)
            target_x0: Ground truth clean image (optional), shape (B, C, H, W)
            tumor_mask: Optional tumor segmentation mask
        
        Returns:
            total_loss: Combined loss scalar
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 1. Noise prediction loss (always computed)
        noise_loss = self.noise_loss(pred_noise, target_noise)
        loss_dict['noise_loss'] = noise_loss.item()
        total_loss += self.noise_weight * noise_loss
        
        # 2-4. Image-level losses (only if x0 is provided)
        if pred_x0 is not None and target_x0 is not None:
            # Weighted PET loss
            pet_loss = self.pet_loss(pred_x0, target_x0, mask=tumor_mask)
            loss_dict['pet_loss'] = pet_loss.item()
            total_loss += self.pet_weight * pet_loss
            
            # Gradient loss
            if self.gradient_weight > 0:
                grad_loss = self.gradient_loss(pred_x0, target_x0)
                loss_dict['gradient_loss'] = grad_loss.item()
                total_loss += self.gradient_weight * grad_loss
            
            # Frequency loss
            if self.frequency_weight > 0:
                freq_loss = self.frequency_loss(pred_x0, target_x0)
                loss_dict['frequency_loss'] = freq_loss.item()
                total_loss += self.frequency_weight * freq_loss
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


# =============================================================================
# Testing / Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("测试损失函数模块")
    print("=" * 50 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试 WeightedPETLoss
    print("1. 测试 WeightedPETLoss")
    pet_loss = WeightedPETLoss(threshold=0.3, high_weight=10.0)
    
    # 模拟 PET 图像（大部分是暗的，少量亮区）
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 20:30, 20:30] = 0.8  # 模拟肿瘤热点
    pred = target + torch.randn_like(target) * 0.1  # 带噪声的预测
    
    loss = pet_loss(pred, target)
    print(f"   WeightedPETLoss: {loss.item():.4f}")
    
    # 测试 CombinedDiffusionLoss
    print("\n2. 测试 CombinedDiffusionLoss")
    combined_loss = CombinedDiffusionLoss()
    
    pred_noise = torch.randn(2, 4, 32, 32)
    target_noise = torch.randn(2, 4, 32, 32)
    pred_x0 = torch.randn(2, 1, 64, 64).clamp(0, 1)
    target_x0 = torch.randn(2, 1, 64, 64).clamp(0, 1)
    
    total, loss_dict = combined_loss(pred_noise, target_noise, pred_x0, target_x0)
    print(f"   Total Loss: {total.item():.4f}")
    for k, v in loss_dict.items():
        print(f"   - {k}: {v:.4f}")
    
    print("\n✅ 所有损失函数测试通过!")
