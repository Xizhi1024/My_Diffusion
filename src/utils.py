import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm


class DiffusionMetrics:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        if pred_np.ndim == 4:
            pred_np = np.transpose(pred_np, (0, 2, 3, 1))
            target_np = np.transpose(target_np, (0, 2, 3, 1))

        scores = []
        for i in range(pred_np.shape[0]):
            if pred_np.shape[-1] == 1:
                scores.append(psnr(target_np[i, :, :, 0], pred_np[i, :, :, 0], data_range=data_range))
            else:
                # 对于多通道图像，计算每个通道的PSNR然后取平均
                channel_scores = []
                for c in range(pred_np.shape[-1]):
                    channel_scores.append(psnr(target_np[i, :, :, c], pred_np[i, :, :, c], data_range=data_range))
                scores.append(np.mean(channel_scores))
        return np.mean(scores)

    @staticmethod
    def compute_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        if pred_np.ndim == 4:
            pred_np = np.transpose(pred_np, (0, 2, 3, 1))
            target_np = np.transpose(target_np, (0, 2, 3, 1))

        scores = []
        for i in range(pred_np.shape[0]):
            if pred_np.shape[-1] == 1:
                scores.append(ssim(target_np[i, :, :, 0], pred_np[i, :, :, 0], data_range=data_range))
            else:
                try:
                    scores.append(ssim(target_np[i], pred_np[i], data_range=data_range, channel_axis=-1, win_size=7))
                except:
                    scores.append(ssim(target_np[i], pred_np[i], data_range=data_range, multichannel=True, win_size=7))
        return np.mean(scores)

    @staticmethod
    def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
        return F.l1_loss(pred, target).item()

    @staticmethod
    def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
        return F.mse_loss(pred, target).item()

    @staticmethod
    def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
        return torch.sqrt(F.mse_loss(pred, target)).item()

    @staticmethod
    def compute_nrmse(pred: torch.Tensor, target: torch.Tensor) -> float:
        mse = F.mse_loss(pred, target)
        rmse = torch.sqrt(mse)
        target_range = target.max() - target.min()
        if target_range == 0:
            return float('inf')
        return (rmse / target_range).item()

    def evaluate_batch(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        metrics = {
            'mae': self.compute_mae(pred, target),
            'mse': self.compute_mse(pred, target),
            'rmse': self.compute_rmse(pred, target),
            'nrmse': self.compute_nrmse(pred, target),
            'psnr': self.compute_psnr(pred, target),
            'ssim': self.compute_ssim(pred, target)
        }
        return metrics

    def evaluate_model(self, model, dataloader, num_samples: Optional[int] = None) -> Dict[str, float]:
        model.eval()
        all_metrics = []

        total_samples = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                if isinstance(batch, dict):
                    pet = batch['pet'].to(self.device)
                    ct = batch['ct'].to(self.device)
                else:
                    pet, ct = batch.chunk(2, dim=1)
                    pet = pet.to(self.device)
                    ct = ct.to(self.device)

                # 使用CT作为条件生成PET（与训练保持一致）
                generated = model.sample(condition=ct, num_inference_steps=1000)

                # 【关键修复】将真实图像从 [-1, 1] 转换到 [0, 1]
                # 因为 Dataset 使用 Normalize(mean=[0.5], std=[0.5]) 将数据转换到 [-1, 1]
                # 而 sample() 返回的是 [0, 1] 范围的图像
                pet_denorm = (pet + 1.0) * 0.5
                pet_denorm = torch.clamp(pet_denorm, 0.0, 1.0)

                batch_metrics = self.evaluate_batch(generated, pet_denorm)
                all_metrics.append(batch_metrics)

                total_samples += pet.shape[0]
                if num_samples and total_samples >= num_samples:
                    break

        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        return avg_metrics


class CTStandardizer:
    """
    CT 标准化类 - 实现软组织窗处理和归一化

    按照计划文档要求：
    1. 应用软组织窗（窗宽 400，窗位 50）
    2. 截断范围之外的数值
    3. 线性映射到 [-1, 1]
    """

    def __init__(self, window_width: float = 400, window_center: float = 50):
        """
        Args:
            window_width: 窗宽（默认 400）
            window_center: 窗位（默认 50）
        """
        self.window_width = window_width
        self.window_center = window_center

        # 计算窗的范围
        self.window_min = window_center - window_width / 2
        self.window_max = window_center + window_width / 2

    def apply_window(self, ct_image: np.ndarray) -> np.ndarray:
        """
        应用软组织窗
        """
        # 截断到窗范围
        windowed = np.clip(ct_image, self.window_min, self.window_max)
        return windowed

    def normalize_to_minus1_1(self, windowed_image: np.ndarray) -> np.ndarray:
        """
        将窗处理后的图像归一化到 [-1, 1]
        """
        # 线性映射
        normalized = 2.0 * (windowed_image - self.window_min) / (self.window_max - self.window_min) - 1.0
        return normalized

    def __call__(self, ct_image: np.ndarray) -> np.ndarray:
        """
        完整的 CT 标准化流程

        Args:
            ct_image: CT 图像 (H, W) 或 (H, W, 1) 或 (C, H, W)

        Returns:
            标准化后的 CT 图像，范围 [-1, 1]
        """
        # 确保输入是 numpy 数组
        if torch.is_tensor(ct_image):
            ct_image = ct_image.detach().cpu().numpy()

        # 处理维度
        if ct_image.ndim == 3 and ct_image.shape[0] == 1:
            # (1, H, W) -> (H, W)
            ct_image = ct_image.squeeze(0)
        elif ct_image.ndim == 3 and ct_image.shape[-1] == 1:
            # (H, W, 1) -> (H, W)
            ct_image = ct_image.squeeze(-1)

        # 应用窗处理
        windowed = self.apply_window(ct_image)

        # 归一化到 [-1, 1]
        normalized = self.normalize_to_minus1_1(windowed)

        return normalized


class PETStandardizer:
    """
    PET 标准化类 - 基于 SUV 值的归一化

    按照计划文档要求：
    1. 计算整个数据集的 max_SUV（通常是 10-20 左右）
    2. 将数值除以该最大值归一化到 [0, 1] 或 [-1, 1]
    """

    def __init__(self, max_suv: Optional[float] = None, target_range: str = "[-1,1]"):
        """
        Args:
            max_suv: 数据集的最大 SUV 值。如果为 None，需要在调用 fit 或手动设置
            target_range: 目标范围，可选 "[-1,1]" 或 "[0,1]"
        """
        self.max_suv = max_suv
        self.target_range = target_range

        if self.target_range not in ["[-1,1]", "[0,1]"]:
            raise ValueError("target_range must be '[-1,1]' or '[0,1]'")

    def fit(self, pet_images: List[np.ndarray]):
        """
        从数据集中计算 max_suv

        Args:
            pet_images: PET 图像列表
        """
        max_values = []
        for img in pet_images:
            if torch.is_tensor(img):
                img = img.detach().cpu().numpy()
            max_values.append(np.max(img))

        self.max_suv = np.max(max_values)
        print(f"Computed max_suv: {self.max_suv:.2f}")

    def set_max_suv(self, max_suv: float):
        """
        手动设置 max_suv
        """
        self.max_suv = max_suv

    def normalize(self, pet_image: np.ndarray) -> np.ndarray:
        """
        归一化 PET 图像
        """
        if self.max_suv is None:
            raise ValueError("max_suv is not set. Call fit() or set_max_suv() first.")

        # 避免除以零
        if self.max_suv == 0:
            return np.zeros_like(pet_image)

        # 基本归一化到 [0, 1]
        normalized = pet_image / self.max_suv

        # 如果目标范围是 [-1, 1]，进行转换
        if self.target_range == "[-1,1]":
            normalized = 2.0 * normalized - 1.0

        return normalized

    def __call__(self, pet_image: np.ndarray) -> np.ndarray:
        """
        完整的 PET 标准化流程

        Args:
            pet_image: PET 图像 (H, W) 或 (H, W, 1) 或 (C, H, W)

        Returns:
            标准化后的 PET 图像，范围由 target_range 决定
        """
        # 确保输入是 numpy 数组
        if torch.is_tensor(pet_image):
            pet_image = pet_image.detach().cpu().numpy()

        # 处理维度
        if pet_image.ndim == 3 and pet_image.shape[0] == 1:
            # (1, H, W) -> (H, W)
            pet_image = pet_image.squeeze(0)
        elif pet_image.ndim == 3 and pet_image.shape[-1] == 1:
            # (H, W, 1) -> (H, W)
            pet_image = pet_image.squeeze(-1)

        # 归一化
        normalized = self.normalize(pet_image)

        return normalized


class MedicalImagePreprocessor:
    """
    医学图像预处理器 - 整合 CT 和 PET 的标准化
    """

    def __init__(self,
                 ct_window_width: float = 400,
                 ct_window_center: float = 50,
                 pet_max_suv: Optional[float] = None,
                 pet_target_range: str = "[-1,1]"):
        """
        Args:
            ct_window_width: CT 窗宽
            ct_window_center: CT 窗位
            pet_max_suv: PET 最大 SUV 值
            pet_target_range: PET 目标范围
        """
        self.ct_standardizer = CTStandardizer(ct_window_width, ct_window_center)
        self.pet_standardizer = PETStandardizer(pet_max_suv, pet_target_range)

    def preprocess_ct(self, ct_image: np.ndarray) -> np.ndarray:
        """预处理 CT 图像"""
        return self.ct_standardizer(ct_image)

    def preprocess_pet(self, pet_image: np.ndarray) -> np.ndarray:
        """预处理 PET 图像"""
        return self.pet_standardizer(pet_image)

    def preprocess_pair(self, ct_image: np.ndarray, pet_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预处理 CT-PET 图像对"""
        ct_processed = self.preprocess_ct(ct_image)
        pet_processed = self.preprocess_pet(pet_image)
        return ct_processed, pet_processed

    def fit_pet_standardizer(self, pet_images: List[np.ndarray]):
        """训练 PET 标准器的 max_suv"""
        self.pet_standardizer.fit(pet_images)

    def set_pet_max_suv(self, max_suv: float):
        """手动设置 PET 的 max_suv"""
        self.pet_standardizer.set_max_suv(max_suv)


class DiffusionLoss(nn.Module):
    """
    复合损失函数，专为医学图像扩散模型设计
    结合多种损失以提高生成质量
    
    新增功能 (CT-to-PET 优化):
    - WeightedPETLoss: 对高 SUV 区域（肿瘤）加权
    - FocalFrequencyLoss: 频率域对齐（减少模糊）
    """

    def __init__(self,
                 noise_mse_weight: float = 1.0,
                 l1_weight: float = 0.1,
                 ssim_weight: float = 0.5,
                 gradient_weight: float = 0.2,
                 # 新增: 肿瘤加权参数
                 pet_weight: float = 0.0,  # 默认关闭，需要 x0 重建时启用
                 pet_threshold: float = 0.3,
                 pet_high_weight: float = 10.0,
                 frequency_weight: float = 0.0):  # 默认关闭
        """
        Args:
            noise_mse_weight: 噪声预测 MSE 损失权重
            l1_weight: L1 损失权重，有助于提高边缘清晰度
            ssim_weight: SSIM 损失权重，保持结构相似性
            gradient_weight: 梯度损失权重，增强边缘特征
            pet_weight: 肿瘤加权 L1 损失权重 (新增)
            pet_threshold: 高 SUV 区域阈值 (新增)
            pet_high_weight: 肿瘤区域权重倍数 (新增)
            frequency_weight: 频率域损失权重 (新增)
        """
        super().__init__()

        self.noise_mse_weight = noise_mse_weight
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.gradient_weight = gradient_weight
        
        # 新增: 肿瘤加权参数
        self.pet_weight = pet_weight
        self.pet_threshold = pet_threshold
        self.pet_high_weight = pet_high_weight
        self.frequency_weight = frequency_weight

        # 用于 SSIM 计算的高斯核
        self.register_buffer('gaussian_kernel', self._create_gaussian_kernel())

    def _create_gaussian_kernel(self, kernel_size: int = 11, sigma: float = 1.5):
        """创建高斯核用于计算 SSIM"""
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        g_2d = g.unsqueeze(0) * g.unsqueeze(1)
        kernel = g_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, kernel_size, kernel_size]
        return kernel

    def _compute_ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算 SSIM 损失"""
        # 转换为灰度图（如果是多通道）
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        if y.shape[1] > 1:
            y = y.mean(dim=1, keepdim=True)

        # 平滑
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.conv2d(x, self.gaussian_kernel, padding=self.gaussian_kernel.shape[-1]//2)
        mu_y = F.conv2d(y, self.gaussian_kernel, padding=self.gaussian_kernel.shape[-1]//2)

        sigma_x = F.conv2d(x * x, self.gaussian_kernel, padding=self.gaussian_kernel.shape[-1]//2) - mu_x * mu_x
        sigma_y = F.conv2d(y * y, self.gaussian_kernel, padding=self.gaussian_kernel.shape[-1]//2) - mu_y * mu_y
        sigma_xy = F.conv2d(x * y, self.gaussian_kernel, padding=self.gaussian_kernel.shape[-1]//2) - mu_x * mu_y

        numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        ssim_map = numerator / denominator
        return 1 - ssim_map.mean()

    def _compute_gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算梯度损失，保持边缘清晰度"""
        # Sobel 算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                              dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                              dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)

        # 计算梯度
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)

        # 梯度 L1 损失
        grad_loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        grad_loss_y = F.l1_loss(pred_grad_y, target_grad_y)

        return (grad_loss_x + grad_loss_y) / 2

    def _compute_weighted_pet_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算肿瘤加权 L1 损失
        
        PET 图像是稀疏的（大部分是暗背景，肿瘤是亮点）。
        标准 MSE/L1 会导致模型生成模糊或全黑图像。
        此损失对高强度区域（肿瘤）赋予更高权重。
        
        Args:
            pred: 预测的 PET 图像
            target: 真实的 PET 图像
            mask: 可选的外部肿瘤 mask
        """
        # 计算像素级误差
        error = torch.abs(pred - target)
        
        # 创建权重图
        if mask is not None:
            weight_map = torch.where(
                mask > 0.5,
                torch.full_like(mask, self.pet_high_weight),
                torch.ones_like(mask)
            )
        else:
            # 根据目标图像强度自动生成权重
            weight_map = torch.where(
                target > self.pet_threshold,
                torch.full_like(target, self.pet_high_weight),
                torch.ones_like(target)
            )
        
        # 加权误差
        weighted_error = error * weight_map
        return weighted_error.mean()

    def _compute_focal_frequency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算频率域损失
        
        在频率域对比图像以确保纹理一致性，
        有助于减少医学图像合成中常见的模糊和伪影。
        """
        # 计算 2D FFT
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        
        # 计算幅度谱
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # 频率差异
        freq_diff = torch.abs(pred_mag - target_mag)
        
        return freq_diff.mean()

    def forward(self,
                noise_pred: torch.Tensor,
                noise: torch.Tensor,
                x_start: Optional[torch.Tensor] = None,
                x_recon: Optional[torch.Tensor] = None,
                tumor_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算复合损失

        Args:
            noise_pred: 预测的噪声
            noise: 真实噪声
            x_start: 原始图像（可选，用于计算结构损失）
            x_recon: 重建图像（可选，用于计算 PET 加权损失）
            tumor_mask: 肿瘤 mask（可选）

        Returns:
            包含各项损失的字典
        """
        losses = {}

        # 1. 噪声 MSE 损失（主要损失）
        noise_mse_loss = F.mse_loss(noise_pred, noise)
        losses['noise_mse'] = noise_mse_loss

        # 2. L1 损失
        l1_loss = F.l1_loss(noise_pred, noise)
        losses['l1'] = l1_loss

        # 计算总损失
        total_loss = (self.noise_mse_weight * noise_mse_loss +
                     self.l1_weight * l1_loss)

        # 3. 新增: 肿瘤加权 PET 损失（需要重建图像）
        if self.pet_weight > 0 and x_recon is not None and x_start is not None:
            pet_loss = self._compute_weighted_pet_loss(x_recon, x_start, tumor_mask)
            losses['pet_weighted'] = pet_loss
            total_loss = total_loss + self.pet_weight * pet_loss
        
        # 4. 新增: 频率域损失（需要重建图像）
        if self.frequency_weight > 0 and x_recon is not None and x_start is not None:
            freq_loss = self._compute_focal_frequency_loss(x_recon, x_start)
            losses['frequency'] = freq_loss
            total_loss = total_loss + self.frequency_weight * freq_loss

        losses['total'] = total_loss
        return losses