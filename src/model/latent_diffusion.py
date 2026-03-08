import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Literal, Union
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.schedulers import DDIMScheduler

# 导入新迁移的 KarrasUnet
from .karras_unet import KarrasUnetWrapper


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    Sigmoid beta schedule - 更适合大尺寸图像 (>64x64)
    参考: Karras et al. 2022 "Elucidating the Design Space of Diffusion-Based Generative Models"
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, clamp_min, 0.999)


def extract(a, t, x_shape):
    """
    从张量 a 中提取索引 t 处的值，并重塑为 x_shape 的形状
    用于获取特定时间步的 alpha, beta 等参数
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class LatentDiffusionModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        image_size: int = 128,
        latent_channels: int = None,  # Parameter kept for compatibility but not used
        unet_config: Optional[dict] = None,
        unet_type: Literal['diffusers', 'karras'] = 'diffusers',
        karras_unet_config: Optional[dict] = None,
        objective: Literal['pred_noise', 'pred_x0', 'pred_v'] = 'pred_noise',
        min_snr_loss_weight: bool = False,
        min_snr_gamma: float = 5.0,
    ):
        """
        Latent Diffusion Model 支持多种 U-Net 架构

        Args:
            unet_type: 选择 U-Net 架构类型
                - 'diffusers': 使用 diffusers 库的 UNet2DModel (默认，稳定)
                - 'karras': 使用 KarrasUnet (幅度保持 U-Net，来自 Karras et al. 2023)
            karras_unet_config: KarrasUnet 的配置 (仅当 unet_type='karras' 时使用)
        """
        super().__init__()

        self.image_size = image_size
        self.objective = objective
        self.unet_type = unet_type
        # Note: latent_channels parameter is kept for backward compatibility but not used in pixel space

        # 根据 unet_type 选择不同的 U-Net 架构
        if unet_type == 'diffusers':
            # 使用 diffusers 库的 UNet2DModel (默认，稳定)
            unet_config = unet_config or {
                "in_channels": 2,  # noisy PET (1) + CT condition (1)
                "out_channels": 1,  # predict PET noise
                "sample_size": image_size,  # Direct pixel space processing
                "layers_per_block": 2,
                "block_out_channels": (64, 128, 256, 256, 512),  # Slightly deeper for pixel space
                "down_block_types": (
                    "DownBlock2D",       # 128 -> 64
                    "DownBlock2D",       # 64 -> 32
                    "DownBlock2D",       # 32 -> 16
                    "AttnDownBlock2D",   # 16 -> 8 (attention at this level)
                    "DownBlock2D",       # 8 -> 4
                ),
                "up_block_types": (
                    "UpBlock2D",         # 4 -> 8
                    "AttnUpBlock2D",     # 8 -> 16
                    "UpBlock2D",         # 16 -> 32
                    "UpBlock2D",         # 32 -> 64
                    "UpBlock2D",         # 64 -> 128
                ),
            }
            self.unet = UNet2DModel(**unet_config)

        elif unet_type == 'karras':
            # 使用 KarrasUnet - 幅度保持 U-Net (来自 Karras et al. 2023)
            # 参考: https://arxiv.org/abs/2312.02696
            karras_unet_config = karras_unet_config or {
                'dim': 192,
                'dim_max': 768,
                'num_downsamples': 4,  # 适应 128x128 图像
                'num_blocks_per_stage': 2,
                'attn_res': (16, 8),  # 在 16x16 和 8x8 分辨率使用注意力
                'fourier_dim': 16,
                'attn_dim_head': 64,
                'attn_flash': False,
                'mp_cat_t': 0.5,
                'mp_add_emb_t': 0.5,
                'attn_res_mp_add_t': 0.3,
                'resnet_mp_add_t': 0.3,
                'dropout': 0.1,
            }
            self.unet = KarrasUnetWrapper(
                image_size=image_size,
                in_channels=2,  # noisy PET (1) + CT condition (1)
                out_channels=1,  # predict PET noise
                **karras_unet_config
            )
        else:
            raise ValueError(f"Unknown unet_type: {unet_type}. Must be 'diffusers' or 'karras'")

        # 【修复】使用 sigmoid schedule - 对 128x128 图像更稳定
        # 参考 diffusion 项目的工作实现
        # 【修复】强制使用 float32 以兼容 AMP
        betas = sigmoid_beta_schedule(1000, start=-3, end=3, tau=1).float()

        # 计算 alpha 相关参数
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # 注册为 buffer（不参与梯度计算，但会随模型移动设备）
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

        # 计算 SNR (signal-to-noise ratio)
        # SNR = alpha_cumprod / (1 - alpha_cumprod)
        snr = alphas_cumprod / (1 - alphas_cumprod + 1e-8)

        # 【新增】Min-SNR 损失权重
        # 参考: https://arxiv.org/abs/2303.09556
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        # 根据目标类型计算损失权重
        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
            prediction_type = "epsilon"  # 预测噪声
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
            prediction_type = "sample"   # 【关键】预测原图 x0
        elif objective == 'pred_v':
            loss_weight = maybe_clipped_snr / (snr + 1)
            prediction_type = "v_prediction"  # 预测 v
        else:
            raise ValueError(f"unknown objective {objective}")

        self.register_buffer('loss_weight', loss_weight.float())
        self.register_buffer('snr', snr.float())

        # 预计算常用的 sqrt 值
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0,  # 忽略，使用自定义 betas
            beta_end=0.0,    # 忽略，使用自定义 betas
            beta_schedule="linear",  # 会用自定义 betas 覆盖
            trained_betas=betas,  # 【关键】使用 sigmoid betas
            prediction_type=prediction_type  # 【关键修复】动态设置预测类型
        )

        # DDIM scheduler for sampling - 使用相同的 sigmoid betas 和 prediction_type
        self.ddim_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0,
            beta_end=0.0,
            beta_schedule="linear",
            trained_betas=betas,  # 【关键】使用 sigmoid betas
            prediction_type=prediction_type  # 【关键修复】动态设置预测类型
        )

    def add_noise(self, x, noise, timesteps):
        """
        Add noise directly in pixel space (manual implementation for SNR loss)
        q_sample: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, timesteps, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x.shape)

        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t, t, noise):
        """
        从噪声预测 x_start (原始图像)
        x_0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
        """
        sqrt_recip_alphas_cumprod_t = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    def predict_noise_from_start(self, x_t, t, x0):
        """
        从 x_start 预测噪声
        epsilon = (x_t - sqrt(alpha_bar_t) * x_0) / sqrt(1 - alpha_bar_t)
        """
        sqrt_recip_alphas_cumprod_t = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

        return (sqrt_recip_alphas_cumprod_t * x_t - x0) / sqrt_recipm1_alphas_cumprod_t

    def predict_v(self, x_start, t, noise):
        """
        计算 v-parameterization
        v = sqrt(alpha_bar_t) * epsilon - sqrt(1 - alpha_bar_t) * x_0
        参考: Progressive Distillation for Fast Sampling of Diffusion Models
        """
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * x_start

    def predict_start_from_v(self, x_t, t, v):
        """
        从 v 预测 x_start
        x_0 = sqrt(alpha_bar_t) * x_t - sqrt(1 - alpha_bar_t) * v
        """
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        return sqrt_alphas_cumprod_t * x_t - sqrt_one_minus_alphas_cumprod_t * v

    def predict_noise(self, noisy_x, timesteps, condition=None):
        """Predict noise directly in pixel space"""
        if condition is not None:
            # Ensure condition and noisy_x have same dimensions
            if condition.shape[2:] != noisy_x.shape[2:]:
                condition = torch.nn.functional.interpolate(
                    condition,
                    size=noisy_x.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            # Concatenate noisy PET and CT condition
            model_input = torch.cat([noisy_x, condition], dim=1)
        else:
            # If no condition, UNet expects single channel input
            # So we need to handle this case
            model_input = noisy_x
            if model_input.shape[1] == 1:
                # Pad with zeros to match expected 2 channels
                zeros = torch.zeros_like(model_input)
                model_input = torch.cat([model_input, zeros], dim=1)

        return self.unet(model_input, timesteps).sample

    @torch.no_grad()
    def sample(self, condition=None, num_inference_steps=50, generator=None):
        """
        使用 diffusers 库的 DDIMScheduler 采样方法
        修复了手动实现时的数值不稳定性问题（除以极小值导致NaN）
        """
        device = next(self.unet.parameters()).device
        batch_size = condition.shape[0] if condition is not None else 1

        # 1. 设置步数
        self.ddim_scheduler.set_timesteps(num_inference_steps)

        # 2. 初始化噪声
        img = torch.randn(
            (batch_size, 1, self.image_size, self.image_size),
            device=device,
            generator=generator,
        )

        # 3. 使用 diffusers 标准调度器循环
        for t in self.ddim_scheduler.timesteps:
            # 构造输入（拼接条件）
            model_input = img

            # 手动拼接条件（这部分保持原有逻辑）
            if condition is not None:
                if condition.shape[2:] != img.shape[2:]:
                    condition = F.interpolate(condition, size=img.shape[2:], mode='bilinear')
                # 拼接 noisy_x (img) 和 condition
                unet_input = torch.cat([img, condition], dim=1)
            else:
                unet_input = img

            # 预测
            # 注意：传入 scalar 类型的 t 批次
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            model_output = self.unet(unet_input, t_batch).sample

            # 使用 Scheduler 计算下一步 (自动处理 pred_noise / pred_x0 / pred_v)
            # step 返回的是一个对象，.prev_sample 是去噪后的图
            img = self.ddim_scheduler.step(model_output, t, img, eta=0.0).prev_sample

        # 4. 后处理 [-1, 1] -> [0, 1]
        img = (img + 1.0) * 0.5
        img = torch.clamp(img, 0.0, 1.0)

        return img

    def forward(self, x, condition=None):
        """
        训练时的前向传播，支持多种目标类型和 SNR 损失加权

        Args:
            x: 目标图像 (PET)
            condition: 条件图像 (CT)

        Returns:
            model_pred: 模型预测（根据 objective 类型可以是噪声、x0 或 v）
            target: 对应的目标（用于计算损失）
            timesteps: 时间步（用于提取损失权重）
        """
        # x is the target PET image in pixel space
        # No encoding needed - work directly in pixel space

        # Sample noise and timestep
        noise = torch.randn_like(x)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (x.shape[0],)
        )
        timesteps = timesteps.to(x.device)

        # Add noise directly to pixel space
        noisy_x = self.add_noise(x, noise, timesteps)

        # Predict using UNet
        model_output = self.predict_noise(noisy_x, timesteps, condition)

        # 根据目标类型确定预测值和目标值
        if self.objective == 'pred_noise':
            # 预测噪声，目标是真实噪声
            model_pred = model_output
            target = noise
        elif self.objective == 'pred_x0':
            # 预测 x0 (原始图像)，目标是真实图像
            model_pred = model_output
            target = x
        elif self.objective == 'pred_v':
            # 预测 v-parameterization
            model_pred = model_output
            # v = sqrt(alpha_bar) * noise - sqrt(1 - alpha_bar) * x0
            target = self.predict_v(x, timesteps, noise)
        else:
            raise ValueError(f"unknown objective {self.objective}")

        return model_pred, target, timesteps