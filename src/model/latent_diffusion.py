import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Literal, Tuple
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.schedulers import DDIMScheduler

# 导入新迁移的 KarrasUnet
from .karras_unet import KarrasUnetWrapper
from .encoder import TrueDualStreamAttentionEncoder


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
        sample_scheduler: Literal['ddpm', 'ddim'] = 'ddpm',
        model_variant: Literal['standard_diffusion1', 'dual_stream_attention'] = 'standard_diffusion1',
        dual_stream_config: Optional[dict] = None,
        min_snr_loss_weight: bool = False,
        min_snr_gamma: float = 5.0,
        enable_heteroscedastic: bool = False,
        heteroscedastic_logvar_min: float = -6.0,
        heteroscedastic_logvar_max: float = 2.0,
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
        self.model_variant = model_variant
        self.prediction_channels = int(out_channels)
        if self.prediction_channels <= 0:
            raise ValueError("out_channels must be > 0")
        self.enable_heteroscedastic = bool(enable_heteroscedastic)
        self.heteroscedastic_logvar_min = float(heteroscedastic_logvar_min)
        self.heteroscedastic_logvar_max = float(heteroscedastic_logvar_max)
        if self.heteroscedastic_logvar_min >= self.heteroscedastic_logvar_max:
            raise ValueError("heteroscedastic_logvar_min must be < heteroscedastic_logvar_max")
        self.model_out_channels = (
            self.prediction_channels * 2 if self.enable_heteroscedastic else self.prediction_channels
        )
        self.sample_scheduler = sample_scheduler.lower()
        if self.sample_scheduler not in ('ddpm', 'ddim'):
            raise ValueError(
                f"Unknown sample_scheduler: {sample_scheduler}. Must be 'ddpm' or 'ddim'"
            )
        if self.model_variant not in ('standard_diffusion1', 'dual_stream_attention'):
            raise ValueError(
                f"Unknown model_variant: {self.model_variant}. "
                "Must be 'standard_diffusion1' or 'dual_stream_attention'"
            )
        # Note: latent_channels parameter is kept for backward compatibility but not used in pixel space

        # 根据 unet_type 选择不同的 U-Net 架构
        if unet_type == 'diffusers':
            # 使用 diffusers 库的 UNet2DModel (默认，稳定)
            unet_config = unet_config or {
                "in_channels": 2,  # noisy PET (1) + CT condition (1)
                "out_channels": self.model_out_channels,  # predict PET noise or mean/logvar
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
            unet_config = dict(unet_config)
            unet_config["out_channels"] = self.model_out_channels
            self.model_in_channels = int(unet_config["in_channels"])
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
                out_channels=self.model_out_channels,  # predict PET noise or mean/logvar
                **karras_unet_config
            )
            self.model_in_channels = int(getattr(self.unet, "in_channels", 2))
        else:
            raise ValueError(f"Unknown unet_type: {unet_type}. Must be 'diffusers' or 'karras'")

        # Optional condition encoder branch:
        # - standard_diffusion1: raw CT directly concatenated to noisy PET
        # - dual_stream_attention: CT -> dual-stream local/global attention encoder -> 1-channel map
        self.condition_encoder = None
        self.condition_proj = None
        if self.model_variant == 'dual_stream_attention':
            dual_stream_config = dual_stream_config or {}
            cond_out_channels = int(dual_stream_config.get('out_channels', 64))
            self.condition_encoder = TrueDualStreamAttentionEncoder(
                in_channels=in_channels,
                base_channels=int(dual_stream_config.get('base_channels', 64)),
                out_channels=cond_out_channels,
                num_heads=int(dual_stream_config.get('num_heads', 4)),
                groups=int(dual_stream_config.get('groups', 8)),
                local_window_size=int(dual_stream_config.get('local_window_size', 8)),
                global_pool_size=int(dual_stream_config.get('global_pool_size', 16)),
            )
            self.condition_proj = nn.Sequential(
                nn.Conv2d(cond_out_channels, 1, kernel_size=1),
                nn.Tanh(),
            )

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

    def _prepare_condition(self, condition: Optional[torch.Tensor], target_size=None):
        """Prepare condition map according to selected model variant."""
        if condition is None:
            return None

        condition_out = condition
        if self.condition_encoder is not None and self.condition_proj is not None:
            condition_out = self.condition_encoder(condition_out)
            condition_out = self.condition_proj(condition_out)

        if target_size is not None and condition_out.shape[2:] != target_size:
            condition_out = F.interpolate(
                condition_out,
                size=target_size,
                mode='bilinear',
                align_corners=False,
            )

        return condition_out

    def _build_model_input(
        self,
        noisy_x: torch.Tensor,
        condition_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Build U-Net input consistently for both training and sampling.

        When condition is missing, pad with zero channels so unconditional sampling
        still matches the configured model input channels.
        """
        if condition_map is not None:
            model_input = torch.cat([noisy_x, condition_map], dim=1)
        else:
            model_input = noisy_x

        input_channels = model_input.shape[1]
        if input_channels == self.model_in_channels:
            return model_input

        if input_channels < self.model_in_channels:
            pad_channels = self.model_in_channels - input_channels
            zeros = torch.zeros(
                model_input.shape[0],
                pad_channels,
                model_input.shape[2],
                model_input.shape[3],
                device=model_input.device,
                dtype=model_input.dtype,
            )
            return torch.cat([model_input, zeros], dim=1)

        raise ValueError(
            f"Model input channels mismatch: got {input_channels}, expected <= {self.model_in_channels}"
        )

    def _split_prediction(
        self,
        raw_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.enable_heteroscedastic:
            return raw_output, None

        expected = self.prediction_channels * 2
        if raw_output.shape[1] == self.prediction_channels:
            # 兼容旧权重：启用异方差但模型仍输出单通道，退化为均值预测。
            return raw_output, None

        if raw_output.shape[1] != expected:
            raise ValueError(
                f"Unexpected model output channels: got {raw_output.shape[1]}, expected {expected}"
            )

        pred_mean, pred_logvar = torch.chunk(raw_output, 2, dim=1)
        pred_logvar = torch.clamp(
            pred_logvar,
            min=self.heteroscedastic_logvar_min,
            max=self.heteroscedastic_logvar_max,
        )
        return pred_mean, pred_logvar

    def predict_noise(self, noisy_x, timesteps, condition=None, return_logvar: bool = False):
        """Predict diffusion target (and optional log-variance) in pixel space."""
        condition_map = self._prepare_condition(condition, target_size=noisy_x.shape[2:])
        model_input = self._build_model_input(noisy_x, condition_map)
        raw_output = self.unet(model_input, timesteps).sample
        pred_mean, pred_logvar = self._split_prediction(raw_output)
        if return_logvar:
            return pred_mean, pred_logvar
        return pred_mean

    @torch.no_grad()
    def sample(self, condition=None, num_inference_steps=50, generator=None):
        """
        使用 diffusers scheduler 采样。
        - ddpm: 与原始 DDPM 训练目标一致（更接近 baseline 参考实现）
        - ddim: 更快的确定性采样
        - condition=None: 自动补零条件通道，确保无条件采样与模型输入通道一致
        """
        device = next(self.unet.parameters()).device
        batch_size = condition.shape[0] if condition is not None else 1

        # 1. 选择采样调度器并设置步数
        scheduler = self.noise_scheduler if self.sample_scheduler == 'ddpm' else self.ddim_scheduler
        scheduler.set_timesteps(num_inference_steps)

        # 2. 初始化噪声
        img = torch.randn(
            (batch_size, 1, self.image_size, self.image_size),
            device=device,
            generator=generator,
        )
        condition_map = self._prepare_condition(condition, target_size=img.shape[2:])

        # 3. 使用 diffusers 标准调度器循环
        for t in scheduler.timesteps:
            # 拼接 noisy PET 和 CT 条件
            if condition_map is not None:
                if condition_map.shape[2:] != img.shape[2:]:
                    cond_for_step = F.interpolate(
                        condition_map,
                        size=img.shape[2:],
                        mode='bilinear',
                        align_corners=False,
                    )
                else:
                    cond_for_step = condition_map
                unet_input = self._build_model_input(img, cond_for_step)
            else:
                unet_input = self._build_model_input(img, None)

            # 预测噪声 / x0 / v（由 prediction_type 控制）
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            raw_output = self.unet(unet_input, t_batch).sample
            model_output, _ = self._split_prediction(raw_output)

            # 使用 scheduler 计算下一步
            step_kwargs = {'eta': 0.0} if self.sample_scheduler == 'ddim' else {}
            img = scheduler.step(model_output, t, img, **step_kwargs).prev_sample

        # 4. 后处理 [-1, 1] -> [0, 1]
        img = (img + 1.0) * 0.5
        img = torch.clamp(img, 0.0, 1.0)

        return img

    @torch.no_grad()
    def sample_multiple(
        self,
        condition=None,
        num_inference_steps: int = 50,
        num_samples: int = 4,
        generator=None,
        return_stats: bool = True,
    ):
        """
        Monte Carlo diffusion sampling.

        Returns:
            if return_stats:
                {'samples': [S, B, C, H, W], 'mean': [B, C, H, W], 'std': [B, C, H, W]}
            else:
                samples tensor [S, B, C, H, W]
        """
        num_samples = int(num_samples)
        if num_samples <= 0:
            raise ValueError("num_samples must be >= 1")

        samples = []
        for _ in range(num_samples):
            samples.append(
                self.sample(
                    condition=condition,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                )
            )

        stacked = torch.stack(samples, dim=0)
        if not return_stats:
            return stacked

        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0, unbiased=False)
        return {
            'samples': stacked,
            'mean': mean,
            'std': std,
        }

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
        model_output = self.predict_noise(
            noisy_x,
            timesteps,
            condition,
            return_logvar=self.enable_heteroscedastic,
        )
        if isinstance(model_output, tuple):
            model_pred, pred_logvar = model_output
        else:
            model_pred, pred_logvar = model_output, None

        # 根据目标类型确定预测值和目标值
        if self.objective == 'pred_noise':
            # 预测噪声，目标是真实噪声
            target = noise
        elif self.objective == 'pred_x0':
            # 预测 x0 (原始图像)，目标是真实图像
            target = x
        elif self.objective == 'pred_v':
            # 预测 v-parameterization
            # v = sqrt(alpha_bar) * noise - sqrt(1 - alpha_bar) * x0
            target = self.predict_v(x, timesteps, noise)
        else:
            raise ValueError(f"unknown objective {self.objective}")

        if pred_logvar is not None:
            return model_pred, target, timesteps, pred_logvar
        return model_pred, target, timesteps
