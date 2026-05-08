from .latent_diffusion import LatentDiffusionModel
# from .training import DiffusionTrainer  # 不存在，已移除
from .training_v2 import DiffusionTrainerV2

# 新迁移的模型架构
from .karras_unet import KarrasUnet, KarrasUnetWrapper
from .continuous_time_diffusion import (
    ContinuousTimeGaussianDiffusion,
    ContinuousTimeGaussianDiffusionConditional
)

# MSRD-Style Condition Encoder for CT-to-PET
from .encoder import MultiScaleStem, DualStreamCTEncoder, TrueDualStreamAttentionEncoder

# ControlNet Injection for CT-to-PET
from .control_net import ZeroConv2d, ControlNetInjection, ControlledUNet

# Loss Functions for CT-to-PET
from .losses import (
    WeightedPETLoss,
    GradientLoss,
    FocalFrequencyLoss,
    QuantitativeConstraintLoss,
    CombinedDiffusionLoss,
)

# Experimental modules are exposed for research iteration only.
EXPERIMENTAL_COMPONENTS = (
    'ContinuousTimeGaussianDiffusion',
    'ContinuousTimeGaussianDiffusionConditional',
    'ControlNetInjection',
    'ControlledUNet',
)

__all__ = [
    'LatentDiffusionModel',
    # 'DiffusionTrainer',  # 不存在，已移除
    'DiffusionTrainerV2',
    # 新迁移的模型
    'KarrasUnet',
    'KarrasUnetWrapper',
    'ContinuousTimeGaussianDiffusion',
    'ContinuousTimeGaussianDiffusionConditional',
    # MSRD Encoder
    'MultiScaleStem',
    'DualStreamCTEncoder',
    'TrueDualStreamAttentionEncoder',
    # ControlNet
    'ZeroConv2d',
    'ControlNetInjection',
    'ControlledUNet',
    # Losses
    'WeightedPETLoss',
    'GradientLoss',
    'FocalFrequencyLoss',
    'QuantitativeConstraintLoss',
    'CombinedDiffusionLoss',
    'EXPERIMENTAL_COMPONENTS',
]
