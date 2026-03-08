"""
KarrasUnet 配置示例

本文件展示如何使用从 Diffusion/diffusion 迁移过来的 KarrasUnet 架构

KarrasUnet 特点：
- 幅度保持操作 (Magnitude-Preserving Operations)
- 无偏置设计 (Bias-Free)
- 权重归一化 (Weight Normalization)
- 参考: https://arxiv.org/abs/2312.02696
"""

import torch
from src.model import LatentDiffusionModel, KarrasUnet, KarrasUnetWrapper


# ============================================================================
# 方式 1: 使用 LatentDiffusionModel + unet_type='karras'
# ============================================================================
# 这是最简单的方式，LatentDiffusionModel 会自动处理所有细节

def create_model_with_karras_unet():
    """
    使用 KarrasUnet 的 LatentDiffusionModel

    适合条件生成任务 (如 CT -> PET)
    """
    model = LatentDiffusionModel(
        in_channels=1,
        out_channels=1,
        image_size=128,
        unet_type='karras',  # 关键: 选择 'karras' 架构
        objective='pred_noise',
        min_snr_loss_weight=True,
        min_snr_gamma=5.0,

        # KarrasUnet 特定配置
        karras_unet_config={
            'dim': 192,              # 基础维度
            'dim_max': 768,          # 最大维度
            'num_downsamples': 4,    # 下采样次数 (128 -> 64 -> 32 -> 16 -> 8)
            'num_blocks_per_stage': 2,  # 每个阶段的 ResNet 块数
            'attn_res': (16, 8),     # 在哪些分辨率使用注意力
            'fourier_dim': 16,       # Fourier 嵌入维度
            'attn_dim_head': 64,     # 注意力头维度
            'attn_flash': False,     # 是否使用 flash attention
            'mp_cat_t': 0.5,         # MP Concat 的 t 参数
            'mp_add_emb_t': 0.5,     # MP Add (embedding) 的 t 参数
            'attn_res_mp_add_t': 0.3,  # MP Add (attention) 的 t 参数
            'resnet_mp_add_t': 0.3,  # MP Add (resnet) 的 t 参数
            'dropout': 0.1,
        }
    )
    return model


# ============================================================================
# 方式 2: 直接使用 KarrasUnetWrapper
# ============================================================================
# 适合需要更多控制的情况

def create_karras_unet_wrapper():
    """直接使用 KarrasUnetWrapper"""
    model = KarrasUnetWrapper(
        image_size=128,
        in_channels=2,  # noisy PET (1) + CT condition (1)
        out_channels=1,  # predict PET noise
        dim=192,
        dim_max=768,
        num_downsamples=4,
        num_blocks_per_stage=2,
        attn_res=(16, 8),
    )
    return model


# ============================================================================
# 方式 3: 使用原始 KarrasUnet (非条件生成)
# ============================================================================

def create_pure_karras_unet():
    """
    使用原始 KarrasUnet

    适合无条件生成或分类条件生成
    """
    model = KarrasUnet(
        image_size=64,
        dim=192,
        dim_max=768,
        num_classes=1000,  # 分类条件生成
        channels=4,        # 4 通道输入
        num_downsamples=3,
        num_blocks_per_stage=4,
        attn_res=(16, 8),
    )
    return model


# ============================================================================
# 配置对比: Diffusers UNet vs KarrasUnet
# ============================================================================

def create_diffusers_unet_model():
    """使用 diffusers 库的 UNet2DModel (默认选项)"""
    model = LatentDiffusionModel(
        in_channels=1,
        out_channels=1,
        image_size=128,
        unet_type='diffusers',  # 默认选项
        objective='pred_noise',

        # Diffusers UNet 配置
        unet_config={
            "in_channels": 2,
            "out_channels": 1,
            "sample_size": 128,
            "layers_per_block": 2,
            "block_out_channels": (64, 128, 256, 256, 512),
            "down_block_types": (
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            "up_block_types": (
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        }
    )
    return model


# ============================================================================
# 使用建议
# ============================================================================

"""
何时使用 KarrasUnet:
- 需要更好的数值稳定性
- 想要尝试最新的架构设计
- 训练高分辨率图像 (KarrasUnet 在大尺寸图像上表现更好)
- 需要更快的收敛速度

何时使用 Diffusers UNet:
- 需要稳定的实现 (已经过大量验证)
- 希望与 diffusers 生态系统兼容
- 训练资源有限 (Diffusers UNet 通常更快)

参数调优建议:

1. image_size 和 num_downsamples:
   - 64x64: num_downsamples=3
   - 128x128: num_downsamples=4
   - 256x256: num_downsamples=5

2. dim 和 dim_max:
   - 较小的数据集: dim=128, dim_max=512
   - 中等数据集: dim=192, dim_max=768 (默认)
   - 大型数据集: dim=256, dim_max=1024

3. attn_res:
   - 在低分辨率 (8x8, 16x16) 使用注意力效果最好
   - 高分辨率注意力计算成本高

4. MP 操作的 t 参数:
   - mp_cat_t=0.5: 默认，推荐
   - resnet_mp_add_t=0.3: 论文推荐值
   - attn_res_mp_add_t=0.3: 论文推荐值
"""

# ============================================================================
# 训练示例
# ============================================================================

def example_training_usage():
    """训练示例"""
    # 创建模型
    model = create_model_with_karras_unet()

    # 模拟输入
    batch_size = 4
    pet_images = torch.randn(batch_size, 1, 128, 128)  # 目标 PET
    ct_images = torch.randn(batch_size, 1, 128, 128)   # 条件 CT

    # 训练模式
    model.train()
    model_pred, target, timesteps = model(pet_images, ct_images)

    print(f"Model prediction shape: {model_pred.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Timesteps: {timesteps}")

    # 采样模式
    model.eval()
    with torch.no_grad():
        samples = model.sample(condition=ct_images, num_inference_steps=50)
        print(f"Generated samples shape: {samples.shape}")


if __name__ == '__main__':
    print("创建 KarrasUnet 模型示例...")

    print("\n1. 使用 LatentDiffusionModel + unet_type='karras':")
    model1 = create_model_with_karras_unet()
    print(f"   模型参数量: {sum(p.numel() for p in model1.parameters()):,}")

    print("\n2. 使用 KarrasUnetWrapper:")
    model2 = create_karras_unet_wrapper()
    print(f"   模型参数量: {sum(p.numel() for p in model2.parameters()):,}")

    print("\n3. 使用原始 KarrasUnet:")
    model3 = create_pure_karras_unet()
    print(f"   模型参数量: {sum(p.numel() for p in model3.parameters()):,}")

    print("\n4. 使用 Diffusers UNet (对比):")
    model4 = create_diffusers_unet_model()
    print(f"   模型参数量: {sum(p.numel() for p in model4.parameters()):,}")

    print("\n训练示例:")
    example_training_usage()
