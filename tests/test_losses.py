"""
验证脚本：测试损失函数模块

运行方式:
    cd My_diffusion
    python tests/test_losses.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.model.losses import WeightedPETLoss, GradientLoss, FocalFrequencyLoss, CombinedDiffusionLoss


def test_weighted_pet_loss():
    """测试 WeightedPETLoss"""
    print("=" * 50)
    print("测试 WeightedPETLoss")
    print("=" * 50)
    
    loss_fn = WeightedPETLoss(threshold=0.3, high_weight=10.0)
    
    # 创建模拟 PET 图像（稀疏：大部分暗，少量亮点）
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 20:30, 20:30] = 0.8  # 模拟肿瘤
    target[:, :, 50:55, 50:55] = 0.9  # 另一个热点
    
    # 完美预测
    pred_perfect = target.clone()
    loss_perfect = loss_fn(pred_perfect, target)
    
    # 带噪声预测
    pred_noisy = target + torch.randn_like(target) * 0.1
    loss_noisy = loss_fn(pred_noisy, target)
    
    # 完全错误预测（全零）
    pred_wrong = torch.zeros_like(target)
    loss_wrong = loss_fn(pred_wrong, target)
    
    print(f"完美预测 Loss: {loss_perfect.item():.6f} (应该接近0)")
    print(f"带噪声 Loss:   {loss_noisy.item():.6f}")
    print(f"全错误 Loss:   {loss_wrong.item():.6f} (应该最大)")
    
    assert loss_perfect < loss_noisy < loss_wrong, "损失排序不正确!"
    print("✅ WeightedPETLoss 测试通过!")
    print()


def test_gradient_loss():
    """测试 GradientLoss"""
    print("=" * 50)
    print("测试 GradientLoss")
    print("=" * 50)
    
    loss_fn = GradientLoss()
    
    # 创建有边缘的图像
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 20:40, 20:40] = 1.0  # 方块
    
    # 完美预测
    pred_perfect = target.clone()
    loss_perfect = loss_fn(pred_perfect, target)
    
    # 模糊预测（使用高斯模糊模拟）
    pred_blurry = torch.nn.functional.avg_pool2d(target, 5, 1, 2)
    loss_blurry = loss_fn(pred_blurry, target)
    
    print(f"完美预测 Loss: {loss_perfect.item():.6f}")
    print(f"模糊预测 Loss: {loss_blurry.item():.6f}")
    
    assert loss_perfect < loss_blurry, "模糊预测应该有更大的梯度损失!"
    print("✅ GradientLoss 测试通过!")
    print()


def test_focal_frequency_loss():
    """测试 FocalFrequencyLoss"""
    print("=" * 50)
    print("测试 FocalFrequencyLoss")
    print("=" * 50)
    
    loss_fn = FocalFrequencyLoss()
    
    target = torch.randn(2, 1, 64, 64)
    pred_perfect = target.clone()
    pred_different = torch.randn(2, 1, 64, 64)
    
    loss_perfect = loss_fn(pred_perfect, target)
    loss_different = loss_fn(pred_different, target)
    
    print(f"完美预测 Loss: {loss_perfect.item():.6f}")
    print(f"不同预测 Loss: {loss_different.item():.6f}")
    
    assert loss_perfect < loss_different
    print("✅ FocalFrequencyLoss 测试通过!")
    print()


def test_combined_loss():
    """测试 CombinedDiffusionLoss"""
    print("=" * 50)
    print("测试 CombinedDiffusionLoss")
    print("=" * 50)
    
    loss_fn = CombinedDiffusionLoss(
        noise_weight=1.0,
        pet_weight=0.5,
        gradient_weight=0.1,
        frequency_weight=0.1,
    )
    
    # 模拟训练数据
    pred_noise = torch.randn(2, 4, 32, 32)
    target_noise = torch.randn(2, 4, 32, 32)
    pred_x0 = torch.rand(2, 1, 64, 64)
    target_x0 = torch.rand(2, 1, 64, 64)
    
    total_loss, loss_dict = loss_fn(pred_noise, target_noise, pred_x0, target_x0)
    
    print(f"总损失: {total_loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  - {k}: {v:.4f}")
    
    assert 'noise_loss' in loss_dict
    assert 'pet_loss' in loss_dict
    print("✅ CombinedDiffusionLoss 测试通过!")
    print()


def test_gradient_flow():
    """测试梯度流动"""
    print("=" * 50)
    print("测试梯度流动")
    print("=" * 50)
    
    loss_fn = CombinedDiffusionLoss()
    
    pred_noise = torch.randn(1, 4, 16, 16, requires_grad=True)
    target_noise = torch.randn(1, 4, 16, 16)
    pred_x0 = torch.randn(1, 1, 32, 32, requires_grad=True)
    target_x0 = torch.randn(1, 1, 32, 32)
    
    total_loss, _ = loss_fn(pred_noise, target_noise, pred_x0, target_x0)
    total_loss.backward()
    
    assert pred_noise.grad is not None
    assert pred_x0.grad is not None
    print("✅ 梯度流动测试通过!")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("开始验证损失函数模块")
    print("=" * 50 + "\n")
    
    try:
        test_weighted_pet_loss()
        test_gradient_loss()
        test_focal_frequency_loss()
        test_combined_loss()
        test_gradient_flow()
        
        print("=" * 50)
        print("🎉 所有损失函数测试通过!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
