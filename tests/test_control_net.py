"""
验证脚本：测试 ControlNet Injection 模块

运行方式:
    cd My_diffusion
    python tests/test_control_net.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径 (跨平台兼容)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.model.encoder import DualStreamCTEncoder
from src.model.control_net import ZeroConv2d, ControlNetInjection


def test_zero_conv():
    """测试 ZeroConv2d 初始化为零"""
    print("=" * 50)
    print("测试 ZeroConv2d")
    print("=" * 50)
    
    zc = ZeroConv2d(64, 128)
    x = torch.randn(2, 64, 32, 32)
    out = zc(x)
    
    # 输出应该全是零（因为权重初始化为0）
    assert torch.allclose(out, torch.zeros_like(out)), "ZeroConv 输出不为零!"
    print(f"输入形状:  {x.shape}")
    print(f"输出形状:  {out.shape}")
    print(f"输出范数:  {out.norm().item():.6f} (应该接近0)")
    print("✅ ZeroConv2d 测试通过!")
    print()


def test_controlnet_injection():
    """测试 ControlNetInjection 模块"""
    print("=" * 50)
    print("测试 ControlNetInjection")
    print("=" * 50)
    
    # 创建条件编码器
    condition_encoder = DualStreamCTEncoder(
        in_channels=1,
        base_channels=64,
        out_channels=320,
    )
    
    # 创建 ControlNet (使用简化版控制模型)
    controlnet = ControlNetInjection(
        condition_encoder=condition_encoder,
        control_model=None,  # 使用简化版本
        block_out_channels=(320, 640, 1280, 1280),
    )
    
    # 测试输入
    batch_size = 2
    noisy_latents = torch.randn(batch_size, 4, 32, 32)  # 256/8 = 32
    timesteps = torch.randint(0, 1000, (batch_size,))
    ct_image = torch.randn(batch_size, 1, 256, 256)
    
    # 前向传播
    control_outputs = controlnet(noisy_latents, timesteps, ct_image)
    
    print(f"CT 图像形状:      {ct_image.shape}")
    print(f"噪声潜变量形状:    {noisy_latents.shape}")
    print(f"控制输出数量:      {len(control_outputs)}")
    
    for i, out in enumerate(control_outputs):
        print(f"  控制输出 {i}: {out.shape}")
    
    # 验证至少有输出
    assert len(control_outputs) > 0, "没有控制输出!"
    print("✅ ControlNetInjection 测试通过!")
    print()
    
    # 打印模型统计
    total_params = sum(p.numel() for p in controlnet.parameters())
    trainable_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
    print(f"模型总参数量:   {total_params:,}")
    print(f"可训练参数量:   {trainable_params:,}")
    print()


def test_gradient_flow():
    """测试梯度是否能正常反向传播"""
    print("=" * 50)
    print("测试梯度流动")
    print("=" * 50)
    
    condition_encoder = DualStreamCTEncoder()
    controlnet = ControlNetInjection(
        condition_encoder=condition_encoder,
        block_out_channels=(320, 640, 1280, 1280),
    )
    
    noisy_latents = torch.randn(1, 4, 16, 16, requires_grad=True)
    timesteps = torch.randint(0, 1000, (1,))
    ct_image = torch.randn(1, 1, 128, 128, requires_grad=True)
    
    control_outputs = controlnet(noisy_latents, timesteps, ct_image)
    
    # 对所有输出求和作为 loss
    loss = sum(o.mean() for o in control_outputs)
    loss.backward()
    
    # 检查 CT 图像是否有梯度
    assert ct_image.grad is not None, "CT 图像没有梯度!"
    print(f"CT 图像梯度形状: {ct_image.grad.shape}")
    print("✅ 梯度流动测试通过!")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("开始验证 ControlNet Injection")
    print("=" * 50 + "\n")
    
    try:
        test_zero_conv()
        test_controlnet_injection()
        test_gradient_flow()
        
        print("=" * 50)
        print("🎉 所有测试通过!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
