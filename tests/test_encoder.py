"""
验证脚本：测试 DualStreamCTEncoder 模块的正确性

运行方式:
    cd My_diffusion
    python tests/test_encoder.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径 (跨平台兼容)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.model.encoder import MultiScaleStem, DualStreamCTEncoder


def test_multi_scale_stem():
    """测试 MultiScaleStem 模块"""
    print("=" * 50)
    print("测试 MultiScaleStem")
    print("=" * 50)
    
    # 创建模块
    stem = MultiScaleStem(in_channels=1, base_channels=64)
    
    # 测试输入
    x = torch.randn(2, 1, 128, 128)  # (Batch=2, Channels=1, H=128, W=128)
    
    # 前向传播
    out = stem(x)
    
    print(f"输入形状:  {x.shape}")
    print(f"输出形状:  {out.shape}")
    print(f"期望形状:  torch.Size([2, 64, 128, 128])")
    
    # 验证
    assert out.shape == (2, 64, 128, 128), f"形状不匹配! 得到 {out.shape}"
    print("✅ MultiScaleStem 测试通过!")
    print()


def test_dual_stream_encoder():
    """测试 DualStreamCTEncoder 模块"""
    print("=" * 50)
    print("测试 DualStreamCTEncoder")
    print("=" * 50)
    
    # 创建模块
    encoder = DualStreamCTEncoder(
        in_channels=1,
        base_channels=64,
        out_channels=320,  # 匹配 Stable Diffusion U-Net
        num_heads=8,
        num_res_blocks=3
    )
    
    # 测试输入 (模拟 CT 切片)
    x = torch.randn(2, 1, 256, 256)  # (Batch=2, Channels=1, H=256, W=256)
    
    # 前向传播
    out = encoder(x)
    
    print(f"输入形状:  {x.shape}")
    print(f"输出形状:  {out.shape}")
    print(f"期望形状:  torch.Size([2, 320, 256, 256])")
    
    # 验证
    assert out.shape == (2, 320, 256, 256), f"形状不匹配! 得到 {out.shape}"
    print("✅ DualStreamCTEncoder 测试通过!")
    print()
    
    # 打印模型统计
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"模型总参数量:   {total_params:,}")
    print(f"可训练参数量:   {trainable_params:,}")
    print()


def test_gradient_flow():
    """测试梯度是否能正常反向传播"""
    print("=" * 50)
    print("测试梯度流动")
    print("=" * 50)
    
    encoder = DualStreamCTEncoder()
    x = torch.randn(1, 1, 64, 64, requires_grad=True)
    
    out = encoder(x)
    loss = out.mean()
    loss.backward()
    
    # 检查输入是否有梯度
    assert x.grad is not None, "输入没有梯度!"
    print(f"输入梯度形状: {x.grad.shape}")
    
    # 检查模型参数是否有梯度
    has_grad = all(p.grad is not None for p in encoder.parameters() if p.requires_grad)
    assert has_grad, "部分参数没有梯度!"
    print("✅ 梯度流动测试通过!")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("开始验证 MSRD-Style Encoder")
    print("=" * 50 + "\n")
    
    try:
        test_multi_scale_stem()
        test_dual_stream_encoder()
        test_gradient_flow()
        
        print("=" * 50)
        print("🎉 所有测试通过!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
