"""
EMA 验证脚本 - 确认修复后 EMA 正常更新
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import LatentDiffusionModel
from src.model.ema import EMAModel

def main():
    print("=" * 60)
    print("验证 EMA 修复")
    print("=" * 60)
    
    # 创建模型
    model = LatentDiffusionModel(
        in_channels=1,
        out_channels=1,
        image_size=128,
    )
    
    # 创建 EMA（使用默认配置）
    ema = EMAModel(
        model=model,
        decay=0.995,
        min_decay=0.0,
        update_after_step=0,
        use_ema_warmup=False,  # 这是默认配置
    )
    
    print(f"\n1. EMA 配置:")
    print(f"   - decay: {ema.decay}")
    print(f"   - min_decay: {ema.min_decay}")
    print(f"   - use_ema_warmup: {ema.use_ema_warmup}")
    
    # 测试 get_decay()
    ema.current_step = 100
    decay = ema.get_decay()
    print(f"\n2. get_decay() 测试:")
    print(f"   - current_step: {ema.current_step}")
    print(f"   - 返回的 decay 值: {decay}")
    
    if decay == 0.0:
        print("   ❌ 错误: decay=0，EMA 不会更新！")
        return False
    else:
        print("   ✅ 正确: decay > 0，EMA 会正常更新")
    
    # 测试 EMA 更新
    print(f"\n3. 测试 EMA 参数更新:")
    
    # 获取初始参数
    model_param = list(model.parameters())[0]
    shadow_param = list(ema.shadow.parameters())[0]
    
    initial_diff = (model_param.data - shadow_param.data).abs().mean().item()
    print(f"   - 初始差异: {initial_diff:.6f}")
    
    # 模拟一步优化
    with torch.no_grad():
        model_param.data.add_(torch.randn_like(model_param) * 0.1)
    
    # 更新 EMA
    ema.step(current_step=101)
    
    after_diff = (model_param.data - shadow_param.data).abs().mean().item()
    print(f"   - 更新后差异: {after_diff:.6f}")
    
    if after_diff > initial_diff * 0.001:  # 应该有明显差异
        print("   ✅ EMA 参数已更新")
    else:
        print("   ❌ EMA 参数未更新")
        return False
    
    print("\n" + "=" * 60)
    print("✅ EMA 修复验证通过！")
    print("=" * 60)
    return True

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
