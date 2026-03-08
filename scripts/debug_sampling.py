"""
快速诊断脚本 - 分析采样过程中的问题
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import LatentDiffusionModel
import yaml

def main():
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("诊断 Diffusion 采样问题")
    print("=" * 60)
    
    # 创建模型
    model = LatentDiffusionModel(
        in_channels=1,
        out_channels=1,
        image_size=config['image_size'],
        objective=config.get('objective', 'pred_noise'),
    )
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"\n1. 模型配置:")
    print(f"   - objective: {model.objective}")
    print(f"   - image_size: {model.image_size}")
    print(f"   - unet_type: {model.unet_type}")
    
    print(f"\n2. DDIM Scheduler 配置:")
    print(f"   - prediction_type: {model.ddim_scheduler.config.prediction_type}")
    print(f"   - num_train_timesteps: {model.ddim_scheduler.config.num_train_timesteps}")
    print(f"   - trained_betas[:5]: {model.ddim_scheduler.betas[:5].tolist()}")
    print(f"   - trained_betas[-5:]: {model.ddim_scheduler.betas[-5:].tolist()}")
    
    print(f"\n3. 噪声调度参数:")
    print(f"   - alphas_cumprod[0]: {model.alphas_cumprod[0].item():.6f}")
    print(f"   - alphas_cumprod[500]: {model.alphas_cumprod[500].item():.6f}")
    print(f"   - alphas_cumprod[-1]: {model.alphas_cumprod[-1].item():.6f}")
    
    # 测试采样时的 timesteps
    print(f"\n4. 测试采样 timesteps:")
    model.ddim_scheduler.set_timesteps(50)
    print(f"   - num_inference_steps=50: timesteps[:10] = {model.ddim_scheduler.timesteps[:10].tolist()}")
    model.ddim_scheduler.set_timesteps(1000)
    print(f"   - num_inference_steps=1000: timesteps[:10] = {model.ddim_scheduler.timesteps[:10].tolist()}")
    
    # 测试一次前向传播
    print(f"\n5. 测试模型前向传播:")
    with torch.no_grad():
        x = torch.randn(1, 1, 128, 128, device=device)
        condition = torch.randn(1, 1, 128, 128, device=device)
        
        # 测试训练时的前向传播
        model_pred, target, timesteps = model(x, condition)
        print(f"   - 输入 x shape: {x.shape}, range: [{x.min():.2f}, {x.max():.2f}]")
        print(f"   - model_pred shape: {model_pred.shape}, range: [{model_pred.min():.2f}, {model_pred.max():.2f}]")
        print(f"   - target shape: {target.shape}, range: [{target.min():.2f}, {target.max():.2f}]")
        print(f"   - timesteps: {timesteps.tolist()}")
    
    # 测试一次采样
    print(f"\n6. 测试采样 (10步快速测试):")
    with torch.no_grad():
        condition = torch.randn(1, 1, 128, 128, device=device) * 0.5  # 模拟[-1,1]范围的CT
        
        # 使用少量步数快速测试
        generated = model.sample(condition=condition, num_inference_steps=10)
        print(f"   - generated shape: {generated.shape}")
        print(f"   - generated range: [{generated.min():.4f}, {generated.max():.4f}]")
        print(f"   - generated mean: {generated.mean():.4f}, std: {generated.std():.4f}")
        
        # 如果生成图像均值接近0.5且std很大，说明接近纯噪声
        if generated.std() > 0.2:
            print(f"   ⚠️ 警告: 生成图像标准差较大 ({generated.std():.4f})，可能接近噪声!")
            
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)

if __name__ == '__main__':
    main()
