import torch
import argparse
import yaml
import os
import sys
import random
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_dataloaders
from src.model import LatentDiffusionModel
from src.model.training_v2 import DiffusionTrainerV2


def set_seed(seed=42):
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--test-ema', action='store_true', help='Test EMA model after training')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set random seeds
    set_seed(config.get('seed', 42))

    # Print configuration
    print("="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Image size: {config['image_size']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"EMA enabled: {config.get('use_ema', True)}")
    if config.get('use_ema', True):
        print(f"  EMA decay: {config.get('ema_decay', 0.995)}")
        print(f"  EMA update every: {config.get('ema_update_every', 1)}")
    print(f"AMP enabled: {config.get('use_amp', True)}")
    print(f"Eval sampling steps: {config.get('num_inference_steps_eval', 1000)}")
    print("="*60)

    # Create dataloaders
    train_loader, val_loader = get_dataloaders(config)

    # Print dataset info
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Create model
    # 【新增】支持选择不同的 U-Net 架构 (diffusers 或 karras)
    unet_type = config.get('unet_type', 'diffusers')
    print(f"\nU-Net Architecture: {unet_type.upper()}")
    if unet_type == 'karras':
        print("  Using KarrasUnet (Magnitude-Preserving Operations)")
        print("  Reference: https://arxiv.org/abs/2312.02696")

    model = LatentDiffusionModel(
        in_channels=1,
        out_channels=1,
        image_size=config['image_size'],
        latent_channels=config.get('latent_channels', 4),
        unet_config=config.get('unet_config', None),
        unet_type=unet_type,
        karras_unet_config=config.get('karras_unet_config', None),
        objective=config.get('objective', 'pred_noise'),
        min_snr_loss_weight=config.get('min_snr_loss_weight', False),
        min_snr_gamma=config.get('min_snr_gamma', 5.0),
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Create trainer with EMA and AMP support
    trainer = DiffusionTrainerV2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from checkpoint: {args.resume}")

    # Start training
    print("\nStarting training...")
    final_model = trainer.train()
    print("\nTraining completed!")

    # Test EMA model if requested
    if args.test_ema and trainer.ema:
        print("\nTesting EMA model...")
        eval_metrics = trainer.evaluate(num_samples=None, use_ema=True)
        print("\nFinal EMA Model Metrics:")
        for k, v in eval_metrics.items():
            print(f"  {k.upper()}: {v:.4f}")


if __name__ == '__main__':
    main()
