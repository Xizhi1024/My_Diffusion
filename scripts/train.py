import torch
import argparse
import yaml
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_dataloaders
from src.model import LatentDiffusionModel, DiffusionTrainer


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set random seeds
    torch.manual_seed(config.get('seed', 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.get('seed', 42))

    # Create dataloaders
    train_loader, val_loader = get_dataloaders(config)

    # Create model
    model = LatentDiffusionModel(
        in_channels=1,
        out_channels=1,
        image_size=config['image_size'],
        latent_channels=config.get('latent_channels', 4),
        unet_config=config.get('unet_config', None),
    )

    # Create trainer
    trainer = DiffusionTrainer(
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
    print("Starting training...")
    trainer.train()
    print("Training completed!")


if __name__ == '__main__':
    main()