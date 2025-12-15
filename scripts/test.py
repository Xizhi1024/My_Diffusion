import torch
import argparse
import yaml
import os
import sys
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_dataloaders
from src.model import LatentDiffusionModel, DiffusionTrainer


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_comparison_images(pet, ct, generated, save_dir, idx, suffix=''):
    """Save PET, CT (ground truth), and generated CT images side by side"""
    # Denormalize from [-1, 1] to [0, 1]
    pet = (pet + 1) / 2
    ct = (ct + 1) / 2
    generated = (generated + 1) / 2

    # Clamp to valid range
    pet = np.clip(pet, 0, 1)
    ct = np.clip(ct, 0, 1)
    generated = np.clip(generated, 0, 1)

    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(pet[0], cmap='gray')
    axes[0].set_title('PET (Condition)', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(ct[0], cmap='gray')
    axes[1].set_title('CT (Ground Truth)', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(generated[0], cmap='gray')
    axes[2].set_title('Generated CT', fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'comparison_{idx:04d}{suffix}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def test_model(model, test_loader, num_samples=None, save_dir=None):
    """Test the model and optionally save comparison images"""
    model.eval()
    device = next(model.parameters()).device  # Get device from model

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    all_metrics = []
    num_processed = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if isinstance(batch, dict):
                pet = batch['pet'].to(device)
                ct = batch['ct'].to(device)
            else:
                pet, ct = batch.chunk(2, dim=1)
                pet = pet.to(device)
                ct = ct.to(device)

            # Generate CT from PET
            generated = model.sample(condition=pet, num_inference_steps=50)

            # Compute metrics for each sample in the batch
            for i in range(pet.shape[0]):
                pred = generated[i:i+1]
                target = ct[i:i+1]

                # Compute all metrics
                mae = torch.nn.functional.l1_loss(pred, target).item()
                mse = torch.nn.functional.mse_loss(pred, target).item()
                rmse = torch.sqrt(torch.tensor(mse)).item()

                # NRMSE
                target_range = target.max() - target.min()
                nrmse = rmse / target_range.item() if target_range > 0 else float('inf')

                # PSNR and SSIM
                pred_np = pred.cpu().numpy()[0, 0]
                target_np = target.cpu().numpy()[0, 0]

                psnr_val = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')

                # Simple SSIM calculation
                from skimage.metrics import structural_similarity as ssim
                ssim_val = ssim(target_np, pred_np, data_range=1.0)

                all_metrics.append({
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'nrmse': nrmse,
                    'psnr': psnr_val,
                    'ssim': ssim_val
                })

                # Save comparison images
                if save_dir and num_processed < 20:  # Save first 20 samples
                    save_comparison_images(
                        pet[i].cpu().numpy(),
                        ct[i].cpu().numpy(),
                        generated[i].cpu().numpy(),
                        save_dir,
                        num_processed
                    )

                num_processed += 1

            if num_samples and num_processed >= num_samples:
                break

    # Compute average metrics
    avg_metrics = {}
    std_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)
        std_metrics[key] = np.std(values)

    return avg_metrics, std_metrics, len(all_metrics)


def main():
    parser = argparse.ArgumentParser(description='Test PET-CT Diffusion Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, default=None,
                        help='Path to test data folder (if different from val)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to test on (None = all)')
    parser.add_argument('--save_dir', type=str, default='test_results',
                        help='Directory to save test results and images')
    parser.add_argument('--notebook', type=str, default=None,
                        help='Save results to notebook-X directory (X is the input value)')
    parser.add_argument('--no_save_images', action='store_true',
                        help='Do not save comparison images')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluation mode (alias for test)')

    args = parser.parse_args()

    # Override save_dir if notebook is specified
    if args.notebook:
        save_dir = f'./notebook/test-{args.notebook}'
    else:
        save_dir = None if args.no_save_images else args.save_dir

    # Load config
    config = load_config(args.config)

    # Set random seeds
    torch.manual_seed(config.get('seed', 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.get('seed', 42))

    # Create model
    model = LatentDiffusionModel(
        in_channels=1,
        out_channels=1,
        image_size=config['image_size'],
        latent_channels=config.get('latent_channels', 4),
        unet_config=config.get('unet_config', None),
    )

    # Load checkpoint
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from: {args.checkpoint}")
        print(f"Checkpoint epoch: {checkpoint.get('epoch', 'Unknown')}")
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}")
        print("Testing with randomly initialized model!")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create dataloader
    if args.test_data:
        # Use provided test data
        from src.dataset import PairedDataset, DataLoader
        test_dataset = PairedDataset(
            pet_folder=os.path.join(args.test_data, 'PET'),
            ct_folder=os.path.join(args.test_data, 'CT'),
            image_size=config['image_size'],
            augment=False,
            normalize=config.get('normalize', True)
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.get('val_batch_size', config['batch_size']),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
    else:
        # Use validation data as test data
        _, test_loader = get_dataloaders(config)

    # save_dir is already set above based on notebook or save_dir arguments

    # Run test
    print(f"\nTesting model on {args.num_samples if args.num_samples else 'all'} samples...")
    print(f"Device: {device}")
    print(f"Saving results to: {save_dir if save_dir else 'Not saving'}")

    avg_metrics, std_metrics, num_samples = test_model(
        model, test_loader, args.num_samples, save_dir
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"TEST RESULTS (evaluated on {num_samples} samples)")
    print(f"{'='*60}")

    for metric in ['mae', 'mse', 'rmse', 'nrmse', 'psnr', 'ssim']:
        print(f"{metric.upper():<8}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

    # Save metrics to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_file = os.path.join(save_dir, f'test_metrics_{timestamp}.txt') if save_dir else None
    if metrics_file:
        with open(metrics_file, 'w') as f:
            f.write(f"TEST RESULTS - {timestamp}\n")
            f.write(f"Model checkpoint: {args.checkpoint}\n")
            f.write(f"Number of samples: {num_samples}\n")
            f.write(f"Device: {device}\n")
            f.write("-" * 40 + "\n")
            for metric in ['mae', 'mse', 'rmse', 'nrmse', 'psnr', 'ssim']:
                f.write(f"{metric.upper()}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}\n")
        print(f"\nMetrics saved to: {metrics_file}")


if __name__ == '__main__':
    main()