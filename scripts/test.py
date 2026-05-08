import argparse
import copy
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import PairedDataset, get_dataloaders
from src.model import LatentDiffusionModel
from src.model.ema import EMA
from src.utils import prepare_pet_eval_tensors


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _resolve_profile_name(profile_name: str) -> str:
    alias_map = {
        'standard': 'standard_diffusion1',
        'diffusion1': 'standard_diffusion1',
        'standard_diffusion1': 'standard_diffusion1',
        'dual': 'dual_stream_attention',
        'dual_stream': 'dual_stream_attention',
        'dual_stream_attention': 'dual_stream_attention',
    }
    key = str(profile_name).strip().lower()
    if key not in alias_map:
        raise ValueError(
            f"Unknown profile '{profile_name}'. "
            "Use one of: standard, dual, standard_diffusion1, dual_stream_attention"
        )
    return alias_map[key]


def apply_profile_overrides(config, cli_profile=None):
    requested_profile = cli_profile or config.get('pipeline_profile') or config.get('model_variant', 'standard_diffusion1')
    profile = _resolve_profile_name(requested_profile)

    config['pipeline_profile'] = profile
    config['model_variant'] = profile

    if profile == 'dual_stream_attention':
        config.setdefault(
            'dual_stream_config',
            {
                'base_channels': 64,
                'out_channels': 64,
                'num_heads': 4,
                'groups': 8,
                'local_window_size': 8,
                'global_pool_size': 16,
            }
        )
        config.setdefault('tumor_weight_min', 10.0)
        config.setdefault('tumor_weight_max', 50.0)
        config['enforce_tumor_weight_range'] = bool(config.get('enforce_tumor_weight_range', True))
    else:
        config.setdefault('enforce_tumor_weight_range', False)

    return profile


def save_comparison_images(
    ct,
    pet,
    generated_pet,
    save_dir,
    idx,
    suffix='',
    uncertainty_map=None,
):
    """Save CT condition, GT PET, generated PET (+ optional uncertainty) side by side."""
    ct = np.clip((ct + 1.0) / 2.0, 0.0, 1.0)
    pet = np.clip(pet, 0.0, 1.0)
    generated_pet = np.clip(generated_pet, 0.0, 1.0)

    show_uncertainty = uncertainty_map is not None
    ncols = 4 if show_uncertainty else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    axes[0].imshow(ct[0], cmap='gray')
    axes[0].set_title('CT (Condition)', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(pet[0], cmap='gray')
    axes[1].set_title('PET (Ground Truth)', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(generated_pet[0], cmap='gray')
    axes[2].set_title('Generated PET', fontsize=12)
    axes[2].axis('off')

    if show_uncertainty:
        unc = np.clip(uncertainty_map[0], 0.0, 1.0)
        axes[3].imshow(unc, cmap='magma')
        axes[3].set_title('Uncertainty (Std)', fontsize=12)
        axes[3].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'comparison_{idx:04d}{suffix}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def resolve_inference_model(model, checkpoint, device, use_ema=True):
    """Resolve inference model from current/legacy checkpoint formats."""
    if not use_ema:
        return model, False

    if checkpoint.get('is_ema', False):
        # model_state_dict is already EMA weights.
        return model, True

    if 'ema_shadow_state_dict' in checkpoint:
        ema_model = copy.deepcopy(model)
        ema_model.load_state_dict(checkpoint['ema_shadow_state_dict'])
        ema_model = ema_model.to(device)
        ema_model.eval()
        return ema_model, True

    # Legacy checkpoint format compatibility.
    if 'ema_state_dict' in checkpoint:
        legacy_ema = EMA(model, beta=0.995, update_every=10)
        legacy_ema.load_state_dict(checkpoint['ema_state_dict'])
        legacy_ema.ema_model = legacy_ema.ema_model.to(device)
        legacy_ema.ema_model.eval()
        return legacy_ema.ema_model, True

    return model, False


def validate_checkpoint_compatibility(model, checkpoint):
    """
    Validate that checkpoint metadata is compatible with the current model config.
    """
    checkpoint_config = checkpoint.get('config', None)
    if not isinstance(checkpoint_config, dict):
        return

    expected = {
        'unet_type': getattr(model, 'unet_type', None),
        'model_variant': getattr(model, 'model_variant', None),
        'objective': getattr(model, 'objective', None),
        'enable_heteroscedastic': getattr(model, 'enable_heteroscedastic', None),
    }
    found = {
        'unet_type': checkpoint_config.get('unet_type', None),
        'model_variant': checkpoint_config.get(
            'model_variant',
            checkpoint_config.get('pipeline_profile', None),
        ),
        'objective': checkpoint_config.get('objective', None),
        'enable_heteroscedastic': checkpoint_config.get('enable_heteroscedastic', None),
    }

    mismatches = []
    for key, expected_value in expected.items():
        found_value = found.get(key, None)
        if expected_value is None or found_value is None:
            continue
        if str(expected_value) != str(found_value):
            mismatches.append((key, found_value, expected_value))

    if not mismatches:
        return

    mismatch_text = "; ".join(
        f"{k}: checkpoint={v_ckpt}, current={v_current}"
        for k, v_ckpt, v_current in mismatches
    )
    raise ValueError(
        "Checkpoint/model compatibility check failed. "
        f"{mismatch_text}. "
        "Please use a matching config/checkpoint pair or migrate the checkpoint."
    )


def test_model(
    inference_model,
    test_loader,
    num_inference_steps,
    num_samples=None,
    save_dir=None,
    invert_pet=False,
    invert_pet_on_output=False,
    num_mc_samples: int = 1,
):
    """Evaluate model on CT->PET task."""
    device = next(inference_model.parameters()).device
    inference_model.eval()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    all_metrics = []
    num_processed = 0

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                pet = batch['pet'].to(device)
                ct = batch['ct'].to(device)
            else:
                pet, ct = batch.chunk(2, dim=1)
                pet = pet.to(device)
                ct = ct.to(device)

            if num_mc_samples > 1:
                sampled = inference_model.sample_multiple(
                    condition=ct,
                    num_inference_steps=num_inference_steps,
                    num_samples=num_mc_samples,
                    return_stats=True,
                )
                generated = sampled['mean']
                uncertainty = sampled['std']
            else:
                generated = inference_model.sample(condition=ct, num_inference_steps=num_inference_steps)
                uncertainty = None
            pred_01, target_01 = prepare_pet_eval_tensors(
                pred_pet_01=generated,
                target_pet_raw=pet,
                invert_pet=invert_pet,
                invert_pet_on_output=invert_pet_on_output,
            )

            for i in range(pet.shape[0]):
                if num_samples and num_processed >= num_samples:
                    break

                pred = pred_01[i:i + 1]
                target = target_01[i:i + 1]

                mae = torch.nn.functional.l1_loss(pred, target).item()
                mse = torch.nn.functional.mse_loss(pred, target).item()
                rmse = float(np.sqrt(mse))

                target_range = (target.max() - target.min()).item()
                nrmse = rmse / target_range if target_range > 0 else float('inf')

                pred_np = pred.cpu().numpy()[0, 0]
                target_np = target.cpu().numpy()[0, 0]
                psnr_val = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
                ssim_val = ssim(target_np, pred_np, data_range=1.0)

                all_metrics.append(
                    {
                        'mae': mae,
                        'mse': mse,
                        'rmse': rmse,
                        'nrmse': nrmse,
                        'psnr': psnr_val,
                        'ssim': ssim_val,
                    }
                )
                if uncertainty is not None:
                    unc_map = uncertainty[i].cpu().numpy()
                    all_metrics[-1]['uncertainty_mean'] = float(unc_map.mean())

                if save_dir and num_processed < 20:
                    unc_image = None
                    if uncertainty is not None:
                        unc_map = uncertainty[i].cpu().numpy()
                        unc_max = float(np.max(unc_map))
                        unc_image = unc_map / unc_max if unc_max > 1e-8 else unc_map
                    save_comparison_images(
                        ct[i].cpu().numpy(),
                        target[i].cpu().numpy(),
                        pred[i].cpu().numpy(),
                        save_dir,
                        num_processed,
                        uncertainty_map=unc_image,
                    )

                num_processed += 1

            if num_samples and num_processed >= num_samples:
                break

    if not all_metrics:
        raise RuntimeError('No samples were evaluated. Check dataloader and --num_samples setting.')

    avg_metrics = {}
    std_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = float(np.mean(values))
        std_metrics[key] = float(np.std(values))

    return avg_metrics, std_metrics, len(all_metrics)


def main():
    parser = argparse.ArgumentParser(description='Test CT->PET diffusion model')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, default=None, help='Path to test data folder (if different from val)')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to test on (None = all)')
    parser.add_argument('--save_dir', type=str, default='test_results', help='Directory to save test results and images')
    parser.add_argument('--notebook', type=str, default=None, help='Save results to notebook-X directory (X is input value)')
    parser.add_argument('--no_save_images', action='store_true', help='Do not save comparison images')
    parser.add_argument('--no_ema', action='store_true', help='Disable EMA inference even if available')
    parser.add_argument('--num_workers', type=int, default=None, help='Override dataloader workers')
    parser.add_argument('--num_inference_steps', type=int, default=None, help='Override sampling steps for evaluation')
    parser.add_argument(
        '--num_mc_samples',
        type=int,
        default=None,
        help='Number of stochastic diffusion samples per input (mean prediction for metrics)',
    )
    parser.add_argument('--eval', action='store_true', help='Alias for evaluation mode')
    parser.add_argument(
        '--profile',
        type=str,
        default=None,
        help='One-click model profile: standard|dual (or standard_diffusion1|dual_stream_attention)',
    )

    args = parser.parse_args()

    save_dir = f'./notebook/test-{args.notebook}' if args.notebook else (None if args.no_save_images else args.save_dir)
    config = load_config(args.config)
    dataloader_workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else int(config.get('num_workers', 4))
    )
    active_profile = apply_profile_overrides(config, args.profile)
    config.setdefault('experimental', {})
    config['experimental'].setdefault('enable_controlnet', False)
    config['experimental'].setdefault('enable_continuous_time', False)
    if config['experimental'].get('enable_controlnet', False) or config['experimental'].get('enable_continuous_time', False):
        raise NotImplementedError(
            "scripts/test.py only supports the default LatentDiffusionModel inference path. "
            "Disable experimental.enable_controlnet / experimental.enable_continuous_time in config."
        )

    torch.manual_seed(config.get('seed', 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.get('seed', 42))

    num_inference_steps = int(
        args.num_inference_steps
        if args.num_inference_steps is not None
        else config.get('num_inference_steps_eval', 1000)
    )
    num_mc_samples = max(
        1,
        int(
            args.num_mc_samples
            if args.num_mc_samples is not None
            else config.get('num_mc_samples_eval', 1)
        ),
    )

    model = LatentDiffusionModel(
        in_channels=1,
        out_channels=1,
        image_size=config['image_size'],
        latent_channels=config.get('latent_channels', 4),
        unet_config=config.get('unet_config', None),
        unet_type=config.get('unet_type', 'diffusers'),
        karras_unet_config=config.get('karras_unet_config', None),
        objective=config.get('objective', 'pred_noise'),
        sample_scheduler=config.get('sample_scheduler', 'ddpm'),
        model_variant=config.get('model_variant', 'standard_diffusion1'),
        dual_stream_config=config.get('dual_stream_config', None),
        min_snr_loss_weight=config.get('min_snr_loss_weight', False),
        min_snr_gamma=config.get('min_snr_gamma', 5.0),
        enable_heteroscedastic=config.get('enable_heteroscedastic', False),
        heteroscedastic_logvar_min=config.get('heteroscedastic_logvar_min', -6.0),
        heteroscedastic_logvar_max=config.get('heteroscedastic_logvar_max', 2.0),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = None
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        validate_checkpoint_compatibility(model, checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from: {args.checkpoint}")
        print(f"Checkpoint epoch: {checkpoint.get('epoch', 'Unknown')}")
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}")
        print('Testing with randomly initialized model!')

    model = model.to(device)
    model.eval()

    inference_model = model
    using_ema = False
    if checkpoint is not None:
        inference_model, using_ema = resolve_inference_model(
            model=model,
            checkpoint=checkpoint,
            device=device,
            use_ema=not args.no_ema,
        )

    if args.test_data:
        test_dataset = PairedDataset(
            pet_folder=os.path.join(args.test_data, 'PET'),
            ct_folder=os.path.join(args.test_data, 'CT'),
            image_size=config['image_size'],
            augment=False,
            normalize=config.get('normalize', True),
            invert_pet=config.get('invert_pet', False),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.get('val_batch_size', config['batch_size']),
            shuffle=False,
            num_workers=dataloader_workers,
            pin_memory=True,
        )
    else:
        config['num_workers'] = dataloader_workers
        _, test_loader = get_dataloaders(config)

    print(f"\nTesting model on {args.num_samples if args.num_samples else 'all'} samples...")
    print(f'Device: {device}')
    print(f'Pipeline profile: {active_profile}')
    print(f"Model variant: {config.get('model_variant', 'standard_diffusion1')}")
    print(f'Sampling steps: {num_inference_steps}')
    print(f'MC samples per input: {num_mc_samples}')
    print(f"Sampling scheduler: {config.get('sample_scheduler', 'ddpm')}")
    print(f"Invert PET (data): {config.get('invert_pet', False)}")
    print(f"Invert PET on output: {config.get('invert_pet_on_output', True)}")
    print(f"Enable heteroscedastic head: {config.get('enable_heteroscedastic', False)}")
    print(f"Dataloader workers: {dataloader_workers}")
    print(f'Using EMA: {using_ema}')
    print(f"Saving results to: {save_dir if save_dir else 'Not saving'}")

    avg_metrics, std_metrics, evaluated_samples = test_model(
        inference_model=inference_model,
        test_loader=test_loader,
        num_inference_steps=num_inference_steps,
        num_samples=args.num_samples,
        save_dir=save_dir,
        invert_pet=config.get('invert_pet', False),
        invert_pet_on_output=config.get('invert_pet_on_output', True),
        num_mc_samples=num_mc_samples,
    )

    print(f"\n{'=' * 60}")
    print(f'TEST RESULTS (evaluated on {evaluated_samples} samples)')
    print(f"{'=' * 60}")

    metric_names = list(avg_metrics.keys())
    for metric in metric_names:
        print(f"{metric.upper():<8}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_file = os.path.join(save_dir, f'test_metrics_{timestamp}.txt') if save_dir else None
    if metrics_file:
        with open(metrics_file, 'w') as f:
            f.write(f'TEST RESULTS - {timestamp}\n')
            f.write(f'Model checkpoint: {args.checkpoint}\n')
            f.write(f'Number of samples: {evaluated_samples}\n')
            f.write(f'Device: {device}\n')
            f.write(f'Using EMA: {using_ema}\n')
            f.write(f'Sampling steps: {num_inference_steps}\n')
            f.write(f'MC samples per input: {num_mc_samples}\n')
            f.write(f"Sampling scheduler: {config.get('sample_scheduler', 'ddpm')}\n")
            f.write(f"Invert PET (data): {config.get('invert_pet', False)}\n")
            f.write(f"Invert PET on output: {config.get('invert_pet_on_output', True)}\n")
            f.write('-' * 40 + '\n')
            for metric in metric_names:
                f.write(f"{metric.upper()}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}\n")
        print(f'\nMetrics saved to: {metrics_file}')


if __name__ == '__main__':
    main()
