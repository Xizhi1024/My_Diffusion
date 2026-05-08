import torch
import argparse
import yaml
import os
import sys
import random
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import get_dataloaders
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
        config['enforce_tumor_weight_range'] = bool(config.get('enforce_tumor_weight_range', True))
        config.setdefault('tumor_weight_min', 10.0)
        config.setdefault('tumor_weight_max', 50.0)

        tumor_weight = float(config.get('tumor_region_weight', config.get('aux_x0_pet_high_weight', 10.0)))
        tumor_min = float(config.get('tumor_weight_min', 10.0))
        tumor_max = float(config.get('tumor_weight_max', 50.0))
        if tumor_min > tumor_max:
            tumor_min, tumor_max = tumor_max, tumor_min
            config['tumor_weight_min'] = tumor_min
            config['tumor_weight_max'] = tumor_max
        tumor_weight = min(max(tumor_weight, tumor_min), tumor_max)
        config['tumor_region_weight'] = tumor_weight
        config['aux_x0_pet_high_weight'] = tumor_weight
        config.setdefault('aux_x0_pet_focus_weight', 0.08)
    else:
        if 'tumor_region_weight' in config:
            config['aux_x0_pet_high_weight'] = float(config['tumor_region_weight'])
        config.setdefault('enforce_tumor_weight_range', False)

    config.setdefault('experimental', {})
    config['experimental'].setdefault('enable_controlnet', False)
    config['experimental'].setdefault('enable_continuous_time', False)

    return profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--test-ema', action='store_true', help='Test EMA model after training')
    parser.add_argument(
        '--profile',
        type=str,
        default=None,
        help='One-click model profile: standard|dual (or standard_diffusion1|dual_stream_attention)',
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    active_profile = apply_profile_overrides(config, args.profile)
    experimental_cfg = config.get('experimental', {})
    if experimental_cfg.get('enable_controlnet', False) or experimental_cfg.get('enable_continuous_time', False):
        raise NotImplementedError(
            "scripts/train_v2.py currently supports the default LatentDiffusionModel pipeline only. "
            "Disable experimental.enable_controlnet / experimental.enable_continuous_time in config."
        )

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
    print(f"Objective: {config.get('objective', 'pred_noise')}")
    print(f"Min-SNR weighting: {config.get('min_snr_loss_weight', False)}")
    print(f"Sampling scheduler: {config.get('sample_scheduler', 'ddpm')}")
    print(f"Pipeline profile: {active_profile}")
    print(f"Model variant: {config.get('model_variant', 'standard_diffusion1')}")
    print(f"Invert PET (data): {config.get('invert_pet', False)}")
    print(f"Invert PET on output: {config.get('invert_pet_on_output', True)}")
    print(
        "Data source: "
        f"format={config.get('data_format', 'png')}, "
        f"use_dicom_mapping={config.get('use_dicom_mapping', False)}, "
        f"enable_dicom_hu_suv={config.get('enable_dicom_hu_suv', False)}"
    )
    print(
        "Aux x0 loss weights: "
        f"l1={config.get('aux_x0_l1_weight', 0.0)}, "
        f"grad={config.get('aux_x0_grad_weight', 0.0)}, "
        f"pet_focus={config.get('aux_x0_pet_focus_weight', 0.0)}, "
        f"tumor_region_weight={config.get('tumor_region_weight', config.get('aux_x0_pet_high_weight', 'NA'))}"
    )
    print(
        "Quantitative constraints: "
        f"enabled={config.get('enable_quantitative_constraints', False)}, "
        f"weight={config.get('quantitative_constraint_weight', 0.0)}"
    )
    print(
        "Heteroscedastic/NLL: "
        f"enable_heteroscedastic={config.get('enable_heteroscedastic', False)}, "
        f"use_nll={config.get('use_heteroscedastic_nll', False)}, "
        f"nll_weight={config.get('heteroscedastic_nll_weight', 1.0)}, "
        f"mse_weight={config.get('heteroscedastic_mse_weight', 0.0)}"
    )
    print(f"EMA enabled: {config.get('use_ema', True)}")
    if config.get('use_ema', True):
        print(f"  EMA decay: {config.get('ema_decay', 0.995)}")
        print(f"  EMA update every: {config.get('ema_update_every', 1)}")
    print(f"AMP enabled: {config.get('use_amp', True)}")
    print(f"Eval sampling steps: {config.get('num_inference_steps_eval', 1000)}")
    print(f"Eval MC samples: {config.get('num_mc_samples_eval', 1)}")
    print(
        "Experimental modules: "
        f"controlnet={config.get('experimental', {}).get('enable_controlnet', False)}, "
        f"continuous_time={config.get('experimental', {}).get('enable_continuous_time', False)}"
    )
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
        sample_scheduler=config.get('sample_scheduler', 'ddpm'),
        model_variant=config.get('model_variant', 'standard_diffusion1'),
        dual_stream_config=config.get('dual_stream_config', None),
        min_snr_loss_weight=config.get('min_snr_loss_weight', False),
        min_snr_gamma=config.get('min_snr_gamma', 5.0),
        enable_heteroscedastic=config.get('enable_heteroscedastic', False),
        heteroscedastic_logvar_min=config.get('heteroscedastic_logvar_min', -6.0),
        heteroscedastic_logvar_max=config.get('heteroscedastic_logvar_max', 2.0),
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
