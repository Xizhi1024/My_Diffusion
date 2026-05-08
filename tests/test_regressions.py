from pathlib import Path
from unittest import mock

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.dataset import get_dataloaders
from src.model import LatentDiffusionModel
from src.model.continuous_time_diffusion import ContinuousTimeGaussianDiffusionConditional
from src.model.training_v2 import DiffusionTrainerV2


def _small_diffusers_unet_config(image_size: int):
    return {
        'in_channels': 2,
        'out_channels': 1,
        'sample_size': image_size,
        'layers_per_block': 1,
        'block_out_channels': (32, 64, 64),
        'down_block_types': ('DownBlock2D', 'DownBlock2D', 'DownBlock2D'),
        'up_block_types': ('UpBlock2D', 'UpBlock2D', 'UpBlock2D'),
    }


def _build_tiny_latent_model(unet_type: str = 'diffusers', image_size: int = 32):
    kwargs = {
        'image_size': image_size,
        'unet_type': unet_type,
        'sample_scheduler': 'ddim',
    }
    if unet_type == 'diffusers':
        kwargs['unet_config'] = _small_diffusers_unet_config(image_size)
    else:
        kwargs['karras_unet_config'] = {
            'dim': 64,
            'dim_max': 128,
            'num_downsamples': 3,
            'num_blocks_per_stage': 1,
            'attn_res': (8,),
            'attn_flash': False,
            'dropout': 0.0,
        }
    return LatentDiffusionModel(**kwargs)


def _build_tiny_loader(num_samples: int = 2, image_size: int = 32):
    dataset = []
    for _ in range(num_samples):
        dataset.append(
            {
                'pet': torch.rand(1, image_size, image_size) * 2 - 1,
                'ct': torch.rand(1, image_size, image_size) * 2 - 1,
            }
        )
    return DataLoader(dataset, batch_size=1, shuffle=False)


def _build_tiny_trainer(use_amp: bool):
    model = _build_tiny_latent_model('diffusers', image_size=32)
    train_loader = _build_tiny_loader()
    val_loader = _build_tiny_loader()
    config = {
        'learning_rate': 1e-4,
        'weight_decay': 0.0,
        'num_epochs': 1,
        'lr_min': 1e-6,
        'use_amp': use_amp,
        'use_ema': False,
        'gradient_accumulate_every': 1,
        'grad_clip_norm': 0.0,
        'batch_size': 1,
        'val_batch_size': 1,
        'num_inference_steps_eval': 2,
        'num_inference_steps_sample': 2,
        'use_wandb': False,
        'invert_pet': False,
        'invert_pet_on_output': False,
    }
    return DiffusionTrainerV2(model=model, train_loader=train_loader, val_loader=val_loader, config=config)


@pytest.mark.parametrize('unet_type', ['diffusers', 'karras'])
def test_unconditional_sampling_works(unet_type: str):
    model = _build_tiny_latent_model(unet_type=unet_type, image_size=32)
    condition = torch.randn(2, 1, 32, 32)

    conditional = model.sample(condition=condition, num_inference_steps=2)
    unconditional = model.sample(condition=None, num_inference_steps=2)

    assert conditional.shape == (2, 1, 32, 32)
    assert unconditional.shape == (1, 1, 32, 32)


def test_heteroscedastic_forward_and_mc_sampling():
    model = LatentDiffusionModel(
        image_size=32,
        unet_type='diffusers',
        unet_config=_small_diffusers_unet_config(32),
        sample_scheduler='ddim',
        enable_heteroscedastic=True,
    )
    x = torch.rand(2, 1, 32, 32) * 2 - 1
    c = torch.rand(2, 1, 32, 32) * 2 - 1

    out = model(x, c)
    assert isinstance(out, tuple)
    assert len(out) == 4
    pred, target, timesteps, logvar = out
    assert pred.shape == target.shape
    assert logvar.shape == pred.shape
    assert timesteps.shape[0] == x.shape[0]

    sampled = model.sample_multiple(condition=c, num_inference_steps=2, num_samples=3, return_stats=True)
    assert sampled['samples'].shape == (3, 2, 1, 32, 32)
    assert sampled['mean'].shape == (2, 1, 32, 32)
    assert sampled['std'].shape == (2, 1, 32, 32)


def test_continuous_time_conditional_forward_runs():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.random_or_learned_sinusoidal_cond = True
            self.self_condition = False
            self.proj = torch.nn.Conv2d(2, 1, kernel_size=1)

        def forward(self, x, log_snr):
            return self.proj(x)

    diffusion = ContinuousTimeGaussianDiffusionConditional(
        model=DummyModel(),
        image_size=32,
        channels=1,
        condition_channels=1,
    )

    target = torch.rand(2, 1, 32, 32)
    condition = torch.rand(2, 1, 32, 32)
    loss = diffusion(target, condition)
    assert torch.isfinite(loss)


def test_dataset_same_paths_uses_independent_val_without_aug(tmp_path: Path):
    pet_dir = tmp_path / 'pet'
    ct_dir = tmp_path / 'ct'
    pet_dir.mkdir()
    ct_dir.mkdir()

    for idx in range(3):
        arr = (torch.rand(16, 16).numpy() * 255).astype('uint8')
        Image.fromarray(arr).save(pet_dir / f'{idx:03d}.png')
        Image.fromarray(arr).save(ct_dir / f'{idx:03d}.png')

    config = {
        'train_pet_path': str(pet_dir),
        'train_ct_path': str(ct_dir),
        'val_pet_path': str(pet_dir),
        'val_ct_path': str(ct_dir),
        'image_size': 16,
        'batch_size': 1,
        'val_batch_size': 1,
        'augment': True,
        'normalize': True,
        'use_medical_preprocessing': False,
        'num_workers': 0,
    }

    train_loader, val_loader = get_dataloaders(config)
    assert train_loader.dataset is not val_loader.dataset
    assert train_loader.dataset.dataset.augment is True
    assert val_loader.dataset.dataset.augment is False


def test_amp_cpu_path_degrades_without_crash():
    trainer = _build_tiny_trainer(use_amp=True)
    batch = next(iter(trainer.train_loader))
    loss = trainer.compute_loss(batch)
    assert torch.isfinite(loss)


def test_autocast_context_used_when_amp_enabled():
    trainer = _build_tiny_trainer(use_amp=False)
    trainer.mixed_precision.enabled = True

    entered = {'value': False}

    class _DummyCtx:
        def __enter__(self):
            entered['value'] = True
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    with mock.patch('torch.amp.autocast', return_value=_DummyCtx()) as patched:
        with trainer._autocast_context():
            pass
        patched.assert_called_once()
    assert entered['value'] is True


def test_checkpoint_compatibility_mismatch_raises(tmp_path: Path):
    trainer = _build_tiny_trainer(use_amp=False)

    checkpoint = {
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'config': {
            'unet_type': 'karras',
            'model_variant': 'dual_stream_attention',
            'objective': 'pred_x0',
        },
        'best_val_loss': 0.0,
        'global_step': 0,
        'is_ema': False,
        'epoch': 0,
    }
    ckpt_path = tmp_path / 'mismatch_ckpt.pth'
    torch.save(checkpoint, ckpt_path)

    with pytest.raises(ValueError, match='compatibility'):
        trainer.load_checkpoint(str(ckpt_path))


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
