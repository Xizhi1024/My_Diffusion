import pytest
import torch
from torch.utils.data import DataLoader

from src.model import LatentDiffusionModel
from src.model.training_v2 import DiffusionTrainerV2


def _small_unet_config(image_size: int):
    return {
        'in_channels': 2,
        'out_channels': 1,
        'sample_size': image_size,
        'layers_per_block': 1,
        'block_out_channels': (32, 64, 64),
        'down_block_types': ('DownBlock2D', 'DownBlock2D', 'DownBlock2D'),
        'up_block_types': ('UpBlock2D', 'UpBlock2D', 'UpBlock2D'),
    }


def _tiny_loader(num_samples: int = 3, image_size: int = 32):
    dataset = []
    for _ in range(num_samples):
        dataset.append(
            {
                'pet': torch.rand(1, image_size, image_size) * 2 - 1,
                'ct': torch.rand(1, image_size, image_size) * 2 - 1,
            }
        )
    return DataLoader(dataset, batch_size=1, shuffle=False)


def test_smoke_train_validate_sample_evaluate():
    image_size = 32
    model = LatentDiffusionModel(
        image_size=image_size,
        unet_type='diffusers',
        unet_config=_small_unet_config(image_size),
        sample_scheduler='ddim',
    )
    train_loader = _tiny_loader(image_size=image_size)
    val_loader = _tiny_loader(image_size=image_size)

    trainer = DiffusionTrainerV2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config={
            'learning_rate': 1e-4,
            'weight_decay': 0.0,
            'num_epochs': 1,
            'lr_min': 1e-6,
            'use_amp': False,
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
            'save_interval': 0,
            'eval_interval': 0,
        },
    )

    train_loss = trainer.train_epoch(1)
    val_loss = trainer.validate(use_ema=False)
    metrics = trainer.evaluate(num_samples=1, use_ema=False)
    samples = trainer.sample_images(num_samples=1, use_ema=False)

    assert torch.isfinite(torch.tensor(train_loss))
    assert torch.isfinite(torch.tensor(val_loss))
    assert 'mae' in metrics and 'ssim' in metrics
    assert samples['generated'].shape == (1, 1, image_size, image_size)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
