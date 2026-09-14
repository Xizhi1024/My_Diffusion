"""Contracts for the complete Haar-wavelet BBDM U-Net."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_wavelet_downsample_mixes_all_subbands_and_backpropagates():
    from src.model.wavelet_unet import WaveletDownsample

    x = torch.randn(2, 8, 32, 32, requires_grad=True)
    layer = WaveletDownsample(8, 16, mix_kernel_size=3)

    y = layer(x)
    y.square().mean().backward()

    assert y.shape == (2, 16, 16, 16)
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_wavelet_upsample_expands_to_four_subbands_and_backpropagates():
    from src.model.wavelet_unet import WaveletUpsample

    x = torch.randn(2, 16, 16, 16, requires_grad=True)
    layer = WaveletUpsample(16, 8, mix_kernel_size=3)

    y = layer(x)
    y.abs().mean().backward()

    assert y.shape == (2, 8, 32, 32)
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_complete_wavelet_unet_preserves_public_shapes_and_skip_contract():
    from src.model.wavelet_unet import (
        WaveletBBDMUNet,
        WaveletDownsample,
        WaveletUpsample,
    )

    model = WaveletBBDMUNet(
        in_channels=2,
        base_channels=8,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=1,
        time_dim=32,
        enable_heteroscedastic=False,
        ca_kv_dim=8,
        ca_num_heads=1,
    )
    x = torch.randn(2, 2, 32, 32)
    timesteps = torch.tensor([10, 20])
    injections = [
        torch.randn(2, 32, 4, 4),
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 16, 16, 16),
        torch.randn(2, 8, 32, 32),
    ]

    output = model(x, timesteps, skip_injections=injections)

    assert output.shape == (2, 1, 32, 32)
    assert sum(isinstance(module, WaveletDownsample) for module in model.modules()) == 3
    assert sum(isinstance(module, WaveletUpsample) for module in model.modules()) == 3


def test_complete_wavelet_unet_scale_changes_do_not_call_interpolate(monkeypatch):
    import torch.nn.functional as functional

    from src.model.wavelet_unet import WaveletBBDMUNet

    def _reject_interpolate(*args, **kwargs):
        raise AssertionError("wavelet backbone called interpolate")

    monkeypatch.setattr(functional, "interpolate", _reject_interpolate)
    model = WaveletBBDMUNet(
        in_channels=2,
        base_channels=8,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=1,
        time_dim=32,
        enable_heteroscedastic=False,
        ca_kv_dim=8,
        ca_num_heads=1,
    )

    output = model(torch.randn(1, 2, 32, 32), torch.tensor([1]))

    assert output.shape == (1, 1, 32, 32)
