import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from src.model.encoder import DualStreamCTEncoder
from src.model.control_net import ZeroConv2d, ControlNetInjection, ControlledUNet


class _ResidualInjectableUNet(nn.Module):
    """Tiny adapter that exposes control_residuals interface."""

    def forward(self, noisy_latents, timesteps, encoder_hidden_states=None, control_residuals=None):
        out = noisy_latents.clone()
        if control_residuals:
            for residual in control_residuals:
                resized = F.interpolate(residual, size=noisy_latents.shape[-2:], mode='nearest')
                reduced = resized.mean(dim=1, keepdim=True).expand_as(noisy_latents)
                out = out + 0.01 * reduced
        return out


class _UnsupportedUNet(nn.Module):
    def forward(self, noisy_latents, timesteps, encoder_hidden_states=None):
        return noisy_latents


class _FakeControlNet(nn.Module):
    """Test double that emits CT-dependent residuals."""

    def forward(self, noisy_latents, timesteps, ct_image, encoder_hidden_states=None):
        residual = F.interpolate(ct_image, size=noisy_latents.shape[-2:], mode='bilinear', align_corners=False)
        residual = residual.repeat(1, noisy_latents.shape[1], 1, 1)
        return [residual]


def _build_small_controlnet():
    encoder = DualStreamCTEncoder(
        in_channels=1,
        base_channels=8,
        out_channels=16,
        num_heads=2,
        num_res_blocks=1,
        groups=4,
    )
    return ControlNetInjection(
        condition_encoder=encoder,
        control_model=None,
        block_out_channels=(8, 16),
        latent_channels=4,
        condition_channels=16,
    )


def test_zero_conv_outputs_zeros():
    """ZeroConv2d should output zeros at initialization."""
    zc = ZeroConv2d(64, 128)
    x = torch.randn(2, 64, 32, 32)
    out = zc(x)

    assert torch.allclose(out, torch.zeros_like(out))


def test_controlnet_injection_outputs():
    """ControlNetInjection should return residual feature maps."""
    controlnet = _build_small_controlnet()
    noisy_latents = torch.randn(2, 4, 16, 16)
    timesteps = torch.randint(0, 1000, (2,))
    ct_image = torch.randn(2, 1, 64, 64)

    control_outputs = controlnet(noisy_latents, timesteps, ct_image)

    assert len(control_outputs) > 0
    assert all(out.ndim == 4 for out in control_outputs)


def test_controlled_unet_requires_injection_interface():
    """ControlledUNet should reject U-Nets without residual injection interface."""
    controlnet = _build_small_controlnet()
    with pytest.raises(NotImplementedError):
        ControlledUNet(unet=_UnsupportedUNet(), controlnet=controlnet)


def test_controlled_unet_uses_control_signal():
    """Changing CT input should change output when control residuals are wired."""
    model = ControlledUNet(unet=_ResidualInjectableUNet(), controlnet=_FakeControlNet(), freeze_unet=False)

    noisy_latents = torch.randn(2, 4, 16, 16)
    timesteps = torch.randint(0, 1000, (2,))
    ct_image_1 = torch.randn(2, 1, 64, 64)
    ct_image_2 = torch.randn(2, 1, 64, 64)

    out1 = model(noisy_latents, timesteps, ct_image_1)
    out2 = model(noisy_latents, timesteps, ct_image_2)
    max_abs_diff = (out1 - out2).abs().max().item()
    assert max_abs_diff > 0.0


def test_controlnet_gradient_flow():
    """ControlNet branch should propagate gradients back to CT input."""
    controlnet = _build_small_controlnet()

    noisy_latents = torch.randn(1, 4, 16, 16, requires_grad=True)
    timesteps = torch.randint(0, 1000, (1,))
    ct_image = torch.randn(1, 1, 64, 64, requires_grad=True)

    control_outputs = controlnet(noisy_latents, timesteps, ct_image)
    loss = sum(out.mean() for out in control_outputs)
    loss.backward()
    assert ct_image.grad is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
