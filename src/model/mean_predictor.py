"""CT-conditioned PET mean predictors used by residual diffusion models."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frequency.haar import haar_dwt2, reconstruct_lowpass


class LowFrequencyPETPredictor(nn.Module):
    """Predict PET LL coefficients and reconstruct with zero detail bands.

    The architectural restriction prevents this branch from directly emitting
    the level-1/level-2 detail coefficients later assigned to the residual
    bridge.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32, levels: int = 2):
        super().__init__()
        if levels != 2:
            raise ValueError("LowFrequencyPETPredictor currently supports exactly two levels")
        if base_channels < 4:
            raise ValueError("base_channels must be at least 4")
        self.levels = levels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, 1, 1),
        )

    def forward(self, ct: torch.Tensor) -> Dict[str, torch.Tensor]:
        if ct.ndim != 4:
            raise ValueError(f"Expected CT tensor [B,C,H,W], got {tuple(ct.shape)}")
        divisor = 2 ** self.levels
        if ct.shape[-2] % divisor or ct.shape[-1] % divisor:
            raise ValueError(
                f"CT spatial dimensions must be divisible by {divisor}, got {tuple(ct.shape[-2:])}"
            )
        ll2 = self.encoder(ct)
        mean_pet = reconstruct_lowpass(ll2, levels=self.levels)
        return {"ll2": ll2, "mean_pet": mean_pet}


class _ConvBlock(nn.Module):
    """Pix2Pix-compatible encoder block used by the full-image mean."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm: bool = True,
        activation: str = "leaky",
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=not norm,
            )
        ]
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        if activation == "leaky":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpBlock(nn.Module):
    """Pix2Pix-compatible decoder block used by the full-image mean."""

    def __init__(self, in_channels: int, out_channels: int, *, dropout: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FullImagePETPredictor(nn.Module):
    """Full-resolution 2D U-Net mean with Pix2Pix-generator-compatible weights.

    The module names and tensor shapes intentionally match
    ``wuzhe.comparison_experiments.models.UNetGenerator``.  A comparison
    checkpoint can therefore be reused after stripping its ``generator.``
    state-dict prefix.  Unlike :class:`LowFrequencyPETPredictor`, this predictor
    is allowed to model both the global uptake field and lesion detail; the
    residual diffusion then learns only the correction left by the strongest
    deterministic baseline.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        levels: int = 2,
    ) -> None:
        super().__init__()
        if levels != 2:
            raise ValueError("FullImagePETPredictor currently supports exactly two Haar levels")
        if base_channels < 4:
            raise ValueError("base_channels must be at least 4")
        self.levels = levels
        c = base_channels
        self.down1 = _ConvBlock(in_channels, c, norm=False)
        self.down2 = _ConvBlock(c, c * 2)
        self.down3 = _ConvBlock(c * 2, c * 4)
        self.down4 = _ConvBlock(c * 4, c * 8)
        self.bottleneck = _ConvBlock(
            c * 8,
            c * 8,
            norm=False,
            activation="relu",
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.up1 = _UpBlock(c * 8, c * 4, dropout=True)
        self.up2 = _UpBlock(c * 8, c * 2)
        self.up3 = _UpBlock(c * 4, c)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(c * 2, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, ct: torch.Tensor) -> Dict[str, torch.Tensor]:
        if ct.ndim != 4:
            raise ValueError(f"Expected CT tensor [B,C,H,W], got {tuple(ct.shape)}")
        input_hw = ct.shape[-2:]
        d1 = self.down1(ct)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        bottleneck = self.bottleneck(d4)

        u1 = self.up1(bottleneck)
        if u1.shape[-2:] != d3.shape[-2:]:
            u1 = F.interpolate(u1, size=d3.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u1, d3], dim=1))
        if u2.shape[-2:] != d2.shape[-2:]:
            u2 = F.interpolate(u2, size=d2.shape[-2:], mode="bilinear", align_corners=False)
        u3 = self.up3(torch.cat([u2, d2], dim=1))
        if u3.shape[-2:] != d1.shape[-2:]:
            u3 = F.interpolate(u3, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        mean_pet = self.final(torch.cat([u3, d1], dim=1))
        if mean_pet.shape[-2:] != input_hw:
            mean_pet = F.interpolate(
                mean_pet,
                size=input_hw,
                mode="bilinear",
                align_corners=False,
            )
        ll1, _ = haar_dwt2(mean_pet)
        ll2, _ = haar_dwt2(ll1)
        return {"ll2": ll2, "mean_pet": mean_pet}


def build_mean_predictor(config: Mapping[str, Any]) -> nn.Module:
    """Build a backward-compatible conditional PET mean predictor."""

    architecture = str(config.get("architecture", "low_frequency")).strip().lower()
    common = {
        "in_channels": int(config.get("in_channels", 1)),
        "base_channels": int(config.get("base_channels", 32)),
        "levels": int(config.get("levels", 2)),
    }
    if architecture in {"low_frequency", "ll2"}:
        return LowFrequencyPETPredictor(**common)
    if architecture in {"full_image_unet", "unet"}:
        return FullImagePETPredictor(
            **common,
            out_channels=int(config.get("out_channels", 1)),
        )
    raise ValueError(
        "modules.conditional_mean.architecture must be one of "
        "{'low_frequency', 'full_image_unet'}"
    )
