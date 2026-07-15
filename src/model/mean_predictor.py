"""CT-conditioned predictor restricted to a low-frequency PET endpoint."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .frequency.haar import reconstruct_lowpass


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
