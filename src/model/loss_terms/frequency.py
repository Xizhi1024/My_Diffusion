"""Focal Frequency Loss – ICCV 2021.

Computes L1 in Fourier domain with Focal weighting: frequency components
with large errors get higher weight.  This directly combats "PET blurriness"
because high-frequency lesion edges are explicitly encouraged.

Active only at mid-to-late denoising steps (τ < active_tau_max).
"""

from typing import Dict

import torch
import torch.nn as nn

from ..interfaces import LossContext, LossTerm
from .base import smooth_tau_gate


class FocalFrequencyLoss(LossTerm):
    name = "focal_frequency"

    def __init__(
        self,
        alpha: float = 1.0,
        active_tau_max: float = 0.4,
        enabled: bool = True,
        weight: float = 0.1,
    ):
        super().__init__(enabled=enabled, weight=weight)
        self.alpha = alpha
        self.active_tau_max = active_tau_max

    def forward(self, ctx: LossContext) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.enabled:
            return (
                torch.tensor(0.0, device=ctx.target_pet.device),
                {f"{self.name}/enabled": torch.tensor(0.0)},
            )

        gate = smooth_tau_gate(ctx.tau, max_tau=self.active_tau_max)

        pred = ctx.pred_x0 if ctx.pred_x0 is not None else ctx.model_pred
        target = ctx.target_pet

        # rfft2 does not support bf16; cast to float32
        pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
        target_fft = torch.fft.rfft2(target.float(), norm="ortho")

        # Frequency distance (real + imag)
        freq_error = torch.abs(pred_fft - target_fft)

        # Focal weight: spectrum matrix weighted by error magnitude
        weight_matrix = freq_error.detach().pow(self.alpha)
        loss = (weight_matrix * freq_error).mean()

        loss = (loss * gate.mean()).mean()

        return loss * self.weight, {
            f"{self.name}/loss": loss.detach(),
            f"{self.name}/gate_mean": gate.mean().detach(),
            f"{self.name}/enabled": torch.tensor(1.0),
        }
