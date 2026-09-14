"""Top-K Focal Lesion Loss – focuses on the brightest k% pixels.

For lesions occupying 0.1% of pixels, this loss is a survival necessity.
Only the top-k% brightest pixels in the target PET contribute,
with Focal weighting so hard-to-predict pixels get larger gradients.

Active only at late denoising steps (τ < active_tau_max).
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces import LossContext, LossTerm
from .base import smooth_tau_gate


class TopKLesionLoss(LossTerm):
    name = "topk_lesion"

    def __init__(
        self,
        topk_percent: float = 0.01,
        focal_gamma: float = 2.0,
        active_tau_max: float = 0.25,
        enabled: bool = True,
        weight: float = 0.2,
    ):
        super().__init__(enabled=enabled, weight=weight)
        self.topk_percent = topk_percent
        self.focal_gamma = focal_gamma
        self.active_tau_max = active_tau_max

    def forward(self, ctx: LossContext) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.enabled:
            return (
                torch.tensor(0.0, device=ctx.target_pet.device),
                {f"{self.name}/enabled": torch.tensor(0.0)},
            )

        gate = smooth_tau_gate(ctx.tau, max_tau=self.active_tau_max)

        # Use pred_x0 if available, else fallback to model_pred
        pred = ctx.pred_x0 if ctx.pred_x0 is not None else ctx.model_pred
        target = ctx.target_pet

        k = max(1, int(target[0].numel() * self.topk_percent))
        flat_target = target.reshape(target.shape[0], -1)
        threshold = torch.topk(flat_target, k, dim=1).values[:, -1].view(-1, 1, 1, 1)
        mask = (target >= threshold).float()

        abs_error = (pred - target).abs()
        focal_weight = (1.0 - torch.exp(-abs_error)).pow(self.focal_gamma)
        loss = (mask * focal_weight * abs_error).sum() / (mask.sum() + 1e-8)

        # Apply tau gate across batch
        loss = (loss * gate.mean()).mean()

        return loss * self.weight, {
            f"{self.name}/loss": loss.detach(),
            f"{self.name}/gate_mean": gate.mean().detach(),
            f"{self.name}/enabled": torch.tensor(1.0),
        }
