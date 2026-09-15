"""Top-K Focal Lesion Loss – per-sample target Top-K within lesion mask.

For lesions occupying 0.1% of pixels, this loss is a survival necessity.
Each sample is processed independently: the brightest k pixels are selected
from the *target* PET within the lesion mask, and the model's prediction
at those same locations is compared against the target.  Using target
Top-K (not independent pred/target Top-K) preserves spatial correspondence.

Active only at late denoising steps (τ < active_tau_max).
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

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
        pred = ctx.pred_x0 if ctx.pred_x0 is not None else ctx.model_pred
        target = ctx.target_pet
        lesion = ctx.batch.get("mask")

        zero = pred.sum() * 0.0
        if not self.enabled or lesion is None:
            return zero, {
                f"{self.name}/loss": zero.detach(),
                f"{self.name}/enabled": pred.new_tensor(float(self.enabled)),
            }

        lesion = lesion.to(device=pred.device)
        gate = smooth_tau_gate(
            ctx.tau.to(device=pred.device, dtype=pred.dtype),
            max_tau=self.active_tau_max,
        ).reshape(-1)

        losses: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        selected_counts: list[torch.Tensor] = []

        for index in range(pred.shape[0]):
            valid = lesion[index] > 0.5
            target_values = target[index][valid]
            pred_values = pred[index][valid]

            if target_values.numel() == 0:
                continue

            count = int(target_values.numel())
            k = min(
                max(math.ceil(count * self.topk_percent), 3),
                16,
                count,
            )

            # Select positions from target, preserving spatial correspondence.
            indices = torch.topk(target_values, k=k).indices
            error = (pred_values[indices] - target_values[indices]).abs()

            focal_weight = (1.0 - torch.exp(-error)).pow(self.focal_gamma)
            sample_loss = (focal_weight * error).mean()
            losses.append(sample_loss * gate[index])
            gates.append(gate[index])
            selected_counts.append(pred.new_tensor(float(k)))

        if not losses:
            return zero, {
                f"{self.name}/loss": zero.detach(),
                f"{self.name}/enabled": pred.new_tensor(1.0),
            }

        # Audit L2: gate-mass normalization — divide by max(sum(gate), 1),
        # not the appended-sample count (see lesion_roi_l1).
        raw_loss = torch.stack(losses).sum() / torch.stack(gates).sum().clamp_min(1.0)
        weighted_loss = raw_loss * self.weight

        return weighted_loss, {
            f"{self.name}/loss": raw_loss.detach(),
            f"{self.name}/weighted_loss": weighted_loss.detach(),
            f"{self.name}/gate_mean": gate.mean().detach(),
            f"{self.name}/k_mean": torch.stack(selected_counts).mean().detach(),
            f"{self.name}/enabled": pred.new_tensor(1.0),
        }
