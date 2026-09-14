"""Dense lesion ROI and peak-ranking losses.

These terms directly target the current failure mode: predicted PET hotspots
forming outside the small lesion mask while the masked lesion remains too cold.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ..interfaces import LossContext, LossTerm
from .base import smooth_tau_gate


def _dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilation via max-pooling, preserving [B, 1, H, W]."""
    mask = (mask > 0.5).float()
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)


class LesionROIL1Loss(LossTerm):
    """Dense reconstruction loss inside a dilated lesion ROI."""

    name = "lesion_roi_l1"

    def __init__(
        self,
        dilate_radius: int = 3,
        beta: float = 0.05,
        active_tau_max: float = 0.7,
        enabled: bool = True,
        weight: float = 1.0,
    ):
        super().__init__(enabled=enabled, weight=weight)
        self.dilate_radius = dilate_radius
        self.beta = beta
        self.active_tau_max = active_tau_max

    def forward(self, ctx: LossContext) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = ctx.target_pet.device
        zero = torch.tensor(0.0, device=device)
        if not self.enabled:
            return zero, {f"{self.name}/enabled": zero}

        lesion = ctx.batch.get("mask")
        if lesion is None or lesion.sum() <= 0:
            return zero, {
                f"{self.name}/enabled": torch.tensor(1.0, device=device),
                f"{self.name}/loss": zero,
                f"{self.name}/roi_pixels": zero,
            }

        pred = ctx.pred_x0 if ctx.pred_x0 is not None else ctx.model_pred
        target = ctx.target_pet
        roi = _dilate_mask(lesion.to(device=device, dtype=target.dtype), self.dilate_radius)

        if self.beta > 0:
            diff = F.smooth_l1_loss(pred, target, beta=self.beta, reduction="none")
        else:
            diff = (pred - target).abs()
        reduce_dims = tuple(range(1, diff.dim()))
        roi_sum = roi.sum(dim=reduce_dims).clamp_min(1.0)
        per_sample = (diff * roi).sum(dim=reduce_dims) / roi_sum
        gate = smooth_tau_gate(ctx.tau, max_tau=self.active_tau_max).to(device=device)
        loss = (per_sample * gate).mean()

        return loss * self.weight, {
            f"{self.name}/loss": loss.detach(),
            f"{self.name}/gate_mean": gate.mean().detach(),
            f"{self.name}/roi_pixels": roi.sum().detach(),
            f"{self.name}/enabled": torch.tensor(1.0, device=device),
        }


class OutsidePeakRankingLoss(LossTerm):
    """Force the brightest lesion-region peak to exceed outside-mask peaks."""

    name = "outside_peak_ranking"

    def __init__(
        self,
        margin: float = 0.05,
        inside_radius: int = 3,
        outside_radius: int = 8,
        topk_percent: float = 0.01,
        active_tau_max: float = 0.65,
        enabled: bool = True,
        weight: float = 0.1,
    ):
        super().__init__(enabled=enabled, weight=weight)
        self.margin = margin
        self.inside_radius = inside_radius
        self.outside_radius = outside_radius
        self.topk_percent = topk_percent
        self.active_tau_max = active_tau_max

    def _topk_mean(self, pred: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = []
        valid = []
        pred_flat = pred.flatten(1)
        mask_flat = (mask.flatten(1) > 0.5)
        for i in range(pred_flat.shape[0]):
            selected = pred_flat[i][mask_flat[i]]
            if selected.numel() == 0:
                values.append(torch.tensor(0.0, device=pred.device, dtype=pred.dtype))
                valid.append(torch.tensor(False, device=pred.device))
                continue
            k = max(1, int(selected.numel() * self.topk_percent))
            k = min(k, selected.numel())
            values.append(torch.topk(selected, k=k).values.mean())
            valid.append(torch.tensor(True, device=pred.device))
        return torch.stack(values), torch.stack(valid)

    def forward(self, ctx: LossContext) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = ctx.target_pet.device
        zero = torch.tensor(0.0, device=device)
        if not self.enabled:
            return zero, {f"{self.name}/enabled": zero}

        lesion = ctx.batch.get("mask")
        if lesion is None or lesion.sum() <= 0:
            return zero, {
                f"{self.name}/enabled": torch.tensor(1.0, device=device),
                f"{self.name}/loss": zero,
            }

        pred = ctx.pred_x0 if ctx.pred_x0 is not None else ctx.model_pred
        lesion = lesion.to(device=device, dtype=pred.dtype)
        inside = _dilate_mask(lesion, self.inside_radius)
        outside = 1.0 - _dilate_mask(lesion, self.outside_radius)

        inside_peak, inside_valid = self._topk_mean(pred, inside)
        outside_peak, outside_valid = self._topk_mean(pred, outside)
        valid = (inside_valid & outside_valid).to(dtype=pred.dtype)
        per_sample = F.relu(outside_peak - inside_peak + self.margin)
        gate = smooth_tau_gate(ctx.tau, max_tau=self.active_tau_max).to(device=device)
        denom = valid.sum().clamp_min(1.0)
        loss = (per_sample * gate * valid).sum() / denom

        return loss * self.weight, {
            f"{self.name}/loss": loss.detach(),
            f"{self.name}/inside_peak": inside_peak.mean().detach(),
            f"{self.name}/outside_peak": outside_peak.mean().detach(),
            f"{self.name}/gate_mean": gate.mean().detach(),
            f"{self.name}/enabled": torch.tensor(1.0, device=device),
        }
