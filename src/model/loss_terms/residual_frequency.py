"""Frequency losses defined on the residual bridge and reconstructed PET."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from ..frequency.haar import haar_dwt2
from ..interfaces import LossContext, LossTerm
from .base import smooth_tau_gate


def _lesion_weight_map(
    mask: torch.Tensor | None,
    reference: torch.Tensor,
    lesion_weight: float,
) -> torch.Tensor:
    if mask is None:
        return torch.ones(
            reference.shape[0], 1, *reference.shape[-2:],
            device=reference.device, dtype=reference.dtype,
        )
    mask = (mask.to(device=reference.device, dtype=reference.dtype) > 0.5).to(reference.dtype)
    if mask.shape[-2:] != reference.shape[-2:]:
        mask = F.adaptive_max_pool2d(mask, reference.shape[-2:])
    return 1.0 + lesion_weight * mask


def _weighted_charbonnier(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    lesion_weight: float,
    epsilon: float,
) -> torch.Tensor:
    weight = _lesion_weight_map(mask, pred, lesion_weight)
    error = torch.sqrt((pred - target).square() + epsilon ** 2)
    return (error * weight).sum() / (weight.sum() * pred.shape[1]).clamp_min(1.0)


class ResidualWaveletLoss(LossTerm):
    name = "residual_wavelet"

    def __init__(
        self,
        lesion_weight: float = 4.0,
        band_weights: Sequence[float] = (0.5, 1.0, 1.5),
        epsilon: float = 1e-3,
        active_tau_max: float = 0.7,
        enabled: bool = True,
        weight: float = 0.05,
    ):
        super().__init__(enabled=enabled, weight=weight)
        if len(band_weights) != 3:
            raise ValueError("band_weights must contain low/mid/high weights")
        self.lesion_weight = float(lesion_weight)
        self.band_weights = tuple(float(x) for x in band_weights)
        self.epsilon = float(epsilon)
        self.active_tau_max = float(active_tau_max)

    def forward(self, ctx: LossContext) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = ctx.target_pet.new_zeros(())
        if not self.enabled:
            return zero, {f"{self.name}/enabled": zero}
        if ctx.pred_residual is None or ctx.target_residual is None:
            return zero, {
                f"{self.name}/enabled": torch.ones_like(zero),
                f"{self.name}/available": zero,
                f"{self.name}/loss": zero,
            }

        pred_ll1, pred_detail1 = haar_dwt2(ctx.pred_residual)
        target_ll1, target_detail1 = haar_dwt2(ctx.target_residual)
        pred_ll2, pred_detail2 = haar_dwt2(pred_ll1)
        target_ll2, target_detail2 = haar_dwt2(target_ll1)
        pred_mid = torch.cat(pred_detail2, dim=1)
        target_mid = torch.cat(target_detail2, dim=1)
        pred_high = torch.cat(pred_detail1, dim=1)
        target_high = torch.cat(target_detail1, dim=1)
        mask = ctx.batch.get("mask")

        low = _weighted_charbonnier(
            pred_ll2, target_ll2, mask, self.lesion_weight, self.epsilon
        )
        middle = _weighted_charbonnier(
            pred_mid, target_mid, mask, self.lesion_weight, self.epsilon
        )
        high = _weighted_charbonnier(
            pred_high, target_high, mask, self.lesion_weight, self.epsilon
        )
        raw = (
            self.band_weights[0] * low
            + self.band_weights[1] * middle
            + self.band_weights[2] * high
        )
        gate = smooth_tau_gate(ctx.tau, max_tau=self.active_tau_max).mean().to(raw)
        loss = raw * gate
        return loss * self.weight, {
            f"{self.name}/enabled": torch.ones_like(zero),
            f"{self.name}/available": torch.ones_like(zero),
            f"{self.name}/low": low.detach(),
            f"{self.name}/mid": middle.detach(),
            f"{self.name}/high": high.detach(),
            f"{self.name}/gate_mean": gate.detach(),
            f"{self.name}/loss": loss.detach(),
        }


class GaborConsistencyLoss(LossTerm):
    name = "gabor_consistency"
    _REQUIRED_MAPS = (
        "gabor_pred_feat",
        "gabor_target_feat",
        "gabor_pred_orientation",
        "gabor_target_orientation",
    )

    def __init__(
        self,
        lesion_weight: float = 3.0,
        orientation_weight: float = 0.1,
        epsilon: float = 1e-6,
        active_tau_max: float = 0.7,
        enabled: bool = True,
        weight: float = 0.02,
    ):
        super().__init__(enabled=enabled, weight=weight)
        self.lesion_weight = float(lesion_weight)
        self.orientation_weight = float(orientation_weight)
        self.epsilon = float(epsilon)
        self.active_tau_max = float(active_tau_max)

    def forward(self, ctx: LossContext) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = ctx.target_pet.new_zeros(())
        if not self.enabled:
            return zero, {f"{self.name}/enabled": zero}
        if any(name not in ctx.condition.maps for name in self._REQUIRED_MAPS):
            return zero, {
                f"{self.name}/enabled": torch.ones_like(zero),
                f"{self.name}/available": zero,
                f"{self.name}/loss": zero,
            }

        pred_feat = ctx.condition.maps["gabor_pred_feat"]
        target_feat = ctx.condition.maps["gabor_target_feat"]
        pred_orientation = ctx.condition.maps["gabor_pred_orientation"]
        target_orientation = ctx.condition.maps["gabor_target_orientation"]
        mask = ctx.batch.get("mask")

        amplitude = _weighted_charbonnier(
            pred_feat, target_feat, mask, self.lesion_weight, self.epsilon
        )
        eps = self.epsilon
        pred_prob = pred_orientation.clamp_min(eps)
        target_prob = target_orientation.clamp_min(eps)
        pred_prob = pred_prob / pred_prob.sum(dim=1, keepdim=True).clamp_min(eps)
        target_prob = target_prob / target_prob.sum(dim=1, keepdim=True).clamp_min(eps)
        mixture = 0.5 * (pred_prob + target_prob)
        js_map = 0.5 * (
            (pred_prob * (pred_prob / mixture.clamp_min(eps)).log()).sum(dim=1, keepdim=True)
            + (target_prob * (target_prob / mixture.clamp_min(eps)).log()).sum(dim=1, keepdim=True)
        )
        spatial_weight = _lesion_weight_map(mask, js_map, self.lesion_weight)
        orientation_js = (js_map * spatial_weight).sum() / spatial_weight.sum().clamp_min(1.0)
        raw = amplitude + self.orientation_weight * orientation_js
        gate = smooth_tau_gate(ctx.tau, max_tau=self.active_tau_max).mean().to(raw)
        loss = raw * gate
        return loss * self.weight, {
            f"{self.name}/enabled": torch.ones_like(zero),
            f"{self.name}/available": torch.ones_like(zero),
            f"{self.name}/amplitude": amplitude.detach(),
            f"{self.name}/orientation_js": orientation_js.detach(),
            f"{self.name}/gate_mean": gate.detach(),
            f"{self.name}/loss": loss.detach(),
        }
