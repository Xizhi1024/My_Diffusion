"""Robust normalized lesion peak supervision for PNG PET targets.

Matches per-sample Top-Q lesion peaks in normalized model space,
with an asymmetric cold penalty that strongly penalizes underestimation.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F

from ..interfaces import LossContext, LossTerm
from .base import smooth_tau_gate


class NormalizedLesionPeakLoss(LossTerm):
    """Match per-sample Top-Q lesion peaks in normalized model space."""

    name = "normalized_lesion_peak"

    def __init__(
        self,
        topk_percent: float = 0.10,
        min_k: int = 3,
        max_k: int = 16,
        beta: float = 0.02,
        active_tau_max: float = 0.25,
        cold_weight: float = 1.0,
        cold_tolerance: float = 0.02,
        enabled: bool = True,
        weight: float = 0.05,
    ) -> None:
        super().__init__(enabled=enabled, weight=weight)
        if not 0.0 < topk_percent <= 1.0:
            raise ValueError("topk_percent must be in (0, 1]")
        if min_k < 1:
            raise ValueError("min_k must be positive")
        if max_k < min_k:
            raise ValueError("max_k must be >= min_k")
        if beta <= 0:
            raise ValueError("beta must be positive")
        if not 0.0 <= active_tau_max <= 1.0:
            raise ValueError("active_tau_max must be in [0, 1]")
        if cold_weight < 0:
            raise ValueError("cold_weight must be non-negative")
        if cold_tolerance < 0:
            raise ValueError("cold_tolerance must be non-negative")
        self.topk_percent = float(topk_percent)
        self.min_k = int(min_k)
        self.max_k = int(max_k)
        self.beta = float(beta)
        self.active_tau_max = float(active_tau_max)
        self.cold_weight = float(cold_weight)
        self.cold_tolerance = float(cold_tolerance)

    def forward(
        self,
        ctx: LossContext,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred = ctx.pred_x0 if ctx.pred_x0 is not None else ctx.model_pred
        zero = pred.sum() * 0.0
        if not self.enabled:
            return zero, {
                f"{self.name}/enabled": zero.detach(),
                f"{self.name}/loss": zero.detach(),
            }

        lesion = ctx.batch.get("mask")
        if lesion is None:
            lesion = torch.zeros_like(ctx.target_pet)
        lesion = lesion.to(device=pred.device)
        target = ctx.target_pet.to(device=pred.device, dtype=pred.dtype)
        batch_size = pred.shape[0]
        gate = smooth_tau_gate(
            ctx.tau.to(device=pred.device, dtype=pred.dtype),
            max_tau=self.active_tau_max,
        ).reshape(-1)
        if gate.numel() == 1 and batch_size > 1:
            gate = gate.expand(batch_size)
        elif gate.numel() != batch_size:
            raise ValueError("tau must have one value per sample")

        losses = []
        pred_peaks = []
        target_peaks = []
        signed_biases = []
        cold_penalties = []
        valid_gates = []
        selected_counts = []
        for index in range(batch_size):
            valid = lesion[index] > 0.5
            selected_pred = pred[index][valid]
            if selected_pred.numel() == 0:
                continue
            selected_target = target[index][valid]
            count = int(selected_pred.numel())
            k = min(
                max(math.ceil(count * self.topk_percent), self.min_k),
                self.max_k,
                count,
            )
            pred_peak = torch.topk(selected_pred, k=k).values.mean()
            target_peak = torch.topk(selected_target, k=k).values.mean()

            # Symmetric smooth-L1 loss for general peak matching
            symmetric = F.smooth_l1_loss(
                pred_peak,
                target_peak,
                beta=self.beta,
            )

            # Asymmetric cold penalty: strongly penalize underestimation
            # (pred_peak < target_peak) while tolerating small overestimation
            cold_error = F.relu(
                target_peak - pred_peak - self.cold_tolerance
            )
            sample_loss = symmetric + self.cold_weight * cold_error

            losses.append(sample_loss * gate[index])
            pred_peaks.append(pred_peak)
            target_peaks.append(target_peak)
            signed_biases.append(pred_peak - target_peak)
            cold_penalties.append(cold_error)
            valid_gates.append(gate[index])
            selected_counts.append(pred.new_tensor(float(k)))

        if not losses:
            return zero, {
                f"{self.name}/loss": zero.detach(),
                f"{self.name}/pred_peak": zero.detach(),
                f"{self.name}/target_peak": zero.detach(),
                f"{self.name}/signed_bias": zero.detach(),
                f"{self.name}/cold_penalty": zero.detach(),
                f"{self.name}/gate_mean": zero.detach(),
                f"{self.name}/valid_count": zero.detach(),
                f"{self.name}/k_mean": zero.detach(),
                f"{self.name}/enabled": pred.new_tensor(1.0),
            }

        # Audit L2: gate-mass normalization — divide by max(sum(gate), 1),
        # not the appended-sample count (see lesion_roi_l1 / topk_lesion).
        loss = torch.stack(losses).sum() / torch.stack(valid_gates).sum().clamp_min(1.0)
        return loss * self.weight, {
            f"{self.name}/loss": loss.detach(),
            f"{self.name}/pred_peak": torch.stack(pred_peaks).mean().detach(),
            f"{self.name}/target_peak": torch.stack(target_peaks).mean().detach(),
            f"{self.name}/signed_bias": torch.stack(signed_biases).mean().detach(),
            f"{self.name}/cold_penalty": torch.stack(cold_penalties).mean().detach(),
            f"{self.name}/gate_mean": torch.stack(valid_gates).mean().detach(),
            f"{self.name}/valid_count": pred.new_tensor(float(len(losses))),
            f"{self.name}/k_mean": torch.stack(selected_counts).mean().detach(),
            f"{self.name}/enabled": pred.new_tensor(1.0),
        }
