"""Boundary-focused PET supervision for lesion and anatomy fidelity."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from ..frequency.haar import haar_dwt2
from ..interfaces import LossContext, LossTerm
from .base import smooth_tau_gate


def _gradient_magnitude(image: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    dx = F.pad(image[..., :, 1:] - image[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(image[..., 1:, :] - image[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + epsilon ** 2)


def _edge_weight(image: torch.Tensor) -> torch.Tensor:
    # Inputs used for consensus are conditions/targets, so an exact zero map is
    # preferable to adding epsilon and manufacturing edges in uniform regions.
    dx = F.pad(image[..., :, 1:] - image[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(image[..., 1:, :] - image[..., :-1, :], (0, 0, 0, 1))
    magnitude = torch.sqrt(dx.square() + dy.square())
    maximum = magnitude.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (magnitude / maximum).clamp(0.0, 1.0)


def _boundary_ring(mask: torch.Tensor, radius: int) -> torch.Tensor:
    binary = (mask > 0.5).to(mask.dtype)
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(binary, kernel, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - binary, kernel, stride=1, padding=radius)
    return (dilated - eroded).clamp(0.0, 1.0)


def _available_boundary(
    mask: Optional[torch.Tensor],
    reference: torch.Tensor,
    radius: int,
    collapse_channels: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    zero_map = reference.new_zeros(reference.shape[0], 1, *reference.shape[-2:])
    zero = reference.new_zeros(())
    if mask is None:
        return zero_map, zero
    mask = mask.to(device=reference.device, dtype=reference.dtype)
    if mask.shape[-2:] != reference.shape[-2:]:
        mask = F.interpolate(mask, size=reference.shape[-2:], mode="nearest")
    if torch.count_nonzero(mask > 0.5).item() == 0:
        return zero_map, zero
    ring = _boundary_ring(mask, radius)
    if collapse_channels and ring.shape[1] > 1:
        ring = ring.amax(dim=1, keepdim=True)
    return ring, torch.ones_like(zero)


def _weighted_charbonnier(
    prediction: torch.Tensor,
    target: torch.Tensor,
    spatial_weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    if spatial_weight.sum().detach().item() <= 0:
        return prediction.new_zeros(())
    error = torch.sqrt((prediction - target).square() + epsilon ** 2) - epsilon
    denominator = spatial_weight.sum() * prediction.shape[1]
    return (error * spatial_weight).sum() / denominator.clamp_min(1.0)


class BoundaryFrequencyLoss(LossTerm):
    """Supervise predicted-x0 gradients and Haar details at valid boundaries."""

    name = "boundary_frequency"

    def __init__(
        self,
        lesion_weight: float = 1.0,
        anatomy_weight: float = 0.5,
        organ_weight: float = 0.5,
        wavelet_weight: float = 0.25,
        boundary_radius: int = 2,
        epsilon: float = 1e-3,
        active_tau_max: float = 0.7,
        enabled: bool = True,
        weight: float = 0.05,
    ) -> None:
        super().__init__(enabled=enabled, weight=weight)
        if boundary_radius < 1:
            raise ValueError("boundary_radius must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if any(value < 0 for value in (lesion_weight, anatomy_weight, organ_weight, wavelet_weight)):
            raise ValueError("boundary component weights must be non-negative")
        self.lesion_weight = float(lesion_weight)
        self.anatomy_weight = float(anatomy_weight)
        self.organ_weight = float(organ_weight)
        self.wavelet_weight = float(wavelet_weight)
        self.boundary_radius = int(boundary_radius)
        self.epsilon = float(epsilon)
        self.active_tau_max = float(active_tau_max)

    def forward(self, ctx: LossContext) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = ctx.target_pet.new_zeros(())
        if not self.enabled or ctx.pred_x0 is None:
            return zero, {
                f"{self.name}/enabled": zero,
                f"{self.name}/available": zero,
                f"{self.name}/loss": zero,
            }

        prediction = ctx.pred_x0
        target = ctx.target_pet
        ct = ctx.batch.get("ct")
        if ct is None:
            ct = torch.zeros_like(target)
        else:
            ct = ct.to(device=target.device, dtype=target.dtype)

        lesion_ring, lesion_available = _available_boundary(
            ctx.batch.get("mask"), target, self.boundary_radius
        )
        organ_ring, organ_available = _available_boundary(
            ctx.batch.get("organ_mask"),
            target,
            self.boundary_radius,
            collapse_channels=True,
        )
        anatomy_consensus = _edge_weight(ct) * _edge_weight(target)

        pred_gradient = _gradient_magnitude(prediction, self.epsilon)
        target_gradient = _gradient_magnitude(target, self.epsilon)
        lesion = _weighted_charbonnier(
            pred_gradient, target_gradient, lesion_ring, self.epsilon
        )
        anatomy = _weighted_charbonnier(
            pred_gradient, target_gradient, anatomy_consensus, self.epsilon
        )
        organ = _weighted_charbonnier(
            pred_gradient, target_gradient, organ_ring, self.epsilon
        )

        boundary_union = torch.maximum(lesion_ring, anatomy_consensus)
        boundary_union = torch.maximum(boundary_union, organ_ring)
        _, pred_details = haar_dwt2(prediction)
        _, target_details = haar_dwt2(target)
        pred_high = torch.cat(pred_details, dim=1)
        target_high = torch.cat(target_details, dim=1)
        wavelet_region = F.adaptive_max_pool2d(
            boundary_union, pred_high.shape[-2:]
        )
        wavelet = _weighted_charbonnier(
            pred_high, target_high, wavelet_region, self.epsilon
        )

        raw = (
            self.lesion_weight * lesion
            + self.anatomy_weight * anatomy
            + self.organ_weight * organ
            + self.wavelet_weight * wavelet
        )
        reliability = ctx.condition.scalars.get("frequency_noise_reliability")
        if reliability is None:
            gate = smooth_tau_gate(ctx.tau, max_tau=self.active_tau_max).mean()
        else:
            gate = reliability.to(device=raw.device, dtype=raw.dtype).mean()
        gated = raw * gate
        available = torch.maximum(
            lesion_available,
            torch.maximum(organ_available, (anatomy_consensus.sum() > 0).to(zero)),
        )
        return gated * self.weight, {
            f"{self.name}/enabled": torch.ones_like(zero),
            f"{self.name}/available": available.detach(),
            f"{self.name}/lesion_available": lesion_available.detach(),
            f"{self.name}/organ_available": organ_available.detach(),
            f"{self.name}/lesion": lesion.detach(),
            f"{self.name}/anatomy": anatomy.detach(),
            f"{self.name}/organ": organ.detach(),
            f"{self.name}/wavelet": wavelet.detach(),
            f"{self.name}/gate_mean": gate.detach(),
            f"{self.name}/loss": gated.detach(),
        }
