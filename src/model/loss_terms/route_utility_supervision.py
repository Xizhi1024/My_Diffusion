"""Explicit destination utility supervision for hierarchical spectral routing.

The end-to-end image loss cannot identify an internal native-vs-shallow
decision when both branches and the U-Net co-adapt.  This term gives the
destination a stable meaning: lesion-concentrated target-residual detail may
move one level shallower, while non-lesion detail stays at its native level.
The lesion mask and target PET are supervision only and never enter inference.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ..frequency.haar import haar_dwt2
from ..interfaces import LossContext, LossTerm
from .base import smooth_tau_gate


def _stack_details(details) -> torch.Tensor:
    return torch.cat(details, dim=1)


class RouteUtilitySupervisionLoss(LossTerm):
    """Teach global level utility and local lesion/background destination."""

    name = "route_utility_supervision"

    def __init__(
        self,
        lesion_dilate_radius: int = 3,
        spatial_weight: float = 1.0,
        global_weight: float = 0.25,
        spatial_tv_weight: float = 1.0e-3,
        positive_weight: float = 4.0,
        background_target: float = 0.02,
        lesion_target: float = 0.90,
        global_target_min: float = 0.05,
        global_target_max: float = 0.55,
        active_tau_max: float = 0.70,
        enabled: bool = True,
        weight: float = 0.05,
    ) -> None:
        super().__init__(enabled=enabled, weight=weight)
        if lesion_dilate_radius < 0:
            raise ValueError("lesion_dilate_radius must be non-negative")
        if spatial_weight < 0 or global_weight < 0 or spatial_tv_weight < 0:
            raise ValueError("route utility weights must be non-negative")
        if positive_weight < 1:
            raise ValueError("positive_weight must be at least 1")
        if not 0.0 < background_target < lesion_target < 1.0:
            raise ValueError(
                "targets must satisfy 0 < background < lesion < 1"
            )
        if not 0.0 < global_target_min < global_target_max < 1.0:
            raise ValueError(
                "global targets must satisfy 0 < min < max < 1"
            )
        if not 0.0 < active_tau_max <= 1.0:
            raise ValueError("active_tau_max must be in (0, 1]")
        self.lesion_dilate_radius = int(lesion_dilate_radius)
        self.spatial_weight = float(spatial_weight)
        self.global_weight = float(global_weight)
        self.spatial_tv_weight = float(spatial_tv_weight)
        self.positive_weight = float(positive_weight)
        self.background_target = float(background_target)
        self.lesion_target = float(lesion_target)
        self.global_target_min = float(global_target_min)
        self.global_target_max = float(global_target_max)
        self.active_tau_max = float(active_tau_max)

    def _resize_lesion(
        self,
        mask: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        lesion = (mask > 0.5).float()
        if self.lesion_dilate_radius > 0:
            width = 2 * self.lesion_dilate_radius + 1
            lesion = F.max_pool2d(
                lesion,
                kernel_size=width,
                stride=1,
                padding=self.lesion_dilate_radius,
            )
        return F.interpolate(lesion, size=size, mode="nearest")

    def _targets(
        self,
        details: torch.Tensor,
        lesion: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            energy = details.detach().float().abs()
            scale = energy.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            normalized_energy = torch.tanh(energy / scale)
            spatial = self.background_target + (
                self.lesion_target - self.background_target
            ) * lesion * (0.5 + 0.5 * normalized_energy)

            lesion_pixels = lesion.sum(dim=(-2, -1)).clamp_min(1.0)
            background = 1.0 - lesion
            background_pixels = background.sum(dim=(-2, -1)).clamp_min(1.0)
            lesion_energy = (energy * lesion).sum(
                dim=(-2, -1)
            ) / lesion_pixels
            background_energy = (energy * background).sum(
                dim=(-2, -1)
            ) / background_pixels
            contrast = lesion_energy / (
                lesion_energy + background_energy + 1e-6
            )
            has_lesion = (
                lesion.flatten(1).sum(dim=1) > 0
            ).float()[:, None]
            global_target = self.global_target_min + (
                self.global_target_max - self.global_target_min
            ) * contrast * has_lesion
        return spatial, global_target

    @staticmethod
    def _per_sample_mean(value: torch.Tensor) -> torch.Tensor:
        return value.flatten(1).mean(dim=1)

    def forward(
        self,
        ctx: LossContext,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        reference = ctx.target_pet
        zero = reference.new_zeros(())
        if not self.enabled:
            return zero, {
                f"{self.name}/enabled": zero,
                f"{self.name}/loss": zero,
            }
        mask = ctx.batch.get("mask")
        if mask is None:
            raise ValueError(f"{self.name} requires batch['mask']")
        spatial_l2 = ctx.condition.maps.get(
            "spectral_route_spatial_shallow_l2"
        )
        spatial_l1 = ctx.condition.maps.get(
            "spectral_route_spatial_shallow_l1"
        )
        global_route = ctx.condition.scalars.get(
            "spectral_route_conditional_shallow"
        )
        if spatial_l2 is None or spatial_l1 is None or global_route is None:
            raise ValueError(
                f"{self.name} requires spatial destination maps and the "
                "global conditional-shallow route"
            )

        target_residual = (
            ctx.target_residual
            if ctx.target_residual is not None
            else ctx.target_pet
        )
        ll1, details_l1 = haar_dwt2(target_residual.float())
        _, details_l2 = haar_dwt2(ll1)
        target_details = (
            _stack_details(details_l2),
            _stack_details(details_l1),
        )
        predictions = (spatial_l2.float(), spatial_l1.float())
        global_route = global_route.float()
        gate = smooth_tau_gate(
            ctx.tau,
            max_tau=self.active_tau_max,
        ).float()

        spatial_losses = []
        global_losses = []
        target_stds = []
        route_stds = []
        global_targets = []
        spatial_tv = []
        for level_index, (prediction, details) in enumerate(
            zip(predictions, target_details)
        ):
            lesion = self._resize_lesion(
                mask.to(device=prediction.device),
                prediction.shape[-2:],
            ).expand(-1, 3, -1, -1)
            spatial_target, global_target = self._targets(details, lesion)
            probability = prediction.clamp(1e-5, 1.0 - 1e-5)
            # Probability-space BCE is explicitly rejected inside CUDA AMP.
            # Converting the bounded route probability back to logits keeps
            # the exact same objective while using the autocast-safe kernel.
            point_loss = F.binary_cross_entropy_with_logits(
                torch.logit(probability),
                spatial_target,
                reduction="none",
            )
            weights = 1.0 + (
                self.positive_weight - 1.0
            ) * lesion
            spatial_losses.append(
                self._per_sample_mean(point_loss * weights)
            )

            global_probability = global_route[:, level_index].float().clamp(
                1e-5,
                1.0 - 1e-5,
            )
            global_losses.append(
                F.binary_cross_entropy_with_logits(
                    torch.logit(global_probability),
                    global_target,
                    reduction="none",
                ).mean(dim=1)
            )
            target_stds.append(spatial_target.std())
            route_stds.append(prediction.float().std())
            global_targets.append(global_target.mean())
            spatial_tv.append(
                (
                    (prediction[..., 1:, :] - prediction[..., :-1, :])
                    .abs()
                    .mean()
                    + (
                        prediction[..., :, 1:]
                        - prediction[..., :, :-1]
                    )
                    .abs()
                    .mean()
                )
            )

        spatial_per_sample = torch.stack(spatial_losses).mean(dim=0)
        global_per_sample = torch.stack(global_losses).mean(dim=0)
        per_sample = (
            self.spatial_weight * spatial_per_sample
            + self.global_weight * global_per_sample
        )
        tv = torch.stack(spatial_tv).mean()
        loss = (
            (per_sample * gate).sum() / gate.sum().clamp_min(1.0)
            + self.spatial_tv_weight * tv
        )
        raw = self.weight * loss
        return raw, {
            f"{self.name}/enabled": reference.new_tensor(1.0),
            f"{self.name}/loss": loss.detach(),
            f"{self.name}/spatial": spatial_per_sample.mean().detach(),
            f"{self.name}/global": global_per_sample.mean().detach(),
            f"{self.name}/spatial_tv": tv.detach(),
            f"{self.name}/route_spatial_std": torch.stack(
                route_stds
            ).mean().detach(),
            f"{self.name}/target_spatial_std": torch.stack(
                target_stds
            ).mean().detach(),
            f"{self.name}/global_target": torch.stack(
                global_targets
            ).mean().detach(),
            f"{self.name}/gate_mean": gate.mean().detach(),
        }
