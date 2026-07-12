"""Hotspot prior supervision loss.

Directly supervises the tiny CT→hotspot prior with lesion masks, optional
PET high-uptake targets, and optional relaxed distance maps.

target_mode (default ``mask_only``):
  - ``mask_only``             : target = lesion mask only.  Strict, never pulls
                                in far-away high-uptake pixels.  Use this for the
                                PNG baseline (no clean SUV / no reliable uptake).
  - ``mask_plus_local_uptake``: target = max(lesion mask, high-uptake pixels
                                inside the *dilated* lesion ROI).  The full-image
                                top-percentile shortcut is forbidden here.
  - ``legacy``                : target = max(lesion mask, full-image top-q% PET).
                                Kept for backward compatibility; NOT recommended
                                for the PNG baseline (pulls in off-mask hotspots).
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from ..interfaces import LossContext, LossTerm
from .base import smooth_tau_gate


def _safe_prob(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(1e-5, 1.0 - 1e-5)


def _normalise_map(x: torch.Tensor) -> torch.Tensor:
    lo = x.amin(dim=(2, 3), keepdim=True)
    hi = x.amax(dim=(2, 3), keepdim=True)
    return (x - lo) / (hi - lo).clamp_min(1e-6)


def _dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Dilate a binary mask via max-pooling. radius<=0 returns the mask unchanged."""
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius)


def _get_distance_target(ctx: LossContext) -> Optional[torch.Tensor]:
    for key in ("hotspot_distance", "lesion_distance", "distance_map", "organ_distance"):
        value = ctx.batch.get(key)
        if value is not None and value.shape[1] == 1:
            return _normalise_map(value)
    return None


class HotspotPriorLoss(LossTerm):
    name = "hotspot_prior"

    def __init__(
        self,
        active_tau_min: float = 0.25,
        active_tau_max: float = 0.75,
        focal_gamma: float = 2.0,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        distance_weight: float = 0.25,
        pet_threshold_quantile: float = 0.95,
        target_mode: str = "mask_only",
        local_uptake_radius: int = 8,
        enabled: bool = True,
        weight: float = 0.1,
    ):
        super().__init__(enabled=enabled, weight=weight)
        if target_mode not in {"mask_only", "mask_plus_local_uptake", "legacy"}:
            raise ValueError(
                f"hotspot_prior.target_mode must be one of "
                f"mask_only / mask_plus_local_uptake / legacy, got {target_mode!r}"
            )
        self.active_tau_min = active_tau_min
        self.active_tau_max = active_tau_max
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.distance_weight = distance_weight
        self.pet_threshold_quantile = pet_threshold_quantile
        self.target_mode = target_mode
        self.local_uptake_radius = local_uptake_radius

    def _target(self, ctx: LossContext) -> torch.Tensor:
        mask = ctx.batch.get("mask")
        if mask is None:
            mask = torch.zeros_like(ctx.target_pet)
        mask = mask.float()

        # mask_only: target strictly from the lesion mask.  No PET-derived
        # pixels are added, so far-away high-uptake regions can never enter.
        if self.target_mode == "mask_only":
            return mask

        pet = _normalise_map(ctx.target_pet.detach())

        if self.target_mode == "mask_plus_local_uptake":
            # Restrict high-uptake selection to the dilated lesion ROI only.
            dilated = _dilate_mask(mask, self.local_uptake_radius)
            local_pet = pet * dilated
            B = pet.shape[0]
            uptake = torch.zeros_like(mask)
            for b in range(B):
                d = dilated[b, 0]
                n_pix = int(d.sum().item())
                if n_pix < 1:
                    continue
                vals = pet[b, 0][d > 0]  # flatten within ROI
                q = torch.quantile(vals, self.pet_threshold_quantile)
                uptake[b, 0] = ((pet[b, 0] >= q) & (d > 0)).float()
            _ = local_pet  # kept for debugging / diagnostics
            return torch.maximum(mask, uptake)

        # legacy: full-image top-percentile (NOT recommended for PNG baseline)
        q = torch.quantile(
            pet.reshape(pet.shape[0], -1),
            self.pet_threshold_quantile,
            dim=1,
        ).view(-1, 1, 1, 1)
        uptake = (pet >= q).to(mask.dtype)
        return torch.maximum(mask, uptake)

    def forward(self, ctx: LossContext) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred = ctx.condition.get_map("hotspot_prior")
        if not self.enabled or pred is None:
            return (
                torch.tensor(0.0, device=ctx.target_pet.device),
                {f"{self.name}/enabled": torch.tensor(0.0, device=ctx.target_pet.device)},
            )

        early_gate = smooth_tau_gate(ctx.tau, max_tau=self.active_tau_max)
        late_gate = 1.0 - smooth_tau_gate(ctx.tau, max_tau=self.active_tau_min)
        gate = early_gate * late_gate

        target = self._target(ctx).to(device=pred.device, dtype=pred.dtype)
        pred = _safe_prob(pred)

        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (union + 1.0)).mean()

        # BCE is unsafe while CUDA autocast is enabled, even if inputs are
        # explicitly cast to float32.
        autocast_device = pred.device.type if pred.device.type in {"cuda", "cpu"} else "cpu"
        with torch.amp.autocast(autocast_device, enabled=False):
            pred_f = pred.float()
            target_f = target.float()
            bce = F.binary_cross_entropy(pred_f, target_f, reduction="none")
            pt = torch.where(target_f > 0.5, pred_f, 1.0 - pred_f)
        focal_loss = ((1.0 - pt).pow(self.focal_gamma) * bce).mean()

        distance_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        distance_target = _get_distance_target(ctx)
        if distance_target is not None:
            distance_target = distance_target.to(device=pred.device, dtype=pred.dtype)
            distance_loss = F.l1_loss(pred, distance_target)

        loss = (
            self.dice_weight * dice_loss
            + self.focal_weight * focal_loss
            + self.distance_weight * distance_loss
        )
        loss = loss * gate.mean()

        return loss * self.weight, {
            f"{self.name}/loss": loss.detach(),
            f"{self.name}/dice": dice_loss.detach(),
            f"{self.name}/focal": focal_loss.detach(),
            f"{self.name}/distance": distance_loss.detach(),
            f"{self.name}/gate_mean": gate.mean().detach(),
            f"{self.name}/enabled": torch.tensor(1.0, device=pred.device),
        }
