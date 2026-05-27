"""False Hotspot Suppression Loss — per-organ differentiation.

Penalises high predicted PET values in regions that should be metabolically
quiet, while allowing physiological uptake in bladder/bowel.

Organ class semantics (from organ_preprocess.py):
  0: uterus/pelvic_region  → exclude (lesion proximity)
  1: bladder               → allow physiological uptake
  2: rectum/bowel          → allow physiological uptake
  3: bone                  → penalise (pathological if high uptake)
  4: fat                   → penalise
  5: muscle/other          → penalise

Active at mid-to-late steps (tau < active_tau_max).
"""

from typing import Dict

import torch
import torch.nn as nn

from ..interfaces import LossContext, LossTerm
from .base import smooth_tau_gate


class FalseHotspotLoss(LossTerm):
    name = "false_hotspot"

    def __init__(
        self,
        active_tau_max: float = 0.3,
        threshold_percentile: float = 0.95,
        enabled: bool = True,
        weight: float = 0.05,
    ):
        super().__init__(enabled=enabled, weight=weight)
        self.active_tau_max = active_tau_max
        self.threshold_percentile = threshold_percentile

    def forward(self, ctx: LossContext) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.enabled:
            return (
                torch.tensor(0.0, device=ctx.target_pet.device),
                {f"{self.name}/enabled": torch.tensor(0.0)},
            )

        gate = smooth_tau_gate(ctx.tau, max_tau=self.active_tau_max)

        pred = ctx.pred_x0 if ctx.pred_x0 is not None else ctx.model_pred

        # Build "should-be-cold" mask from per-organ rules
        bg_mask = self._build_cold_mask(ctx)

        # Penalise high values in the cold mask
        pred_bg = pred * bg_mask
        q = torch.quantile(
            pred_bg.reshape(pred_bg.shape[0], -1), self.threshold_percentile, dim=1
        )
        high_mask = (pred > q.view(-1, 1, 1, 1)).float() * bg_mask

        n_high = high_mask.sum().clamp_min(1)
        loss = (high_mask * pred.abs()).sum() / n_high
        loss = (loss * gate.mean()).mean()

        # Per-region breakdown for logging
        logs: Dict[str, torch.Tensor] = {
            f"{self.name}/loss": loss.detach(),
            f"{self.name}/enabled": torch.tensor(1.0),
        }

        # Log mean intensity in each cold region type
        for region_name, region_mask in self._region_masks(ctx).items():
            if region_mask.sum() > 0:
                mean_val = (pred * region_mask).sum() / region_mask.sum().clamp_min(1)
                logs[f"{self.name}/{region_name}_mean"] = mean_val.detach()

        return loss * self.weight, logs

    def _build_cold_mask(self, ctx: LossContext) -> torch.Tensor:
        """Regions where high PET values indicate false positives.

        Cold = non-lesion AND (bone | fat | muscle | non-organ background).
        Excludes: lesion, bladder, rectum/bowel, uterus/pelvic.
        """
        lesion_mask = ctx.batch.get("mask")
        organ_mask = ctx.batch.get("organ_mask")  # [B, 6, H, W]

        B = ctx.target_pet.shape[0]
        device = ctx.target_pet.device
        H, W = ctx.target_pet.shape[2], ctx.target_pet.shape[3]

        # Start with all-cold
        cold = torch.ones(B, 1, H, W, device=device)

        # Exclude lesion
        if lesion_mask is not None:
            cold = cold * (1.0 - lesion_mask)

        if organ_mask is not None:
            # Class 0 (uterus/pelvic): exclude (lesion proximity)
            # Class 1 (bladder): exclude (physiological uptake allowed)
            # Class 2 (rectum/bowel): exclude (physiological uptake allowed)
            # Class 3-5 (bone/fat/muscle): keep COLD (penalise uptake)
            warm_classes = [0, 1, 2]
            for c in warm_classes:
                if c < organ_mask.shape[1]:
                    cold = cold * (1.0 - organ_mask[:, c:c+1])

        return cold

    def _region_masks(self, ctx: LossContext) -> Dict[str, torch.Tensor]:
        """Return per-region cold masks for logging."""
        organ_mask = ctx.batch.get("organ_mask")
        lesion_mask = ctx.batch.get("mask")
        B = ctx.target_pet.shape[0]
        device = ctx.target_pet.device
        H, W = ctx.target_pet.shape[2], ctx.target_pet.shape[3]

        base = torch.ones(B, 1, H, W, device=device)
        if lesion_mask is not None:
            base = base * (1.0 - lesion_mask)

        regions: Dict[str, torch.Tensor] = {}
        if organ_mask is not None:
            # Cold regions
            regions["bone"] = base * organ_mask[:, 3:4] if organ_mask.shape[1] > 3 else torch.zeros_like(base)
            regions["fat"] = base * organ_mask[:, 4:5] if organ_mask.shape[1] > 4 else torch.zeros_like(base)
            regions["muscle_cold"] = base * organ_mask[:, 5:6] if organ_mask.shape[1] > 5 else torch.zeros_like(base)
        else:
            regions["other_background"] = base

        return regions
