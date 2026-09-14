"""Segmenter Consistency Loss (L_seg) — frozen TinySegmenter as quality oracle.

Computes L1 distance between the frozen segmenter's outputs on synthetic PET
and real PET.  This provides a "does it look like a real lesion to a segmenter
trained on real PET?" signal that backpropagates through the diffusion model.

The segmenter uses Stage 2 (relaxed distance-transform output, not hard mask)
to provide dense, continuous gradients even for pixels far from lesion boundaries.

Active only at late steps (tau < active_tau_max): the segmenter is useless on
coarse intermediate diffusion states.

Design constraint (v2.0 doc):
  Enabled only when frozen segmenter achieves lesion recall ≥ 0.70 on val set.
  The `enabled` flag in config controls this; the validation check is manual.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces import LossContext, LossTerm
from .base import smooth_tau_gate


class SegmenterConsistencyLoss(LossTerm):
    name = "segmenter_consistency"

    def __init__(
        self,
        segmenter: Optional[nn.Module] = None,
        active_tau_max: float = 0.3,
        enabled: bool = True,
        weight: float = 0.1,
    ):
        super().__init__(enabled=enabled, weight=weight)
        self.segmenter = segmenter
        self.active_tau_max = active_tau_max

    def set_segmenter(self, segmenter: nn.Module):
        self.segmenter = segmenter

    def forward(self, ctx: LossContext) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.enabled or self.segmenter is None:
            return (
                torch.tensor(0.0, device=ctx.target_pet.device),
                {f"{self.name}/enabled": torch.tensor(0.0, device=ctx.target_pet.device)},
            )

        gate = smooth_tau_gate(ctx.tau, max_tau=self.active_tau_max)

        pred_pet = ctx.pred_x0 if ctx.pred_x0 is not None else ctx.model_pred
        target_pet = ctx.target_pet

        # ---- Frozen segmenter forward on both PETs ----
        with torch.no_grad():
            seg_target = self.segmenter(target_pet)      # [B, 1, H, W]

        # Synthetic PET goes through segmenter (gradient passes through here)
        seg_pred = self.segmenter(pred_pet)               # [B, 1, H, W]

        # L1 consistency: relaxed distance-transform outputs should match
        loss = F.l1_loss(seg_pred, seg_target, reduction="none").mean(dim=(1, 2, 3))
        loss = (loss * gate).mean()

        return loss * self.weight, {
            f"{self.name}/loss": loss.detach(),
            f"{self.name}/gate_mean": gate.mean().detach(),
            f"{self.name}/enabled": torch.tensor(1.0, device=pred_pet.device),
        }
