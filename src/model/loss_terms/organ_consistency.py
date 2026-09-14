"""Organ Consistency Loss — per-organ PET value reasonableness.

Penalises high PET values in metabolically-cold organs (bone, fat, muscle)
while allowing physiological uptake in bladder/rectum.  Differs from
FalseHotspotLoss: this is a continuous mean-level constraint across the
entire organ region, not quantile-based outlier detection.

Organ class semantics:
  0: uterus/pelvic_region  → mild constraint (lesion proximity)
  1: bladder               → no constraint (physiological uptake)
  2: rectum/bowel          → no constraint (physiological uptake)
  3: bone                  → strong penalty on high values
  4: fat                   → strong penalty on high values
  5: muscle/other          → moderate penalty on high values

Active at early-to-mid steps (tau > 0.25): organ-level PET distribution
is determined early; fine lesion details come later.
"""

from typing import Dict

import torch

from ..interfaces import LossContext, LossTerm


class OrganConsistencyLoss(LossTerm):
    name = "organ_consistency"

    def __init__(
        self,
        active_tau_min: float = 0.25,   # active for tau > 0.25 (early-mid)
        cold_weight: float = 0.02,       # penalty scale for cold organs
        enabled: bool = True,
        weight: float = 0.1,
    ):
        super().__init__(enabled=enabled, weight=weight)
        self.active_tau_min = active_tau_min
        self.cold_weight = cold_weight

    def forward(self, ctx: LossContext) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.enabled:
            return (
                torch.tensor(0.0, device=ctx.target_pet.device),
                {f"{self.name}/enabled": torch.tensor(0.0, device=ctx.target_pet.device)},
            )

        organ_mask = ctx.batch.get("organ_mask")
        if organ_mask is None:
            return (
                torch.tensor(0.0, device=ctx.target_pet.device),
                {f"{self.name}/loss": torch.tensor(0.0, device=ctx.target_pet.device)},
            )

        # ---- tau gate: active only at early-mid steps ----
        # Use early gate: ~1 when tau > active_tau_min, ~0 when tau < active_tau_min
        tau_f = ctx.tau.float()
        gate = torch.sigmoid(10.0 * (tau_f - self.active_tau_min))
        gate_mean = gate.mean()

        pred = ctx.pred_x0 if ctx.pred_x0 is not None else ctx.model_pred
        B, C, H, W = pred.shape
        device = pred.device

        # ---- Build per-organ masks ----
        num_organs = organ_mask.shape[1]
        loss_terms: Dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=device)

        # Cold organs: penalise positive PET values (only the positive part)
        # PET in [-1,1]; positive → above mean in normalised space → suspicious in cold tissue.
        cold_classes = {
            "bone": 3,
            "fat": 4,
            "muscle": 5,
        }
        cold_weights = {
            "bone": 1.0,
            "fat": 1.0,
            "muscle": 0.5,
        }

        for name, cls in cold_classes.items():
            if cls >= num_organs:
                continue
            mask = organ_mask[:, cls:cls + 1]       # [B, 1, H, W]
            if mask.sum() < 1:
                continue
            # Penalise only positive predictions in cold tissue
            pos_pred = torch.relu(pred)              # clip negative → 0
            region_loss = (pos_pred * mask).sum() / mask.sum().clamp_min(1)
            w = cold_weights[name] * self.cold_weight
            total_loss = total_loss + w * region_loss
            loss_terms[f"{self.name}/{name}_loss"] = region_loss.detach()

        # Uterus (class 0): mild penalty on extreme values (> 0.8 in [-1,1])
        if num_organs > 0:
            uterus = organ_mask[:, 0:1]
            if uterus.sum() > 0:
                extreme = torch.relu(pred - 0.8)
                uterus_loss = (extreme * uterus).sum() / uterus.sum().clamp_min(1)
                total_loss = total_loss + 0.01 * uterus_loss
                loss_terms[f"{self.name}/uterus_loss"] = uterus_loss.detach()

        loss = total_loss * gate_mean

        logs: Dict[str, torch.Tensor] = {
            f"{self.name}/loss": loss.detach(),
            f"{self.name}/gate_mean": gate_mean.detach(),
            f"{self.name}/enabled": torch.tensor(1.0, device=device),
        }
        logs.update(loss_terms)

        return loss * self.weight, logs
