"""ROI-SUV Loss -- clinical quantitative accuracy in physical SUV space.

Denormalises predictions and targets from [-1, 1] back to physical SUV
using per-sample scale_meta (pet_suv_max, pet_raw_min/max, suv_ok).
Then constrains SUVmax, SUVmean and TBR (Tumour-to-Background Ratio)
within the lesion mask.

Active only at late steps (tau < active_tau_max).
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn

from ..interfaces import LossContext, LossTerm
from .base import smooth_tau_gate


def _denormalise_pet(pet_norm: torch.Tensor, meta_list: List[dict]) -> torch.Tensor:
    """Reverse [-1,1] normalisation back to physical SUV per sample.

    Normalisation chain (from CacheBuilder):
      suv_ok=True:  pet_norm = clip(suv / pet_suv_max, 0, 1)  -> 2*x - 1  -> [-1,1]
      suv_ok=False: pet_norm = (val - v_min)/(v_max - v_min)  -> 2*x - 1  -> [-1,1]

    Reverse:
      suv_ok=True:  suv   = (pet_norm + 1) / 2 * pet_suv_max
      suv_ok=False: approx = (pet_norm + 1) / 2 * (v_max - v_min) + v_min
    """
    B = pet_norm.shape[0]
    suv = torch.zeros_like(pet_norm)
    for i in range(B):
        m = meta_list[i] if i < len(meta_list) else {}
        # Map [-1, 1] -> [0, 1]
        p01 = (pet_norm[i:i+1] + 1.0) / 2.0
        if m.get("suv_ok", False):
            suv_max = float(m.get("pet_suv_max", 20.0))
            suv[i:i+1] = p01 * suv_max
        else:
            v_min = float(m.get("pet_raw_min", 0.0))
            v_max = float(m.get("pet_raw_max", 1.0))
            suv[i:i+1] = p01 * max(v_max - v_min, 1e-8) + v_min
    return suv


class ROISUVLoss(LossTerm):
    name = "roi_suv"

    def __init__(
        self,
        active_tau_max: float = 0.25,
        enabled: bool = True,
        weight: float = 0.2,
    ):
        super().__init__(enabled=enabled, weight=weight)
        self.active_tau_max = active_tau_max

    def forward(self, ctx: LossContext) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.enabled:
            return (
                torch.tensor(0.0, device=ctx.target_pet.device),
                {f"{self.name}/enabled": torch.tensor(0.0)},
            )

        gate = smooth_tau_gate(ctx.tau, max_tau=self.active_tau_max)

        pred = ctx.pred_x0 if ctx.pred_x0 is not None else ctx.model_pred
        target = ctx.target_pet

        lesion_mask = ctx.batch.get("mask")
        if lesion_mask is None:
            return (
                torch.tensor(0.0, device=ctx.target_pet.device),
                {f"{self.name}/loss": torch.tensor(0.0)},
            )

        # ---- Denormalise to physical SUV space ----
        meta_list = _extract_meta_list(ctx)
        pred_suv = _denormalise_pet(pred, meta_list)
        target_suv = _denormalise_pet(target, meta_list)

        # ---- SUVmax error within lesion mask ----
        pred_masked = pred_suv * lesion_mask
        target_masked = target_suv * lesion_mask

        pred_max = pred_masked.amax(dim=(2, 3))
        target_max = target_masked.amax(dim=(2, 3))
        suv_max_loss = (pred_max - target_max).abs().mean()

        # ---- SUVmean within lesion mask ----
        mask_sum = lesion_mask.sum(dim=(2, 3)).clamp_min(1)
        pred_mean = pred_masked.sum(dim=(2, 3)) / mask_sum
        target_mean = target_masked.sum(dim=(2, 3)) / mask_sum
        suv_mean_loss = (pred_mean - target_mean).abs().mean()

        # ---- TBR: Tumour-to-Background Ratio ----
        # Background = non-lesion, non-organ, non-bladder (physiological uptake)
        organ_mask = ctx.batch.get("organ_mask")
        bg_mask = torch.ones_like(lesion_mask)
        bg_mask = bg_mask * (1.0 - lesion_mask)
        if organ_mask is not None:
            organ_any = (organ_mask.sum(dim=1, keepdim=True) > 0).float()
            bg_mask = bg_mask * (1.0 - organ_any)

        bg_sum = bg_mask.sum(dim=(2, 3)).clamp_min(1)
        pred_bg_mean = (pred_suv * bg_mask).sum(dim=(2, 3)) / bg_sum
        target_bg_mean = (target_suv * bg_mask).sum(dim=(2, 3)) / bg_sum

        pred_tbr = pred_mean / pred_bg_mean.clamp_min(1e-6)
        target_tbr = target_mean / target_bg_mean.clamp_min(1e-6)
        tbr_loss = (pred_tbr - target_tbr).abs().mean()

        loss = (suv_max_loss + suv_mean_loss + 0.5 * tbr_loss) * gate.mean()

        return loss * self.weight, {
            f"{self.name}/loss": loss.detach(),
            f"{self.name}/suv_max_error": suv_max_loss.detach(),
            f"{self.name}/suv_mean_error": suv_mean_loss.detach(),
            f"{self.name}/tbr_error": tbr_loss.detach(),
            f"{self.name}/pred_suv_mean": pred_mean.mean().detach(),
            f"{self.name}/target_suv_mean": target_mean.mean().detach(),
            f"{self.name}/enabled": torch.tensor(1.0),
        }


def _de_collate_meta(meta: Any, batch_size: int) -> List[dict]:
    """Normalise meta from DataLoader collation into per-sample dicts.

    DataLoader's default collate_fn turns a list of dicts into a dict of lists.
    We need to reverse that to get per-sample dicts back.
    """
    if meta is None:
        return [{}] * batch_size

    if isinstance(meta, list):
        # Already per-sample list
        if len(meta) == batch_size:
            return meta
        # Pad/truncate
        result = list(meta)
        while len(result) < batch_size:
            result.append({})
        return result[:batch_size]

    if isinstance(meta, dict):
        # Check if collated (values are lists) or single-sample (values are scalars)
        first_val = next(iter(meta.values()), None)
        if first_val is None:
            return [{}] * batch_size

        # A collated dict has list/tensor values with len == batch_size
        is_collated = False
        if isinstance(first_val, (list, tuple)):
            is_collated = len(first_val) == batch_size
        elif hasattr(first_val, '__len__') and not isinstance(first_val, str):
            # Tensor or ndarray — check first dim
            try:
                is_collated = len(first_val) == batch_size
            except TypeError:
                pass

        if is_collated:
            # Split collated dict back into per-sample dicts
            samples: List[dict] = []
            for i in range(batch_size):
                sample_meta = {}
                for k, v in meta.items():
                    if isinstance(v, (list, tuple)):
                        sample_meta[k] = v[i] if i < len(v) else None
                    elif hasattr(v, '__getitem__') and not isinstance(v, str):
                        try:
                            sample_meta[k] = v[i]
                        except (IndexError, TypeError):
                            sample_meta[k] = None
                    else:
                        sample_meta[k] = v
                samples.append(sample_meta)
            return samples
        else:
            # Single-sample dict — broadcast to all batch items
            return [dict(meta)] * batch_size

    return [{}] * batch_size


def _extract_meta_list(ctx: LossContext) -> List[dict]:
    """Extract per-sample meta dicts from batch, handling collation."""
    return _de_collate_meta(ctx.batch.get("meta"), ctx.target_pet.shape[0])
