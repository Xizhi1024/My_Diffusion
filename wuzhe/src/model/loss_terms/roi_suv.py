"""ROI-SUV Loss -- clinical quantitative accuracy in physical SUV space.

Denormalises predictions and targets from [-1, 1] back to physical SUV
using per-sample scale_meta (pet_suv_max, suv_ok). Samples without a
validated SUV conversion are skipped instead of being approximated from
raw PET intensity. When CachedDataset provides ``pet_suv``, the target
uses that unclipped physical SUV tensor directly.

Active only at late steps (tau < active_tau_max).
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn

from ..interfaces import LossContext, LossTerm
from .base import smooth_tau_gate


def _as_bool(value: Any) -> bool:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return False
        return bool(value.detach().cpu().reshape(-1)[0].item())
    return bool(value)


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return float(value.detach().cpu().reshape(-1)[0].item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _suv_valid_mask(meta_list: List[dict], batch_size: int, device: torch.device) -> torch.Tensor:
    valid = []
    for i in range(batch_size):
        m = meta_list[i] if i < len(meta_list) else {}
        has_scale = m.get("pet_suv_max") is not None
        valid.append(1.0 if _as_bool(m.get("suv_ok", False)) and has_scale else 0.0)
    return torch.tensor(valid, device=device, dtype=torch.float32)


def _denormalise_pet(pet_norm: torch.Tensor, meta_list: List[dict]) -> torch.Tensor:
    """Reverse [-1,1] normalisation back to physical SUV per valid sample.

    Normalisation chain (from CacheBuilder):
      suv_ok=True:  pet_norm = clip(suv / pet_suv_max, 0, 1)  -> 2*x - 1  -> [-1,1]

    Reverse:
      suv_ok=True:  suv   = (pet_norm + 1) / 2 * pet_suv_max
      suv_ok=False: zero-filled and masked out by ROISUVLoss
    """
    B = pet_norm.shape[0]
    suv = torch.zeros_like(pet_norm)
    for i in range(B):
        m = meta_list[i] if i < len(meta_list) else {}
        # Map [-1, 1] -> [0, 1]
        p01 = (pet_norm[i:i+1] + 1.0) / 2.0
        if _as_bool(m.get("suv_ok", False)):
            suv_max = _as_float(m.get("pet_suv_max"), 20.0)
            suv[i:i+1] = p01 * suv_max
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
        device = ctx.target_pet.device
        zero = torch.tensor(0.0, device=device)
        if not self.enabled:
            return (
                zero,
                {f"{self.name}/enabled": zero},
            )

        pred = ctx.pred_x0 if ctx.pred_x0 is not None else ctx.model_pred
        target = ctx.target_pet
        B = target.shape[0]

        gate = smooth_tau_gate(ctx.tau.to(device=device, dtype=torch.float32), max_tau=self.active_tau_max).reshape(-1)
        if gate.numel() == 1 and B > 1:
            gate = gate.expand(B)
        elif gate.numel() != B:
            gate = gate[:1].expand(B)

        lesion_mask = ctx.batch.get("mask")
        if lesion_mask is None:
            return (
                zero,
                {f"{self.name}/loss": zero, f"{self.name}/valid_fraction": zero},
            )
        lesion_mask = lesion_mask.to(device=device, dtype=pred.dtype)

        # ---- Denormalise to physical SUV space ----
        meta_list = _extract_meta_list(ctx)
        suv_valid = _suv_valid_mask(meta_list, B, device)
        valid_count = suv_valid.sum()
        if valid_count.item() == 0:
            return zero, {
                f"{self.name}/loss": zero,
                f"{self.name}/suv_max_error": zero,
                f"{self.name}/suv_mean_error": zero,
                f"{self.name}/tbr_error": zero,
                f"{self.name}/pred_suv_mean": zero,
                f"{self.name}/target_suv_mean": zero,
                f"{self.name}/valid_fraction": zero,
                f"{self.name}/enabled": torch.tensor(1.0, device=device),
            }

        pred_suv = _denormalise_pet(pred, meta_list)
        target_suv_tensor = ctx.batch.get("pet_suv")
        if torch.is_tensor(target_suv_tensor) and target_suv_tensor.shape == target.shape:
            target_suv = target_suv_tensor.to(device=device, dtype=target.dtype)
        else:
            target_suv = _denormalise_pet(target, meta_list)

        # ---- SUVmax error within lesion mask ----
        pred_masked = pred_suv * lesion_mask
        target_masked = target_suv * lesion_mask

        pred_max = pred_masked.amax(dim=(2, 3))
        target_max = target_masked.amax(dim=(2, 3))
        suv_max_per = (pred_max - target_max).abs().flatten(1).mean(dim=1)

        # ---- SUVmean within lesion mask ----
        mask_sum = lesion_mask.sum(dim=(2, 3)).clamp_min(1)
        pred_mean = pred_masked.sum(dim=(2, 3)) / mask_sum
        target_mean = target_masked.sum(dim=(2, 3)) / mask_sum
        suv_mean_per = (pred_mean - target_mean).abs().flatten(1).mean(dim=1)

        # ---- TBR: Tumour-to-Background Ratio ----
        # Background = non-lesion, non-organ, non-bladder (physiological uptake)
        organ_mask = ctx.batch.get("organ_mask")
        bg_mask = torch.ones_like(lesion_mask)
        bg_mask = bg_mask * (1.0 - lesion_mask)
        if organ_mask is not None:
            organ_mask = organ_mask.to(device=device, dtype=pred.dtype)
            organ_any = (organ_mask.sum(dim=1, keepdim=True) > 0).float()
            bg_mask = bg_mask * (1.0 - organ_any)

        bg_sum = bg_mask.sum(dim=(2, 3)).clamp_min(1)
        pred_bg_mean = (pred_suv * bg_mask).sum(dim=(2, 3)) / bg_sum
        target_bg_mean = (target_suv * bg_mask).sum(dim=(2, 3)) / bg_sum

        pred_tbr = pred_mean / pred_bg_mean.clamp_min(1e-6)
        target_tbr = target_mean / target_bg_mean.clamp_min(1e-6)
        tbr_per = (pred_tbr - target_tbr).abs().flatten(1).mean(dim=1)

        per_sample_loss = suv_max_per + suv_mean_per + 0.5 * tbr_per
        loss = (per_sample_loss * suv_valid * gate).sum() / valid_count.clamp_min(1.0)

        suv_max_loss = (suv_max_per * suv_valid).sum() / valid_count.clamp_min(1.0)
        suv_mean_loss = (suv_mean_per * suv_valid).sum() / valid_count.clamp_min(1.0)
        tbr_loss = (tbr_per * suv_valid).sum() / valid_count.clamp_min(1.0)
        pred_suv_mean = (pred_mean.flatten(1).mean(dim=1) * suv_valid).sum() / valid_count.clamp_min(1.0)
        target_suv_mean = (target_mean.flatten(1).mean(dim=1) * suv_valid).sum() / valid_count.clamp_min(1.0)

        return loss * self.weight, {
            f"{self.name}/loss": loss.detach(),
            f"{self.name}/suv_max_error": suv_max_loss.detach(),
            f"{self.name}/suv_mean_error": suv_mean_loss.detach(),
            f"{self.name}/tbr_error": tbr_loss.detach(),
            f"{self.name}/pred_suv_mean": pred_suv_mean.detach(),
            f"{self.name}/target_suv_mean": target_suv_mean.detach(),
            f"{self.name}/valid_fraction": suv_valid.mean().detach(),
            f"{self.name}/enabled": torch.tensor(1.0, device=device),
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
