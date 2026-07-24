"""Standalone training utilities for the low-frequency PET mean predictor."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.lineage import attach_data_lineage, load_checkpoint_data_lineage
from src.mechanism_validation.common import canonical_json_sha256

from .frequency.haar import haar_dwt2
from .mean_predictor import LowFrequencyPETPredictor


# Lineage fields that a strict (pathology-excluded) production mean must carry
# beside its policy so the H2 exclusion can be audited without loading weights.
_CHECKPOINT_LINEAGE_FINGERPRINT_FIELDS = (
    "manifest_semantic_sha256",
    "raw_png_combined_sha256",
    "preprocessing_config_sha256",
    "dataset_contract_sha256",
    "cache_payload_sha256",
    "cache_metadata_sha256",
)


def write_checkpoint_fingerprint(
    checkpoint_path: Path,
    checkpoint: Dict[str, Any],
    data_lineage: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Write a tamper-evident sidecar summarising a saved mean checkpoint.

    The ``.pt`` itself intentionally keeps the versioned key set expected by
    existing loaders (``format_version/model/mean_config/epoch/val_loss`` plus
    optional ``data_lineage``).  This sidecar adds the production-audit fields
    that cannot live inside the checkpoint (notably its own SHA-256) and
    re-surfaces the pathology-exclusion policy + cache fingerprints so a reviewer
    can confirm an excluded mean without executing any code.
    """

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    policy = dict(checkpoint.get("mean_config", {}).get("pathology_exclusion", {}) or {})
    lineage = dict(data_lineage or {})
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "stage": "excluded_mean_checkpoint_fingerprint",
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_format_version": checkpoint.get("format_version"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_loss": checkpoint.get("val_loss"),
        "checkpoint_sha256": checkpoint_sha256,
        "pathology_exclusion": {
            "enabled": bool(policy.get("enabled", False)),
            "guard_radius_px": int(policy.get("guard_radius_px", 0)),
        },
        "lineage_present": lineage is not None and bool(lineage),
        "lineage_fingerprints": {
            field: lineage.get(field) for field in _CHECKPOINT_LINEAGE_FINGERPRINT_FIELDS
        },
    }
    payload["fingerprint_sha256"] = canonical_json_sha256(payload)
    sidecar = checkpoint_path.with_name(checkpoint_path.name + ".fingerprint.json")
    sidecar.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def mean_target_ll2(pet: torch.Tensor) -> torch.Tensor:
    """Return the level-2 orthonormal Haar low-pass PET coefficient."""
    ll1, _ = haar_dwt2(pet)
    ll2, _ = haar_dwt2(ll1)
    return ll2


def mean_charbonnier_loss(
    pred_ll2: torch.Tensor,
    target_ll2: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    if epsilon <= 0:
        raise ValueError("Charbonnier epsilon must be positive")
    return torch.sqrt((pred_ll2 - target_ll2).square() + epsilon ** 2).mean()


def pathology_excluded_mean_loss(
    pred_ll2: torch.Tensor,
    target_ll2: torch.Tensor,
    mask: torch.Tensor,
    *,
    epsilon: float,
    guard_radius_px: int,
) -> torch.Tensor:
    """Charbonnier LL2 loss outside dilated pathology support.

    Mirrors the validated H2 exclusion
    (``scripts/validate_h2_pathology_excluded_residual.py``): the pixel-space lesion
    mask is downsampled to LL2 resolution (``LowFrequencyPETPredictor`` is fixed at two
    Haar levels, so LL2 is at 1/4), dilated by ``guard_radius_px``, and the loss is
    averaged only over background LL2 pixels.  Masks never enter the predictor.
    """
    if pred_ll2.shape != target_ll2.shape:
        raise ValueError("pred_ll2 and target_ll2 must have identical shapes")
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must have shape [B,1,H,W]")
    support = F.max_pool2d(mask.float(), kernel_size=4, stride=4)
    support = (support > 0).to(pred_ll2)
    ll2_radius = int(math.ceil(max(guard_radius_px, 0) / 4))
    if ll2_radius > 0:
        kernel = 2 * ll2_radius + 1
        support = F.max_pool2d(
            support, kernel_size=kernel, stride=1, padding=ll2_radius
        )
    valid = (1.0 - support).clamp(0.0, 1.0)
    error = torch.sqrt((pred_ll2 - target_ll2).square() + epsilon ** 2)
    denominator = valid.sum().clamp_min(1.0)
    return (error * valid).sum() / denominator


class MeanPretrainer:
    """Train and select only :class:`LowFrequencyPETPredictor`."""

    def __init__(
        self,
        predictor: LowFrequencyPETPredictor,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        if val_loader is None:
            raise ValueError("Conditional mean pretraining requires a validation loader")
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.data_lineage = load_checkpoint_data_lineage(config)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.predictor = predictor.to(self.device)

        mean_cfg = config.get("modules", {}).get("conditional_mean", {})
        self.mean_config = dict(mean_cfg)
        self.epsilon = float(mean_cfg.get("charbonnier_eps", 1e-3))
        exclusion_cfg = mean_cfg.get("pathology_exclusion", {}) or {}
        self.pathology_exclusion_enabled = bool(exclusion_cfg.get("enabled", False))
        self.guard_radius_px = int(exclusion_cfg.get("guard_radius_px", 8))
        if self.pathology_exclusion_enabled and self.guard_radius_px < 0:
            raise ValueError(
                "modules.conditional_mean.pathology_exclusion.guard_radius_px "
                "cannot be negative"
            )
        training_cfg = config.get("training", {})
        self.learning_rate = float(training_cfg.get("learning_rate", 1e-4))
        self.lr_min = float(training_cfg.get("lr_min", 1e-6))
        weight_decay = float(training_cfg.get("weight_decay", 0.01))
        self.optimizer = torch.optim.AdamW(
            self.predictor.parameters(),
            lr=self.learning_rate,
            weight_decay=weight_decay,
        )
        run_cfg = config.get("runtime", {})
        self.grad_clip_norm = float(run_cfg.get("grad_clip_norm", 1.0))
        self.amp_enabled = bool(run_cfg.get("amp", True)) and self.device.type == "cuda"
        if self.amp_enabled and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.amp_enabled and self.amp_dtype == torch.float16,
        )

    def _batch_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        ct = batch["ct"].to(self.device, non_blocking=True)
        pet = batch["pet"].to(self.device, non_blocking=True)
        with torch.amp.autocast(
            self.device.type,
            enabled=self.amp_enabled,
            dtype=self.amp_dtype,
        ):
            pred_ll2 = self.predictor(ct)["ll2"]
            target_ll2 = mean_target_ll2(pet)
            if not self.pathology_exclusion_enabled:
                return mean_charbonnier_loss(pred_ll2, target_ll2, self.epsilon)
            if "mask" not in batch:
                raise ValueError(
                    "modules.conditional_mean.pathology_exclusion.enabled=true "
                    "requires 'mask' in the batch"
                )
            mask = batch["mask"].to(self.device, non_blocking=True)
            return pathology_excluded_mean_loss(
                pred_ll2,
                target_ll2,
                mask,
                epsilon=self.epsilon,
                guard_radius_px=self.guard_radius_px,
            )

    def train_epoch(self) -> float:
        self.predictor.train()
        total = 0.0
        count = 0
        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._batch_loss(batch)
            self.scaler.scale(loss).backward()
            if self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.predictor.parameters(), self.grad_clip_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += float(loss.detach().item())
            count += 1
        if count == 0:
            raise ValueError("Conditional mean training loader is empty")
        return total / count

    @torch.no_grad()
    def validate(self) -> float:
        self.predictor.eval()
        total = 0.0
        count = 0
        for batch in self.val_loader:
            loss = self._batch_loss(batch)
            total += float(loss.detach().item())
            count += 1
        if count == 0:
            raise ValueError("Conditional mean validation loader is empty")
        value = total / count
        if not math.isfinite(value):
            raise FloatingPointError(f"Non-finite conditional mean validation loss: {value}")
        return value

    def _checkpoint(self, epoch: int, val_loss: float) -> Dict[str, Any]:
        checkpoint = {
            "format_version": 2 if self.data_lineage is not None else 1,
            "model": self.predictor.state_dict(),
            "mean_config": self.mean_config,
            "epoch": int(epoch),
            "val_loss": float(val_loss),
        }
        return attach_data_lineage(checkpoint, self.data_lineage)

    def run(self, epochs: int, output_dir: str | Path) -> list[dict[str, float]]:
        if epochs < 1:
            raise ValueError("Conditional mean pretraining epochs must be positive")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=self.lr_min,
        )
        best_val = float("inf")
        history: list[dict[str, float]] = []
        for epoch_index in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            epoch = epoch_index + 1
            row = {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            }
            history.append(row)
            checkpoint = self._checkpoint(epoch, val_loss)
            torch.save(checkpoint, output_path / "mean_last.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(checkpoint, output_path / "mean_best.pt")
                # Tamper-evident audit sidecar: policy + cache lineage + the
                # checkpoint's own SHA-256 (cannot be embedded in the .pt).
                write_checkpoint_fingerprint(
                    output_path / "mean_best.pt",
                    checkpoint,
                    self.data_lineage,
                )
            scheduler.step()
            with (output_path / "history.json").open("w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2, ensure_ascii=False)
            print(
                f"Mean epoch {epoch:3d}/{epochs} | "
                f"train={train_loss:.6f} | val={val_loss:.6f}"
            )
        return history
