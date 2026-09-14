#!/usr/bin/env python3
"""Pretrain the frozen PET feature encoder for the perceptual-x0 ablation.

This script is the ONLY producer of the ``PETFeatureEncoder`` checkpoint that
arms P2/P3 consume.  It trains the encoder AND a tiny segmentation head jointly
on the binary lesion mask, over the mechanism training/calibration patients
only, then freezes the encoder and writes a sidecar ``<checkpoint>.lineage.json``
that the loss's fail-closed lineage validation requires.

The produced checkpoint contains ``PETFeatureEncoder.state_dict()`` (the jointly
trained weights) plus metadata.  The segmentation head is a training-time
scaffold whose weights are discarded.

Lineage sidecar contract (written next to the checkpoint):
    {
      "schema_version": 1,
      "checkpoint_sha256": "<sha256 of the checkpoint .pt file>",
      "train_patient_split": {"patient_id": "mechanism_train|calibration", ...},
      "eval_metrics": {
        "lesion_recall": <float >= 0.70>,
        "small_lesion_recall": <float >= 0.70>,
        "dataset": "<manifest path>",
        "small_lesion_quartile": <float>,
        "small_lesion_area_threshold": <int>,
        "base_channels": 16
      }
    }

The encoder and seg head are jointly optimised: gradients from the segmentation
loss flow through the decoder into the encoder parameters.  Only AFTER training
completes is the encoder frozen for downstream consumption.  This fixes the
original design error where ``requires_grad=False`` was set before training, so
the checkpoint silently contained a randomly initialised encoder.

Usage (local preflight only — real data is cloud-only):
    python scripts/pretrain_pet_encoder.py --dry-run --output checkpoints/pet_feature_encoder_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.split_manifest import SplitManifest  # noqa: E402
from src.model.loss_terms.perceptual_x0 import (  # noqa: E402
    MIN_LESION_RECALL,
    MIN_SMALL_LESION_RECALL,
    PETFeatureEncoder,
)

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path("checkpoints/pet_feature_encoder_v1")
SMALL_LESION_QUANTILE = 0.25
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-3


class PretrainError(RuntimeError):
    """A fail-closed pretraining contract violation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> SplitManifest:
    if not path.is_file():
        raise PretrainError(f"Split manifest not found: {path}")
    return SplitManifest(path)


def patient_partition(manifest: SplitManifest) -> dict[str, str]:
    """Map patient_id -> mechanism role (mechanism_train/calibration/validation).

    Uses the locked H1 mechanism partition: ``mechanism_train`` + ``calibration``
    are derived deterministically from the manifest's training patients;
    ``validation`` is exactly the held-out ``val`` patients.  The encoder must
    never train or calibrate on validation patients — that would leak the final
    held-out cohort into the pretrained representation.
    """
    from src.mechanism_validation.common import patient_partition as _mechanism_partition

    rows = [dict(row) for row in manifest._by_sample.values()]
    partition = _mechanism_partition(rows)
    if not any(role == "validation" for role in partition.values()):
        raise PretrainError(
            "Manifest has no held-out validation patients; refusing to run "
            "with an ambiguous partition"
        )
    return partition


def load_pet_masks(
    manifest: SplitManifest,
    cache_dir: Path,
) -> tuple[list[dict[str, Any]], int]:
    """Collect (npz path, mask_area) samples for encoder training.

    Returns (samples, missing_count).  Missing cache entries are counted rather
    than fatal so a preflight can report data-completeness honestly.
    """
    samples: list[dict[str, Any]] = []
    missing = 0
    cache_dir = cache_dir.resolve()
    for sid, row in manifest._by_sample.items():
        split = row["split"]
        if split == "test":
            continue
        cache_path = row.get("cache_path", "")
        npz = (cache_dir / Path(cache_path)).resolve()
        if not npz.is_file():
            npz = (cache_dir / Path(cache_path).name).resolve()
            if not npz.is_file():
                missing += 1
                continue
        samples.append(
            {
                "sample_id": sid,
                "patient_id": row["patient_id"],
                "split": split,
                "npz": npz,
            }
        )
    if not samples:
        raise PretrainError(f"No cache entries found under {cache_dir}")
    return samples, missing


def _as_single_image_bchw(array: Any, *, name: str, source: Path) -> torch.Tensor:
    """Normalise one cached image to ``[1, 1, H, W]`` or fail closed.

    PNG caches store single-channel arrays as ``[1, H, W]`` while small test
    fixtures may use ``[H, W]`` and callers may already provide BCHW.  The
    pretrainer consumes one sample at a time, so multi-channel or batched
    arrays are ambiguous and must not be silently reinterpreted.
    """
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        if tensor.shape[0] != 1:
            raise PretrainError(
                f"{name} in {source} must be single-channel [1,H,W]; "
                f"got {tuple(tensor.shape)}"
            )
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 4:
        if tensor.shape[0] != 1 or tensor.shape[1] != 1:
            raise PretrainError(
                f"{name} in {source} must be one BCHW sample [1,1,H,W]; "
                f"got {tuple(tensor.shape)}"
            )
    else:
        raise PretrainError(
            f"{name} in {source} must have rank 2, 3, or 4; "
            f"got rank {tensor.ndim} with shape {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise PretrainError(f"{name} in {source} contains NaN or Inf")
    return tensor


def _load_sample(sample: Mapping[str, Any], image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(pet, mask)`` as finite single-sample BCHW tensors."""
    source = Path(sample["npz"])
    with np.load(source, allow_pickle=False) as data:
        missing = {key for key in ("pet", "mask") if key not in data}
        if missing:
            raise PretrainError(
                f"Cache entry {source} is missing required arrays: {sorted(missing)}"
            )
        pet = _as_single_image_bchw(data["pet"], name="pet", source=source)
        mask = _as_single_image_bchw(data["mask"], name="mask", source=source)

    target_size = (image_size, image_size)
    if pet.shape[-2:] != target_size:
        pet = F.interpolate(
            pet, size=target_size, mode="bilinear", align_corners=False
        )
    if mask.shape[-2:] != target_size:
        mask = F.interpolate(mask, size=target_size, mode="nearest")
    return pet, mask


class _SegHead(nn.Module):
    """Tiny segmentation scaffold on encoder features; discarded after training.

    Gradients flow through the decoder into the encoder parameters so they are
    jointly trained.  Callers that need a detached feature extraction (e.g. the
    calibration recall gate) must wrap the call in ``torch.no_grad()``.
    """

    def __init__(self, encoder: PETFeatureEncoder, base_channels: int = 16):
        super().__init__()
        self.encoder = encoder
        # Decoder from the deepest feature scale back to full resolution.
        self.up = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.out = nn.Conv2d(base_channels * 2, 1, 3, padding=1)

    def forward(self, pet: torch.Tensor) -> torch.Tensor:
        features = self.encoder.forward_features(pet)
        quarter = features["quarter"]          # [B, 64, H/4, W/4]
        half = features["half"]                # [B, 32, H/2, W/2]
        up = F.silu(self.up(quarter))          # [B, 32, H/2, W/2]
        fused = up + half
        logits = self.out(F.silu(fused))       # [B, 1, H/2, W/2]
        return F.interpolate(logits, scale_factor=2, mode="bilinear", align_corners=False)


def _recall_gate_metrics(
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    small_quartile: float,
) -> dict[str, Any]:
    """Compute overall and small-lesion-quartile recall from predicted masks.

    ``pairs`` is a list of (mask, logit) where mask is the binary ground truth
    and logit is the model's unthresholded prediction.  A sample is a true
    positive when the predicted lesion overlaps the ground truth
    (intersection-over-union > 0) at a fixed 0.5 threshold.
    """
    areas = sorted(int((mask > 0.5).sum().item()) for mask, _ in pairs)
    threshold = areas[int(len(areas) * small_quartile)] if areas else 0
    detections: list[bool] = []
    small: list[bool] = []
    for mask, logit in pairs:
        pred = (torch.sigmoid(logit) > 0.5)
        mask_b = (mask > 0.5)
        inter = (pred & mask_b).sum().item()
        union = (pred | mask_b).sum().item()
        hit = union > 0 and (inter / max(union, 1)) > 0.0
        detections.append(bool(hit))
        small.append(int((mask > 0.5).sum().item()) <= threshold)
    recall = sum(detections) / len(detections) if detections else 0.0
    small_hits = sum(1 for d, s in zip(detections, small) if d and s)
    small_recall = small_hits / sum(small) if any(small) else 0.0
    return {
        "lesion_recall": float(recall),
        "small_lesion_recall": float(small_recall),
        "small_lesion_quartile": float(small_quartile),
        "small_lesion_area_threshold": float(threshold),
        "samples": len(pairs),
        "small_samples": int(sum(small)),
    }


def run_training(
    *,
    manifest_path: Path,
    cache_dir: Path,
    output_dir: Path,
    base_channels: int,
    image_size: int,
    epochs: int,
    learning_rate: float,
    small_quantile: float,
    seed: int,
) -> dict[str, Any]:
    """Train the encoder+seg scaffold on train/calibration, write checkpoint."""
    torch.manual_seed(seed)
    manifest = load_manifest(manifest_path)
    partition = patient_partition(manifest)
    samples, missing = load_pet_masks(manifest, cache_dir)
    # Held-out validation patients must never feed the encoder.  They are
    # present in the mechanism partition but absent from the two encoder-facing
    # roles, so their samples are dropped here.
    encoder_roles = {
        patient: ("train" if role == "mechanism_train" else role)
        for patient, role in partition.items()
        if role != "validation"
    }
    samples = [s for s in samples if s["patient_id"] in encoder_roles]
    if missing:
        raise PretrainError(
            f"Cache incomplete: {missing} of {len(manifest._by_sample)} "
            "manifest samples missing from cache; refusing a partial run"
        )

    train_samples = [s for s in samples if encoder_roles[s["patient_id"]] == "train"]
    cal_samples = [s for s in samples if encoder_roles[s["patient_id"]] == "calibration"]
    if not train_samples:
        raise PretrainError("No training patients for the encoder")

    encoder = PETFeatureEncoder(base_channels=base_channels)
    head = _SegHead(encoder, base_channels=base_channels)
    # Joint training: the encoder MUST be trainable here.  Freezing it before
    # training (the original bug) left a randomly initialised encoder in the
    # checkpoint.  It is frozen only after training completes, below.
    for param in encoder.parameters():
        param.requires_grad = True
    encoder.train()
    # head subsumes the encoder (self.encoder is a submodule), so deduplicate
    # by id to avoid giving the optimizer two groups over the same params.
    seen: set[int] = set()
    trainable = []
    for module in (encoder, head):
        for param in module.parameters():
            if param.requires_grad and id(param) not in seen:
                seen.add(id(param))
                trainable.append(param)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=0.01,
    )

    for epoch in range(epochs):
        order = torch.randperm(len(train_samples))
        total = 0.0
        for idx in order:
            pet, mask = _load_sample(train_samples[int(idx)], image_size)
            logits = head(pet)
            loss = F.binary_cross_entropy_with_logits(logits, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[pretrain] epoch {epoch + 1}/{epochs} loss={total / len(order):.4f}")

    # Freeze the encoder only AFTER joint training completes.  Downstream loss
    # terms (and the lineage validation) rely on these weights being frozen,
    # but the pretrained representation must be the trained one, not random.
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    # ---- Calibration recall gate (measures the exact frozen encoder shipped) ----
    cal_pairs = []
    for s in cal_samples:
        pet, mask = _load_sample(s, image_size)
        logit = head(pet).detach()
        cal_pairs.append((mask, logit))
    metrics = _recall_gate_metrics(
        cal_pairs,
        small_quartile=small_quantile,
    )

    from src.mechanism_validation.common import partition_sha256 as _partition_sha
    partition_sha = _partition_sha(partition)

    checkpoint_path = output_dir / "encoder_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": encoder.state_dict(),
        "base_channels": base_channels,
        "epochs": epochs,
        "seed": seed,
        "partition_sha256": partition_sha,
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)

    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha,
        "train_patient_split": encoder_roles,
        "partition_sha256": partition_sha,
        "eval_metrics": {
            "lesion_recall": metrics["lesion_recall"],
            "small_lesion_recall": metrics["small_lesion_recall"],
            "dataset": manifest_path.as_posix(),
            "small_lesion_quartile": small_quantile,
            "small_lesion_area_threshold": metrics["small_lesion_area_threshold"],
            "base_channels": base_channels,
        },
    }
    sidecar_path = checkpoint_path.with_name(checkpoint_path.name + ".lineage.json")
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    passed = (
        metrics["lesion_recall"] >= MIN_LESION_RECALL
        and metrics["small_lesion_recall"] >= MIN_SMALL_LESION_RECALL
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "train",
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": checkpoint_sha,
        "lineage": sidecar_path.as_posix(),
        "eval_metrics": metrics,
        "gates": {
            "lesion_recall_min": MIN_LESION_RECALL,
            "small_lesion_recall_min": MIN_SMALL_LESION_RECALL,
            "pass": passed,
        },
        "train_patients": len(train_samples),
        "calibration_patients": len(cal_samples),
        "status": "COMPLETE" if passed else "GATE_FAIL",
    }


def dry_run(
    *,
    manifest_path: Path,
    cache_dir: Path,
    output_dir: Path,
    base_channels: int,
) -> dict[str, Any]:
    """Preflight audit: resolve data and partition, report BLOCKED honestly."""
    from src.mechanism_validation.common import partition_sha256 as _partition_sha

    manifest = load_manifest(manifest_path)
    partition = patient_partition(manifest)
    encoder_roles = {
        patient: ("train" if role == "mechanism_train" else role)
        for patient, role in partition.items()
        if role != "validation"
    }
    samples, missing = load_pet_masks(manifest, cache_dir)
    excluded_validation = sum(
        1 for s in samples if s["patient_id"] not in encoder_roles
    )
    samples = [s for s in samples if s["patient_id"] in encoder_roles]
    encoder = PETFeatureEncoder(base_channels=base_channels)
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "dry_run",
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "cache_dir": cache_dir.as_posix(),
        "partition_roles": sorted(set(partition.values())),
        "partition_sha256": _partition_sha(partition),
        "train_patients": sum(1 for r in encoder_roles.values() if r == "train"),
        "calibration_patients": sum(1 for r in encoder_roles.values() if r == "calibration"),
        "excluded_validation_samples": excluded_validation,
        "samples_found": len(samples),
        "samples_missing_from_cache": missing,
        "cache_complete": missing == 0,
        "encoder_params": encoder.get_total_params(),
        "recall_gate": {
            "lesion_recall_min": MIN_LESION_RECALL,
            "small_lesion_recall_min": MIN_SMALL_LESION_RECALL,
        },
        "output_dir": output_dir.as_posix(),
        "status": "BLOCKED",
        "blockers": [
            "Dry-run preflight only. Real encoder training is a cloud-owned "
            "step that requires the validated PNG tensor cache and CUDA.",
        ],
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(Path("main_data/split_manifest.csv")).replace("\\", "/"))
    parser.add_argument("--cache-dir", default=str(Path("cache/tensors")).replace("\\", "/"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT).replace("\\", "/"))
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--small-lesion-quantile", type=float, default=SMALL_LESION_QUANTILE)
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifest_path = (ROOT / args.manifest).resolve()
    cache_dir = (ROOT / args.cache_dir).resolve()
    output_dir = (ROOT / args.output).resolve()

    if args.dry_run:
        report = dry_run(
            manifest_path=manifest_path,
            cache_dir=cache_dir,
            output_dir=output_dir,
            base_channels=args.base_channels,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["status"] == "READY" else 1

    if not torch.cuda.is_available():
        raise PretrainError("Real encoder training requires CUDA (cloud).")

    report = run_training(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        output_dir=output_dir,
        base_channels=args.base_channels,
        image_size=args.image_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        small_quantile=args.small_lesion_quantile,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
