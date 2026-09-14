#!/usr/bin/env python3
"""Pretrain the frozen Stage-2 TinySegmenter consumed by P1_SEG_OUT.

The model is trained only on mechanism-training patients, calibrated on the
locked calibration patients, and never sees held-out validation patients.
The canonical ``segmenter.pt`` is written only when both the overall and
small-lesion recall gates pass.  Its state dict is saved directly because
``SLMFBBDM`` loads the checkpoint with ``TinySegmenter.load_state_dict``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pretrain_pet_encoder import (  # noqa: E402
    PretrainError,
    _load_sample,
    load_manifest,
    load_pet_masks,
    patient_partition,
)
from src.mechanism_validation.common import partition_sha256  # noqa: E402
from src.model.segmenter import TinySegmenter, segmenter_loss  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path("checkpoints/tiny_segmenter_v1")
DEFAULT_STAGE1_EPOCHS = 5
DEFAULT_STAGE2_EPOCHS = 20
DEFAULT_RECALL_MIN = 0.70


class SegmenterPretrainError(PretrainError):
    """A fail-closed TinySegmenter producer contract violation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encoder_roles_from_partition(partition: Mapping[str, str]) -> dict[str, str]:
    """Return the two patient roles allowed to influence the segmenter."""
    return {
        patient: ("train" if role == "mechanism_train" else role)
        for patient, role in partition.items()
        if role != "validation"
    }


def prepare_records(
    samples: list[Mapping[str, Any]],
    *,
    image_size: int,
    include_stage2_target: bool,
) -> list[dict[str, Any]]:
    """Load each NPZ once; optionally precompute the expensive EDT target."""
    records: list[dict[str, Any]] = []
    for sample in samples:
        pet, mask = _load_sample(sample, image_size)
        record: dict[str, Any] = {
            "patient_id": str(sample["patient_id"]),
            "pet": pet.contiguous(),
            "mask": mask.contiguous(),
        }
        if include_stage2_target:
            record["stage2_target"] = TinySegmenter.build_target(
                mask, stage=2
            ).contiguous()
        records.append(record)
    return records


def _batches(order: torch.Tensor, batch_size: int) -> Iterable[list[int]]:
    for start in range(0, len(order), batch_size):
        yield [int(index) for index in order[start : start + batch_size]]


def train_stage(
    model: TinySegmenter,
    optimizer: torch.optim.Optimizer,
    records: list[Mapping[str, Any]],
    *,
    stage: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> None:
    """Train one declared stage and print every epoch for cloud monitoring."""
    model.stage = stage
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(records))
        total = 0.0
        steps = 0
        for indices in _batches(order, batch_size):
            pet = torch.cat([records[index]["pet"] for index in indices]).to(device)
            mask = torch.cat([records[index]["mask"] for index in indices]).to(device)
            if stage == 1:
                target = TinySegmenter.build_target(
                    mask, stage=1, gaussian_sigma=model.gaussian_sigma
                )
            else:
                target = torch.cat(
                    [records[index]["stage2_target"] for index in indices]
                ).to(device)

            prediction = model(pet)
            loss, _ = segmenter_loss(
                prediction,
                target,
                mask=mask,
                stage=stage,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        print(
            f"[segmenter] stage={stage} epoch={epoch + 1}/{epochs} "
            f"loss={total / max(steps, 1):.6f}",
            flush=True,
        )


def recall_metrics(
    model: TinySegmenter,
    records: list[Mapping[str, Any]],
    *,
    device: torch.device,
    small_quantile: float,
) -> dict[str, Any]:
    """Evaluate pixel recall plus non-degeneracy diagnostics on calibration."""
    model.stage = 2
    model.eval()
    rows: list[dict[str, float | bool]] = []
    with torch.no_grad():
        for record in records:
            mask = record["mask"].to(device) > 0.5
            prediction = model(record["pet"].to(device)) > 0.0
            true_positive = int((prediction & mask).sum().item())
            predicted_positive = int(prediction.sum().item())
            lesion_pixels = int(mask.sum().item())
            rows.append(
                {
                    "area": float(lesion_pixels),
                    "tp": float(true_positive),
                    "predicted": float(predicted_positive),
                    "detected": bool(true_positive > 0),
                }
            )

    positive_rows = [row for row in rows if row["area"] > 0]
    if not positive_rows:
        raise SegmenterPretrainError("Calibration partition has no positive lesion masks")
    areas = sorted(float(row["area"]) for row in positive_rows)
    threshold = float(np.quantile(areas, small_quantile))
    small_rows = [row for row in positive_rows if float(row["area"]) <= threshold]

    def aggregate(selected: list[dict[str, float | bool]]) -> dict[str, float]:
        tp = sum(float(row["tp"]) for row in selected)
        target = sum(float(row["area"]) for row in selected)
        predicted = sum(float(row["predicted"]) for row in selected)
        recall = tp / target if target else 0.0
        precision = tp / predicted if predicted else 0.0
        dice = 2.0 * tp / (target + predicted) if target + predicted else 0.0
        detection = sum(bool(row["detected"]) for row in selected) / len(selected)
        return {
            "recall": recall,
            "precision": precision,
            "dice": dice,
            "detection_recall": detection,
        }

    overall = aggregate(positive_rows)
    small = aggregate(small_rows)
    return {
        "lesion_recall": overall["recall"],
        "small_lesion_recall": small["recall"],
        "lesion_precision": overall["precision"],
        "lesion_dice": overall["dice"],
        "slice_detection_recall": overall["detection_recall"],
        "small_slice_detection_recall": small["detection_recall"],
        "small_lesion_quartile": small_quantile,
        "small_lesion_area_threshold": threshold,
        "samples": len(positive_rows),
        "small_samples": len(small_rows),
    }


def run_training(
    *,
    manifest_path: Path,
    cache_dir: Path,
    output_dir: Path,
    image_size: int,
    base_channels: int,
    batch_size: int,
    stage1_epochs: int,
    stage2_epochs: int,
    learning_rate: float,
    seed: int,
    small_quantile: float,
    lesion_recall_min: float,
    small_lesion_recall_min: float,
) -> dict[str, Any]:
    checkpoint_path = output_dir / "segmenter.pt"
    if checkpoint_path.exists():
        raise SegmenterPretrainError(
            f"Refusing to overwrite existing canonical checkpoint: {checkpoint_path}"
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    manifest = load_manifest(manifest_path)
    partition = patient_partition(manifest)
    roles = encoder_roles_from_partition(partition)
    samples, missing = load_pet_masks(manifest, cache_dir)
    if missing:
        raise SegmenterPretrainError(
            f"Cache incomplete: {missing} manifest samples are missing"
        )
    samples = [sample for sample in samples if sample["patient_id"] in roles]
    train_samples = [sample for sample in samples if roles[sample["patient_id"]] == "train"]
    calibration_samples = [
        sample for sample in samples if roles[sample["patient_id"]] == "calibration"
    ]
    if not train_samples or not calibration_samples:
        raise SegmenterPretrainError(
            "Both mechanism-training and calibration samples are required"
        )

    print("[segmenter] loading training cache and precomputing Stage-2 targets", flush=True)
    train_records = prepare_records(
        train_samples, image_size=image_size, include_stage2_target=True
    )
    calibration_records = prepare_records(
        calibration_samples, image_size=image_size, include_stage2_target=False
    )

    device = torch.device("cuda")
    model = TinySegmenter(
        in_channels=1,
        base_channels=base_channels,
        stage=1,
        gaussian_sigma=6.0,
        enabled=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=0.01
    )
    train_stage(
        model,
        optimizer,
        train_records,
        stage=1,
        epochs=stage1_epochs,
        batch_size=batch_size,
        device=device,
    )
    train_stage(
        model,
        optimizer,
        train_records,
        stage=2,
        epochs=stage2_epochs,
        batch_size=batch_size,
        device=device,
    )

    metrics = recall_metrics(
        model,
        calibration_records,
        device=device,
        small_quantile=small_quantile,
    )
    passed = (
        metrics["lesion_recall"] >= lesion_recall_min
        and metrics["small_lesion_recall"] >= small_lesion_recall_min
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "mode": "train",
        "eval_metrics": metrics,
        "gates": {
            "lesion_recall_min": lesion_recall_min,
            "small_lesion_recall_min": small_lesion_recall_min,
            "pass": passed,
        },
        "train_samples": len(train_samples),
        "calibration_samples": len(calibration_samples),
        "train_patients": len({sample["patient_id"] for sample in train_samples}),
        "calibration_patients": len(
            {sample["patient_id"] for sample in calibration_samples}
        ),
        "status": "COMPLETE" if passed else "GATE_FAIL",
    }
    if not passed:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return report

    output_dir.mkdir(parents=True, exist_ok=True)
    model.cpu().stage = 2
    torch.save(model.state_dict(), checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_format": "TinySegmenter.state_dict",
        "architecture": {
            "in_channels": 1,
            "base_channels": base_channels,
            "stage": 2,
            "gaussian_sigma": 6.0,
        },
        "train_patient_split": roles,
        "partition_sha256": partition_sha256(partition),
        "eval_metrics": metrics,
        "gates": report["gates"],
        "dataset": manifest_path.as_posix(),
        "cache_dir": cache_dir.as_posix(),
        "seed": seed,
        "stage1_epochs": stage1_epochs,
        "stage2_epochs": stage2_epochs,
    }
    sidecar_path = checkpoint_path.with_name(checkpoint_path.name + ".lineage.json")
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report.update(
        {
            "checkpoint": checkpoint_path.as_posix(),
            "checkpoint_sha256": checkpoint_sha,
            "lineage": sidecar_path.as_posix(),
        }
    )
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="main_data/split_manifest.csv")
    parser.add_argument("--cache-dir", default="cache/tensors")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT).replace("\\", "/"))
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stage1-epochs", type=int, default=DEFAULT_STAGE1_EPOCHS)
    parser.add_argument("--stage2-epochs", type=int, default=DEFAULT_STAGE2_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--small-lesion-quantile", type=float, default=0.25)
    parser.add_argument("--lesion-recall-min", type=float, default=DEFAULT_RECALL_MIN)
    parser.add_argument(
        "--small-lesion-recall-min", type=float, default=DEFAULT_RECALL_MIN
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not torch.cuda.is_available():
        raise SegmenterPretrainError("Real TinySegmenter training requires CUDA")
    if args.stage1_epochs < 1 or args.stage2_epochs < 1:
        raise SegmenterPretrainError("Both training stages require at least one epoch")
    if args.batch_size < 1:
        raise SegmenterPretrainError("batch_size must be positive")

    report = run_training(
        manifest_path=(ROOT / args.manifest).resolve(),
        cache_dir=(ROOT / args.cache_dir).resolve(),
        output_dir=(ROOT / args.output).resolve(),
        image_size=args.image_size,
        base_channels=args.base_channels,
        batch_size=args.batch_size,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        small_quantile=args.small_lesion_quantile,
        lesion_recall_min=args.lesion_recall_min,
        small_lesion_recall_min=args.small_lesion_recall_min,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
