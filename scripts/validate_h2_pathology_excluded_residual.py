"""H2: paired fixed-epoch conditional means and residual-enrichment gate.

Two predictors receive identical CT-only inputs, initialization, patient
partition, batch order, optimizer, and epoch budget.  The only intervention is
the training loss:

* ``included`` uses every LL2 pixel;
* ``excluded`` removes the dilated pathology support from the LL2 loss.

Masks are never accepted by either predictor at inference.  Calibration fixes
the boundary non-inferiority margin; held-out validation is evaluated once with
patient-level bootstrap and sign-flip inference.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import CachedDataset
from src.data.lineage import (
    REQUIRED_CHECKPOINT_LINEAGE_FIELDS,
    attach_data_lineage,
    load_checkpoint_data_lineage,
)
from src.mechanism_validation.common import (
    aggregate_patient_rows,
    decision_guardrails,
    file_sha256,
    load_json,
    paired_patient_bootstrap,
    partition_counts,
    partition_sha256,
    patient_partition,
    read_manifest,
    require_upstream_pass,
    write_csv,
    write_json,
    write_mechanism_manifest,
)
from src.model.frequency.haar import reconstruct_lowpass
from src.model.mean_predictor import LowFrequencyPETPredictor
from src.model.mean_pretraining import mean_target_ll2


SCHEMA_VERSION = 1
ANALYSIS_SEED = 20260724
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 8
DEFAULT_GUARD_RADIUS_PX = 8


class IndexedDataset(Dataset):
    """Stable view over CachedDataset with patient/sample entries exposed."""

    def __init__(self, base: CachedDataset, indices: Sequence[int]) -> None:
        self.base = base
        self.indices = [int(index) for index in indices]
        self.entries = [base.entries[index] for index in self.indices]
        if not self.indices:
            raise ValueError("IndexedDataset cannot be empty")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.base[self.indices[index]]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be an object: {path}")
    return payload


def _make_partition_datasets(
    *,
    cache_dir: Path,
    manifest_path: Path,
    partition: Mapping[str, str],
    required_keys: Sequence[str],
) -> dict[str, IndexedDataset]:
    train_base = CachedDataset(
        cache_dir,
        split="train",
        augment=False,
        split_manifest=manifest_path,
        required_keys=list(required_keys),
    )
    validation_base = CachedDataset(
        cache_dir,
        split="val",
        augment=False,
        split_manifest=manifest_path,
        required_keys=list(required_keys),
    )

    def select(base: CachedDataset, role: str) -> IndexedDataset:
        indices = [
            index
            for index, entry in enumerate(base.entries)
            if partition.get(entry.patient_id) == role
        ]
        return IndexedDataset(base, indices)

    return {
        "mechanism_train": select(train_base, "mechanism_train"),
        "calibration": select(train_base, "calibration"),
        "validation": select(validation_base, "validation"),
    }


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def pathology_excluded_mean_loss(
    pred_ll2: torch.Tensor,
    target_ll2: torch.Tensor,
    mask: torch.Tensor,
    *,
    epsilon: float,
    guard_radius_px: int,
) -> torch.Tensor:
    """Charbonnier LL2 loss outside dilated pathology support."""

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
            support,
            kernel_size=kernel,
            stride=1,
            padding=ll2_radius,
        )
    valid = (1.0 - support).clamp(0.0, 1.0)
    error = torch.sqrt((pred_ll2 - target_ll2).square() + epsilon**2)
    denominator = valid.sum().clamp_min(1.0)
    return (error * valid).sum() / denominator


def pathology_included_mean_loss(
    pred_ll2: torch.Tensor,
    target_ll2: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    return torch.sqrt((pred_ll2 - target_ll2).square() + epsilon**2).mean()


def _train_variant(
    *,
    variant: str,
    initial_state: Mapping[str, torch.Tensor],
    model_kwargs: Mapping[str, Any],
    dataset: IndexedDataset,
    data_lineage: Mapping[str, Any],
    partition_hash: str,
    output_path: Path,
    epochs: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
    lr_min: float,
    weight_decay: float,
    epsilon: float,
    guard_radius_px: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, Any]:
    model = LowFrequencyPETPredictor(**model_kwargs).to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr_min,
    )
    loader = _loader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
        device=device,
    )
    amp_enabled = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if amp_enabled and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16,
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_total = 0.0
        batches = 0
        for batch in loader:
            ct = batch["ct"].to(device, non_blocking=True)
            pet = batch["pet"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device.type,
                enabled=amp_enabled,
                dtype=amp_dtype,
            ):
                pred_ll2 = model(ct)["ll2"]
                target_ll2 = mean_target_ll2(pet)
                if variant == "excluded":
                    loss = pathology_excluded_mean_loss(
                        pred_ll2,
                        target_ll2,
                        mask,
                        epsilon=epsilon,
                        guard_radius_px=guard_radius_px,
                    )
                elif variant == "included":
                    loss = pathology_included_mean_loss(
                        pred_ll2,
                        target_ll2,
                        epsilon=epsilon,
                    )
                else:
                    raise ValueError(f"Unknown H2 mean variant: {variant}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_total += float(loss.detach().item())
            batches += 1
        if batches == 0:
            raise RuntimeError("H2 mechanism-train loader is empty")
        row = {
            "epoch": float(epoch),
            "train_loss": epoch_total / batches,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        scheduler.step()
        print(
            f"H2 {variant:8s} epoch {epoch:3d}/{epochs} "
            f"train={row['train_loss']:.6f}",
            flush=True,
        )

    checkpoint = attach_data_lineage(
        {
            "format_version": 2,
            "model": model.state_dict(),
            "mean_config": {
                **dict(model_kwargs),
                "charbonnier_eps": epsilon,
                "pathology_policy": variant,
                "guard_radius_px": guard_radius_px,
            },
            "epoch": epochs,
            "fixed_epoch_endpoint": True,
            "validation_used_for_selection": False,
            "mechanism_partition_sha256": partition_hash,
            "training_seed": seed,
            "history": history,
        },
        data_lineage,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return {
        "path": output_path.as_posix(),
        "sha256": file_sha256(output_path),
        "epochs": epochs,
        "final_train_loss": history[-1]["train_loss"],
    }


def _validate_reusable_checkpoint(
    path: Path,
    *,
    data_lineage: Mapping[str, Any],
    expected_partition_hash: str,
    expected_variant: str,
    expected_epochs: int,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"H2 checkpoint is not a mapping: {path}")
    embedded = checkpoint.get("data_lineage")
    if not isinstance(embedded, Mapping):
        raise RuntimeError(f"H2 checkpoint lacks data_lineage: {path}")
    mismatches = [
        field
        for field in REQUIRED_CHECKPOINT_LINEAGE_FIELDS
        if embedded.get(field) != data_lineage.get(field)
    ]
    if mismatches:
        raise RuntimeError(
            f"H2 checkpoint lineage mismatch in {path}: {mismatches}"
        )
    if checkpoint.get("mechanism_partition_sha256") != expected_partition_hash:
        raise RuntimeError(f"H2 checkpoint partition mismatch: {path}")
    mean_config = checkpoint.get("mean_config", {})
    if mean_config.get("pathology_policy") != expected_variant:
        raise RuntimeError(f"H2 checkpoint variant mismatch: {path}")
    if checkpoint.get("epoch") != expected_epochs:
        raise RuntimeError(f"H2 checkpoint epoch mismatch: {path}")
    return checkpoint


def _load_predictor(
    checkpoint: Mapping[str, Any],
    *,
    device: torch.device,
) -> LowFrequencyPETPredictor:
    mean_config = checkpoint.get("mean_config", {})
    model = LowFrequencyPETPredictor(
        in_channels=int(mean_config.get("in_channels", 1)),
        base_channels=int(mean_config.get("base_channels", 32)),
        levels=int(mean_config.get("levels", 2)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    return model


def _region_mean(values: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    axes = tuple(range(1, values.ndim))
    numerator = (values * region).sum(dim=axes)
    denominator = region.sum(dim=axes).clamp_min(1.0)
    return numerator / denominator


def _gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    kernel_x = image.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    gx = F.conv2d(image, kernel_x, padding=1)
    gy = F.conv2d(image, kernel_y, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-12)


def _spatial_regions(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    binary = (mask > 0.5).float()
    ring = (
        F.max_pool2d(binary, kernel_size=25, stride=1, padding=12) - binary
    ).clamp(0.0, 1.0)
    dilated = F.max_pool2d(binary, kernel_size=5, stride=1, padding=2)
    eroded = 1.0 - F.max_pool2d(
        1.0 - binary,
        kernel_size=5,
        stride=1,
        padding=2,
    )
    boundary = (dilated - eroded).clamp(0.0, 1.0)
    return ring, boundary


@torch.no_grad()
def _evaluate_variants(
    *,
    models: Mapping[str, LowFrequencyPETPredictor],
    datasets: Mapping[str, IndexedDataset],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for partition_name, dataset in datasets.items():
        loader = _loader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
            num_workers=num_workers,
            device=device,
        )
        offset = 0
        for batch in loader:
            ct = batch["ct"].to(device, non_blocking=True)
            pet = batch["pet"].to(device, non_blocking=True)
            mask = (batch["mask"].to(device, non_blocking=True) > 0.5).float()
            target_ll2 = mean_target_ll2(pet)
            target_low = reconstruct_lowpass(target_ll2, levels=2)
            ring, boundary = _spatial_regions(mask)
            target_gradient = _gradient_magnitude(target_low)
            batch_size_actual = int(ct.shape[0])
            batch_records = [
                {
                    "partition": partition_name,
                    "sample_id": dataset.entries[offset + index].sample_id,
                    "patient_id": dataset.entries[offset + index].patient_id,
                    "mask_area": float(mask[index].sum().item()),
                }
                for index in range(batch_size_actual)
            ]
            for variant, model in models.items():
                prediction = model(ct)
                mean_pet = prediction["mean_pet"]
                residual_abs = (pet - mean_pet).abs()
                gradient_error = (
                    _gradient_magnitude(mean_pet) - target_gradient
                ).abs()
                lesion_residual = _region_mean(residual_abs, mask)
                ring_residual = _region_mean(residual_abs, ring)
                boundary_error = _region_mean(gradient_error, boundary)
                full_lowpass_mae = (mean_pet - target_low).abs().mean(
                    dim=(1, 2, 3)
                )
                mean_gradient_boundary = _region_mean(
                    _gradient_magnitude(mean_pet),
                    boundary,
                )
                for index, record in enumerate(batch_records):
                    record[f"{variant}_residual_enrichment"] = float(
                        (lesion_residual[index] - ring_residual[index]).item()
                    )
                    record[f"{variant}_boundary_gradient_error"] = float(
                        boundary_error[index].item()
                    )
                    record[f"{variant}_boundary_gradient_magnitude"] = float(
                        mean_gradient_boundary[index].item()
                    )
                    record[f"{variant}_lowpass_mae"] = float(
                        full_lowpass_mae[index].item()
                    )
            rows.extend(batch_records)
            offset += batch_size_actual
    return rows


def _decision_from_metrics(
    patient_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary = paired_patient_bootstrap(
        patient_rows,
        partition="validation",
        left="excluded_residual_enrichment",
        right="included_residual_enrichment",
        seed=ANALYSIS_SEED,
    )
    boundary = paired_patient_bootstrap(
        patient_rows,
        partition="validation",
        left="excluded_boundary_gradient_error",
        right="included_boundary_gradient_error",
        seed=ANALYSIS_SEED + 100,
    )
    calibration = [
        row for row in patient_rows if row["partition"] == "calibration"
    ]
    calibration_boundary_harm = np.asarray(
        [
            float(row["excluded_boundary_gradient_error"])
            - float(row["included_boundary_gradient_error"])
            for row in calibration
        ],
        dtype=np.float64,
    )
    calibration_boundary_harm = calibration_boundary_harm[
        np.isfinite(calibration_boundary_harm)
    ]
    if calibration_boundary_harm.size == 0:
        raise RuntimeError("H2 calibration boundary metrics are empty")
    boundary_margin = float(
        max(0.0, np.quantile(np.abs(calibration_boundary_harm), 0.95))
    )
    enrichment_pass = bool(
        primary["ci95_low"] > 0.0 and primary["sign_flip_p"] < 0.05
    )
    boundary_pass = bool(boundary["ci95_high"] <= boundary_margin)
    decision = "PASS" if enrichment_pass and boundary_pass else "FAIL"
    evidence = {
        "decision": decision,
        "residual_enrichment": {
            "status": "PASS" if enrichment_pass else "FAIL",
            "contrast": "excluded - included; positive means more lesion residual enrichment",
            "validation": primary,
        },
        "mask_boundary_safety": {
            "status": "PASS" if boundary_pass else "FAIL",
            "contrast": "excluded - included boundary gradient error; lower is safer",
            "calibration_noninferiority_margin": boundary_margin,
            "validation": boundary,
            "structural_mask_independence": True,
        },
    }
    thresholds = {
        "primary_null": 0.0,
        "primary_alpha": 0.05,
        "boundary_margin_source": "calibration patient absolute paired differences q95",
        "boundary_noninferiority_margin": boundary_margin,
    }
    return evidence, thresholds


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    output = _resolve(root, args.output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = _resolve(root, args.config)
    config = _load_config(config_path)
    contract_path = _resolve(root, args.contract)
    contract = load_json(contract_path)
    h1_decision = require_upstream_pass(
        _resolve(root, args.h1_decision),
        expected_stage="00B_H1_local_spectral_asymmetry",
    )
    if h1_decision.get("dataset_contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise RuntimeError("H1 decision and locked dataset contract differ")

    manifest_path = _resolve(root, args.manifest)
    rows = read_manifest(manifest_path)
    partition = patient_partition(rows)
    counts = partition_counts(rows, partition)
    if counts != {
        "mechanism_train": {"patients": 99, "samples": counts["mechanism_train"]["samples"]},
        "calibration": {"patients": 25, "samples": counts["calibration"]["samples"]},
        "validation": {"patients": 31, "samples": counts["validation"]["samples"]},
    }:
        observed_patients = {
            role: values["patients"] for role, values in counts.items()
        }
        if observed_patients != {
            "mechanism_train": 99,
            "calibration": 25,
            "validation": 31,
        }:
            raise RuntimeError(
                f"Unexpected formal patient partition: {observed_patients}"
            )
    current_partition_hash = partition_sha256(partition)
    derived_manifest = write_mechanism_manifest(
        rows,
        partition,
        output / "mechanism_split_manifest.csv",
    )

    data_cfg = config.setdefault("data", {})
    data_cfg["cache_dir"] = str(_resolve(root, args.cache_dir))
    data_cfg["cache_lineage"] = str(_resolve(root, args.cache_lineage))
    data_cfg["dataset_contract"] = str(contract_path)
    data_cfg["require_cache_lineage"] = True
    data_lineage = load_checkpoint_data_lineage(config, root=root)
    if data_lineage is None:
        raise RuntimeError("H2 requires sealed cache lineage")
    if data_lineage["dataset_contract_sha256"] != contract["contract_sha256"]:
        raise RuntimeError("Cache lineage and locked dataset contract differ")

    cache_dir = _resolve(root, args.cache_dir)
    datasets = _make_partition_datasets(
        cache_dir=cache_dir,
        manifest_path=manifest_path,
        partition=partition,
        required_keys=("ct", "pet", "mask"),
    )
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _set_seed(args.seed)

    mean_cfg = config.get("modules", {}).get("conditional_mean", {})
    model_kwargs = {
        "in_channels": int(mean_cfg.get("in_channels", 1)),
        "base_channels": int(mean_cfg.get("base_channels", 32)),
        "levels": int(mean_cfg.get("levels", 2)),
    }
    initial_model = LowFrequencyPETPredictor(**model_kwargs)
    initial_state = copy.deepcopy(initial_model.state_dict())
    del initial_model

    training_cfg = config.get("training", {})
    checkpoints: dict[str, dict[str, Any]] = {}
    loaded_checkpoints: dict[str, Mapping[str, Any]] = {}
    for variant in ("included", "excluded"):
        path = output / "checkpoints" / f"mean_{variant}_fixed.pt"
        if path.exists() and not args.force:
            loaded = _validate_reusable_checkpoint(
                path,
                data_lineage=data_lineage,
                expected_partition_hash=current_partition_hash,
                expected_variant=variant,
                expected_epochs=args.epochs,
            )
            checkpoints[variant] = {
                "path": path.as_posix(),
                "sha256": file_sha256(path),
                "epochs": int(loaded["epoch"]),
                "reused": True,
            }
            loaded_checkpoints[variant] = loaded
            print(f"Reusing verified H2 checkpoint: {path}", flush=True)
        else:
            metadata = _train_variant(
                variant=variant,
                initial_state=initial_state,
                model_kwargs=model_kwargs,
                dataset=datasets["mechanism_train"],
                data_lineage=data_lineage,
                partition_hash=current_partition_hash,
                output_path=path,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                learning_rate=float(training_cfg.get("learning_rate", 1e-4)),
                lr_min=float(training_cfg.get("lr_min", 1e-6)),
                weight_decay=float(training_cfg.get("weight_decay", 0.01)),
                epsilon=float(mean_cfg.get("charbonnier_eps", 1e-3)),
                guard_radius_px=args.guard_radius_px,
                num_workers=args.num_workers,
                device=device,
            )
            metadata["reused"] = False
            checkpoints[variant] = metadata
            loaded_checkpoints[variant] = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )

    models = {
        variant: _load_predictor(checkpoint, device=device)
        for variant, checkpoint in loaded_checkpoints.items()
    }
    sample_rows = _evaluate_variants(
        models=models,
        datasets=datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    metric_names = [
        f"{variant}_{metric}"
        for variant in ("included", "excluded")
        for metric in (
            "residual_enrichment",
            "boundary_gradient_error",
            "boundary_gradient_magnitude",
            "lowpass_mae",
        )
    ]
    patient_rows = aggregate_patient_rows(sample_rows, metric_names)
    evidence, thresholds = _decision_from_metrics(patient_rows)
    write_csv(output / "sample_metrics.csv", sample_rows)
    write_csv(output / "patient_metrics.csv", patient_rows)
    write_json(output / "thresholds.json", thresholds)
    write_json(
        output / "analysis_spec.json",
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis": "H2",
            "training_intervention": "included full LL2 loss vs excluded dilated pathology support",
            "inference_inputs": ["ct"],
            "mask_used_at_inference": False,
            "fixed_epochs": args.epochs,
            "same_initialization": True,
            "same_batch_order": True,
            "guard_radius_px": args.guard_radius_px,
            "partition": counts,
            "partition_sha256": current_partition_hash,
            "derived_manifest": derived_manifest,
            "thresholds": thresholds,
        },
    )

    h2_pass = evidence["decision"] == "PASS"
    decision = {
        "schema_version": SCHEMA_VERSION,
        "stage": "01_H2_pathology_excluded_residual",
        "decision": evidence["decision"],
        "dataset_contract_sha256": contract["contract_sha256"],
        "cache_metadata_sha256": data_lineage["cache_metadata_sha256"],
        "mechanism_partition_sha256": current_partition_hash,
        "partition_counts": counts,
        "guardrails": decision_guardrails(),
        "H2": evidence,
        "checkpoints": checkpoints,
        "downstream_mean_checkpoint": (
            checkpoints["excluded"]["path"] if h2_pass else None
        ),
        "next_stage_allowed": h2_pass,
        "stop_rule": (
            "H2 passed; H3 may use only the pathology-excluded fixed endpoint."
            if h2_pass
            else "Stop the pathology-excluded residual module; do not tune this gate to preserve the story."
        ),
    }
    write_json(output / "decision.json", decision)
    write_json(
        output / "execution_metadata.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "command": " ".join(sys.argv),
            "script": Path(__file__).resolve().as_posix(),
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "config": config_path.as_posix(),
            "config_sha256": file_sha256(config_path),
            "device": str(device),
            "training_used_validation_for_selection": False,
        },
    )
    return decision


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/slmf_png_residual_frequency.yaml"),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/dataset_contract_stage0a_v1.json"),
    )
    parser.add_argument(
        "--h1-decision",
        type=Path,
        default=Path(
            "results/mechanism_validation/00B_h1_spectral_asymmetry/decision.json"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("main_data/split_manifest.csv"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache/tensors_main"),
    )
    parser.add_argument(
        "--cache-lineage",
        type=Path,
        default=Path("cache/tensors_main/cache_lineage.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mechanism_validation/01_h2_residual_enrichment"),
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--guard-radius-px", type=int, default=DEFAULT_GUARD_RADIUS_PX)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.root = args.root.resolve()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")
    if args.guard_radius_px < 0 or args.num_workers < 0:
        parser.error("--guard-radius-px and --num-workers cannot be negative")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        decision = run(args)
    except Exception as exc:
        output = _resolve(args.root, args.output)
        failure = {
            "schema_version": SCHEMA_VERSION,
            "stage": "01_H2_pathology_excluded_residual",
            "decision": "FAIL",
            "dataset_contract_sha256": None,
            "guardrails": decision_guardrails(checkpoint_lineage="NOT_EVALUATED"),
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "next_stage_allowed": False,
            "stop_rule": "Stop H2 and all dependent stages.",
        }
        write_json(output / "decision.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
        return 2
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    return 0 if decision["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
