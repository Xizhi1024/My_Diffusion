"""H3: patient-level recoverability crossings over fixed bridge log-SNR.

The probe corrupts the H2 pathology-excluded residual with the analytic
Brownian-bridge noise law.  It measures four predeclared targets:

* coarse LL2 alignment;
* lesion-shape support AUC;
* lesion residual-intensity recovery;
* lesion-local L1-HH alignment.

Calibration freezes task thresholds and selects the most separated task pair.
Validation is then used once for a paired patient bootstrap/sign-flip gate.
No validation curve changes the time grid, thresholds, or course.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_h2_pathology_excluded_residual import (
    IndexedDataset,
    _load_predictor,
    _loader,
    _make_partition_datasets,
    _spatial_regions,
)
from src.data.lineage import (
    REQUIRED_CHECKPOINT_LINEAGE_FIELDS,
    load_checkpoint_data_lineage,
)
from src.mechanism_validation.common import (
    bootstrap_mean,
    decision_guardrails,
    file_sha256,
    load_json,
    partition_counts,
    partition_sha256,
    patient_partition,
    read_manifest,
    sign_flip_p,
    write_csv,
    write_json,
)
from src.model.frequency.haar import haar_dwt2
from src.model.noise.base import BBDMBridgeSchedule


SCHEMA_VERSION = 1
ANALYSIS_SEED = 20260725
TASKS = ("coarse", "shape", "intensity", "local_frequency")
DEFAULT_TIMESTEPS = (0, 50, 100, 200, 350, 500, 650, 800, 900, 950)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be an object: {path}")
    return payload


def _cosine_score(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.float().flatten(1)
    right_flat = right.float().flatten(1)
    numerator = (left_flat * right_flat).sum(dim=1)
    denominator = (
        left_flat.square().sum(dim=1).sqrt()
        * right_flat.square().sum(dim=1).sqrt()
    ).clamp_min(1e-8)
    return ((numerator / denominator).clamp(-1.0, 1.0) + 1.0) * 0.5


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numerator = (values * mask).sum(dim=(1, 2, 3))
    denominator = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return numerator / denominator


def _shape_auc(
    score: torch.Tensor,
    mask: torch.Tensor,
    ring: torch.Tensor,
) -> list[float]:
    values: list[float] = []
    score_np = score.detach().float().cpu().numpy()
    mask_np = mask.detach().cpu().numpy() > 0.5
    ring_np = ring.detach().cpu().numpy() > 0.5
    for index in range(score_np.shape[0]):
        selected = mask_np[index, 0] | ring_np[index, 0]
        labels = mask_np[index, 0][selected].astype(np.uint8)
        current = score_np[index, 0][selected].astype(np.float64)
        if (
            labels.size < 4
            or np.unique(labels).size != 2
            or np.ptp(current) <= 1e-12
        ):
            values.append(0.5)
        else:
            values.append(float(roc_auc_score(labels, current)))
    return values


def _hh1(image: torch.Tensor) -> torch.Tensor:
    _, details = haar_dwt2(image)
    return details[2]


def _ll2(image: torch.Tensor) -> torch.Tensor:
    ll1, _ = haar_dwt2(image)
    ll2, _ = haar_dwt2(ll1)
    return ll2


@torch.no_grad()
def _intensity_scale(
    model: torch.nn.Module,
    dataset: IndexedDataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> float:
    values: list[float] = []
    loader = _loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=num_workers,
        device=device,
    )
    for batch in loader:
        ct = batch["ct"].to(device, non_blocking=True)
        pet = batch["pet"].to(device, non_blocking=True)
        mask = (batch["mask"].to(device, non_blocking=True) > 0.5).float()
        residual = pet - model(ct)["mean_pet"]
        current = _masked_mean(residual, mask).abs()
        values.extend(current.detach().cpu().tolist())
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise RuntimeError("Cannot derive H3 intensity scale")
    return float(max(np.median(array), 1e-3))


def _seeded_noise_like(
    reference: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(
        reference.shape,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    return noise.to(device=reference.device, dtype=reference.dtype)


@torch.no_grad()
def _recoverability_rows(
    *,
    model: torch.nn.Module,
    datasets: Mapping[str, IndexedDataset],
    schedule: BBDMBridgeSchedule,
    timesteps: Sequence[int],
    intensity_scale: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for partition_index, (partition_name, dataset) in enumerate(datasets.items()):
        loader = _loader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
            num_workers=num_workers,
            device=device,
        )
        offset = 0
        for batch_index, batch in enumerate(loader):
            ct = batch["ct"].to(device, non_blocking=True)
            pet = batch["pet"].to(device, non_blocking=True)
            mask = (batch["mask"].to(device, non_blocking=True) > 0.5).float()
            residual = pet - model(ct)["mean_pet"]
            clean_ll2 = _ll2(residual)
            clean_hh = _hh1(residual)
            hh_mask = F.max_pool2d(mask, kernel_size=2, stride=2)
            ring, _ = _spatial_regions(mask)
            clean_intensity = _masked_mean(residual, mask)
            batch_actual = int(ct.shape[0])
            entries = dataset.entries[offset : offset + batch_actual]
            for timestep_index, timestep in enumerate(timesteps):
                t = torch.full(
                    (batch_actual,),
                    int(timestep),
                    dtype=torch.long,
                    device=device,
                )
                m = schedule.m_t[t].to(device=device, dtype=residual.dtype)
                sigma = schedule.sigma_t[t].to(
                    device=device,
                    dtype=residual.dtype,
                )
                noise = _seeded_noise_like(
                    residual,
                    seed=(
                        ANALYSIS_SEED
                        + partition_index * 1_000_000
                        + batch_index * 10_000
                        + timestep_index
                    ),
                )
                noisy = (
                    (1.0 - m)[:, None, None, None] * residual
                    + sigma[:, None, None, None] * noise
                )
                log_snr = torch.log(
                    (1.0 - m).square().clamp_min(1e-8)
                    / sigma.square().clamp_min(1e-8)
                ).clamp(-20.0, 20.0)
                coarse = _cosine_score(_ll2(noisy), clean_ll2)
                shape = _shape_auc(noisy.abs(), mask, ring)
                estimated_intensity = _masked_mean(noisy, mask) / (
                    1.0 - m
                ).clamp_min(0.05)
                intensity = torch.exp(
                    -(estimated_intensity - clean_intensity).abs()
                    / intensity_scale
                )
                local_frequency = _cosine_score(
                    _hh1(noisy) * hh_mask,
                    clean_hh * hh_mask,
                )
                task_values = {
                    "coarse": coarse.detach().cpu().tolist(),
                    "shape": shape,
                    "intensity": intensity.detach().cpu().tolist(),
                    "local_frequency": local_frequency.detach().cpu().tolist(),
                }
                for index, entry in enumerate(entries):
                    for task in TASKS:
                        rows.append(
                            {
                                "partition": partition_name,
                                "patient_id": entry.patient_id,
                                "sample_id": entry.sample_id,
                                "task": task,
                                "timestep": int(timestep),
                                "log_snr": float(log_snr[index].item()),
                                "recoverability": float(task_values[task][index]),
                            }
                        )
            offset += batch_actual
    return rows


def _patient_curves(
    sample_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    log_snr_by_timestep: dict[int, float] = {}
    for row in sample_rows:
        key = (
            str(row["partition"]),
            str(row["patient_id"]),
            str(row["task"]),
            int(row["timestep"]),
        )
        grouped[key].append(float(row["recoverability"]))
        log_snr_by_timestep[int(row["timestep"])] = float(row["log_snr"])
    result = []
    for (partition, patient, task, timestep), values in sorted(grouped.items()):
        result.append(
            {
                "partition": partition,
                "patient_id": patient,
                "task": task,
                "timestep": timestep,
                "log_snr": log_snr_by_timestep[timestep],
                "recoverability": float(np.mean(values)),
                "samples": len(values),
            }
        )
    return result


def freeze_recoverability_thresholds(
    patient_curves: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    calibration = [
        row for row in patient_curves if row["partition"] == "calibration"
    ]
    if not calibration:
        raise RuntimeError("H3 calibration curves are empty")
    timesteps = sorted({int(row["timestep"]) for row in calibration})
    clean_timestep = min(timesteps)
    noise_timestep = max(timesteps)
    thresholds: dict[str, float] = {}
    for task in TASKS:
        clean = [
            float(row["recoverability"])
            for row in calibration
            if row["task"] == task and int(row["timestep"]) == clean_timestep
        ]
        noise = [
            float(row["recoverability"])
            for row in calibration
            if row["task"] == task and int(row["timestep"]) == noise_timestep
        ]
        if not clean or not noise:
            raise RuntimeError(f"H3 threshold endpoints missing for {task}")
        clean_mean = float(np.mean(clean))
        noise_mean = float(np.mean(noise))
        thresholds[task] = float(noise_mean + 0.5 * (clean_mean - noise_mean))
    return thresholds


def crossing_log_snr(
    curve: Sequence[Mapping[str, Any]],
    threshold: float,
) -> float:
    ordered = sorted(curve, key=lambda row: float(row["log_snr"]))
    values = np.asarray(
        [float(row["recoverability"]) for row in ordered],
        dtype=np.float64,
    )
    log_snr = np.asarray(
        [float(row["log_snr"]) for row in ordered],
        dtype=np.float64,
    )
    # Monotone envelope is fixed a priori; it removes Monte-Carlo reversals
    # without fitting anything to validation.
    envelope = np.maximum.accumulate(values)
    indices = np.flatnonzero(envelope >= threshold)
    if indices.size == 0:
        return float("nan")
    return float(log_snr[int(indices[0])])


def _patient_crossings(
    patient_curves: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in patient_curves:
        grouped[
            (
                str(row["partition"]),
                str(row["patient_id"]),
                str(row["task"]),
            )
        ].append(row)
    by_patient: dict[tuple[str, str], dict[str, Any]] = {}
    for (partition, patient, task), curve in grouped.items():
        record = by_patient.setdefault(
            (partition, patient),
            {"partition": partition, "patient_id": patient},
        )
        record[f"{task}_crossing_log_snr"] = crossing_log_snr(
            curve,
            thresholds[task],
        )
    return [by_patient[key] for key in sorted(by_patient)]


def select_calibration_pair(
    patient_crossings: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    calibration = [
        row for row in patient_crossings if row["partition"] == "calibration"
    ]
    task_means = {}
    for task in TASKS:
        values = np.asarray(
            [float(row[f"{task}_crossing_log_snr"]) for row in calibration],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        if values.size < max(5, len(calibration) // 2):
            raise RuntimeError(f"H3 calibration crossing not identifiable for {task}")
        task_means[task] = float(values.mean())
    pairs = [
        (left, right)
        for index, left in enumerate(TASKS)
        for right in TASKS[index + 1 :]
    ]
    first, second = max(
        pairs,
        key=lambda pair: (
            abs(task_means[pair[0]] - task_means[pair[1]]),
            -pairs.index(pair),
        ),
    )
    return (
        (first, second)
        if task_means[first] >= task_means[second]
        else (second, first)
    )


def _h3_decision(
    patient_crossings: Sequence[Mapping[str, Any]],
    selected_pair: tuple[str, str],
) -> dict[str, Any]:
    validation = [
        row for row in patient_crossings if row["partition"] == "validation"
    ]
    left, right = selected_pair
    differences = np.asarray(
        [
            float(row[f"{left}_crossing_log_snr"])
            - float(row[f"{right}_crossing_log_snr"])
            for row in validation
        ],
        dtype=np.float64,
    )
    differences = differences[np.isfinite(differences)]
    comparison = bootstrap_mean(differences, seed=ANALYSIS_SEED)
    comparison["sign_flip_p"] = sign_flip_p(
        differences,
        seed=ANALYSIS_SEED + 1,
    )
    comparison["contrast"] = f"{left} - {right} crossing log-SNR"
    identifiable_fraction = float(
        differences.size / max(len(validation), 1)
    )
    passed = bool(
        identifiable_fraction >= 0.80
        and comparison["ci95_low"] > 0.0
        and comparison["sign_flip_p"] < 0.05
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "selected_calibration_pair": [left, right],
        "validation_identifiable_fraction": identifiable_fraction,
        "validation_comparison": comparison,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    output = _resolve(root, args.output)
    output.mkdir(parents=True, exist_ok=True)
    contract = load_json(_resolve(root, args.contract))
    h2 = load_json(_resolve(root, args.h2_decision))
    if h2.get("decision") != "PASS" or not h2.get("next_stage_allowed"):
        raise RuntimeError("H2 is not PASS; H3 is blocked")
    if h2.get("dataset_contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("H2 decision and locked contract differ")
    mean_checkpoint_path = Path(str(h2["downstream_mean_checkpoint"]))
    if not mean_checkpoint_path.is_absolute():
        mean_checkpoint_path = _resolve(root, mean_checkpoint_path)
    if not mean_checkpoint_path.is_file():
        raise FileNotFoundError(mean_checkpoint_path)

    config_path = _resolve(root, args.config)
    config = _load_config(config_path)
    data_cfg = config.setdefault("data", {})
    data_cfg["cache_dir"] = str(_resolve(root, args.cache_dir))
    data_cfg["cache_lineage"] = str(_resolve(root, args.cache_lineage))
    data_cfg["dataset_contract"] = str(_resolve(root, args.contract))
    data_cfg["require_cache_lineage"] = True
    data_lineage = load_checkpoint_data_lineage(config, root=root)
    if data_lineage is None:
        raise RuntimeError("H3 requires sealed cache lineage")
    checkpoint = torch.load(
        mean_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    embedded = checkpoint.get("data_lineage", {})
    lineage_mismatches = [
        field
        for field in REQUIRED_CHECKPOINT_LINEAGE_FIELDS
        if embedded.get(field) != data_lineage.get(field)
    ]
    if lineage_mismatches:
        raise RuntimeError(
            f"H2 mean checkpoint lineage mismatch: {lineage_mismatches}"
        )

    manifest_path = _resolve(root, args.manifest)
    manifest = read_manifest(manifest_path)
    partition = patient_partition(manifest)
    current_partition_sha = partition_sha256(partition)
    if current_partition_sha != h2.get("mechanism_partition_sha256"):
        raise RuntimeError("H2 and H3 patient partitions differ")
    datasets = _make_partition_datasets(
        cache_dir=_resolve(root, args.cache_dir),
        manifest_path=manifest_path,
        partition=partition,
        required_keys=("ct", "pet", "mask"),
    )
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = _load_predictor(checkpoint, device=device)
    intensity_scale = _intensity_scale(
        model,
        datasets["mechanism_train"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    schedule = BBDMBridgeSchedule(
        num_train_timesteps=args.num_train_timesteps,
        m_schedule="linear",
        sigma_scale=args.sigma_scale,
    ).to(device)
    timesteps = tuple(args.timesteps)
    if (
        min(timesteps) < 0
        or max(timesteps) >= args.num_train_timesteps
        or tuple(sorted(set(timesteps))) != timesteps
    ):
        raise ValueError("H3 timesteps must be sorted, unique, and in schedule range")
    sample_rows = _recoverability_rows(
        model=model,
        datasets=datasets,
        schedule=schedule,
        timesteps=timesteps,
        intensity_scale=intensity_scale,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    patient_curves = _patient_curves(sample_rows)
    thresholds = freeze_recoverability_thresholds(patient_curves)
    patient_crossings = _patient_crossings(patient_curves, thresholds)
    selected_pair = select_calibration_pair(patient_crossings)
    h3 = _h3_decision(patient_crossings, selected_pair)
    task_statistics = {}
    for task_index, task in enumerate(TASKS):
        values = [
            float(row[f"{task}_crossing_log_snr"])
            for row in patient_crossings
            if row["partition"] == "validation"
        ]
        task_statistics[task] = bootstrap_mean(
            values,
            seed=ANALYSIS_SEED + 100 + task_index,
        )
    write_csv(output / "sample_recoverability.csv", sample_rows)
    write_csv(output / "patient_recoverability_curves.csv", patient_curves)
    write_csv(output / "patient_crossings.csv", patient_crossings)
    write_json(
        output / "thresholds.json",
        {
            "source": "calibration only",
            "recoverability_thresholds": thresholds,
            "selected_pair": list(selected_pair),
            "intensity_scale_source": "mechanism_train median absolute lesion residual",
            "intensity_scale": intensity_scale,
        },
    )
    write_json(
        output / "analysis_spec.json",
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis": "H3",
            "tasks": list(TASKS),
            "timesteps": list(timesteps),
            "num_train_timesteps": args.num_train_timesteps,
            "sigma_scale": args.sigma_scale,
            "crossing_policy": "first threshold crossing of cumulative-max recoverability when ordered by log-SNR",
            "calibration_pair_policy": "largest absolute mean crossing separation; direction frozen on calibration",
            "partition_counts": partition_counts(manifest, partition),
        },
    )
    passed = h3["status"] == "PASS"
    decision = {
        "schema_version": SCHEMA_VERSION,
        "stage": "02_H3_logsnr_recoverability",
        "decision": "PASS" if passed else "FAIL",
        "dataset_contract_sha256": contract["contract_sha256"],
        "cache_metadata_sha256": data_lineage["cache_metadata_sha256"],
        "mechanism_partition_sha256": current_partition_sha,
        "guardrails": decision_guardrails(),
        "H3": h3,
        "recoverability_thresholds": thresholds,
        "validation_task_crossing_statistics": task_statistics,
        "mean_checkpoint": {
            "path": mean_checkpoint_path.as_posix(),
            "sha256": file_sha256(mean_checkpoint_path),
        },
        "next_stage_allowed": passed,
        "stop_rule": (
            "H3 passed; H4 may test the predeclared noise evidence."
            if passed
            else "Stop time-level curriculum and recoverability-conditioned routing."
        ),
    }
    write_json(output / "decision.json", decision)
    write_json(
        output / "execution_metadata.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "device": str(device),
            "validation_used_to_choose_time_grid_or_thresholds": False,
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
        "--h2-decision",
        type=Path,
        default=Path(
            "results/mechanism_validation/01_h2_residual_enrichment/decision.json"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("main_data/split_manifest.csv"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/tensors_main"))
    parser.add_argument(
        "--cache-lineage",
        type=Path,
        default=Path("cache/tensors_main/cache_lineage.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mechanism_validation/02_h3_recoverability_curves"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--sigma-scale", type=float, default=1.0)
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=list(DEFAULT_TIMESTEPS),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.root = args.root.resolve()
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        decision = run(args)
    except Exception as exc:
        output = _resolve(args.root, args.output)
        failure = {
            "schema_version": SCHEMA_VERSION,
            "stage": "02_H3_logsnr_recoverability",
            "decision": "FAIL",
            "dataset_contract_sha256": None,
            "guardrails": decision_guardrails(checkpoint_lineage="NOT_EVALUATED"),
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "next_stage_allowed": False,
            "stop_rule": "Stop H3 and all dependent time/router stages.",
        }
        write_json(output / "decision.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
        return 2
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    return 0 if decision["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
