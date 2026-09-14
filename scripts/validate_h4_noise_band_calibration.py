"""H4: pure-noise calibrated band evidence predicts real recoverability.

For each patient, Haar band, and fixed H3 timestep, the probe computes:

* evidence = log(observed noisy-residual band energy / matched pure-noise
  band energy);
* true recoverability = cosine alignment of noisy and clean residual bands.

A mechanism-train ridge probe compares log-SNR+band against
log-SNR+band+evidence.  Alphas and the patient-permutation null q95 are frozen
on calibration.  Validation is tested once with a patient bootstrap and
sign-flip test.  This is predictive evidence, not causality or mutual
information.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_h2_pathology_excluded_residual import (
    IndexedDataset,
    _load_predictor,
    _loader,
    _make_partition_datasets,
)
from scripts.validate_h3_logsnr_recoverability import (
    _seeded_noise_like,
)
from src.data.lineage import (
    REQUIRED_CHECKPOINT_LINEAGE_FIELDS,
    load_checkpoint_data_lineage,
)
from src.mechanism_validation.common import (
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
ANALYSIS_SEED = 20260726
BANDS = ("l2_lh", "l2_hl", "l2_hh", "l1_lh", "l1_hl", "l1_hh")
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_NULL_REPLICATES = 2_000
DEFAULT_BOOTSTRAP_REPLICATES = 10_000


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be an object: {path}")
    return payload


def _band_tensors(image: torch.Tensor) -> dict[str, torch.Tensor]:
    ll1, details1 = haar_dwt2(image)
    _, details2 = haar_dwt2(ll1)
    return {
        "l2_lh": details2[0],
        "l2_hl": details2[1],
        "l2_hh": details2[2],
        "l1_lh": details1[0],
        "l1_hl": details1[1],
        "l1_hh": details1[2],
    }


def _masked_band_energy(
    band: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    numerator = (band.abs() * mask).sum(dim=(1, 2, 3))
    denominator = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return numerator / denominator


def _masked_band_alignment(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    noisy_flat = (noisy * mask).float().flatten(1)
    clean_flat = (clean * mask).float().flatten(1)
    numerator = (noisy_flat * clean_flat).sum(dim=1)
    denominator = (
        noisy_flat.square().sum(dim=1).sqrt()
        * clean_flat.square().sum(dim=1).sqrt()
    ).clamp_min(1e-8)
    return ((numerator / denominator).clamp(-1.0, 1.0) + 1.0) * 0.5


@torch.no_grad()
def _sample_band_rows(
    *,
    model: torch.nn.Module,
    datasets: Mapping[str, IndexedDataset],
    schedule: BBDMBridgeSchedule,
    timesteps: Sequence[int],
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
            clean_bands = _band_tensors(residual)
            mask_l1 = F.max_pool2d(mask, kernel_size=2, stride=2)
            mask_l2 = F.max_pool2d(mask, kernel_size=4, stride=4)
            masks = {
                band: mask_l2 if band.startswith("l2_") else mask_l1
                for band in BANDS
            }
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
                pure_noise = sigma[:, None, None, None] * noise
                noisy_bands = _band_tensors(noisy)
                noise_bands = _band_tensors(pure_noise)
                log_snr = torch.log(
                    (1.0 - m).square().clamp_min(1e-8)
                    / sigma.square().clamp_min(1e-8)
                ).clamp(-20.0, 20.0)
                for band in BANDS:
                    observed_energy = _masked_band_energy(
                        noisy_bands[band],
                        masks[band],
                    )
                    pure_noise_energy = _masked_band_energy(
                        noise_bands[band],
                        masks[band],
                    )
                    evidence = torch.log(
                        (observed_energy + 1e-6)
                        / (pure_noise_energy + 1e-6)
                    ).clamp(-20.0, 20.0)
                    recoverability = _masked_band_alignment(
                        noisy_bands[band],
                        clean_bands[band],
                        masks[band],
                    )
                    for index, entry in enumerate(entries):
                        rows.append(
                            {
                                "partition": partition_name,
                                "patient_id": entry.patient_id,
                                "sample_id": entry.sample_id,
                                "band": band,
                                "timestep": int(timestep),
                                "log_snr": float(log_snr[index].item()),
                                "noise_calibrated_evidence": float(
                                    evidence[index].item()
                                ),
                                "recoverability": float(
                                    recoverability[index].item()
                                ),
                            }
                        )
            offset += batch_actual
    return rows


def _patient_band_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str, int],
        list[Mapping[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["partition"]),
                str(row["patient_id"]),
                str(row["band"]),
                int(row["timestep"]),
            )
        ].append(row)
    result = []
    for (partition, patient, band, timestep), current in sorted(grouped.items()):
        result.append(
            {
                "partition": partition,
                "patient_id": patient,
                "band": band,
                "timestep": timestep,
                "log_snr": float(np.mean([row["log_snr"] for row in current])),
                "noise_calibrated_evidence": float(
                    np.mean(
                        [row["noise_calibrated_evidence"] for row in current]
                    )
                ),
                "recoverability": float(
                    np.mean([row["recoverability"] for row in current])
                ),
                "samples": len(current),
            }
        )
    return result


def _feature_matrices(
    rows: Sequence[Mapping[str, Any]],
    *,
    means: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    band_index = {band: index for index, band in enumerate(BANDS)}
    numeric = np.asarray(
        [
            [
                float(row["log_snr"]) / 20.0,
                float(row["noise_calibrated_evidence"]) / 20.0,
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    if means is None:
        means = numeric.mean(axis=0)
    if scales is None:
        scales = numeric.std(axis=0)
    scales = np.maximum(scales, 1e-8)
    standardized = (numeric - means) / scales
    one_hot = np.zeros((len(rows), len(BANDS) - 1), dtype=np.float64)
    for row_index, row in enumerate(rows):
        index = band_index[str(row["band"])]
        if index > 0:
            one_hot[row_index, index - 1] = 1.0
    baseline = np.concatenate((standardized[:, :1], one_hot), axis=1)
    full = np.concatenate((baseline, standardized[:, 1:2]), axis=1)
    targets = np.asarray(
        [float(row["recoverability"]) for row in rows],
        dtype=np.float64,
    )
    return baseline, full, targets, means, scales


def _mse_skill(
    target: np.ndarray,
    baseline_prediction: np.ndarray,
    full_prediction: np.ndarray,
) -> float:
    baseline_error = float(np.mean((target - baseline_prediction) ** 2))
    full_error = float(np.mean((target - full_prediction) ** 2))
    return float(1.0 - full_error / max(baseline_error, 1e-12))


def fit_calibrated_probes(
    patient_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    train_rows = [
        row for row in patient_rows if row["partition"] == "mechanism_train"
    ]
    calibration_rows = [
        row for row in patient_rows if row["partition"] == "calibration"
    ]
    train_baseline, train_full, train_target, means, scales = _feature_matrices(
        train_rows
    )
    cal_baseline, cal_full, cal_target, _, _ = _feature_matrices(
        calibration_rows,
        means=means,
        scales=scales,
    )

    def choose_alpha(
        train_features: np.ndarray,
        calibration_features: np.ndarray,
    ) -> tuple[float, list[dict[str, float]]]:
        grid = []
        for alpha in RIDGE_ALPHAS:
            model = Ridge(alpha=alpha).fit(train_features, train_target)
            prediction = model.predict(calibration_features)
            grid.append(
                {
                    "alpha": float(alpha),
                    "calibration_mse": float(
                        np.mean((cal_target - prediction) ** 2)
                    ),
                }
            )
        best = min(grid, key=lambda row: (row["calibration_mse"], row["alpha"]))
        return float(best["alpha"]), grid

    baseline_alpha, baseline_grid = choose_alpha(train_baseline, cal_baseline)
    full_alpha, full_grid = choose_alpha(train_full, cal_full)
    baseline_model = Ridge(alpha=baseline_alpha).fit(
        train_baseline,
        train_target,
    )
    full_model = Ridge(alpha=full_alpha).fit(train_full, train_target)
    return {
        "baseline_model": baseline_model,
        "full_model": full_model,
        "means": means,
        "scales": scales,
        "baseline_alpha": baseline_alpha,
        "full_alpha": full_alpha,
        "baseline_grid": baseline_grid,
        "full_grid": full_grid,
        "calibration_skill": _mse_skill(
            cal_target,
            baseline_model.predict(cal_baseline),
            full_model.predict(cal_full),
        ),
        "train_rows": train_rows,
        "calibration_rows": calibration_rows,
    }


def calibration_permutation_null(
    fitted: Mapping[str, Any],
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    train_rows = list(fitted["train_rows"])
    calibration_rows = list(fitted["calibration_rows"])
    train_baseline, train_full, train_target, means, scales = _feature_matrices(
        train_rows,
        means=fitted["means"],
        scales=fitted["scales"],
    )
    cal_baseline, cal_full, cal_target, _, _ = _feature_matrices(
        calibration_rows,
        means=means,
        scales=scales,
    )
    baseline_model = Ridge(alpha=fitted["baseline_alpha"]).fit(
        train_baseline,
        train_target,
    )
    baseline_prediction = baseline_model.predict(cal_baseline)
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(train_rows):
        groups[(str(row["band"]), int(row["timestep"]))].append(index)
    rng = np.random.default_rng(seed)
    null = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        permuted = train_full.copy()
        for indices in groups.values():
            source = np.asarray(indices, dtype=np.int64)
            permuted[source, -1] = permuted[
                rng.permutation(source),
                -1,
            ]
        model = Ridge(alpha=fitted["full_alpha"]).fit(permuted, train_target)
        null[replicate] = _mse_skill(
            cal_target,
            baseline_prediction,
            model.predict(cal_full),
        )
    return null


def validation_probe_statistics(
    patient_rows: Sequence[Mapping[str, Any]],
    fitted: Mapping[str, Any],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    validation_rows = [
        row for row in patient_rows if row["partition"] == "validation"
    ]
    baseline, full, target, _, _ = _feature_matrices(
        validation_rows,
        means=fitted["means"],
        scales=fitted["scales"],
    )
    baseline_prediction = fitted["baseline_model"].predict(baseline)
    full_prediction = fitted["full_model"].predict(full)
    patient_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(validation_rows):
        patient_indices[str(row["patient_id"])].append(index)
    patients = sorted(patient_indices)
    baseline_mse = np.asarray(
        [
            np.mean(
                (
                    target[patient_indices[patient]]
                    - baseline_prediction[patient_indices[patient]]
                )
                ** 2
            )
            for patient in patients
        ],
        dtype=np.float64,
    )
    full_mse = np.asarray(
        [
            np.mean(
                (
                    target[patient_indices[patient]]
                    - full_prediction[patient_indices[patient]]
                )
                ** 2
            )
            for patient in patients
        ],
        dtype=np.float64,
    )
    estimate = float(1.0 - full_mse.mean() / max(baseline_mse.mean(), 1e-12))
    rng = np.random.default_rng(seed)
    draws = np.empty(bootstrap_replicates, dtype=np.float64)
    for index in range(bootstrap_replicates):
        sampled = rng.integers(0, len(patients), size=len(patients))
        draws[index] = 1.0 - full_mse[sampled].mean() / max(
            baseline_mse[sampled].mean(),
            1e-12,
        )
    low, high = np.quantile(draws, (0.025, 0.975))
    improvement = baseline_mse - full_mse
    return {
        "patients": len(patients),
        "incremental_mse_skill": estimate,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "sign_flip_p": sign_flip_p(
            improvement,
            seed=seed + 1,
            replicates=bootstrap_replicates,
        ),
        "patient_rows": [
            {
                "patient_id": patient,
                "baseline_mse": float(baseline_mse[index]),
                "full_mse": float(full_mse[index]),
                "improvement": float(improvement[index]),
            }
            for index, patient in enumerate(patients)
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    output = _resolve(root, args.output)
    output.mkdir(parents=True, exist_ok=True)
    contract = load_json(_resolve(root, args.contract))
    h3 = load_json(_resolve(root, args.h3_decision))
    if h3.get("decision") != "PASS" or not h3.get("next_stage_allowed"):
        raise RuntimeError("H3 is not PASS; H4 is blocked")
    if h3.get("dataset_contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("H3 decision and locked contract differ")
    mean_checkpoint_path = Path(str(h3["mean_checkpoint"]["path"]))
    if not mean_checkpoint_path.is_absolute():
        mean_checkpoint_path = _resolve(root, mean_checkpoint_path)

    config_path = _resolve(root, args.config)
    config = _load_config(config_path)
    data_cfg = config.setdefault("data", {})
    data_cfg["cache_dir"] = str(_resolve(root, args.cache_dir))
    data_cfg["cache_lineage"] = str(_resolve(root, args.cache_lineage))
    data_cfg["dataset_contract"] = str(_resolve(root, args.contract))
    data_cfg["require_cache_lineage"] = True
    data_lineage = load_checkpoint_data_lineage(config, root=root)
    if data_lineage is None:
        raise RuntimeError("H4 requires sealed cache lineage")
    checkpoint = torch.load(
        mean_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    embedded = checkpoint.get("data_lineage", {})
    mismatch = [
        field
        for field in REQUIRED_CHECKPOINT_LINEAGE_FIELDS
        if embedded.get(field) != data_lineage.get(field)
    ]
    if mismatch:
        raise RuntimeError(f"H4 mean checkpoint lineage mismatch: {mismatch}")

    manifest_path = _resolve(root, args.manifest)
    manifest = read_manifest(manifest_path)
    partition = patient_partition(manifest)
    current_partition_sha = partition_sha256(partition)
    if current_partition_sha != h3.get("mechanism_partition_sha256"):
        raise RuntimeError("H3 and H4 patient partitions differ")
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
    analysis_spec = load_json(
        _resolve(root, args.h3_decision).parent / "analysis_spec.json"
    )
    timesteps = tuple(int(value) for value in analysis_spec["timesteps"])
    schedule = BBDMBridgeSchedule(
        num_train_timesteps=int(analysis_spec["num_train_timesteps"]),
        m_schedule="linear",
        sigma_scale=float(analysis_spec["sigma_scale"]),
    ).to(device)
    sample_rows = _sample_band_rows(
        model=model,
        datasets=datasets,
        schedule=schedule,
        timesteps=timesteps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    patient_rows = _patient_band_rows(sample_rows)
    fitted = fit_calibrated_probes(patient_rows)
    null = calibration_permutation_null(
        fitted,
        replicates=args.null_replicates,
        seed=ANALYSIS_SEED,
    )
    null_q95 = float(np.quantile(null, 0.95))
    validation = validation_probe_statistics(
        patient_rows,
        fitted,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=ANALYSIS_SEED + 10_000,
    )
    passed = bool(
        validation["ci95_low"] > null_q95
        and validation["sign_flip_p"] < 0.05
    )
    validation_patient_rows = validation.pop("patient_rows")
    write_csv(output / "sample_band_evidence.csv", sample_rows)
    write_csv(output / "patient_band_evidence.csv", patient_rows)
    write_csv(output / "validation_patient_errors.csv", validation_patient_rows)
    write_json(
        output / "calibration.json",
        {
            "baseline_alpha": fitted["baseline_alpha"],
            "full_alpha": fitted["full_alpha"],
            "baseline_grid": fitted["baseline_grid"],
            "full_grid": fitted["full_grid"],
            "calibration_skill": fitted["calibration_skill"],
            "permutation_null_replicates": args.null_replicates,
            "permutation_null_q95": null_q95,
            "feature_means": fitted["means"].tolist(),
            "feature_scales": fitted["scales"].tolist(),
        },
    )
    write_json(
        output / "analysis_spec.json",
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis": "H4",
            "bands": list(BANDS),
            "timesteps": list(timesteps),
            "evidence": "log observed residual-band energy / matched pure-noise band energy",
            "target": "masked cosine alignment of noisy and clean residual bands",
            "baseline_features": ["bridge_log_snr", "band_identity"],
            "full_features": [
                "bridge_log_snr",
                "band_identity",
                "noise_calibrated_band_evidence",
            ],
            "threshold_source": "calibration patient-permutation null q95",
            "partition_counts": partition_counts(manifest, partition),
        },
    )
    decision = {
        "schema_version": SCHEMA_VERSION,
        "stage": "03_H4_noise_band_calibration",
        "decision": "PASS" if passed else "FAIL",
        "dataset_contract_sha256": contract["contract_sha256"],
        "cache_metadata_sha256": data_lineage["cache_metadata_sha256"],
        "mechanism_partition_sha256": current_partition_sha,
        "guardrails": decision_guardrails(),
        "H4": {
            "status": "PASS" if passed else "FAIL",
            "validation_incremental_skill": validation,
            "calibration_permutation_null_q95": null_q95,
            "wording": "predictive evidence only; no causal or mutual-information claim",
        },
        "mean_checkpoint": {
            "path": mean_checkpoint_path.as_posix(),
            "sha256": file_sha256(mean_checkpoint_path),
        },
        "next_stage_allowed": passed,
        "stop_rule": (
            "H4 passed; calibrated band evidence may enter the CT-head/router prerequisites."
            if passed
            else "Stop residual-band evidence routing; do not tune the null threshold on validation."
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
            "validation_used_for_probe_or_threshold_selection": False,
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
        "--h3-decision",
        type=Path,
        default=Path(
            "results/mechanism_validation/02_h3_recoverability_curves/decision.json"
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
        default=Path("results/mechanism_validation/03_h4_noise_calibration"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--null-replicates",
        type=int,
        default=DEFAULT_NULL_REPLICATES,
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
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
            "stage": "03_H4_noise_band_calibration",
            "decision": "FAIL",
            "dataset_contract_sha256": None,
            "guardrails": decision_guardrails(checkpoint_lineage="NOT_EVALUATED"),
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "next_stage_allowed": False,
            "stop_rule": "Stop H4 and all evidence-router stages.",
        }
        write_json(output / "decision.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
        return 2
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    return 0 if decision["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
