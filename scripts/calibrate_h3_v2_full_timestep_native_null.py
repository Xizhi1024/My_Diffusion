"""Calibrate the independent H3-v2 full-timestep native/null schedule.

This is an offline mechanism-development worker.  It may read PET and lesion
masks from the 99-patient mechanism-train and 25-patient calibration roles, but
it never reads the exposed validation role.  Its runtime artifact maps one
above-chance recoverability degree of freedom to
``[native, shallow, null] = [a, 0, 1-a]`` and defines every timestep 0..999
directly.  It neither restores H4-v2 nor authorizes production, H5, or H6.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_h2_pathology_excluded_residual import (  # noqa: E402
    IndexedDataset,
    _load_predictor,
    _loader,
)
from scripts.validate_h4_noise_band_calibration import (  # noqa: E402
    BANDS,
    _band_tensors,
    _masked_band_alignment,
)
from src.data.dataset import CachedDataset  # noqa: E402
from src.data.lineage import (  # noqa: E402
    REQUIRED_CHECKPOINT_LINEAGE_FIELDS,
    load_checkpoint_data_lineage,
    validate_checkpoint_data_lineage,
)
from src.mechanism_validation.common import (  # noqa: E402
    bootstrap_mean,
    canonical_json_sha256,
    file_sha256,
    load_json,
    partition_counts,
    partition_sha256,
    patient_partition,
    read_manifest,
    write_csv,
    write_json,
)
from src.model.noise.base import BBDMBridgeSchedule  # noqa: E402


SCHEMA_VERSION = 1
PIPELINE_ID = "H3_V2_FULL_TIMESTEP_NATIVE_NULL_CALIBRATION_V1"
ROLES = ("mechanism_train", "calibration")
ROUTE_ORDER = ("native", "shallow", "null")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _config_self_hash(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("config_sha256", None)
    return canonical_json_sha256(body)


def _schedule_self_hash(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("schedule_sha256", None)
    return canonical_json_sha256(body)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError("Expected a scalar array")
        value = value.reshape(()).item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _validate_protocol_config(
    root: Path,
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    config = _read_object(config_path)
    declared = str(config.get("config_sha256", ""))
    computed = _config_self_hash(config)
    if declared != computed:
        raise ValueError(
            f"H3-v2 config self-hash mismatch: declared={declared}, "
            f"computed={computed}"
        )
    if config.get("pipeline_id") != PIPELINE_ID:
        raise ValueError("Unexpected H3-v2 pipeline_id")
    if config.get("status") != (
        "PREREGISTERED_BEFORE_H3_V2_CALIBRATION_OR_MODEL_EVALUATION"
    ):
        raise ValueError("H3-v2 protocol is not in its preregistered state")
    if config.get("historical_results_preserved", {}).get(
        "must_not_overwrite_or_relabel"
    ) is not True:
        raise ValueError("H3-v2 protocol does not preserve historical failures")
    role = config.get("mechanism_role", {})
    if (
        not isinstance(role, Mapping)
        or role.get("stage") != "A1_availability_screen"
        or role.get("tested_component")
        != "full_timestep_band_conditioned_native_null_availability"
        or role.get("destination_component_status") != "NOT_EVALUATED"
        or role.get("full_ternary_router_claim_allowed") is not False
    ):
        raise ValueError(
            "H3-v2 must remain a Stage-A1 availability-only protocol"
        )
    if config.get("calibration", {}).get("validation_images_read") is not False:
        raise ValueError("H3-v2 calibration must exclude validation images")
    mapping = config.get("calibration", {}).get("mapping", {})
    if mapping.get("shallow_route_policy") != "structurally_zero":
        raise ValueError("H3-v2 protocol must structurally exclude shallow routing")
    claim_rules = config.get("stop_and_claim_rules", {})
    required_false = (
        "production_training",
        "production_activation",
        "h5_started",
        "h6_started",
        "full_ternary_router_claim",
        "destination_mechanism_claim",
        "artifact_safety_mechanism_claim",
        "h3_driven_curriculum_claim",
        "recoverability_schedule_specificity_claim",
        "next_production_stage_allowed",
    )
    if not isinstance(claim_rules, Mapping) or any(
        claim_rules.get(field) is not False for field in required_false
    ):
        raise ValueError("H3-v2 protocol claim boundary is not fail-closed")

    source_hashes: dict[str, str] = {}
    for label, spec in config.get("runtime_sources", {}).items():
        if not isinstance(spec, Mapping):
            raise ValueError(f"runtime_sources.{label} must be an object")
        source_path = _resolve(root, str(spec.get("path", "")))
        expected = str(spec.get("file_sha256", "")).lower()
        if len(expected) != 64:
            raise ValueError(f"runtime_sources.{label} is not frozen")
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        observed = file_sha256(source_path)
        if observed != expected:
            raise ValueError(
                f"runtime_sources.{label} SHA-256 mismatch: "
                f"expected={expected}, observed={observed}"
            )
        source_hashes[str(source_path.relative_to(root))] = observed
    return config, source_hashes


def _validate_mean_checkpoint(
    root: Path,
    config: Mapping[str, Any],
    *,
    lineage: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    spec = config["mean_checkpoint"]
    decision_path = _resolve(root, spec["decision_path"])
    if file_sha256(decision_path) != spec["decision_file_sha256"]:
        raise ValueError("V2-03 excluded-mean decision SHA-256 mismatch")
    decision = load_json(decision_path)
    if (
        decision.get("pipeline_id") != spec["decision_pipeline_id"]
        or decision.get("decision") != spec["decision_required"]
    ):
        raise ValueError("V2-03 excluded-mean decision is not the frozen PASS")

    checkpoint_path = _resolve(root, spec["path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    observed_checkpoint_hash = file_sha256(checkpoint_path)
    if observed_checkpoint_hash != spec["file_sha256"]:
        raise ValueError("V2-03 excluded-mean checkpoint SHA-256 mismatch")

    sidecar_path = _resolve(root, spec["sidecar_path"])
    sidecar = _read_object(sidecar_path)
    sidecar_body = dict(sidecar)
    declared_fingerprint = str(sidecar_body.pop("fingerprint_sha256", ""))
    if declared_fingerprint != canonical_json_sha256(sidecar_body):
        raise ValueError("V2-03 excluded-mean sidecar self-hash mismatch")
    if declared_fingerprint != spec["sidecar_fingerprint_sha256"]:
        raise ValueError("V2-03 excluded-mean sidecar fingerprint mismatch")
    if sidecar.get("checkpoint_sha256") != observed_checkpoint_hash:
        raise ValueError("V2-03 sidecar and checkpoint hashes differ")
    exclusion = sidecar.get("pathology_exclusion", {})
    if (
        exclusion.get("enabled") is not True
        or int(exclusion.get("guard_radius_px", -1)) != int(spec["guard_radius_px"])
    ):
        raise ValueError("V2-03 checkpoint pathology-exclusion policy mismatch")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("V2-03 checkpoint must contain an object")
    validate_checkpoint_data_lineage(
        checkpoint,
        lineage,
        required=True,
        context="H3-v2 excluded-mean checkpoint",
    )
    return checkpoint_path, checkpoint, sidecar


def _mechanism_datasets(
    *,
    cache_dir: Path,
    manifest_path: Path,
    partition: Mapping[str, str],
) -> tuple[IndexedDataset, list[str]]:
    base = CachedDataset(
        cache_dir,
        split="train",
        augment=False,
        split_manifest=manifest_path,
        required_keys=["ct", "pet", "mask"],
    )
    indices = [
        index
        for index, entry in enumerate(base.entries)
        if partition.get(entry.patient_id) in ROLES
    ]
    dataset = IndexedDataset(base, indices)
    roles = [str(partition[entry.patient_id]) for entry in dataset.entries]
    if any(role not in ROLES for role in roles):
        raise ValueError("Unexpected role in H3-v2 development dataset")
    return dataset, roles


def _noise_seed(analysis_seed: int, role: str, sample_id: str) -> int:
    digest = hashlib.sha256(
        f"{analysis_seed}|{role}|{sample_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _fixed_noise(
    shape: Sequence[int],
    *,
    analysis_seed: int,
    role: str,
    sample_id: str,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_noise_seed(analysis_seed, role, sample_id))
    return torch.randn(tuple(shape), generator=generator, dtype=torch.float32)


def _validate_prepared_cache(
    path: Path,
    expected_identity: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        identity = json.loads(str(_json_scalar(payload["identity_json"])))
        if identity != dict(expected_identity):
            raise ValueError("Prepared H3-v2 input cache identity mismatch")
        required = {
            "residual",
            "mask",
            "noise",
            "sample_ids",
            "patient_ids",
            "roles",
        }
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"Prepared H3-v2 cache is incomplete: {missing}")
        arrays = {name: np.asarray(payload[name]) for name in required}
    count = len(arrays["sample_ids"])
    if arrays["residual"].shape != arrays["noise"].shape:
        raise ValueError("Prepared residual/noise shapes differ")
    if arrays["residual"].shape != arrays["mask"].shape:
        raise ValueError("Prepared residual/mask shapes differ")
    if arrays["residual"].shape[0] != count:
        raise ValueError("Prepared H3-v2 sample dimension mismatch")
    if not np.isfinite(arrays["residual"]).all():
        raise ValueError("Prepared H3-v2 residual contains non-finite values")
    if not np.isfinite(arrays["noise"]).all():
        raise ValueError("Prepared H3-v2 noise contains non-finite values")
    return arrays


@torch.no_grad()
def _prepare_inputs(
    *,
    output_dir: Path,
    dataset: IndexedDataset,
    roles: Sequence[str],
    model: torch.nn.Module,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    identity: Mapping[str, Any],
    analysis_seed: int,
) -> tuple[Path, dict[str, np.ndarray]]:
    destination = output_dir / "prepared" / "development_residual_inputs.npz"
    if destination.exists():
        return destination, _validate_prepared_cache(destination, identity)

    residual_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    noise_rows: list[np.ndarray] = []
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
        mask = (batch["mask"] > 0.5).to(torch.uint8)
        residual = (pet - model(ct)["mean_pet"]).float().cpu()
        batch_actual = int(residual.shape[0])
        current_entries = dataset.entries[offset : offset + batch_actual]
        current_roles = roles[offset : offset + batch_actual]
        fixed = torch.stack(
            [
                _fixed_noise(
                    residual[index].shape,
                    analysis_seed=analysis_seed,
                    role=current_roles[index],
                    sample_id=current_entries[index].sample_id,
                )
                for index in range(batch_actual)
            ],
            dim=0,
        )
        residual_rows.append(residual.numpy())
        mask_rows.append(mask.numpy())
        noise_rows.append(fixed.numpy())
        offset += batch_actual
        print(
            f"prepared batch {batch_index + 1}: {offset}/{len(dataset)} samples",
            flush=True,
        )
    if offset != len(dataset):
        raise RuntimeError("Prepared H3-v2 cache sample count mismatch")

    arrays = {
        "residual": np.concatenate(residual_rows, axis=0).astype(np.float32),
        "mask": np.concatenate(mask_rows, axis=0).astype(np.uint8),
        "noise": np.concatenate(noise_rows, axis=0).astype(np.float32),
        "sample_ids": np.asarray(
            [entry.sample_id for entry in dataset.entries], dtype="U64"
        ),
        "patient_ids": np.asarray(
            [entry.patient_id for entry in dataset.entries], dtype="U64"
        ),
        "roles": np.asarray(list(roles), dtype="U32"),
    }
    _atomic_npz(
        destination,
        identity_json=np.asarray(
            json.dumps(dict(identity), sort_keys=True, separators=(",", ":"))
        ),
        **arrays,
    )
    return destination, _validate_prepared_cache(destination, identity)


def _shard_path(output_dir: Path, start: int, stop: int) -> Path:
    return output_dir / "shards" / f"t{start:04d}-{stop - 1:04d}.npz"


def _validate_shard(
    path: Path,
    *,
    identity: Mapping[str, Any],
    patient_ids: Sequence[str],
    patient_roles: Sequence[str],
    start: int,
    stop: int,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        observed_identity = json.loads(str(_json_scalar(payload["identity_json"])))
        if observed_identity != dict(identity):
            raise ValueError(f"H3-v2 shard identity mismatch: {path}")
        if int(_json_scalar(payload["start"])) != start:
            raise ValueError(f"H3-v2 shard start mismatch: {path}")
        if int(_json_scalar(payload["stop"])) != stop:
            raise ValueError(f"H3-v2 shard stop mismatch: {path}")
        observed_patients = [str(value) for value in payload["patient_ids"].tolist()]
        observed_roles = [str(value) for value in payload["patient_roles"].tolist()]
        recoverability = np.asarray(payload["recoverability"], dtype=np.float64)
    expected_shape = (len(patient_ids), len(BANDS), stop - start)
    if observed_patients != list(patient_ids):
        raise ValueError(f"H3-v2 shard patient order mismatch: {path}")
    if observed_roles != list(patient_roles):
        raise ValueError(f"H3-v2 shard patient roles mismatch: {path}")
    if recoverability.shape != expected_shape:
        raise ValueError(
            f"H3-v2 shard shape mismatch: {recoverability.shape} != {expected_shape}"
        )
    if not np.isfinite(recoverability).all():
        raise ValueError(f"H3-v2 shard contains non-finite values: {path}")
    if np.any((recoverability < 0.0) | (recoverability > 1.0)):
        raise ValueError(f"H3-v2 shard leaves [0,1]: {path}")
    return recoverability


@torch.no_grad()
def _compute_shard(
    *,
    path: Path,
    arrays: Mapping[str, np.ndarray],
    patient_ids: Sequence[str],
    patient_roles: Sequence[str],
    patient_index: Mapping[str, int],
    patient_sample_counts: np.ndarray,
    start: int,
    stop: int,
    batch_size: int,
    timestep_chunk_size: int,
    schedule: BBDMBridgeSchedule,
    device: torch.device,
    identity: Mapping[str, Any],
) -> np.ndarray:
    if path.exists():
        return _validate_shard(
            path,
            identity=identity,
            patient_ids=patient_ids,
            patient_roles=patient_roles,
            start=start,
            stop=stop,
        )
    sums = np.zeros(
        (len(patient_ids), len(BANDS), stop - start),
        dtype=np.float64,
    )
    total_samples = int(arrays["residual"].shape[0])
    for batch_start in range(0, total_samples, batch_size):
        batch_stop = min(batch_start + batch_size, total_samples)
        residual = torch.from_numpy(
            arrays["residual"][batch_start:batch_stop]
        ).to(device)
        fixed_noise = torch.from_numpy(
            arrays["noise"][batch_start:batch_stop]
        ).to(device)
        mask = torch.from_numpy(
            arrays["mask"][batch_start:batch_stop]
        ).to(device=device, dtype=torch.float32)
        clean_bands = _band_tensors(residual)
        mask_l1 = F.max_pool2d(mask, kernel_size=2, stride=2)
        mask_l2 = F.max_pool2d(mask, kernel_size=4, stride=4)
        masks = {
            band: mask_l2 if band.startswith("l2_") else mask_l1
            for band in BANDS
        }
        current_patient_indices = np.asarray(
            [
                patient_index[str(value)]
                for value in arrays["patient_ids"][batch_start:batch_stop]
            ],
            dtype=np.int64,
        )
        current_batch = batch_stop - batch_start
        for chunk_start in range(start, stop, timestep_chunk_size):
            chunk_stop = min(chunk_start + timestep_chunk_size, stop)
            timestep_values = torch.arange(
                chunk_start,
                chunk_stop,
                dtype=torch.long,
                device=device,
            )
            chunk_count = int(timestep_values.numel())
            m = schedule.m_t[timestep_values].to(
                device=device, dtype=residual.dtype
            )
            sigma = schedule.sigma_t[timestep_values].to(
                device=device, dtype=residual.dtype
            )
            noisy = (
                (1.0 - m)[:, None, None, None, None]
                * residual[None]
                + sigma[:, None, None, None, None] * fixed_noise[None]
            )
            noisy_flat = noisy.reshape(
                chunk_count * current_batch,
                *residual.shape[1:],
            )
            noisy_bands = _band_tensors(noisy_flat)
            for band_index, band in enumerate(BANDS):
                clean = (
                    clean_bands[band][None]
                    .expand(chunk_count, -1, -1, -1, -1)
                    .reshape(chunk_count * current_batch, 1, *clean_bands[band].shape[-2:])
                )
                current_mask = (
                    masks[band][None]
                    .expand(chunk_count, -1, -1, -1, -1)
                    .reshape(chunk_count * current_batch, 1, *masks[band].shape[-2:])
                )
                values = _masked_band_alignment(
                    noisy_bands[band],
                    clean,
                    current_mask,
                ).reshape(chunk_count, current_batch)
                values_np = values.transpose(0, 1).float().cpu().numpy()
                destination = sums[
                    :,
                    band_index,
                    chunk_start - start : chunk_stop - start,
                ]
                np.add.at(destination, current_patient_indices, values_np)
        print(
            f"shard {start:04d}-{stop - 1:04d}: "
            f"{batch_stop}/{total_samples} samples",
            flush=True,
        )
    recoverability = sums / patient_sample_counts[:, None, None]
    _atomic_npz(
        path,
        identity_json=np.asarray(
            json.dumps(dict(identity), sort_keys=True, separators=(",", ":"))
        ),
        start=np.asarray(start, dtype=np.int64),
        stop=np.asarray(stop, dtype=np.int64),
        patient_ids=np.asarray(patient_ids, dtype="U64"),
        patient_roles=np.asarray(patient_roles, dtype="U32"),
        recoverability=recoverability.astype(np.float32),
    )
    return _validate_shard(
        path,
        identity=identity,
        patient_ids=patient_ids,
        patient_roles=patient_roles,
        start=start,
        stop=stop,
    )


def _bootstrap_contrast(
    differences: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    result = bootstrap_mean(
        differences,
        seed=seed,
        replicates=replicates,
    )
    result["contrast"] = (
        "h3_v2_schedule_mse_minus_mechanism_train_band_constant_mse"
    )
    return result


def _fit_and_gate(
    *,
    patient_recoverability: np.ndarray,
    patient_ids: Sequence[str],
    patient_roles: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[dict[str, list[float]], dict[str, Any], list[dict[str, Any]]]:
    train_indices = np.asarray(
        [index for index, role in enumerate(patient_roles) if role == "mechanism_train"],
        dtype=np.int64,
    )
    calibration_indices = np.asarray(
        [index for index, role in enumerate(patient_roles) if role == "calibration"],
        dtype=np.int64,
    )
    gates = config["calibration_gates"]
    if len(train_indices) != int(gates["required_mechanism_train_patients"]):
        raise ValueError("H3-v2 mechanism-train patient count mismatch")
    if len(calibration_indices) != int(gates["required_calibration_patients"]):
        raise ValueError("H3-v2 calibration patient count mismatch")

    train_mean = patient_recoverability[train_indices].mean(axis=0)
    raw_active = np.clip((train_mean - 0.5) / 0.5, 0.0, 1.0)
    timesteps = np.arange(train_mean.shape[1], dtype=np.float64)
    frozen_active = np.empty_like(raw_active)
    for band_index in range(len(BANDS)):
        frozen_active[band_index] = IsotonicRegression(
            increasing=False,
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        ).fit_transform(timesteps, raw_active[band_index])
    frozen_active = np.clip(frozen_active, 0.0, 1.0)

    band_constant = train_mean.mean(axis=1)
    schedule_prediction = 0.5 + 0.5 * frozen_active
    calibration = patient_recoverability[calibration_indices]
    schedule_mse = np.mean(
        (calibration - schedule_prediction[None]) ** 2,
        axis=(1, 2),
    )
    constant_mse = np.mean(
        (calibration - band_constant[None, :, None]) ** 2,
        axis=(1, 2),
    )
    differences = schedule_mse - constant_mse
    comparison_spec = gates["calibration_schedule_vs_band_constant"]
    comparison = _bootstrap_contrast(
        differences,
        seed=int(comparison_spec["bootstrap_seed"]),
        replicates=int(comparison_spec["bootstrap_replicates"]),
    )
    improved_fraction = float(np.mean(differences < 0.0))
    comparison["improved_patient_fraction"] = improved_fraction
    comparison["patient_rows"] = [
        {
            "patient_id": patient_ids[int(patient_index)],
            "h3_v2_schedule_mse": float(schedule_mse[row_index]),
            "band_constant_mse": float(constant_mse[row_index]),
            "difference": float(differences[row_index]),
        }
        for row_index, patient_index in enumerate(calibration_indices)
    ]

    finite = bool(np.isfinite(patient_recoverability).all())
    in_range = bool(
        np.all(
            (patient_recoverability >= 0.0)
            & (patient_recoverability <= 1.0)
        )
    )
    monotonic = bool(np.all(np.diff(frozen_active, axis=1) <= 1e-8))
    t0_min = float(frozen_active[:, 0].min())
    t999_max = float(frozen_active[:, -1].max())
    checks = {
        "mechanism_train_patient_count": len(train_indices)
        == int(gates["required_mechanism_train_patients"]),
        "calibration_patient_count": len(calibration_indices)
        == int(gates["required_calibration_patients"]),
        "band_count": patient_recoverability.shape[1]
        == int(gates["required_band_count"]),
        "timestep_count": patient_recoverability.shape[2]
        == int(gates["required_timestep_count"]),
        "all_values_finite": finite,
        "all_values_in_range": in_range,
        "all_frozen_curves_non_increasing": monotonic,
        "minimum_active_mass_at_t0": t0_min
        >= float(gates["minimum_active_mass_at_t0"]),
        "maximum_active_mass_at_t999": t999_max
        <= float(gates["maximum_active_mass_at_t999"]),
        "calibration_bootstrap_ci95_high_below_zero": float(
            comparison["ci95_high"]
        )
        < float(comparison_spec["required_bootstrap_ci95_high_below"]),
        "calibration_improved_patient_fraction": improved_fraction
        >= float(comparison_spec["minimum_improved_patient_fraction"]),
    }
    gate = {
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "active_mass_at_t0_minimum": t0_min,
        "active_mass_at_t999_maximum": t999_max,
        "calibration_schedule_vs_band_constant": comparison,
    }
    schedule_values = {
        band: [float(value) for value in frozen_active[index].tolist()]
        for index, band in enumerate(BANDS)
    }
    summary_rows: list[dict[str, Any]] = []
    calibration_mean = calibration.mean(axis=0)
    for band_index, band in enumerate(BANDS):
        for timestep in range(patient_recoverability.shape[2]):
            summary_rows.append(
                {
                    "band": band,
                    "timestep": timestep,
                    "mechanism_train_mean_recoverability": float(
                        train_mean[band_index, timestep]
                    ),
                    "calibration_mean_recoverability": float(
                        calibration_mean[band_index, timestep]
                    ),
                    "raw_above_chance_active_mass": float(
                        raw_active[band_index, timestep]
                    ),
                    "frozen_isotonic_active_mass": float(
                        frozen_active[band_index, timestep]
                    ),
                    "native_probability": float(
                        frozen_active[band_index, timestep]
                    ),
                    "shallow_probability": 0.0,
                    "null_probability": float(
                        1.0 - frozen_active[band_index, timestep]
                    ),
                }
            )
    return schedule_values, gate, summary_rows


def _existing_complete_decision(
    output_dir: Path,
    *,
    config_sha256: str,
) -> dict[str, Any] | None:
    decision_path = output_dir / "decision.json"
    if not decision_path.exists():
        return None
    decision = _read_object(decision_path)
    if (
        decision.get("pipeline_id") != PIPELINE_ID
        or decision.get("config_sha256") != config_sha256
        or decision.get("calibration_complete") is not True
    ):
        raise ValueError("Existing H3-v2 decision is not reusable")
    decision_body = dict(decision)
    declared_decision_hash = str(
        decision_body.pop("decision_sha256", "")
    )
    if declared_decision_hash != canonical_json_sha256(decision_body):
        raise ValueError("Existing H3-v2 decision self-hash mismatch")
    schedule_path = output_dir / "frozen_schedule.json"
    if decision.get("decision") == "PASS":
        if not schedule_path.is_file():
            raise ValueError("Existing PASS decision lacks frozen schedule")
        if file_sha256(schedule_path) != decision.get("schedule_file_sha256"):
            raise ValueError("Existing H3-v2 schedule hash mismatch")
    elif schedule_path.exists():
        raise ValueError("Existing FAIL decision unexpectedly has a schedule")
    return decision


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    output_dir = _resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = _resolve(root, args.config)
    config, source_hashes = _validate_protocol_config(root, config_path)
    existing = _existing_complete_decision(
        output_dir,
        config_sha256=config["config_sha256"],
    )
    if existing is not None:
        return existing

    dataset_spec = config["dataset_contract"]
    data_config = {
        "data": {
            "cache_dir": str(_resolve(root, dataset_spec["cache_dir"])),
            "cache_lineage": str(
                _resolve(root, dataset_spec["cache_lineage_path"])
            ),
            "dataset_contract": str(
                _resolve(root, dataset_spec["dataset_contract_path"])
            ),
            "require_cache_lineage": True,
            "use_fake_data": False,
        }
    }
    lineage = load_checkpoint_data_lineage(data_config, root=root)
    if lineage is None:
        raise ValueError("H3-v2 requires sealed cache lineage")
    for field in REQUIRED_CHECKPOINT_LINEAGE_FIELDS:
        expected = dataset_spec.get(field)
        if expected is not None and lineage.get(field) != expected:
            raise ValueError(f"H3-v2 data lineage mismatch for {field}")

    manifest_path = _resolve(root, dataset_spec["manifest_path"])
    if file_sha256(manifest_path) != dataset_spec["manifest_file_sha256"]:
        raise ValueError("H3-v2 authoritative manifest file SHA-256 mismatch")
    manifest = read_manifest(manifest_path)
    partition = patient_partition(manifest)
    current_partition_hash = partition_sha256(partition)
    if current_partition_hash != dataset_spec["mechanism_partition_sha256"]:
        raise ValueError("H3-v2 patient partition fingerprint mismatch")
    counts = partition_counts(manifest, partition)
    expected_counts = {
        "mechanism_train": (
            dataset_spec["mechanism_train_patients"],
            766,
        ),
        "calibration": (dataset_spec["calibration_patients"], 188),
        "validation": (dataset_spec["validation_patients"], 237),
    }
    for role, (patients, samples) in expected_counts.items():
        if counts[role] != {"patients": patients, "samples": samples}:
            raise ValueError(f"H3-v2 partition counts differ for {role}")

    checkpoint_path, checkpoint, sidecar = _validate_mean_checkpoint(
        root,
        config,
        lineage=lineage,
    )
    calibration_spec = config["calibration"]
    num_train_timesteps = int(calibration_spec["num_train_timesteps"])
    batch_size = int(args.batch_size or calibration_spec["batch_size"])
    timestep_chunk_size = int(
        args.timestep_chunk_size or calibration_spec["timestep_chunk_size"]
    )
    num_workers = int(
        calibration_spec["num_workers"]
        if args.num_workers is None
        else args.num_workers
    )
    if batch_size != int(calibration_spec["batch_size"]):
        raise ValueError("H3-v2 batch size is frozen by the protocol")
    if timestep_chunk_size != int(calibration_spec["timestep_chunk_size"]):
        raise ValueError("H3-v2 timestep chunk size is frozen by the protocol")
    if args.shard_size <= 0 or num_train_timesteps % args.shard_size != 0:
        raise ValueError("H3-v2 shard size must divide 1000 exactly")
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("Formal H3-v2 calibration requires CUDA")

    dataset, sample_roles = _mechanism_datasets(
        cache_dir=_resolve(root, dataset_spec["cache_dir"]),
        manifest_path=manifest_path,
        partition=partition,
    )
    role_counts = {
        role: len(
            {
                entry.patient_id
                for entry, current_role in zip(dataset.entries, sample_roles)
                if current_role == role
            }
        )
        for role in ROLES
    }
    if role_counts != {
        "mechanism_train": dataset_spec["mechanism_train_patients"],
        "calibration": dataset_spec["calibration_patients"],
    }:
        raise ValueError("H3-v2 development dataset patient counts mismatch")
    if any(entry.patient_id in {"044", "080", "153"} for entry in dataset.entries):
        raise ValueError("H3-v2 calibration unexpectedly read exposed validation")

    model = _load_predictor(checkpoint, device=device)
    identity = {
        "schema_version": 1,
        "pipeline_id": PIPELINE_ID,
        "config_sha256": config["config_sha256"],
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "mechanism_partition_sha256": current_partition_hash,
        "analysis_seed": int(calibration_spec["analysis_seed"]),
        "batch_size": batch_size,
        "timestep_chunk_size": timestep_chunk_size,
        "noise_identity": calibration_spec["noise_identity"],
    }
    prepared_path, arrays = _prepare_inputs(
        output_dir=output_dir,
        dataset=dataset,
        roles=sample_roles,
        model=model,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        identity=identity,
        analysis_seed=int(calibration_spec["analysis_seed"]),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    patient_ids = sorted(set(str(value) for value in arrays["patient_ids"]))
    patient_index = {patient: index for index, patient in enumerate(patient_ids)}
    patient_roles = []
    patient_sample_counts = np.zeros(len(patient_ids), dtype=np.float64)
    for patient in patient_ids:
        rows = np.flatnonzero(arrays["patient_ids"] == patient)
        roles = sorted(set(str(value) for value in arrays["roles"][rows]))
        if len(roles) != 1:
            raise ValueError(f"H3-v2 patient {patient} has multiple roles")
        patient_roles.append(roles[0])
        patient_sample_counts[patient_index[patient]] = len(rows)
    if np.any(patient_sample_counts <= 0):
        raise ValueError("H3-v2 patient has no samples")

    schedule = BBDMBridgeSchedule(
        num_train_timesteps=num_train_timesteps,
        m_schedule=str(calibration_spec["m_schedule"]),
        sigma_scale=float(calibration_spec["sigma_scale"]),
    ).to(device)
    shard_arrays: list[np.ndarray] = []
    for start in range(0, num_train_timesteps, args.shard_size):
        stop = start + args.shard_size
        shard_arrays.append(
            _compute_shard(
                path=_shard_path(output_dir, start, stop),
                arrays=arrays,
                patient_ids=patient_ids,
                patient_roles=patient_roles,
                patient_index=patient_index,
                patient_sample_counts=patient_sample_counts,
                start=start,
                stop=stop,
                batch_size=batch_size,
                timestep_chunk_size=timestep_chunk_size,
                schedule=schedule,
                device=device,
                identity=identity,
            )
        )
    patient_recoverability = np.concatenate(shard_arrays, axis=2)
    if patient_recoverability.shape != (
        len(patient_ids),
        len(BANDS),
        num_train_timesteps,
    ):
        raise ValueError("H3-v2 combined patient tensor has the wrong shape")
    _atomic_npz(
        output_dir / "patient_recoverability.npz",
        patient_ids=np.asarray(patient_ids, dtype="U64"),
        patient_roles=np.asarray(patient_roles, dtype="U32"),
        band_order=np.asarray(BANDS, dtype="U16"),
        timesteps=np.arange(num_train_timesteps, dtype=np.int64),
        recoverability=patient_recoverability.astype(np.float32),
    )

    schedule_values, gate, summary_rows = _fit_and_gate(
        patient_recoverability=patient_recoverability,
        patient_ids=patient_ids,
        patient_roles=patient_roles,
        config=config,
    )
    write_csv(output_dir / "full_timestep_schedule_summary.csv", summary_rows)
    write_csv(
        output_dir / "calibration_patient_errors.csv",
        gate["calibration_schedule_vs_band_constant"]["patient_rows"],
    )
    passed = gate["decision"] == "PASS"
    schedule_file_hash = None
    schedule_self_hash = None
    if passed:
        schedule_payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "stage": config["stage"],
            "pipeline_id": PIPELINE_ID,
            "decision": "PASS",
            "mechanism_component": "availability",
            "destination_mechanism_evaluated": False,
            "artifact_safety_mechanism_evaluated": False,
            "h3_driven_curriculum_evaluated": False,
            "full_ternary_router_claim_allowed": False,
            "config_sha256": config["config_sha256"],
            "inference_schedule_allowed": True,
            "production_activation_allowed": False,
            "h5_h6_allowed": False,
            "num_train_timesteps": num_train_timesteps,
            "band_order": list(BANDS),
            "route_order": list(ROUTE_ORDER),
            "native_active_mass": schedule_values,
            "mapping": "[active_mass,0,1-active_mass]",
            "shallow_route_policy": "structurally_zero",
            "lookup_policy": "exact_integer_timestep_only",
            "forbidden_runtime_inputs": list(
                config["runtime_contract"]["forbidden_runtime_inputs"]
            ),
            "source_partitions": [
                "mechanism_train_fit",
                "calibration_gate",
            ],
            "validation_images_read": False,
            "created_at_utc": utc_now(),
        }
        schedule_payload["schedule_sha256"] = _schedule_self_hash(
            schedule_payload
        )
        schedule_path = output_dir / "frozen_schedule.json"
        if schedule_path.exists():
            # A power loss can occur after the atomically-written schedule and
            # before decision.json.  Resume may reuse only that exact,
            # self-sealed orphan; it never overwrites or silently adopts a
            # different table.
            existing_schedule = _read_object(schedule_path)
            declared_self_hash = str(
                existing_schedule.get("schedule_sha256", "")
            )
            if declared_self_hash != _schedule_self_hash(existing_schedule):
                raise ValueError(
                    "Existing orphan H3-v2 schedule has an invalid self-hash"
                )
            comparable_existing = dict(existing_schedule)
            comparable_expected = dict(schedule_payload)
            comparable_existing.pop("created_at_utc", None)
            comparable_expected.pop("created_at_utc", None)
            if comparable_existing != comparable_expected:
                raise ValueError(
                    "Existing orphan H3-v2 schedule differs from the "
                    "deterministically recomputed schedule"
                )
            schedule_payload = existing_schedule
        else:
            write_json(schedule_path, schedule_payload)
        schedule_file_hash = file_sha256(schedule_path)
        schedule_self_hash = schedule_payload["schedule_sha256"]
    elif (output_dir / "frozen_schedule.json").exists():
        raise ValueError(
            "Calibration recomputed FAIL while an orphan PASS schedule exists"
        )

    decision = {
        "schema_version": SCHEMA_VERSION,
        "stage": config["stage"],
        "pipeline_id": PIPELINE_ID,
        "decision": "PASS" if passed else "FAIL",
        "mechanism_component": "availability",
        "stage_A1_calibration_complete": True,
        "destination_mechanism_evaluated": False,
        "artifact_safety_mechanism_evaluated": False,
        "h3_driven_curriculum_evaluated": False,
        "recoverability_schedule_specificity_controls_complete": False,
        "full_ternary_router_claim_allowed": False,
        "calibration_complete": True,
        "config_path": str(config_path),
        "config_sha256": config["config_sha256"],
        "runtime_source_hashes": source_hashes,
        "dataset_lineage": {
            field: lineage[field] for field in REQUIRED_CHECKPOINT_LINEAGE_FIELDS
        },
        "mechanism_partition_sha256": current_partition_hash,
        "partition_counts": counts,
        "validation_images_read": False,
        "mean_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "sidecar_fingerprint_sha256": sidecar["fingerprint_sha256"],
            "production_cutover_claimed": False,
        },
        "prepared_inputs": {
            "path": str(prepared_path),
            "sha256": file_sha256(prepared_path),
            "samples": int(arrays["residual"].shape[0]),
        },
        "direct_grid": {
            "minimum": 0,
            "maximum": num_train_timesteps - 1,
            "count": num_train_timesteps,
            "bands": len(BANDS),
            "cells": num_train_timesteps * len(BANDS),
            "interpolation_used": False,
            "nearest_neighbor_used": False,
            "global_mean_fallback_used": False,
        },
        "mapping": {
            "route_order": list(ROUTE_ORDER),
            "formula": "[active_mass,0,1-active_mass]",
            "shallow_route_structurally_zero": True,
            "scalar_degrees_of_freedom": 1,
            "route_degrees_of_freedom_used": 1,
        },
        "gate": gate,
        "schedule_path": (
            str(output_dir / "frozen_schedule.json") if passed else None
        ),
        "schedule_file_sha256": schedule_file_hash,
        "schedule_sha256": schedule_self_hash,
        "model_experiment_allowed": passed,
        "production_training_allowed": False,
        "production_activation_allowed": False,
        "h5_h6_allowed": False,
        "next_production_stage_allowed": False,
        "stop_rule": (
            "Calibration PASS permits only the isolated no-route versus "
            "H3-v2 internal exploratory experiment."
            if passed
            else "Stop before any H3-v2 model training."
        ),
        "created_at_utc": utc_now(),
    }
    decision["decision_sha256"] = canonical_json_sha256(decision)
    write_json(output_dir / "decision.json", decision)
    return decision


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/h3_v2_full_timestep_native_null_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--timestep-chunk-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=25)
    parser.add_argument(
        "--allow-scientific-fail-exit-zero",
        action="store_true",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.root = args.root.resolve()
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        decision = run(args)
    except Exception as exc:
        output_dir = _resolve(args.root, args.output_dir)
        failure = {
            "schema_version": SCHEMA_VERSION,
            "stage": "05C_h3_v2_full_timestep_native_null",
            "pipeline_id": PIPELINE_ID,
            "decision": "FAIL",
            "mechanism_component": "availability",
            "destination_mechanism_evaluated": False,
            "artifact_safety_mechanism_evaluated": False,
            "h3_driven_curriculum_evaluated": False,
            "full_ternary_router_claim_allowed": False,
            "calibration_complete": False,
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "model_experiment_allowed": False,
            "production_training_allowed": False,
            "production_activation_allowed": False,
            "h5_h6_allowed": False,
            "created_at_utc": utc_now(),
        }
        failure["decision_sha256"] = canonical_json_sha256(failure)
        write_json(output_dir / "technical_failure.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
        return 2
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    if decision["decision"] == "PASS":
        return 0
    return 0 if args.allow_scientific_fail_exit_zero else 3


if __name__ == "__main__":
    raise SystemExit(main())
