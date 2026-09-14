"""Fail-closed loading for formal and direct-PNG prior-anchor schedules.

The formal H3-v2 contract remains owned by
``h3_native_null_schedule.load_h3_native_null_schedule``.  This module only
adds a narrow dispatcher and a separately labelled loader for local
direct-PNG previews.  Preview consumption is deliberately opt-in: its source
checkpoint/cache lineage is not established for the current runtime even
when the file hash and payload schema are valid.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

import torch

from .h3_native_null_schedule import (
    H3_V2_BAND_ORDER,
    H3_V2_ROUTE_ORDER,
    load_h3_native_null_schedule,
)


FORMAL_H3_V2 = "formal_h3_v2"
DIRECT_PNG_PREVIEW = "direct_png_preview"
PRIOR_ANCHOR_SCHEDULE_SOURCES = frozenset(
    {FORMAL_H3_V2, DIRECT_PNG_PREVIEW}
)

DIRECT_PNG_PREVIEW_SCHEMA_VERSION = 1
DIRECT_PNG_PREVIEW_PIPELINE_ID = "H3_DIRECT_PNG_PRIOR_PREVIEW_V1"
DIRECT_PNG_PREVIEW_PATH_POLICY = (
    "paths are serialized repo-relative with POSIX separators; "
    "resolved machine paths are never persisted"
)
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
_MONOTONIC_TOLERANCE = 1e-7


def _file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: object, *, field: str) -> str:
    digest = str(value)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{field} must be a full SHA-256 hex digest")
    return digest.lower()


def _mapping(container: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Direct-PNG preview {key} must be an object")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _resolve_preview_path(
    path: str | Path,
    *,
    repository_root: str | Path | None,
) -> tuple[Path, Path, str]:
    root_declared = Path.cwd() if repository_root is None else Path(repository_root)
    try:
        root = root_declared.resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"repository_root cannot be resolved: {root_declared}"
        ) from exc
    if not root.is_dir():
        raise ValueError(f"repository_root must be a directory: {root_declared}")

    declared = Path(path)
    candidate = declared if declared.is_absolute() else root / declared
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Direct-PNG prior preview not found: {declared}"
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Direct-PNG prior preview is not a file: {declared}"
        )
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Direct-PNG prior preview must remain within repository_root"
        ) from exc
    return resolved, root, relative.as_posix()


def _portable_path(
    value: object,
    *,
    field: str,
    repository_root: Path,
    allow_none: bool = False,
) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty repo-relative path")
    if "\\" in value:
        raise ValueError(f"{field} must use POSIX separators")
    if "://" in value:
        raise ValueError(f"{field} must not be a URI")

    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    raw_parts = value.split("/")
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise ValueError(
            f"{field} must be a normalized repository-relative path"
        )

    resolved = repository_root.joinpath(*posix.parts).resolve(strict=False)
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes repository_root") from exc
    return posix.as_posix()


def _validate_preview_paths(
    payload: Mapping[str, Any],
    *,
    repository_root: Path,
    preview_relative_path: str,
) -> dict[str, str | None]:
    if payload.get("path_policy") != DIRECT_PNG_PREVIEW_PATH_POLICY:
        raise ValueError("Direct-PNG preview path_policy mismatch")

    paths = _mapping(payload, "paths")
    required_paths = {
        "png_root",
        "manifest",
        "dataset_contract",
        "mean_checkpoint",
        "output_dir",
    }
    missing_paths = required_paths.difference(paths)
    if missing_paths:
        raise ValueError(
            "Direct-PNG preview paths omit required entries: "
            f"{sorted(missing_paths)}"
        )
    validated_paths: dict[str, str | None] = {}
    for key, value in paths.items():
        validated_paths[f"paths.{key}"] = _portable_path(
            value,
            field=f"paths.{key}",
            repository_root=repository_root,
        )

    outputs = _mapping(payload, "outputs")
    required_outputs = {"curve_csv", "patient_npz", "plot_png"}
    missing_outputs = required_outputs.difference(outputs)
    if missing_outputs:
        raise ValueError(
            "Direct-PNG preview outputs omit required entries: "
            f"{sorted(missing_outputs)}"
        )
    for key, value in outputs.items():
        validated_paths[f"outputs.{key}"] = _portable_path(
            value,
            field=f"outputs.{key}",
            repository_root=repository_root,
            allow_none=(key == "plot_png"),
        )

    output_dir = PurePosixPath(str(paths["output_dir"]))
    preview_parent = PurePosixPath(preview_relative_path).parent
    if preview_parent != output_dir:
        raise ValueError(
            "Direct-PNG preview file must be stored in paths.output_dir"
        )
    for key in required_outputs:
        value = validated_paths[f"outputs.{key}"]
        if value is None:
            continue
        try:
            PurePosixPath(value).relative_to(output_dir)
        except ValueError as exc:
            raise ValueError(
                f"outputs.{key} must remain within paths.output_dir"
            ) from exc
    return validated_paths


def _validate_preview_flags(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != DIRECT_PNG_PREVIEW_SCHEMA_VERSION:
        raise ValueError("Direct-PNG preview schema_version must be 1")
    if payload.get("pipeline_id") != DIRECT_PNG_PREVIEW_PIPELINE_ID:
        raise ValueError("Direct-PNG preview pipeline_id mismatch")
    if payload.get("decision") != "PREVIEW_ONLY":
        raise ValueError("Direct-PNG preview decision must be PREVIEW_ONLY")
    required_flags = {
        "preview_only": True,
        "inference_schedule_allowed": False,
        "production_activation_allowed": False,
        "formal_h3_v2_claim_allowed": False,
    }
    for key, required in required_flags.items():
        if payload.get(key) is not required:
            raise ValueError(
                f"Direct-PNG preview {key} must be {required!r}"
            )


def _validate_preview_lineage_shape(payload: Mapping[str, Any]) -> None:
    input_contract = _mapping(payload, "input_contract")
    required_input_values = {
        "source": "raw_png_direct",
        "cache_read": False,
        "cache_written": False,
        "physical_split_policy": "index_all_split_folders_by_sample_id",
    }
    for key, required in required_input_values.items():
        observed = input_contract.get(key)
        if isinstance(required, bool):
            matches = observed is required
        else:
            matches = observed == required
        if not matches:
            raise ValueError(
                f"Direct-PNG preview input_contract.{key} mismatch"
            )
    if input_contract.get("image_size") != 192:
        raise ValueError(
            "Direct-PNG preview input_contract.image_size must be 192"
        )
    for key in ("manifest_file_sha256", "mechanism_partition_sha256"):
        _sha256(
            input_contract.get(key),
            field=f"input_contract.{key}",
        )
    raw_fingerprint = _mapping(input_contract, "raw_png_fingerprint")
    if raw_fingerprint.get("algorithm") != "sha256":
        raise ValueError(
            "Direct-PNG preview raw_png_fingerprint.algorithm must be sha256"
        )
    _sha256(
        raw_fingerprint.get("sha256"),
        field="input_contract.raw_png_fingerprint.sha256",
    )
    _positive_int(
        raw_fingerprint.get("file_count"),
        field="input_contract.raw_png_fingerprint.file_count",
    )

    dataset_contract = _mapping(payload, "dataset_contract")
    _sha256(
        dataset_contract.get("contract_sha256"),
        field="dataset_contract.contract_sha256",
    )
    contract_checks = _mapping(dataset_contract, "checks")
    if not contract_checks or any(
        value is not True for value in contract_checks.values()
    ):
        raise ValueError(
            "Direct-PNG preview dataset_contract checks must all be true"
        )

    checkpoint = _mapping(payload, "checkpoint")
    _sha256(
        checkpoint.get("file_sha256"),
        field="checkpoint.file_sha256",
    )
    if not isinstance(checkpoint.get("has_data_lineage"), bool):
        raise ValueError(
            "Direct-PNG preview checkpoint.has_data_lineage must be boolean"
        )
    if checkpoint.get("checkpoint_lineage_verified_for_current_png_run") is not False:
        raise ValueError(
            "Direct-PNG preview checkpoint lineage must remain explicitly unverified"
        )
    declared_partition = checkpoint.get("mechanism_partition_sha256")
    if declared_partition is not None:
        declared_partition = _sha256(
            declared_partition,
            field="checkpoint.mechanism_partition_sha256",
        )
        if declared_partition != str(
            input_contract["mechanism_partition_sha256"]
        ).lower():
            raise ValueError(
                "Direct-PNG preview checkpoint/manifest partition mismatch"
            )
    mean_config = checkpoint.get("mean_config")
    if not isinstance(mean_config, Mapping):
        raise ValueError("Direct-PNG preview checkpoint.mean_config must be an object")
    exclusion = mean_config.get("pathology_exclusion")
    pathology_excluded = (
        mean_config.get("pathology_policy") == "excluded"
        or (
            isinstance(exclusion, Mapping)
            and exclusion.get("enabled") is True
        )
    )
    if not pathology_excluded:
        raise ValueError(
            "Direct-PNG preview requires a pathology-excluded mean checkpoint"
        )


def _validate_preview_analysis(
    payload: Mapping[str, Any],
    *,
    expected_num_train_timesteps: int,
) -> int:
    analysis = _mapping(payload, "analysis")
    timestep_count = _positive_int(
        analysis.get("num_train_timesteps"),
        field="analysis.num_train_timesteps",
    )
    if timestep_count != expected_num_train_timesteps:
        raise ValueError(
            "Direct-PNG preview timestep count mismatch: "
            f"expected {expected_num_train_timesteps}, got {timestep_count}"
        )
    if tuple(analysis.get("band_order", ())) != H3_V2_BAND_ORDER:
        raise ValueError(
            "Direct-PNG preview analysis.band_order must be "
            f"{H3_V2_BAND_ORDER}"
        )
    required_analysis = {
        "m_schedule": "linear",
        "sigma_scale": 1.0,
        "patient_weighting": "equal_patient_weight",
        "active_transform": "clip((recoverability-0.5)/0.5,0,1)",
        "shape_constraint": "per-band isotonic non-increasing over timestep",
        "dense_curve_method": (
            "exact A=<x,x>, B=<e,e>, C=<x,e> sufficient-statistic "
            "evaluation; no timestep interpolation"
        ),
    }
    for key, required in required_analysis.items():
        if analysis.get(key) != required:
            raise ValueError(f"Direct-PNG preview analysis.{key} mismatch")

    route_mapping = _mapping(payload, "route_mapping_preview")
    if tuple(route_mapping.get("route_order", ())) != H3_V2_ROUTE_ORDER:
        raise ValueError(
            "Direct-PNG preview route_order must be "
            f"{H3_V2_ROUTE_ORDER}"
        )
    if route_mapping.get("formula") != "[a,0,1-a]":
        raise ValueError("Direct-PNG preview route formula mismatch")
    if route_mapping.get("shallow_probability") != "structurally_zero":
        raise ValueError(
            "Direct-PNG preview shallow probability must be structurally zero"
        )
    return timestep_count


def _preview_active_mass(
    payload: Mapping[str, Any],
    *,
    timestep_count: int,
) -> torch.Tensor:
    active_payload = _mapping(payload, "preview_native_active_mass")
    declared_bands = set(active_payload)
    expected_bands = set(H3_V2_BAND_ORDER)
    if declared_bands != expected_bands:
        missing = sorted(expected_bands - declared_bands)
        extra = sorted(declared_bands - expected_bands)
        raise ValueError(
            "Direct-PNG preview active-mass bands mismatch: "
            f"missing={missing}, extra={extra}"
        )

    rows: list[list[float]] = []
    for band in H3_V2_BAND_ORDER:
        values = active_payload[band]
        if not isinstance(values, list) or len(values) != timestep_count:
            raise ValueError(
                f"Direct-PNG preview band {band!r} must contain exactly "
                f"{timestep_count} values"
            )
        row: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"Direct-PNG preview band {band!r} contains a non-number"
                )
            row.append(float(value))
        tensor_row = torch.tensor(row, dtype=torch.float64)
        if not torch.isfinite(tensor_row).all():
            raise ValueError(
                f"Direct-PNG preview band {band!r} contains a non-finite value"
            )
        if bool(((tensor_row < 0.0) | (tensor_row > 1.0)).any()):
            raise ValueError(
                f"Direct-PNG preview band {band!r} leaves [0, 1]"
            )
        if bool(
            (
                tensor_row[1:] - tensor_row[:-1]
                > _MONOTONIC_TOLERANCE
            ).any()
        ):
            raise ValueError(
                f"Direct-PNG preview band {band!r} is not non-increasing"
            )
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32).reshape(
        2, 3, timestep_count
    )


def _load_direct_png_preview(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_num_train_timesteps: int,
    repository_root: str | Path | None,
    allow_unverified_preview_lineage: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if allow_unverified_preview_lineage is not True:
        raise ValueError(
            "direct_png_preview requires "
            "allow_unverified_preview_lineage=True"
        )
    expected_steps = _positive_int(
        expected_num_train_timesteps,
        field="expected_num_train_timesteps",
    )
    preview_path, root, relative_path = _resolve_preview_path(
        path,
        repository_root=repository_root,
    )
    expected_hash = _sha256(
        expected_file_sha256,
        field="expected_file_sha256",
    )
    observed_hash = _file_sha256(preview_path)
    if not hmac.compare_digest(observed_hash, expected_hash):
        raise ValueError(
            "Direct-PNG preview file SHA-256 mismatch: "
            f"expected {expected_hash}, observed {observed_hash}"
        )

    try:
        payload = json.loads(preview_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot load Direct-PNG prior preview {relative_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Direct-PNG preview root must be a JSON object")

    _validate_preview_flags(payload)
    _validate_preview_paths(
        payload,
        repository_root=root,
        preview_relative_path=relative_path,
    )
    _validate_preview_lineage_shape(payload)
    timestep_count = _validate_preview_analysis(
        payload,
        expected_num_train_timesteps=expected_steps,
    )
    active_mass = _preview_active_mass(
        payload,
        timestep_count=timestep_count,
    )
    input_contract = _mapping(payload, "input_contract")
    checkpoint = _mapping(payload, "checkpoint")
    metadata = {
        "path": relative_path,
        "file_sha256": observed_hash,
        "schema_version": DIRECT_PNG_PREVIEW_SCHEMA_VERSION,
        "pipeline_id": DIRECT_PNG_PREVIEW_PIPELINE_ID,
        "schedule_source": DIRECT_PNG_PREVIEW,
        "num_train_timesteps": timestep_count,
        "band_order": list(H3_V2_BAND_ORDER),
        "route_order": list(H3_V2_ROUTE_ORDER),
        "preview_only": True,
        "lineage_verified": False,
        "allow_unverified_preview_lineage": True,
        "inference_schedule_allowed": False,
        "production_activation_allowed": False,
        "manifest_file_sha256": str(
            input_contract["manifest_file_sha256"]
        ).lower(),
        "mean_checkpoint_file_sha256": str(
            checkpoint["file_sha256"]
        ).lower(),
    }
    return active_mass, metadata


def load_prior_anchor_schedule(
    path: str | Path,
    *,
    schedule_source: str,
    expected_file_sha256: str,
    expected_num_train_timesteps: int,
    repository_root: str | Path | None = None,
    allow_unverified_preview_lineage: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load a formal H3 schedule or an explicitly opted-in PNG preview.

    ``formal_h3_v2`` delegates without weakening the existing formal loader.
    ``direct_png_preview`` remains visibly non-production and requires the
    caller to acknowledge that its current-runtime lineage is unverified.
    """

    if schedule_source == FORMAL_H3_V2:
        if allow_unverified_preview_lineage:
            raise ValueError(
                "allow_unverified_preview_lineage is only valid for "
                "direct_png_preview"
            )
        return load_h3_native_null_schedule(
            path,
            expected_file_sha256=expected_file_sha256,
            expected_num_train_timesteps=expected_num_train_timesteps,
        )
    if schedule_source == DIRECT_PNG_PREVIEW:
        return _load_direct_png_preview(
            path,
            expected_file_sha256=expected_file_sha256,
            expected_num_train_timesteps=expected_num_train_timesteps,
            repository_root=repository_root,
            allow_unverified_preview_lineage=(
                allow_unverified_preview_lineage
            ),
        )
    raise ValueError(
        "schedule_source must be one of "
        f"{sorted(PRIOR_ANCHOR_SCHEDULE_SOURCES)}, got {schedule_source!r}"
    )
