"""Stage 0C cloud data/cache/checkpoint-lineage gate.

This script is a detector, not a trainer.  It verifies that a cloud-built PNG
cache is a complete, numerically faithful realization of the locked Stage 0A
dataset contract before training is allowed.

The gate is intentionally exhaustive:

* the authoritative manifest is checked semantically and for patient leakage;
* all raw CT/PET/mask PNG files are content-fingerprinted;
* every manifest sample must have exactly one NPZ and one JSON sidecar;
* every cached CT/PET/mask tensor is compared with the contract preprocessing;
* PET polarity, mask thresholding, tensor metadata, dtype, and shape are checked;
* an optional content-addressed cache-lineage file is sealed only after PASS;
* optional checkpoints must embed all locked data and cache fingerprints.

No training, model evaluation, or model-mechanism claim is performed here.

Typical cloud use::

    python scripts/run_cloud_stage0c_gate.py \
      --cache-dir cache/tensors_main \
      --seal-cache

After a checkpoint exists::

    python scripts/run_cloud_stage0c_gate.py \
      --cache-dir cache/tensors_main \
      --checkpoint checkpoints/run_name/best.pt

Exit code 0 means the requested gate passed.  Exit code 2 means a hard gate
failed.  An omitted checkpoint leaves checkpoint lineage DEFERRED and does not
block a pre-training cache gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


SCHEMA_VERSION = 1
MANIFEST_COLUMNS = ("sample_id", "patient_id", "slice_id", "split", "cache_path")
MODALITY_DIRS = {"ct": "ct", "pet": "pet", "mask": "label"}
REQUIRED_CACHE_KEYS = ("ct", "pet", "mask")
SAMPLE_ID_RE = re.compile(r"^(\d{3})(\d{3})$")
ENGINEERING_ATOL = 1.0e-6
CACHE_LINEAGE_NAME = "cache_lineage.json"
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        converted = float(value)
        return converted if np.isfinite(converted) else None
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)
        + "\n"
    )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = sorted({key for row in rows for key in row}) if rows else ()
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(columns), extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = [
            {
                str(key): "" if value is None else str(value).strip()
                for key, value in row.items()
            }
            for row in reader
        ]
    return rows, columns


def _manifest_semantic_sha256(rows: Sequence[Mapping[str, str]]) -> str:
    canonical_rows = [
        {column: row.get(column, "") for column in MANIFEST_COLUMNS}
        for row in sorted(rows, key=lambda item: item.get("sample_id", ""))
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "columns": list(MANIFEST_COLUMNS),
        "rows": canonical_rows,
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def _validate_contract(contract: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    body = dict(contract)
    declared_contract_hash = str(body.pop("contract_sha256", ""))
    computed_contract_hash = _sha256_bytes(_canonical_json_bytes(body))

    preprocessing = contract.get("preprocessing", {})
    declared_preprocessing_hash = str(
        contract.get("preprocessing_config_sha256", "")
    )
    computed_preprocessing_hash = _sha256_bytes(
        _canonical_json_bytes(preprocessing)
    )
    required_metadata = contract.get("cloud_cache_metadata_required", {})
    metadata_matches = (
        isinstance(required_metadata, Mapping)
        and required_metadata.get("manifest_semantic_sha256")
        == contract.get("manifest", {}).get("semantic_sha256")
        and required_metadata.get("raw_png_combined_sha256")
        == contract.get("raw_png", {}).get("combined_sha256")
        and required_metadata.get("preprocessing_config_sha256")
        == declared_preprocessing_hash
        and required_metadata.get("dataset_contract_sha256")
        == "must equal this file's contract_sha256"
    )
    rules_supported = (
        contract.get("schema_version") == SCHEMA_VERSION
        and contract.get("contract_status") == "LOCKED"
        and preprocessing.get("output_image_size") == [192, 192]
        and preprocessing.get("ct", {}).get("resize")
        == "PIL.Image.Resampling.BILINEAR"
        and preprocessing.get("ct", {}).get("normalization")
        == "uint8 / 127.5 - 1.0"
        and preprocessing.get("pet", {}).get("resize")
        == "PIL.Image.Resampling.BILINEAR"
        and preprocessing.get("pet", {}).get("polarity")
        == "invert white-canvas/dark-uptake as 255 - pixel"
        and preprocessing.get("pet", {}).get("normalization")
        == "inverted_uint8 / 127.5 - 1.0"
        and preprocessing.get("mask", {}).get("resize")
        == "PIL.Image.Resampling.NEAREST"
        and preprocessing.get("mask", {})
        .get("threshold_uint8", {})
        .get("operator")
        == ">"
        and preprocessing.get("mask", {})
        .get("threshold_uint8", {})
        .get("value")
        == 127
    )
    passed = bool(
        declared_contract_hash
        and declared_contract_hash == computed_contract_hash
        and declared_preprocessing_hash
        and declared_preprocessing_hash == computed_preprocessing_hash
        and metadata_matches
        and rules_supported
    )
    return passed, {
        "declared_contract_sha256": declared_contract_hash,
        "computed_contract_sha256": computed_contract_hash,
        "declared_preprocessing_sha256": declared_preprocessing_hash,
        "computed_preprocessing_sha256": computed_preprocessing_hash,
        "required_metadata_consistent": bool(metadata_matches),
        "preprocessing_rules_supported_by_detector": bool(rules_supported),
    }


def _validate_manifest(
    rows: Sequence[Mapping[str, str]],
    columns: Sequence[str],
    contract: Mapping[str, Any],
) -> tuple[bool, dict[str, Any], dict[str, Mapping[str, str]]]:
    sample_counts = Counter(row.get("sample_id", "") for row in rows)
    duplicate_samples = sorted(
        sample_id for sample_id, count in sample_counts.items() if count != 1
    )
    invalid_rows: list[dict[str, str]] = []
    patient_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        sample_id = row.get("sample_id", "")
        match = SAMPLE_ID_RE.fullmatch(sample_id)
        reasons: list[str] = []
        if match is None:
            reasons.append("sample_id_not_six_digits")
        else:
            if row.get("patient_id") != match.group(1):
                reasons.append("patient_id_not_sample_prefix")
            try:
                if int(row.get("slice_id", "")) != int(match.group(2)):
                    reasons.append("slice_id_not_sample_suffix")
            except ValueError:
                reasons.append("slice_id_not_integer")
        split = row.get("split", "")
        if split not in {"train", "val", "test"}:
            reasons.append("invalid_split")
        patient_splits[row.get("patient_id", "")].add(split)
        if reasons:
            invalid_rows.append(
                {"sample_id": sample_id, "reasons": ";".join(reasons)}
            )

    overlap = {
        patient_id: sorted(splits)
        for patient_id, splits in patient_splits.items()
        if len(splits) > 1
    }
    actual_splits: dict[str, dict[str, int]] = {}
    expected_splits = contract.get("splits", {})
    split_counts_match = True
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row.get("split") == split]
        actual = {
            "samples": len(split_rows),
            "patients": len({row.get("patient_id", "") for row in split_rows}),
        }
        actual_splits[split] = actual
        expected = expected_splits.get(split, {})
        split_counts_match = split_counts_match and actual == {
            "samples": expected.get("samples"),
            "patients": expected.get("patients"),
        }

    semantic_sha256 = _manifest_semantic_sha256(rows)
    expected_semantic_sha256 = contract.get("manifest", {}).get(
        "semantic_sha256"
    )
    required_columns_present = set(MANIFEST_COLUMNS).issubset(columns)
    passed = bool(
        required_columns_present
        and not duplicate_samples
        and not invalid_rows
        and not overlap
        and split_counts_match
        and semantic_sha256 == expected_semantic_sha256
    )
    evidence = {
        "row_count": len(rows),
        "required_columns_present": required_columns_present,
        "duplicate_sample_count": len(duplicate_samples),
        "invalid_row_count": len(invalid_rows),
        "patient_overlap_count": len(overlap),
        "actual_splits": actual_splits,
        "split_counts_match_contract": split_counts_match,
        "computed_semantic_sha256": semantic_sha256,
        "expected_semantic_sha256": expected_semantic_sha256,
        "semantic_sha256_matches": semantic_sha256 == expected_semantic_sha256,
        "invalid_rows_preview": invalid_rows[:20],
        "patient_overlap_preview": dict(list(overlap.items())[:20]),
    }
    manifest_by_id = {
        row["sample_id"]: row
        for row in rows
        if row.get("sample_id") and sample_counts[row["sample_id"]] == 1
    }
    return passed, evidence, manifest_by_id


def _index_pngs(
    raw_root: Path, modality_dir: str
) -> tuple[dict[str, Path], dict[str, list[Path]], int]:
    candidates: dict[str, list[Path]] = defaultdict(list)
    file_count = 0
    for physical_split in ("train", "val", "test"):
        directory = raw_root / physical_split / modality_dir
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.png")):
            candidates[path.stem].append(path)
            file_count += 1
    unique = {
        sample_id: paths[0]
        for sample_id, paths in candidates.items()
        if len(paths) == 1
    }
    duplicates = {
        sample_id: paths for sample_id, paths in candidates.items() if len(paths) != 1
    }
    return unique, duplicates, file_count


def _raw_root_inventory(raw_root: Path) -> dict[str, Any]:
    file_counts: dict[str, int] = {}
    layout_directory_count = 0
    for modality, folder in MODALITY_DIRS.items():
        _, _, file_count = _index_pngs(raw_root, folder)
        file_counts[modality] = file_count
        layout_directory_count += sum(
            (raw_root / split / folder).is_dir()
            for split in ("train", "val", "test")
        )
    return {
        "path": raw_root.as_posix(),
        "exists": raw_root.is_dir(),
        "layout_directory_count": layout_directory_count,
        "file_counts": file_counts,
        "total_file_count": sum(file_counts.values()),
    }


def _select_raw_root(
    root: Path,
    contract: Mapping[str, Any],
    explicit_raw_root: Path | None,
) -> tuple[Path, dict[str, Any]]:
    contract_root = _resolve(root, contract["raw_png"]["root"]).resolve()
    if explicit_raw_root is not None:
        selected = explicit_raw_root.resolve()
        return selected, {
            "mode": "explicit",
            "selected": _relative(selected, root),
            "candidates": [_raw_root_inventory(selected)],
        }

    environment_value = os.environ.get("CT_PET_RAW_ROOT", "").strip()
    candidate_values = [
        contract_root,
        root / "Data" / "data",
        root / "main_data",
        root.parent / "main_data",
        root / "data",
        root / "Data",
        root.parent / "Data" / "data",
    ]
    if environment_value:
        candidate_values.insert(0, _resolve(root, environment_value))

    candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidate_values:
        resolved = candidate.resolve()
        identity = os.path.normcase(str(resolved))
        if identity not in seen:
            seen.add(identity)
            candidates.append(resolved)

    expected = contract["raw_png"]
    expected_modalities = expected.get("modalities", {})
    inventories = [_raw_root_inventory(candidate) for candidate in candidates]
    matching_candidates = [
        candidate
        for candidate, inventory in zip(candidates, inventories)
        if inventory["total_file_count"] == expected.get("file_count")
        and all(
            inventory["file_counts"].get(modality)
            == expected_modalities.get(modality)
            for modality in MODALITY_DIRS
        )
    ]
    if contract_root in matching_candidates:
        selected = contract_root
        mode = "contract_path"
    elif matching_candidates:
        selected = matching_candidates[0]
        mode = "auto_discovered_by_locked_counts"
    else:
        selected = contract_root
        mode = "contract_path_no_matching_candidate"

    return selected, {
        "mode": mode,
        "selected": _relative(selected, root),
        "environment_variable": "CT_PET_RAW_ROOT",
        "environment_value_present": bool(environment_value),
        "matching_candidate_count": len(matching_candidates),
        "candidates": [
            {**inventory, "path": _relative(Path(inventory["path"]), root)}
            for inventory in inventories
        ],
        "selection_safety": (
            "Auto-discovery uses locked per-modality file counts only; the selected "
            "root must still pass sample bijection and the full raw-PNG SHA-256."
        ),
    }


def _validate_and_fingerprint_raw(
    raw_root: Path,
    manifest_ids: set[str],
    contract: Mapping[str, Any],
) -> tuple[bool, dict[str, Any], dict[str, dict[str, Path]]]:
    indexes: dict[str, dict[str, Path]] = {}
    duplicate_counts: dict[str, int] = {}
    file_counts: dict[str, int] = {}
    missing: dict[str, list[str]] = {}
    extras: dict[str, list[str]] = {}
    hash_entries: list[str] = []
    hash_errors: list[dict[str, str]] = []

    for modality, folder in MODALITY_DIRS.items():
        index, duplicates, file_count = _index_pngs(raw_root, folder)
        indexes[modality] = index
        duplicate_counts[modality] = len(duplicates)
        file_counts[modality] = file_count
        missing[modality] = sorted(manifest_ids - set(index))
        extras[modality] = sorted(set(index) - manifest_ids)

    if not any(duplicate_counts.values()) and not any(missing.values()):
        for sample_id in sorted(manifest_ids):
            for modality in MODALITY_DIRS:
                path = indexes[modality][sample_id]
                try:
                    hash_entries.append(
                        f"{sample_id}|{modality}|{_sha256_file(path)}"
                    )
                except Exception as exc:
                    hash_errors.append(
                        {
                            "sample_id": sample_id,
                            "modality": modality,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

    combined_sha256 = (
        _sha256_bytes("\n".join(sorted(hash_entries)).encode("utf-8"))
        if hash_entries
        else ""
    )
    expected = contract.get("raw_png", {})
    total_file_count = sum(file_counts.values())
    modalities_match = all(
        file_counts.get(modality, 0) == expected.get("modalities", {}).get(modality)
        for modality in MODALITY_DIRS
    )
    passed = bool(
        raw_root.is_dir()
        and not any(duplicate_counts.values())
        and not any(missing.values())
        and not any(extras.values())
        and not hash_errors
        and total_file_count == expected.get("file_count")
        and modalities_match
        and combined_sha256 == expected.get("combined_sha256")
    )
    evidence = {
        "raw_root": raw_root.as_posix(),
        "file_counts": file_counts,
        "total_file_count": total_file_count,
        "expected_total_file_count": expected.get("file_count"),
        "modality_counts_match": modalities_match,
        "duplicate_counts": duplicate_counts,
        "missing_counts": {key: len(value) for key, value in missing.items()},
        "extra_counts": {key: len(value) for key, value in extras.items()},
        "missing_preview": {key: value[:20] for key, value in missing.items()},
        "extra_preview": {key: value[:20] for key, value in extras.items()},
        "hash_error_count": len(hash_errors),
        "hash_errors_preview": hash_errors[:20],
        "computed_combined_sha256": combined_sha256,
        "expected_combined_sha256": expected.get("combined_sha256"),
        "combined_sha256_matches": combined_sha256
        == expected.get("combined_sha256"),
    }
    return passed, evidence, indexes


def _load_contract_expected_arrays(
    paths: Mapping[str, Path], image_size: tuple[int, int]
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for modality, path in paths.items():
        with Image.open(path) as image:
            image = image.convert("L")
            resampling = (
                Image.Resampling.NEAREST
                if modality == "mask"
                else Image.Resampling.BILINEAR
            )
            if image.size != (image_size[1], image_size[0]):
                image = image.resize((image_size[1], image_size[0]), resampling)
            array = np.asarray(image, dtype=np.uint8)
        if modality == "ct":
            arrays["ct"] = array.astype(np.float32) / 127.5 - 1.0
        elif modality == "pet":
            arrays["pet"] = 1.0 - array.astype(np.float32) / 127.5
            arrays["pet_noninverted"] = (
                array.astype(np.float32) / 127.5 - 1.0
            )
        else:
            arrays["mask"] = (array > 127).astype(np.float32)
            arrays["mask_positive"] = (array > 0).astype(np.float32)
    return arrays


def _decode_json_array(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype == np.uint8:
        raw = array.tobytes()
    elif array.dtype.kind in {"S", "U"}:
        raw = str(array.reshape(-1)[0]).encode("utf-8")
    else:
        raise TypeError(f"Unsupported scale_meta_json dtype: {array.dtype}")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("scale_meta_json must decode to a JSON object")
    return payload


def _validate_cache(
    cache_dir: Path,
    manifest_by_id: Mapping[str, Mapping[str, str]],
    raw_indexes: Mapping[str, Mapping[str, Path]],
    contract: Mapping[str, Any],
) -> tuple[
    bool,
    bool,
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any] | None,
]:
    manifest_ids = set(manifest_by_id)
    cache_paths = {path.stem: path for path in cache_dir.glob("*.npz")}
    sidecar_paths = {
        path.name[: -len("_meta.json")]: path
        for path in cache_dir.glob("*_meta.json")
    }
    missing_npz = sorted(manifest_ids - set(cache_paths))
    extra_npz = sorted(set(cache_paths) - manifest_ids)
    missing_sidecars = sorted(manifest_ids - set(sidecar_paths))
    extra_sidecars = sorted(set(sidecar_paths) - manifest_ids)
    bijection_pass = bool(
        cache_dir.is_dir()
        and not missing_npz
        and not extra_npz
        and not missing_sidecars
        and not extra_sidecars
    )

    image_size_value = contract.get("preprocessing", {}).get(
        "output_image_size", [192, 192]
    )
    image_size = (int(image_size_value[0]), int(image_size_value[1]))
    rows: list[dict[str, Any]] = []
    payload_hash_entries: list[str] = []

    for sample_id in sorted(manifest_ids & set(cache_paths)):
        manifest_row = manifest_by_id[sample_id]
        npz_path = cache_paths[sample_id]
        sidecar_path = sidecar_paths.get(sample_id)
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "patient_id": manifest_row.get("patient_id", ""),
            "split": manifest_row.get("split", ""),
            "npz_path": npz_path.as_posix(),
            "sidecar_path": "" if sidecar_path is None else sidecar_path.as_posix(),
            "ct_max_abs_error": None,
            "pet_max_abs_error": None,
            "pet_noninverted_max_abs_error": None,
            "mask_disagreement_fraction": None,
            "mask_positive_disagreement_fraction": None,
            "pet_matches_noninverted": False,
            "npz_pet_invert": None,
            "sidecar_pet_invert": None,
            "format_pass": False,
            "numerical_pass": False,
            "metadata_pass": False,
            "sample_pass": False,
            "errors": "",
        }
        errors: list[str] = []
        try:
            npz_sha256 = _sha256_file(npz_path)
            row["npz_sha256"] = npz_sha256
            payload_hash_entries.append(f"{sample_id}|npz|{npz_sha256}")
            with np.load(npz_path, allow_pickle=False) as archive:
                keys = set(archive.files)
                missing_keys = sorted(set(REQUIRED_CACHE_KEYS) - keys)
                if missing_keys:
                    errors.append(f"missing_npz_keys={','.join(missing_keys)}")
                arrays = {
                    key: np.asarray(archive[key])
                    for key in REQUIRED_CACHE_KEYS
                    if key in keys
                }
                shape_pass = all(
                    array.shape == (1, image_size[0], image_size[1])
                    for array in arrays.values()
                ) and len(arrays) == len(REQUIRED_CACHE_KEYS)
                dtype_pass = all(
                    array.dtype == np.float32 for array in arrays.values()
                ) and len(arrays) == len(REQUIRED_CACHE_KEYS)
                finite_pass = all(
                    np.isfinite(array).all() for array in arrays.values()
                ) and len(arrays) == len(REQUIRED_CACHE_KEYS)
                if not shape_pass:
                    errors.append("invalid_tensor_shape")
                if not dtype_pass:
                    errors.append("tensor_dtype_not_float32")
                if not finite_pass:
                    errors.append("nonfinite_tensor_values")

                scale_meta: dict[str, Any] = {}
                if "scale_meta_json" not in keys:
                    errors.append("missing_scale_meta_json")
                else:
                    scale_meta = _decode_json_array(archive["scale_meta_json"])
                row["npz_pet_invert"] = scale_meta.get("pet_invert")
                scale_meta_pass = bool(
                    scale_meta.get("pet_physical_kind") == "png_intensity"
                    and scale_meta.get("pet_invert") is True
                    and scale_meta.get("suv_ok") is False
                    and scale_meta.get("pet_suv_available") is False
                    and str(scale_meta.get("patient_id", ""))
                    == manifest_row.get("patient_id", "")
                    and int(scale_meta.get("slice_id", -1))
                    == int(manifest_row.get("slice_id", "-2"))
                )
                if not scale_meta_pass:
                    errors.append("npz_scale_metadata_mismatch")

                row["format_pass"] = bool(
                    not missing_keys and shape_pass and dtype_pass and finite_pass
                )
                if row["format_pass"]:
                    expected = _load_contract_expected_arrays(
                        {
                            modality: raw_indexes[modality][sample_id]
                            for modality in MODALITY_DIRS
                        },
                        image_size,
                    )
                    cached = {
                        key: arrays[key][0].astype(np.float32, copy=False)
                        for key in REQUIRED_CACHE_KEYS
                    }
                    row["ct_max_abs_error"] = float(
                        np.max(np.abs(cached["ct"] - expected["ct"]))
                    )
                    row["pet_max_abs_error"] = float(
                        np.max(np.abs(cached["pet"] - expected["pet"]))
                    )
                    row["pet_noninverted_max_abs_error"] = float(
                        np.max(
                            np.abs(
                                cached["pet"] - expected["pet_noninverted"]
                            )
                        )
                    )
                    row["mask_disagreement_fraction"] = float(
                        np.mean(
                            (cached["mask"] > 0.5)
                            != (expected["mask"] > 0.5)
                        )
                    )
                    row["mask_positive_disagreement_fraction"] = float(
                        np.mean(
                            (cached["mask"] > 0.5)
                            != (expected["mask_positive"] > 0.5)
                        )
                    )
                    row["pet_matches_noninverted"] = bool(
                        row["pet_noninverted_max_abs_error"]
                        <= ENGINEERING_ATOL
                        and row["pet_max_abs_error"] > ENGINEERING_ATOL
                    )
                    mask_binary = bool(
                        np.isin(np.unique(cached["mask"]), [0.0, 1.0]).all()
                    )
                    row["numerical_pass"] = bool(
                        row["ct_max_abs_error"] <= ENGINEERING_ATOL
                        and row["pet_max_abs_error"] <= ENGINEERING_ATOL
                        and row["mask_disagreement_fraction"] == 0.0
                        and mask_binary
                    )
                    if not row["numerical_pass"]:
                        errors.append("cache_differs_from_contract_preprocessing")

                sidecar_pass = False
                if sidecar_path is None:
                    errors.append("missing_sidecar")
                else:
                    sidecar_sha256 = _sha256_file(sidecar_path)
                    row["sidecar_sha256"] = sidecar_sha256
                    payload_hash_entries.append(
                        f"{sample_id}|meta|{sidecar_sha256}"
                    )
                    sidecar = _read_json(sidecar_path)
                    sidecar_scale = sidecar.get("scale_meta", {})
                    row["sidecar_pet_invert"] = sidecar_scale.get("pet_invert")
                    sidecar_pass = bool(
                        sidecar.get("sample_id") == sample_id
                        and str(sidecar.get("patient_id", ""))
                        == manifest_row.get("patient_id", "")
                        and int(sidecar.get("slice_id", -1))
                        == int(manifest_row.get("slice_id", "-2"))
                        and sidecar.get("split") == manifest_row.get("split")
                        and sidecar.get("source") == "png"
                        and sidecar.get("has_label") is True
                        and sidecar_scale.get("pet_physical_kind")
                        == "png_intensity"
                        and sidecar_scale.get("pet_invert") is True
                        and sidecar_scale.get("suv_ok") is False
                    )
                    if not sidecar_pass:
                        errors.append("sidecar_metadata_mismatch")
                row["metadata_pass"] = bool(scale_meta_pass and sidecar_pass)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        row["sample_pass"] = bool(
            row["format_pass"]
            and row["numerical_pass"]
            and row["metadata_pass"]
            and not errors
        )
        row["errors"] = ";".join(errors)
        rows.append(row)

    sample_pass_count = sum(bool(row["sample_pass"]) for row in rows)
    format_pass_count = sum(bool(row["format_pass"]) for row in rows)
    numerical_pass_count = sum(bool(row["numerical_pass"]) for row in rows)
    metadata_pass_count = sum(bool(row["metadata_pass"]) for row in rows)
    pet_noninverted_match_count = sum(
        bool(row["pet_matches_noninverted"]) for row in rows
    )
    error_counts: Counter[str] = Counter()
    for row in rows:
        error_counts.update(
            error for error in str(row.get("errors", "")).split(";") if error
        )

    def _finite_max(field: str) -> float | None:
        values = [
            float(row[field])
            for row in rows
            if row.get(field) is not None and np.isfinite(float(row[field]))
        ]
        return max(values) if values else None

    content_pass = bool(
        bijection_pass
        and len(rows) == len(manifest_ids)
        and sample_pass_count == len(manifest_ids)
    )
    cache_lineage: dict[str, Any] | None = None
    if content_pass:
        payload_sha256 = _sha256_bytes(
            "\n".join(sorted(payload_hash_entries)).encode("utf-8")
        )
        lineage_body = {
            "schema_version": SCHEMA_VERSION,
            "lineage_type": "verified_png_tensor_cache",
            "dataset_contract_sha256": contract.get("contract_sha256"),
            "manifest_semantic_sha256": contract.get("manifest", {}).get(
                "semantic_sha256"
            ),
            "raw_png_combined_sha256": contract.get("raw_png", {}).get(
                "combined_sha256"
            ),
            "preprocessing_config_sha256": contract.get(
                "preprocessing_config_sha256"
            ),
            "cache_payload_sha256": payload_sha256,
            "sample_count": len(manifest_ids),
            "required_keys": list(REQUIRED_CACHE_KEYS),
            "tensor_shape": [1, image_size[0], image_size[1]],
            "tensor_dtype": "float32",
            "numerical_gate": {
                "comparison": "exhaustive_per_sample_max_absolute_error",
                "ct_pet_atol": ENGINEERING_ATOL,
                "mask_disagreement_required": 0.0,
                "threshold_origin": "fixed numerical-equivalence tolerance; not fitted to validation/test metrics",
            },
        }
        cache_lineage = {
            **lineage_body,
            "cache_metadata_sha256": _sha256_bytes(
                _canonical_json_bytes(lineage_body)
            ),
        }

    evidence = {
        "cache_dir": cache_dir.as_posix(),
        "manifest_sample_count": len(manifest_ids),
        "npz_count": len(cache_paths),
        "sidecar_count": len(sidecar_paths),
        "missing_npz_count": len(missing_npz),
        "extra_npz_count": len(extra_npz),
        "missing_sidecar_count": len(missing_sidecars),
        "extra_sidecar_count": len(extra_sidecars),
        "missing_npz_preview": missing_npz[:20],
        "extra_npz_preview": extra_npz[:20],
        "missing_sidecar_preview": missing_sidecars[:20],
        "extra_sidecar_preview": extra_sidecars[:20],
        "bijection_pass": bijection_pass,
        "validated_sample_count": len(rows),
        "format_pass_count": format_pass_count,
        "numerical_pass_count": numerical_pass_count,
        "metadata_pass_count": metadata_pass_count,
        "sample_pass_count": sample_pass_count,
        "sample_failure_count": len(rows) - sample_pass_count,
        "error_counts": dict(sorted(error_counts.items())),
        "ct_max_abs_error_max": _finite_max("ct_max_abs_error"),
        "pet_inverted_max_abs_error_max": _finite_max("pet_max_abs_error"),
        "pet_noninverted_max_abs_error_max": _finite_max(
            "pet_noninverted_max_abs_error"
        ),
        "mask_disagreement_fraction_max": _finite_max(
            "mask_disagreement_fraction"
        ),
        "mask_positive_disagreement_fraction_max": _finite_max(
            "mask_positive_disagreement_fraction"
        ),
        "pet_noninverted_match_count": pet_noninverted_match_count,
        "npz_pet_invert_counts": dict(
            Counter(str(row.get("npz_pet_invert")) for row in rows)
        ),
        "sidecar_pet_invert_counts": dict(
            Counter(str(row.get("sidecar_pet_invert")) for row in rows)
        ),
        "engineering_atol": ENGINEERING_ATOL,
        "cache_payload_sha256": (
            None if cache_lineage is None else cache_lineage["cache_payload_sha256"]
        ),
    }
    return bijection_pass, content_pass, evidence, rows, cache_lineage


def _validate_lineage_self_hash(payload: Mapping[str, Any]) -> bool:
    body = dict(payload)
    declared = str(body.pop("cache_metadata_sha256", ""))
    return bool(
        declared and declared == _sha256_bytes(_canonical_json_bytes(body))
    )


def _seal_or_validate_cache_lineage(
    cache_dir: Path,
    expected: Mapping[str, Any] | None,
    *,
    seal: bool,
) -> tuple[bool, dict[str, Any]]:
    path = cache_dir / CACHE_LINEAGE_NAME
    existed_before = path.is_file()
    existing: dict[str, Any] | None = None
    read_error = ""
    if existed_before:
        try:
            existing = _read_json(path)
        except Exception as exc:
            read_error = f"{type(exc).__name__}: {exc}"

    wrote = False
    if expected is not None and seal and existing != expected:
        _write_json(path, expected)
        existing = _read_json(path)
        wrote = True

    self_hash_pass = bool(
        existing is not None and _validate_lineage_self_hash(existing)
    )
    exact_match = bool(existing is not None and existing == expected)
    passed = bool(expected is not None and self_hash_pass and exact_match)
    return passed, {
        "path": path.as_posix(),
        "existed_before": existed_before,
        "seal_requested": seal,
        "written": wrote,
        "read_error": read_error,
        "self_hash_pass": self_hash_pass,
        "matches_current_validated_cache": exact_match,
        "cache_metadata_sha256": (
            None if expected is None else expected.get("cache_metadata_sha256")
        ),
    }


def _expand_checkpoint_inputs(
    values: Sequence[str], root: Path
) -> tuple[list[Path], list[str]]:
    paths: list[Path] = []
    errors: list[str] = []
    for value in values:
        candidate = _resolve(root, value)
        if candidate.is_file():
            paths.append(candidate)
        elif candidate.is_dir():
            paths.extend(
                path
                for path in candidate.rglob("*")
                if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES
            )
        else:
            errors.append(f"checkpoint path does not exist: {candidate}")
    return sorted(set(path.resolve() for path in paths)), errors


def _collect_named_values(value: Any, result: dict[str, list[Any]]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            result.setdefault(key_text, []).append(child)
            _collect_named_values(child, result)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _collect_named_values(child, result)


def _validate_checkpoints(
    checkpoint_values: Sequence[str],
    root: Path,
    cache_lineage: Mapping[str, Any] | None,
    *,
    hash_checkpoints: bool = True,
    metadata_only: bool = False,
    show_progress: bool = False,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    if not checkpoint_values:
        return (
            "DEFERRED",
            {
                "requested_checkpoint_count": 0,
                "reason": "No checkpoint supplied; this does not block pre-training Stage 0C.",
            },
            [],
        )

    paths, input_errors = _expand_checkpoint_inputs(checkpoint_values, root)
    expected_fields = (
        {}
        if cache_lineage is None
        else {
            key: cache_lineage[key]
            for key in (
                "manifest_semantic_sha256",
                "raw_png_combined_sha256",
                "preprocessing_config_sha256",
                "dataset_contract_sha256",
                "cache_payload_sha256",
                "cache_metadata_sha256",
            )
        }
    )
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        if show_progress:
            size_gib = path.stat().st_size / (1024 ** 3)
            print(
                f"[checkpoint {index}/{len(paths)}] "
                f"{_relative(path, root)} ({size_gib:.2f} GiB)",
                flush=True,
            )
        row: dict[str, Any] = {
            "checkpoint": _relative(path, root),
            "sha256": "",
            "load_pass": False,
            "required_fingerprint_count": len(expected_fields),
            "matched_fingerprint_count": 0,
            "use_fake_true_found": False,
            "lineage_pass": False,
            "error": "",
        }
        try:
            import torch

            if hash_checkpoints:
                if show_progress:
                    print("  hashing file...", flush=True)
                row["sha256"] = _sha256_file(path)
            if show_progress:
                mode = "metadata-only" if metadata_only else "full safe load"
                print(f"  loading {mode}...", flush=True)
            if metadata_only:
                from torch._subclasses.fake_tensor import FakeTensorMode

                with FakeTensorMode():
                    checkpoint = torch.load(
                        path, map_location="cpu", weights_only=True
                    )
            else:
                checkpoint = torch.load(
                    path, map_location="cpu", weights_only=True
                )
            named_values: dict[str, list[Any]] = {}
            _collect_named_values(checkpoint, named_values)
            matched = 0
            missing_fields: list[str] = []
            for field, expected_value in expected_fields.items():
                values = named_values.get(field, [])
                if expected_value in values:
                    matched += 1
                else:
                    missing_fields.append(field)
            use_fake_values = named_values.get("use_fake", [])
            use_fake_true = any(value is True for value in use_fake_values)
            row["load_pass"] = True
            row["matched_fingerprint_count"] = matched
            row["use_fake_true_found"] = use_fake_true
            row["missing_fields"] = ";".join(missing_fields)
            row["lineage_pass"] = bool(
                expected_fields
                and matched == len(expected_fields)
                and not use_fake_true
            )
            if show_progress:
                result = "PASS" if row["lineage_pass"] else "QUARANTINE"
                print(
                    f"  {result}: matched {matched}/{len(expected_fields)} "
                    "required fingerprints",
                    flush=True,
                )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            if show_progress:
                print(f"  QUARANTINE: {row['error']}", flush=True)
        rows.append(row)

    passed = bool(
        not input_errors
        and paths
        and cache_lineage is not None
        and all(bool(row["lineage_pass"]) for row in rows)
    )
    status = "PASS" if passed else "FAIL"
    evidence = {
        "requested_checkpoint_count": len(checkpoint_values),
        "resolved_checkpoint_count": len(paths),
        "input_errors": input_errors,
        "passed_checkpoint_count": sum(
            bool(row["lineage_pass"]) for row in rows
        ),
        "required_named_fingerprints": sorted(expected_fields),
        "safe_loading_policy": "torch.load(weights_only=True); unsafe pickle loading is forbidden",
        "checkpoint_file_hashing": hash_checkpoints,
        "metadata_only_loading": metadata_only,
    }
    return status, evidence, rows


def _gate(
    gate_id: str, description: str, passed: bool, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "description": description,
        "status": "PASS" if passed else "FAIL",
        "evidence": dict(evidence),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_path = args.contract.resolve()
    cache_dir = args.cache_dir.resolve()

    contract = _read_json(contract_path)
    contract_pass, contract_evidence = _validate_contract(contract)
    gates: list[dict[str, Any]] = [
        _gate(
            "G0C.0_locked_contract",
            "The copied Stage 0A dataset contract is self-consistent and supported.",
            contract_pass,
            {
                **contract_evidence,
                "contract_path": _relative(contract_path, root),
            },
        )
    ]

    if not contract_pass:
        decision = {
            "stage": "00C_cloud_data_gate",
            "scope": "cloud_pretraining_validation_only",
            "decision": "FAIL",
            "data_gate": "FAIL",
            "cloud_training_gate": "FAIL",
            "checkpoint_lineage": "NOT_EVALUATED",
            "training_allowed": False,
            "formal_checkpoint_claims_allowed": False,
            "model_mechanism_claims_allowed": False,
            "training_actions_performed": False,
            "gates": gates,
            "stop_rule": "Stop: the locked dataset contract is invalid or unsupported.",
        }
        _write_json(output / "decision.json", decision)
        return decision

    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else _resolve(root, contract["manifest"]["path"]).resolve()
    )
    raw_root, raw_root_resolution = _select_raw_root(
        root,
        contract,
        args.raw_root,
    )
    manifest_rows, manifest_columns = _read_manifest(manifest_path)
    manifest_pass, manifest_evidence, manifest_by_id = _validate_manifest(
        manifest_rows, manifest_columns, contract
    )
    manifest_evidence["manifest_path"] = _relative(manifest_path, root)
    manifest_evidence["source_file_sha256"] = _sha256_file(manifest_path)
    manifest_evidence["source_file_sha256_matches_contract"] = (
        manifest_evidence["source_file_sha256"]
        == contract.get("manifest", {}).get("source_file_sha256")
    )
    gates.append(
        _gate(
            "G0C.1_authoritative_manifest",
            "Manifest semantics, patient grouping, and split counts match Stage 0A.",
            manifest_pass,
            manifest_evidence,
        )
    )

    raw_pass, raw_evidence, raw_indexes = _validate_and_fingerprint_raw(
        raw_root, set(manifest_by_id), contract
    )
    raw_evidence["raw_root"] = _relative(raw_root, root)
    raw_evidence["raw_root_resolution"] = raw_root_resolution
    physical_split_mismatches: list[str] = []
    if raw_pass:
        for sample_id, manifest_row in manifest_by_id.items():
            if any(
                raw_indexes[modality][sample_id].parent.parent.name
                != manifest_row.get("split")
                for modality in MODALITY_DIRS
            ):
                physical_split_mismatches.append(sample_id)
    raw_evidence["physical_split_mismatch_sample_count"] = len(
        physical_split_mismatches
    )
    raw_evidence["physical_split_mismatch_preview"] = sorted(
        physical_split_mismatches
    )[:20]
    raw_evidence["split_authority_policy"] = (
        "Manifest split is authoritative; physical train/val folders are file stores only."
    )
    gates.append(
        _gate(
            "G0C.2_raw_png_lineage",
            "All raw CT/PET/mask PNG inputs are bijective and content-identical.",
            bool(manifest_pass and raw_pass),
            raw_evidence,
        )
    )

    cache_bijection_pass = False
    cache_content_pass = False
    cache_evidence: dict[str, Any] = {
        "cache_dir": _relative(cache_dir, root),
        "reason": "Manifest/raw gate failed; cache comparison not run.",
    }
    cache_rows: list[dict[str, Any]] = []
    cache_lineage: dict[str, Any] | None = None
    if manifest_pass and raw_pass:
        (
            cache_bijection_pass,
            cache_content_pass,
            cache_evidence,
            cache_rows,
            cache_lineage,
        ) = _validate_cache(
            cache_dir,
            manifest_by_id,
            raw_indexes,
            contract,
        )
        cache_evidence["cache_dir"] = _relative(cache_dir, root)
    gates.append(
        _gate(
            "G0C.3_cache_bijection",
            "Cache NPZ files and sidecars are bijective with manifest sample IDs.",
            cache_bijection_pass,
            cache_evidence,
        )
    )
    gates.append(
        _gate(
            "G0C.4_cache_numerical_equivalence",
            "Every tensor matches locked resize, PET inversion, normalization, and mask thresholding.",
            cache_content_pass,
            cache_evidence,
        )
    )
    _write_csv(output / "cache_sample_validation.csv", cache_rows)
    if cache_lineage is not None:
        _write_json(output / CACHE_LINEAGE_NAME, cache_lineage)

    cache_metadata_pass = False
    metadata_evidence: dict[str, Any] = {
        "path": _relative(cache_dir / CACHE_LINEAGE_NAME, root),
        "reason": "Cache content gate failed; lineage metadata cannot be sealed.",
        "written": False,
    }
    if cache_content_pass:
        cache_metadata_pass, metadata_evidence = (
            _seal_or_validate_cache_lineage(
                cache_dir, cache_lineage, seal=args.seal_cache
            )
        )
        metadata_evidence["path"] = _relative(
            cache_dir / CACHE_LINEAGE_NAME, root
        )
    gates.append(
        _gate(
            "G0C.5_cache_lineage_metadata",
            "A self-hashed cache-lineage file matches the currently validated cache.",
            cache_metadata_pass,
            metadata_evidence,
        )
    )

    checkpoint_status, checkpoint_evidence, checkpoint_rows = (
        _validate_checkpoints(args.checkpoint, root, cache_lineage)
    )
    _write_csv(output / "checkpoint_lineage.csv", checkpoint_rows)
    gates.append(
        {
            "gate_id": "G1.0_checkpoint_lineage",
            "description": "Supplied checkpoints embed all named data/cache fingerprints and are not fake-data runs.",
            "status": checkpoint_status,
            "evidence": checkpoint_evidence,
        }
    )

    cloud_training_pass = bool(
        contract_pass
        and manifest_pass
        and raw_pass
        and cache_bijection_pass
        and cache_content_pass
        and cache_metadata_pass
    )
    requested_checkpoint_gate_pass = checkpoint_status != "FAIL"
    overall_pass = bool(cloud_training_pass and requested_checkpoint_gate_pass)
    decision = {
        "stage": "00C_cloud_data_gate",
        "scope": "cloud_pretraining_validation_only",
        "decision": "PASS" if overall_pass else "FAIL",
        "data_gate": "PASS" if manifest_pass and raw_pass else "FAIL",
        "cloud_training_gate": "PASS" if cloud_training_pass else "FAIL",
        "checkpoint_lineage": checkpoint_status,
        "dataset_hypothesis_allowed": bool(
            contract.get("claim_boundary", {}).get(
                "dataset_hypothesis_allowed", False
            )
            and manifest_pass
            and raw_pass
        ),
        "training_allowed": cloud_training_pass,
        "formal_checkpoint_claims_allowed": checkpoint_status == "PASS",
        "model_mechanism_claims_allowed": False,
        "training_actions_performed": False,
        "cache_metadata_written": bool(metadata_evidence.get("written", False)),
        "fingerprints": {
            "manifest_semantic_sha256": contract["manifest"]["semantic_sha256"],
            "raw_png_combined_sha256": contract["raw_png"]["combined_sha256"],
            "preprocessing_config_sha256": contract[
                "preprocessing_config_sha256"
            ],
            "dataset_contract_sha256": contract["contract_sha256"],
            "cache_payload_sha256": (
                None
                if cache_lineage is None
                else cache_lineage["cache_payload_sha256"]
            ),
            "cache_metadata_sha256": (
                None
                if cache_lineage is None
                else cache_lineage["cache_metadata_sha256"]
            ),
        },
        "threshold_policy": {
            "inferential_thresholds_fitted_from_validation_or_test": False,
            "cache_numerical_atol": ENGINEERING_ATOL,
            "cache_numerical_atol_role": "fixed implementation-equivalence tolerance, not a model/mechanism threshold",
        },
        "statistical_policy": {
            "inferential_comparisons_performed": False,
            "patient_bootstrap_required": False,
            "reason": "Manifest, file hashes, bijection, and tensor equivalence are exhaustive deterministic identity gates, not sampled statistical comparisons.",
        },
        "gates": gates,
        "stop_rule": (
            "Cloud cache is locked; training may start, but model-mechanism claims remain forbidden."
            if cloud_training_pass
            else "Stop before training. Rebuild or correct the cache; do not relax a failed gate."
        ),
        "next_stage_allowed": cloud_training_pass,
    }
    _write_json(output / "decision.json", decision)

    execution_metadata = {
        "stage": decision["stage"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "root": root.as_posix(),
        "contract": contract_path.as_posix(),
        "manifest": manifest_path.as_posix(),
        "raw_root": raw_root.as_posix(),
        "cache_dir": cache_dir.as_posix(),
        "output": output.as_posix(),
        "checkpoint_arguments": list(args.checkpoint),
        "training_actions_performed": False,
    }
    _write_json(output / "execution_metadata.json", execution_metadata)

    failed_gates = [
        gate["gate_id"] for gate in gates if gate["status"] == "FAIL"
    ]
    report_lines = [
        "# Stage 0C cloud data gate",
        "",
        f"- Decision: **{decision['decision']}**",
        f"- Cloud training gate: **{decision['cloud_training_gate']}**",
        f"- Checkpoint lineage: **{checkpoint_status}**",
        "- Training actions performed: **false**",
        f"- Manifest samples: {len(manifest_rows)}",
        f"- Cache samples passed: {cache_evidence.get('sample_pass_count', 0)}",
        f"- Failed gates: {', '.join(failed_gates) if failed_gates else 'none'}",
        "",
        "## Claim boundary",
        "",
        decision["stop_rule"],
        "Passing Stage 0C proves data/cache lineage only. It does not prove a model mechanism or final performance.",
        "",
        "## Locked fingerprints",
        "",
    ]
    report_lines.extend(
        f"- `{key}`: `{value}`" for key, value in decision["fingerprints"].items()
    )
    (output / "report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return decision


def _default_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    initial_root = _default_root()
    parser = argparse.ArgumentParser(
        description="Verify and optionally seal the Stage 0C cloud PNG-cache gate."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=initial_root,
        help="Repository root (default: inferred from this script).",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=initial_root / "configs" / "dataset_contract_stage0a_v1.json",
        help="Locked Stage 0A dataset contract.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Authoritative manifest override; default comes from the contract.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help=(
            "Raw PNG root override. Without it, the detector safely checks the "
            "contract path plus common repo/main_data locations; CT_PET_RAW_ROOT "
            "may also provide a candidate."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=initial_root / "cache" / "tensors_main",
        help="Cloud-built NPZ cache to verify.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=initial_root
        / "results"
        / "mechanism_validation"
        / "00C_cloud_data_gate",
        help="Directory receiving decision.json and audit tables.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Optional checkpoint file/directory. Repeat to inspect multiple inputs.",
    )
    parser.add_argument(
        "--seal-cache",
        action="store_true",
        help=(
            "After all content checks pass, atomically write/refresh "
            "<cache-dir>/cache_lineage.json. Required for cloud_training_gate PASS "
            "when no matching sealed metadata exists."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.root = args.root.resolve()
    if not args.contract.is_absolute():
        args.contract = args.root / args.contract
    if args.manifest is not None and not args.manifest.is_absolute():
        args.manifest = args.root / args.manifest
    if args.raw_root is not None and not args.raw_root.is_absolute():
        args.raw_root = args.root / args.raw_root
    if not args.cache_dir.is_absolute():
        args.cache_dir = args.root / args.cache_dir
    if not args.output.is_absolute():
        args.output = args.root / args.output
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        decision = run(args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        decision = {
            "stage": "00C_cloud_data_gate",
            "scope": "cloud_pretraining_validation_only",
            "decision": "FAIL",
            "data_gate": "NOT_EVALUATED",
            "cloud_training_gate": "FAIL",
            "checkpoint_lineage": "NOT_EVALUATED",
            "training_allowed": False,
            "formal_checkpoint_claims_allowed": False,
            "model_mechanism_claims_allowed": False,
            "training_actions_performed": False,
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "stop_rule": "Stop before training; correct the detector input or data error.",
        }
        _write_json(args.output / "decision.json", decision)
        print(json.dumps(decision, indent=2, ensure_ascii=False))
        return 2
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0 if decision["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
