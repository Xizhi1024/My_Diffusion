"""Stage 0A local data-lineage audit for CT->PET mechanism validation.

This script is deliberately diagnostic.  It does not build a cache, alter a
split, load a model for inference, or change training code.  The authoritative
split is ``main_data/split_manifest.csv``.  The raw PNG store may be laid out
under local ``Data/data`` or cloud ``main_data``; identity is established by
the locked content fingerprint rather than by the physical directory name.

All image comparisons are first aggregated within patient and then bootstrapped
over patients.  No validation result is used to choose a preprocessing rule.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import re
import sys
import tomllib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage


SCHEMA_VERSION = 1
AUDIT_SEED = 20260723
BOOTSTRAP_REPLICATES = 10_000
IMAGE_SIZE = 192
MANIFEST_COLUMNS = ("sample_id", "patient_id", "slice_id", "split", "cache_path")
MODALITY_DIRS = {"ct": "ct", "pet": "pet", "mask": "label"}
SAMPLE_ID_RE = re.compile(r"^(\d{3})(\d{3})$")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = sorted({key for row in rows for key in row}) if rows else ()
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_locked_contract(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read locked dataset contract {path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Locked dataset contract must be an object: {path}")
    body = dict(payload)
    claimed = str(body.pop("contract_sha256", ""))
    computed = _sha256_bytes(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if not claimed or claimed != computed:
        raise RuntimeError(
            "Locked dataset contract self-hash mismatch: "
            f"claimed={claimed!r}, computed={computed!r}"
        )
    if payload.get("contract_status") != "LOCKED":
        raise RuntimeError(
            "Locked dataset contract has invalid status: "
            f"{payload.get('contract_status')!r}"
        )
    return payload


def _locked_contract_mismatches(
    contract: Mapping[str, Any],
    *,
    manifest_semantic_sha256: str,
    raw_png_combined_sha256: str,
    raw_png_file_count: int,
    modality_counts: Mapping[str, int],
    split_sample_counts: Mapping[str, int],
    split_patient_counts: Mapping[str, int],
    preprocessing_config_sha256: str,
) -> list[str]:
    """Compare audited content to a path-portable locked contract."""

    mismatches: list[str] = []

    def check(label: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            mismatches.append(
                f"{label}: audited={actual!r}, contract={expected!r}"
            )

    check(
        "manifest.semantic_sha256",
        manifest_semantic_sha256,
        contract.get("manifest", {}).get("semantic_sha256"),
    )
    check(
        "raw_png.combined_sha256",
        raw_png_combined_sha256,
        contract.get("raw_png", {}).get("combined_sha256"),
    )
    check(
        "raw_png.file_count",
        int(raw_png_file_count),
        contract.get("raw_png", {}).get("file_count"),
    )
    for modality in MODALITY_DIRS:
        check(
            f"raw_png.modalities.{modality}",
            int(modality_counts.get(modality, 0)),
            contract.get("raw_png", {}).get("modalities", {}).get(modality),
        )
    for split in ("train", "val", "test"):
        expected = contract.get("splits", {}).get(split, {})
        check(
            f"splits.{split}.samples",
            int(split_sample_counts.get(split, 0)),
            expected.get("samples"),
        )
        check(
            f"splits.{split}.patients",
            int(split_patient_counts.get(split, 0)),
            expected.get("patients"),
        )
    check(
        "preprocessing_config_sha256",
        preprocessing_config_sha256,
        contract.get("preprocessing_config_sha256"),
    )
    return mismatches


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _resolve_from_root(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = [
            {key: "" if value is None else str(value).strip() for key, value in row.items()}
            for row in reader
        ]
    return rows, columns


def _manifest_fingerprint(path: Path, rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    canonical_rows = [
        {column: row.get(column, "") for column in MANIFEST_COLUMNS}
        for row in sorted(rows, key=lambda item: item.get("sample_id", ""))
    ]
    canonical_payload = {
        "schema_version": SCHEMA_VERSION,
        "columns": list(MANIFEST_COLUMNS),
        "rows": canonical_rows,
    }
    canonical_bytes = json.dumps(
        canonical_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "algorithm": "sha256",
        "canonicalization": "UTF-8 JSON; sorted sample_id; fixed columns; sorted keys; compact separators",
        "semantic_sha256": _sha256_bytes(canonical_bytes),
        "source_file_sha256": _sha256_file(path),
        "source_file": path.as_posix(),
        "row_count": len(rows),
        "columns": list(MANIFEST_COLUMNS),
    }


def _index_pngs(raw_root: Path, modality_dir: str) -> tuple[dict[str, Path], dict[str, list[str]]]:
    candidates: dict[str, list[Path]] = defaultdict(list)
    for physical_split in ("train", "val", "test"):
        folder = raw_root / physical_split / modality_dir
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.png")):
            candidates[path.stem].append(path)
    duplicates = {
        sample_id: [path.as_posix() for path in paths]
        for sample_id, paths in candidates.items()
        if len(paths) != 1
    }
    return {sample_id: paths[0] for sample_id, paths in candidates.items()}, duplicates


def _load_grayscale(path: Path) -> tuple[np.ndarray, str, tuple[int, int]]:
    with Image.open(path) as image:
        mode = image.mode
        size = tuple(int(value) for value in image.size)
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    return array, mode, size


def _resize(array: np.ndarray, *, nearest: bool) -> np.ndarray:
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    image = Image.fromarray(array, mode="L")
    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), resample)
    return np.asarray(image, dtype=np.uint8)


def _local_ring(mask: np.ndarray, radius: int = 12) -> np.ndarray:
    return ndimage.binary_dilation(mask, iterations=radius) & ~mask


def _border_mean(image: np.ndarray, width: int = 8) -> float:
    border = np.zeros(image.shape, dtype=bool)
    border[:width] = True
    border[-width:] = True
    border[:, :width] = True
    border[:, -width:] = True
    return float(image[border].mean())


def _ncc(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.astype(np.float64).ravel()
    right_flat = right.astype(np.float64).ravel()
    left_flat -= left_flat.mean()
    right_flat -= right_flat.mean()
    denominator = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    return float(left_flat @ right_flat / denominator) if denominator > 1e-12 else 0.0


def _patient_aggregate(rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["patient_id"]))].append(row)
    result: list[dict[str, Any]] = []
    for (split, patient_id), patient_rows in sorted(grouped.items()):
        record: dict[str, Any] = {
            "split": split,
            "patient_id": patient_id,
            "sample_count": len(patient_rows),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in patient_rows], dtype=np.float64)
            values = values[np.isfinite(values)]
            record[metric] = float(values.mean()) if values.size else float("nan")
        result.append(record)
    return result


def _bootstrap_mean(
    patient_rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    split: str,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(
        [float(row[metric]) for row in patient_rows if row["split"] == split],
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "split": split,
            "metric": metric,
            "patients": 0,
            "estimate": None,
            "ci95_low": None,
            "ci95_high": None,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "seed": seed,
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_REPLICATES, values.size))
    bootstrap = values[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "split": split,
        "metric": metric,
        "patients": int(values.size),
        "estimate": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
        "unit": "patient (slice metrics averaged within patient before resampling)",
    }


def _bootstrap_table(
    patient_rows: Sequence[Mapping[str, Any]], metrics: Sequence[str], seed_offset: int
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for split_index, split in enumerate(("train", "val")):
        for metric_index, metric in enumerate(metrics):
            result.append(
                _bootstrap_mean(
                    patient_rows,
                    metric,
                    split=split,
                    seed=AUDIT_SEED + seed_offset + split_index * 100 + metric_index,
                )
            )
    return result


def _decode_json_array(value: np.ndarray) -> dict[str, Any]:
    try:
        if value.dtype == np.uint8:
            text = bytes(value.tolist()).decode("utf-8")
        else:
            text = str(value.item())
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _nested_fingerprint_entries(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            key_lower = str(key).lower()
            if "fingerprint" in key_lower and isinstance(child, (str, int, float, bool)):
                result.append((path, str(child)))
            result.extend(_nested_fingerprint_entries(child, path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            result.extend(_nested_fingerprint_entries(child, f"{prefix}[{index}]"))
    return result


def _checkpoint_embedded_metadata(path: Path) -> dict[str, Any]:
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            return {"load_status": "not_a_dict"}
        config = checkpoint.get("config")
        config = config if isinstance(config, dict) else {}
        data = config.get("data") if isinstance(config.get("data"), dict) else {}
        fingerprints = _nested_fingerprint_entries(checkpoint)
        return {
            "load_status": "ok",
            "top_level_keys": ";".join(sorted(str(key) for key in checkpoint)),
            "embedded_config": config,
            "embedded_cache_dir": data.get("cache_dir", ""),
            "embedded_split_manifest": data.get("split_manifest", ""),
            "embedded_use_fake_data": data.get("use_fake_data", ""),
            "fingerprint_entries": fingerprints,
        }
    except Exception as exc:
        return {"load_status": f"error: {type(exc).__name__}: {exc}"}


def _audit_checkpoints(root: Path, manifest_fingerprint: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    extensions = {".pt", ".pth", ".ckpt"}
    for path in sorted(
        candidate
        for candidate in (root / "checkpoints").rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in extensions
    ):
        embedded = _checkpoint_embedded_metadata(path)
        config = embedded.pop("embedded_config", {})
        config_source = "embedded" if config else ""
        sibling_config_path = path.parent / "resolved_config.yaml"
        if not config and sibling_config_path.exists():
            try:
                parsed = yaml.safe_load(sibling_config_path.read_text(encoding="utf-8"))
                config = parsed if isinstance(parsed, dict) else {}
                config_source = "sibling_resolved_config"
            except Exception:
                config = {}
        data = config.get("data") if isinstance(config.get("data"), dict) else {}
        cache_dir = str(data.get("cache_dir", embedded.get("embedded_cache_dir", "")) or "")
        split_manifest = str(
            data.get("split_manifest", embedded.get("embedded_split_manifest", "")) or ""
        )
        use_fake = data.get("use_fake_data", embedded.get("embedded_use_fake_data", ""))
        fingerprint_entries = embedded.get("fingerprint_entries", [])
        exact_fingerprint = any(value == manifest_fingerprint for _, value in fingerprint_entries)
        cache_exists = bool(cache_dir) and _resolve_from_root(cache_dir, root).is_dir()
        manifest_exists = bool(split_manifest) and _resolve_from_root(split_manifest, root).is_file()
        eligible = bool(exact_fingerprint and cache_exists and manifest_exists and use_fake is not True)
        rows.append(
            {
                "checkpoint": _relative(path, root),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "load_status": embedded.get("load_status", ""),
                "top_level_keys": embedded.get("top_level_keys", ""),
                "config_source": config_source,
                "declared_cache_dir": cache_dir,
                "declared_cache_exists_from_repo_root": cache_exists,
                "declared_split_manifest": split_manifest,
                "declared_manifest_exists_from_repo_root": manifest_exists,
                "use_fake_data": use_fake,
                "fingerprint_entries": json.dumps(fingerprint_entries, ensure_ascii=False),
                "matches_audited_manifest_fingerprint": exact_fingerprint,
                "eligible_for_formal_mechanism_conclusion": eligible,
                "quarantine_reason": "" if eligible else "no exact audited data-lineage proof",
            }
        )
    return rows


def _gate(
    gate_id: str,
    description: str,
    passed: bool | None,
    evidence: Mapping[str, Any],
    *,
    scope: str,
    blocks_local_data_gate: bool,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "description": description,
        "scope": scope,
        "blocks_local_data_gate": blocks_local_data_gate,
        "status": "deferred" if passed is None else ("pass" if passed else "fail"),
        "evidence": dict(evidence),
    }


def run(
    root: Path,
    output: Path,
    *,
    audit_checkpoint_inventory: bool = True,
    raw_root_override: Path | None = None,
    manifest_path_override: Path | None = None,
    locked_contract_path: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    manifest_path = (
        manifest_path_override.resolve()
        if manifest_path_override is not None
        else root / "main_data" / "split_manifest.csv"
    )
    raw_root = (
        raw_root_override.resolve()
        if raw_root_override is not None
        else root / "Data" / "data"
    )
    locked_contract = (
        _load_locked_contract(locked_contract_path.resolve())
        if locked_contract_path is not None
        else None
    )
    config_path = root / "configs" / "experiments" / "slmf_png_spectral_router_v5.yaml"
    pixi_path = root / "pixi.toml"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)

    manifest, manifest_columns = _read_manifest(manifest_path)
    fingerprint = _manifest_fingerprint(manifest_path, manifest)
    _write_json(output / "manifest_fingerprint.json", fingerprint)

    sample_counts = Counter(row["sample_id"] for row in manifest)
    duplicate_samples = sorted(sample_id for sample_id, count in sample_counts.items() if count != 1)
    invalid_rows: list[dict[str, Any]] = []
    patient_splits: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        match = SAMPLE_ID_RE.fullmatch(row.get("sample_id", ""))
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
        if row.get("split") not in {"train", "val", "test"}:
            reasons.append("invalid_split")
        if reasons:
            invalid_rows.append({**row, "reasons": ";".join(reasons)})
        patient_splits[row.get("patient_id", "")].add(row.get("split", ""))
    patient_overlap = {
        patient_id: sorted(splits)
        for patient_id, splits in patient_splits.items()
        if len(splits) > 1
    }

    indexes: dict[str, dict[str, Path]] = {}
    raw_duplicates: dict[str, dict[str, list[str]]] = {}
    for modality, folder in MODALITY_DIRS.items():
        indexes[modality], raw_duplicates[modality] = _index_pngs(raw_root, folder)

    manifest_ids = {row["sample_id"] for row in manifest}
    missing_raw = {
        modality: sorted(manifest_ids - set(index)) for modality, index in indexes.items()
    }
    extra_raw = {
        modality: sorted(set(index) - manifest_ids) for modality, index in indexes.items()
    }

    lineage_rows: list[dict[str, Any]] = []
    image_metric_rows: list[dict[str, Any]] = []
    image_errors: list[dict[str, str]] = []
    raw_hash_entries: list[str] = []
    manifest_by_id = {row["sample_id"]: row for row in manifest}
    raw_arrays: dict[str, dict[str, np.ndarray]] = {}

    for row in sorted(manifest, key=lambda item: item["sample_id"]):
        sample_id = row["sample_id"]
        paths = {modality: indexes[modality].get(sample_id) for modality in MODALITY_DIRS}
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "patient_id": row["patient_id"],
            "manifest_split": row["split"],
            "slice_id": row["slice_id"],
            "manifest_cache_path": row["cache_path"],
        }
        if any(path is None for path in paths.values()):
            for modality, path in paths.items():
                record[f"{modality}_path"] = "" if path is None else _relative(path, root)
            lineage_rows.append(record)
            continue
        try:
            loaded: dict[str, np.ndarray] = {}
            modes: dict[str, str] = {}
            sizes: dict[str, tuple[int, int]] = {}
            for modality, path_or_none in paths.items():
                assert path_or_none is not None
                array, mode, size = _load_grayscale(path_or_none)
                loaded[modality] = array
                modes[modality] = mode
                sizes[modality] = size
                file_hash = _sha256_file(path_or_none)
                record[f"{modality}_path"] = _relative(path_or_none, root)
                record[f"{modality}_physical_split"] = path_or_none.parent.parent.name
                record[f"{modality}_sha256"] = file_hash
                record[f"{modality}_mode"] = mode
                record[f"{modality}_size"] = f"{size[0]}x{size[1]}"
                raw_hash_entries.append(f"{sample_id}|{modality}|{file_hash}")

            ct_192 = _resize(loaded["ct"], nearest=False)
            pet_192 = _resize(loaded["pet"], nearest=False)
            mask_192 = _resize(loaded["mask"], nearest=True)
            mask_positive = mask_192 > 0
            mask_midpoint = mask_192 > 127
            ring = _local_ring(mask_midpoint)
            ct_unit = ct_192.astype(np.float32) / 255.0
            raw_pet = pet_192.astype(np.float32) / 255.0
            pet_uptake = 1.0 - raw_pet
            registration_ncc = _ncc(ct_unit, pet_uptake)
            registration_ncc_lr = _ncc(np.fliplr(ct_unit), pet_uptake)
            registration_ncc_ud = _ncc(np.flipud(ct_unit), pet_uptake)
            lesion_mean = float(raw_pet[mask_midpoint].mean()) if mask_midpoint.any() else float("nan")
            ring_mean = float(raw_pet[ring].mean()) if ring.any() else float("nan")
            raw_difference = lesion_mean - ring_mean
            raw_mask_midpoint = loaded["mask"] > 127
            raw_mask_fraction = float(raw_mask_midpoint.mean())
            resized_mask_fraction = float(mask_midpoint.mean())
            midpoint_area = int(mask_midpoint.sum())
            positive_area = int(mask_positive.sum())
            metric_row = {
                "sample_id": sample_id,
                "patient_id": row["patient_id"],
                "split": row["split"],
                "registration_ncc_unflipped": registration_ncc,
                "registration_unflipped_minus_lr": registration_ncc - registration_ncc_lr,
                "registration_unflipped_minus_ud": registration_ncc - registration_ncc_ud,
                "pet_raw_lesion_minus_ring": raw_difference,
                "pet_inverted_lesion_minus_ring": -raw_difference,
                "pet_raw_border_minus_lesion": _border_mean(raw_pet) - lesion_mean,
                "pet_raw_border_mean": _border_mean(raw_pet),
                "mask_area_positive": positive_area,
                "mask_area_midpoint": midpoint_area,
                "mask_positive_minus_midpoint_fraction": (
                    float(positive_area - midpoint_area) / max(midpoint_area, 1)
                ),
                "mask_resized_minus_raw_area_fraction": resized_mask_fraction - raw_mask_fraction,
            }
            image_metric_rows.append(metric_row)
            raw_arrays[sample_id] = {
                "ct_png_expected": ct_192.astype(np.float32) / 127.5 - 1.0,
                "pet_png_inverted_expected": 1.0 - pet_192.astype(np.float32) / 127.5,
                "pet_png_noninverted_expected": pet_192.astype(np.float32) / 127.5 - 1.0,
                "mask_positive_expected": mask_positive.astype(np.float32),
                "mask_midpoint_expected": mask_midpoint.astype(np.float32),
            }
            record.update(
                {
                    "label_unique_values": int(np.unique(loaded["mask"]).size),
                    "label_has_intermediate_values": bool(
                        np.any((loaded["mask"] > 0) & (loaded["mask"] < 255))
                    ),
                    "mask_area_positive_192": positive_area,
                    "mask_area_midpoint_192": midpoint_area,
                    "manifest_matches_physical_split": all(
                        path.parent.parent.name == row["split"]
                        for path in paths.values()
                        if path is not None
                    ),
                }
            )
        except Exception as exc:
            image_errors.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
        lineage_rows.append(record)

    _write_csv(output / "sample_lineage.csv", lineage_rows)
    physical_mismatches = [
        row for row in lineage_rows if row.get("manifest_matches_physical_split") is False
    ]
    _write_csv(output / "physical_split_mismatches.csv", physical_mismatches)
    _write_csv(output / "invalid_manifest_rows.csv", invalid_rows)

    raw_dataset_fingerprint = {
        "algorithm": "sha256",
        "canonicalization": "sorted sample_id|modality|file_sha256 records joined with LF",
        "sha256": _sha256_bytes("\n".join(sorted(raw_hash_entries)).encode("utf-8")),
        "file_count": len(raw_hash_entries),
    }
    _write_json(output / "raw_png_fingerprint.json", raw_dataset_fingerprint)

    pet_metrics = (
        "pet_raw_lesion_minus_ring",
        "pet_inverted_lesion_minus_ring",
        "pet_raw_border_minus_lesion",
        "pet_raw_border_mean",
    )
    mask_metrics = (
        "mask_area_positive",
        "mask_area_midpoint",
        "mask_positive_minus_midpoint_fraction",
        "mask_resized_minus_raw_area_fraction",
    )
    registration_metrics = (
        "registration_ncc_unflipped",
        "registration_unflipped_minus_lr",
        "registration_unflipped_minus_ud",
    )
    pet_patient_rows = _patient_aggregate(image_metric_rows, pet_metrics)
    mask_patient_rows = _patient_aggregate(image_metric_rows, mask_metrics)
    registration_patient_rows = _patient_aggregate(image_metric_rows, registration_metrics)
    pet_bootstrap = _bootstrap_table(pet_patient_rows, pet_metrics, 0)
    mask_bootstrap = _bootstrap_table(mask_patient_rows, mask_metrics, 1_000)
    registration_bootstrap = _bootstrap_table(
        registration_patient_rows, registration_metrics, 1_500
    )
    _write_csv(output / "pet_polarity_patient_metrics.csv", pet_patient_rows)
    _write_csv(output / "pet_polarity_bootstrap_95ci.csv", pet_bootstrap)
    _write_csv(output / "mask_resize_patient_metrics.csv", mask_patient_rows)
    _write_csv(output / "mask_resize_bootstrap_95ci.csv", mask_bootstrap)
    _write_csv(output / "registration_patient_metrics.csv", registration_patient_rows)
    _write_csv(output / "registration_bootstrap_95ci.csv", registration_bootstrap)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config = config.get("data", {}) if isinstance(config, dict) else {}
    configured_cache_dir = _resolve_from_root(str(data_config.get("cache_dir", "")), root)
    configured_manifest_path = _resolve_from_root(str(data_config.get("split_manifest", "")), root)

    manifest_cache_dirs = sorted(
        {
            _resolve_from_root(row["cache_path"], root).parent.resolve()
            for row in manifest
            if row.get("cache_path")
        },
        key=lambda path: path.as_posix(),
    )
    cache_files_by_id: dict[str, list[Path]] = defaultdict(list)
    for directory in manifest_cache_dirs:
        if directory.is_dir():
            for path in directory.glob("*.npz"):
                cache_files_by_id[path.stem].append(path)
    cache_duplicate_ids = {
        sample_id: [path.as_posix() for path in paths]
        for sample_id, paths in cache_files_by_id.items()
        if len(paths) != 1
    }
    cache_ids = set(cache_files_by_id)
    cache_missing_ids = sorted(manifest_ids - cache_ids)
    cache_extra_ids = sorted(cache_ids - manifest_ids)
    cache_overlap_ids = sorted(manifest_ids & cache_ids)

    cache_inventory_rows: list[dict[str, Any]] = []
    cache_comparison_rows: list[dict[str, Any]] = []
    cache_errors: list[dict[str, str]] = []
    for sample_id in sorted(cache_files_by_id):
        path = cache_files_by_id[sample_id][0]
        inventory: dict[str, Any] = {
            "sample_id": sample_id,
            "path": _relative(path, root),
            "in_main_manifest": sample_id in manifest_ids,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        try:
            with np.load(path, allow_pickle=False) as archive:
                inventory["keys"] = ";".join(sorted(archive.files))
                scale_meta = (
                    _decode_json_array(archive["scale_meta_json"])
                    if "scale_meta_json" in archive.files
                    else {}
                )
                inventory["pet_physical_kind"] = scale_meta.get("pet_physical_kind", "")
                inventory["suv_ok"] = scale_meta.get("suv_ok", "")
                inventory["pet_invert"] = scale_meta.get("pet_invert", "")
                inventory["scale_meta_patient_id"] = scale_meta.get("patient_id", "")
                inventory["scale_meta_slice_id"] = scale_meta.get("slice_id", "")
                if sample_id in manifest_by_id and sample_id in raw_arrays:
                    main_row = manifest_by_id[sample_id]
                    expected = raw_arrays[sample_id]
                    arrays = {
                        key: np.asarray(archive[key], dtype=np.float32).squeeze()
                        for key in ("ct", "pet", "mask")
                        if key in archive.files
                    }
                    if all(key in arrays for key in ("ct", "pet", "mask")) and all(
                        arrays[key].shape == (IMAGE_SIZE, IMAGE_SIZE) for key in arrays
                    ):
                        comparison = {
                            "sample_id": sample_id,
                            "patient_id": main_row["patient_id"],
                            "split": main_row["split"],
                            "cache_ct_mae_to_png": float(
                                np.mean(np.abs(arrays["ct"] - expected["ct_png_expected"]))
                            ),
                            "cache_pet_mae_to_inverted_png": float(
                                np.mean(
                                    np.abs(arrays["pet"] - expected["pet_png_inverted_expected"])
                                )
                            ),
                            "cache_pet_mae_to_noninverted_png": float(
                                np.mean(
                                    np.abs(
                                        arrays["pet"] - expected["pet_png_noninverted_expected"]
                                    )
                                )
                            ),
                            "cache_mask_disagreement_positive": float(
                                np.mean(
                                    (arrays["mask"] > 0.5)
                                    != (expected["mask_positive_expected"] > 0.5)
                                )
                            ),
                            "cache_mask_disagreement_midpoint": float(
                                np.mean(
                                    (arrays["mask"] > 0.5)
                                    != (expected["mask_midpoint_expected"] > 0.5)
                                )
                            ),
                        }
                        cache_comparison_rows.append(comparison)
        except Exception as exc:
            cache_errors.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
        cache_inventory_rows.append(inventory)
    _write_csv(output / "cache_inventory.csv", cache_inventory_rows)
    cache_comparison_metrics = (
        "cache_ct_mae_to_png",
        "cache_pet_mae_to_inverted_png",
        "cache_pet_mae_to_noninverted_png",
        "cache_mask_disagreement_positive",
        "cache_mask_disagreement_midpoint",
    )
    cache_patient_rows = _patient_aggregate(cache_comparison_rows, cache_comparison_metrics)
    cache_bootstrap = _bootstrap_table(cache_patient_rows, cache_comparison_metrics, 2_000)
    _write_csv(output / "cache_png_comparison_patient_metrics.csv", cache_patient_rows)
    _write_csv(output / "cache_png_comparison_bootstrap_95ci.csv", cache_bootstrap)

    with pixi_path.open("rb") as handle:
        pixi = tomllib.load(handle)
    png_cache_task = str(pixi.get("tasks", {}).get("data-cache-png-main", ""))
    from src.data.png_cache import build_png_cache

    cache_builder_signature = inspect.signature(build_png_cache)
    builder_pet_invert_default = cache_builder_signature.parameters["pet_invert"].default
    builder_mask_threshold_default = cache_builder_signature.parameters["mask_threshold"].default
    normalized_cache_task = png_cache_task.replace("\\", "/")
    task_has_pet_invert = "--pet-invert" in png_cache_task
    task_has_midpoint_threshold = bool(
        re.search(r"--mask-threshold(?:=|\s+)0?\.5(?:\s|$)", png_cache_task)
    )
    task_png_root_match = any(
        marker in normalized_cache_task
        for marker in ("--png-root main_data", "--png-root Data/data")
    )

    checkpoint_rows = (
        _audit_checkpoints(root, fingerprint["semantic_sha256"])
        if audit_checkpoint_inventory
        else []
    )
    _write_csv(output / "checkpoint_inventory.csv", checkpoint_rows)

    split_sample_counts = Counter(row["split"] for row in manifest)
    split_patient_counts = {
        split: len({row["patient_id"] for row in manifest if row["split"] == split})
        for split in ("train", "val", "test")
    }
    raw_input_sizes = {
        modality: dict(
            Counter(
                row.get(f"{modality}_size", "")
                for row in lineage_rows
                if row.get(f"{modality}_size")
            )
        )
        for modality in MODALITY_DIRS
    }
    midpoint_areas = np.asarray(
        [float(row["mask_area_midpoint"]) for row in image_metric_rows], dtype=np.float64
    )
    positive_areas = np.asarray(
        [float(row["mask_area_positive"]) for row in image_metric_rows], dtype=np.float64
    )
    train_pet_raw = next(
        row for row in pet_bootstrap if row["split"] == "train" and row["metric"] == "pet_raw_lesion_minus_ring"
    )
    train_pet_inverted = next(
        row
        for row in pet_bootstrap
        if row["split"] == "train" and row["metric"] == "pet_inverted_lesion_minus_ring"
    )
    train_registration_lr = next(
        row
        for row in registration_bootstrap
        if row["split"] == "train"
        and row["metric"] == "registration_unflipped_minus_lr"
    )
    train_registration_ud = next(
        row
        for row in registration_bootstrap
        if row["split"] == "train"
        and row["metric"] == "registration_unflipped_minus_ud"
    )

    manifest_gate = (
        all(column in manifest_columns for column in MANIFEST_COLUMNS)
        and not duplicate_samples
        and not invalid_rows
        and len(manifest) > 0
    )
    patient_gate = not patient_overlap
    raw_pair_gate = (
        all(not values for values in missing_raw.values())
        and all(not values for values in extra_raw.values())
        and all(not values for values in raw_duplicates.values())
        and not image_errors
    )
    registration_gate = bool(
        train_registration_lr["ci95_low"] is not None
        and train_registration_ud["ci95_low"] is not None
        and float(train_registration_lr["ci95_low"]) > 0.0
        and float(train_registration_ud["ci95_low"]) > 0.0
    )
    pet_polarity_gate = bool(
        train_pet_raw["ci95_high"] is not None
        and train_pet_inverted["ci95_low"] is not None
        and float(train_pet_raw["ci95_high"]) < 0.0
        and float(train_pet_inverted["ci95_low"]) > 0.0
    )
    mask_rule_characterized = bool(
        image_metric_rows
        and all(float(row["mask_area_midpoint"]) > 0 for row in image_metric_rows)
        and all(float(row["mask_area_positive"]) > 0 for row in image_metric_rows)
    )
    cache_build_route_gate = bool(
        task_png_root_match and task_has_pet_invert and task_has_midpoint_threshold
    )
    configured_cache_gate = bool(
        configured_cache_dir.is_dir()
        and configured_manifest_path.resolve() == manifest_path.resolve()
        and not cache_missing_ids
        and not cache_extra_ids
        and not cache_duplicate_ids
    )
    cache_png_provenance_gate = bool(
        configured_cache_gate
        and cache_inventory_rows
        and all(row.get("pet_physical_kind") == "png_intensity" for row in cache_inventory_rows)
        and all(row.get("pet_invert") is True for row in cache_inventory_rows)
    )
    eligible_checkpoints = [
        row for row in checkpoint_rows if row["eligible_for_formal_mechanism_conclusion"]
    ]
    checkpoint_gate = bool(eligible_checkpoints)

    local_gates = [
        _gate(
            "G0A.1_manifest_integrity",
            "Manifest schema, sample identity, and uniqueness are valid.",
            manifest_gate,
            {
                "rows": len(manifest),
                "duplicate_sample_ids": len(duplicate_samples),
                "invalid_rows": len(invalid_rows),
                "semantic_sha256": fingerprint["semantic_sha256"],
            },
            scope="local_raw_data",
            blocks_local_data_gate=True,
        ),
        _gate(
            "G0A.2_patient_disjointness",
            "No patient appears in more than one authoritative split.",
            patient_gate,
            {"overlap_patients": len(patient_overlap), "split_patient_counts": split_patient_counts},
            scope="local_raw_data",
            blocks_local_data_gate=True,
        ),
        _gate(
            "G0A.3_manifest_raw_png_bijection",
            "Every manifest sample has exactly one CT, PET, and mask PNG and there are no extras.",
            raw_pair_gate,
            {
                "missing_by_modality": {key: len(value) for key, value in missing_raw.items()},
                "extra_by_modality": {key: len(value) for key, value in extra_raw.items()},
                "duplicate_by_modality": {key: len(value) for key, value in raw_duplicates.items()},
                "read_errors": len(image_errors),
            },
            scope="local_raw_data",
            blocks_local_data_gate=True,
        ),
        _gate(
            "G0A.4_registration_orientation",
            "Unflipped CT/PET pairing is favored over left-right and up-down flips on manifest-train patients.",
            registration_gate,
            {
                "unflipped_minus_lr": train_registration_lr,
                "unflipped_minus_ud": train_registration_ud,
                "interpretation": "cross-modal normalized correlation is a registration proxy, not a causal or mutual-information claim",
            },
            scope="local_raw_data",
            blocks_local_data_gate=True,
        ),
        _gate(
            "G0A.5_pet_polarity",
            "Manifest-train patient bootstrap confirms white-canvas/dark-uptake PET must be inverted.",
            pet_polarity_gate,
            {"raw_lesion_minus_ring": train_pet_raw, "inverted_lesion_minus_ring": train_pet_inverted},
            scope="local_raw_data",
            blocks_local_data_gate=True,
        ),
        _gate(
            "G0A.6_mask_rule_characterized",
            "Nearest-neighbor 512->192 resizing preserves non-empty masks and threshold sensitivity is quantified.",
            mask_rule_characterized,
            {
                "median_area_positive": float(np.median(positive_areas)),
                "median_area_midpoint": float(np.median(midpoint_areas)),
                "different_area_samples": int(np.sum(positive_areas != midpoint_areas)),
            },
            scope="local_raw_data",
            blocks_local_data_gate=True,
        ),
    ]
    deferred_gates = [
        _gate(
            "G0C.1_png_cache_build_route",
            "Cloud cache construction uses the locked preprocessing contract.",
            None,
            {
                "current_local_task_ready": cache_build_route_gate,
                "task": png_cache_task,
                "task_uses_supported_png_root": task_png_root_match,
                "task_has_pet_invert": task_has_pet_invert,
                "task_has_mask_threshold_0_5": task_has_midpoint_threshold,
                "builder_pet_invert_default": builder_pet_invert_default,
                "builder_mask_threshold_default": builder_mask_threshold_default,
            },
            scope="cloud_cache",
            blocks_local_data_gate=False,
        ),
        _gate(
            "G0C.2_configured_cache_bijection",
            "Cloud-built cache is bijective with the authoritative manifest.",
            None,
            {
                "current_configured_cache_ready": configured_cache_gate,
                "configured_cache_dir": _relative(configured_cache_dir, root),
                "configured_cache_exists": configured_cache_dir.is_dir(),
                "legacy_manifest_cache_overlap": len(cache_overlap_ids),
                "legacy_manifest_cache_missing": len(cache_missing_ids),
                "legacy_manifest_cache_extras": len(cache_extra_ids),
            },
            scope="cloud_cache",
            blocks_local_data_gate=False,
        ),
        _gate(
            "G0C.3_cache_png_provenance",
            "Cloud cache metadata matches manifest, raw-PNG, and preprocessing fingerprints.",
            None,
            {
                "current_configured_cache_provenance_ready": cache_png_provenance_gate,
                "legacy_cache_pet_physical_kind_counts": dict(
                    Counter(str(row.get("pet_physical_kind", "")) for row in cache_inventory_rows)
                ),
                "legacy_cache_suv_ok_counts": dict(
                    Counter(str(row.get("suv_ok", "")) for row in cache_inventory_rows)
                ),
            },
            scope="cloud_cache",
            blocks_local_data_gate=False,
        ),
        _gate(
            "G1.0_checkpoint_lineage",
            "New cloud checkpoints embed the locked dataset-contract and cache fingerprints.",
            None,
            {
                "legacy_checkpoint_count": len(checkpoint_rows),
                "legacy_eligible_checkpoint_count": len(eligible_checkpoints),
                "legacy_policy": "quarantine",
                "current_legacy_checkpoint_gate_ready": checkpoint_gate,
            },
            scope="cloud_training",
            blocks_local_data_gate=False,
        ),
    ]
    preprocessing_config = {
        "schema_version": SCHEMA_VERSION,
        "output_image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "ct": {
            "operation_order": ["decode", "resize", "normalize"],
            "decode": "PIL.Image.convert('L')",
            "resize": "PIL.Image.Resampling.BILINEAR",
            "normalization": "uint8 / 127.5 - 1.0",
            "output_dtype": "float32",
            "output_range": [-1.0, 1.0],
        },
        "pet": {
            "operation_order": ["decode", "resize", "invert", "normalize"],
            "decode": "PIL.Image.convert('L')",
            "resize": "PIL.Image.Resampling.BILINEAR",
            "polarity": "invert white-canvas/dark-uptake as 255 - pixel",
            "normalization": "inverted_uint8 / 127.5 - 1.0",
            "output_dtype": "float32",
            "output_range": [-1.0, 1.0],
        },
        "mask": {
            "operation_order": ["decode", "resize", "threshold", "cast"],
            "decode": "PIL.Image.convert('L')",
            "resize": "PIL.Image.Resampling.NEAREST",
            "threshold_uint8": {"operator": ">", "value": 127},
            "threshold_normalized_equivalent": {"operator": ">", "value": 0.5},
            "output_dtype": "float32",
            "output_values": [0.0, 1.0],
        },
        "selection_policy": {
            "split_authority": "main_data/split_manifest.csv",
            "physical_train_val_directories_are_file_stores_only": True,
        },
    }
    preprocessing_bytes = json.dumps(
        preprocessing_config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    preprocessing_sha256 = _sha256_bytes(preprocessing_bytes)
    _write_json(
        output / "preprocessing_config.json",
        {**preprocessing_config, "semantic_sha256": preprocessing_sha256},
    )

    locked_contract_errors: list[str] = []
    if locked_contract is not None:
        locked_contract_errors = _locked_contract_mismatches(
            locked_contract,
            manifest_semantic_sha256=fingerprint["semantic_sha256"],
            raw_png_combined_sha256=raw_dataset_fingerprint["sha256"],
            raw_png_file_count=raw_dataset_fingerprint["file_count"],
            modality_counts={
                modality: len(indexes[modality]) for modality in MODALITY_DIRS
            },
            split_sample_counts=split_sample_counts,
            split_patient_counts=split_patient_counts,
            preprocessing_config_sha256=preprocessing_sha256,
        )
        local_gates.append(
            _gate(
                "G0A.7_locked_contract_content",
                "Audited manifest, PNG bytes, splits, and preprocessing exactly match the locked contract; physical root may differ.",
                not locked_contract_errors,
                {
                    "contract_path": _relative(locked_contract_path, root),
                    "declared_raw_root": locked_contract.get("raw_png", {}).get(
                        "root"
                    ),
                    "audited_raw_root": _relative(raw_root, root),
                    "mismatches": locked_contract_errors,
                },
                scope="local_raw_data",
                blocks_local_data_gate=True,
            )
        )

    gates = [*local_gates, *deferred_gates]
    failed_local_gates = [
        gate["gate_id"] for gate in local_gates if gate["status"] == "fail"
    ]
    data_gate_pass = not failed_local_gates

    contract_body = {
        "schema_version": SCHEMA_VERSION,
        "contract_name": "ct_pet_png_dataset_contract",
        "contract_status": "LOCKED" if data_gate_pass else "INVALID_LOCAL_DATA_GATE",
        "immutability": "content-addressed; any field change invalidates contract_sha256",
        "hash_canonicalization": "UTF-8 JSON of contract body; sorted keys; compact separators",
        "manifest": {
            "path": _relative(manifest_path, root),
            "semantic_sha256": fingerprint["semantic_sha256"],
            "source_file_sha256": fingerprint["source_file_sha256"],
        },
        "raw_png": {
            "root": _relative(raw_root, root),
            "file_count": raw_dataset_fingerprint["file_count"],
            "combined_sha256": raw_dataset_fingerprint["sha256"],
            "modalities": {modality: len(indexes[modality]) for modality in MODALITY_DIRS},
        },
        "splits": {
            split: {
                "samples": int(split_sample_counts.get(split, 0)),
                "patients": int(split_patient_counts.get(split, 0)),
            }
            for split in ("train", "val", "test")
        },
        "preprocessing": preprocessing_config,
        "preprocessing_config_sha256": preprocessing_sha256,
        "cloud_cache_metadata_required": {
            "manifest_semantic_sha256": fingerprint["semantic_sha256"],
            "raw_png_combined_sha256": raw_dataset_fingerprint["sha256"],
            "preprocessing_config_sha256": preprocessing_sha256,
            "dataset_contract_sha256": "must equal this file's contract_sha256",
        },
        "claim_boundary": {
            "dataset_hypothesis_allowed": data_gate_pass,
            "model_mechanism_claims_allowed": False,
        },
    }
    contract_bytes = json.dumps(
        contract_body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    generated_contract = {
        **contract_body,
        "contract_sha256": _sha256_bytes(contract_bytes),
    }
    dataset_contract = (
        dict(locked_contract)
        if locked_contract is not None and not locked_contract_errors
        else generated_contract
    )
    _write_json(output / "dataset_contract.json", dataset_contract)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "00A_local_raw_data_audit",
        "data_gate": "PASS" if data_gate_pass else "FAIL",
        "dataset_hypothesis_allowed": data_gate_pass,
        "cloud_training_gate": "DEFERRED",
        "checkpoint_lineage": "DEFERRED",
        "model_mechanism_claims_allowed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": root.as_posix(),
        "manifest": {
            "path": _relative(manifest_path, root),
            "fingerprint": fingerprint,
            "sample_counts": dict(split_sample_counts),
            "patient_counts": split_patient_counts,
            "patient_overlap": patient_overlap,
            "test_set_present": split_sample_counts.get("test", 0) > 0,
        },
        "raw_png": {
            "root": _relative(raw_root, root),
            "contract_declared_root": dataset_contract.get("raw_png", {}).get(
                "root"
            ),
            "fingerprint": raw_dataset_fingerprint,
            "input_size_counts": raw_input_sizes,
            "physical_split_mismatch_samples": len(physical_mismatches),
            "missing_by_modality": {key: len(value) for key, value in missing_raw.items()},
            "extra_by_modality": {key: len(value) for key, value in extra_raw.items()},
            "read_errors": image_errors,
        },
        "registration": {
            "method": "patient-level bootstrap of unflipped CT/PET normalized correlation minus flipped controls",
            "bootstrap_95ci": registration_bootstrap,
            "claim_boundary": "registration proxy only; not mutual information or causality",
        },
        "pet_polarity": {
            "conclusion": "invert as 1 - PNG/255 before mapping to [-1,1]" if pet_polarity_gate else "unresolved",
            "bootstrap_95ci": pet_bootstrap,
        },
        "mask": {
            "resize": "PIL grayscale nearest-neighbor to 192x192",
            "midpoint_rule": "pixel > 127 (equivalent to normalized pixel > 0.5)",
            "builder_default_rule": "pixel > 0 because mask_threshold defaults to 0.0",
            "median_area_positive": float(np.median(positive_areas)),
            "median_area_midpoint": float(np.median(midpoint_areas)),
            "different_area_samples": int(np.sum(positive_areas != midpoint_areas)),
            "bootstrap_95ci": mask_bootstrap,
        },
        "cache": {
            "configured_cache_dir": _relative(configured_cache_dir, root),
            "configured_cache_exists": configured_cache_dir.is_dir(),
            "manifest_declared_cache_dirs": [_relative(path, root) for path in manifest_cache_dirs],
            "legacy_cache_files": len(cache_ids),
            "overlap_with_manifest": len(cache_overlap_ids),
            "missing_manifest_ids": len(cache_missing_ids),
            "extra_cache_ids": len(cache_extra_ids),
            "errors": cache_errors,
            "comparison_bootstrap_95ci": cache_bootstrap,
        },
        "checkpoints": {
            "gate_status": "DEFERRED",
            "inventory_status": (
                "AUDITED" if audit_checkpoint_inventory else "SKIPPED_SEPARATE_GATE"
            ),
            "count": len(checkpoint_rows),
            "eligible_for_formal_conclusions": [row["checkpoint"] for row in eligible_checkpoints],
            "quarantined_count": len(checkpoint_rows) - len(eligible_checkpoints),
        },
        "dataset_contract": {
            "path": _relative(output / "dataset_contract.json", root),
            "contract_sha256": dataset_contract["contract_sha256"],
            "preprocessing_config_sha256": preprocessing_sha256,
        },
        "gates": gates,
    }
    _write_json(output / "audit_summary.json", summary)

    decision = {
        "schema_version": SCHEMA_VERSION,
        "stage": "00A_local_raw_data_audit",
        "data_gate": "PASS" if data_gate_pass else "FAIL",
        "dataset_hypothesis_allowed": data_gate_pass,
        "cloud_training_gate": "DEFERRED",
        "checkpoint_lineage": "DEFERRED",
        "model_mechanism_claims_allowed": False,
        "stage0b_local_dataset_hypothesis": "ALLOWED" if data_gate_pass else "BLOCKED",
        "stage0c_cloud_cache": "DEFERRED",
        "stage1_cloud_model": "BLOCKED_PENDING_STAGE_0C",
        "next_allowed_stage": "Stage_0B_local_dataset_hypothesis" if data_gate_pass else None,
        "manifest_semantic_sha256": fingerprint["semantic_sha256"],
        "raw_png_combined_sha256": raw_dataset_fingerprint["sha256"],
        "preprocessing_config_sha256": preprocessing_sha256,
        "dataset_contract_sha256": dataset_contract["contract_sha256"],
        "failed_local_gates": failed_local_gates,
        "deferred_gates": [gate["gate_id"] for gate in deferred_gates],
        "gates": gates,
        "legacy_checkpoint_policy": {
            "status": (
                "QUARANTINED"
                if audit_checkpoint_inventory
                else "DEFERRED_TO_CHECKPOINT_LINEAGE_GATE"
            ),
            "inventory_performed": audit_checkpoint_inventory,
            "eligible_for_formal_claims": [],
            "quarantined": [
                row["checkpoint"]
                for row in checkpoint_rows
                if not row["eligible_for_formal_mechanism_conclusion"]
            ],
        },
        "cloud_requirements_before_training": [
            "Build cache from the locked dataset contract without changing manifest sample or patient splits.",
            "Store manifest, raw-PNG, preprocessing, and dataset-contract fingerprints in cache metadata.",
            "Pass Stage 0C cache numerical-consistency checks before cloud model training.",
            "Embed the locked dataset-contract and cache fingerprints in every new checkpoint.",
        ],
    }
    _write_json(output / "decision.json", decision)

    report_lines = [
        "# Stage 0A local raw-data audit",
        "",
        f"Local data gate: **{decision['data_gate']}**. Stage 0B dataset-hypothesis work allowed = "
        f"`{str(data_gate_pass).lower()}`.",
        "",
        "Cloud cache/training and checkpoint-lineage gates are **DEFERRED**. Model-mechanism "
        "claims remain forbidden.",
        "",
        f"Manifest fingerprint (semantic SHA-256): `{fingerprint['semantic_sha256']}`.",
        f"Dataset-contract SHA-256: `{dataset_contract['contract_sha256']}`.",
        "",
        "## Findings",
        "",
        f"- Manifest: {len(manifest)} slices; train {split_patient_counts['train']} patients/"
        f"{split_sample_counts['train']} slices; val {split_patient_counts['val']} patients/"
        f"{split_sample_counts['val']} slices; test {split_patient_counts['test']} patients/"
        f"{split_sample_counts['test']} slices.",
        f"- Patient overlap across authoritative splits: {len(patient_overlap)}.",
        f"- Manifest/raw PNG bijection: CT missing {len(missing_raw['ct'])}, PET missing "
        f"{len(missing_raw['pet'])}, mask missing {len(missing_raw['mask'])}; physical-folder "
        f"split mismatches {len(physical_mismatches)} (folder labels are not used as splits).",
        f"- Registration proxy: unflipped-minus-LR train estimate "
        f"{train_registration_lr['estimate']:.6f}, 95% CI "
        f"[{train_registration_lr['ci95_low']:.6f}, {train_registration_lr['ci95_high']:.6f}]; "
        f"unflipped-minus-UD {train_registration_ud['estimate']:.6f}, 95% CI "
        f"[{train_registration_ud['ci95_low']:.6f}, {train_registration_ud['ci95_high']:.6f}].",
        f"- PET polarity: train patient-bootstrap inverted lesion-minus-ring estimate "
        f"{train_pet_inverted['estimate']:.6f}, 95% CI "
        f"[{train_pet_inverted['ci95_low']:.6f}, {train_pet_inverted['ci95_high']:.6f}].",
        f"- Mask at 192x192: median area `>0` = {float(np.median(positive_areas)):.1f} px; "
        f"median area `>127` = {float(np.median(midpoint_areas)):.1f} px; "
        f"{int(np.sum(positive_areas != midpoint_areas))}/{len(image_metric_rows)} samples differ.",
        f"- Configured cache `{_relative(configured_cache_dir, root)}` exists: "
        f"{configured_cache_dir.is_dir()}. Legacy manifest-declared cache overlap/missing/extra: "
        f"{len(cache_overlap_ids)}/{len(cache_missing_ids)}/{len(cache_extra_ids)}.",
        f"- Legacy checkpoints audited/quarantined: {len(checkpoint_rows)}/{len(checkpoint_rows)}; "
        "they do not block Stage 0B and remain ineligible for formal claims.",
        "",
        "## Gates",
        "",
    ]
    report_lines.extend(
        f"- `{gate['gate_id']}` [{gate['scope']}]: **{gate['status'].upper()}** -- {gate['description']}"
        for gate in gates
    )
    report_lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "Stage 0B may test data-only hypotheses, including H1. Stage 0C must pass before "
            "cloud training. Conditional-mean, residual, router, curriculum, diffusion optimization, "
            "ablation, and final performance claims remain out of scope.",
        ]
    )
    (output / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    execution = {
        "script": _relative(Path(__file__), root),
        "script_sha256": _sha256_file(Path(__file__)),
        "python": sys.version,
        "command": " ".join(sys.argv),
        "checkpoint_inventory_performed": audit_checkpoint_inventory,
        "audited_raw_root": _relative(raw_root, root),
        "locked_contract_path": (
            None
            if locked_contract_path is None
            else _relative(locked_contract_path, root)
        ),
        "audit_seed": AUDIT_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "image_size": IMAGE_SIZE,
    }
    _write_json(output / "execution_metadata.json", execution)
    return decision


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mechanism_validation/00_data_audit"),
    )
    parser.add_argument(
        "--skip-checkpoint-inventory",
        action="store_true",
        help=(
            "Do not inspect legacy checkpoints. Checkpoint lineage is a separate "
            "cloud-training gate and never blocks the local data gate."
        ),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help=(
            "Physical CT/PET/label PNG root. Defaults to Data/data for the "
            "original local layout; cloud runs should pass main_data."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Authoritative manifest override. This is required when "
            "auditing a new/external H4-v2 confirmation cohort."
        ),
    )
    parser.add_argument(
        "--locked-contract",
        type=Path,
        default=None,
        help=(
            "Optional immutable contract. When supplied, audited content must "
            "match it exactly even if the physical raw-root path differs."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    raw_root = (
        None
        if args.raw_root is None
        else (
            args.raw_root.resolve()
            if args.raw_root.is_absolute()
            else (root / args.raw_root).resolve()
        )
    )
    locked_contract_path = (
        None
        if args.locked_contract is None
        else (
            args.locked_contract.resolve()
            if args.locked_contract.is_absolute()
            else (root / args.locked_contract).resolve()
        )
    )
    manifest_path = (
        None
        if args.manifest is None
        else (
            args.manifest.resolve()
            if args.manifest.is_absolute()
            else (root / args.manifest).resolve()
        )
    )
    decision = run(
        root,
        output,
        audit_checkpoint_inventory=not args.skip_checkpoint_inventory,
        raw_root_override=raw_root,
        manifest_path_override=manifest_path,
        locked_contract_path=locked_contract_path,
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False, default=_json_default))
    return 0 if decision["data_gate"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
