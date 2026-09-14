"""Shared, fail-closed helpers for formal patient-level mechanism gates."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PARTITION_SEED = 42
CALIBRATION_FRACTION = 0.20
BOOTSTRAP_REPLICATES = 10_000
MANIFEST_COLUMNS = ("sample_id", "patient_id", "slice_id", "split", "cache_path")


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if math.isfinite(value) else None
    raise TypeError(f"Cannot JSON-encode {type(value)!r}")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row}) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        missing = [column for column in MANIFEST_COLUMNS if column not in columns]
        if missing:
            raise ValueError(f"Manifest is missing columns {missing}: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Manifest has duplicate sample_id values")
    patient_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        patient_splits[row["patient_id"]].add(row["split"])
    overlap = {
        patient: sorted(splits)
        for patient, splits in patient_splits.items()
        if len(splits) > 1
    }
    if overlap:
        raise ValueError(f"Manifest patient leakage: {overlap}")
    return rows


def patient_partition(
    rows: Sequence[Mapping[str, str]],
    *,
    seed: int = PARTITION_SEED,
    calibration_fraction: float = CALIBRATION_FRACTION,
) -> dict[str, str]:
    """Reproduce the locked H1 99/25/31 patient partition exactly."""

    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    train_patients = sorted(
        {row["patient_id"] for row in rows if row["split"] == "train"}
    )
    validation_patients = sorted(
        {row["patient_id"] for row in rows if row["split"] == "val"}
    )
    if not train_patients or not validation_patients:
        raise ValueError("Formal mechanism partition requires train and val patients")
    shuffled = np.asarray(train_patients, dtype=object)
    np.random.default_rng(seed).shuffle(shuffled)
    calibration_count = max(1, round(calibration_fraction * len(shuffled)))
    calibration = set(shuffled[:calibration_count].tolist())
    result = {
        patient: (
            "calibration" if patient in calibration else "mechanism_train"
        )
        for patient in train_patients
    }
    result.update({patient: "validation" for patient in validation_patients})
    return result


def partition_sha256(partition: Mapping[str, str]) -> str:
    return canonical_json_sha256(sorted(partition.items()))


def partition_counts(
    rows: Sequence[Mapping[str, str]],
    partition: Mapping[str, str],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for role in ("mechanism_train", "calibration", "validation"):
        patients = {patient for patient, value in partition.items() if value == role}
        result[role] = {
            "patients": len(patients),
            "samples": sum(row["patient_id"] in patients for row in rows),
        }
    return result


def write_mechanism_manifest(
    source_rows: Sequence[Mapping[str, str]],
    partition: Mapping[str, str],
    output: Path,
) -> dict[str, Any]:
    """Write train=mechanism_train, val=calibration, test=held-out validation."""

    split_for_role = {
        "mechanism_train": "train",
        "calibration": "val",
        "validation": "test",
    }
    derived: list[dict[str, str]] = []
    for row in source_rows:
        role = partition.get(row["patient_id"])
        if role not in split_for_role:
            continue
        derived.append(
            {
                **{column: str(row.get(column, "")) for column in MANIFEST_COLUMNS},
                "split": split_for_role[role],
            }
        )
    write_csv(output, derived)
    return {
        "path": output.as_posix(),
        "sha256": file_sha256(output),
        "partition_sha256": partition_sha256(partition),
        "counts": partition_counts(source_rows, partition),
        "mapping": split_for_role,
    }


def aggregate_patient_rows(
    rows: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["partition"]), str(row["patient_id"]))].append(row)
    result: list[dict[str, Any]] = []
    for (partition, patient_id), current in sorted(grouped.items()):
        record: dict[str, Any] = {
            "partition": partition,
            "patient_id": patient_id,
            "samples": len(current),
        }
        for metric in metric_names:
            values = np.asarray(
                [float(row[metric]) for row in current if metric in row],
                dtype=np.float64,
            )
            values = values[np.isfinite(values)]
            record[metric] = (
                float(values.mean()) if values.size else float("nan")
            )
        result.append(record)
    return result


def bootstrap_mean(
    values: Iterable[float],
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "patients": 0,
            "estimate": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "bootstrap_replicates": replicates,
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(replicates, array.size))
    draws = array[indices].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "patients": int(array.size),
        "estimate": float(array.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_replicates": replicates,
    }


def paired_patient_bootstrap(
    patient_rows: Sequence[Mapping[str, Any]],
    *,
    partition: str,
    left: str,
    right: str,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    current = [row for row in patient_rows if row["partition"] == partition]
    differences = np.asarray(
        [float(row[left]) - float(row[right]) for row in current],
        dtype=np.float64,
    )
    result = bootstrap_mean(differences, seed=seed, replicates=replicates)
    result.update(
        {
            "partition": partition,
            "contrast": f"{left} - {right}",
            "sign_flip_p": sign_flip_p(
                differences,
                seed=seed + 1,
                replicates=replicates,
            ),
        }
    )
    return result


def sign_flip_p(
    effects: Iterable[float],
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
    two_sided: bool = False,
) -> float:
    array = np.asarray(list(effects), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    observed = float(array.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(replicates, array.size))
    null = (signs * array[None]).mean(axis=1)
    if two_sided:
        exceed = np.abs(null) >= abs(observed)
    else:
        exceed = null >= observed
    return float((np.count_nonzero(exceed) + 1) / (replicates + 1))


def decision_guardrails(
    *,
    checkpoint_lineage: str = "PASS",
) -> dict[str, Any]:
    return {
        "patient_unit": "patient",
        "threshold_source": "mechanism_train_and_calibration_only",
        "validation_or_test_drives_training_or_course": False,
        "patient_bootstrap_95ci": True,
        "causal_or_mutual_information_claimed": False,
        "checkpoint_lineage": checkpoint_lineage,
    }


def require_upstream_pass(path: Path, expected_stage: str | None = None) -> dict[str, Any]:
    decision = load_json(path)
    observed = decision.get("decision", decision.get("data_gate"))
    if observed != "PASS":
        raise RuntimeError(f"Upstream gate is not PASS: {path} ({observed!r})")
    if expected_stage is not None and decision.get("stage") != expected_stage:
        raise RuntimeError(
            f"Unexpected upstream stage in {path}: {decision.get('stage')!r}"
        )
    return decision
