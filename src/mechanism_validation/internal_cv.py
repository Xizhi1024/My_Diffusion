"""Frozen patient-level partitions for internal exploratory H4-v2 analysis.

Only patient attributes that are available before an H4-v2 outer-fold
evaluation are used to construct the partitions.  Recoverability targets and
comparator errors are deliberately absent from this module.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.mechanism_validation.common import canonical_json_sha256
from src.mechanism_validation.h4_v2 import normalize_ids


BALANCE_CONTINUOUS_FIELDS = (
    "slice_count",
    "mean_mask_area",
    "mean_pet_ll2_log_contrast",
)
BALANCE_CATEGORICAL_FIELDS = (
    "manifest_split",
    "original_partition",
)


def seal_mapping(
    payload: Mapping[str, Any],
    *,
    hash_field: str,
) -> dict[str, Any]:
    body = dict(payload)
    body.pop(hash_field, None)
    return {**body, hash_field: canonical_json_sha256(body)}


def validate_sealed_mapping(
    payload: Mapping[str, Any],
    *,
    hash_field: str,
) -> str:
    body = dict(payload)
    claimed = str(body.pop(hash_field, ""))
    computed = canonical_json_sha256(body)
    if not claimed or claimed != computed:
        raise ValueError(
            f"{hash_field} self-hash mismatch: {claimed!r} != {computed!r}"
        )
    return claimed


def patient_balance_attributes(h1_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the three pre-registered balance variables by patient."""

    frame = normalize_ids(h1_rows)
    required = {
        "patient_id",
        "sample_id",
        "manifest_split",
        "partition",
        "mask_area",
        "pet_ll2_lesion_ring_log_ratio",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"H1 rows lack patient-balance fields: {sorted(missing)}"
        )
    uniqueness = frame.groupby("patient_id", sort=True).agg(
        manifest_splits=("manifest_split", "nunique"),
        original_partitions=("partition", "nunique"),
    )
    if (uniqueness["manifest_splits"] != 1).any():
        raise ValueError("A patient spans multiple manifest splits")
    if (uniqueness["original_partitions"] != 1).any():
        raise ValueError("A patient spans multiple mechanism partitions")
    result = (
        frame.groupby("patient_id", as_index=False, sort=True)
        .agg(
            manifest_split=("manifest_split", "first"),
            original_partition=("partition", "first"),
            slice_count=("sample_id", "nunique"),
            mean_mask_area=("mask_area", "mean"),
            mean_pet_ll2_log_contrast=(
                "pet_ll2_lesion_ring_log_ratio",
                "mean",
            ),
        )
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    for field in BALANCE_CONTINUOUS_FIELDS:
        values = result[field].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite patient balance field: {field}")
    return result


def _standardize(column: np.ndarray) -> np.ndarray:
    values = np.asarray(column, dtype=np.float64)
    scale = float(values.std())
    if scale <= 1e-12:
        return np.zeros_like(values)
    return (values - float(values.mean())) / scale


def balance_feature_matrix(
    attributes: pd.DataFrame,
    *,
    quantile_bins: int,
) -> tuple[np.ndarray, list[str]]:
    """Build continuous-rank, quantile-bin, and categorical balance features."""

    if quantile_bins < 2:
        raise ValueError("quantile_bins must be at least 2")
    columns: list[np.ndarray] = []
    names: list[str] = []
    for field in BALANCE_CONTINUOUS_FIELDS:
        ranks = (
            attributes[field]
            .rank(method="average", pct=True)
            .to_numpy(dtype=np.float64)
        )
        columns.append(_standardize(ranks))
        names.append(f"rank:{field}")
        stable_ranks = attributes[field].rank(method="first").to_numpy()
        bins = pd.qcut(
            stable_ranks,
            q=min(quantile_bins, len(attributes)),
            labels=False,
            duplicates="drop",
        )
        for bin_index in sorted(np.unique(bins)):
            indicator = np.asarray(bins == bin_index, dtype=np.float64)
            columns.append(_standardize(indicator))
            names.append(f"quantile:{field}:{int(bin_index)}")
    for field in BALANCE_CATEGORICAL_FIELDS:
        values = attributes[field].astype(str)
        for category in sorted(values.unique()):
            indicator = np.asarray(values == category, dtype=np.float64)
            columns.append(_standardize(indicator))
            names.append(f"category:{field}:{category}")
    if not columns:
        raise ValueError("No balance features were constructed")
    return np.column_stack(columns), names


def balanced_partition_search(
    attributes: pd.DataFrame,
    *,
    folds: int,
    seed: int,
    candidates: int,
    quantile_bins: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Choose the best fixed-size patient partition from seeded candidates."""

    frame = attributes.sort_values("patient_id").reset_index(drop=True)
    if folds < 2 or len(frame) < folds:
        raise ValueError("Invalid patient fold count")
    if candidates < 1:
        raise ValueError("candidates must be positive")
    matrix, feature_names = balance_feature_matrix(
        frame,
        quantile_bins=quantile_bins,
    )
    labels = np.arange(len(frame), dtype=np.int64) % folds
    rng = np.random.default_rng(seed)
    best_assignment: np.ndarray | None = None
    best_score = float("inf")
    for _ in range(candidates):
        permutation = rng.permutation(len(frame))
        assignment = np.empty(len(frame), dtype=np.int64)
        assignment[permutation] = labels
        fold_means = np.vstack(
            [matrix[assignment == fold].mean(axis=0) for fold in range(folds)]
        )
        score = float(np.mean(np.square(fold_means)))
        if score < best_score:
            best_score = score
            best_assignment = assignment.copy()
    if best_assignment is None:
        raise RuntimeError("Patient partition search produced no candidate")
    result = frame[["patient_id"]].copy()
    result["fold"] = best_assignment
    fold_feature_means = np.vstack(
        [
            matrix[best_assignment == fold].mean(axis=0)
            for fold in range(folds)
        ]
    )
    diagnostics = {
        "seed": int(seed),
        "candidates": int(candidates),
        "folds": int(folds),
        "quantile_bins": int(quantile_bins),
        "objective": best_score,
        "maximum_absolute_standardized_feature_mean": float(
            np.max(np.abs(fold_feature_means))
        ),
        "feature_names": feature_names,
        "fold_patient_counts": {
            str(fold): int(np.count_nonzero(best_assignment == fold))
            for fold in range(folds)
        },
    }
    return result, diagnostics


def fold_balance_table(
    attributes: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    fold_column: str = "fold",
) -> pd.DataFrame:
    merged = attributes.merge(
        assignments[["patient_id", fold_column]],
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )
    records: list[dict[str, Any]] = []
    for fold, group in merged.groupby(fold_column, sort=True):
        record: dict[str, Any] = {
            fold_column: int(fold),
            "patients": int(len(group)),
        }
        for field in BALANCE_CONTINUOUS_FIELDS:
            record[f"mean_{field}"] = float(group[field].mean())
            record[f"median_{field}"] = float(group[field].median())
        for field in BALANCE_CATEGORICAL_FIELDS:
            for category, count in group[field].value_counts().items():
                record[f"{field}_{category}_patients"] = int(count)
        records.append(record)
    return pd.DataFrame(records).sort_values(fold_column).reset_index(drop=True)


def partition_fingerprint(
    outer_assignments: pd.DataFrame,
    nested_roles: pd.DataFrame,
) -> str:
    outer = [
        {
            "patient_id": str(row.patient_id),
            "outer_fold": int(row.outer_fold),
        }
        for row in outer_assignments.sort_values(
            ["patient_id"]
        ).itertuples(index=False)
    ]
    nested = [
        {
            "outer_fold": int(row.outer_fold),
            "patient_id": str(row.patient_id),
            "role": str(row.role),
            "inner_fold": (
                None if pd.isna(row.inner_fold) else int(row.inner_fold)
            ),
        }
        for row in nested_roles.sort_values(
            ["outer_fold", "patient_id"]
        ).itertuples(index=False)
    ]
    return canonical_json_sha256({"outer": outer, "nested": nested})


def build_nested_roles(
    attributes: pd.DataFrame,
    outer_assignments: pd.DataFrame,
    *,
    outer_folds: int,
    inner_folds: int,
    inner_seed: int,
    inner_candidates: int,
    quantile_bins: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Freeze inner train/calibration roles independently in every outer fold."""

    assignment_map = dict(
        zip(
            outer_assignments["patient_id"].astype(str),
            outer_assignments["outer_fold"].astype(int),
            strict=True,
        )
    )
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for outer_fold in range(outer_folds):
        held = {
            patient
            for patient, fold in assignment_map.items()
            if fold == outer_fold
        }
        remaining = attributes[
            ~attributes["patient_id"].astype(str).isin(held)
        ].copy()
        inner, current_diagnostics = balanced_partition_search(
            remaining,
            folds=inner_folds,
            seed=inner_seed + outer_fold,
            candidates=inner_candidates,
            quantile_bins=quantile_bins,
        )
        calibration_fold = outer_fold % inner_folds
        inner_map = dict(
            zip(
                inner["patient_id"].astype(str),
                inner["fold"].astype(int),
                strict=True,
            )
        )
        for patient in attributes["patient_id"].astype(str):
            if patient in held:
                role = "outer_evaluation"
                current_inner_fold: int | None = None
            else:
                current_inner_fold = int(inner_map[patient])
                role = (
                    "calibration"
                    if current_inner_fold == calibration_fold
                    else "mechanism_train"
                )
            rows.append(
                {
                    "outer_fold": outer_fold,
                    "patient_id": patient,
                    "role": role,
                    "inner_fold": current_inner_fold,
                }
            )
        current_diagnostics.update(
            {
                "outer_fold": outer_fold,
                "calibration_inner_fold": calibration_fold,
                "outer_evaluation_patients": len(held),
                "mechanism_train_patients": sum(
                    row["outer_fold"] == outer_fold
                    and row["role"] == "mechanism_train"
                    for row in rows
                ),
                "calibration_patients": sum(
                    row["outer_fold"] == outer_fold
                    and row["role"] == "calibration"
                    for row in rows
                ),
            }
        )
        diagnostics.append(current_diagnostics)
    result = pd.DataFrame(rows).sort_values(
        ["outer_fold", "patient_id"]
    ).reset_index(drop=True)
    expected = len(attributes) * outer_folds
    if len(result) != expected:
        raise ValueError("Nested role table does not cover every fold/patient")
    if result.duplicated(["outer_fold", "patient_id"]).any():
        raise ValueError("Nested role table contains duplicate patients")
    return result, diagnostics


def assert_patient_partition_integrity(
    outer_assignments: pd.DataFrame,
    nested_roles: pd.DataFrame,
    *,
    patient_ids: Sequence[str],
    outer_folds: int,
) -> None:
    expected = {str(value) for value in patient_ids}
    observed = set(outer_assignments["patient_id"].astype(str))
    if observed != expected:
        raise ValueError("Outer assignments do not cover the patient cohort")
    if outer_assignments["patient_id"].duplicated().any():
        raise ValueError("A patient occurs in multiple outer assignments")
    for outer_fold in range(outer_folds):
        current = nested_roles[nested_roles["outer_fold"] == outer_fold]
        if set(current["patient_id"].astype(str)) != expected:
            raise ValueError(f"Nested roles incomplete for fold {outer_fold}")
        role_sets = {
            role: set(
                current.loc[current["role"] == role, "patient_id"].astype(str)
            )
            for role in (
                "mechanism_train",
                "calibration",
                "outer_evaluation",
            )
        }
        if any(
            role_sets[left] & role_sets[right]
            for left, right in (
                ("mechanism_train", "calibration"),
                ("mechanism_train", "outer_evaluation"),
                ("calibration", "outer_evaluation"),
            )
        ):
            raise ValueError(f"Patient role leakage in fold {outer_fold}")
        expected_held = set(
            outer_assignments.loc[
                outer_assignments["outer_fold"] == outer_fold,
                "patient_id",
            ].astype(str)
        )
        if role_sets["outer_evaluation"] != expected_held:
            raise ValueError(f"Outer held set mismatch in fold {outer_fold}")
