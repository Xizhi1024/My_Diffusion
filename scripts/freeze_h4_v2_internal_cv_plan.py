"""Freeze patient-level nested-CV partitions before H4-v2 outer evaluation."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mechanism_validation.common import file_sha256, load_json, write_json
from src.mechanism_validation.h4_v2 import BANDS, normalize_ids
from src.mechanism_validation.internal_cv import (
    assert_patient_partition_integrity,
    balanced_partition_search,
    build_nested_roles,
    fold_balance_table,
    partition_fingerprint,
    patient_balance_attributes,
    seal_mapping,
    validate_sealed_mapping,
)


SCHEMA_VERSION = 2
STAGE = "00_H4_v2_internal_exploratory_plan_freeze"


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return normalize_ids(
        pd.read_csv(
            path,
            dtype={"patient_id": str, "sample_id": str},
            **kwargs,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze balanced patient-level outer and inner folds without "
            "reading H4-v2 recoverability outcomes."
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--config",
        default="configs/h4_v2_internal_exploratory_nested_cv_v1.json",
    )
    parser.add_argument(
        "--h4-v1-decision",
        default=(
            "results/mechanism_validation/"
            "03_h4_noise_calibration/decision.json"
        ),
    )
    parser.add_argument(
        "--h4-v1-samples",
        default=(
            "results/mechanism_validation/"
            "03_h4_noise_calibration/sample_band_evidence.csv"
        ),
    )
    parser.add_argument(
        "--h1-samples",
        default=(
            "results/mechanism_validation/"
            "00B_h1_spectral_asymmetry/sample_band_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "results/mechanism_validation_v2/"
            "02_h4_v2_internal_exploratory_nested_cv/00_frozen_plan"
        ),
    )
    return parser


def _validate_existing_plan(
    plan_path: Path,
    *,
    config_sha256: str,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    plan = load_json(plan_path)
    validate_sealed_mapping(plan, hash_field="plan_sha256")
    if plan.get("config_sha256") != config_sha256:
        raise ValueError("Frozen plan config hash differs from current config")
    if plan.get("source_artifacts") != source_hashes:
        raise ValueError("Frozen plan source hashes differ from current inputs")
    if plan.get("status") != "FROZEN_BEFORE_OUTER_EVALUATION":
        raise ValueError("Existing plan is not a valid frozen plan")
    return plan


def _run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    output = _resolve(root, args.output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = _resolve(root, args.config)
    h4_decision_path = _resolve(root, args.h4_v1_decision)
    h4_samples_path = _resolve(root, args.h4_v1_samples)
    h1_samples_path = _resolve(root, args.h1_samples)

    config = load_json(config_path)
    config_sha256 = validate_sealed_mapping(
        config,
        hash_field="config_sha256",
    )
    source_hashes = {
        "config_file_sha256": file_sha256(config_path),
        "h4_v1_decision_sha256": file_sha256(h4_decision_path),
        "h4_v1_sample_evidence_sha256": file_sha256(h4_samples_path),
        "h1_sample_metrics_sha256": file_sha256(h1_samples_path),
    }
    plan_path = output / "frozen_plan.json"
    if plan_path.exists():
        plan = _validate_existing_plan(
            plan_path,
            config_sha256=config_sha256,
            source_hashes=source_hashes,
        )
        decision = load_json(output / "decision.json")
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        return 0

    h4_v1 = load_json(h4_decision_path)
    if h4_v1.get("decision") != "FAIL":
        raise ValueError("The immutable H4-v1 outcome must remain FAIL")
    h1_rows = _read_csv(h1_samples_path)
    h4_identity = _read_csv(
        h4_samples_path,
        usecols=["sample_id", "patient_id", "band"],
    )
    if set(h4_identity["band"].astype(str)) != set(BANDS):
        raise ValueError("H4-v1 band set differs from the frozen v2 band set")
    if set(h1_rows["sample_id"]) != set(h4_identity["sample_id"]):
        raise ValueError("H1 and H4-v1 sample identities differ")
    if set(h1_rows["patient_id"]) != set(h4_identity["patient_id"]):
        raise ValueError("H1 and H4-v1 patient identities differ")

    constraints = config["dataset_constraints"]
    patients = int(h1_rows["patient_id"].nunique())
    samples = int(h1_rows["sample_id"].nunique())
    if patients != int(constraints["patients"]):
        raise ValueError(
            f"Expected {constraints['patients']} patients, observed {patients}"
        )
    if samples != int(constraints["samples"]):
        raise ValueError(
            f"Expected {constraints['samples']} samples, observed {samples}"
        )
    attributes = patient_balance_attributes(h1_rows)
    observed_partitions = {
        str(name): int(count)
        for name, count in attributes["original_partition"]
        .value_counts()
        .items()
    }
    expected_partitions = {
        str(name): int(count)
        for name, count in constraints["original_partitions"].items()
    }
    if observed_partitions != expected_partitions:
        raise ValueError(
            "Original patient partition counts differ: "
            f"{observed_partitions} != {expected_partitions}"
        )
    difficult = {
        str(value) for value in constraints["required_difficult_patients"]
    }
    observed_patients = set(attributes["patient_id"].astype(str))
    missing_difficult = difficult - observed_patients
    if missing_difficult:
        raise ValueError(
            f"Required difficult patients are missing: {missing_difficult}"
        )

    partition_spec = config["partition"]
    outer_folds = int(partition_spec["outer_folds"])
    outer_assignments, outer_diagnostics = balanced_partition_search(
        attributes,
        folds=outer_folds,
        seed=int(partition_spec["outer_seed"]),
        candidates=int(partition_spec["outer_balance_candidates"]),
        quantile_bins=int(partition_spec["quantile_bins"]),
    )
    outer_assignments = outer_assignments.rename(
        columns={"fold": "outer_fold"}
    )
    nested_roles, inner_diagnostics = build_nested_roles(
        attributes,
        outer_assignments,
        outer_folds=outer_folds,
        inner_folds=int(partition_spec["inner_folds"]),
        inner_seed=int(partition_spec["inner_seed"]),
        inner_candidates=int(partition_spec["inner_balance_candidates"]),
        quantile_bins=int(partition_spec["quantile_bins"]),
    )
    assert_patient_partition_integrity(
        outer_assignments,
        nested_roles,
        patient_ids=attributes["patient_id"],
        outer_folds=outer_folds,
    )
    partition_sha256 = partition_fingerprint(
        outer_assignments,
        nested_roles,
    )
    balance = fold_balance_table(
        attributes,
        outer_assignments,
        fold_column="outer_fold",
    )

    assignment_rows = attributes.merge(
        outer_assignments,
        on="patient_id",
        how="inner",
        validate="one_to_one",
    ).sort_values(["outer_fold", "patient_id"])
    assignment_path = output / "outer_patient_assignments.csv"
    nested_path = output / "nested_patient_roles.csv"
    balance_path = output / "outer_fold_balance.csv"
    assignment_rows.to_csv(assignment_path, index=False)
    nested_roles.to_csv(nested_path, index=False)
    balance.to_csv(balance_path, index=False)

    plan_body = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "pipeline_id": config["pipeline_id"],
        "status": "FROZEN_BEFORE_OUTER_EVALUATION",
        "scope": config["scope"],
        "config_sha256": config_sha256,
        "source_artifacts": source_hashes,
        "dataset": {
            "patients": patients,
            "samples": samples,
            "original_partitions": observed_partitions,
            "required_difficult_patients_present": sorted(difficult),
            "patients_excluded": 0,
        },
        "partition": {
            "fingerprint_sha256": partition_sha256,
            "outer_folds": outer_folds,
            "inner_folds": int(partition_spec["inner_folds"]),
            "unit": "patient",
            "balance_uses_h4_v2_outcomes": False,
            "outer_diagnostics": outer_diagnostics,
            "inner_diagnostics": inner_diagnostics,
        },
        "frozen_files": {
            "outer_patient_assignments": {
                "path": assignment_path.relative_to(root).as_posix(),
                "sha256": file_sha256(assignment_path),
            },
            "nested_patient_roles": {
                "path": nested_path.relative_to(root).as_posix(),
                "sha256": file_sha256(nested_path),
            },
            "outer_fold_balance": {
                "path": balance_path.relative_to(root).as_posix(),
                "sha256": file_sha256(balance_path),
            },
        },
        "freeze_guardrails": {
            "frozen_before_outer_evaluation": True,
            "slice_level_random_split": False,
            "outer_patient_used_for_corresponding_fit_or_threshold": False,
            "recoverability_or_comparator_error_used_for_balance": False,
            "H4_v1_outcome": "FAIL",
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    plan = seal_mapping(plan_body, hash_field="plan_sha256")
    write_json(plan_path, plan)
    decision = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "pipeline_id": config["pipeline_id"],
        "decision": "FROZEN",
        "plan_sha256": plan["plan_sha256"],
        "partition_fingerprint": partition_sha256,
        "patients": patients,
        "samples": samples,
        "patients_excluded": 0,
        "H4_v1_preserved": "FAIL",
        "outer_evaluation_allowed": True,
    }
    write_json(output / "decision.json", decision)
    write_json(
        output / "execution_metadata.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "command": " ".join(sys.argv),
            "script": Path(__file__).resolve().as_posix(),
            "script_sha256": file_sha256(Path(__file__).resolve()),
        },
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _run(argv)
    except Exception as exc:
        args = build_parser().parse_args(argv)
        root = args.root.resolve()
        output = _resolve(root, args.output)
        output.mkdir(parents=True, exist_ok=True)
        decision = {
            "schema_version": SCHEMA_VERSION,
            "stage": STAGE,
            "decision": "FAIL",
            "failure_phase": "PLAN_FREEZE_EXCEPTION_FAIL_CLOSED",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "outer_evaluation_allowed": False,
        }
        write_json(output / "decision.json", decision)
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
