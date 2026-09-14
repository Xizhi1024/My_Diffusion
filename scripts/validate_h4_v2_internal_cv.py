"""Evaluate frozen H4-v2 partitions on all 155 patients exactly out of fold."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mechanism_validation.common import (
    bootstrap_mean,
    file_sha256,
    load_json,
    sign_flip_p,
    write_json,
)
from src.mechanism_validation.h4_v2 import (
    BANDS,
    add_context_features,
    apply_comparators,
    apply_hierarchical_calibration,
    assign_subgroups,
    fit_comparators,
    fit_original_standardization,
    fit_patient_attributes,
    fit_population_stats,
    freeze_subgroup_spec,
    normalize_ids,
    patient_errors,
)
from src.mechanism_validation.internal_cv import (
    assert_patient_partition_integrity,
    partition_fingerprint,
    validate_sealed_mapping,
)


SCHEMA_VERSION = 2
STAGE = "01_H4_v2_internal_exploratory_outer_evaluation"


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_csv(path: Path) -> pd.DataFrame:
    return normalize_ids(
        pd.read_csv(
            path,
            dtype={"patient_id": str, "sample_id": str},
        )
    )


def _read_patient_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"patient_id": str})
    if "patient_id" not in frame:
        raise ValueError(f"Patient table lacks patient_id: {path}")
    frame["patient_id"] = frame["patient_id"].map(
        lambda value: (
            str(value).strip().removesuffix(".0").zfill(3)
            if str(value).strip().removesuffix(".0").isdigit()
            else str(value).strip()
        )
    )
    return frame


def _effect_summary(
    values: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    result = bootstrap_mean(values, seed=seed, replicates=replicates)
    result["sign_flip_p"] = sign_flip_p(
        values,
        seed=seed + 1,
        replicates=replicates,
    )
    return result


def _improvement_summary(
    effects: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    improved = np.asarray(effects > 0.0, dtype=np.float64)
    result = bootstrap_mean(
        improved,
        seed=seed,
        replicates=replicates,
    )
    result["improved_patients"] = int(np.count_nonzero(improved))
    result["not_improved_patients"] = int(len(improved) - improved.sum())
    result["definition"] = "strictly lower patient MSE for the left method"
    return result


def _comparison(
    errors: pd.DataFrame,
    *,
    left: str,
    right: str,
    seed: int,
    replicates: int,
    minimum_positive_folds: int,
) -> dict[str, Any]:
    effect = (
        errors[f"mse_{right}"].to_numpy(dtype=np.float64)
        - errors[f"mse_{left}"].to_numpy(dtype=np.float64)
    )
    result = _effect_summary(
        effect,
        seed=seed,
        replicates=replicates,
    )
    result["effect"] = (
        f"MSE({right}) - MSE({left}); positive favors {left}"
    )
    result["improved_patient_fraction"] = _improvement_summary(
        effect,
        seed=seed + 10,
        replicates=replicates,
    )
    fold_results: list[dict[str, Any]] = []
    for fold, group in errors.groupby("outer_fold", sort=True):
        fold_effect = (
            group[f"mse_{right}"].to_numpy(dtype=np.float64)
            - group[f"mse_{left}"].to_numpy(dtype=np.float64)
        )
        fold_results.append(
            {
                "outer_fold": int(fold),
                **_effect_summary(
                    fold_effect,
                    seed=seed + 100 + int(fold) * 10,
                    replicates=replicates,
                ),
                "improved_patient_fraction": _improvement_summary(
                    fold_effect,
                    seed=seed + 105 + int(fold) * 10,
                    replicates=replicates,
                ),
            }
        )
    result["outer_fold_results"] = fold_results
    result["positive_outer_folds"] = int(
        sum(float(row["estimate"]) > 0.0 for row in fold_results)
    )
    result["minimum_positive_outer_folds"] = int(minimum_positive_folds)
    return result


def _all_comparisons(
    errors: pd.DataFrame,
    gates: Mapping[str, Any],
    *,
    seed_offset: int = 0,
) -> dict[str, Any]:
    seed = int(gates["analysis_seed"]) + seed_offset
    replicates = int(gates["bootstrap_replicates"])
    positive_folds = int(gates["minimum_positive_outer_folds"])
    specs = (
        ("h3_vs_no_route", "h3_fixed_schedule", "no_route"),
        ("original_vs_h3", "original_evidence", "h3_fixed_schedule"),
        ("uncertainty_vs_h3", "uncertainty_aware", "h3_fixed_schedule"),
        (
            "uncertainty_vs_original",
            "uncertainty_aware",
            "original_evidence",
        ),
    )
    return {
        name: _comparison(
            errors,
            left=left,
            right=right,
            seed=seed + index * 1000,
            replicates=replicates,
            minimum_positive_folds=positive_folds,
        )
        for index, (name, left, right) in enumerate(specs)
    }


def _subgroup_statistics(
    report: pd.DataFrame,
    gates: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    results: dict[str, Any] = {}
    all_passed = True
    replicates = int(gates["bootstrap_replicates"])
    base_seed = int(gates["analysis_seed"]) + 20_000
    minimum = int(gates["subgroup_minimum_patients"])
    for index, name in enumerate(gates["robustness_subgroups"]):
        selected = report[report[str(name)].fillna(False)].copy()
        raw_effect = (
            selected["mse_h3_fixed_schedule"].to_numpy(dtype=np.float64)
            - selected["mse_uncertainty_aware"].to_numpy(dtype=np.float64)
        )
        margins = selected["fold_noninferiority_margin"].to_numpy(
            dtype=np.float64
        )
        adjusted = raw_effect + margins
        raw_stats = _effect_summary(
            raw_effect,
            seed=base_seed + index * 100,
            replicates=replicates,
        )
        adjusted_stats = _effect_summary(
            adjusted,
            seed=base_seed + 50 + index * 100,
            replicates=replicates,
        )
        enough = len(selected) >= minimum
        point_ok = float(raw_stats["estimate"]) >= 0.0
        noninferior = float(adjusted_stats["ci95_low"]) >= 0.0
        passed = bool(enough and point_ok and noninferior)
        all_passed = all_passed and passed
        results[str(name)] = {
            "status": "PASS" if passed else "FAIL",
            "patients": int(len(selected)),
            "minimum_patients": minimum,
            "raw_effect": {
                **raw_stats,
                "definition": (
                    "MSE(H3 fixed schedule) - MSE(uncertainty-aware)"
                ),
            },
            "improved_patient_fraction": _improvement_summary(
                raw_effect,
                seed=base_seed + 75 + index * 100,
                replicates=replicates,
            ),
            "fold_margin_adjusted_effect": {
                **adjusted_stats,
                "definition": "raw effect + fold-frozen margin",
            },
            "requirements": {
                "raw_effect_estimate": ">= 0",
                "adjusted_effect_ci95_low": ">= 0",
            },
        }
    return results, all_passed


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one frozen five-fold patient-level H4-v2 internal "
            "exploratory outer evaluation."
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--config",
        default="configs/h4_v2_internal_exploratory_nested_cv_v1.json",
    )
    parser.add_argument(
        "--plan",
        default=(
            "results/mechanism_validation_v2/"
            "02_h4_v2_internal_exploratory_nested_cv/"
            "00_frozen_plan/frozen_plan.json"
        ),
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
            "02_h4_v2_internal_exploratory_nested_cv/"
            "01_outer_evaluation"
        ),
    )
    return parser


def _verify_frozen_inputs(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    plan: Mapping[str, Any],
    h4_decision_path: Path,
    h4_samples_path: Path,
    h1_samples_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config_sha256 = validate_sealed_mapping(
        config,
        hash_field="config_sha256",
    )
    validate_sealed_mapping(plan, hash_field="plan_sha256")
    if plan.get("status") != "FROZEN_BEFORE_OUTER_EVALUATION":
        raise ValueError("H4-v2 internal plan is not frozen")
    if plan.get("pipeline_id") != config.get("pipeline_id"):
        raise ValueError("Plan and config pipeline IDs differ")
    if plan.get("config_sha256") != config_sha256:
        raise ValueError("Plan and config hashes differ")
    current_sources = {
        "config_file_sha256": file_sha256(config_path),
        "h4_v1_decision_sha256": file_sha256(h4_decision_path),
        "h4_v1_sample_evidence_sha256": file_sha256(h4_samples_path),
        "h1_sample_metrics_sha256": file_sha256(h1_samples_path),
    }
    if plan.get("source_artifacts") != current_sources:
        raise ValueError("A source artifact changed after plan freeze")
    h4_v1 = load_json(h4_decision_path)
    if h4_v1.get("decision") != "FAIL":
        raise ValueError("The immutable H4-v1 outcome must remain FAIL")
    frozen_files = plan["frozen_files"]
    outer_path = _resolve(
        root,
        frozen_files["outer_patient_assignments"]["path"],
    )
    nested_path = _resolve(
        root,
        frozen_files["nested_patient_roles"]["path"],
    )
    if file_sha256(outer_path) != frozen_files[
        "outer_patient_assignments"
    ]["sha256"]:
        raise ValueError("Frozen outer patient assignments changed")
    if file_sha256(nested_path) != frozen_files[
        "nested_patient_roles"
    ]["sha256"]:
        raise ValueError("Frozen nested patient roles changed")
    outer = _read_patient_csv(outer_path)
    nested = _read_patient_csv(nested_path)
    observed_fingerprint = partition_fingerprint(outer, nested)
    if observed_fingerprint != plan["partition"]["fingerprint_sha256"]:
        raise ValueError("Frozen partition fingerprint mismatch")
    assert_patient_partition_integrity(
        outer,
        nested,
        patient_ids=outer["patient_id"],
        outer_folds=int(config["partition"]["outer_folds"]),
    )
    return outer, nested


def _run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    output = _resolve(root, args.output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = _resolve(root, args.config)
    plan_path = _resolve(root, args.plan)
    h4_decision_path = _resolve(root, args.h4_v1_decision)
    h4_samples_path = _resolve(root, args.h4_v1_samples)
    h1_samples_path = _resolve(root, args.h1_samples)
    config = load_json(config_path)
    plan = load_json(plan_path)

    existing_decision_path = output / "decision.json"
    if existing_decision_path.exists():
        existing = load_json(existing_decision_path)
        if (
            existing.get("stage") == STAGE
            and existing.get("pipeline_id") == config.get("pipeline_id")
            and existing.get("plan_sha256") == plan.get("plan_sha256")
            and existing.get("decision") in {"PASS", "FAIL"}
            and "conclusion" in existing
        ):
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return 0 if existing["decision"] == "PASS" else 2
        if (
            existing.get("failure_phase")
            == "OUTER_EVALUATION_EXCEPTION_FAIL_CLOSED"
            and existing.get("conclusion_not_issued_due_to_technical_failure")
            is True
            and not (output / "out_of_fold_patient_report.csv").exists()
        ):
            archive = output / "technical_failure_before_outer_evaluation.json"
            if archive.exists():
                raise ValueError(
                    "A prior technical-failure archive already exists; "
                    "refusing a second retry"
                )
            existing_decision_path.replace(archive)
        else:
            raise ValueError(
                "Outer-evaluation output already exists but is not a reusable "
                "completed result; refusing to overwrite it"
            )

    outer, nested = _verify_frozen_inputs(
        root=root,
        config=config,
        config_path=config_path,
        plan=plan,
        h4_decision_path=h4_decision_path,
        h4_samples_path=h4_samples_path,
        h1_samples_path=h1_samples_path,
    )
    h4_rows = _read_csv(h4_samples_path)
    h1_rows = _read_csv(h1_samples_path)
    if set(h4_rows["sample_id"]) != set(h1_rows["sample_id"]):
        raise ValueError("H1 and H4-v1 sample identities differ")
    if set(h4_rows["band"].astype(str)) != set(BANDS):
        raise ValueError("H4-v1 band set differs from H4-v2")
    all_patients = set(outer["patient_id"].astype(str))
    if set(h4_rows["patient_id"]) != all_patients:
        raise ValueError("Frozen patients and H4-v1 patients differ")

    context = add_context_features(h4_rows)
    heldout_predictions: list[pd.DataFrame] = []
    heldout_errors: list[pd.DataFrame] = []
    heldout_attributes: list[pd.DataFrame] = []
    fold_fitted_components: list[dict[str, Any]] = []
    outer_folds = int(config["partition"]["outer_folds"])
    confidence_quantile = float(
        config["mechanism"]["confidence_quantile_from_inner_calibration"]
    )
    for outer_fold in range(outer_folds):
        current_roles = nested[nested["outer_fold"] == outer_fold]
        roles = dict(
            zip(
                current_roles["patient_id"].astype(str),
                current_roles["role"].astype(str),
                strict=True,
            )
        )
        fold_context = context.copy()
        fold_context["partition"] = fold_context["patient_id"].map(roles)
        population_stats = fit_population_stats(fold_context)
        calibrated = apply_hierarchical_calibration(
            fold_context,
            population_stats,
        )
        original_standardization = fit_original_standardization(calibrated)
        models = fit_comparators(
            calibrated,
            original_standardization,
            alpha=float(config["mechanism"]["ridge_alpha"]),
        )
        calibration_confidence = calibrated.loc[
            calibrated["partition"] == "calibration",
            "evidence_confidence",
        ].to_numpy(dtype=np.float64)
        if not calibration_confidence.size:
            raise ValueError(f"No calibration confidence in fold {outer_fold}")
        confidence_threshold = float(
            np.quantile(calibration_confidence, confidence_quantile)
        )
        predictions = apply_comparators(
            calibrated,
            models=models,
            original_standardization=original_standardization,
            confidence_threshold=confidence_threshold,
        )
        all_fold_errors = patient_errors(predictions)

        fold_h1 = h1_rows.copy()
        fold_h1["partition"] = fold_h1["patient_id"].map(roles)
        attributes = fit_patient_attributes(fold_h1)
        subgroup_spec = freeze_subgroup_spec(
            attributes,
            all_fold_errors,
        )
        attributes = assign_subgroups(attributes, subgroup_spec)

        held_patients = {
            patient
            for patient, role in roles.items()
            if role == "outer_evaluation"
        }
        held_prediction = predictions[
            predictions["patient_id"].isin(held_patients)
        ].copy()
        held_prediction["partition"] = "internal_cv"
        held_prediction["outer_fold"] = outer_fold
        held_error = patient_errors(held_prediction)
        held_error["outer_fold"] = outer_fold
        held_attribute = attributes[
            attributes["patient_id"].isin(held_patients)
        ].copy()
        held_attribute["partition"] = "internal_cv"
        held_attribute["outer_fold"] = outer_fold
        held_attribute["fold_noninferiority_margin"] = float(
            subgroup_spec["noninferiority_margin"]
        )
        original_meta = outer[
            [
                "patient_id",
                "manifest_split",
                "original_partition",
            ]
        ]
        held_attribute = held_attribute.merge(
            original_meta,
            on="patient_id",
            how="left",
            validate="one_to_one",
        )

        heldout_predictions.append(held_prediction)
        heldout_errors.append(held_error)
        heldout_attributes.append(held_attribute)
        role_counts = current_roles["role"].value_counts().to_dict()
        fold_fitted_components.append(
            {
                "outer_fold": outer_fold,
                "patient_counts": {
                    str(role): int(role_counts.get(role, 0))
                    for role in (
                        "mechanism_train",
                        "calibration",
                        "outer_evaluation",
                    )
                },
                "confidence_threshold": confidence_threshold,
                "confidence_quantile": confidence_quantile,
                "subgroup_spec": subgroup_spec,
                "population_stats": population_stats,
                "original_standardization": original_standardization,
                "models": models,
                "outer_patient_used_for_fit_or_threshold": False,
            }
        )
        print(
            f"[H4-v2 internal exploratory] outer fold "
            f"{outer_fold + 1}/{outer_folds}: "
            f"{len(held_patients)} held patients",
            flush=True,
        )

    predictions = pd.concat(heldout_predictions, ignore_index=True)
    errors = pd.concat(heldout_errors, ignore_index=True)
    attributes = pd.concat(heldout_attributes, ignore_index=True)
    if errors["patient_id"].duplicated().any():
        raise ValueError("A patient has multiple out-of-fold error rows")
    if set(errors["patient_id"]) != all_patients:
        raise ValueError("Out-of-fold errors do not cover every patient")
    if len(errors) != int(config["dataset_constraints"]["patients"]):
        raise ValueError("Out-of-fold patient count differs from the config")

    report = errors.merge(
        attributes.drop(columns=["partition"]),
        on=["patient_id", "outer_fold"],
        how="left",
        validate="one_to_one",
    )
    report["effect_uncertainty_vs_h3"] = (
        report["mse_h3_fixed_schedule"]
        - report["mse_uncertainty_aware"]
    )
    report["effect_uncertainty_vs_original"] = (
        report["mse_original_evidence"]
        - report["mse_uncertainty_aware"]
    )
    report["improved_vs_h3"] = report[
        "effect_uncertainty_vs_h3"
    ] > 0.0
    report["improved_vs_original"] = report[
        "effect_uncertainty_vs_original"
    ] > 0.0

    gates = config["gates"]
    comparisons = _all_comparisons(errors, gates)
    subgroup_results, subgroup_pass = _subgroup_statistics(report, gates)
    primary_pass = True
    for name in gates["primary_comparisons"]:
        current = comparisons[str(name)]
        current_pass = bool(
            float(current["ci95_low"]) > 0.0
            and float(current["sign_flip_p"]) < 0.05
            and int(current["positive_outer_folds"])
            >= int(gates["minimum_positive_outer_folds"])
        )
        current["gate_status"] = "PASS" if current_pass else "FAIL"
        primary_pass = primary_pass and current_pass
    active_fraction = float(report["active_fraction"].mean())
    active_pass = active_fraction >= float(
        gates["minimum_router_active_patient_mean"]
    )
    required_difficult = {
        str(value).zfill(3)
        for value in config["dataset_constraints"][
            "required_difficult_patients"
        ]
    }
    difficult = report[
        report["patient_id"].astype(str).isin(required_difficult)
    ].sort_values("patient_id")
    difficult_present = set(difficult["patient_id"].astype(str))
    inclusion_pass = (
        difficult_present == required_difficult
        and len(report) == len(all_patients)
        and set(report["patient_id"].astype(str)) == all_patients
    )
    passed = bool(
        primary_pass and active_pass and subgroup_pass and inclusion_pass
    )

    fold_results: list[dict[str, Any]] = []
    for outer_fold, group in errors.groupby("outer_fold", sort=True):
        fold_results.append(
            {
                "outer_fold": int(outer_fold),
                "patients": int(len(group)),
                "comparisons": _all_comparisons(
                    group,
                    gates,
                    seed_offset=50_000 + int(outer_fold) * 10_000,
                ),
                "router_active_fraction_patient_mean": float(
                    group["active_fraction"].mean()
                ),
            }
        )
    original_partition_results: dict[str, Any] = {}
    for index, (name, group) in enumerate(
        report.groupby("original_partition", sort=True)
    ):
        original_partition_results[str(name)] = {
            "patients": int(len(group)),
            "previously_exposed": str(name) == "validation",
            "comparisons": _all_comparisons(
                group,
                gates,
                seed_offset=100_000 + index * 10_000,
            ),
        }

    predictions.to_csv(
        output / "out_of_fold_sample_predictions.csv",
        index=False,
    )
    errors.to_csv(output / "out_of_fold_patient_errors.csv", index=False)
    report.to_csv(output / "out_of_fold_patient_report.csv", index=False)
    difficult.to_csv(output / "difficult_patients_044_153.csv", index=False)
    write_json(
        output / "difficult_patients_044_153.json",
        {
            "required": sorted(required_difficult),
            "excluded": [],
            "records": _frame_records(difficult),
        },
    )
    write_json(
        output / "fold_fitted_components.json",
        fold_fitted_components,
    )
    write_json(output / "outer_fold_results.json", fold_results)
    write_json(output / "comparison_statistics.json", comparisons)
    write_json(output / "subgroup_statistics.json", subgroup_results)
    write_json(
        output / "original_partition_statistics.json",
        original_partition_results,
    )

    allowed_conclusions = config["allowed_conclusions"]
    conclusion = (
        allowed_conclusions["supported"]
        if passed
        else allowed_conclusions["not_supported"]
    )
    decision = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "pipeline_id": config["pipeline_id"],
        "decision": "PASS" if passed else "FAIL",
        "conclusion": conclusion,
        "scope": config["scope"],
        "plan_sha256": plan["plan_sha256"],
        "partition_fingerprint": plan["partition"]["fingerprint_sha256"],
        "config_sha256": config["config_sha256"],
        "patients": int(len(report)),
        "samples": int(h1_rows["sample_id"].nunique()),
        "outer_folds": outer_folds,
        "patients_excluded": 0,
        "H4_v1_preserved": "FAIL",
        "H4_v1_outcome_changed": False,
        "validation_exposure_disclosure": {
            "original_validation_patients": 31,
            "previously_used_in_v1_H1_to_H4": True,
            "role_in_this_analysis": (
                "included in patient-level nested cross-validation; "
                "not treated as a fresh cohort"
            ),
        },
        "gate_status": {
            "primary_comparisons": "PASS" if primary_pass else "FAIL",
            "router_active_coverage": "PASS" if active_pass else "FAIL",
            "robustness_subgroups": "PASS" if subgroup_pass else "FAIL",
            "complete_patient_inclusion": (
                "PASS" if inclusion_pass else "FAIL"
            ),
        },
        "comparisons": comparisons,
        "router_active_fraction_patient_mean": active_fraction,
        "minimum_router_active_patient_mean": float(
            gates["minimum_router_active_patient_mean"]
        ),
        "robustness_subgroups": subgroup_results,
        "difficult_patients": {
            "required": sorted(required_difficult),
            "present": sorted(difficult_present),
            "excluded": [],
            "records": _frame_records(difficult),
        },
        "guardrails": {
            "split_unit": "patient",
            "slice_level_random_split": False,
            "each_patient_out_of_fold_exactly_once": True,
            "outer_patient_used_for_corresponding_fit_or_threshold": False,
            "threshold_source": "corresponding inner patient partitions only",
            "patient_bootstrap_95ci": True,
            "outer_results_used_to_modify_frozen_spec": False,
            "causal_or_mutual_information_claimed": False,
        },
        "recommended_for_subsequent_internal_use": passed,
        "source_artifacts": plan["source_artifacts"],
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
            "frozen_before_outer_evaluation": True,
            "outer_results_used_for_tuning": False,
        },
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if passed else 2


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
            "failure_phase": "OUTER_EVALUATION_EXCEPTION_FAIL_CLOSED",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "conclusion_not_issued_due_to_technical_failure": True,
            "recommended_for_subsequent_internal_use": False,
        }
        write_json(output / "decision.json", decision)
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
