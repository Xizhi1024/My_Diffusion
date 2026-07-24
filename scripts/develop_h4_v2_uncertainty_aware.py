"""Freeze the H4-v2 uncertainty-aware evidence protocol.

This command is intentionally a development-only operation.  The original
31 validation patients have already been exposed by H4-v1, so their v2
results are diagnostics and can never make this stage PASS.  Formal H4-v2
confirmation requires the separately frozen external-confirmation command.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mechanism_validation.common import (
    canonical_json_sha256,
    file_sha256,
    load_json,
    write_json,
)
from src.mechanism_validation.h4_v2 import (
    BANDS,
    CONFIDENCE_QUANTILE,
    MIN_ACTIVE_COVERAGE,
    MIN_CONFIRMATION_PATIENTS,
    MIN_SUBGROUP_PATIENTS,
    MODEL_ORDER,
    NEIGHBOR_RADIUS,
    PATIENT_SHRINKAGE_K,
    RIDGE_ALPHA,
    add_context_features,
    apply_comparators,
    apply_hierarchical_calibration,
    assign_subgroups,
    development_diagnostics,
    effect_statistics,
    fit_comparators,
    fit_original_standardization,
    fit_patient_attributes,
    fit_population_stats,
    freeze_subgroup_spec,
    normalize_ids,
    patient_errors,
    patient_set_sha256,
    seal_probe,
    validate_probe,
)


SCHEMA_VERSION = 2
ANALYSIS_SEED = 20260730
PROTOCOL_FROZEN_AT_UTC = "2026-07-23T08:40:00+00:00"


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _validate_contract(contract: dict[str, Any]) -> str:
    claimed = str(contract.get("contract_sha256", ""))
    body = dict(contract)
    body.pop("contract_sha256", None)
    computed = canonical_json_sha256(body)
    if not claimed or claimed != computed:
        raise ValueError(
            f"Development dataset contract self-hash mismatch: "
            f"{claimed!r} != {computed!r}"
        )
    return claimed


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"patient_id": str, "sample_id": str},
    )


def _check_sources(
    *,
    contract: dict[str, Any],
    h3_decision: dict[str, Any],
    h4_decision: dict[str, Any],
    h4_rows: pd.DataFrame,
    h1_rows: pd.DataFrame,
) -> tuple[str, list[int]]:
    contract_sha = str(contract["contract_sha256"])
    if h3_decision.get("decision") != "PASS":
        raise ValueError("H4-v2 requires the frozen H3 stage to be PASS")
    if h4_decision.get("decision") != "FAIL":
        raise ValueError(
            "H4-v2 is a new protocol after an explicit H4-v1 FAIL"
        )
    for name, decision in (
        ("H3", h3_decision),
        ("H4-v1", h4_decision),
    ):
        if decision.get("dataset_contract_sha256") != contract_sha:
            raise ValueError(f"{name} dataset contract lineage mismatch")
    h4_rows = normalize_ids(h4_rows)
    h1_rows = normalize_ids(h1_rows)
    required_partitions = {"mechanism_train", "calibration", "validation"}
    if set(h4_rows["partition"]) != required_partitions:
        raise ValueError("H4-v1 sample rows have unexpected partitions")
    if set(h1_rows["partition"]) != required_partitions:
        raise ValueError("H1 sample rows have unexpected partitions")
    h4_samples = set(h4_rows["sample_id"])
    h1_samples = set(h1_rows["sample_id"])
    if h4_samples != h1_samples:
        raise ValueError(
            "H1/H4 sample identity mismatch; v2 development is not allowed"
        )
    patient_partition_counts = (
        h4_rows.groupby("partition")["patient_id"].nunique().to_dict()
    )
    if patient_partition_counts != {
        "calibration": 25,
        "mechanism_train": 99,
        "validation": 31,
    }:
        raise ValueError(
            "H4-v2 development patient partitions do not match the "
            f"locked 99/25/31 split: {patient_partition_counts}"
        )
    timesteps = sorted(
        int(value) for value in h4_rows["timestep"].unique()
    )
    if len(timesteps) < 3:
        raise ValueError("H4-v2 needs a multi-time bridge evidence grid")
    expected_rows = len(h4_samples) * len(BANDS) * len(timesteps)
    if len(h4_rows) != expected_rows:
        raise ValueError(
            f"H4-v1 sample evidence is incomplete: "
            f"{len(h4_rows)} != {expected_rows}"
        )
    return contract_sha, timesteps


def _subgroup_diagnostics(
    errors: pd.DataFrame,
    attributes: pd.DataFrame,
    subgroup_spec: dict[str, Any],
) -> dict[str, Any]:
    assigned = assign_subgroups(attributes, subgroup_spec)
    merged = errors.merge(
        assigned,
        on=["partition", "patient_id"],
        how="left",
        validate="one_to_one",
    )
    diagnostics: dict[str, Any] = {
        "partition": "validation",
        "exploratory_only": True,
        "formal_gate_eligible": False,
    }
    validation = merged[merged["partition"] == "validation"]
    for index, name in enumerate(
        ("few_slices", "small_lesion", "low_contrast")
    ):
        member_ids = set(
            validation.loc[validation[name].fillna(False), "patient_id"]
        )
        subset = errors[
            (errors["partition"] == "validation")
            & errors["patient_id"].isin(member_ids)
        ]
        diagnostics[name] = {
            "patients": len(member_ids),
            "uncertainty_vs_h3": effect_statistics(
                subset,
                partition="validation",
                left="uncertainty_aware",
                right="h3_fixed_schedule",
                seed=ANALYSIS_SEED + 1000 + index * 100,
            ),
        }
    return diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Develop and freeze H4-v2; never confirms it on the exposed "
            "31-patient validation cohort."
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--contract",
        default="configs/dataset_contract_stage0a_v1.json",
    )
    parser.add_argument(
        "--h3-decision",
        default=(
            "results/mechanism_validation/"
            "02_h3_recoverability_curves/decision.json"
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
            "00_h4_v2_development"
        ),
    )
    parser.add_argument(
        "--allow-refreeze",
        action="store_true",
        help=(
            "Development migration only: replace a mismatching frozen "
            "probe while archiving the previous probe. Never use this "
            "during external confirmation."
        ),
    )
    return parser


def _run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    contract_path = _resolve(root, args.contract)
    h3_decision_path = _resolve(root, args.h3_decision)
    h4_decision_path = _resolve(root, args.h4_v1_decision)
    h4_samples_path = _resolve(root, args.h4_v1_samples)
    h1_samples_path = _resolve(root, args.h1_samples)
    output = _resolve(root, args.output)
    output.mkdir(parents=True, exist_ok=True)

    contract = load_json(contract_path)
    h3_decision = load_json(h3_decision_path)
    h3_spec_path = h3_decision_path.parent / "analysis_spec.json"
    h3_spec = load_json(h3_spec_path)
    h4_decision = load_json(h4_decision_path)
    h4_rows = _read_csv(h4_samples_path)
    h1_rows = _read_csv(h1_samples_path)
    contract_sha, timesteps = _check_sources(
        contract=contract,
        h3_decision=h3_decision,
        h4_decision=h4_decision,
        h4_rows=h4_rows,
        h1_rows=h1_rows,
    )
    if timesteps != [int(value) for value in h3_spec["timesteps"]]:
        raise ValueError("H4-v1 evidence grid differs from the frozen H3 grid")
    h4_rows = normalize_ids(h4_rows)
    h1_rows = normalize_ids(h1_rows)

    # Fit paths below use mechanism_train and calibration only.  The exposed
    # validation patients are not evaluated until after the probe is sealed.
    context = add_context_features(h4_rows)
    population_stats = fit_population_stats(context)
    calibrated = apply_hierarchical_calibration(
        context,
        population_stats,
    )
    original_standardization = fit_original_standardization(calibrated)
    fit_only = calibrated[
        calibrated["partition"].isin(("mechanism_train", "calibration"))
    ].copy()
    models = fit_comparators(fit_only, original_standardization)
    calibration_confidence = fit_only.loc[
        fit_only["partition"] == "calibration",
        "evidence_confidence",
    ].to_numpy(dtype=np.float64)
    confidence_threshold = float(
        np.quantile(calibration_confidence, CONFIDENCE_QUANTILE)
    )
    fit_predictions = apply_comparators(
        fit_only,
        models=models,
        original_standardization=original_standardization,
        confidence_threshold=confidence_threshold,
    )
    fit_errors = patient_errors(fit_predictions)
    attributes = fit_patient_attributes(h1_rows)
    fit_attributes = attributes[
        attributes["partition"].isin(("mechanism_train", "calibration"))
    ]
    subgroup_spec = freeze_subgroup_spec(fit_attributes, fit_errors)

    all_patient_ids = sorted(set(h4_rows["patient_id"]))
    exposed_validation_ids = sorted(
        set(
            h4_rows.loc[
                h4_rows["partition"] == "validation",
                "patient_id",
            ]
        )
    )
    checkpoint = h4_decision.get("mean_checkpoint", {})
    probe = seal_probe(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": "H4_uncertainty_aware_evidence_v2",
            "status": "FROZEN_FOR_NEW_COHORT_CONFIRMATION",
            "development_only": True,
            "frozen_at_utc": PROTOCOL_FROZEN_AT_UTC,
            "claim": (
                "confidence-qualified residual band evidence predicts "
                "band recoverability beyond the H3 fixed schedule"
            ),
            "wording_boundary": (
                "predictive evidence only; no causal or "
                "mutual-information claim"
            ),
            "v1_H4_outcome": "FAIL",
            "development_dataset": {
                "cohort_id": "internal_png_stage0a_v1",
                "dataset_contract_sha256": contract_sha,
                "preprocessing_config_sha256": contract[
                    "preprocessing_config_sha256"
                ],
                "patient_ids": all_patient_ids,
                "patient_set_sha256": patient_set_sha256(all_patient_ids),
                "exposed_validation_patient_ids": exposed_validation_ids,
                "patients": len(all_patient_ids),
                "validation_patients": len(exposed_validation_ids),
                "formal_confirmation_eligible": False,
            },
            "source_artifacts": {
                "dataset_contract": {
                    "sha256": file_sha256(contract_path),
                    "semantic_contract_sha256": contract_sha,
                    "manifest_semantic_sha256": contract["manifest"][
                        "semantic_sha256"
                    ],
                    "raw_png_combined_sha256": contract["raw_png"][
                        "combined_sha256"
                    ],
                },
                "h3_decision": {
                    "sha256": file_sha256(h3_decision_path),
                },
                "h3_analysis_spec": {
                    "sha256": file_sha256(h3_spec_path),
                },
                "h4_v1_decision": {
                    "sha256": file_sha256(h4_decision_path),
                },
                "h4_v1_sample_evidence": {
                    "sha256": file_sha256(h4_samples_path),
                },
                "h1_sample_metrics": {
                    "sha256": file_sha256(h1_samples_path),
                },
                "pathology_excluded_mean_checkpoint": {
                    "sha256": str(checkpoint.get("sha256", "")),
                },
                "implementations": {
                    "h4_v2_module_sha256": file_sha256(
                        root / "src/mechanism_validation/h4_v2.py"
                    ),
                    "development_script_sha256": file_sha256(
                        Path(__file__).resolve()
                    ),
                    "external_confirmation_script_sha256": file_sha256(
                        root
                        / "scripts/validate_h4_v2_external_confirmation.py"
                    ),
                    "h4_v1_evidence_script_sha256": file_sha256(
                        root
                        / "scripts/validate_h4_noise_band_calibration.py"
                    ),
                    "h1_attribute_script_sha256": file_sha256(
                        root
                        / "scripts/validate_h1_local_spectral_asymmetry.py"
                    ),
                },
            },
            "bands": list(BANDS),
            "timesteps": timesteps,
            "model_order": list(MODEL_ORDER),
            "mechanism": {
                "bridge_schedule": {
                    "num_train_timesteps": int(
                        h3_spec["num_train_timesteps"]
                    ),
                    "m_schedule": "linear",
                    "sigma_scale": float(h3_spec["sigma_scale"]),
                },
                "evidence_generation": {
                    "analysis_seed": 20260726,
                    "batch_size": 8,
                    "noise_pairing": (
                        "same seeded noise for observed residual and "
                        "pure-noise reference"
                    ),
                },
                "context": {
                    "adjacent_slice_radius": NEIGHBOR_RADIUS,
                    "cross_scale_peer": (
                        "same sample, timestep, and orientation"
                    ),
                    "aggregator": "median",
                    "dispersion": "median_absolute_deviation",
                },
                "hierarchical_calibration": {
                    "population_unit": "band_by_timestep",
                    "population_fit_partition": "mechanism_train",
                    "patient_band_centering": "shrunken_median_offset",
                    "patient_shrinkage_k": PATIENT_SHRINKAGE_K,
                    "population_stats": population_stats,
                },
                "confidence": {
                    "formula": (
                        "sqrt(local_confidence * "
                        "patient_band_timestep_median_local_confidence), "
                        "where local_confidence = "
                        "exp(-context_MAD/population_scale) * "
                        "min(context_count/4, 1)"
                    ),
                    "decision_unit": "patient_by_band_by_timestep_and_slice",
                    "threshold": confidence_threshold,
                    "threshold_source": "calibration_q25",
                    "quantile": CONFIDENCE_QUANTILE,
                    "abstention_fallback": "h3_fixed_schedule",
                },
                "original_standardization": original_standardization,
                "models": models,
                "ridge_alpha_fixed_without_sweep": RIDGE_ALPHA,
            },
            "robustness_subgroups": subgroup_spec,
            "formal_confirmation_gate": {
                "confirmation_data": (
                    "new unseen patients or an external cohort only"
                ),
                "development_patient_overlap_required": 0,
                "different_dataset_contract_required": True,
                "same_preprocessing_config_required": True,
                "minimum_total_patients": MIN_CONFIRMATION_PATIENTS,
                "minimum_patients_per_subgroup": MIN_SUBGROUP_PATIENTS,
                "minimum_router_active_fraction": MIN_ACTIVE_COVERAGE,
                "patient_bootstrap_replicates": 10_000,
                "confidence_level": 0.95,
                "patient_sign_flip_replicates": 10_000,
                "primary_superiority": {
                    "comparisons": [
                        "uncertainty_aware_vs_h3_fixed_schedule",
                        "uncertainty_aware_vs_original_evidence",
                    ],
                    "required_ci95_low": "> 0",
                    "required_one_sided_sign_flip_p": "< 0.05",
                },
                "robustness_noninferiority": {
                    "subgroups": [
                        "few_slices",
                        "small_lesion",
                        "low_contrast",
                    ],
                    "required_effect_estimate": ">= 0",
                    "required_ci95_low": (
                        ">= -frozen_noninferiority_margin"
                    ),
                },
                "reported_not_gated": [
                    "h3_fixed_schedule_vs_no_route",
                    "original_evidence_vs_h3_fixed_schedule",
                ],
            },
            "downstream_unlock_policy": {
                "requires_formal_H4_v2_PASS": True,
                "blocked_until_pass": [
                    "CT_support_head",
                    "curriculum",
                    "H5_router_role_separation",
                    "H6_final_integration",
                ],
            },
        }
    )
    probe_path = output / "frozen_probe.json"
    if probe_path.is_file():
        existing_probe = load_json(probe_path)
        validate_probe(existing_probe)
        if existing_probe["probe_sha256"] != probe["probe_sha256"]:
            if not args.allow_refreeze:
                raise ValueError(
                    "Existing H4-v2 probe differs from the current "
                    "candidate. Refusing to overwrite a frozen protocol. "
                    "Use --allow-refreeze only for an explicitly declared "
                    "development migration before any external confirmation."
                )
            archive = output / (
                "superseded_probe_"
                f"{existing_probe['probe_sha256']}.json"
            )
            if not archive.exists():
                write_json(archive, existing_probe)
            write_json(probe_path, probe)
        else:
            probe = existing_probe
    else:
        write_json(probe_path, probe)

    # The sealed probe now exists.  Only after sealing do we inspect the
    # already-exposed validation patients for development diagnostics.
    predictions = apply_comparators(
        calibrated,
        models=models,
        original_standardization=original_standardization,
        confidence_threshold=confidence_threshold,
    )
    errors = patient_errors(predictions)
    diagnostics = development_diagnostics(errors)
    calibration_diagnostics = development_diagnostics(
        errors,
        partition="calibration",
        seed=ANALYSIS_SEED + 5000,
    )
    coverage = {
        str(partition): {
            "patients": int(len(group)),
            "router_active_fraction_patient_mean": float(
                group["active_fraction"].mean()
            ),
            "router_abstained_fraction_patient_mean": float(
                1.0 - group["active_fraction"].mean()
            ),
            "mean_confidence_patient_mean": float(
                group["mean_confidence"].mean()
            ),
        }
        for partition, group in errors.groupby("partition", sort=True)
    }
    subgroup_diagnostics = _subgroup_diagnostics(
        errors,
        attributes,
        subgroup_spec,
    )
    assigned_attributes = assign_subgroups(attributes, subgroup_spec)

    predictions.to_csv(
        output / "development_sample_predictions.csv",
        index=False,
    )
    errors.to_csv(output / "development_patient_errors.csv", index=False)
    assigned_attributes.to_csv(
        output / "development_patient_attributes.csv",
        index=False,
    )
    write_json(output / "development_diagnostics.json", diagnostics)
    write_json(
        output / "calibration_diagnostics.json",
        calibration_diagnostics,
    )
    write_json(output / "router_coverage.json", coverage)
    write_json(
        output / "development_subgroup_diagnostics.json",
        subgroup_diagnostics,
    )

    decision = {
        "schema_version": SCHEMA_VERSION,
        "stage": "00_H4_v2_development_and_freeze",
        "decision": "DEVELOPMENT_ONLY",
        "development_status": "FROZEN",
        "formal_confirmation_status": "DEFERRED_NEW_COHORT",
        "H4_v1_preserved": "FAIL",
        "dataset_contract_sha256": contract_sha,
        "probe_sha256": probe["probe_sha256"],
        "development_patients": len(all_patient_ids),
        "exposed_validation_patients": len(exposed_validation_ids),
        "development_diagnostics": diagnostics,
        "calibration_diagnostics": calibration_diagnostics,
        "router_coverage": coverage,
        "confidence_threshold": confidence_threshold,
        "development_subgroup_diagnostics": subgroup_diagnostics,
        "guardrails": {
            "patient_unit": "patient",
            "threshold_source": "mechanism_train_and_calibration_only",
            "validation_used_for_failure_localization_and_v2_design": True,
            "validation_used_for_numeric_model_fit_or_threshold_selection": (
                False
            ),
            "validation_diagnostics_are_exploratory_only": True,
            "patient_bootstrap_95ci": True,
            "causal_or_mutual_information_claimed": False,
            "new_cohort_required_for_formal_confirmation": True,
        },
        "model_mechanism_claims_allowed": False,
        "next_stage_allowed": False,
        "stop_rule": (
            "Do not resume CT head, curriculum, H5, or H6 until the "
            "frozen H4-v2 probe passes on a zero-overlap new cohort."
        ),
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
            "probe_sha256": probe["probe_sha256"],
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
            "stage": "00_H4_v2_development_and_freeze",
            "decision": "FAIL",
            "failure_phase": "EXCEPTION_FAIL_CLOSED",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "H4_v1_preserved": "FAIL",
            "formal_confirmation_status": "BLOCKED",
            "model_mechanism_claims_allowed": False,
            "next_stage_allowed": False,
            "stop_rule": (
                "H4-v2 development/freeze did not complete cleanly; "
                "do not run external confirmation or downstream stages."
            ),
        }
        write_json(output / "decision.json", decision)
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
