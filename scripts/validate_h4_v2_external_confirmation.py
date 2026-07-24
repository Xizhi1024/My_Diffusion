"""Confirm the frozen H4-v2 probe on a genuinely new patient cohort.

The command fails closed on patient overlap, data/preprocessing lineage,
probe mutation, undersized cohorts, weak active coverage, either primary
comparison, or any pre-registered robustness subgroup.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cloud_stage0c_gate import _manifest_semantic_sha256
from scripts.validate_h1_local_spectral_asymmetry import (
    RING_RADIUS,
    _band_energy_maps,
)
from scripts.validate_h2_pathology_excluded_residual import (
    IndexedDataset,
    _load_predictor,
)
from scripts.validate_h4_noise_band_calibration import _sample_band_rows
from src.data.dataset import CachedDataset
from src.data.lineage import (
    load_checkpoint_data_lineage,
    validate_checkpoint_data_lineage,
)
from src.mechanism_validation.common import (
    canonical_json_sha256,
    file_sha256,
    load_json,
    read_manifest,
    write_json,
)
from src.mechanism_validation.h4_v2 import (
    MODEL_ORDER,
    add_context_features,
    apply_comparators,
    apply_hierarchical_calibration,
    assign_subgroups,
    effect_statistics,
    fit_patient_attributes,
    normalize_ids,
    patient_errors,
    patient_set_sha256,
    validate_probe,
)
from src.model.noise.base import BBDMBridgeSchedule


SCHEMA_VERSION = 2
ANALYSIS_SEED = 20260731


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _validate_contract(contract: Mapping[str, Any]) -> str:
    body = dict(contract)
    claimed = str(body.pop("contract_sha256", ""))
    computed = canonical_json_sha256(body)
    if not claimed or claimed != computed:
        raise ValueError(
            f"External dataset contract self-hash mismatch: "
            f"{claimed!r} != {computed!r}"
        )
    if contract.get("contract_status") != "LOCKED":
        raise ValueError("External dataset contract is not LOCKED")
    return claimed


def _preflight_failure(
    *,
    output: Path,
    probe: Mapping[str, Any],
    cohort_id: str,
    checks: Mapping[str, Any],
) -> int:
    failed = [
        name
        for name, value in checks.items()
        if isinstance(value, Mapping) and value.get("status") != "PASS"
    ]
    decision = {
        "schema_version": SCHEMA_VERSION,
        "stage": "01_H4_v2_external_confirmation",
        "decision": "FAIL",
        "failure_phase": "PREFLIGHT",
        "confirmation_cohort_id": cohort_id,
        "probe_sha256": probe.get("probe_sha256"),
        "preflight_checks": dict(checks),
        "failed_checks": failed,
        "H4_v2": {"status": "FAIL"},
        "next_stage_allowed": False,
        "model_mechanism_claims_allowed": False,
        "stop_rule": (
            "Do not compute or claim H4-v2 on an ineligible cohort; "
            "CT head, curriculum, H5, and H6 remain blocked."
        ),
    }
    write_json(output / "decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 2


def _status(passed: bool, **evidence: Any) -> dict[str, Any]:
    return {"status": "PASS" if passed else "FAIL", **evidence}


def _external_attributes(dataset: IndexedDataset) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(dataset.entries):
        sample = dataset[index]
        pet = (
            (sample["pet"].detach().cpu().numpy()[0].astype(np.float32) + 1.0)
            * 0.5
        )
        mask = (
            sample["mask"].detach().cpu().numpy()[0].astype(np.float32)
            > 0.5
        )
        ring = ndimage.binary_dilation(
            mask,
            iterations=RING_RADIUS,
        ) & ~mask
        if not mask.any() or not ring.any():
            raise ValueError(
                f"External sample has empty lesion/ring: {entry.sample_id}"
            )
        ll2 = _band_energy_maps(pet)["ll2"]
        lesion = float(ll2[mask].mean())
        ring_mean = float(ll2[ring].mean())
        rows.append(
            {
                "partition": "external_confirmation",
                "patient_id": entry.patient_id,
                "sample_id": entry.sample_id,
                "mask_area": int(mask.sum()),
                "pet_ll2_lesion_ring_log_ratio": math.log(
                    (lesion + 1e-8) / (ring_mean + 1e-8)
                ),
            }
        )
        if (index + 1) % 200 == 0:
            print(
                f"[H4-v2 attributes] {index + 1}/{len(dataset)}",
                flush=True,
            )
    return fit_patient_attributes(pd.DataFrame(rows))


def _subgroup_statistics(
    *,
    errors: pd.DataFrame,
    attributes: pd.DataFrame,
    subgroup_spec: Mapping[str, Any],
    minimum_patients: int,
) -> tuple[dict[str, Any], bool]:
    assigned = assign_subgroups(attributes, subgroup_spec)
    merged = errors.merge(
        assigned,
        on=["partition", "patient_id"],
        how="left",
        validate="one_to_one",
    )
    margin = float(subgroup_spec["noninferiority_margin"])
    results: dict[str, Any] = {}
    all_passed = True
    for index, name in enumerate(
        ("few_slices", "small_lesion", "low_contrast")
    ):
        subset = merged[merged[name].fillna(False)].copy()
        stats = effect_statistics(
            subset,
            partition="external_confirmation",
            left="uncertainty_aware",
            right="h3_fixed_schedule",
            seed=ANALYSIS_SEED + 1000 + index * 100,
        )
        enough = int(stats["patients"]) >= minimum_patients
        estimate_ok = float(stats["estimate"]) >= 0.0
        ci_ok = float(stats["ci95_low"]) >= -margin
        passed = bool(enough and estimate_ok and ci_ok)
        all_passed = all_passed and passed
        results[name] = {
            "status": "PASS" if passed else "FAIL",
            "minimum_patients": minimum_patients,
            "noninferiority_margin": margin,
            "requirements": {
                "patients": ">= minimum_patients",
                "effect_estimate": ">= 0",
                "ci95_low": ">= -noninferiority_margin",
            },
            "statistics": stats,
        }
    return results, all_passed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zero-overlap external confirmation of frozen H4-v2."
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--probe",
        default=(
            "results/mechanism_validation_v2/"
            "00_h4_v2_development/frozen_probe.json"
        ),
    )
    parser.add_argument("--external-contract", required=True)
    parser.add_argument("--external-manifest", required=True)
    parser.add_argument("--external-cache-dir", required=True)
    parser.add_argument("--external-cache-lineage", required=True)
    parser.add_argument(
        "--mean-checkpoint",
        default=(
            "results/mechanism_validation/"
            "01_h2_residual_enrichment/checkpoints/"
            "mean_excluded_fixed.pt"
        ),
    )
    parser.add_argument("--confirmation-cohort-id", required=True)
    parser.add_argument("--confirmation-split", default="test")
    parser.add_argument(
        "--output",
        default=(
            "results/mechanism_validation_v2/"
            "01_h4_v2_external_confirmation"
        ),
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    return parser


def _run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    output = _resolve(root, args.output)
    output.mkdir(parents=True, exist_ok=True)
    probe_path = _resolve(root, args.probe)
    contract_path = _resolve(root, args.external_contract)
    manifest_path = _resolve(root, args.external_manifest)
    cache_dir = _resolve(root, args.external_cache_dir)
    cache_lineage_path = _resolve(root, args.external_cache_lineage)
    checkpoint_path = _resolve(root, args.mean_checkpoint)

    probe = load_json(probe_path)
    validate_probe(probe)
    implementation_paths = {
        "h4_v2_module_sha256": (
            root / "src/mechanism_validation/h4_v2.py"
        ),
        "external_confirmation_script_sha256": Path(__file__).resolve(),
        "h4_v1_evidence_script_sha256": (
            root / "scripts/validate_h4_noise_band_calibration.py"
        ),
        "h1_attribute_script_sha256": (
            root / "scripts/validate_h1_local_spectral_asymmetry.py"
        ),
    }
    frozen_implementations = probe["source_artifacts"]["implementations"]
    implementation_mismatches = {
        name: {
            "frozen": frozen_implementations.get(name),
            "current": file_sha256(path),
        }
        for name, path in implementation_paths.items()
        if frozen_implementations.get(name) != file_sha256(path)
    }
    if implementation_mismatches:
        raise ValueError(
            "H4-v2 confirmation implementation differs from frozen probe: "
            + json.dumps(implementation_mismatches, sort_keys=True)
        )
    contract = load_json(contract_path)
    external_contract_sha = _validate_contract(contract)
    manifest = read_manifest(manifest_path)
    actual_manifest_sha = _manifest_semantic_sha256(manifest)
    development = probe["development_dataset"]
    gate = probe["formal_confirmation_gate"]
    cohort_id = str(args.confirmation_cohort_id).strip()

    selected_manifest = [
        row for row in manifest if row["split"] == args.confirmation_split
    ]
    selected_patients = sorted(
        {str(row["patient_id"]) for row in selected_manifest}
    )
    development_patients = {
        str(value) for value in development["patient_ids"]
    }
    overlap = sorted(set(selected_patients) & development_patients)
    checks = {
        "probe_frozen": _status(
            probe.get("status") == "FROZEN_FOR_NEW_COHORT_CONFIRMATION"
            and probe.get("development_only") is True,
            probe_sha256=probe["probe_sha256"],
        ),
        "cohort_identity": _status(
            bool(cohort_id)
            and cohort_id != str(development["cohort_id"]),
            confirmation_cohort_id=cohort_id,
            development_cohort_id=development["cohort_id"],
        ),
        "different_dataset_contract": _status(
            external_contract_sha
            != str(development["dataset_contract_sha256"]),
            external_dataset_contract_sha256=external_contract_sha,
            development_dataset_contract_sha256=development[
                "dataset_contract_sha256"
            ],
        ),
        "same_preprocessing": _status(
            contract.get("preprocessing_config_sha256")
            == development["preprocessing_config_sha256"],
            external_preprocessing_config_sha256=contract.get(
                "preprocessing_config_sha256"
            ),
            required_preprocessing_config_sha256=development[
                "preprocessing_config_sha256"
            ],
        ),
        "manifest_contract": _status(
            actual_manifest_sha
            == contract.get("manifest", {}).get("semantic_sha256"),
            computed_manifest_semantic_sha256=actual_manifest_sha,
            contract_manifest_semantic_sha256=contract.get(
                "manifest", {}
            ).get("semantic_sha256"),
        ),
        "new_raw_bytes": _status(
            contract.get("raw_png", {}).get("combined_sha256")
            != probe["source_artifacts"]["dataset_contract"][
                "raw_png_combined_sha256"
            ],
            external_raw_png_combined_sha256=contract.get(
                "raw_png", {}
            ).get("combined_sha256"),
            development_raw_png_combined_sha256=probe[
                "source_artifacts"
            ]["dataset_contract"]["raw_png_combined_sha256"],
        ),
        "new_manifest_content": _status(
            actual_manifest_sha
            != probe["source_artifacts"]["dataset_contract"][
                "manifest_semantic_sha256"
            ],
            external_manifest_semantic_sha256=actual_manifest_sha,
            development_manifest_semantic_sha256=probe[
                "source_artifacts"
            ]["dataset_contract"]["manifest_semantic_sha256"],
        ),
        "patient_zero_overlap": _status(
            not overlap,
            overlap_count=len(overlap),
            overlap_preview=overlap[:20],
            external_patient_set_sha256=patient_set_sha256(
                selected_patients
            ),
        ),
        "minimum_patient_count": _status(
            len(selected_patients)
            >= int(gate["minimum_total_patients"]),
            patients=len(selected_patients),
            minimum=int(gate["minimum_total_patients"]),
        ),
        "confirmation_split_nonempty": _status(
            bool(selected_manifest),
            confirmation_split=args.confirmation_split,
            samples=len(selected_manifest),
        ),
    }

    dev_contract_sha = development["dataset_contract_sha256"]
    if external_contract_sha == dev_contract_sha:
        checks["new_raw_bytes"]["status"] = "FAIL"

    if any(value["status"] != "PASS" for value in checks.values()):
        return _preflight_failure(
            output=output,
            probe=probe,
            cohort_id=cohort_id,
            checks=checks,
        )

    external_config = {
        "data": {
            "cache_dir": str(cache_dir),
            "cache_lineage": str(cache_lineage_path),
            "dataset_contract": str(contract_path),
            "require_cache_lineage": True,
            "use_fake_data": False,
        }
    }
    external_lineage = load_checkpoint_data_lineage(
        external_config,
        root=root,
    )
    if external_lineage is None:
        raise RuntimeError("External confirmation cache lineage is required")

    expected_checkpoint_sha = probe["source_artifacts"][
        "pathology_excluded_mean_checkpoint"
    ]["sha256"]
    actual_checkpoint_sha = file_sha256(checkpoint_path)
    if actual_checkpoint_sha != expected_checkpoint_sha:
        raise ValueError(
            "Pathology-excluded mean checkpoint hash differs from the "
            "frozen H4-v2 probe"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    embedded = checkpoint.get("data_lineage")
    development_lineage = validate_checkpoint_data_lineage(
        checkpoint,
        embedded if isinstance(embedded, Mapping) else None,
        required=True,
        context="H4-v2 pathology-excluded mean checkpoint",
    )
    if (
        development_lineage["dataset_contract_sha256"]
        != development["dataset_contract_sha256"]
        or development_lineage["preprocessing_config_sha256"]
        != development["preprocessing_config_sha256"]
    ):
        raise ValueError(
            "Mean checkpoint does not carry the frozen development lineage"
        )

    base = CachedDataset(
        cache_dir,
        split=args.confirmation_split,
        augment=False,
        split_manifest=manifest_path,
        required_keys=["ct", "pet", "mask"],
    )
    dataset = IndexedDataset(base, range(len(base)))
    cache_samples = {entry.sample_id for entry in dataset.entries}
    expected_samples = {row["sample_id"] for row in selected_manifest}
    if cache_samples != expected_samples:
        raise ValueError(
            "External confirmation cache/manifest sample mismatch"
        )
    if {entry.patient_id for entry in dataset.entries} != set(
        selected_patients
    ):
        raise ValueError(
            "CachedDataset patient IDs differ from the authoritative manifest"
        )

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = _load_predictor(checkpoint, device=device)
    bridge = probe["mechanism"]["bridge_schedule"]
    schedule = BBDMBridgeSchedule(
        num_train_timesteps=int(bridge["num_train_timesteps"]),
        m_schedule=str(bridge["m_schedule"]),
        sigma_scale=float(bridge["sigma_scale"]),
    ).to(device)
    frozen_batch_size = int(
        probe["mechanism"]["evidence_generation"]["batch_size"]
    )
    sampled = _sample_band_rows(
        model=model,
        datasets={"external_confirmation": dataset},
        schedule=schedule,
        timesteps=[int(value) for value in probe["timesteps"]],
        batch_size=frozen_batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    slice_by_sample = {
        entry.sample_id: int(entry.slice_id) for entry in dataset.entries
    }
    for row in sampled:
        row["slice_id"] = slice_by_sample[str(row["sample_id"])]
    sample_frame = normalize_ids(pd.DataFrame(sampled))
    context = add_context_features(sample_frame)
    calibrated = apply_hierarchical_calibration(
        context,
        probe["mechanism"]["hierarchical_calibration"][
            "population_stats"
        ],
        shrinkage_k=float(
            probe["mechanism"]["hierarchical_calibration"][
                "patient_shrinkage_k"
            ]
        ),
    )
    predictions = apply_comparators(
        calibrated,
        models=probe["mechanism"]["models"],
        original_standardization=probe["mechanism"][
            "original_standardization"
        ],
        confidence_threshold=float(
            probe["mechanism"]["confidence"]["threshold"]
        ),
    )
    errors = patient_errors(predictions)
    attributes = _external_attributes(dataset)

    comparisons = {
        "h3_vs_no_route": effect_statistics(
            errors,
            partition="external_confirmation",
            left="h3_fixed_schedule",
            right="no_route",
            seed=ANALYSIS_SEED,
        ),
        "original_vs_h3": effect_statistics(
            errors,
            partition="external_confirmation",
            left="original_evidence",
            right="h3_fixed_schedule",
            seed=ANALYSIS_SEED + 100,
        ),
        "uncertainty_vs_h3": effect_statistics(
            errors,
            partition="external_confirmation",
            left="uncertainty_aware",
            right="h3_fixed_schedule",
            seed=ANALYSIS_SEED + 200,
        ),
        "uncertainty_vs_original": effect_statistics(
            errors,
            partition="external_confirmation",
            left="uncertainty_aware",
            right="original_evidence",
            seed=ANALYSIS_SEED + 300,
        ),
    }
    primary_pass = all(
        float(comparisons[name]["ci95_low"]) > 0.0
        and float(comparisons[name]["sign_flip_p"]) < 0.05
        for name in ("uncertainty_vs_h3", "uncertainty_vs_original")
    )
    active_fraction = float(errors["active_fraction"].mean())
    active_pass = active_fraction >= float(
        gate["minimum_router_active_fraction"]
    )
    subgroup_results, subgroups_pass = _subgroup_statistics(
        errors=errors,
        attributes=attributes,
        subgroup_spec=probe["robustness_subgroups"],
        minimum_patients=int(gate["minimum_patients_per_subgroup"]),
    )
    passed = bool(primary_pass and active_pass and subgroups_pass)

    predictions.to_csv(
        output / "external_sample_predictions.csv",
        index=False,
    )
    errors.to_csv(output / "external_patient_errors.csv", index=False)
    assigned = assign_subgroups(
        attributes,
        probe["robustness_subgroups"],
    )
    assigned.to_csv(
        output / "external_patient_attributes.csv",
        index=False,
    )
    write_json(output / "preflight_checks.json", checks)
    decision = {
        "schema_version": SCHEMA_VERSION,
        "stage": "01_H4_v2_external_confirmation",
        "decision": "PASS" if passed else "FAIL",
        "confirmation_cohort_id": cohort_id,
        "probe_sha256": probe["probe_sha256"],
        "external_dataset_contract_sha256": external_contract_sha,
        "external_cache_metadata_sha256": external_lineage[
            "cache_metadata_sha256"
        ],
        "external_patients": len(selected_patients),
        "external_samples": len(selected_manifest),
        "preflight_checks": checks,
        "H4_v2": {
            "status": "PASS" if passed else "FAIL",
            "router_active_fraction_patient_mean": active_fraction,
            "minimum_router_active_fraction": gate[
                "minimum_router_active_fraction"
            ],
            "active_coverage_status": (
                "PASS" if active_pass else "FAIL"
            ),
            "comparisons": comparisons,
            "primary_superiority_status": (
                "PASS" if primary_pass else "FAIL"
            ),
            "robustness_subgroups": subgroup_results,
            "robustness_status": (
                "PASS" if subgroups_pass else "FAIL"
            ),
            "wording": probe["wording_boundary"],
        },
        "guardrails": {
            "patient_unit": "patient",
            "thresholds_and_models_frozen_before_confirmation": True,
            "confirmation_drives_training_or_course": False,
            "patient_bootstrap_95ci": True,
            "development_patient_overlap": 0,
            "causal_or_mutual_information_claimed": False,
            "dual_lineage_verified": {
                "development_checkpoint": True,
                "external_cache": True,
            },
        },
        "model_mechanism_claims_allowed": passed,
        "next_stage_allowed": passed,
        "next_stage": "CT_support_head" if passed else None,
        "stop_rule": (
            "If FAIL, stop evidence routing and keep CT head, curriculum, "
            "H5, and H6 blocked; do not change the frozen gate to preserve "
            "the mechanism story."
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
            "device": str(device),
            "frozen_batch_size": frozen_batch_size,
            "model_order": list(MODEL_ORDER),
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
            "stage": "01_H4_v2_external_confirmation",
            "decision": "FAIL",
            "failure_phase": "EXCEPTION_FAIL_CLOSED",
            "confirmation_cohort_id": str(
                args.confirmation_cohort_id
            ),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "H4_v2": {"status": "FAIL"},
            "next_stage_allowed": False,
            "model_mechanism_claims_allowed": False,
            "stop_rule": (
                "The external confirmation did not complete cleanly; "
                "CT head, curriculum, H5, and H6 remain blocked."
            ),
        }
        write_json(output / "decision.json", decision)
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
