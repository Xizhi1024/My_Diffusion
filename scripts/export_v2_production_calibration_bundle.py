"""Export the frozen V2 production calibration bundle (Stage V2-04 enabler).

Refits the locked H4-v2 mechanism on the ORIGINAL development partition
(99 mechanism_train + 25 calibration) -- NOT a nested-CV fold -- and seals a
single bundle that downstream code needs in order to (a) set the router's
confidence threshold, (b) compute leakage-free context-aware confidence
(population stats + patient/band shrinkage + ridge), and (c) source the
recoverability schedule that a future recoverability->route mapping consumes.

This does NOT introduce new science: it reuses the exact fit functions from
``src/mechanism_validation/h4_v2.py`` that the frozen H4-v2 nested CV used.
The source sample-evidence CSV is SHA-256-verified against the frozen plan.

The bundle is a NEW frozen artifact (own pipeline_id + self-hash), distinct
from the H4-v2 nested-CV fit, and explicitly NOT a PASS for any gate: it is a
calibration export that the still-unbuilt context-aware inference path will
consume.  Producing it does not activate the selector in production.

Run from the repo root (CPU-only; works on the local working copy and on cloud):
    python scripts/export_v2_production_calibration_bundle.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mechanism_validation.common import (
    canonical_json_sha256,
    file_sha256,
    load_json,
)
from src.mechanism_validation.h4_v2 import (
    BANDS,
    CONFIDENCE_QUANTILE,
    MODEL_ORDER,
    NEIGHBOR_RADIUS,
    PATIENT_SHRINKAGE_K,
    RIDGE_ALPHA,
    add_context_features,
    apply_hierarchical_calibration,
    fit_comparators,
    fit_original_standardization,
    fit_population_stats,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    REPO_ROOT
    / "results"
    / "mechanism_validation_v2"
    / "02_h4_v2_internal_exploratory_nested_cv"
    / "00_frozen_plan"
    / "frozen_plan.json"
)
DEFAULT_SAMPLES = (
    "results/mechanism_validation/03_h4_noise_calibration/sample_band_evidence.csv"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "mechanism_validation_v2"
    / "04_main_integration_freeze"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _partition_fingerprint(rows: pd.DataFrame) -> str:
    """Original-partition (99/25/31) patient->role fingerprint."""
    mapping = sorted(
        (str(pid), str(role))
        for pid, role in zip(rows["patient_id"], rows["partition"])
    )
    return canonical_json_sha256(mapping)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default=DEFAULT_SAMPLES,
                        help="H4-v1 sample_band_evidence.csv (original partition)")
    parser.add_argument("--plan", default=str(PLAN_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    samples_path = (REPO_ROOT / args.samples).resolve()
    plan_path = (REPO_ROOT / args.plan).resolve()
    plan = load_json(plan_path)

    # SHA-verify the source evidence against what the H4-v2 freeze recorded.
    samples_sha = file_sha256(samples_path)
    expected_sha = plan["source_artifacts"]["h4_v1_sample_evidence_sha256"]
    if samples_sha != expected_sha:
        raise ValueError(
            f"sample_band_evidence.csv SHA mismatch:\n"
            f"  observed={samples_sha}\n  frozen={expected_sha}\n"
            f"Refusing to export a bundle from a changed source."
        )

    rows = pd.read_csv(samples_path, dtype={"patient_id": str})
    if set(rows["band"].astype(str)) != set(BANDS):
        raise ValueError("sample band set differs from the frozen H4-v2 band order")
    patient_counts = rows.groupby("partition")["patient_id"].nunique().to_dict()
    if patient_counts != {"mechanism_train": 99, "calibration": 25, "validation": 31}:
        raise ValueError(f"unexpected original partition counts: {patient_counts}")

    # Exact H4-v2 recipe, fit on the ORIGINAL development partition.
    context = add_context_features(rows)
    population_stats = fit_population_stats(context, partition="mechanism_train")
    calibrated = apply_hierarchical_calibration(context, population_stats)
    original_standardization = fit_original_standardization(
        calibrated, partition="mechanism_train"
    )
    models = fit_comparators(
        calibrated,
        original_standardization,
        partition="mechanism_train",
        alpha=RIDGE_ALPHA,
    )
    calibration_confidence = calibrated.loc[
        calibrated["partition"] == "calibration", "evidence_confidence"
    ].to_numpy(dtype=np.float64)
    if not calibration_confidence.size:
        raise ValueError("No calibration-partition confidence for threshold")
    confidence_threshold = float(
        np.quantile(calibration_confidence, CONFIDENCE_QUANTILE)
    )

    original_fp = _partition_fingerprint(rows)
    bundle = {
        "schema_version": 1,
        "stage": "04_v2_production_calibration_bundle",
        "pipeline_id": "V2_PRODUCTION_CALIBRATION_BUNDLE_V1",
        "status": "FROZEN_CALIBRATION_EXPORT",
        "scope": (
            "frozen H4-v2 mechanism refit on the original 99/25/31 development "
            "partition; consumed by the (unbuilt) context-aware inference path "
            "and a future recoverability->route mapping"
        ),
        "fit_recipe": {
            "population_partition": "mechanism_train (99)",
            "standardization_partition": "mechanism_train (99)",
            "comparator_partition": "mechanism_train (99)",
            "threshold_partition": "calibration (25)",
            "neighbor_radius": NEIGHBOR_RADIUS,
            "patient_band_shrinkage_k": PATIENT_SHRINKAGE_K,
            "ridge_alpha": RIDGE_ALPHA,
            "confidence_quantile": CONFIDENCE_QUANTILE,
        },
        "confidence_threshold": confidence_threshold,
        "population_stats": population_stats,
        "original_standardization": original_standardization,
        "models": {name: models[name] for name in MODEL_ORDER},
        "frozen_grid": {
            "band_order": list(BANDS),
            "model_order": list(MODEL_ORDER),
            "log_snr_mapping": (
                "bridge log-SNR clamped [-20,20] then /20 -> [-1,1]; "
                "ridge feature uses log_snr/20.0"
            ),
        },
        "source": {
            "h4_v2_pipeline_id": plan["pipeline_id"],
            "h4_v2_plan_sha256": plan["plan_sha256"],
            "h4_v2_config_sha256": plan["config_sha256"],
            "h4_v2_nested_partition_fingerprint_sha256": plan["partition"][
                "fingerprint_sha256"
            ],
            "original_partition_fingerprint_sha256": original_fp,
            "sample_band_evidence_sha256": samples_sha,
            "h4_v1_outcome": plan["freeze_guardrails"]["H4_v1_outcome"],
        },
        "usage_contract": {
            "selector_confidence_threshold": (
                "router loads bundle.confidence_threshold into "
                "UncertaintyAwareRouteSelector"
            ),
            "context_confidence": (
                "context-aware inference path uses population_stats + "
                "patient_band_shrinkage_k + ridge to produce router_confidence; "
                "this script does NOT build that path"
            ),
            "recoverability_route_mapping": (
                "a separate module must map models.h3_fixed_schedule (band x "
                "timestep recoverability) to a route distribution; this bundle "
                "sources the recoverability but does NOT define the mapping"
            ),
        },
        "explicit_non_claim": (
            "producing this bundle does NOT activate the selector, does NOT "
            "pass H5-v2/H6-v2, and does NOT assert H4-v2 is in production"
        ),
        "created_at_utc": _utc_now(),
    }
    bundle["bundle_sha256"] = canonical_json_sha256(bundle)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "production_calibration_bundle.json"
    bundle_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    decision = {
        "schema_version": 2,
        "stage": "04_v2_production_calibration_bundle_export",
        "pipeline_id": "V2_PRODUCTION_CALIBRATION_BUNDLE_V1",
        "decision": "FROZEN",
        "bundle_sha256": bundle["bundle_sha256"],
        "confidence_threshold": confidence_threshold,
        "source_verified": {
            "sample_band_evidence_sha256": samples_sha,
            "matches_h4_v2_freeze": samples_sha == expected_sha,
            "original_partition_fingerprint_sha256": original_fp,
        },
        "conclusion": (
            "Calibration bundle exported and sealed from the SHA-verified H4-v1 "
            "sample evidence on the original 99/25/31 partition. It is a frozen "
            "input for the (still-unbuilt) context-aware inference path and a "
            "future recoverability->route mapping; it is not a gate PASS."
        ),
        "created_at_utc": _utc_now(),
    }
    (out_dir / "calibration_bundle_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Exported production calibration bundle: {bundle_path.relative_to(REPO_ROOT)}")
    print(f"bundle_sha256      = {bundle['bundle_sha256']}")
    print(f"confidence_threshold = {confidence_threshold:.6f}")
    print(f"source SHA verified  = {samples_sha == expected_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
