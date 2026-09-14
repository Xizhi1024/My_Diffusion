"""Freeze the V2 main-model integration contract (Stage V2-04).

Locks, BEFORE any outer evaluation, what the four comparators are, how the
uncertainty-aware evidence is composed, how context must be obtained without
leakage, and the exact fallback behaviour.  The frozen values are derived from
the already-locked ``src/mechanism_validation/h4_v2.py`` constants and the
frozen H4-v2 nested-CV plan; this script does not invent new science.

The output is a self-hashed JSON contract + a decision.json.  The decision is
FROZEN_BEFORE_EVALUATION with an armed FAIL condition: if a leakage-free
context-aware inference path cannot be built, the integration stops with FAIL
and must not be renamed to a degraded mechanism.

Run from the repo root:
    python scripts/freeze_v2_main_integration_contract.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mechanism_validation.common import canonical_json_sha256, load_json
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
    SUBGROUP_MARGIN_FRACTION,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "results" / "mechanism_validation_v2" / "04_main_integration_freeze"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    plan = load_json(
        REPO_ROOT
        / "results/mechanism_validation_v2/02_h4_v2_internal_exploratory_nested_cv"
        / "00_frozen_plan/frozen_plan.json"
    )

    contract = {
        "schema_version": 2,
        "stage": "04_v2_main_integration_freeze",
        "pipeline_id": "V2_MAIN_INTEGRATION_CONTRACT_V1",
        "status": "FROZEN_BEFORE_EVALUATION",
        "scope": "lock_v2_main_model_integration_before_outer_evaluation",
        "comparators": {
            "order": list(MODEL_ORDER),
            "definitions": {
                "no_route": (
                    "frozen per-band mean recoverability; no time conditioning, "
                    "no evidence"
                ),
                "h3_fixed_schedule": (
                    "frozen per-(band, timestep) recoverability schedule at the "
                    "pre-registered H3 bridge log-SNR grid; time-conditioned, "
                    "evidence-free"
                ),
                "original_evidence": (
                    "single-slice noise-calibrated evidence ridge, fit as an "
                    "additive correction on top of the H3 fixed schedule"
                ),
                "uncertainty_aware": (
                    "confidence-weighted hierarchical (adjacent-slice + "
                    "cross-scale) evidence ridge, additive over H3, abstaining "
                    "element-wise to the H3 fixed schedule below the frozen "
                    "confidence threshold"
                ),
            },
        },
        "v2_evidence_recipe": {
            "context_source": (
                "explicit patient_id + slice_id indexed adjacent slices "
                "(+/- NEIGHBOR_RADIUS) within the same patient/band/timestep, "
                "plus the cross-scale peer band; assembled via a deterministic "
                "context index built once from the manifest"
            ),
            "population_normalization": "per (band, timestep) median + 1.4826*MAD scale",
            "patient_band_shrinkage": {
                "weight_formula": "count / (count + K)",
                "K": PATIENT_SHRINKAGE_K,
            },
            "evidence_confidence": (
                "sqrt(local_stability*count_score * patient_band_time_median); "
                "stability = exp(-context_dispersion / population_scale)"
            ),
            "ridge": {"alpha": RIDGE_ALPHA, "fit_intercept": False},
            "abstention": {
                "confidence_threshold_source": (
                    "development-calibration lower quantile of evidence_confidence"
                ),
                "confidence_quantile": CONFIDENCE_QUANTILE,
                "below_threshold": "element-wise exact fallback to h3_fixed_schedule",
            },
        },
        "frozen_grid": {
            "band_order": list(BANDS),
            "log_snr_mapping": (
                "log10-free bridge log-SNR: log((signal^2)/(sigma^2)) clamped to "
                "[-20, 20], then /20 -> [-1, 1]; feature uses log_snr/20.0"
            ),
            "neighbor_radius": NEIGHBOR_RADIUS,
            "min_active_coverage": MIN_ACTIVE_COVERAGE,
            "min_confirmation_patients": MIN_CONFIRMATION_PATIENTS,
            "min_subgroup_patients": MIN_SUBGROUP_PATIENTS,
            "subgroup_noninferiority_margin_fraction": SUBGROUP_MARGIN_FRACTION,
        },
        "context_acquisition_rule": {
            "must_use": [
                "explicit patient_id",
                "explicit slice_id",
                "a reproducible context index (deterministic, manifest-derived)",
            ],
            "forbidden_as_inference_input": [
                "target PET",
                "recoverability label",
                "lesion mask (mask never enters the router)",
                "random-batch adjacency (a batch ordering must not determine context)",
            ],
            "fail_closed_clause": (
                "If a leakage-free adjacent-slice/multi-scale context path cannot "
                "be implemented, the integration stops with a FAIL decision. It "
                "must NOT be renamed to a single-slice proxy and still called "
                "H4-v2 / uncertainty-aware."
            ),
            "current_status": "NOT_YET_IMPLEMENTED_in_the_production_forward",
        },
        "fixed_fallback_behavior": {
            "router_level": (
                "abstaining (level, band) entries become exactly the configured "
                "fixed-policy route vector (frozen by construction); gradient "
                "flows only through active evidence routes; L3 injection stays "
                "zero; mask never routed"
            ),
            "mechanism_validation_level": "h3_fixed_schedule scalar recoverability",
        },
        "h4_v2_reference": {
            "pipeline_id": plan["pipeline_id"],
            "config_sha256": plan["config_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "partition_fingerprint_sha256": plan["partition"]["fingerprint_sha256"],
            "h4_v1_outcome": plan["freeze_guardrails"]["H4_v1_outcome"],
            "h4_v2_pipeline_decision": "PASS (internal exploratory; existing_155_patient_dataset_only)",
        },
        "created_at_utc": _utc_now(),
    }
    contract_sha = canonical_json_sha256(contract)
    contract["contract_sha256"] = contract_sha

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contract_path = OUTPUT_DIR / "resolved_integration_contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    decision = {
        "schema_version": 2,
        "stage": "04_v2_main_integration_freeze",
        "pipeline_id": "V2_MAIN_INTEGRATION_CONTRACT_V1",
        "decision": "FROZEN_BEFORE_EVALUATION",
        "contract_sha256": contract_sha,
        "h4_v2_plan_sha256": plan["plan_sha256"],
        "h4_v2_partition_fingerprint_sha256": plan["partition"]["fingerprint_sha256"],
        "conclusion": (
            "Integration contract locked before outer evaluation: four "
            "comparators, the uncertainty-aware evidence recipe, the frozen "
            "band/log-SNR grid, and the fail-closed context-acquisition rule. "
            "Activation is blocked until a leakage-free context path is built."
        ),
        "armed_fail_condition": (
            "If the context-aware inference path cannot be constructed without "
            "using target PET / recoverability / mask / random-batch adjacency, "
            "Stage V2-05/V2-08 must emit FAIL and must not relabel a degraded "
            "single-slice proxy as H4-v2."
        ),
        "current_context_path_status": "NOT_YET_IMPLEMENTED",
        "created_at_utc": _utc_now(),
    }
    decision_path = OUTPUT_DIR / "decision.json"
    decision_path.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Froze V2 integration contract: {contract_path.relative_to(REPO_ROOT)}")
    print(f"contract_sha256 = {contract_sha}")
    print(f"Wrote decision: {decision_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
