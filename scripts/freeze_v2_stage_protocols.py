"""Freeze the V2 main-pipeline stage protocols (Stage V2-06A/B, 07, 08, 09).

Each downstream gate is PRE-REGISTERED here before any outer evaluation: the
comparators, metrics, threshold source, stop-rule, and the upstream it chains
from are locked.  Every decision is written as ``DEFERRED_NOT_RUN`` because the
patient-level evaluation requires the cloud Win cache + GPU and has not been
executed on this working copy.  No stage may be relabelled PASS without a real
patient-level run.

Crucially, the V2 chain hangs off the H4-v2 PASS, NOT the H4-v1 FAIL: the old
scripts/validate_ct_support_head.py / validate_mechanism_curriculum.py chains
that required the H4-v1 decision are NOT reused.

Run from the repo root:
    python scripts/freeze_v2_stage_protocols.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mechanism_validation.common import canonical_json_sha256, load_json


REPO_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = REPO_ROOT / "results" / "mechanism_validation_v2"
H4_V2_PIPELINE_DECISION = (
    V2_ROOT
    / "02_h4_v2_internal_exploratory_nested_cv"
    / "99_pipeline"
    / "decision.json"
)
DIFFICULT_PATIENTS = ("044", "153")
EXTRA_REPORT_PATIENTS = ("002", "022", "080")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(stage_dir: str, payload: dict) -> None:
    out = V2_ROOT / stage_dir
    out.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body.pop("protocol_sha256", None)
    payload["protocol_sha256"] = canonical_json_sha256(body)
    (out / "decision.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "execution_metadata.json").write_text(
        json.dumps(
            {
                "stage": payload["stage"],
                "pipeline_id": payload["pipeline_id"],
                "decision": payload["decision"],
                "run_executed": False,
                "reason": "patient-level evaluation is cloud-only (Win + GPU + sealed cache); no local run performed",
                "created_at_utc": _utc_now(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _h4_v2_ref() -> dict:
    plan = load_json(
        V2_ROOT
        / "02_h4_v2_internal_exploratory_nested_cv"
        / "00_frozen_plan"
        / "frozen_plan.json"
    )
    pipeline = load_json(H4_V2_PIPELINE_DECISION)
    return {
        "upstream_pipeline_id": plan["pipeline_id"],
        "upstream_decision": pipeline["decision"],
        "upstream_h4_v1_preserved": pipeline["H4_v1_preserved"],
        "chains_through_h4_v1_FAIL": False,
        "plan_sha256": plan["plan_sha256"],
        "partition_fingerprint_sha256": plan["partition"]["fingerprint_sha256"],
    }


COMMON_GUARDRAILS = {
    "patient_unit": "patient",
    "slice_level_random_split_forbidden": True,
    "thresholds_margin_curriculum_from_outer_fold_inner_mechanism_train_or_calibration_only": True,
    "outer_patient_not_in_corresponding_fit_or_threshold": True,
    "patient_level_bootstrap_95ci": True,
    "no_causal_or_mutual_information_claim": True,
    "mask_never_a_router_input": True,
    "validation_metric_does_not_drive_training_curriculum": True,
    "checkpoint_lineage_required": True,
    "exploratory_only_conclusion": True,
}


def stage_06a_ct_support() -> dict:
    return {
        "schema_version": 2,
        "stage": "06A_v2_ct_support_gate",
        "pipeline_id": "V2_CT_SUPPORT_GATE_V1",
        "decision": "DEFERRED_NOT_RUN",
        "upstream": _h4_v2_ref(),
        "module_class": "OPTIONAL (must not block the base uncertainty-aware v2)",
        "ct_role": {
            "allowed": "bounded spatial support / reliability only",
            "forbidden": "predict or substitute PET band amplitude; mask input",
        },
        "comparators": [
            "v2 uncertainty-aware WITHOUT CT support",
            "v2 uncertainty-aware WITH fixed analytic CT support",
        ],
        "threshold_source": "inner mechanism_train / calibration of the corresponding outer fold",
        "stop_rule": {
            "if_ct_support_does_not_improve_localization_or_adds_error": "keep CT support OFF",
            "must_not_block_base_v2": True,
            "must_not_reuse_h4_v1_decision_chain": True,
        },
        "cloud_command": "pixi run python scripts/run_v2_main_pipeline.py --stage ct_support",
        "must_not_pass_without_real_run": True,
        "guardrails": COMMON_GUARDRAILS,
    }


def stage_06b_curriculum() -> dict:
    return {
        "schema_version": 2,
        "stage": "06B_v2_curriculum_gate",
        "pipeline_id": "V2_CURRICULUM_GATE_V1",
        "decision": "DEFERRED_NOT_RUN",
        "upstream": _h4_v2_ref(),
        "module_class": "OPTIONAL (must not block the base uncertainty-aware v2)",
        "frozen_recoverability_order": [
            "intensity",
            "coarse",
            "shape",
            "local-frequency",
        ],
        "order_source": "H3 frozen recoverability only",
        "comparators": [
            "fixed training schedule",
            "H3-derived curriculum (frozen order, no outer/validation tuning)",
        ],
        "stop_rule": {
            "if_no_internal_support": "use the fixed training schedule",
            "curriculum_fixed_after_lock": True,
            "must_not_block_base_v2": True,
        },
        "cloud_command": "pixi run python scripts/run_v2_main_pipeline.py --stage curriculum",
        "must_not_pass_without_real_run": True,
        "guardrails": COMMON_GUARDRAILS,
    }


def stage_07_artifact_safety() -> dict:
    return {
        "schema_version": 2,
        "stage": "07_v2_artifact_safety",
        "pipeline_id": "V2_ARTIFACT_SAFETY_V1",
        "decision": "DEFERRED_NOT_RUN",
        "upstream": _h4_v2_ref(),
        "module_class": "eSafe implemented ONLY if the pre-registered artifact score passes",
        "pre_registered_artifact_metrics": [
            "false_hotspot_density",
            "stripe_or_directional_excess",
            "isolated_hotspot_rate",
            "lesion_boundary_gradient_error",
            "whole_image_mae",
        ],
        "threshold_source": "calibration-derived thresholds + non-inferiority margin",
        "artifact_score_must": [
            "identify stripes, false hotspots, isolated hotspots",
            "not lower small-lesion recall",
            "not manufacture a mask boundary",
        ],
        "separation": "eSafe must be independent of CT support and recoverability evidence",
        "ablation_contract": {
            "no_artifact_safety_must_truly_disable_eSafe": True,
            "no_artifact_safety_is_not_equivalent_to_null_route_removal": True,
        },
        "stop_rule": {
            "if_artifact_score_fails": "keep H3 fallback / null safety; do NOT implement eSafe",
            "implement_eSafe_only_on_pass": True,
        },
        "cloud_command": "pixi run python scripts/run_v2_main_pipeline.py --stage artifact_safety",
        "must_not_pass_without_real_run": True,
        "guardrails": COMMON_GUARDRAILS,
    }


def stage_08_h5_v2() -> dict:
    return {
        "schema_version": 2,
        "stage": "08_v2_h5_router_role_ablations",
        "pipeline_id": "V2_H5_ROUTER_ROLES_V1",
        "decision": "DEFERRED_NOT_RUN",
        "upstream": _h4_v2_ref(),
        "base_comparators": [
            "no route",
            "h3 fixed schedule",
            "original h4 evidence",
            "uncertainty-aware v2",
        ],
        "role_ablations": [
            "uncertainty-aware without CT support",
            "uncertainty-aware without recoverability evidence",
            "uncertainty-aware without artifact safety",
            "confidence shuffle (negative control)",
            "wrong-band evidence (negative control)",
            "timestep permutation (negative control)",
            "forced always-active",
            "forced always-abstain",
        ],
        "report_per_outer_fold": True,
        "patient_level_report": [
            "overall effect + bootstrap 95% CI",
            "fraction of patients improved",
            "few-slices subgroup",
            "small-lesion subgroup",
            "low-contrast subgroup",
            [f"patient {p}" for p in DIFFICULT_PATIENTS + EXTRA_REPORT_PATIENTS],
            "router active fraction",
            "router abstain fraction",
            "role distinguishability: CT support vs recoverability vs artifact safety",
        ],
        "stop_rule": {
            "any_pre_registered_main_role_fails": "stop that role; do not edit metrics and rerun",
            "must_not_pass_without_real_run": True,
        },
        "cloud_command": "pixi run python scripts/run_v2_main_pipeline.py --stage h5_v2",
        "must_not_pass_without_real_run": True,
        "guardrails": COMMON_GUARDRAILS,
    }


def stage_09_h6_v2() -> dict:
    return {
        "schema_version": 2,
        "stage": "09_v2_h6_final_integration",
        "pipeline_id": "V2_H6_FINAL_INTEGRATION_V1",
        "decision": "DEFERRED_NOT_RUN",
        "upstream": _h4_v2_ref(),
        "comparators": [
            "plain residual diffusion",
            "fixed-frequency injection",
            "h3 fixed route",
            "uncertainty-aware v2",
            "uncertainty-aware + optional modules that passed their gates",
        ],
        "patient_level_report": [
            "small-lesion metrics",
            "lesion localization and detection",
            "normalized uptake proxy (no physical SUV)",
            "false hotspots",
            "stripes",
            "boundary error",
            "whole-image MAE",
            "PSNR / SSIM",
        ],
        "forbidden_claims": [
            "no physical SUV -> MUST NOT write 'SUV accuracy'",
            "all slices are lesion slices -> MUST declare the negative-slice false-positive rate cannot be fully demonstrated",
        ],
        "must_satisfy_all": [
            "small-lesion primary metric improves",
            "artifact metric non-inferior",
            "whole-image error non-inferior",
            ">=4 of 5 outer folds directionally consistent",
            "patients 044 and 153 retained",
            "no retuning from outer results",
        ],
        "cloud_command": "pixi run python scripts/run_v2_main_pipeline.py --stage h6_v2",
        "must_not_pass_without_real_run": True,
        "guardrails": COMMON_GUARDRAILS,
    }


def main() -> int:
    stages = {
        "06A_ct_support_gate": stage_06a_ct_support(),
        "06B_curriculum_gate": stage_06b_curriculum(),
        "07_artifact_safety": stage_07_artifact_safety(),
        "08_h5_v2_router_ablations": stage_08_h5_v2(),
        "09_h6_v2_final_integration": stage_09_h6_v2(),
    }
    for stage_dir, payload in stages.items():
        payload["created_at_utc"] = _utc_now()
        _write(stage_dir, payload)
        print(
            f"  {payload['decision']:18s} {stage_dir} "
            f"(protocol_sha256={payload['protocol_sha256'][:12]}...)"
        )
    print(f"Wrote {len(stages)} V2 stage protocol decisions under {V2_ROOT.name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
