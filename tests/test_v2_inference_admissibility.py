import json
from pathlib import Path

import pytest

from scripts.audit_v2_inference_admissibility import (
    PIPELINE_ID,
    ROOT,
    build_decision,
    build_missing_artifact_decision,
    inspect_h4_generator,
)

# The frozen V2 calibration artifacts are research outputs stored under
# results/, which is not part of the git tree.  Skip (not fail) on code-only
# checkouts; full local/cloud checkouts still run these contracts.
_V2_BUNDLE = (
    ROOT
    / "results"
    / "mechanism_validation_v2"
    / "04_main_integration_freeze"
    / "production_calibration_bundle.json"
)
_V2_CONTRACT = (
    ROOT
    / "results"
    / "mechanism_validation_v2"
    / "04_main_integration_freeze"
    / "resolved_integration_contract.json"
)
_V1_H4_DECISION = (
    ROOT / "results" / "mechanism_validation" / "03_h4_noise_calibration" / "decision.json"
)
requires_v2_frozen_artifacts = pytest.mark.skipif(
    not (_V2_BUNDLE.is_file() and _V2_CONTRACT.is_file()),
    reason=(
        "frozen calibration bundle/contract not in this checkout: "
        f"{_V2_BUNDLE} / {_V2_CONTRACT}"
    ),
)
requires_v1_h4_decision = pytest.mark.skipif(
    not _V1_H4_DECISION.is_file(),
    reason=f"sealed H4-v1 decision not in this checkout: {_V1_H4_DECISION}",
)


def test_h4_generator_semantics_expose_forbidden_inputs():
    source = (
        ROOT / "scripts" / "validate_h4_noise_band_calibration.py"
    ).read_text(encoding="utf-8")
    semantics = inspect_h4_generator(source)

    assert semantics["reads_target_pet"] is True
    assert semantics["reads_lesion_mask"] is True
    assert semantics["residual_uses_target_pet"] is True
    assert semantics["uses_masked_band_energy"] is True
    assert semantics["uses_masked_band_alignment"] is True


@requires_v2_frozen_artifacts
def test_frozen_bundle_fails_production_inference_admissibility():
    decision = build_decision(
        bundle_path=(
            ROOT
            / "results"
            / "mechanism_validation_v2"
            / "04_main_integration_freeze"
            / "production_calibration_bundle.json"
        ),
        contract_path=(
            ROOT
            / "results"
            / "mechanism_validation_v2"
            / "04_main_integration_freeze"
            / "resolved_integration_contract.json"
        ),
        generator_path=(
            ROOT / "scripts" / "validate_h4_noise_band_calibration.py"
        ),
    )

    assert decision["pipeline_id"] == PIPELINE_ID
    assert decision["decision"] == "FAIL"
    assert decision["conflict_proven"] is True
    assert decision["production_inference_admissible"] is False
    assert decision["dataset_internal_h4_v2_result_preserved"] is True
    assert decision["h4_v1_outcome_preserved"] == "FAIL"
    assert decision["production_router_activation_allowed"] is False
    assert decision["next_stage_allowed"] is False


@requires_v2_frozen_artifacts
@requires_v1_h4_decision
def test_admissibility_decision_is_new_and_does_not_overwrite_v1(tmp_path):
    v1_path = (
        ROOT
        / "results"
        / "mechanism_validation"
        / "03_h4_noise_calibration"
        / "decision.json"
    )
    before = v1_path.read_bytes()
    decision = build_decision(
        bundle_path=(
            ROOT
            / "results"
            / "mechanism_validation_v2"
            / "04_main_integration_freeze"
            / "production_calibration_bundle.json"
        ),
        contract_path=(
            ROOT
            / "results"
            / "mechanism_validation_v2"
            / "04_main_integration_freeze"
            / "resolved_integration_contract.json"
        ),
        generator_path=(
            ROOT / "scripts" / "validate_h4_noise_band_calibration.py"
        ),
    )
    output = tmp_path / "decision.json"
    output.write_text(json.dumps(decision), encoding="utf-8")

    assert v1_path.read_bytes() == before
    assert json.loads(output.read_text(encoding="utf-8"))["stage"].startswith(
        "05A_"
    )


def test_missing_frozen_artifact_fails_closed_without_scientific_claim(tmp_path):
    missing_bundle = tmp_path / "production_calibration_bundle.json"
    decision = build_missing_artifact_decision(
        [missing_bundle],
        bundle_path=missing_bundle,
        contract_path=tmp_path / "resolved_integration_contract.json",
        generator_path=tmp_path / "validate_h4_noise_band_calibration.py",
    )

    assert decision["decision"] == "FAIL"
    assert decision["failure_phase"] == "MISSING_FROZEN_ARTIFACT_FAIL_CLOSED"
    assert decision["scientific_gate_evaluated"] is False
    assert decision["production_router_activation_allowed"] is False
    assert decision["next_stage_allowed"] is False
    assert str(missing_bundle) in decision["missing_artifacts"]
