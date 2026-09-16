import json
from pathlib import Path

import pytest

import scripts.audit_h3_fixed_schedule_inference as audit_module
from scripts.audit_h3_fixed_schedule_inference import (
    DEFAULT_BUNDLE,
    DEFAULT_CONFIG,
    DEFAULT_CONTRACT,
    DEFAULT_MODEL_SOURCE,
    PIPELINE_ID,
    build_decision,
    main,
)
from src.mechanism_validation.common import canonical_json_sha256, file_sha256


SEALED_DECISION_PATH = (
    audit_module.DEFAULT_OUTPUT / "decision.json"
)
SEALED_DECISION_FILE_SHA256 = (
    "756acd11a6ffde9aedba6885ac302552d19d511c7afb151e087e88307d283564"
)

# The sealed decision and the frozen calibration bundle/contract are research
# outputs that live on full local/cloud checkouts, not in git.  Skip (not
# fail) when absent so a code-only CI run stays green; machines that have the
# artifacts still verify every SHA-256 pin above.
requires_sealed_decision = pytest.mark.skipif(
    not SEALED_DECISION_PATH.is_file(),
    reason=f"sealed decision artifact not in this checkout: {SEALED_DECISION_PATH}",
)
requires_frozen_bundle = pytest.mark.skipif(
    not (DEFAULT_BUNDLE.is_file() and DEFAULT_CONTRACT.is_file()),
    reason=(
        "frozen calibration bundle/contract not in this checkout: "
        f"{DEFAULT_BUNDLE} / {DEFAULT_CONTRACT}"
    ),
)


def _sealed_decision():
    assert file_sha256(SEALED_DECISION_PATH) == SEALED_DECISION_FILE_SHA256
    decision = json.loads(
        SEALED_DECISION_PATH.read_text(encoding="utf-8")
    )
    body = dict(decision)
    declared = body.pop("decision_sha256")
    assert declared == canonical_json_sha256(body)
    return decision


def _mock_sealed_runtime_integrity(monkeypatch):
    sealed = _sealed_decision()

    def frozen_runtime_file(
        path,
        *,
        expected_file_sha256,
        parse_error,
    ):
        return {
            "decision": "PASS",
            "path": str(path),
            "checks": {
                "file_exists": True,
                "file_sha256_matches_spec": True,
                "source_parses": True,
            },
            "parse_error": None,
            "hash_error": None,
            "observed_file_sha256": expected_file_sha256,
            "expected_file_sha256": expected_file_sha256,
        }

    monkeypatch.setattr(
        audit_module,
        "inspect_runtime_file",
        frozen_runtime_file,
    )
    monkeypatch.setattr(
        audit_module,
        "inspect_production_code_state",
        lambda **kwargs: sealed["gates"]["production_code_state"],
    )


@requires_sealed_decision
def test_formal_gate_splits_grid_pass_from_two_independent_failures():
    decision = _sealed_decision()

    assert decision["pipeline_id"] == PIPELINE_ID
    assert decision["decision"] == "FAIL"
    assert decision["scientific_gate_evaluated"] is True
    assert decision["gates"]["frozen_input_integrity"]["decision"] == "PASS"
    assert (
        decision["gates"]["frozen_grid_lookup"]["decision"]
        == "PASS"
    )
    assert decision["frozen_grid_lookup_available"] is True
    assert "schedule_runtime_available" not in decision
    timestep_gate = decision["gates"]["production_timestep_consumption"]
    assert timestep_gate["decision"] == "FAIL"
    assert timestep_gate["definition_status"] == "UNDEFINED"
    assert timestep_gate["consumption_defined"] is False
    assert (
        decision["gates"]["mapping_identifiability"]["decision"] == "FAIL"
    )
    assert decision["mapping_identifiable"] is False
    assert (
        decision["gates"]["mapping_identifiability"][
            "undefined_mapping_proven"
        ]
        is True
    )
    assert decision["production_timestep_consumption_defined"] is False


@requires_sealed_decision
def test_nineteen_of_twenty_production_eval_steps_are_off_grid():
    decision = _sealed_decision()
    gate = decision["gates"]["production_timestep_consumption"]

    assert gate["eval_sampling_steps"] == 20
    assert gate["frozen_grid_overlap"] == [0]
    assert gate["off_grid_count"] == 19
    assert len(gate["off_grid_timesteps"]) == 19
    assert set(gate["off_grid_timesteps"]).isdisjoint(
        {0, 50, 100, 200, 350, 500, 650, 800, 900, 950}
    )


@requires_sealed_decision
def test_h3_source_and_h4_nested_partition_lineage_are_distinct():
    partition_gate = _sealed_decision()["gates"]["frozen_input_integrity"][
        "frozen_partitions"
    ]

    assert partition_gate["decision"] == "PASS"
    assert (
        partition_gate["h3_schedule_source"][
            "expected_fingerprint_sha256"
        ]
        == "744327eb274d9ff230e57e9a737f30bbb63fdea05ea2abec12413a0469b5c865"
    )
    assert (
        partition_gate["h4_v2_nested_lineage"][
            "expected_fingerprint_sha256"
        ]
        == "0146d21dc3b242e65977fda52f32cf3c6efe79b8fd11011b392068b69a0ba9ad"
    )


@requires_sealed_decision
@pytest.mark.parametrize(
    ("inspector_name", "proof_field"),
    [
        (
            "inspect_production_timestep_consumption",
            "undefined_consumption_proven",
        ),
        ("inspect_mapping_identifiability", "undefined_mapping_proven"),
    ],
)
def test_unproven_fail_is_not_fully_evaluated_or_exit_zero(
    monkeypatch,
    tmp_path,
    inspector_name,
    proof_field,
):
    _mock_sealed_runtime_integrity(monkeypatch)
    original = getattr(audit_module, inspector_name)

    def unproven(*args, **kwargs):
        result = original(*args, **kwargs)
        result[proof_field] = False
        return result

    monkeypatch.setattr(audit_module, inspector_name, unproven)
    decision = audit_module.build_decision()

    assert decision["decision"] == "FAIL"
    assert decision["scientific_proofs_complete"] is False
    assert decision["scientific_gate_evaluated"] is False
    assert decision["failure_phase"] == "SCIENTIFIC_PROOF_INCOMPLETE_FAIL_CLOSED"
    assert (
        audit_module.main(
            [
                "--output-dir",
                str(tmp_path / inspector_name),
                "--allow-scientific-fail-exit-zero",
            ]
        )
        == 2
    )


def test_fresh_audit_fails_closed_after_runtime_source_advance():
    decision = build_decision()
    runtime_gate = decision["gates"]["frozen_input_integrity"][
        "runtime_sources"
    ]

    assert decision["decision"] == "FAIL"
    assert decision["failure_phase"] == "FROZEN_INPUT_INTEGRITY_FAIL_CLOSED"
    assert decision["scientific_gate_evaluated"] is False
    assert runtime_gate["decision"] == "FAIL"
    assert (
        runtime_gate["slmf_bbdm"]["expected_file_sha256"]
        == audit_module.SPEC_MODEL_SOURCE_SHA256
    )
    assert (
        runtime_gate["spectral_router"]["expected_file_sha256"]
        == audit_module.SPEC_ROUTER_SOURCE_SHA256
    )
    assert (
        runtime_gate["slmf_bbdm"]["observed_file_sha256"]
        != audit_module.SPEC_MODEL_SOURCE_SHA256
    )
    assert (
        runtime_gate["spectral_router"]["observed_file_sha256"]
        != audit_module.SPEC_ROUTER_SOURCE_SHA256
    )
    assert (
        decision["gates"]["production_timestep_consumption"]["decision"]
        == "NOT_EVALUATED"
    )


def test_missing_bundle_fails_closed(tmp_path):
    decision = build_decision(bundle_path=tmp_path / "missing_bundle.json")

    assert decision["decision"] == "FAIL"
    assert decision["failure_phase"] == "FROZEN_INPUT_INTEGRITY_FAIL_CLOSED"
    assert decision["scientific_gate_evaluated"] is False
    assert (
        decision["gates"]["frozen_input_integrity"]["decision"] == "FAIL"
    )
    assert (
        decision["gates"]["frozen_grid_lookup"]["decision"]
        == "NOT_EVALUATED"
    )
    assert decision["frozen_grid_lookup_available"] is False
    assert decision["next_stage_allowed"] is False
    assert decision["model_evaluation_allowed"] is False
    assert decision["production_activation_allowed"] is False


@requires_frozen_bundle
def test_tampered_bundle_and_contract_fail_closed(tmp_path):
    bundle = json.loads(DEFAULT_BUNDLE.read_text(encoding="utf-8"))
    bundle["models"]["h3_fixed_schedule"]["schedule"]["l1_hh|50"] = 0.25
    tampered_bundle = tmp_path / "production_calibration_bundle.json"
    tampered_bundle.write_text(json.dumps(bundle), encoding="utf-8")

    contract = json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    contract["fixed_fallback_behavior"][
        "mechanism_validation_level"
    ] = "invented route mapping"
    tampered_contract = tmp_path / "resolved_integration_contract.json"
    tampered_contract.write_text(json.dumps(contract), encoding="utf-8")

    bundle_decision = build_decision(bundle_path=tampered_bundle)
    contract_decision = build_decision(contract_path=tampered_contract)

    for decision in (bundle_decision, contract_decision):
        assert decision["decision"] == "FAIL"
        assert (
            decision["failure_phase"]
            == "FROZEN_INPUT_INTEGRITY_FAIL_CLOSED"
        )
        assert decision["scientific_gate_evaluated"] is False
        assert decision["next_stage_allowed"] is False
        assert decision["model_evaluation_allowed"] is False
        assert decision["production_activation_allowed"] is False


@requires_frozen_bundle
def test_tampered_runtime_source_fails_closed_but_mapping_stays_independent(
    tmp_path,
):
    tampered_source = tmp_path / "slmf_bbdm.py"
    tampered_source.write_text(
        DEFAULT_MODEL_SOURCE.read_text(encoding="utf-8") + "\n# tampered\n",
        encoding="utf-8",
    )
    decision = build_decision(model_source_path=tampered_source)

    assert decision["decision"] == "FAIL"
    assert decision["failure_phase"] == "FROZEN_INPUT_INTEGRITY_FAIL_CLOSED"
    assert decision["scientific_gate_evaluated"] is False
    assert (
        decision["gates"]["production_timestep_consumption"]["decision"]
        == "NOT_EVALUATED"
    )
    assert decision["gates"]["mapping_identifiability"]["decision"] == "FAIL"
    assert decision["next_stage_allowed"] is False


@requires_sealed_decision
def test_hashes_and_decision_fields_are_self_consistent():
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    declared_config_hash = config.pop("config_sha256")
    decision = _sealed_decision()
    declared_decision_hash = decision.pop("decision_sha256")

    assert declared_config_hash == canonical_json_sha256(config)
    assert declared_decision_hash == canonical_json_sha256(decision)
    assert decision["decision"] == "FAIL"
    assert decision["h3_fixed_schedule_inference_admissible"] is False
    assert decision["implementation_performed"] is False
    assert decision["conservative_substitute_implemented"] is False
    assert decision["existing_model_modified"] is False


@requires_sealed_decision
def test_all_downstream_actions_remain_disabled():
    decision = _sealed_decision()

    assert decision["next_stage_allowed"] is False
    assert decision["model_evaluation_allowed"] is False
    assert decision["production_activation_allowed"] is False
    assert set(decision["blocked_actions"]) == {
        "off_grid_timestep_resolver_without_frozen_contract",
        "recoverability_to_native_shallow_null_mapping_implementation",
        "model_evaluation",
        "production_activation",
    }


def test_cli_does_not_reclassify_runtime_integrity_failure(tmp_path):
    output_dir = tmp_path / "formal"
    assert main(["--output-dir", str(output_dir)]) == 2
    first = json.loads(
        (output_dir / "decision.json").read_text(encoding="utf-8")
    )
    assert first["decision"] == "FAIL"

    assert (
        main(
            [
                "--output-dir",
                str(output_dir),
                "--allow-scientific-fail-exit-zero",
            ]
        )
        == 2
    )
    refreshed = json.loads(
        (output_dir / "decision.json").read_text(encoding="utf-8")
    )
    assert refreshed["decision"] == "FAIL"
    assert refreshed["scientific_gate_evaluated"] is False
    assert (
        refreshed["failure_phase"]
        == "FROZEN_INPUT_INTEGRITY_FAIL_CLOSED"
    )
    assert refreshed["decision_sha256"] != first["decision_sha256"]

    missing_dir = tmp_path / "missing"
    assert (
        main(
            [
                "--bundle",
                str(tmp_path / "absent.json"),
                "--output-dir",
                str(missing_dir),
                "--allow-scientific-fail-exit-zero",
            ]
        )
        == 2
    )
