from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_h3_v2_overnight import (
    TARGETED_TESTS,
    KeepWindowsAwake,
    OvernightRunner,
    PIPELINE_ID,
    RUNS_ROOT,
    build_plan,
    parse_args,
)


def test_overnight_plan_is_isolated_and_conditional() -> None:
    args = parse_args(["--dry-run", "--epochs", "7"])
    run_dir = RUNS_ROOT / "unit-test-plan"
    plan = build_plan(args, run_dir)

    assert plan["pipeline_id"] == PIPELINE_ID
    assert plan["epochs_per_variant"] == 7
    assert Path(plan["run_dir"]) == run_dir
    assert plan["hard_guards"] == {
        "cuda_required": True,
        "cpu_fallback_allowed": False,
        "calibration_fail_stops_before_training": True,
        "all_mutable_training_outputs_below_run_dir": True,
        "mechanism_component": "availability",
        "destination_mechanism_evaluated": False,
        "artifact_safety_mechanism_evaluated": False,
        "h3_driven_curriculum_evaluated": False,
        "full_ternary_router_claim_allowed": False,
        "production_training_started": False,
        "production_activation_allowed": False,
        "h5_h6_started": False,
        "does_not_override_05B": True,
    }
    stages = plan["stages"]
    assert stages.index("04_full_timestep_calibration") < stages.index(
        "IF calibration PASS: 07_train_no_route"
    )
    assert stages.index("IF calibration PASS: 07_train_no_route") < stages.index(
        "IF calibration PASS: 08_train_h3_v2"
    )


def test_skip_full_tests_is_explicit_in_plan() -> None:
    args = parse_args(["--dry-run", "--skip-full-tests"])
    plan = build_plan(args, RUNS_ROOT / "unit-test-skip")
    assert "03_full_tests" not in plan["stages"]


def test_resume_requires_exact_run_directory() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--resume"])
    with pytest.raises(SystemExit):
        parse_args(["--resume", "--run-dir", "x", "--dry-run"])


def test_frozen_v1_accepts_only_visible_cuda_device_zero() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--device-index", "1", "--dry-run"])


def test_scientific_controls_do_not_have_short_or_invalid_values() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--epochs", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--shard-size", "30"])
    with pytest.raises(SystemExit):
        parse_args(["--heartbeat-seconds", "0"])


def test_runner_contract_freezes_cuda_selection_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    runner = OvernightRunner(
        parse_args(["--dry-run"]),
        tmp_path / "contract",
        KeepWindowsAwake(),
    )
    try:
        contract = runner._runner_contract()
        assert contract["cuda_visible_devices"] == "3"
        assert contract["cuda_device_order"] == "PCI_BUS_ID"
    finally:
        runner.logger.close()


def test_resume_resource_preflight_preserves_initial_stage_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = OvernightRunner(
        parse_args(["--resume", "--run-dir", str(tmp_path)]),
        tmp_path / "resume",
        KeepWindowsAwake(),
    )
    initial = runner.artifacts_dir / "preflight" / "gpu_and_disk.json"
    initial.parent.mkdir(parents=True, exist_ok=True)
    initial.write_text('{"sealed":true}\n', encoding="utf-8")
    runner.state = {"resume_resource_preflights": []}
    monkeypatch.setattr(
        runner,
        "_resource_preflight_payload",
        lambda: {"created_at_utc": "2026-07-24T00:00:00+00:00"},
    )
    monkeypatch.setattr(
        runner,
        "_historical_failure_checks",
        lambda: {"05A": {"decision": "FAIL"}, "05B": {"decision": "FAIL"}},
    )
    try:
        resume_artifact = runner._resume_resource_preflight()
        assert initial.read_text(encoding="utf-8") == '{"sealed":true}\n'
        assert resume_artifact.is_file()
        assert resume_artifact != initial
        assert resume_artifact.parent.name == "resume_launches"
        assert len(runner.state["resume_resource_preflights"]) == 1
        assert (
            runner.state["resume_resource_preflights"][0]["path"]
            == str(resume_artifact.resolve())
        )
    finally:
        runner.logger.close()


def test_targeted_tests_exclude_retired_h3_fixed_schedule_suite() -> None:
    names = {path.name for path in TARGETED_TESTS}
    assert "test_h3_fixed_schedule_inference.py" not in names
    assert names == {
        "test_h3_v2_full_timestep_native_null.py",
        "test_h3_v2_overnight_runner.py",
        "test_v2_inference_admissibility.py",
    }
