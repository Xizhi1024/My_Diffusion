"""Tests for scripts/run_dual_experiment_queue.py — the sequential dual-worktree queue.

Covers the fail-closed launch/eligibility behaviours:

 13. non-empty experiment output dir is rejected
 14. same-GPU parallel request is rejected
 15. a changing checkpoint refuses to launch
 16. causality does not run after an upstream failure / failed gates
 17. smoke/fake can never produce a formal COMPLETE
     + structural analysis checks (module missing ⇒ fail-closed)
     + the analysis-code verification against the real feature_causality module
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_dual_experiment_queue import (  # noqa: E402
    QueueError,
    checkpoint_stable,
    compute_eligibility,
    emergence_passed,
    main,
    read_emergence_gates,
    verify_analysis_code,
)


# ---------------------------------------------------------------------------
# 13. non-empty experiment output dir is rejected
# ---------------------------------------------------------------------------


def test_nonempty_output_dir_rejected(tmp_path):
    out_root = tmp_path / "cloud_runs"
    out_root.mkdir(parents=True)
    run_dir = out_root / "router" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "stale.json").write_text("x", encoding="utf-8")

    from run_dual_experiment_queue import _require_empty_run_dir

    with pytest.raises(QueueError, match="not empty"):
        _require_empty_run_dir(run_dir)


# ---------------------------------------------------------------------------
# 14. same-GPU parallel request is rejected
# ---------------------------------------------------------------------------


def test_same_gpu_parallel_rejected(tmp_path, monkeypatch):
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("dummy", encoding="utf-8")
    monkeypatch.setattr(
        "run_dual_experiment_queue._cuda_available", lambda: False
    )
    argv = [
        "--checkpoint", str(ckpt),
        "--output-root", str(tmp_path / "cloud_runs"),
        "--router-command", "echo router",
        "--feature-emergence-command", "echo emergence",
        "--feature-causality-command", "echo causality",
        "--parallel",  # both default to the same device
        "--dry-run",
    ]
    rc = main(argv)
    assert rc == 2  # REFUSING


# ---------------------------------------------------------------------------
# 15. a changing checkpoint refuses to launch
# ---------------------------------------------------------------------------


def test_checkpoint_changing_refuses_launch(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("version-1", encoding="utf-8")

    # Interleave a write so consecutive snapshots differ.
    def _write_between():
        ckpt.write_text("version-2", encoding="utf-8")

    stable, _ = checkpoint_stable(ckpt, interval=0.0, samples=2)
    assert stable
    # Now force a mutation mid-check.
    import run_dual_experiment_queue as queue

    orig = queue._snapshot_checkpoint
    calls = {"n": 0}

    def flaky(path):
        calls["n"] += 1
        if calls["n"] == 2:
            ckpt.write_text("version-3", encoding="utf-8")
        return orig(path)

    queue._snapshot_checkpoint = flaky
    try:
        stable2, _ = queue.checkpoint_stable(ckpt, interval=0.0, samples=2)
    finally:
        queue._snapshot_checkpoint = orig
    assert not stable2


def test_last_pt_checkpoint_refused(tmp_path):
    ckpt = tmp_path / "last.pt"
    ckpt.write_text("in-training", encoding="utf-8")
    argv = [
        "--checkpoint", str(ckpt),
        "--output-root", str(tmp_path / "cloud_runs"),
        "--skip-router",
        "--feature-emergence-command", "echo e",
        "--skip-causality",
        "--dry-run",
    ]
    with pytest.raises(QueueError, match="last.pt"):
        main(argv)


# ---------------------------------------------------------------------------
# 16. causality does not run after upstream failure / failed gates
# ---------------------------------------------------------------------------


def test_emergence_gates_fail_closed_when_files_missing(tmp_path):
    assert emergence_passed(read_emergence_gates(tmp_path)) is False


def test_emergence_gates_require_both_true(tmp_path):
    # A1 passes, A2 missing → not passed.
    (tmp_path / "cross_modal_similarity.json").write_text(
        json.dumps({"s1_gate": {"passed": True}}), encoding="utf-8"
    )
    gates = read_emergence_gates(tmp_path)
    assert gates == {"s1_gate": True, "m1_gate": None}
    assert emergence_passed(gates) is False


def test_gates_read_json(tmp_path):
    (tmp_path / "cross_modal_similarity.json").write_text(
        json.dumps({"s1_gate": {"passed": True}}), encoding="utf-8"
    )
    (tmp_path / "lesion_emergence.json").write_text(
        json.dumps({"m1_gate": {"passed": True}}), encoding="utf-8"
    )
    assert emergence_passed(read_emergence_gates(tmp_path)) is True


def test_full_queue_blocks_causality_when_emergence_gates_fail(tmp_path, monkeypatch):
    """Integration: emergence runs but its gates are missing ⇒ causality skipped."""
    router_wt = tmp_path / "router_wt"
    feature_wt = tmp_path / "feature_wt"
    router_wt.mkdir()
    feature_wt.mkdir()
    # Provide the real analysis module so structural checks can pass.
    src_mv = feature_wt / "src" / "mechanism_validation"
    src_mv.mkdir(parents=True)
    real = (
        Path(__file__).resolve().parents[1]
        / "src" / "mechanism_validation" / "feature_causality.py"
    )
    (src_mv / "feature_causality.py").write_text(
        real.read_text(encoding="utf-8"), encoding="utf-8"
    )

    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("dummy", encoding="utf-8")

    # NOTE: do NOT pre-create the emergence output dir — read_emergence_gates
    # then finds no gate files and fails closed (causality is skipped).

    def _fake_run_experiment(spec, *, checkpoint_sha256, formal):
        spec.run_dir.mkdir(parents=True, exist_ok=True)
        import run_dual_experiment_queue as queue

        record = {
            "experiment": spec.name,
            "command": spec.command,
            "worktree_root": str(spec.worktree_root),
            "device": spec.device,
            "returncode": 0,
            "status": "passed",
            "checkpoint_sha256": checkpoint_sha256,
            "formal_eligibility": formal,
            "started_at": "t",
            "finished_at": "t",
            "environment": {},
        }
        queue.write_json(spec.manifest_path, record)
        return record

    monkeypatch.setattr("run_dual_experiment_queue._cuda_available", lambda: False)
    captured = {}

    def _fake_lock_run_experiment(spec, **kwargs):
        captured[spec.name] = True
        return _fake_run_experiment(spec, **kwargs)

    monkeypatch.setattr(
        "run_dual_experiment_queue.run_experiment", _fake_lock_run_experiment
    )

    argv = [
        "--router-worktree", str(router_wt),
        "--feature-worktree", str(feature_wt),
        "--checkpoint", str(ckpt),
        "--output-root", str(tmp_path / "cloud_runs"),
        "--run-id", "run-x",
        "--router-command", "echo router",
        "--feature-emergence-command",
        "echo emergence --split val --checkpoint {checkpoint}",
        "--feature-causality-command", "echo causality",
        "--check-interval", "0.0",
        "--check-samples", "1",
    ]
    rc = main(argv)
    assert rc == 1  # INCOMPLETE (causality skipped)
    assert "router" in captured
    assert "emergence" in captured
    assert "causality" not in captured
    complete = json.loads(
        (tmp_path / "cloud_runs" / "interpretability" / "run-x" / "COMPLETE.json")
        .read_text(encoding="utf-8")
    )
    assert complete["status"] == "INCOMPLETE"


# ---------------------------------------------------------------------------
# 17. smoke/fake cannot produce formal COMPLETE
# ---------------------------------------------------------------------------


def test_smoke_fake_never_formal():
    checks = verify_analysis_code(Path(__file__).resolve().parents[1])
    elig = compute_eligibility(
        cmd=["python", "x.py", "--fake-data", "--max-samples", "4"],
        split="val",
        has_frozen_quartiles=True,
        analysis_checks=checks,
    )
    assert elig["eligibility"] == "exploratory"
    assert "fake-data" in elig["reasons"]
    assert "max-samples smoke" in elig["reasons"]


def test_formal_requires_all_checks():
    checks = dict.fromkeys(
        ["margin_constant_present", "computed_background_gate", "module_exists"],
        True,
    )
    elig = compute_eligibility(
        cmd=["python", "x.py"],
        split="val",
        has_frozen_quartiles=True,
        analysis_checks=checks,
    )
    assert elig["eligibility"] == "formal"


def test_missing_module_is_fail_closed():
    elig = compute_eligibility(
        cmd=["python", "x.py"],
        split="train",
        has_frozen_quartiles=True,
        analysis_checks=None,  # module not inspected
    )
    assert elig["eligibility"] == "exploratory"
    assert any("not inspected" in r for r in elig["reasons"])

def test_no_frozen_quartiles_on_val_is_exploratory():
    elig = compute_eligibility(
        cmd=["python", "x.py"],
        split="val",
        has_frozen_quartiles=False,
        analysis_checks={},
    )
    assert elig["eligibility"] == "exploratory"
    assert any("quartile" in r for r in elig["reasons"])


def test_missing_spatial_controls_block_formal():
    checks = dict.fromkeys(
        ["margin_constant_present", "computed_background_gate", "module_exists"],
        True,
    )
    elig = compute_eligibility(
        cmd=["python", "x.py", "--interventions", "c2_lesion_zero", "baseline"],
        split="val",
        has_frozen_quartiles=True,
        analysis_checks=checks,
    )
    assert elig["eligibility"] == "exploratory"
    assert any("control" in r for r in elig["reasons"])
    # Full matrix (no --interventions) passes the controls check.
    elig2 = compute_eligibility(
        cmd=["python", "x.py"],
        split="val",
        has_frozen_quartiles=True,
        analysis_checks=checks,
    )
    assert elig2["eligibility"] == "formal"


# ---------------------------------------------------------------------------
# Structural analysis-code verification against the real module
# ---------------------------------------------------------------------------


def test_verify_analysis_code_real_module(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    checks = verify_analysis_code(repo)
    assert checks["module_exists"] is True
    assert checks["margin_constant_present"] is True
    assert checks["computed_background_gate"] is True
    assert checks["samearea_control_present"] is True
    assert checks["shifted_control_present"] is True
    assert checks["body_mask_code_path"] is True
    assert checks["tissue_coverage_recorded"] is True


def test_verify_analysis_code_missing_module(tmp_path):
    checks = verify_analysis_code(tmp_path)
    assert checks["module_exists"] is False
    assert all(v is False for k, v in checks.items())
