"""Script tests for scripts/verify_rc_brd_math.py (FR-6.4; [计划] §9 M1).

Covers: the synthetic config passes every M1 check (function level and via
main(argv) -> exit 0 with the report written to the pinned results path);
a stub schedule with a forced non-monotone clock fails fail-closed (exit 1);
float64 dtype also passes; report metrics honor the pinned thresholds.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.rc_brd.schedule import BandwiseBridgeSchedule  # noqa: E402


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module  # dataclasses resolves cls.__module__ eagerly
    spec.loader.exec_module(module)
    return module


VERIFY = _load_script("verify_rc_brd_math")

EXPECTED_CHECKS = {
    "haar_roundtrip",
    "kappa0_m_identity",
    "kappa0_q_marginal_scalar_equivalence",
    "kappa0_step_scalar_equivalence",
    "clock_monotone_endpoints",
    "posterior_variance_nonnegative",
    "fixed_seed_reproducibility",
    "no_nan_inf",
    "contract_permutation_scoping",
}


@pytest.fixture()
def t_dir() -> Path:
    """Repo-anchored scratch dir (pytest tmp_path is unusable in the sandbox)."""
    run_dir = REPO_ROOT / ".t_dir" / "rc_brd_verify" / uuid.uuid4().hex[:12]
    run_dir.mkdir(parents=True)
    yield run_dir
    shutil.rmtree(run_dir, ignore_errors=True)


class _BrokenClockSchedule(BandwiseBridgeSchedule):
    """Stub forcing a flat (non-monotone) clock segment ([计划] §9 fail case)."""

    def m_sequence(self, group):
        m = super().m_sequence(group).clone()
        m[5] = m[4]
        return m


def test_all_checks_pass_function_level():
    cfg = VERIFY.VerifyConfig(num_samples=8, seed=0, dtype="float32")
    report = VERIFY.run_verification(cfg)
    assert set(report["checks"]) == EXPECTED_CHECKS
    assert report["all_passed"] is True
    assert report["exit_code"] == 0
    assert report["checks"]["kappa0_m_identity"]["metric"] < 1e-12
    assert report["checks"]["haar_roundtrip"]["metric"] < 1e-5
    assert report["checks"]["posterior_variance_nonnegative"]["metric"] >= 0.0


def test_float64_dtype_also_passes():
    report = VERIFY.run_verification(VERIFY.VerifyConfig(num_samples=4, dtype="float64"))
    assert report["all_passed"] is True


def test_main_writes_report_and_exits_zero():
    rc = VERIFY.main(["--num-samples", "4", "--seed", "0", "--dtype", "float32"])
    assert rc == 0
    payload = json.loads(VERIFY.REPORT_PATH.read_text(encoding="utf-8"))
    assert payload["all_passed"] is True
    assert set(payload["checks"]) == EXPECTED_CHECKS


def test_broken_clock_stub_fails_closed():
    cfg = VERIFY.VerifyConfig(num_samples=4)
    report = VERIFY.run_verification(
        cfg, schedule_factory=lambda config, contract: _BrokenClockSchedule(config, contract))
    assert report["all_passed"] is False
    assert report["exit_code"] == 1
    assert report["checks"]["clock_monotone_endpoints"]["passed"] is False
    assert report["checks"]["kappa0_m_identity"]["passed"] is False


def test_permutation_scoping_moves_only_expected_dimensions():
    entry = VERIFY.run_verification(
        VERIFY.VerifyConfig(num_samples=4))["checks"]["contract_permutation_scoping"]
    assert entry["passed"] is True
    assert entry["metric"] < 1e-9
