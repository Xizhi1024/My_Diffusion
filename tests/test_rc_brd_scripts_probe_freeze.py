"""Script tests: run_recoverability_probe.py (R0) + freeze step (FR-6.2/6.3).

Covers: synthetic data with known recoverable bands -> PASS judged correctly;
CT independent of the residual under the permutation null -> no false
positive (LCB <= 0); insufficient patients -> exit 2; compressed/full grid
shapes; the v1.5 cell-based PASS rule ([计划] v1.5 §3.4: >=2 small-lesion
cells covering >=2 lambda regions; single-cell and single-region FAIL);
probe JSON -> contract JSON round trip readable by
RecoverabilityContract.load; computed vs passed mean SHA agree; kappa grid
without 0 -> exit 1; s_ref <= 0 -> exit 1; fold mismatch -> exit 1.
Function-level main(argv) only (no subprocess); .t_dir scratch fixture.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.rc_brd import BAND_NAMES, RecoverabilityContract, mean_weights_sha256  # noqa: E402


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module  # dataclasses resolves cls.__module__ eagerly
    spec.loader.exec_module(module)
    return module


PROBE = _load_script("run_recoverability_probe")
FREEZE = _load_script("freeze_recoverability_contract")
MEAN_SHA = "a" * 64


@pytest.fixture()
def t_dir() -> Path:
    """Repo-anchored scratch dir (pytest tmp_path is unusable in the sandbox)."""
    run_dir = REPO_ROOT / ".t_dir" / "rc_brd_probe_freeze" / uuid.uuid4().hex[:12]
    run_dir.mkdir(parents=True)
    yield run_dir
    shutil.rmtree(run_dir, ignore_errors=True)


def probe_config(n_patients: int = 60, recoverable=("low", "mid"),
                 n_coef: int = 64, seed: int = 3) -> dict[str, Any]:
    return {
        "probe": {"source": "synthetic", "n_patients": n_patients, "n_coef": n_coef,
                  "n_features": 3, "seed": seed, "signal_rho": 0.85,
                  "recoverable_groups": list(recoverable)},
        "cross_fit": {"n_folds": 5, "ridge_lambda": 0.01},
        "seed": 11,
    }


class _Scratch:
    """Small helper so tests can share the run_probe/freeze plumbing."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.counter = 0

    def run_probe(self, config: dict[str, Any], grid: str = "compressed",
                  n_perm: int = 200, fold: str = "fold_0") -> tuple[int, Path, dict]:
        self.counter += 1
        cfg_path = self.base / f"cfg{self.counter}.json"
        cfg_path.write_text(json.dumps(config), encoding="utf-8")
        out = self.base / f"probe{self.counter}.json"
        rc = PROBE.main(["--config", str(cfg_path), "--fold", fold, "--grid", grid,
                        "--n-perm", str(n_perm), "--out", str(out)])
        result = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
        return rc, out, result


@pytest.fixture()
def scratch(t_dir: Path) -> _Scratch:
    return _Scratch(t_dir)


# ---------------------------------------------------------------------------
# R0 probe (FR-6.2)
# ---------------------------------------------------------------------------

def test_probe_true_recoverable_bands_pass(scratch):
    rc, out, result = scratch.run_probe(probe_config(recoverable=("low", "mid")))
    assert rc == 0
    assert result["pass_rule"]["passed"] is True
    assert result["pass_rule"]["n_passing_cells"] == 6  # 2 groups x 3 lambda
    assert {c.split("|")[0] for c in result["pass_rule"]["passing_cells"]} == {"low", "mid"}
    assert result["pass_rule"]["covered_lambda_regions"] == [0, 1, 2]
    assert result["pass_rule"]["per_cell"]["mid|lam1"]["pass"] is True
    for group in ("low", "mid"):
        for k in range(3):
            cell = result["delta"]["small_lesion"][group][str(k)]
            assert cell["lcb95"] > 0.0 and cell["exceeds_maxt_null"] is True
    assert result["grid"]["groups"] == ["low", "mid", "high"]
    assert result["grid"]["strata"] == ["small_lesion"]
    assert result["nulls"]["patient_permuted_ct"]["max_t_q95"] > 0.0


def test_probe_null_ct_gives_no_false_positive(scratch):
    rc, _out, result = scratch.run_probe(probe_config(recoverable=()))
    assert rc == 0
    assert result["pass_rule"]["passed"] is False
    assert result["pass_rule"]["n_passing_cells"] == 0
    assert result["pass_rule"]["passing_cells"] == []
    for group in result["grid"]["groups"]:
        for k in range(result["grid"]["n_lambda"]):
            cell = result["delta"]["small_lesion"][group][str(k)]
            assert cell["lcb95"] <= 0.0
            assert cell["exceeds_maxt_null"] is False


def test_probe_insufficient_patients_exit_2(scratch):
    rc, out, result = scratch.run_probe(probe_config(n_patients=8))
    assert rc == 2
    assert result["status"] == "insufficient_patients"
    assert result["threshold"] == 12
    rc, _out2, result = scratch.run_probe(probe_config(n_patients=20), grid="full")
    assert rc == 2
    assert result["threshold"] == 24


def test_probe_compressed_grid_shape(scratch):
    rc, _out, result = scratch.run_probe(probe_config(), grid="compressed", n_perm=100)
    assert rc == 0
    assert len(result["grid"]["groups"]) == 3
    assert len(result["grid"]["strata"]) == 1
    assert result["grid"]["n_lambda"] == 3
    cells = [g for s in result["delta"].values() for g in s.values() for _ in g]
    assert len(cells) == 9  # 3 groups x 3 lambda x 1 stratum ([审计] §1.1)


def test_probe_full_grid_shape(scratch):
    config = probe_config(n_patients=30, recoverable=("LL2", "LH2"), n_coef=33)
    rc, _out, result = scratch.run_probe(config, grid="full", n_perm=100)
    assert rc == 0
    assert result["grid"]["groups"] == list(BAND_NAMES)
    assert len(result["grid"]["strata"]) == 3
    cells = [g for s in result["delta"].values() for g in s.values() for _ in g]
    assert len(cells) == 63  # 7 bands x 3 lambda x 3 strata


# ---------------------------------------------------------------------------
# v1.5 PASS rule ([计划] v1.5 §3.4; DESIGN §10 v1.0f; PRD v1.0.1 addendum 4)
# ---------------------------------------------------------------------------

def _rule_input(passing: set[tuple[str, int]]) -> tuple[dict, float]:
    """Nested delta structure (3 groups x 3 lambda) with chosen passing cells.

    Passing cells: lcb95=0.05, t_obs=5.0 (> q95=2.0); failing cells keep
    lcb95<=0 and t_obs below the max-T threshold."""
    stratum = PROBE.CONFIRMATORY_STRATUM
    nested: dict[str, Any] = {stratum: {}}
    for g in ("low", "mid", "high"):
        nested[stratum][g] = {}
        for k in range(3):
            ok = (g, k) in passing
            nested[stratum][g][str(k)] = {
                "delta": 0.1 if ok else -0.001, "r_base": 1.0,
                "r_cond": 0.9 if ok else 1.001, "se": 0.02,
                "t_obs": 5.0 if ok else 0.5, "lcb95": 0.05 if ok else -0.01,
                "exceeds_maxt_null": ok}
    return nested, 2.0


def test_pass_rule_two_adjacent_lambda_same_group():
    nested, q95 = _rule_input({("mid", 1), ("mid", 2)})
    rule = PROBE.evaluate_pass_rule(nested, ("low", "mid", "high"), 3, q95)
    assert rule["passed"] is True
    assert rule["n_passing_cells"] == 2
    assert rule["passing_cells"] == ["mid|lam1", "mid|lam2"]
    assert rule["covered_lambda_regions"] == [1, 2]
    assert rule["per_cell"]["mid|lam1"]["pass"] is True
    assert rule["per_cell"]["low|lam0"]["pass"] is False


def test_pass_rule_two_groups_one_cell_each():
    nested, q95 = _rule_input({("low", 0), ("high", 2)})
    rule = PROBE.evaluate_pass_rule(nested, ("low", "mid", "high"), 3, q95)
    assert rule["passed"] is True
    assert rule["n_passing_cells"] == 2
    assert rule["covered_lambda_regions"] == [0, 2]


def test_pass_rule_single_cell_fails():
    nested, q95 = _rule_input({("mid", 1)})
    rule = PROBE.evaluate_pass_rule(nested, ("low", "mid", "high"), 3, q95)
    assert rule["passed"] is False
    assert rule["n_passing_cells"] == 1
    assert rule["per_cell"]["mid|lam1"]["pass"] is True  # judgment retained


def test_pass_rule_single_lambda_region_fails():
    # 3 passing cells but all inside lambda region 1 -> coverage 1 -> FAIL
    nested, q95 = _rule_input({("low", 1), ("mid", 1), ("high", 1)})
    rule = PROBE.evaluate_pass_rule(nested, ("low", "mid", "high"), 3, q95)
    assert rule["passed"] is False
    assert rule["n_passing_cells"] == 3
    assert rule["covered_lambda_regions"] == [1]


# ---------------------------------------------------------------------------
# Contract freeze (FR-6.3)
# ---------------------------------------------------------------------------

def _freeze(scratch: _Scratch, probe_out: Path, **kw: Any) -> tuple[int, Path]:
    scratch.counter += 1
    out = scratch.base / f"contract{scratch.counter}.json"
    args = ["--probe-result", str(probe_out), "--fold", kw.get("fold", "fold_0"),
            "--out", str(out), "--s-ref", str(kw.get("s_ref", 0.05)),
            "--kappa-grid", kw.get("kappa", "0.0,0.5")]
    if kw.get("mean_checkpoint"):
        args += ["--mean-checkpoint", str(kw["mean_checkpoint"])]
    else:
        args += ["--mean-sha", kw.get("mean_sha", MEAN_SHA)]
    if "sigma_floor" in kw:
        args += ["--sigma-floor", str(kw["sigma_floor"])]
    rc = FREEZE.main(args)
    return rc, out


def test_freeze_roundtrip_loadable_and_c_matches_probe(scratch):
    rc, probe_out, probe = scratch.run_probe(probe_config())
    assert rc == 0
    rc, out = _freeze(scratch, probe_out)
    assert rc == 0
    contract = RecoverabilityContract.load(out, expected_fold="fold_0",
                                           expected_mean_sha=MEAN_SHA)
    contract.validate()
    s_ref = 0.05
    for group, row in contract.c_values.items():
        for k, value in enumerate(row):
            lcb = probe["delta"]["small_lesion"][group][str(k)]["lcb95"]
            assert value == pytest.approx(min(1.0, max(0.0, lcb / s_ref)))
    # psd floors come from the probe data (synthetic noise-variance proxy)
    assert set(contract.psd_floors) == set(probe["psd_floors"])
    assert contract.log_snr_grid == tuple(probe["grid"]["log_snr_centers"])


def test_freeze_mean_checkpoint_sha_matches_direct(scratch, t_dir):
    rc, probe_out, _ = scratch.run_probe(probe_config())
    assert rc == 0
    state = {"w": torch.arange(6, dtype=torch.float32).reshape(2, 3),
             "b": torch.zeros(1, dtype=torch.float32)}
    ckpt = t_dir / "mean.pt"
    torch.save(state, ckpt)
    rc, out = _freeze(scratch, probe_out, mean_checkpoint=ckpt)
    assert rc == 0
    contract = RecoverabilityContract.load(out, expected_mean_sha=mean_weights_sha256(state))
    assert contract.mean_checkpoint_sha256 == mean_weights_sha256(state)


def test_freeze_kappa_grid_missing_zero_exits_1(scratch):
    rc, probe_out, _ = scratch.run_probe(probe_config())
    assert rc == 0
    rc, _ = _freeze(scratch, probe_out, kappa="0.25,0.5")
    assert rc == 1


def test_freeze_nonpositive_s_ref_exits_1(scratch):
    rc, probe_out, _ = scratch.run_probe(probe_config())
    assert rc == 0
    for bad in ("0", "-2.0"):
        rc, _ = _freeze(scratch, probe_out, s_ref=bad)
        assert rc == 1


def test_freeze_fold_mismatch_exits_1(scratch):
    rc, probe_out, _ = scratch.run_probe(probe_config())
    assert rc == 0
    rc, _ = _freeze(scratch, probe_out, fold="fold_9")
    assert rc == 1
