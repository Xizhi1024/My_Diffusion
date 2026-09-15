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

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.rc_brd import (  # noqa: E402
    BAND_NAMES,
    RecoverabilityContract,
    compute_contract_sha256,
    mean_weights_sha256,
)


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


def _ragged_npz(base: Path, *, zero_patient: bool = False,
                group_varying: bool = False) -> Path:
    """Cached probe matrix with ragged per-patient stratum membership.

    Half the patients hold 20 small-lesion coefficients, the other half 12,
    so every cell is ragged and must subsample to quota=12.  zero_patient
    empties patient 0's confirmatory coefficients entirely (hard error).
    group_varying makes odd groups use a 16/10 split instead, so adjacent
    groups' natural quotas differ (12 vs 10) and the wrong-band null must
    align the pair to the smaller quota (review finding 4).
    """
    P, C, D = 15, 32, 3
    rng = np.random.default_rng(7)
    z = rng.standard_normal((3, 3, P, C))
    u = rng.standard_normal((3, 3, P, C))
    x = rng.standard_normal((3, 3, P, C, D))
    sid = np.zeros((3, 3, P, C), dtype=np.int64)
    for gi in range(3):
        for p in range(P):
            if group_varying and gi % 2 == 1:
                keep = 16 if p < P // 2 else 10
            else:
                keep = 20 if p < P // 2 else 12
            sid[gi, :, p, keep:] = 1
    if zero_patient:
        sid[:, :, 0, :] = 1
    path = base / "ragged.npz"
    np.savez(path,
             patient_ids=np.array([f"p{i:03d}" for i in range(P)]),
             groups=np.array(["low", "mid", "high"]),
             lambdas=np.array([-2.0, 0.0, 2.0]),
             strata=np.array(["small_lesion"]),
             z=z, u=u, x=x, stratum_ids=sid)
    return path


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
    args += [str(x) for x in kw.get("extra", [])]  # raw CLI passthrough
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
    # AUDIT evidence-chain gate: b_active holds only groups with >=1 passing
    # confirmatory cell (probe recoverable=("low","mid") by default).
    assert set(contract.b_active) == {"low", "mid"}


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


# ---------------------------------------------------------------------------
# AUDIT evidence-chain gates: PASS-only freeze + b_active passing groups
# ---------------------------------------------------------------------------

def test_freeze_refuses_probe_that_failed_pass_rule(scratch):
    rc, probe_out, result = scratch.run_probe(probe_config(recoverable=()))
    assert rc == 0
    assert result["pass_rule"]["passed"] is False
    rc, _ = _freeze(scratch, probe_out)
    assert rc == 1  # FAILED probes must never freeze


def test_freeze_refuses_probe_json_without_pass_rule(scratch, t_dir):
    rc, probe_out, probe = scratch.run_probe(probe_config())
    assert rc == 0
    stripped = dict(probe)
    stripped.pop("pass_rule")
    p2 = t_dir / "no_pass_rule.json"
    p2.write_text(json.dumps(stripped), encoding="utf-8")
    rc, _ = _freeze(scratch, p2)
    assert rc == 1  # a missing pass_rule is not a PASS


def test_freeze_malformed_per_cell_exits_1(scratch, t_dir):
    # Review finding 3: a truthy non-mapping per_cell must fail cleanly.
    rc, probe_out, probe = scratch.run_probe(probe_config())
    assert rc == 0
    edited = json.loads(json.dumps(probe))
    edited["pass_rule"]["per_cell"] = 5  # malformed
    p2 = t_dir / "bad_per_cell.json"
    p2.write_text(json.dumps(edited), encoding="utf-8")
    rc, _ = _freeze(scratch, p2)
    assert rc == 1


def test_freeze_b_active_only_groups_with_passing_cells(scratch, t_dir):
    rc, probe_out, probe = scratch.run_probe(probe_config(recoverable=("low",)))
    assert rc == 0
    assert probe["pass_rule"]["passed"] is True
    # Keep the overall PASS verdict but leave only "low" with passing cells.
    edited = json.loads(json.dumps(probe))
    for cell, judgement in edited["pass_rule"]["per_cell"].items():
        if not cell.startswith("low|"):
            judgement["pass"] = False
    p2 = t_dir / "only_low.json"
    p2.write_text(json.dumps(edited), encoding="utf-8")
    rc, out = _freeze(scratch, p2)
    assert rc == 0
    contract = RecoverabilityContract.load(out)
    assert tuple(contract.b_active) == ("low",)  # canonical order, passing only


# ---------------------------------------------------------------------------
# AUDIT evidence-chain fix: patient-level variable-length cells
# ---------------------------------------------------------------------------

def test_probe_ragged_cells_subsampled_deterministically(scratch, t_dir):
    npz = _ragged_npz(t_dir)
    config = {"probe": {"source": "npz", "path": str(npz)},
              "cross_fit": {"n_folds": 3, "ridge_lambda": 0.01}, "seed": 5}
    rc1, _out1, r1 = scratch.run_probe(config, n_perm=50)
    rc2, _out2, r2 = scratch.run_probe(config, n_perm=50)
    assert rc1 == 0 and rc2 == 0
    cell = r1["delta"]["small_lesion"]["mid"]["1"]
    assert cell["n_coef"] == 12      # per-cell quota = min per-patient count
    assert cell["ragged_cell"] is True
    assert r1 == r2                  # seeded subsample -> bit-identical reruns


def test_probe_zero_coefficient_patient_exits_1(scratch, t_dir):
    npz = _ragged_npz(t_dir, zero_patient=True)
    config = {"probe": {"source": "npz", "path": str(npz)}, "seed": 5}
    rc, _out, _result = scratch.run_probe(config)
    assert rc == 1  # a patient with zero coefficients in a cell is a hard error


def test_probe_group_varying_quotas_wrong_band_null_runs(scratch, t_dir):
    # Review finding 4: adjacent groups with different natural quotas (12 vs
    # 10) must not shape-error the wrong-band null — the pair aligns to the
    # smaller quota and the probe completes.
    npz = _ragged_npz(t_dir, group_varying=True)
    config = {"probe": {"source": "npz", "path": str(npz)},
              "cross_fit": {"n_folds": 3, "ridge_lambda": 0.01}, "seed": 5}
    rc, _out, result = scratch.run_probe(config, n_perm=50)
    assert rc == 0
    even = result["delta"]["small_lesion"]["low"]["0"]["n_coef"]
    odd = result["delta"]["small_lesion"]["mid"]["0"]["n_coef"]
    assert even == 12 and odd == 10  # per-group natural minima preserved


# ---------------------------------------------------------------------------
# schema v2: freeze writes band_powers (never silent; DESIGN_RC_BRD_clock_v2)
# ---------------------------------------------------------------------------

def test_freeze_writes_explicit_default_band_powers(scratch):
    rc, probe_out, _ = scratch.run_probe(probe_config())
    assert rc == 0
    rc, out = _freeze(scratch, probe_out)
    assert rc == 0
    contract = RecoverabilityContract.load(out)
    # absent CLI/probe powers -> explicit all-1.0 (power-blind control A)
    assert contract.band_powers == {"low": 1.0, "mid": 1.0, "high": 1.0}
    assert contract.band_powers_split == "outer_train_fold_0"


def test_freeze_cli_band_powers_propagated(scratch):
    rc, probe_out, _ = scratch.run_probe(probe_config())
    assert rc == 0
    scratch.counter += 1
    out = scratch.base / ("contract" + str(scratch.counter) + ".json")
    rc = FREEZE.main([
        "--probe-result", str(probe_out), "--fold", "fold_0",
        "--out", str(out), "--s-ref", "0.05",
        "--kappa-grid", "0.0,0.5", "--mean-sha", MEAN_SHA,
        "--band-powers", "low=0.2", "--band-powers", "mid=1.0",
        "--band-powers", "high=5.0",
        "--band-power-split", "outer_train_fold_2"])
    assert rc == 0
    contract = RecoverabilityContract.load(out)
    assert contract.band_powers == {"low": 0.2, "mid": 1.0, "high": 5.0}
    assert contract.band_powers_split == "outer_train_fold_2"


# ---------------------------------------------------------------------------
# schema v2: --band-powers-file (sealed compute_band_powers.py artifact)
# ---------------------------------------------------------------------------

GROUPS3 = {"low": ["LL2"], "mid": ["LH2", "HL2", "HH2"],
           "high": ["LH1", "HL1", "HH1"]}


def _sealed_powers_file(base: Path, powers: dict[str, float], *, fold: str = "fold_0",
                        split: str = "train", mean_sha: str = MEAN_SHA,
                        band_groups: dict | None = None,
                        tamper: str | None = None) -> Path:
    """A sealed band-powers artifact in the compute_band_powers.py v2 schema.

    Mirrors scripts/compute_band_powers.py exactly: provenance fields
    (fold / split / split_label / mean_checkpoint_sha256 / band_groups)
    plus the canonical-JSON artifact_sha256 over the de-hashed payload.
    `tamper` mutates the artifact AFTER sealing ("edit_power" -> stale
    hash, "drop_hash" -> no hash at all); both must be rejected.
    """
    payload: dict[str, Any] = {
        "schema_version": 2, "stage": "rc_brd_band_powers", "fold": fold,
        "split": split,
        "split_label": (f"outer_train_{fold}" if split == "train"
                        else f"{split}_{fold}"),
        "n_samples": 48, "n_patients": 12, "image_size": 192,
        "band_groups": band_groups if band_groups is not None else GROUPS3,
        "band_powers": dict(powers),
        "mean_checkpoint": "checkpoints/rc_brd_mean_pretrain/mean_best.pt",
        "mean_checkpoint_sha256": mean_sha,
    }
    payload["artifact_sha256"] = compute_contract_sha256(payload)
    if tamper == "edit_power":
        payload["band_powers"]["low"] = 42.0   # edited after sealing
    elif tamper == "drop_hash":
        del payload["artifact_sha256"]
    path = base / f"band_powers_{fold}_{split}_{tamper or 'ok'}_{len(powers)}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path

def test_freeze_band_powers_file_seals_values_and_verified_split(scratch, t_dir):
    rc, probe_out, _ = scratch.run_probe(probe_config())
    assert rc == 0
    powers_file = _sealed_powers_file(t_dir, {"low": 0.2, "mid": 1.0, "high": 5.0})
    rc, out = _freeze(scratch, probe_out,
                      extra=["--band-powers-file", str(powers_file)])
    assert rc == 0
    contract = RecoverabilityContract.load(out)
    assert contract.band_powers == {"low": 0.2, "mid": 1.0, "high": 5.0}
    # The split label comes from the VERIFIED artifact provenance (never a
    # hand default): fold_0 + split=train => outer_train_fold_0.
    assert contract.band_powers_split == "outer_train_fold_0"


@pytest.mark.parametrize("kwargs, needle", [
    ({"fold": "fold_1"}, "fold mismatch"),                       # cross-fold P_g
    ({"split": "val"}, "outer-train"),                           # val-split P_g
    ({"mean_sha": "b" * 64}, "mean_checkpoint_sha256 mismatch"),  # foreign MeanNet
    ({"band_groups": {"low": ["LL2"], "mid": ["LH2", "HL2", "HH2"],
      "high": ["LH1", "HL1", "HH1"], "phantom": ["XX9"]}}, "band_groups"),
    ({"tamper": "edit_power"}, "artifact_sha256 mismatch"),      # edited after seal
    ({"tamper": "drop_hash"}, "missing artifact_sha256"),        # unsealed artifact
])
def test_freeze_band_powers_file_provenance_violations_exit_1(
        scratch, t_dir, capsys, kwargs, needle):
    rc, probe_out, _ = scratch.run_probe(probe_config())
    assert rc == 0
    powers_file = _sealed_powers_file(t_dir, {"low": 0.2, "mid": 1.0, "high": 5.0},
                                       **kwargs)
    rc, out = _freeze(scratch, probe_out,
                      extra=["--band-powers-file", str(powers_file)])
    assert rc == 1
    assert not out.exists()  # a rejected artifact never writes a contract
    assert needle in capsys.readouterr().err


def test_freeze_band_powers_and_file_are_mutually_exclusive(scratch, t_dir, capsys):
    # Provenance audit: explicit pairs used to override the file's numbers
    # while silently inheriting its split label (numbers and label from two
    # different sources).  The CLI now rejects the combination outright.
    rc, probe_out, _ = scratch.run_probe(probe_config())
    assert rc == 0
    powers_file = _sealed_powers_file(t_dir, {"low": 9.9, "mid": 8.8, "high": 7.7})
    rc, out = _freeze(scratch, probe_out, extra=[
        "--band-powers-file", str(powers_file),
        "--band-powers", "low=0.2", "--band-powers", "mid=1.0",
        "--band-powers", "high=5.0"])
    assert rc == 1
    assert not out.exists()
    assert "mutually exclusive" in capsys.readouterr().err


def test_freeze_band_power_split_cannot_override_sealed_file_label(scratch, t_dir, capsys):
    # The sealed artifact's VERIFIED split_label is authoritative; a manual
    # --band-power-split next to --band-powers-file could relabel provenance.
    rc, probe_out, _ = scratch.run_probe(probe_config())
    assert rc == 0
    powers_file = _sealed_powers_file(t_dir, {"low": 0.2, "mid": 1.0, "high": 5.0})
    rc, out = _freeze(scratch, probe_out, extra=[
        "--band-powers-file", str(powers_file),
        "--band-power-split", "outer_train_fold_2"])
    assert rc == 1
    assert not out.exists()
    assert "cannot override" in capsys.readouterr().err


def _bad_powers_file(base: Path, kind: str) -> Path:
    """A malformed or absent band-powers artifact (fail-closed cases)."""
    if kind == "missing_path":
        return base / "no_such_band_powers.json"  # never written
    contents = {
        "not_an_object": "[1, 2, 3]",
        "missing_band_powers": json.dumps({"split_label": "val_fold_0"}),
        "empty_band_powers": json.dumps({"band_powers": {}}),
        "non_numeric_value": json.dumps({"band_powers": {"low": "fast"}}),
        "not_json": "{not json",
    }
    path = base / f"bad_powers_{kind}.json"
    path.write_text(contents[kind], encoding="utf-8")
    return path

@pytest.mark.parametrize("kind", ["not_an_object", "missing_band_powers", "empty_band_powers",
                    "non_numeric_value", "not_json", "missing_path"])
def test_freeze_band_powers_file_malformed_or_absent_exits_1(scratch, t_dir, kind):
    rc, probe_out, _ = scratch.run_probe(probe_config())
    assert rc == 0
    bad = _bad_powers_file(t_dir, kind)
    rc, out = _freeze(scratch, probe_out,
                      extra=["--band-powers-file", str(bad)])
    assert rc == 1
    assert not out.exists()  # a rejected artifact never writes a contract
