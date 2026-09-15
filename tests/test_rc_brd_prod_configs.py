"""Config-consistency tests for the RC-BRD production experiment arms.

configs/experiments/{rc_brd_prod_d1,rc_brd_prod_rc,rc_brd_prod_rc_band_snr,
rc_brd_smoke}.yaml form a matched minimax matrix (AUDIT "最小矩阵"): the
three prod arms must stay key-for-key identical except for the explicitly
allowed mechanism switches, and the band-SNR v2 arm must wire
clock_mode=band_snr together with its v2 contract (contract_band_snr.json),
the per-band Min-SNR-gamma loss weighting, the clipped residual output, the
fast EMA (0.995 x every step) and an early-stopping patience that can
actually fire between evaluations (patience >= eval_interval; AUDIT
P1-1/P1-3/B5/P1-11).
"""

from __future__ import annotations

import pathlib

import yaml

CONFIG_DIR = (pathlib.Path(__file__).resolve().parent.parent
              / "configs" / "experiments")

PROD_ARMS = ("rc_brd_prod_d1", "rc_brd_prod_rc", "rc_brd_prod_rc_band_snr")


def _load(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _recursive_diff(left: dict, right: dict, prefix: str = "") -> list[str]:
    """Dotted paths at which two nested config dicts differ.

    A key present on only one side counts as a difference at its own path
    (e.g. rc omitting modules.rc_brd.min_snr_gamma vs band_snr's explicit
    5.0); differing leaves report the leaf's own dotted path.
    """
    paths: list[str] = []
    for key in sorted(set(left) | set(right)):
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in left or key not in right:
            paths.append(path)
            continue
        value_left, value_right = left[key], right[key]
        if isinstance(value_left, dict) and isinstance(value_right, dict):
            paths.extend(_recursive_diff(value_left, value_right, path))
        elif value_left != value_right:
            paths.append(path)
    return paths


class TestBandSnrArm:
    def test_clock_and_loss_contract(self):
        cfg = _load("rc_brd_prod_rc_band_snr")
        rc = cfg["modules"]["rc_brd"]
        # v2 main switch + its v2 contract (with frozen band_powers) + A8
        # per-band Min-SNR weighting on the same query axis + kappa in the
        # frozen kappa_grid.
        assert rc["clock_mode"] == "band_snr"
        assert rc["contract_path"].endswith("contract_band_snr.json")
        assert rc["loss_weighting"] == "per_band_min_snr"
        assert rc["kappa"] == 0.5

    def test_clip_ema_and_early_stopping(self):
        cfg = _load("rc_brd_prod_rc_band_snr")
        # Audit B5/P1-11: PET in [-1,1] -> composed output must be clamped.
        assert cfg["modules"]["residual_bridge"]["clip_output"] is True
        # Audit P1-3: 0.995 x every step (not 0.999 x every 10).
        assert cfg["training"]["ema"] == {"decay": 0.995, "update_every": 1}
        runtime = cfg["runtime"]
        # Audit P1-1: patience (100) must be >= eval_interval (50), else
        # early stopping can never fire between evaluations.
        assert runtime["early_stopping"]["patience"] >= runtime["eval_interval"]


class TestMatchedMatrix:
    def test_band_snr_vs_rc_only_clock_and_loss_keys(self):
        diff = set(_recursive_diff(_load("rc_brd_prod_rc_band_snr"),
                                   _load("rc_brd_prod_rc")))
        allowed = {
            "experiment.name",
            "modules.rc_brd.clock_mode",
            "modules.rc_brd.contract_path",
            "modules.rc_brd.loss_weighting",
            "modules.rc_brd.min_snr_gamma",
        }
        assert diff <= allowed, f"unexpected drift: {sorted(diff - allowed)}"

    def test_d1_vs_rc_only_kappa_and_contract(self):
        diff = set(_recursive_diff(_load("rc_brd_prod_d1"),
                                   _load("rc_brd_prod_rc")))
        allowed = {
            "experiment.name",
            "modules.rc_brd.kappa",
            "modules.rc_brd.contract_path",
        }
        assert diff <= allowed, f"unexpected drift: {sorted(diff - allowed)}"


class TestSharedProdSettings:
    def test_all_prod_arms_clip_residual_output(self):
        for arm in PROD_ARMS:
            cfg = _load(arm)
            assert cfg["modules"]["residual_bridge"]["clip_output"] is True, arm

    def test_smoke_ema_decay_matches_prod(self):
        assert _load("rc_brd_smoke")["training"]["ema"]["decay"] == 0.995
