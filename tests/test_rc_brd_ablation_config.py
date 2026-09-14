"""RC-BRD ablation config generator tests (DESIGN §11; PRD FR-7, C8).

Minimum assertion set (DESIGN §11 test_rc_brd_ablation_config row,
"可加不可减"): all nine arms generate and each differs from the base by
exactly its expected key set (subset AND superset directions); the hash is
stable on identical input and distinct across arms; both A3 variants
generate and differ; a3_variant on non-A3 arms raises; unknown arms raise;
A4 rotation and A5 grid-reversal semantics verified on a small contract
object (group evidence rotated; grid reversed with c flipped accordingly);
the base config is never mutated. No tmp_path usage (sandbox-safe, mirrors
test_rc_brd_contract.py policy).
"""

from __future__ import annotations

import copy
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.rc_brd.ablations import (
    A3_VARIANTS,
    ABLATION_ARMS,
    ablation_config_hash,
    apply_contract_transform,
    build_ablation_config,
    flatten_contract,
    permute_contract_groups,
    reverse_contract_grid,
)
from src.model.rc_brd.contract import (
    RecoverabilityContract,
    compute_contract_sha256,
)


def make_base_config() -> dict:
    """Base config exercising every touched key path (DESIGN §8 schema).

    Initial values deliberately differ from the arm override values so the
    per-arm diff is observable; specialist.enabled starts True so the A2
    bypass shows up in the diff (the A1 direction is covered by its own
    dedicated test below).
    """
    return {
        "modules": {
            "rc_brd": {
                "enabled": False,
                "forward_mode": "bridge_time_changed",
                "kappa": 0.25,
                "endpoint_mode": "zeros",
                "contract_path": "contracts/recoverability_contract_fold0.json",
                "specialist": {"enabled": True, "a_max": 0.25, "gate_init": 0.1},
                "contract": {"support_mode": "floor_gated", "eta_max": 0.8, "floor_rho": 0.1},
                "readout": {"mode": "mc_mean", "mc_samples": 8},
            },
        },
        "losses": {
            "topk_lesion": {"enabled": True, "weight": 0.4, "active_tau_max": 0.5},
            "lesion_roi_l1": {"enabled": True, "weight": 1.0, "dilate_radius": 3},
            "false_hotspot": {"enabled": True, "weight": 0.10, "active_tau_max": 0.45},
            "hotspot_prior": {"enabled": True, "weight": 0.15, "distance_weight": 0.5},
            "organ_consistency": {"enabled": True, "weight": 0.05, "cold_weight": 0.02},
            "focal_frequency": {"enabled": True, "weight": 0.03},
        },
        "model": {"base_loss": {"min_snr_enabled": True, "min_snr_gamma": 5.0}},
    }


# Expected dotlist overrides per arm on make_base_config() (DESIGN §6).
EXPECTED_ARM_OVERRIDES = {
    "A1": {
        "modules.rc_brd.enabled": True,
        "modules.rc_brd.kappa": 0.0,
        "modules.rc_brd.specialist.enabled": True,
    },
    "A2": {
        "modules.rc_brd.enabled": True,
        "modules.rc_brd.specialist.enabled": False,
    },
    "A3a": {
        "modules.rc_brd.enabled": True,
        "modules.rc_brd.contract_transform": "flat_c",
        "modules.rc_brd.a3_variant": "budget_matched",
    },
    "A3b": {
        "modules.rc_brd.enabled": True,
        "modules.rc_brd.contract_transform": "flat_c",
        "modules.rc_brd.a3_variant": "density_matched",
        "modules.rc_brd.density_match": "logsnr",
    },
    "A4": {
        "modules.rc_brd.enabled": True,
        "modules.rc_brd.contract_transform": "group_rotation",
    },
    "A5": {
        "modules.rc_brd.enabled": True,
        "modules.rc_brd.contract_transform": "grid_reversal",
    },
    "A6": {
        "modules.rc_brd.enabled": True,
        "losses.topk_lesion.weight": 0.0,
        "losses.lesion_roi_l1.weight": 0.0,
        "losses.false_hotspot.weight": 0.0,
        "losses.hotspot_prior.weight": 0.0,
        "losses.hotspot_prior.distance_weight": 0.0,
        "losses.organ_consistency.cold_weight": 0.0,
    },
    "A8": {
        "modules.rc_brd.enabled": True,
        "modules.rc_brd.kappa": 0.0,
        "modules.rc_brd.loss_weighting": "per_band_min_snr",
        "modules.rc_brd.min_snr_gamma": 5.0,
    },
    "A9": {
        "modules.rc_brd.enabled": True,
        "modules.rc_brd.endpoint_mode": "ct_minus_mean",
    },
}


def a3_kwargs(arm: str) -> dict:
    """Keyword arguments each arm needs (C8 pairing)."""
    return {"a3_variant": "budget_matched"} if arm == "A3a" else {}


def flatten_leaves(node: dict, prefix: str = "") -> dict:
    """Flatten nested dicts to dotted path -> leaf value (dicts recurse)."""
    flat: dict = {}
    for key, value in node.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_leaves(value, dotted))
        else:
            flat[dotted] = value
    return flat


def get_dotted(config: dict, dotted: str):
    """Fetch a dotted-path value; None when any segment is missing."""
    cursor = config
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def diff_keys(base: dict, other: dict) -> set:
    """Dotted keys whose value differs or that exist in only one config."""
    base_flat = flatten_leaves(base)
    other_flat = flatten_leaves(other)
    keys = set(base_flat) | set(other_flat)
    return {k for k in keys if base_flat.get(k) != other_flat.get(k)}


BAND_GROUPS = {"low": ["LL2"], "mid": ["LH2", "HL2", "HH2"], "high": ["LH1", "HL1", "HH1"]}
LOG_SNR_GRID = (-8.0, -2.0, 3.0, 9.0)  # asymmetric on purpose: mirror ≠ identity
C_VALUES = {
    "low": (0.2, 0.25, 0.3, 0.4),
    "mid": (0.4, 0.5, 0.55, 0.6),
    "high": (0.6, 0.65, 0.7, 0.8),
}
PSD_FLOORS = {"low": 1e-4, "mid": 2e-4, "high": 5e-4}


def make_contract() -> RecoverabilityContract:
    """Small contract object for the A3/A4/A5 transform semantics tests."""
    payload = {
        "fold_id": "fold_0",
        "band_groups": {g: list(bands) for g, bands in BAND_GROUPS.items()},
        "log_snr_grid": list(LOG_SNR_GRID),
        "c_values": {g: list(vals) for g, vals in C_VALUES.items()},
        "mean_checkpoint_sha256": "a" * 64,
        "b_active": ["low", "mid", "high"],
        "psd_floors": dict(PSD_FLOORS),
        "size_thresholds": {"small_lesion_q25": 12.0},
        "kappa_grid": [0.0, 0.25, 0.5],
        "s_ref": 1.0,
        "support_mode": "floor_gated",
        "eta_max": 0.8,
        "floor_rho": 0.1,
    }
    contract = RecoverabilityContract.from_payload(payload)
    contract.validate()
    return contract


@pytest.fixture()
def base_config() -> dict:
    return make_base_config()


# ---------------------------------------------------------------------------
# Nine arms: generation + exact touched-key sets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", list(ABLATION_ARMS))
def test_arm_generates_and_diff_is_exactly_expected_keys(base_config, arm):
    """Each arm builds and differs from base by exactly its expected keys."""
    expected = EXPECTED_ARM_OVERRIDES[arm]
    cfg = build_ablation_config(base_config, arm, **a3_kwargs(arm))
    # 1) every expected key carries the expected value (override applied)
    for dotted, value in expected.items():
        assert get_dotted(cfg, dotted) == value, dotted
    touched = diff_keys(base_config, cfg)
    # 2) subset direction: nothing outside the expected key set changed
    assert touched <= set(expected), (arm, touched - set(expected))
    # 3) superset direction: every expected key whose value really differs
    #    from base appears in the observed diff
    changed = {
        k for k, v in expected.items() if get_dotted(cfg, k) != get_dotted(base_config, k)
    }
    assert changed <= touched, (arm, changed - touched)
    assert touched == changed


def test_ablation_arms_constant_is_final_form():
    """Nine adjudicated arms, no A7 (DESIGN §6 / [裁决] C-2/C-3)."""
    assert ABLATION_ARMS == ("A1", "A2", "A3a", "A3b", "A4", "A5", "A6", "A8", "A9")
    assert A3_VARIANTS == ("budget_matched", "density_matched")


# ---------------------------------------------------------------------------
# Hashing (FR-7.2)
# ---------------------------------------------------------------------------

def test_hash_stable_on_same_input_and_distinct_across_arms(base_config):
    hashes = {}
    for arm in ABLATION_ARMS:
        cfg = build_ablation_config(base_config, arm, **a3_kwargs(arm))
        again = build_ablation_config(copy.deepcopy(base_config), arm, **a3_kwargs(arm))
        assert ablation_config_hash(cfg) == ablation_config_hash(again)
        hashes[arm] = ablation_config_hash(cfg)
    values = list(hashes.values()) + [ablation_config_hash(base_config)]
    assert len(set(values)) == len(values)  # 9 arms + base all distinct


def test_hash_reuses_canonical_json_convention():
    """Same serialization as contract.compute_contract_sha256 (DESIGN §3)."""
    cfg = {"b": {"z": 1, "a": 2.0}, "a": [1, 2, 3]}
    assert ablation_config_hash(cfg) == compute_contract_sha256(cfg)
    reordered = {"a": [1, 2, 3], "b": {"a": 2.0, "z": 1}}
    assert ablation_config_hash(reordered) == ablation_config_hash(cfg)


# ---------------------------------------------------------------------------
# A3 variants and C8 pairing
# ---------------------------------------------------------------------------

def test_a3_both_variants_generate_and_differ(base_config):
    a = build_ablation_config(base_config, "A3a", a3_variant="budget_matched")
    b = build_ablation_config(base_config, "A3b")  # default density_matched
    assert a["modules"]["rc_brd"]["a3_variant"] == "budget_matched"
    assert b["modules"]["rc_brd"]["a3_variant"] == "density_matched"
    assert b["modules"]["rc_brd"]["density_match"] == "logsnr"
    assert "density_match" not in a["modules"]["rc_brd"]
    assert a != b
    assert ablation_config_hash(a) != ablation_config_hash(b)


def test_a3_variant_arm_pairing_is_enforced(base_config):
    """A3a is pinned to budget_matched and A3b to density_matched (C8)."""
    with pytest.raises(ValueError):  # default variant does not pair with A3a
        build_ablation_config(base_config, "A3a")
    with pytest.raises(ValueError):
        build_ablation_config(base_config, "A3b", a3_variant="budget_matched")
    with pytest.raises(ValueError):
        build_ablation_config(base_config, "A3a", a3_variant="mystery")


@pytest.mark.parametrize("arm", ["A1", "A4", "A6", "A8", "A9"])
def test_non_a3_arm_rejects_foreign_a3_variant(base_config, arm):
    with pytest.raises(ValueError):
        build_ablation_config(base_config, arm, a3_variant="budget_matched")
    # the C8 default is a tolerated no-op for non-A3 arms (PRD §3)
    assert build_ablation_config(base_config, arm) == build_ablation_config(
        base_config, arm, a3_variant="density_matched")


@pytest.mark.parametrize("arm", ["", "A0", "A3", "A7", "A10", "a1", "baseline"])
def test_unknown_arm_raises(base_config, arm):
    with pytest.raises(ValueError):
        build_ablation_config(base_config, arm)


# ---------------------------------------------------------------------------
# A4 permutation and A5 reversal semantics on a small contract
# ---------------------------------------------------------------------------

def test_a4_group_rotation_semantics():
    """Group evidence rotates along the sorted-group cycle; structure stays."""
    contract = make_contract()
    rotated = permute_contract_groups(contract)
    order = sorted(BAND_GROUPS)  # ["high", "low", "mid"]
    n = len(order)
    for i, group in enumerate(order):
        source = order[(i + 1) % n]
        assert rotated.c_values[group] == contract.c_values[source]
        assert rotated.psd_floors[group] == pytest.approx(contract.psd_floors[source])
    # structural fields untouched
    assert rotated.band_groups == contract.band_groups
    assert rotated.b_active == contract.b_active
    assert rotated.log_snr_grid == contract.log_snr_grid
    assert rotated.fold_id == contract.fold_id
    rotated.validate()  # rebuilt contract is fail-closed valid
    assert apply_contract_transform(
        contract, "group_rotation").to_payload() == rotated.to_payload()


def test_a5_grid_reversal_semantics():
    """Grid is mirrored (reversed and negated, still ascending) with c flipped."""
    contract = make_contract()
    mirrored = reverse_contract_grid(contract)
    old_grid = contract.log_snr_grid
    assert mirrored.log_snr_grid == tuple(-x for x in reversed(old_grid))
    assert all(
        mirrored.log_snr_grid[i] < mirrored.log_snr_grid[i + 1]
        for i in range(len(old_grid) - 1)
    )
    for group in C_VALUES:
        assert mirrored.c_values[group] == tuple(reversed(contract.c_values[group]))
    # pointwise pairing: the c measured at λ keeps its value at −λ
    for group in C_VALUES:
        lhs = mirrored.effective_c(
            group, torch.tensor([-lam for lam in old_grid], dtype=torch.float64))
        rhs = contract.effective_c(
            group, torch.tensor(list(old_grid), dtype=torch.float64))
        assert torch.allclose(lhs, rhs, rtol=0.0, atol=1e-12)
    mirrored.validate()
    assert apply_contract_transform(
        contract, "grid_reversal").to_payload() == mirrored.to_payload()


def test_a3_flat_contract_semantics():
    """Flat contract pins every c to 0.5 - the shrinkage fixed point."""
    contract = make_contract()
    flat = flatten_contract(contract)
    n_points = len(contract.log_snr_grid)
    for group in C_VALUES:
        assert flat.c_values[group] == (0.5,) * n_points
        probes = torch.tensor([-100.0, -8.0, 0.0, 3.0, 100.0])
        out = flat.effective_c(group, probes)
        assert torch.allclose(out, torch.full_like(out, 0.5))
    assert flat.log_snr_grid == contract.log_snr_grid  # grid untouched
    flat.validate()
    assert apply_contract_transform(
        contract, "flat_c").to_payload() == flat.to_payload()


def test_apply_contract_transform_none_and_unknown():
    contract = make_contract()
    same = apply_contract_transform(contract, "none")
    assert same.to_payload() == contract.to_payload()
    with pytest.raises(ValueError):
        apply_contract_transform(contract, "rotate_groups")


# ---------------------------------------------------------------------------
# A6 / A9 / A1 arm-specific behaviour
# ---------------------------------------------------------------------------

def test_a6_zeroes_only_existing_lesion_safety_weight_leaves(base_config):
    cfg = build_ablation_config(base_config, "A6")
    for dotted, value in EXPECTED_ARM_OVERRIDES["A6"].items():
        assert get_dotted(cfg, dotted) == value, dotted
    # non-weight leaves and unrelated weights stay untouched
    assert cfg["losses"]["topk_lesion"]["active_tau_max"] == 0.5
    assert cfg["losses"]["topk_lesion"]["enabled"] is True
    assert cfg["losses"]["lesion_roi_l1"]["dilate_radius"] == 3
    assert cfg["losses"]["focal_frequency"]["weight"] == 0.03
    assert cfg["model"]["base_loss"]["min_snr_gamma"] == 5.0


def test_a6_without_matching_losses_keys_invents_nothing():
    base = {"modules": {"rc_brd": {"enabled": False}}}
    cfg = build_ablation_config(base, "A6")
    assert cfg["modules"]["rc_brd"]["enabled"] is True
    assert "losses" not in cfg  # existing keys only, no phantom keys


def test_a9_endpoint_mode_flips_both_ways(base_config):
    cfg = build_ablation_config(base_config, "A9")
    assert cfg["modules"]["rc_brd"]["endpoint_mode"] == "ct_minus_mean"
    swapped = copy.deepcopy(base_config)
    swapped["modules"]["rc_brd"]["endpoint_mode"] = "ct_minus_mean"
    back = build_ablation_config(swapped, "A9")
    assert back["modules"]["rc_brd"]["endpoint_mode"] == "zeros"
    broken = copy.deepcopy(base_config)
    broken["modules"]["rc_brd"]["endpoint_mode"] = "mu_image"
    with pytest.raises(ValueError):
        build_ablation_config(broken, "A9")


def test_a9_defaults_to_zeros_when_endpoint_mode_missing():
    base = {"modules": {"rc_brd": {"enabled": False}}}
    cfg = build_ablation_config(base, "A9")  # DESIGN §8 default endpoint_mode
    assert cfg["modules"]["rc_brd"]["endpoint_mode"] == "ct_minus_mean"


def test_a1_forces_specialist_on_when_base_has_it_off(base_config):
    base = copy.deepcopy(base_config)
    base["modules"]["rc_brd"]["specialist"]["enabled"] = False
    cfg = build_ablation_config(base, "A1")
    assert cfg["modules"]["rc_brd"]["specialist"]["enabled"] is True
    assert cfg["modules"]["rc_brd"]["kappa"] == 0.0
    assert cfg["modules"]["rc_brd"]["enabled"] is True


# ---------------------------------------------------------------------------
# Base config immutability
# ---------------------------------------------------------------------------

def test_base_config_not_mutated(base_config):
    snapshot = copy.deepcopy(base_config)
    base_hash = ablation_config_hash(base_config)
    for arm in ABLATION_ARMS:
        cfg = build_ablation_config(base_config, arm, **a3_kwargs(arm))
        assert cfg is not base_config
        assert cfg["modules"] is not base_config["modules"]
        # tampering with the returned config must not leak into the base
        cfg["modules"]["rc_brd"]["enabled"] = "tampered"
        cfg.setdefault("losses", {})["leak"] = True
    assert base_config == snapshot
    assert ablation_config_hash(base_config) == base_hash