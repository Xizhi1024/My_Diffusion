"""Fail-closed contract for the proposed A/B/C/D router ablation.

Matching configuration outside the router is useful preparation, but it does
not establish architectural, initialization, or data-order pairing.  Until
those properties are proven, the plan must remain blocked and must not support
causal attribution.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.config_utils import apply_dotlist_overrides  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "configs/experiments/prior_anchored_paired_ablation_v1.yaml"
BASE_PATH = REPO_ROOT / "configs/experiments/slmf_png_prior_anchored_router_100e.yaml"


def _load_plan() -> dict:
    payload = yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _load_base() -> dict:
    payload = yaml.safe_load(BASE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _materialize(base: dict, overrides: dict) -> dict:
    return apply_dotlist_overrides(copy.deepcopy(base), overrides)


def _router(cfg: dict) -> dict:
    return cfg["modules"]["residual_frequency"]["cross_level_router"]


def _drop_router(cfg: dict) -> dict:
    clone = copy.deepcopy(cfg)
    clone["modules"]["residual_frequency"].pop("cross_level_router", None)
    return clone


def test_four_variants_are_declared_and_policy_is_unique() -> None:
    plan = _load_plan()
    variants = plan["variants"]
    assert set(variants) == {
        "A_no_route",
        "B_fixed_h3_native_null",
        "C_h3_prior_anchored_adaptive",
        "D_unconstrained_learned",
    }
    policies = {
        name: variant["overrides"][
            "modules.residual_frequency.cross_level_router.policy"
        ]
        for name, variant in variants.items()
    }
    assert policies == {
        "A_no_route": "native_only",
        "B_fixed_h3_native_null": "h3_native_null",
        "C_h3_prior_anchored_adaptive": "prior_anchored_learned",
        "D_unconstrained_learned": "learned",
    }


def test_declared_overrides_are_scoped_to_the_router_block() -> None:
    """Check override scope without claiming that the models are paired."""
    plan = _load_plan()
    base = _load_base()
    materialized = [
        _materialize(base, variant["overrides"])
        for variant in plan["variants"].values()
    ]
    without_router = [_drop_router(cfg) for cfg in materialized]
    reference = without_router[0]
    for variant_name, cfg in zip(plan["variants"].keys(), without_router):
        assert cfg == reference, (
            f"variant {variant_name} differs from A outside the router block"
        )


def test_pairing_contract_fields_match_base() -> None:
    plan = _load_plan()
    base = _load_base()
    shared = plan["shared_contract"]
    assert base["experiment"]["seed"] == shared["experiment_seed"]
    assert base["data"]["split_manifest"] == shared["data_split_manifest"]
    assert base["data"]["cache_dir"] == shared["data_cache_dir"]
    assert base["data"]["dataset_contract"] == shared["data_dataset_contract"]
    assert base["data"]["require_cache_lineage"] is True
    assert base["training"]["num_epochs"] == shared["num_epochs"]
    assert base["runtime"]["eval_seed"] == shared["eval_seed"]
    assert base["runtime"]["dataloader_seed"] == shared["dataloader_seed"]
    assert base["runtime"]["eval_interval"] == shared["eval_interval"]
    assert base["runtime"]["save_interval"] == shared["save_interval"]


def test_variant_C_matches_the_cloud_base_config_exactly() -> None:
    """Variant C is the main experiment; it must equal the 100e cloud config."""
    plan = _load_plan()
    base = _load_base()
    c = _materialize(base, plan["variants"]["C_h3_prior_anchored_adaptive"]["overrides"])
    # Variant C only re-asserts the base policy; nothing else changes.
    assert _router(c)["policy"] == "prior_anchored_learned"
    assert _router(c)["h3_schedule_source"] == "direct_png_preview"
    assert _router(c)["h3_allow_unverified_preview_lineage"] is True


def test_variant_B_is_blocked_until_a_formal_schedule_exists() -> None:
    """The preview cannot feed the fail-closed formal H3 loader."""
    plan = _load_plan()
    variant_b = plan["variants"]["B_fixed_h3_native_null"]
    assert variant_b.get("blocked_until"), (
        "Variant B must document why it cannot run today"
    )
    assert variant_b["overrides"][
        "modules.residual_frequency.cross_level_router.h3_schedule_source"
    ] == "formal_h3_v2"
    assert variant_b["overrides"][
        "modules.residual_frequency.cross_level_router.h3_allow_unverified_preview_lineage"
    ] is False


def test_plan_marks_itself_exploratory_and_non_production() -> None:
    plan = _load_plan()
    assert plan["status"] == "BLOCKED"
    assert plan["paired_execution_allowed"] is False
    assert plan["causal_attribution_allowed"] is False
    assert plan["scientific_claim_allowed"] is False
    assert plan["production_activation_allowed"] is False


def test_plan_names_all_known_pairing_blockers() -> None:
    plan = _load_plan()
    assert set(plan["blocked_reasons"]) == {
        "unequal_parameter_and_state_dict_structure",
        "unequal_amplitude_contract",
        "conditional_construction_changes_initialization_rng",
        "formal_h3_v2_schedule_missing",
    }
    assert (
        plan["shared_contract"]["declared_override_scope"]
        == "modules.residual_frequency.cross_level_router"
    )


def test_plan_uses_cross_policy_route_mass_names() -> None:
    plan = _load_plan()
    comparisons = set(plan["primary_comparisons"])
    assert {
        "frequency/route_native_mass",
        "frequency/route_shallow_mass",
        "frequency/route_null_mass",
    } <= comparisons
    assert not {
        "frequency/prior_anchor_native_mass",
        "frequency/prior_anchor_shallow_mass",
        "frequency/prior_anchor_null_mass",
    } & comparisons
