"""Fail-closed V2-05B audit for H3 fixed-schedule inference.

The frozen H3 table supports exact lookup on its ten-point probe grid from only
band, timestep, and constants.  Production training/sampling uses a broader
timestep domain, but no off-grid resolver is frozen.  The table also contains
one scalar while the router needs ``[native, shallow, null]``.  This audit
records those two independent failures and stops before model evaluation.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "h3_fixed_schedule_inference_v1.json"
DEFAULT_BUNDLE = (
    ROOT
    / "results"
    / "mechanism_validation_v2"
    / "04_main_integration_freeze"
    / "production_calibration_bundle.json"
)
DEFAULT_CONTRACT = (
    ROOT
    / "results"
    / "mechanism_validation_v2"
    / "04_main_integration_freeze"
    / "resolved_integration_contract.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "mechanism_validation_v2"
    / "05B_h3_fixed_schedule_inference"
)
DEFAULT_MODEL_SOURCE = ROOT / "src" / "model" / "slmf_bbdm.py"
DEFAULT_ROUTER_SOURCE = (
    ROOT / "src" / "model" / "frequency" / "spectral_router.py"
)
DEFAULT_PRODUCTION_CONFIG = (
    ROOT / "configs" / "experiments" / "slmf_png_spectral_router_v5.yaml"
)

sys.path.insert(0, str(ROOT))

from src.mechanism_validation.common import (  # noqa: E402
    canonical_json_sha256,
    file_sha256,
)


PIPELINE_ID = "H3_FIXED_SCHEDULE_INFERENCE_V1"
SPEC_CONFIG_SHA256 = (
    "103878ed0fb4ea67e5d9f6dd9d13fd53d49d8d0661f0d86150ccd21969681036"
)
SPEC_BUNDLE_FILE_SHA256 = (
    "b25e98994d057769087351f238bcdf4d551220022d5cd0f9c5be579c180990ef"
)
SPEC_BUNDLE_SELF_SHA256 = (
    "02042c9c959038610de21bea9bc732caa15ba84f37b3f48198cf9ed3660c5525"
)
SPEC_CONTRACT_FILE_SHA256 = (
    "0016c0380ead1d4ed8ad2fcc1c7665cfcb79cbcbb241424ea44db4ab6e837dd1"
)
SPEC_CONTRACT_SELF_SHA256 = (
    "15fed00ba875a4cb78ad9a3c9217e871f9037970dd372582d15252f3b2f76a35"
)
SPEC_H3_SOURCE_PARTITION_FINGERPRINT = (
    "744327eb274d9ff230e57e9a737f30bbb63fdea05ea2abec12413a0469b5c865"
)
SPEC_H4_V2_NESTED_PARTITION_FINGERPRINT = (
    "0146d21dc3b242e65977fda52f32cf3c6efe79b8fd11011b392068b69a0ba9ad"
)
SPEC_MODEL_SOURCE_SHA256 = (
    "8216d9e100bd5682a473fbf3dff8b888d666d963edec45db1cc3495423c2ce6c"
)
SPEC_ROUTER_SOURCE_SHA256 = (
    "8337db1cfc0ac1d47260f449588cd8f6de3ee5137ea04e53162a6353f0615718"
)
SPEC_PRODUCTION_CONFIG_SHA256 = (
    "42f77c7c9460ecba596ac3ca16bc3d234af8d437611fa68febf5d99d1e74f59d"
)
SPEC_BANDS = ("l2_lh", "l2_hl", "l2_hh", "l1_lh", "l1_hl", "l1_hh")
SPEC_TIMESTEPS = (0, 50, 100, 200, 350, 500, 650, 800, 900, 950)
SPEC_ROUTES = ("native", "shallow", "null")
SPEC_NUM_TRAIN_TIMESTEPS = 1000
SPEC_EVAL_SAMPLING_STEPS = 20
SPEC_EVAL_TIMESTEPS = (
    999, 946, 893, 841, 788, 736, 683, 630, 578, 525,
    473, 420, 368, 315, 262, 210, 157, 105, 52, 0,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _safe_hash(path: Path) -> tuple[str | None, str | None]:
    if not path.is_file():
        return None, "file_not_found"
    try:
        return file_sha256(path), None
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _safe_load(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "file_not_found"
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, "json_root_is_not_an_object"
    return payload, None


def _safe_source(path: Path) -> tuple[str | None, ast.Module | None, str | None]:
    if not path.is_file():
        return None, None, "file_not_found"
    try:
        source = path.read_text(encoding="utf-8")
        return source, ast.parse(source), None
    except (OSError, UnicodeError, SyntaxError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def _safe_yaml(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "file_not_found"
    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (ImportError, OSError, UnicodeError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, "yaml_root_is_not_an_object"
    return payload, None


def _nested(payload: Mapping[str, Any] | None, *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _json_list_tuple(value: Any) -> tuple[Any, ...]:
    return tuple(value) if isinstance(value, list) else ()


def _finite_real(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _self_hash(
    payload: Mapping[str, Any] | None,
    field: str,
) -> tuple[str | None, str | None]:
    if payload is None:
        return None, None
    declared = payload.get(field)
    if not isinstance(declared, str):
        return None, None
    body = dict(payload)
    body.pop(field, None)
    try:
        computed = canonical_json_sha256(body)
    except (OverflowError, TypeError, ValueError, UnicodeError):
        computed = None
    return declared, computed


def inspect_config(
    config: Mapping[str, Any] | None,
    *,
    load_error: str | None = None,
) -> dict[str, Any]:
    declared, computed = _self_hash(config, "config_sha256")
    bundle_spec = _nested(
        config, "source_artifacts", "production_calibration_bundle"
    )
    contract_spec = _nested(
        config, "source_artifacts", "resolved_integration_contract"
    )
    schedule = _nested(config, "h3_schedule_contract")
    mapping = _nested(config, "mapping_contract")
    policy = _nested(config, "decision_policy")
    runtime_sources = _nested(config, "runtime_sources")
    timestep_contract = _nested(config, "production_timestep_contract")
    code_contract = _nested(config, "production_code_state_contract")
    checks = {
        "json_object_loaded": config is not None and load_error is None,
        "pipeline_id_matches_spec": _nested(config, "pipeline_id") == PIPELINE_ID,
        "status_is_frozen_before_model_evaluation": (
            _nested(config, "status") == "FROZEN_BEFORE_MODEL_EVALUATION"
        ),
        "config_self_hash_recomputes": (
            declared is not None and declared == computed
        ),
        "config_self_hash_matches_spec": declared == SPEC_CONFIG_SHA256,
        "bundle_file_hash_is_spec_locked": (
            _nested(bundle_spec, "file_sha256") == SPEC_BUNDLE_FILE_SHA256
        ),
        "bundle_self_hash_is_spec_locked": (
            _nested(bundle_spec, "self_sha256") == SPEC_BUNDLE_SELF_SHA256
        ),
        "contract_file_hash_is_spec_locked": (
            _nested(contract_spec, "file_sha256") == SPEC_CONTRACT_FILE_SHA256
        ),
        "contract_self_hash_is_spec_locked": (
            _nested(contract_spec, "self_sha256") == SPEC_CONTRACT_SELF_SHA256
        ),
        "h3_source_partition_fingerprint_is_spec_locked": (
            _nested(
                config,
                "frozen_partitions",
                "h3_schedule_source",
                "fingerprint_sha256",
            )
            == SPEC_H3_SOURCE_PARTITION_FINGERPRINT
        ),
        "h4_v2_nested_lineage_fingerprint_is_spec_locked": (
            _nested(
                config,
                "frozen_partitions",
                "h4_v2_nested_lineage",
                "fingerprint_sha256",
            )
            == SPEC_H4_V2_NESTED_PARTITION_FINGERPRINT
        ),
        "model_source_hash_is_spec_locked": (
            _nested(runtime_sources, "slmf_bbdm", "file_sha256")
            == SPEC_MODEL_SOURCE_SHA256
        ),
        "router_source_hash_is_spec_locked": (
            _nested(runtime_sources, "spectral_router", "file_sha256")
            == SPEC_ROUTER_SOURCE_SHA256
        ),
        "production_config_hash_is_spec_locked": (
            _nested(runtime_sources, "production_v5_config", "file_sha256")
            == SPEC_PRODUCTION_CONFIG_SHA256
        ),
        "band_order_is_spec_locked": (
            _json_list_tuple(_nested(schedule, "bands")) == SPEC_BANDS
        ),
        "timestep_grid_is_spec_locked": (
            _json_list_tuple(_nested(schedule, "timesteps"))
            == SPEC_TIMESTEPS
        ),
        "schedule_entry_count_is_spec_locked": (
            _nested(schedule, "entry_count")
            == len(SPEC_BANDS) * len(SPEC_TIMESTEPS)
        ),
        "frozen_grid_scope_is_exact_only": (
            _nested(schedule, "availability_scope")
            == "frozen_grid_exact_lookup_only"
            and _nested(schedule, "off_grid_lookup_authorized") is False
        ),
        "production_timestep_domain_is_spec_locked": (
            _nested(timestep_contract, "num_train_timesteps")
            == SPEC_NUM_TRAIN_TIMESTEPS
            and _nested(timestep_contract, "training_timestep_minimum") == 0
            and _nested(timestep_contract, "training_timestep_maximum") == 999
            and _nested(timestep_contract, "eval_sampling_steps")
            == SPEC_EVAL_SAMPLING_STEPS
            and _json_list_tuple(
                _nested(timestep_contract, "expected_eval_timesteps")
            )
            == SPEC_EVAL_TIMESTEPS
        ),
        "off_grid_resolver_is_explicitly_undefined": (
            _nested(timestep_contract, "off_grid_resolver_status")
            == "UNDEFINED_FAIL_CLOSED"
            and _nested(timestep_contract, "authorized_off_grid_resolver")
            is None
            and set(
                _json_list_tuple(
                    _nested(
                        timestep_contract,
                        "forbidden_unfrozen_substitutes",
                    )
                )
            )
            == {"nearest", "interpolation", "global_mean_fallback"}
        ),
        "production_code_state_is_spec_locked": (
            _nested(code_contract, "h3_fixed_schedule_activation")
            == "ABSENT_NOT_ACTIVATED"
            and _nested(
                code_contract,
                "uncertainty_aware_router_default_enabled",
            )
            is False
            and _nested(code_contract, "l3_injection") == "zero_tensor"
            and _json_list_tuple(_nested(code_contract, "route_order"))
            == SPEC_ROUTES
        ),
        "ternary_route_order_is_spec_locked": (
            _json_list_tuple(_nested(mapping, "required_route_order"))
            == SPEC_ROUTES
        ),
        "mapping_is_explicitly_undefined": (
            _nested(mapping, "frozen_mapping_status")
            == "UNDEFINED_FAIL_CLOSED"
            and _nested(mapping, "authorized_mapping") is None
        ),
        "conservative_substitute_is_forbidden": (
            _nested(mapping, "conservative_substitute_allowed") is False
        ),
        "decision_policy_disables_all_downstream_actions": (
            _nested(policy, "expected_decision") == "FAIL"
            and _nested(policy, "next_stage_allowed") is False
            and _nested(policy, "model_evaluation_allowed") is False
            and _nested(policy, "production_activation_allowed") is False
            and _nested(policy, "stop_before_model_change") is True
        ),
    }
    return {
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "load_error": load_error,
        "declared_self_sha256": declared,
        "computed_self_sha256": computed,
        "spec_self_sha256": SPEC_CONFIG_SHA256,
    }


def inspect_frozen_artifact(
    path: Path,
    payload: Mapping[str, Any] | None,
    *,
    load_error: str | None,
    expected_file_sha256: str,
    self_hash_field: str,
    expected_self_sha256: str,
) -> dict[str, Any]:
    observed_file_sha256, hash_error = _safe_hash(path)
    declared, computed = _self_hash(payload, self_hash_field)
    checks = {
        "file_exists": path.is_file(),
        "json_object_loaded": payload is not None and load_error is None,
        "file_sha256_matches_spec": (
            observed_file_sha256 == expected_file_sha256
        ),
        "declared_self_hash_matches_spec": declared == expected_self_sha256,
        "self_hash_recomputes": (
            declared is not None and declared == computed
        ),
    }
    return {
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "path": _display_path(path),
        "checks": checks,
        "load_error": load_error,
        "hash_error": hash_error,
        "observed_file_sha256": observed_file_sha256,
        "expected_file_sha256": expected_file_sha256,
        "declared_self_sha256": declared,
        "computed_self_sha256": computed,
        "expected_self_sha256": expected_self_sha256,
    }


def inspect_runtime_file(
    path: Path,
    *,
    expected_file_sha256: str,
    parse_error: str | None,
) -> dict[str, Any]:
    observed, hash_error = _safe_hash(path)
    checks = {
        "file_exists": path.is_file(),
        "file_sha256_matches_spec": observed == expected_file_sha256,
        "source_parses": parse_error is None,
    }
    return {
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "path": _display_path(path),
        "checks": checks,
        "parse_error": parse_error,
        "hash_error": hash_error,
        "observed_file_sha256": observed,
        "expected_file_sha256": expected_file_sha256,
    }


def _function_source(
    source: str | None,
    tree: ast.Module | None,
    name: str,
    *,
    class_name: str | None = None,
) -> str:
    if source is None or tree is None:
        return ""
    scope: ast.AST = tree
    if class_name is not None:
        scope = next(
            (
                item
                for item in tree.body
                if isinstance(item, ast.ClassDef) and item.name == class_name
            ),
            tree,
        )
    node = next(
        (
            item
            for item in getattr(scope, "body", ())
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == name
        ),
        None,
    )
    return ast.get_source_segment(source, node) if node is not None else ""


def inspect_production_code_state(
    *,
    model_source: str | None,
    model_tree: ast.Module | None,
    router_source: str | None,
    router_tree: ast.Module | None,
    production_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    model_source = model_source or ""
    router_source = router_source or ""
    forward_source = _function_source(
        router_source,
        router_tree,
        "forward",
        class_name="SpectralEvidenceFrequencyRouter",
    )
    injection_source = _function_source(
        model_source, model_tree, "_build_frequency_injections"
    )
    router_cfg = _nested(
        production_config,
        "modules",
        "residual_frequency",
        "cross_level_router",
    )
    checks = {
        "production_num_train_timesteps_is_1000": (
            _nested(
                production_config,
                "modules",
                "bbdm_bridge",
                "num_train_timesteps",
            )
            == SPEC_NUM_TRAIN_TIMESTEPS
        ),
        "production_eval_sampling_steps_is_20": (
            _nested(production_config, "runtime", "eval_sampling_steps")
            == SPEC_EVAL_SAMPLING_STEPS
        ),
        "training_samples_full_timestep_domain": (
            "torch.randint(0, T, (B,), device=device)" in model_source
        ),
        "sampling_uses_long_linspace": (
            "torch.linspace(T - 1, 0, steps, device=device, dtype=torch.long)"
            in model_source
        ),
        "uncertainty_aware_router_is_default_off": (
            'cross_level_router_cfg.get(\n                        "uncertainty_aware_enabled", False'
            in model_source
            and (
                not isinstance(router_cfg, Mapping)
                or router_cfg.get("uncertainty_aware_enabled", False) is False
            )
        ),
        "h3_bundle_consumer_is_absent": (
            "production_calibration_bundle" not in model_source
            and "production_calibration_bundle" not in router_source
            and "h3_fixed_schedule" not in json.dumps(
                production_config or {}, sort_keys=True
            )
        ),
        "l3_injection_is_zero": (
            "l3 = current_residual.new_zeros(" in forward_source
            and "injections = [l3, l2, l1, l0]" in forward_source
        ),
        "route_order_is_native_shallow_null": (
            'destinations = ("native", "shallow", "null")' in model_source
            and "native_l2 = gated_l2 * routes_l2[..., 0" in forward_source
            and "shallow_l2 = gated_l2 * routes_l2[..., 1" in forward_source
        ),
        "router_does_not_receive_direct_pet_or_mask": (
            "batch" not in injection_source
            and "target_pet" not in forward_source
            and "lesion_mask" not in forward_source
            and "_ = lesion_score, topq_mask" in forward_source
        ),
    }
    return {
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "h3_fixed_schedule_activation": "ABSENT_NOT_ACTIVATED",
        "uncertainty_aware_router_default_enabled": False,
        "l3_injection": "zero_tensor",
        "route_order": list(SPEC_ROUTES),
        "forbidden_direct_inputs_observed": [],
    }


def inspect_h3_schedule(
    bundle: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    model = _nested(bundle, "models", "h3_fixed_schedule")
    schedule = _nested(model, "schedule")
    expected_keys = {
        f"{band}|{timestep}"
        for band in SPEC_BANDS
        for timestep in SPEC_TIMESTEPS
    }
    observed_keys = set(schedule) if isinstance(schedule, Mapping) else set()
    values = list(schedule.values()) if isinstance(schedule, Mapping) else []
    numeric_values = [_finite_real(value) for value in values]
    lower = _nested(config, "h3_schedule_contract", "scalar_minimum")
    upper = _nested(config, "h3_schedule_contract", "scalar_maximum")
    numeric_lower = _finite_real(lower)
    numeric_upper = _finite_real(upper)
    values_in_range = (
        numeric_lower is not None
        and numeric_upper is not None
        and all(
            value is not None and numeric_lower <= value <= numeric_upper
            for value in numeric_values
        )
    )
    checks = {
        "model_is_object": isinstance(model, Mapping),
        "schedule_is_object": isinstance(schedule, Mapping),
        "schedule_type_is_frozen_band_timestep": (
            _nested(model, "type") == "frozen_band_timestep_schedule"
        ),
        "time_conditioned_is_true": _nested(model, "time_conditioned") is True,
        "evidence_conditioned_is_false": (
            _nested(model, "evidence_conditioned") is False
        ),
        "frozen_band_order_matches": (
            _json_list_tuple(_nested(bundle, "frozen_grid", "band_order"))
            == SPEC_BANDS
        ),
        "all_expected_entries_present": not (expected_keys - observed_keys),
        "no_unexpected_entries_present": not (observed_keys - expected_keys),
        "entry_count_is_complete": (
            len(observed_keys) == len(SPEC_BANDS) * len(SPEC_TIMESTEPS)
        ),
        "all_scalars_are_finite_and_in_range": values_in_range,
    }
    missing = sorted(expected_keys - observed_keys)
    unexpected = sorted(observed_keys - expected_keys)
    passed = all(checks.values())
    return {
        "decision": "PASS" if passed else "FAIL",
        "lookup_available": passed,
        "scope": "exact_lookup_on_the_frozen_band_timestep_grid",
        "runtime_lookup_inputs": [
            "band",
            "timestep",
            "frozen_schedule_constants",
        ],
        "forbidden_inputs_read": [],
        "checks": checks,
        "expected_entry_count": len(expected_keys),
        "observed_entry_count": len(observed_keys),
        "missing_entries": missing,
        "unexpected_entries": unexpected,
    }


def inspect_production_timestep_consumption(
    bundle: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None,
    code_state: Mapping[str, Any],
) -> dict[str, Any]:
    frozen_grid = set(SPEC_TIMESTEPS)
    eval_timesteps = list(SPEC_EVAL_TIMESTEPS)
    overlap = [value for value in eval_timesteps if value in frozen_grid]
    off_grid = [value for value in eval_timesteps if value not in frozen_grid]
    timestep_contract = _nested(config, "production_timestep_contract")
    model = _nested(bundle, "models", "h3_fixed_schedule")
    resolver_keys = {
        "off_grid_resolver",
        "interpolation",
        "nearest",
        "timestep_resolver",
    }
    model_keys = set(model) if isinstance(model, Mapping) else set()
    checks = {
        "production_code_shape_verified": code_state.get("decision") == "PASS",
        "num_train_timesteps_is_1000": (
            _nested(timestep_contract, "num_train_timesteps")
            == SPEC_NUM_TRAIN_TIMESTEPS
        ),
        "eval_sampling_steps_is_20": (
            _nested(timestep_contract, "eval_sampling_steps")
            == SPEC_EVAL_SAMPLING_STEPS
        ),
        "eval_timestep_sequence_matches": (
            _json_list_tuple(
                _nested(timestep_contract, "expected_eval_timesteps")
            )
            == SPEC_EVAL_TIMESTEPS
        ),
        "nineteen_of_twenty_eval_steps_are_off_grid": (
            len(off_grid) == 19 and overlap == [0]
        ),
        "bundle_has_no_frozen_off_grid_resolver": not (
            model_keys & resolver_keys
        ),
        "off_grid_resolver_is_undefined": (
            _nested(timestep_contract, "off_grid_resolver_status")
            == "UNDEFINED_FAIL_CLOSED"
            and _nested(timestep_contract, "authorized_off_grid_resolver")
            is None
        ),
        "unfrozen_substitutes_are_forbidden": (
            set(
                _json_list_tuple(
                    _nested(
                        timestep_contract,
                        "forbidden_unfrozen_substitutes",
                    )
                )
            )
            == {"nearest", "interpolation", "global_mean_fallback"}
        ),
    }
    return {
        "decision": "FAIL",
        "definition_status": "UNDEFINED",
        "consumption_defined": False,
        "undefined_consumption_proven": all(checks.values()),
        "checks": checks,
        "num_train_timesteps": SPEC_NUM_TRAIN_TIMESTEPS,
        "eval_sampling_steps": SPEC_EVAL_SAMPLING_STEPS,
        "eval_timesteps": eval_timesteps,
        "frozen_grid_overlap": overlap,
        "off_grid_timesteps": off_grid,
        "off_grid_count": len(off_grid),
        "reason": (
            "The production timestep domain is broader than the frozen H3 "
            "probe grid, and no interpolation, nearest, or global-mean "
            "off-grid resolver is frozen."
        ),
    }


def inspect_mapping_identifiability(
    bundle: Mapping[str, Any] | None,
    contract: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    usage = _nested(
        bundle, "usage_contract", "recoverability_route_mapping"
    )
    mechanism_level = _nested(
        contract, "fixed_fallback_behavior", "mechanism_validation_level"
    )
    mapping = _nested(config, "mapping_contract")
    model = _nested(bundle, "models", "h3_fixed_schedule")
    explicit_mapping_keys = {
        "route_mapping",
        "recoverability_to_route",
        "route_probabilities",
        "route_logits",
    }
    observed_model_keys = set(model) if isinstance(model, Mapping) else set()
    checks = {
        "input_is_one_scalar": (
            _nested(mapping, "input")
            == "one_scalar_recoverability_per_band_timestep"
        ),
        "router_requires_native_shallow_null": (
            _json_list_tuple(_nested(mapping, "required_route_order"))
            == SPEC_ROUTES
        ),
        "router_requires_probability_distribution": (
            _nested(mapping, "required_output")
            == "three_component_probability_distribution"
        ),
        "bundle_explicitly_says_mapping_is_undefined": (
            isinstance(usage, str)
            and isinstance(
                _nested(mapping, "bundle_undefined_marker"), str
            )
            and _nested(mapping, "bundle_undefined_marker") in usage
        ),
        "contract_exposes_only_scalar_recoverability": (
            mechanism_level
            == _nested(mapping, "contract_scalar_marker")
        ),
        "no_frozen_mapping_object_or_parameters": not (
            observed_model_keys & explicit_mapping_keys
        ),
        "authorized_mapping_is_absent": (
            _nested(mapping, "authorized_mapping") is None
        ),
    }
    undefined_proven = all(checks.values())
    return {
        "decision": "FAIL",
        "identifiable": False,
        "undefined_mapping_proven": undefined_proven,
        "checks": checks,
        "input_degrees": 1,
        "ternary_simplex_degrees_of_freedom": 2,
        "frozen_mapping_parameter_count": 0,
        "required_route_order": list(SPEC_ROUTES),
        "reason": (
            "The frozen artifacts provide one recoverability scalar but no "
            "frozen function or constraints that uniquely map it to the "
            "[native, shallow, null] probability simplex."
        ),
    }


def build_decision(
    *,
    config_path: Path = DEFAULT_CONFIG,
    bundle_path: Path | None = None,
    contract_path: Path | None = None,
    model_source_path: Path | None = None,
    router_source_path: Path | None = None,
    production_config_path: Path | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    bundle_path = (bundle_path or DEFAULT_BUNDLE).resolve()
    contract_path = (contract_path or DEFAULT_CONTRACT).resolve()
    model_source_path = (model_source_path or DEFAULT_MODEL_SOURCE).resolve()
    router_source_path = (router_source_path or DEFAULT_ROUTER_SOURCE).resolve()
    production_config_path = (
        production_config_path or DEFAULT_PRODUCTION_CONFIG
    ).resolve()

    config, config_error = _safe_load(config_path)
    bundle, bundle_error = _safe_load(bundle_path)
    contract, contract_error = _safe_load(contract_path)
    model_source, model_tree, model_error = _safe_source(model_source_path)
    router_source, router_tree, router_error = _safe_source(
        router_source_path
    )
    production_config, production_config_error = _safe_yaml(
        production_config_path
    )

    config_gate = inspect_config(config, load_error=config_error)
    bundle_gate = inspect_frozen_artifact(
        bundle_path,
        bundle,
        load_error=bundle_error,
        expected_file_sha256=SPEC_BUNDLE_FILE_SHA256,
        self_hash_field="bundle_sha256",
        expected_self_sha256=SPEC_BUNDLE_SELF_SHA256,
    )
    contract_gate = inspect_frozen_artifact(
        contract_path,
        contract,
        load_error=contract_error,
        expected_file_sha256=SPEC_CONTRACT_FILE_SHA256,
        self_hash_field="contract_sha256",
        expected_self_sha256=SPEC_CONTRACT_SELF_SHA256,
    )
    model_source_gate = inspect_runtime_file(
        model_source_path,
        expected_file_sha256=SPEC_MODEL_SOURCE_SHA256,
        parse_error=model_error,
    )
    router_source_gate = inspect_runtime_file(
        router_source_path,
        expected_file_sha256=SPEC_ROUTER_SOURCE_SHA256,
        parse_error=router_error,
    )
    production_config_gate = inspect_runtime_file(
        production_config_path,
        expected_file_sha256=SPEC_PRODUCTION_CONFIG_SHA256,
        parse_error=production_config_error,
    )
    code_state_gate = inspect_production_code_state(
        model_source=model_source,
        model_tree=model_tree,
        router_source=router_source,
        router_tree=router_tree,
        production_config=production_config,
    )

    bundle_h3_source_partition = _nested(
        bundle, "source", "original_partition_fingerprint_sha256"
    )
    config_h3_source_partition = _nested(
        config,
        "frozen_partitions",
        "h3_schedule_source",
        "fingerprint_sha256",
    )
    bundle_nested_partition = _nested(
        bundle, "source", "h4_v2_nested_partition_fingerprint_sha256"
    )
    contract_nested_partition = _nested(
        contract, "h4_v2_reference", "partition_fingerprint_sha256"
    )
    config_nested_partition = _nested(
        config,
        "frozen_partitions",
        "h4_v2_nested_lineage",
        "fingerprint_sha256",
    )
    partition_checks = {
        "h3_source_config_matches_spec": (
            config_h3_source_partition
            == SPEC_H3_SOURCE_PARTITION_FINGERPRINT
        ),
        "h3_source_bundle_matches_spec": (
            bundle_h3_source_partition
            == SPEC_H3_SOURCE_PARTITION_FINGERPRINT
        ),
        "h3_source_config_matches_bundle": (
            config_h3_source_partition == bundle_h3_source_partition
        ),
        "h4_v2_nested_config_matches_spec": (
            config_nested_partition
            == SPEC_H4_V2_NESTED_PARTITION_FINGERPRINT
        ),
        "h4_v2_nested_bundle_matches_spec": (
            bundle_nested_partition
            == SPEC_H4_V2_NESTED_PARTITION_FINGERPRINT
        ),
        "h4_v2_nested_contract_matches_spec": (
            contract_nested_partition
            == SPEC_H4_V2_NESTED_PARTITION_FINGERPRINT
        ),
        "h4_v2_nested_lineage_matches": (
            config_nested_partition
            == bundle_nested_partition
            == contract_nested_partition
        ),
    }
    partition_gate = {
        "decision": (
            "PASS" if all(partition_checks.values()) else "FAIL"
        ),
        "checks": partition_checks,
        "h3_schedule_source": {
            "expected_fingerprint_sha256": (
                SPEC_H3_SOURCE_PARTITION_FINGERPRINT
            ),
            "config_fingerprint_sha256": config_h3_source_partition,
            "bundle_fingerprint_sha256": bundle_h3_source_partition,
        },
        "h4_v2_nested_lineage": {
            "expected_fingerprint_sha256": (
                SPEC_H4_V2_NESTED_PARTITION_FINGERPRINT
            ),
            "config_fingerprint_sha256": config_nested_partition,
            "bundle_fingerprint_sha256": bundle_nested_partition,
            "contract_fingerprint_sha256": contract_nested_partition,
        },
    }

    core_integrity_ok = all(
        gate["decision"] == "PASS"
        for gate in (
            config_gate,
            bundle_gate,
            contract_gate,
            partition_gate,
        )
    )
    runtime_integrity_ok = all(
        gate["decision"] == "PASS"
        for gate in (
            model_source_gate,
            router_source_gate,
            production_config_gate,
            code_state_gate,
        )
    )
    integrity_ok = core_integrity_ok and runtime_integrity_ok
    integrity_gate = {
        "decision": "PASS" if integrity_ok else "FAIL",
        "config": config_gate,
        "production_calibration_bundle": bundle_gate,
        "resolved_integration_contract": contract_gate,
        "frozen_partitions": partition_gate,
        "runtime_sources": {
            "decision": "PASS" if runtime_integrity_ok else "FAIL",
            "slmf_bbdm": model_source_gate,
            "spectral_router": router_source_gate,
            "production_v5_config": production_config_gate,
        },
    }

    raw_grid_gate = inspect_h3_schedule(bundle, config)
    if core_integrity_ok:
        grid_gate = raw_grid_gate
    else:
        grid_gate = {
            **raw_grid_gate,
            "decision": "NOT_EVALUATED",
            "lookup_available": False,
            "blocked_by": "frozen_input_integrity",
        }

    grid_ok = grid_gate["decision"] == "PASS"
    if core_integrity_ok and grid_ok:
        mapping_gate = inspect_mapping_identifiability(
            bundle, contract, config
        )
    else:
        mapping_gate = {
            "decision": "NOT_EVALUATED",
            "identifiable": False,
            "undefined_mapping_proven": False,
            "blocked_by": (
                "frozen_input_integrity"
                if not core_integrity_ok
                else "frozen_grid_lookup"
            ),
        }

    if core_integrity_ok and runtime_integrity_ok and grid_ok:
        timestep_gate = inspect_production_timestep_consumption(
            bundle, config, code_state_gate
        )
    else:
        timestep_gate = {
            "decision": "NOT_EVALUATED",
            "definition_status": "UNDEFINED",
            "consumption_defined": False,
            "undefined_consumption_proven": False,
            "blocked_by": (
                "frozen_grid_lookup"
                if core_integrity_ok and not grid_ok
                else "frozen_input_integrity"
            ),
        }

    scientific_proofs_complete = (
        timestep_gate.get("undefined_consumption_proven") is True
        and mapping_gate.get("undefined_mapping_proven") is True
    )
    scientific_gate_evaluated = (
        integrity_ok
        and grid_ok
        and timestep_gate["decision"] == "FAIL"
        and mapping_gate["decision"] == "FAIL"
        and scientific_proofs_complete
    )
    if not integrity_ok:
        failure_phase = "FROZEN_INPUT_INTEGRITY_FAIL_CLOSED"
    elif not grid_ok:
        failure_phase = "FROZEN_GRID_LOOKUP_FAIL_CLOSED"
    elif not scientific_proofs_complete:
        failure_phase = "SCIENTIFIC_PROOF_INCOMPLETE_FAIL_CLOSED"
    else:
        failure_phase = "INDEPENDENT_TIMESTEP_AND_MAPPING_GATES_FAIL_CLOSED"

    payload: dict[str, Any] = {
        "schema_version": 1,
        "stage": "05B_h3_fixed_schedule_inference",
        "pipeline_id": PIPELINE_ID,
        "decision": "FAIL",
        "failure_phase": failure_phase,
        "scientific_gate_evaluated": scientific_gate_evaluated,
        "scientific_proofs_complete": scientific_proofs_complete,
        "gates": {
            "frozen_input_integrity": integrity_gate,
            "production_code_state": code_state_gate,
            "frozen_grid_lookup": grid_gate,
            "production_timestep_consumption": timestep_gate,
            "mapping_identifiability": mapping_gate,
        },
        "frozen_grid_lookup_available": grid_ok,
        "production_timestep_consumption_defined": False,
        "mapping_identifiable": False,
        "h3_fixed_schedule_inference_admissible": False,
        "next_stage_allowed": False,
        "model_evaluation_allowed": False,
        "production_activation_allowed": False,
        "implementation_performed": False,
        "conservative_substitute_implemented": False,
        "existing_model_modified": False,
        "blocked_actions": [
            "off_grid_timestep_resolver_without_frozen_contract",
            "recoverability_to_native_shallow_null_mapping_implementation",
            "model_evaluation",
            "production_activation",
        ],
        "conclusion": (
            "FAIL: exact H3 frozen-grid lookup is available, but production "
            "uses off-grid timesteps without a frozen resolver, and the scalar "
            "still does not uniquely identify a [native, shallow, null] route "
            "distribution."
            if scientific_gate_evaluated
            else (
                "FAIL CLOSED: the scientific mapping gate was not evaluated "
                "because frozen inputs are missing, invalid, or incomplete."
            )
        ),
        "stop_rule": (
            "Do not invent nearest, interpolation, global-mean fallback, or a "
            "conservative route mapping. Do not modify/evaluate the model or "
            "activate production until both contracts are separately justified "
            "and frozen."
        ),
        "artifacts": {
            "config": _display_path(config_path),
            "config_file_sha256": _safe_hash(config_path)[0],
            "bundle": _display_path(bundle_path),
            "bundle_file_sha256": bundle_gate["observed_file_sha256"],
            "contract": _display_path(contract_path),
            "contract_file_sha256": contract_gate["observed_file_sha256"],
            "model_source": _display_path(model_source_path),
            "model_source_file_sha256": model_source_gate[
                "observed_file_sha256"
            ],
            "router_source": _display_path(router_source_path),
            "router_source_file_sha256": router_source_gate[
                "observed_file_sha256"
            ],
            "production_v5_config": _display_path(production_config_path),
            "production_v5_config_file_sha256": production_config_gate[
                "observed_file_sha256"
            ],
        },
        "created_at_utc": _utc_now(),
    }
    payload["decision_sha256"] = canonical_json_sha256(payload)
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--contract", type=Path, default=None)
    parser.add_argument("--model-source", type=Path, default=None)
    parser.add_argument("--router-source", type=Path, default=None)
    parser.add_argument("--production-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-scientific-fail-exit-zero",
        action="store_true",
        help=(
            "Return zero only for the expected, fully evaluated scientific "
            "mapping FAIL; integrity failures still return 2."
        ),
    )
    args = parser.parse_args(argv)
    decision = build_decision(
        config_path=args.config,
        bundle_path=args.bundle,
        contract_path=args.contract,
        model_source_path=args.model_source,
        router_source_path=args.router_source,
        production_config_path=args.production_config,
    )
    output = args.output_dir.resolve() / "decision.json"
    _write_json_atomic(output, decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(f"Wrote: {output}")
    if (
        args.allow_scientific_fail_exit_zero
        and decision["scientific_gate_evaluated"] is True
        and decision["scientific_proofs_complete"] is True
        and decision["gates"]["frozen_input_integrity"]["decision"] == "PASS"
        and decision["gates"]["frozen_grid_lookup"]["decision"] == "PASS"
        and decision["gates"]["production_timestep_consumption"].get(
            "undefined_consumption_proven"
        )
        is True
        and decision["gates"]["mapping_identifiability"].get(
            "undefined_mapping_proven"
        )
        is True
        and decision["failure_phase"]
        == "INDEPENDENT_TIMESTEP_AND_MAPPING_GATES_FAIL_CLOSED"
    ):
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
