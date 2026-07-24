"""Build the isolated, matched H3-v2 model-experiment configurations.

This builder is deliberately non-executing: it verifies the preregistered
protocol and the completed calibration, derives the locked 99/25/31 manifest,
and writes two run-owned YAML configurations plus an experiment plan.  It
never starts training, evaluates a model, or writes into legacy checkpoint or
evidence directories.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from src.data.lineage import (  # noqa: E402
    REQUIRED_CHECKPOINT_LINEAGE_FIELDS,
    load_checkpoint_data_lineage,
)
from src.mechanism_validation.common import (  # noqa: E402
    MANIFEST_COLUMNS,
    canonical_json_sha256,
    file_sha256,
    partition_counts,
    partition_sha256,
    patient_partition,
    read_manifest,
    write_mechanism_manifest,
)
from src.model.frequency.h3_native_null_schedule import (  # noqa: E402
    H3_V2_PIPELINE_ID,
    load_h3_native_null_schedule,
)


SCHEMA_VERSION = 1
STAGE = "05C_h3_v2_full_timestep_native_null"
PLAN_PIPELINE_ID = "H3_V2_MATCHED_MODEL_EXPERIMENT_PLAN_V1"
PROTOCOL_STATUS = (
    "PREREGISTERED_BEFORE_H3_V2_CALIBRATION_OR_MODEL_EVALUATION"
)
BASE_CONFIG_RELATIVE = Path(
    "configs/experiments/slmf_png_spectral_router_v5.yaml"
)
EXPECTED_BASE_CONFIG_SHA256 = (
    "42f77c7c9460ecba596ac3ca16bc3d234af8d437611fa68febf5d99d1e74f59d"
)
VARIANTS = ("no_route", "h3_v2_native_null")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _resolve(root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _require_safe_id(value: str, label: str) -> str:
    current = str(value).strip()
    if (
        not SAFE_ID.fullmatch(current)
        or current in {".", ".."}
        or ".." in current
    ):
        raise ValueError(
            f"{label} must contain only a conservative identifier alphabet"
        )
    return current


def _require_run_owned_output(root: Path, output: Path) -> None:
    if output == root:
        raise ValueError("output_dir cannot be the repository root")
    protected = (
        root / "cache",
        root / "checkpoints",
        root / "configs",
        root / "main_data",
        root / "Data",
        root / "results" / "mechanism_validation",
    )
    for protected_root in protected:
        try:
            output.relative_to(protected_root.resolve())
        except ValueError:
            continue
        raise ValueError(
            f"output_dir overlaps a protected legacy/input tree: {protected_root}"
        )
    mechanism_v2 = (
        root / "results" / "mechanism_validation_v2"
    ).resolve()
    try:
        relative = output.relative_to(mechanism_v2)
    except ValueError:
        return
    if not relative.parts or relative.parts[0] != STAGE:
        raise ValueError(
            "H3-v2 output under results/mechanism_validation_v2 must stay "
            f"inside {STAGE}"
        )


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be a JSON object: {path}")
    return payload


def _config_self_hash(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("config_sha256", None)
    return canonical_json_sha256(body)


def _schedule_self_hash(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("schedule_sha256", None)
    return canonical_json_sha256(body)


def _plan_self_hash(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("plan_sha256", None)
    return canonical_json_sha256(body)


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value).lower()
    if not SHA256.fullmatch(digest):
        raise ValueError(f"{label} must be a frozen lowercase SHA-256 digest")
    return digest


def _validate_runtime_sources(
    root: Path, protocol: Mapping[str, Any]
) -> dict[str, str]:
    specifications = protocol.get("runtime_sources")
    if not isinstance(specifications, Mapping) or not specifications:
        raise ValueError("Protocol runtime_sources must be a non-empty object")
    identities: dict[str, str] = {}
    for label, specification in sorted(specifications.items()):
        if not isinstance(specification, Mapping):
            raise ValueError(f"runtime_sources.{label} must be an object")
        source = _resolve(root, str(specification.get("path", "")))
        expected = _require_sha256(
            specification.get("file_sha256"),
            f"runtime_sources.{label}.file_sha256",
        )
        if not source.is_file():
            raise FileNotFoundError(f"Frozen runtime source not found: {source}")
        observed = file_sha256(source)
        if observed != expected:
            raise ValueError(
                f"runtime_sources.{label} SHA-256 mismatch: "
                f"expected={expected}, observed={observed}"
            )
        try:
            # Match the calibration worker's platform-native relative-key
            # canonicalization exactly; the decision stores this dictionary.
            key = str(source.relative_to(root))
        except ValueError:
            key = str(source)
        identities[key] = observed
    return identities


def _validate_protocol(
    root: Path, path: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    protocol = _read_object(path, "H3-v2 protocol")
    declared = _require_sha256(
        protocol.get("config_sha256"), "protocol.config_sha256"
    )
    computed = _config_self_hash(protocol)
    if declared != computed:
        raise ValueError(
            "H3-v2 protocol self-hash mismatch: "
            f"declared={declared}, computed={computed}"
        )
    if (
        protocol.get("schema_version") != SCHEMA_VERSION
        or protocol.get("stage") != STAGE
        or protocol.get("pipeline_id") != H3_V2_PIPELINE_ID
        or protocol.get("status") != PROTOCOL_STATUS
    ):
        raise ValueError("Unexpected or unsealed H3-v2 protocol identity")
    historical = protocol.get("historical_results_preserved", {})
    if not isinstance(historical, Mapping) or historical.get(
        "must_not_overwrite_or_relabel"
    ) is not True:
        raise ValueError("Protocol does not preserve the historical H3/H4 results")
    rules = protocol.get("stop_and_claim_rules", {})
    required_false = (
        "production_training",
        "production_activation",
        "h5_started",
        "h6_started",
        "next_production_stage_allowed",
    )
    if not isinstance(rules, Mapping) or any(
        rules.get(field) is not False for field in required_false
    ):
        raise ValueError("Protocol stop-and-claim rules are not fail-closed")
    experiment = protocol.get("exploratory_model_experiment", {})
    if (
        not isinstance(experiment, Mapping)
        or experiment.get("runs_only_after_calibration_pass") is not True
        or list(experiment.get("comparators", ()))
        != ["no_route", "h3_v2_native_null"]
        or experiment.get("same_epoch_budget") is not True
        or experiment.get("exact_final_epoch_checkpoint") is not True
    ):
        raise ValueError("Protocol does not declare the matched H3-v2 experiment")
    return protocol, _validate_runtime_sources(root, protocol)


def _validate_calibration_decision(
    path: Path,
    protocol: Mapping[str, Any],
    runtime_source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    decision = _read_object(path, "H3-v2 calibration decision")
    decision_body = dict(decision)
    declared_decision_hash = _require_sha256(
        decision_body.pop("decision_sha256", None),
        "calibration_decision.decision_sha256",
    )
    if declared_decision_hash != canonical_json_sha256(decision_body):
        raise ValueError("H3-v2 calibration decision self-hash mismatch")
    if (
        decision.get("schema_version") != SCHEMA_VERSION
        or decision.get("stage") != STAGE
        or decision.get("pipeline_id") != H3_V2_PIPELINE_ID
        or decision.get("decision") != "PASS"
        or decision.get("calibration_complete") is not True
    ):
        raise ValueError("H3-v2 calibration decision is not the completed PASS")
    if decision.get("config_sha256") != protocol.get("config_sha256"):
        raise ValueError("Calibration decision used a different protocol identity")
    if decision.get("runtime_source_hashes") != dict(runtime_source_hashes):
        raise ValueError("Calibration decision runtime-source identity mismatch")
    dataset = protocol["dataset_contract"]
    if (
        decision.get("mechanism_partition_sha256")
        != dataset["mechanism_partition_sha256"]
        or decision.get("validation_images_read") is not False
        or decision.get("model_experiment_allowed") is not True
        or decision.get("production_training_allowed") is not False
        or decision.get("production_activation_allowed") is not False
        or decision.get("h5_h6_allowed") is not False
        or decision.get("next_production_stage_allowed") is not False
    ):
        raise ValueError("Calibration decision violates H3-v2 isolation rules")
    mapping = decision.get("mapping", {})
    if (
        not isinstance(mapping, Mapping)
        or mapping.get("route_order") != ["native", "shallow", "null"]
        or mapping.get("shallow_route_structurally_zero") is not True
        or int(mapping.get("scalar_degrees_of_freedom", -1)) != 1
        or int(mapping.get("route_degrees_of_freedom_used", -1)) != 1
    ):
        raise ValueError("Calibration decision does not preserve native/null mapping")
    gate = decision.get("gate", {})
    checks = gate.get("checks", {}) if isinstance(gate, Mapping) else {}
    if (
        not isinstance(gate, Mapping)
        or gate.get("decision") != "PASS"
        or not isinstance(checks, Mapping)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise ValueError("Calibration decision does not contain an all-PASS gate")
    comparison = gate.get("calibration_schedule_vs_band_constant", {})
    comparison_spec = protocol["calibration_gates"][
        "calibration_schedule_vs_band_constant"
    ]
    if not isinstance(comparison, Mapping):
        raise ValueError("Calibration comparison must be an object")
    comparison_ci_high = float(comparison.get("ci95_high", float("inf")))
    improved_fraction = float(
        comparison.get("improved_patient_fraction", -1.0)
    )
    if (
        not math.isfinite(comparison_ci_high)
        or not math.isfinite(improved_fraction)
        or comparison_ci_high
        >= float(comparison_spec["required_bootstrap_ci95_high_below"])
        or improved_fraction
        < float(comparison_spec["minimum_improved_patient_fraction"])
        or len(comparison.get("patient_rows", ()))
        != int(protocol["dataset_contract"]["calibration_patients"])
    ):
        raise ValueError("Calibration comparison no longer satisfies its frozen gate")
    direct_grid = decision.get("direct_grid", {})
    timesteps = int(protocol["calibration"]["num_train_timesteps"])
    bands = len(protocol["calibration"]["band_order"])
    if (
        not isinstance(direct_grid, Mapping)
        or int(direct_grid.get("minimum", -1)) != 0
        or int(direct_grid.get("maximum", -1)) != timesteps - 1
        or int(direct_grid.get("count", -1)) != timesteps
        or int(direct_grid.get("bands", -1)) != bands
        or int(direct_grid.get("cells", -1)) != timesteps * bands
        or direct_grid.get("interpolation_used") is not False
        or direct_grid.get("nearest_neighbor_used") is not False
        or direct_grid.get("global_mean_fallback_used") is not False
    ):
        raise ValueError("Calibration decision does not prove the full direct grid")
    return decision


def _validate_schedule(
    decision_path: Path,
    decision: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], str]:
    raw_path = decision.get("schedule_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("PASS calibration decision lacks schedule_path")
    schedule_path = Path(raw_path)
    if not schedule_path.is_absolute():
        schedule_path = (decision_path.parent / schedule_path).resolve()
    else:
        schedule_path = schedule_path.resolve()
    expected_local = (decision_path.parent / "frozen_schedule.json").resolve()
    if schedule_path != expected_local:
        raise ValueError(
            "Calibration schedule must be the run-owned frozen_schedule.json"
        )
    expected_file_hash = _require_sha256(
        decision.get("schedule_file_sha256"),
        "calibration_decision.schedule_file_sha256",
    )
    observed_file_hash = file_sha256(schedule_path)
    if observed_file_hash != expected_file_hash:
        raise ValueError(
            "Frozen schedule file hash mismatch: "
            f"expected={expected_file_hash}, observed={observed_file_hash}"
        )
    schedule = _read_object(schedule_path, "H3-v2 frozen schedule")
    declared_self_hash = _require_sha256(
        schedule.get("schedule_sha256"), "schedule.schedule_sha256"
    )
    if declared_self_hash != _schedule_self_hash(schedule):
        raise ValueError("Frozen schedule canonical self-hash mismatch")
    if (
        declared_self_hash != decision.get("schedule_sha256")
        or schedule.get("config_sha256") != protocol.get("config_sha256")
    ):
        raise ValueError("Schedule and calibration decision identities disagree")
    timesteps = int(protocol["calibration"]["num_train_timesteps"])
    active_mass, metadata = load_h3_native_null_schedule(
        schedule_path,
        expected_file_sha256=expected_file_hash,
        expected_num_train_timesteps=timesteps,
    )
    gates = protocol["calibration_gates"]
    if (
        float(active_mass[:, :, 0].min())
        < float(gates["minimum_active_mass_at_t0"]) - 1e-7
        or float(active_mass[:, :, -1].max())
        > float(gates["maximum_active_mass_at_t999"]) + 1e-7
    ):
        raise ValueError("Frozen schedule no longer satisfies endpoint gates")
    return schedule_path, metadata, observed_file_hash


def _manifest_semantic_sha256(
    rows: Iterable[Mapping[str, str]],
) -> str:
    canonical_rows = [
        {column: row.get(column, "") for column in MANIFEST_COLUMNS}
        for row in sorted(rows, key=lambda item: item.get("sample_id", ""))
    ]
    return canonical_json_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "columns": list(MANIFEST_COLUMNS),
            "rows": canonical_rows,
        }
    )


def _validate_source_data(
    root: Path,
    protocol: Mapping[str, Any],
) -> tuple[
    list[dict[str, str]],
    dict[str, str],
    dict[str, dict[str, int]],
    dict[str, Any],
    dict[str, Any],
]:
    dataset = protocol["dataset_contract"]
    manifest_path = _resolve(root, dataset["manifest_path"])
    if file_sha256(manifest_path) != dataset["manifest_file_sha256"]:
        raise ValueError("Authoritative manifest file SHA-256 mismatch")
    rows = read_manifest(manifest_path)
    if len(rows) != int(dataset["samples"]):
        raise ValueError("Source manifest sample count differs from protocol")
    patients = {row["patient_id"] for row in rows}
    if len(patients) != int(dataset["patients"]):
        raise ValueError("Source manifest patient count differs from protocol")
    semantic = _manifest_semantic_sha256(rows)
    if semantic != dataset["manifest_semantic_sha256"]:
        raise ValueError("Source manifest semantic SHA-256 mismatch")

    contract_path = _resolve(root, dataset["dataset_contract_path"])
    contract = _read_object(contract_path, "dataset contract")
    contract_body = dict(contract)
    declared_contract_hash = _require_sha256(
        contract_body.pop("contract_sha256", None),
        "dataset_contract.contract_sha256",
    )
    if declared_contract_hash != canonical_json_sha256(contract_body):
        raise ValueError("Dataset contract canonical self-hash mismatch")
    if declared_contract_hash != dataset["dataset_contract_sha256"]:
        raise ValueError("Protocol and dataset-contract identities disagree")
    if (
        contract.get("manifest", {}).get("semantic_sha256") != semantic
        or contract.get("manifest", {}).get("source_file_sha256")
        != file_sha256(manifest_path)
    ):
        raise ValueError("Dataset contract does not identify the source manifest")

    lineage_path = _resolve(root, dataset["cache_lineage_path"])
    cache_dir = _resolve(root, dataset["cache_dir"])
    lineage = load_checkpoint_data_lineage(
        {
            "data": {
                "cache_dir": str(cache_dir),
                "cache_lineage": str(lineage_path),
                "dataset_contract": str(contract_path),
                "require_cache_lineage": True,
                "use_fake_data": False,
            }
        },
        root=root,
    )
    if lineage is None:
        raise ValueError("Strict cache lineage unexpectedly resolved to None")
    if (
        lineage.get("cache_payload_sha256")
        != dataset["cache_payload_sha256"]
        or lineage.get("cache_metadata_sha256")
        != dataset["cache_metadata_sha256"]
        or lineage.get("manifest_semantic_sha256") != semantic
    ):
        raise ValueError("Protocol and sealed cache-lineage identities disagree")

    partition = patient_partition(rows)
    observed_partition_hash = partition_sha256(partition)
    if observed_partition_hash != dataset["mechanism_partition_sha256"]:
        raise ValueError("Mechanism partition SHA-256 mismatch")
    counts = partition_counts(rows, partition)
    expected_patients = {
        "mechanism_train": int(dataset["mechanism_train_patients"]),
        "calibration": int(dataset["calibration_patients"]),
        "validation": int(dataset["validation_patients"]),
    }
    if any(
        counts[role]["patients"] != expected
        for role, expected in expected_patients.items()
    ):
        raise ValueError(f"Unexpected 99/25/31 partition counts: {counts}")
    source_identity = {
        "manifest": {
            "path": str(manifest_path),
            "file_sha256": file_sha256(manifest_path),
            "semantic_sha256": semantic,
            "samples": len(rows),
            "patients": len(patients),
        },
        "dataset_contract": {
            "path": str(contract_path),
            "file_sha256": file_sha256(contract_path),
            "contract_sha256": declared_contract_hash,
        },
        "cache_lineage": {
            "path": str(lineage_path),
            "file_sha256": file_sha256(lineage_path),
            **{
                field: lineage[field]
                for field in REQUIRED_CHECKPOINT_LINEAGE_FIELDS
            },
        },
    }
    paths = {
        "manifest": str(manifest_path),
        "contract": str(contract_path),
        "cache_dir": str(cache_dir),
        "cache_lineage": str(lineage_path),
    }
    return rows, partition, counts, source_identity, paths


def _validate_mean_checkpoint(
    root: Path,
    protocol: Mapping[str, Any],
    calibration_decision: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    specification = protocol["mean_checkpoint"]
    decision_path = _resolve(root, specification["decision_path"])
    if (
        file_sha256(decision_path)
        != specification["decision_file_sha256"]
    ):
        raise ValueError("Excluded-mean decision file SHA-256 mismatch")
    decision = _read_object(decision_path, "excluded-mean decision")
    if (
        decision.get("pipeline_id") != specification["decision_pipeline_id"]
        or decision.get("decision") != specification["decision_required"]
    ):
        raise ValueError("Excluded-mean upstream decision is not the locked PASS")
    checkpoint_path = _resolve(root, specification["path"])
    expected_checkpoint_hash = _require_sha256(
        specification["file_sha256"], "mean_checkpoint.file_sha256"
    )
    if (
        not checkpoint_path.is_file()
        or file_sha256(checkpoint_path) != expected_checkpoint_hash
    ):
        raise ValueError("Excluded-mean checkpoint file identity mismatch")
    calibrated_mean = calibration_decision.get("mean_checkpoint", {})
    if (
        not isinstance(calibrated_mean, Mapping)
        or calibrated_mean.get("sha256") != expected_checkpoint_hash
        or calibrated_mean.get("production_cutover_claimed") is not False
    ):
        raise ValueError("Calibration decision used a different mean checkpoint")

    sidecar_path = _resolve(root, specification["sidecar_path"])
    sidecar = _read_object(sidecar_path, "excluded-mean sidecar")
    sidecar_body = dict(sidecar)
    declared_fingerprint = _require_sha256(
        sidecar_body.pop("fingerprint_sha256", None),
        "mean_checkpoint.sidecar.fingerprint_sha256",
    )
    if declared_fingerprint != canonical_json_sha256(sidecar_body):
        raise ValueError("Excluded-mean sidecar canonical self-hash mismatch")
    if (
        declared_fingerprint
        != specification["sidecar_fingerprint_sha256"]
        or sidecar.get("checkpoint_sha256") != expected_checkpoint_hash
    ):
        raise ValueError("Excluded-mean checkpoint and sidecar identities disagree")
    exclusion = sidecar.get("pathology_exclusion", {})
    if (
        not isinstance(exclusion, Mapping)
        or exclusion.get("enabled") is not True
        or int(exclusion.get("guard_radius_px", -1))
        != int(specification["guard_radius_px"])
    ):
        raise ValueError("Excluded-mean pathology guard is not the locked policy")
    return checkpoint_path, {
        "decision_path": str(decision_path),
        "decision_file_sha256": file_sha256(decision_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_file_sha256": expected_checkpoint_hash,
        "sidecar_path": str(sidecar_path),
        "sidecar_file_sha256": file_sha256(sidecar_path),
        "sidecar_fingerprint_sha256": declared_fingerprint,
        "pathology_exclusion_enabled": True,
        "guard_radius_px": int(specification["guard_radius_px"]),
        "production_cutover_claimed": False,
    }


def _load_base_config(root: Path) -> tuple[Path, dict[str, Any], str]:
    path = _resolve(root, BASE_CONFIG_RELATIVE)
    if not path.is_file():
        raise FileNotFoundError(f"Base experiment config not found: {path}")
    observed_hash = file_sha256(path)
    if observed_hash != EXPECTED_BASE_CONFIG_SHA256:
        raise ValueError(
            "Base experiment config SHA-256 mismatch: "
            f"expected={EXPECTED_BASE_CONFIG_SHA256}, observed={observed_hash}"
        )
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load base experiment config: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Base experiment config root must be an object")
    return path, payload, observed_hash


def _set_nested(payload: dict[str, Any], dotted: str, value: Any) -> None:
    cursor = payload
    parts = dotted.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def _prepare_variant_config(
    *,
    base: Mapping[str, Any],
    variant: str,
    run_id: str,
    attempt: str,
    epochs: int,
    training_seed: int,
    evaluation_seed: int,
    evaluation_steps: int,
    training_monitor_samples: int,
    paths: Mapping[str, str],
    mean_checkpoint: Path,
    schedule_path: Path,
    schedule_file_sha256: str,
    checkpoint_dir: Path,
    sample_dir: Path,
    manifest_file_sha256: str,
    partition_digest: str,
    protocol_config_sha256: str,
    calibration_decision_sha256: str,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown H3-v2 experiment variant: {variant}")
    config = copy.deepcopy(dict(base))
    fixed: dict[str, Any] = {
        "experiment.name": f"h3_v2_{run_id}_{attempt}_{variant}",
        "experiment.seed": training_seed,
        "data.mode": "png",
        "data.cache_dir": paths["cache_dir"],
        "data.cache_lineage": paths["cache_lineage"],
        "data.dataset_contract": paths["contract"],
        "data.require_cache_lineage": True,
        "data.use_fake_data": False,
        "data.split_manifest": paths["derived_manifest"],
        "data.augment": True,
        "runtime.device": "cuda",
        "runtime.require_cuda": True,
        "runtime.eval_interval": epochs,
        "runtime.eval_num_samples": training_monitor_samples,
        "runtime.eval_seed": evaluation_seed,
        "runtime.eval_sampling_steps": evaluation_steps,
        "runtime.sample_interval": epochs + 1,
        "runtime.save_interval": epochs,
        "runtime.tracked_sample_ids": [],
        "runtime.best_checkpoint.enabled": False,
        "runtime.early_stopping.enabled": False,
        "runtime.sample_dir": str(sample_dir),
        "training.num_epochs": epochs,
        "training.checkpoint_dir": str(checkpoint_dir),
        "training.init_from": None,
        "training.resume_from": None,
        "modules.conditional_mean.checkpoint": str(mean_checkpoint),
        "modules.conditional_mean.freeze": True,
        "modules.conditional_mean.detach_bridge": True,
        "modules.residual_frequency.enabled": True,
        "modules.residual_frequency.mode": "spectral_evidence_router",
        "modules.residual_frequency.use_noise_release": False,
        "modules.residual_frequency.use_ct_reliability": False,
        "modules.residual_frequency.use_content_reliability": False,
        "modules.residual_frequency.use_subband_gates": False,
        "modules.residual_frequency.use_directional_reliability": False,
        "modules.residual_frequency.use_gabor_gate": False,
        "modules.residual_frequency.use_gabor_agreement": False,
        "modules.residual_frequency.dct_descriptor.enabled": False,
        "modules.residual_frequency.gabor_descriptor.enabled": False,
        "modules.residual_frequency.ct_support_head.enabled": False,
        "modules.residual_frequency.ct_support_head.support_only": True,
        "modules.residual_frequency.cross_level_router.enabled": True,
        "modules.residual_frequency.cross_level_router.fixed_prior": [
            0.0,
            0.0,
            1.0,
        ],
        "modules.residual_frequency.cross_level_router.native_warmup_epochs": 0,
        "modules.residual_frequency.cross_level_router.routing_ramp_epochs": 0,
        "modules.residual_frequency.cross_level_router.uncertainty_aware_enabled": False,
        "modules.residual_frequency.cross_level_router.uncertainty_aware_confidence_threshold": None,
        "modules.gabor.enabled": False,
        "modules.gabor.inject_adapter": False,
        "modules.gabor.use_for_noise": False,
        "modules.gabor.use_for_hotspot": False,
        "modules.gabor.use_for_loss": False,
        "losses.spectral_router_regularization.enabled": False,
    }
    if variant == "no_route":
        fixed.update(
            {
                "modules.residual_frequency.cross_level_router.hard_all_null": True,
                "modules.residual_frequency.cross_level_router.policy": "fixed_prior",
            }
        )
    else:
        fixed.update(
            {
                "modules.residual_frequency.cross_level_router.hard_all_null": False,
                "modules.residual_frequency.cross_level_router.policy": "h3_native_null",
                "modules.residual_frequency.cross_level_router.h3_schedule_path": str(
                    schedule_path
                ),
                "modules.residual_frequency.cross_level_router.h3_schedule_sha256": (
                    schedule_file_sha256
                ),
            }
        )
    for dotted, value in fixed.items():
        _set_nested(config, dotted, value)
    router = config["modules"]["residual_frequency"]["cross_level_router"]
    if variant == "no_route":
        router.pop("h3_schedule_path", None)
        router.pop("h3_schedule_sha256", None)

    config["formal_h3_v2_experiment"] = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "pipeline_id": PLAN_PIPELINE_ID,
        "scope": "internal_exploratory_only_on_previously_exposed_dataset",
        "run_id": run_id,
        "attempt": attempt,
        "variant": variant,
        "training_partition": "mechanism_train_99_patients_only",
        "checkpoint_selection_partition": "calibration_25_patients_only",
        "calibration_partition": "val_25_patients_final_endpoint_monitor_only",
        "validation_partition": "test_31_patients_post_hoc_only",
        "training_seed": training_seed,
        "fixed_endpoint_epochs": epochs,
        "same_epoch_budget": True,
        "checkpoint_policy": "exact_final_epoch_no_best_checkpoint_selection",
        "validation_drives_training_or_course": False,
        "derived_manifest_file_sha256": manifest_file_sha256,
        "mechanism_partition_sha256": partition_digest,
        "protocol_config_sha256": protocol_config_sha256,
        "calibration_decision_file_sha256": calibration_decision_sha256,
        "schedule_file_sha256": (
            schedule_file_sha256
            if variant == "h3_v2_native_null"
            else None
        ),
        "learned_route_enabled": False,
        "uncertainty_aware_or_h4_enabled": False,
        "production_training": False,
        "production_activation": False,
        "h5_h6_started": False,
    }
    return config


def _yaml_bytes(payload: Mapping[str, Any]) -> bytes:
    rendered = yaml.safe_dump(
        dict(payload),
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    return rendered.encode("utf-8")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
        )
        + "\n"
    ).encode("utf-8")


def _commit_new_or_identical(path: Path, content: bytes, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file():
            raise FileExistsError(f"{label} target is not a file: {path}")
        observed = path.read_bytes()
        if observed != content:
            raise FileExistsError(
                f"Refusing to overwrite a different existing {label}: {path}"
            )
        return
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != content:
            raise FileExistsError(
                f"Concurrent different {label} appeared at {path}"
            )


def _materialize_manifest(
    *,
    attempt_dir: Path,
    rows: list[dict[str, str]],
    partition: Mapping[str, str],
) -> tuple[Path, dict[str, Any]]:
    target = attempt_dir / "data" / "mechanism_split_manifest.csv"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".manifest_candidate_",
        dir=attempt_dir,
    ) as temporary:
        candidate = Path(temporary) / "mechanism_split_manifest.csv"
        metadata = write_mechanism_manifest(rows, partition, candidate)
        content = candidate.read_bytes()
    _commit_new_or_identical(target, content, "derived mechanism manifest")
    metadata = dict(metadata)
    metadata.update(
        {
            "path": str(target),
            "file_sha256": file_sha256(target),
            "sha256": file_sha256(target),
        }
    )
    return target, metadata


def build(args: argparse.Namespace) -> Path:
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Repository root not found: {root}")
    run_id = _require_safe_id(args.run_id, "run_id")
    attempt = _require_safe_id(args.attempt, "attempt")
    output_root = _resolve(root, args.output_dir)
    try:
        output_root.relative_to(root)
    except ValueError as exc:
        raise ValueError("output_dir must stay inside the repository root") from exc
    _require_run_owned_output(root, output_root)
    attempt_dir = output_root / "attempts" / attempt

    protocol_path = _resolve(root, args.protocol_config)
    protocol, runtime_source_hashes = _validate_protocol(root, protocol_path)
    calibration_decision_path = _resolve(root, args.calibration_decision)
    calibration_decision = _validate_calibration_decision(
        calibration_decision_path,
        protocol,
        runtime_source_hashes,
    )
    schedule_path, schedule_metadata, schedule_file_hash = _validate_schedule(
        calibration_decision_path,
        calibration_decision,
        protocol,
    )
    rows, partition, counts, data_identity, data_paths = _validate_source_data(
        root, protocol
    )
    if calibration_decision.get("partition_counts") != counts:
        raise ValueError("Calibration decision and live partition counts disagree")
    mean_checkpoint, mean_identity = _validate_mean_checkpoint(
        root,
        protocol,
        calibration_decision,
    )
    base_config_path, base_config, base_config_hash = _load_base_config(root)

    experiment_spec = protocol["exploratory_model_experiment"]
    epochs = (
        int(args.epochs)
        if args.epochs is not None
        else int(experiment_spec["default_epochs"])
    )
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    training_seed = int(experiment_spec["same_initialization_seed"])
    evaluation_seed = int(experiment_spec["evaluation_seed"])
    evaluation_steps = int(experiment_spec["evaluation_sampling_steps"])
    training_monitor_samples = int(
        experiment_spec["training_monitor_samples"]
    )
    if not 1 <= training_monitor_samples <= 16:
        raise ValueError("training_monitor_samples must stay in [1,16]")

    derived_manifest, manifest_metadata = _materialize_manifest(
        attempt_dir=attempt_dir,
        rows=rows,
        partition=partition,
    )
    if (
        manifest_metadata["partition_sha256"]
        != protocol["dataset_contract"]["mechanism_partition_sha256"]
        or manifest_metadata["counts"] != counts
    ):
        raise ValueError("Derived manifest metadata differs from locked partition")
    data_paths = {
        **data_paths,
        "derived_manifest": str(derived_manifest),
    }

    calibration_decision_hash = file_sha256(calibration_decision_path)
    protocol_file_hash = file_sha256(protocol_path)
    variant_plan: dict[str, Any] = {}
    configs: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        checkpoint_dir = attempt_dir / "checkpoints" / variant
        sample_dir = attempt_dir / "samples" / variant
        config_path = attempt_dir / "configs" / f"{variant}.yaml"
        checkpoint_path = checkpoint_dir / f"ckpt_epoch{epochs:04d}.pt"
        evaluation_path = (
            attempt_dir / "evaluations" / variant / "validation.json"
        )
        config = _prepare_variant_config(
            base=base_config,
            variant=variant,
            run_id=run_id,
            attempt=attempt,
            epochs=epochs,
            training_seed=training_seed,
            evaluation_seed=evaluation_seed,
            evaluation_steps=evaluation_steps,
            training_monitor_samples=training_monitor_samples,
            paths=data_paths,
            mean_checkpoint=mean_checkpoint,
            schedule_path=schedule_path,
            schedule_file_sha256=schedule_file_hash,
            checkpoint_dir=checkpoint_dir,
            sample_dir=sample_dir,
            manifest_file_sha256=manifest_metadata["file_sha256"],
            partition_digest=manifest_metadata["partition_sha256"],
            protocol_config_sha256=protocol["config_sha256"],
            calibration_decision_sha256=calibration_decision_hash,
        )
        content = _yaml_bytes(config)
        _commit_new_or_identical(config_path, content, f"{variant} config")
        observed_config_hash = file_sha256(config_path)
        configs[variant] = config
        variant_plan[variant] = {
            "variant": variant,
            "config_path": str(config_path),
            "config_sha256": observed_config_hash,
            "config_canonical_sha256": canonical_json_sha256(config),
            "checkpoint_dir": str(checkpoint_dir),
            "sample_dir": str(sample_dir),
            "checkpoint_path": str(checkpoint_path),
            "eval_output_path": str(evaluation_path),
            "evaluation_provenance_path": str(
                evaluation_path.with_name("validation_provenance.json")
            ),
            "train_log_path": str(
                attempt_dir / "logs" / f"train_{variant}.log"
            ),
            "eval_log_path": str(
                attempt_dir / "logs" / f"evaluate_{variant}.log"
            ),
            "route_policy": (
                "fixed_prior"
                if variant == "no_route"
                else "h3_native_null"
            ),
            "hard_all_null": variant == "no_route",
            "schedule_file_sha256": (
                schedule_file_hash
                if variant == "h3_v2_native_null"
                else None
            ),
            "expected_final_epoch": epochs,
            "checkpoint_weights_for_evaluation": "ema",
            "evaluation_split": "test",
            "evaluation_full_split": True,
        }

    # Fail closed on the intended matched design.  Apart from identity/output
    # fields and the explicit route intervention, both arms must remain equal.
    no_route_audit = copy.deepcopy(configs["no_route"])
    h3_audit = copy.deepcopy(configs["h3_v2_native_null"])
    for current in (no_route_audit, h3_audit):
        current["experiment"]["name"] = "<variant>"
        current["training"]["checkpoint_dir"] = "<variant>"
        current["runtime"]["sample_dir"] = "<variant>"
        formal = current["formal_h3_v2_experiment"]
        formal["variant"] = "<variant>"
        formal["schedule_file_sha256"] = "<route-intervention>"
        router = current["modules"]["residual_frequency"]["cross_level_router"]
        router["hard_all_null"] = "<route-intervention>"
        router["policy"] = "<route-intervention>"
        router["h3_schedule_path"] = "<route-intervention>"
        router["h3_schedule_sha256"] = "<route-intervention>"
    if no_route_audit != h3_audit:
        raise ValueError("Generated experiment arms differ outside the route intervention")

    derived_manifest_payload = {
        "path": str(derived_manifest),
        "file_sha256": manifest_metadata["file_sha256"],
        "partition_sha256": manifest_metadata["partition_sha256"],
        "counts": counts,
        "mapping": manifest_metadata["mapping"],
        "training_split": "train=mechanism_train_99",
        "calibration_split": "val=calibration_25",
        "evaluation_split": "test=validation_31_previously_exposed",
    }
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "pipeline_id": PLAN_PIPELINE_ID,
        "execution_status": "PLANNED_NOT_STARTED",
        "scope": "isolated_internal_exploratory_model_experiment",
        "run_id": run_id,
        "attempt": attempt,
        "attempt_dir": str(attempt_dir),
        "protocol_config_sha256": protocol["config_sha256"],
        "protocol_config": {
            "path": str(protocol_path),
            "file_sha256": protocol_file_hash,
            "config_sha256": protocol["config_sha256"],
        },
        "calibration_decision_path": str(calibration_decision_path),
        "calibration_decision_sha256": calibration_decision_hash,
        "calibration_decision": {
            "path": str(calibration_decision_path),
            "file_sha256": calibration_decision_hash,
            "pipeline_id": calibration_decision["pipeline_id"],
            "decision": "PASS",
            "calibration_complete": True,
        },
        "schedule_path": str(schedule_path),
        "schedule_file_sha256": schedule_file_hash,
        "schedule_sha256": schedule_metadata["schedule_sha256"],
        "schedule": {
            "path": str(schedule_path),
            "file_sha256": schedule_file_hash,
            "schedule_sha256": schedule_metadata["schedule_sha256"],
            "num_train_timesteps": schedule_metadata[
                "num_train_timesteps"
            ],
            "route_order": schedule_metadata["route_order"],
            "band_order": schedule_metadata["band_order"],
            "mapping": "[native,shallow,null]=[active_mass,0,1-active_mass]",
        },
        "derived_manifest": derived_manifest_payload,
        "budget": {
            "training_seed": training_seed,
            "epochs_per_variant": epochs,
            "same_epoch_budget": True,
            "same_initialization_seed": True,
            "checkpoint_selection_partition": "calibration_25_patients_only",
            "calibration_final_endpoint_monitor_only": True,
            "best_checkpoint_selection_enabled": False,
            "exact_final_epoch_checkpoint": True,
            "evaluation_seed": evaluation_seed,
            "evaluation_sampling_steps": evaluation_steps,
            "evaluation_partition": "validation_31_previously_exposed",
        },
        "variants": variant_plan,
        "source_identity": {
            **data_identity,
            "base_config": {
                "path": str(base_config_path),
                "file_sha256": base_config_hash,
            },
            "mean_checkpoint": mean_identity,
            "runtime_source_hashes": runtime_source_hashes,
        },
        "stop_and_claim_rules": {
            "development_model_training_allowed": True,
            "production_training": False,
            "production_activation": False,
            "h5_started": False,
            "h6_started": False,
            "next_production_stage_allowed": False,
            "independent_validation_claim": False,
            "confirmatory_evidence_claim": False,
        },
    }
    plan["plan_sha256"] = _plan_self_hash(plan)
    plan_path = attempt_dir / "experiment_plan.json"
    _commit_new_or_identical(plan_path, _json_bytes(plan), "experiment plan")
    observed_plan = _read_object(plan_path, "experiment plan")
    if observed_plan.get("plan_sha256") != _plan_self_hash(observed_plan):
        raise ValueError("Written experiment plan failed self-hash verification")
    return plan_path


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SCRIPT_ROOT)
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=Path("configs/h3_v2_full_timestep_native_null_v1.json"),
    )
    parser.add_argument("--calibration-decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--attempt", default="attempt_01")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        plan_path = build(args)
    except Exception as exc:
        print(
            f"H3-v2 experiment-config build FAILED CLOSED: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 14
    print(
        json.dumps(
            {
                "status": "PLANNED_NOT_STARTED",
                "experiment_plan_path": str(plan_path),
                "experiment_plan_sha256": file_sha256(plan_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
