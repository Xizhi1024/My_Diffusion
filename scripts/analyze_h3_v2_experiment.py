"""Analyze the isolated H3-v2 no-route versus native/null experiment.

The analysis unit is the patient.  This script validates the preregistered
protocol, calibration PASS, frozen schedule, experiment plan, derived
mechanism manifest, both final checkpoints, and both evaluation artifacts
before it computes any contrast.  The exposed 31-patient validation cohort is
used once and must be present in full in both arms.

This is an internal exploratory decision only.  Neither SUPPORT nor
NO_SUPPORT authorizes production, H5, or H6, and neither changes the historical
05B fixed-schedule FAIL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mechanism_validation.common import (  # noqa: E402
    MANIFEST_COLUMNS,
    canonical_json_sha256,
    file_sha256,
    partition_counts,
    partition_sha256,
    patient_partition,
    read_manifest,
)


SCHEMA_VERSION = 1
PIPELINE_ID = "H3_V2_NATIVE_NULL_MODEL_EXPERIMENT_ANALYSIS_V1"
CALIBRATION_PIPELINE_ID = "H3_V2_FULL_TIMESTEP_NATIVE_NULL_CALIBRATION_V1"
EVALUATION_PROVENANCE_PIPELINE_ID = (
    "H3_V2_MATCHED_MODEL_EVALUATION_PROVENANCE_V1"
)
EXPERIMENT_PLAN_PIPELINE_ID = "H3_V2_MATCHED_MODEL_EXPERIMENT_PLAN_V1"
VARIANT_REFERENCE = "no_route"
VARIANT_CANDIDATE = "h3_v2_native_null"
HARD_PATIENTS = ("002", "022", "044", "080", "153")
DECISION_FILENAME = "decision.json"
PATIENT_CSV_FILENAME = "patient_differences.csv"


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON constant is forbidden: {value}")
            ),
        )
    except OSError as exc:
        raise ValueError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _require_mapping(
    payload: Mapping[str, Any], key: str, context: str
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}.{key} must be an object")
    return value


def _require_text(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value


def _require_int(payload: Mapping[str, Any], key: str, context: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context}.{key} must be an integer")
    return value


def _require_finite_number(
    payload: Mapping[str, Any], key: str, context: str
) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}.{key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context}.{key} must be finite")
    return number


def _require_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{context} must be a lowercase 64-hex SHA-256")
    return normalized


def _assert_file_hash(path: Path, expected: Any, context: str) -> str:
    expected_hash = _require_sha256(expected, f"{context}.file_sha256")
    if not path.is_file():
        raise ValueError(f"{context} is missing: {path}")
    observed = file_sha256(path)
    if observed != expected_hash:
        raise ValueError(
            f"{context} SHA-256 mismatch: expected {expected_hash}, "
            f"observed {observed} ({path})"
        )
    return observed


def _canonical_self_hash(payload: Mapping[str, Any], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return canonical_json_sha256(body)


def _validate_config(
    root: Path, config_path: Path
) -> tuple[dict[str, Any], str, dict[str, str]]:
    config = _load_json(config_path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported H3-v2 protocol schema")
    if config.get("pipeline_id") != CALIBRATION_PIPELINE_ID:
        raise ValueError("Unexpected H3-v2 protocol pipeline_id")
    declared = _require_sha256(
        config.get("config_sha256"), "protocol.config_sha256"
    )
    observed_self = _canonical_self_hash(config, "config_sha256")
    if declared != observed_self:
        raise ValueError(
            "Protocol config self-hash mismatch: "
            f"declared {declared}, computed {observed_self}"
        )

    source_hashes: dict[str, str] = {}
    runtime_sources = _require_mapping(config, "runtime_sources", "protocol")
    for label, raw_spec in runtime_sources.items():
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"protocol.runtime_sources.{label} must be an object")
        source_path = _resolve(
            root,
            _require_text(
                raw_spec, "path", f"protocol.runtime_sources.{label}"
            ),
        )
        try:
            source_key = str(source_path.relative_to(root))
        except ValueError:
            source_key = str(source_path)
        source_hashes[source_key] = _assert_file_hash(
            source_path,
            raw_spec.get("file_sha256"),
            f"protocol.runtime_sources.{label}",
        )

    stop_rules = _require_mapping(config, "stop_and_claim_rules", "protocol")
    for field in (
        "production_training",
        "production_activation",
        "h5_started",
        "h6_started",
        "next_production_stage_allowed",
    ):
        if stop_rules.get(field) is not False:
            raise ValueError(f"protocol.stop_and_claim_rules.{field} must be false")
    return config, declared, source_hashes


def _validate_schedule(
    schedule_path: Path,
    schedule_file_hash: Any,
    schedule_self_hash: Any,
    config_sha256: str,
    expected_timesteps: int,
) -> tuple[dict[str, Any], str, str]:
    observed_file_hash = _assert_file_hash(
        schedule_path, schedule_file_hash, "frozen_schedule"
    )
    schedule = _load_json(schedule_path)
    if schedule.get("pipeline_id") != CALIBRATION_PIPELINE_ID:
        raise ValueError("Frozen schedule pipeline_id mismatch")
    if schedule.get("decision") != "PASS":
        raise ValueError("Frozen schedule is not a PASS artifact")
    if schedule.get("config_sha256") != config_sha256:
        raise ValueError("Frozen schedule config_sha256 mismatch")
    if schedule.get("inference_schedule_allowed") is not True:
        raise ValueError("Frozen schedule does not allow isolated inference")
    if schedule.get("production_activation_allowed") is not False:
        raise ValueError("Frozen schedule unexpectedly allows production")
    if schedule.get("h5_h6_allowed") is not False:
        raise ValueError("Frozen schedule unexpectedly allows H5/H6")
    if schedule.get("num_train_timesteps") != expected_timesteps:
        raise ValueError("Frozen schedule timestep count mismatch")
    if schedule.get("route_order") != ["native", "shallow", "null"]:
        raise ValueError("Frozen schedule route order mismatch")
    if schedule.get("shallow_route_policy") != "structurally_zero":
        raise ValueError("Frozen schedule does not structurally disable shallow")
    declared_self = _require_sha256(
        schedule.get("schedule_sha256"), "frozen_schedule.schedule_sha256"
    )
    expected_self = _require_sha256(
        schedule_self_hash, "calibration.schedule_sha256"
    )
    observed_self = _canonical_self_hash(schedule, "schedule_sha256")
    if declared_self != expected_self or declared_self != observed_self:
        raise ValueError(
            "Frozen schedule self-hash mismatch among schedule, calibration, "
            "and computed content"
        )
    return schedule, observed_file_hash, declared_self


def _validate_calibration(
    root: Path,
    path: Path,
    config: Mapping[str, Any],
    config_sha256: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    decision = _load_json(path)
    decision_body = dict(decision)
    declared_decision_hash = _require_sha256(
        decision_body.pop("decision_sha256", None),
        "calibration.decision_sha256",
    )
    if declared_decision_hash != canonical_json_sha256(decision_body):
        raise ValueError("Calibration decision self-hash mismatch")
    if decision.get("pipeline_id") != CALIBRATION_PIPELINE_ID:
        raise ValueError("Calibration decision pipeline_id mismatch")
    if decision.get("decision") != "PASS":
        raise ValueError("H3-v2 calibration must PASS before model analysis")
    if decision.get("calibration_complete") is not True:
        raise ValueError("H3-v2 calibration is not complete")
    if decision.get("config_sha256") != config_sha256:
        raise ValueError("Calibration decision config_sha256 mismatch")
    if decision.get("model_experiment_allowed") is not True:
        raise ValueError("Calibration decision does not allow model experiment")
    for field in (
        "production_training_allowed",
        "production_activation_allowed",
        "h5_h6_allowed",
        "next_production_stage_allowed",
    ):
        if decision.get(field) is not False:
            raise ValueError(f"Calibration decision {field} must be false")

    declared_config_path = _resolve(
        root, _require_text(decision, "config_path", "calibration")
    )
    expected_config_path = _resolve(
        root, _require_text(config, "_source_path", "protocol")
    )
    if declared_config_path != expected_config_path:
        raise ValueError("Calibration decision points to a different protocol")

    runtime_source_hashes = _require_mapping(
        decision, "runtime_source_hashes", "calibration"
    )
    config_sources = _require_mapping(config, "runtime_sources", "protocol")
    expected_runtime_hashes: dict[str, str] = {}
    for label, raw_spec in config_sources.items():
        if not isinstance(raw_spec, Mapping):
            raise ValueError(
                f"protocol.runtime_sources.{label} must be an object"
            )
        source_path = _resolve(
            root,
            _require_text(
                raw_spec,
                "path",
                f"protocol.runtime_sources.{label}",
            ),
        )
        try:
            source_key = str(source_path.relative_to(root))
        except ValueError:
            source_key = str(source_path)
        expected_runtime_hashes[source_key] = _require_sha256(
            raw_spec.get("file_sha256"),
            f"protocol.runtime_sources.{label}.file_sha256",
        )
    if dict(runtime_source_hashes) != expected_runtime_hashes:
        raise ValueError(
            "Calibration runtime source hashes do not match the frozen protocol"
        )

    checked: dict[str, str] = {}
    mean_checkpoint = _require_mapping(
        decision, "mean_checkpoint", "calibration"
    )
    mean_path = _resolve(
        root, _require_text(mean_checkpoint, "path", "calibration.mean_checkpoint")
    )
    checked["calibration_mean_checkpoint"] = _assert_file_hash(
        mean_path,
        mean_checkpoint.get("sha256"),
        "calibration.mean_checkpoint",
    )
    prepared = _require_mapping(decision, "prepared_inputs", "calibration")
    prepared_path = _resolve(
        root, _require_text(prepared, "path", "calibration.prepared_inputs")
    )
    checked["calibration_prepared_inputs"] = _assert_file_hash(
        prepared_path,
        prepared.get("sha256"),
        "calibration.prepared_inputs",
    )
    return decision, checked


def _validate_plan(
    root: Path,
    path: Path,
    config_path: Path,
    config_file_hash: str,
    config_sha256: str,
    calibration_path: Path,
    calibration_file_hash: str,
    calibration: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    plan = _load_json(path)
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported experiment plan schema")
    if plan.get("pipeline_id") != EXPERIMENT_PLAN_PIPELINE_ID:
        raise ValueError("Unexpected experiment plan pipeline_id")
    if plan.get("execution_status") != "PLANNED_NOT_STARTED":
        raise ValueError("Experiment plan execution_status was mutated")
    declared_self = _require_sha256(
        plan.get("plan_sha256"), "experiment_plan.plan_sha256"
    )
    observed_self = _canonical_self_hash(plan, "plan_sha256")
    if declared_self != observed_self:
        raise ValueError("Experiment plan self-hash mismatch")

    checked: dict[str, str] = {}
    protocol_ref = _require_mapping(
        plan, "protocol_config", "experiment_plan"
    )
    plan_config_path = _resolve(
        root,
        _require_text(
            protocol_ref, "path", "experiment_plan.protocol_config"
        ),
    )
    if plan_config_path != config_path:
        raise ValueError("Experiment plan points to a different protocol config")
    checked["protocol_config"] = _assert_file_hash(
        plan_config_path,
        protocol_ref.get("file_sha256"),
        "experiment_plan.protocol_config",
    )
    if checked["protocol_config"] != config_file_hash:
        raise ValueError("Protocol file changed while validating experiment plan")
    if protocol_ref.get("config_sha256") != config_sha256:
        raise ValueError("Experiment plan protocol semantic hash mismatch")
    if plan.get("protocol_config_sha256") != config_sha256:
        raise ValueError("Experiment plan flat protocol hash alias mismatch")

    calibration_ref = _require_mapping(
        plan, "calibration_decision", "experiment_plan"
    )
    plan_calibration_path = _resolve(
        root,
        _require_text(
            calibration_ref, "path", "experiment_plan.calibration_decision"
        ),
    )
    if plan_calibration_path != calibration_path:
        raise ValueError("Experiment plan points to a different calibration")
    checked["calibration_decision"] = _assert_file_hash(
        plan_calibration_path,
        calibration_ref.get("file_sha256"),
        "experiment_plan.calibration_decision",
    )
    if checked["calibration_decision"] != calibration_file_hash:
        raise ValueError("Calibration changed while validating experiment plan")
    if (
        calibration_ref.get("pipeline_id") != CALIBRATION_PIPELINE_ID
        or calibration_ref.get("decision") != "PASS"
    ):
        raise ValueError("Experiment plan does not bind the calibration PASS")
    if (
        _resolve(
            root,
            _require_text(
                plan,
                "calibration_decision_path",
                "experiment_plan",
            ),
        )
        != plan_calibration_path
        or plan.get("calibration_decision_sha256")
        != calibration_file_hash
    ):
        raise ValueError("Experiment plan flat calibration aliases mismatch")

    schedule_ref = _require_mapping(plan, "schedule", "experiment_plan")
    calibration_schedule_path = _resolve(
        root,
        _require_text(calibration, "schedule_path", "calibration"),
    )
    plan_schedule_path = _resolve(
        root, _require_text(schedule_ref, "path", "experiment_plan.schedule")
    )
    if plan_schedule_path != calibration_schedule_path:
        raise ValueError("Experiment plan points to a different frozen schedule")
    if (
        schedule_ref.get("file_sha256")
        != calibration.get("schedule_file_sha256")
        or schedule_ref.get("schedule_sha256")
        != calibration.get("schedule_sha256")
    ):
        raise ValueError("Experiment plan schedule hashes mismatch calibration")
    if (
        _resolve(
            root, _require_text(plan, "schedule_path", "experiment_plan")
        )
        != plan_schedule_path
        or plan.get("schedule_file_sha256")
        != schedule_ref.get("file_sha256")
        or plan.get("schedule_sha256")
        != schedule_ref.get("schedule_sha256")
    ):
        raise ValueError("Experiment plan flat schedule aliases mismatch")

    derived = _require_mapping(plan, "derived_manifest", "experiment_plan")
    derived_path = _resolve(
        root,
        _require_text(derived, "path", "experiment_plan.derived_manifest"),
    )
    checked["derived_manifest"] = _assert_file_hash(
        derived_path,
        derived.get("file_sha256"),
        "experiment_plan.derived_manifest",
    )

    source_identity = _require_mapping(plan, "source_identity", "experiment_plan")
    source_paths: dict[str, Path] = {}
    for label in ("manifest", "dataset_contract", "cache_lineage", "base_config"):
        value = _require_mapping(
            source_identity, label, "experiment_plan.source_identity"
        )
        source_path = _resolve(
            root,
            _require_text(
                value, "path", f"experiment_plan.source_identity.{label}"
            ),
        )
        checked[f"source_identity.{label}"] = _assert_file_hash(
            source_path,
            value.get("file_sha256"),
            f"experiment_plan.source_identity.{label}",
        )
        source_paths[label] = source_path

    contract_payload = _load_json(source_paths["dataset_contract"])
    contract_declared = _require_sha256(
        contract_payload.get("contract_sha256"),
        "dataset_contract.contract_sha256",
    )
    if contract_declared != _canonical_self_hash(
        contract_payload, "contract_sha256"
    ):
        raise ValueError("Dataset contract canonical self-hash mismatch")
    contract_identity = _require_mapping(
        source_identity, "dataset_contract", "experiment_plan.source_identity"
    )
    if contract_identity.get("contract_sha256") != contract_declared:
        raise ValueError("Experiment plan dataset contract self-hash mismatch")

    lineage_payload = _load_json(source_paths["cache_lineage"])
    lineage_declared = _require_sha256(
        lineage_payload.get("cache_metadata_sha256"),
        "cache_lineage.cache_metadata_sha256",
    )
    if lineage_declared != _canonical_self_hash(
        lineage_payload, "cache_metadata_sha256"
    ):
        raise ValueError("Cache lineage canonical self-hash mismatch")
    lineage_identity = _require_mapping(
        source_identity, "cache_lineage", "experiment_plan.source_identity"
    )
    if lineage_identity.get("cache_metadata_sha256") != lineage_declared:
        raise ValueError("Experiment plan cache lineage self-hash mismatch")

    mean_identity = _require_mapping(
        source_identity, "mean_checkpoint", "experiment_plan.source_identity"
    )
    mean_paths: dict[str, Path] = {}
    for artifact, path_field, hash_field in (
        ("upstream_decision", "decision_path", "decision_file_sha256"),
        ("checkpoint", "checkpoint_path", "checkpoint_file_sha256"),
        ("sidecar", "sidecar_path", "sidecar_file_sha256"),
    ):
        artifact_path = _resolve(
            root,
            _require_text(
                mean_identity,
                path_field,
                "experiment_plan.source_identity.mean_checkpoint",
            ),
        )
        checked[f"source_identity.mean_checkpoint.{artifact}"] = (
            _assert_file_hash(
                artifact_path,
                mean_identity.get(hash_field),
                f"experiment_plan.source_identity.mean_checkpoint.{artifact}",
            )
        )
        mean_paths[artifact] = artifact_path
    _require_sha256(
        mean_identity.get("sidecar_fingerprint_sha256"),
        (
            "experiment_plan.source_identity.mean_checkpoint."
            "sidecar_fingerprint_sha256"
        ),
    )
    if mean_identity.get("pathology_exclusion_enabled") is not True:
        raise ValueError("Experiment plan mean checkpoint exclusion is disabled")
    if mean_identity.get("production_cutover_claimed") is not False:
        raise ValueError("Experiment plan claims a forbidden mean cutover")
    sidecar_payload = _load_json(mean_paths["sidecar"])
    sidecar_declared = _require_sha256(
        sidecar_payload.get("fingerprint_sha256"),
        "mean_checkpoint.sidecar.fingerprint_sha256",
    )
    if sidecar_declared != _canonical_self_hash(
        sidecar_payload, "fingerprint_sha256"
    ):
        raise ValueError("Mean checkpoint sidecar canonical self-hash mismatch")
    if (
        sidecar_declared != mean_identity.get("sidecar_fingerprint_sha256")
        or sidecar_payload.get("checkpoint_sha256")
        != checked["source_identity.mean_checkpoint.checkpoint"]
    ):
        raise ValueError("Mean checkpoint, sidecar, and experiment plan disagree")

    plan_runtime_hashes = _require_mapping(
        source_identity,
        "runtime_source_hashes",
        "experiment_plan.source_identity",
    )
    calibration_runtime_hashes = _require_mapping(
        calibration, "runtime_source_hashes", "calibration"
    )
    if dict(plan_runtime_hashes) != dict(calibration_runtime_hashes):
        raise ValueError(
            "Experiment plan runtime source hashes mismatch calibration"
        )

    stop_rules = _require_mapping(
        plan, "stop_and_claim_rules", "experiment_plan"
    )
    forbidden_true = (
        "production_training",
        "production_training_allowed",
        "production_activation",
        "production_activation_allowed",
        "h5_started",
        "h6_started",
        "h5_h6_allowed",
        "next_production_stage_allowed",
    )
    for field in forbidden_true:
        if field in stop_rules and stop_rules.get(field) is not False:
            raise ValueError(
                f"experiment_plan.stop_and_claim_rules.{field} must be false"
            )
    return plan, checked


def _manifest_semantic_sha256(
    rows: Sequence[Mapping[str, str]],
) -> str:
    canonical_rows = [
        {
            column: str(row.get(column, "")).strip()
            for column in MANIFEST_COLUMNS
        }
        for row in sorted(rows, key=lambda item: item.get("sample_id", ""))
    ]
    payload = {
        "schema_version": 1,
        "columns": list(MANIFEST_COLUMNS),
        "rows": canonical_rows,
    }
    return canonical_json_sha256(payload)


def _validate_manifests(
    root: Path,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> tuple[dict[str, str], list[str], int, dict[str, Any]]:
    dataset = _require_mapping(config, "dataset_contract", "protocol")
    manifest_path = _resolve(
        root,
        _require_text(dataset, "manifest_path", "protocol.dataset_contract"),
    )
    _assert_file_hash(
        manifest_path,
        dataset.get("manifest_file_sha256"),
        "protocol.dataset_contract.authoritative_manifest",
    )
    rows = read_manifest(manifest_path)
    observed_semantic = _manifest_semantic_sha256(rows)
    expected_semantic = _require_sha256(
        dataset.get("manifest_semantic_sha256"),
        "protocol.dataset_contract.manifest_semantic_sha256",
    )
    if observed_semantic != expected_semantic:
        raise ValueError("Authoritative manifest semantic SHA-256 mismatch")

    partition = patient_partition(rows)
    observed_partition_hash = partition_sha256(partition)
    expected_partition_hash = _require_sha256(
        dataset.get("mechanism_partition_sha256"),
        "protocol.dataset_contract.mechanism_partition_sha256",
    )
    if observed_partition_hash != expected_partition_hash:
        raise ValueError("Mechanism partition SHA-256 mismatch")
    counts = partition_counts(rows, partition)
    expected_patient_counts = {
        "mechanism_train": _require_int(
            dataset,
            "mechanism_train_patients",
            "protocol.dataset_contract",
        ),
        "calibration": _require_int(
            dataset, "calibration_patients", "protocol.dataset_contract"
        ),
        "validation": _require_int(
            dataset, "validation_patients", "protocol.dataset_contract"
        ),
    }
    for role, expected in expected_patient_counts.items():
        if counts[role]["patients"] != expected:
            raise ValueError(
                f"Authoritative {role} patient count mismatch: "
                f"{counts[role]['patients']} != {expected}"
            )

    derived_ref = _require_mapping(plan, "derived_manifest", "experiment_plan")
    if derived_ref.get("partition_sha256") != observed_partition_hash:
        raise ValueError("Derived manifest plan partition SHA-256 mismatch")
    if derived_ref.get("counts") != counts:
        raise ValueError("Derived manifest plan counts mismatch")
    mapping = derived_ref.get("mapping")
    if mapping != {
        "mechanism_train": "train",
        "calibration": "val",
        "validation": "test",
    }:
        raise ValueError("Derived manifest role-to-split mapping mismatch")

    derived_path = _resolve(
        root,
        _require_text(
            derived_ref, "path", "experiment_plan.derived_manifest"
        ),
    )
    _assert_file_hash(
        derived_path,
        derived_ref.get("file_sha256"),
        "experiment_plan.derived_manifest",
    )
    derived_rows = read_manifest(derived_path)
    expected_validation = sorted(
        patient for patient, role in partition.items() if role == "validation"
    )
    derived_test = sorted(
        {row["patient_id"] for row in derived_rows if row["split"] == "test"}
    )
    if derived_test != expected_validation:
        raise ValueError(
            "Derived manifest test patients are not exactly the locked "
            "validation cohort"
        )
    expected_validation_samples = counts["validation"]["samples"]
    derived_test_samples = sum(row["split"] == "test" for row in derived_rows)
    if derived_test_samples != expected_validation_samples:
        raise ValueError("Derived manifest validation sample count mismatch")
    if len(derived_rows) != len(rows):
        raise ValueError("Derived manifest dropped or added samples")
    source_by_sample = {row["sample_id"]: row for row in rows}
    derived_by_sample = {row["sample_id"]: row for row in derived_rows}
    if set(source_by_sample) != set(derived_by_sample):
        raise ValueError("Derived manifest sample identity set mismatch")
    split_for_role = {
        "mechanism_train": "train",
        "calibration": "val",
        "validation": "test",
    }
    for sample_id, source_row in source_by_sample.items():
        derived_row = derived_by_sample[sample_id]
        patient = source_row["patient_id"]
        expected_split = split_for_role[partition[patient]]
        for field in ("patient_id", "slice_id", "cache_path"):
            if derived_row[field] != source_row[field]:
                raise ValueError(
                    f"Derived manifest changed {field} for sample {sample_id}"
                )
        if derived_row["split"] != expected_split:
            raise ValueError(
                f"Derived manifest assigned wrong split for sample {sample_id}"
            )

    hard_report: dict[str, Any] = {}
    for patient in HARD_PATIENTS:
        if patient not in partition:
            raise ValueError(f"Required difficult patient {patient} is absent")
        hard_report[patient] = {
            "partition": partition[patient],
            "evaluated_on_validation": partition[patient] == "validation",
        }
    return (
        partition,
        expected_validation,
        expected_validation_samples,
        {
            "authoritative_manifest_path": str(manifest_path),
            "authoritative_manifest_file_sha256": file_sha256(manifest_path),
            "authoritative_manifest_semantic_sha256": observed_semantic,
            "mechanism_partition_sha256": observed_partition_hash,
            "counts": counts,
            "derived_manifest_path": str(derived_path),
            "derived_manifest_file_sha256": file_sha256(derived_path),
            "hard_patient_partitions": hard_report,
        },
    )


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load experiment YAML {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Experiment YAML root must be an object: {path}")
    return payload


def _validate_evaluation_provenance(
    root: Path,
    plan: Mapping[str, Any],
    plan_path: Path,
    variant: str,
    spec: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    *,
    expected_patient_ids: Sequence[str],
    expected_samples: int,
) -> tuple[Path, str, str]:
    provenance_path = _resolve(
        root,
        _require_text(
            spec,
            "evaluation_provenance_path",
            f"experiment_plan.variants.{variant}",
        ),
    )
    provenance = _load_json(provenance_path)
    if provenance.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{variant} evaluation provenance schema mismatch")
    if (
        provenance.get("pipeline_id")
        != EVALUATION_PROVENANCE_PIPELINE_ID
    ):
        raise ValueError(f"{variant} evaluation provenance pipeline mismatch")
    declared_self = _require_sha256(
        provenance.get("provenance_sha256"),
        f"{variant}.evaluation_provenance.provenance_sha256",
    )
    if declared_self != _canonical_self_hash(
        provenance, "provenance_sha256"
    ):
        raise ValueError(f"{variant} evaluation provenance self-hash mismatch")

    expected_exact = {
        "variant": variant,
        "plan_path": str(plan_path),
        "plan_sha256": plan.get("plan_sha256"),
        "config_path": artifacts["config_path"],
        "config_sha256": artifacts["config_file_sha256"],
        "checkpoint_path": artifacts["checkpoint_path"],
        "checkpoint_sha256": artifacts["checkpoint_file_sha256"],
        "checkpoint_epoch": spec.get("expected_final_epoch"),
        "weights": "ema",
        "split": "test",
        "full_split": True,
        "eval_path": artifacts["evaluation_path"],
        "eval_sha256": artifacts["evaluation_file_sha256"],
        "num_patients": len(expected_patient_ids),
        "num_samples": expected_samples,
        "patient_ids": list(expected_patient_ids),
        "production_activation_allowed": False,
        "h5_h6_allowed": False,
    }
    budget = _require_mapping(plan, "budget", "experiment_plan")
    expected_exact["seed"] = budget.get("evaluation_seed")
    expected_exact["mc_steps"] = budget.get("evaluation_sampling_steps")
    mismatches = {
        key: {"expected": expected, "observed": provenance.get(key)}
        for key, expected in expected_exact.items()
        if provenance.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"{variant} evaluation provenance mismatch: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    created_at = provenance.get("created_at_utc")
    if not isinstance(created_at, str) or not created_at:
        raise ValueError(f"{variant} provenance created_at_utc is missing")
    try:
        timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"{variant} provenance created_at_utc is not ISO-8601"
        ) from exc
    if timestamp.tzinfo is None:
        raise ValueError(f"{variant} provenance timestamp lacks timezone")
    return provenance_path, file_sha256(provenance_path), declared_self


def _validate_variants(
    root: Path,
    plan: Mapping[str, Any],
    plan_path: Path,
    *,
    expected_patient_ids: Sequence[str],
    expected_samples: int,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, dict[str, Any]]]:
    variants = _require_mapping(plan, "variants", "experiment_plan")
    if set(variants) != {VARIANT_REFERENCE, VARIANT_CANDIDATE}:
        raise ValueError(
            "Experiment plan variants must be exactly no_route and "
            "h3_v2_native_null"
        )
    result: dict[str, Mapping[str, Any]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    parsed_configs: dict[str, Mapping[str, Any]] = {}
    budget = _require_mapping(plan, "budget", "experiment_plan")
    for field in (
        "same_epoch_budget",
        "same_initialization_seed",
        "exact_final_epoch_checkpoint",
    ):
        if budget.get(field) is not True:
            raise ValueError(f"experiment_plan.budget.{field} must be true")
    epochs = _require_int(
        budget, "epochs_per_variant", "experiment_plan.budget"
    )
    training_seed = _require_int(
        budget, "training_seed", "experiment_plan.budget"
    )
    for name in (VARIANT_REFERENCE, VARIANT_CANDIDATE):
        spec = _require_mapping(variants, name, "experiment_plan.variants")
        config_path = _resolve(
            root,
            _require_text(
                spec, "config_path", f"experiment_plan.variants.{name}"
            ),
        )
        config_hash = _assert_file_hash(
            config_path,
            spec.get("config_sha256"),
            f"experiment_plan.variants.{name}.config",
        )
        parsed_config = _load_yaml_mapping(config_path)
        parsed_configs[name] = parsed_config
        declared_canonical = _require_sha256(
            spec.get("config_canonical_sha256"),
            f"experiment_plan.variants.{name}.config_canonical_sha256",
        )
        observed_canonical = canonical_json_sha256(parsed_config)
        if declared_canonical != observed_canonical:
            raise ValueError(f"{name} experiment config canonical hash mismatch")

        checkpoint_path = _resolve(
            root,
            _require_text(
                spec, "checkpoint_path", f"experiment_plan.variants.{name}"
            ),
        )
        if not checkpoint_path.is_file():
            raise ValueError(f"{name} exact final checkpoint is missing")
        checkpoint_hash = file_sha256(checkpoint_path)
        eval_path = _resolve(
            root,
            _require_text(
                spec, "eval_output_path", f"experiment_plan.variants.{name}"
            ),
        )
        if not eval_path.is_file():
            raise ValueError(f"{name} evaluation artifact is missing: {eval_path}")
        if spec.get("expected_final_epoch") != epochs:
            raise ValueError(f"{name} final epoch differs from matched budget")
        if spec.get("checkpoint_weights_for_evaluation") != "ema":
            raise ValueError(f"{name} did not preregister EMA evaluation")
        if (
            spec.get("evaluation_split") != "test"
            or spec.get("evaluation_full_split") is not True
        ):
            raise ValueError(f"{name} did not preregister full test-split evaluation")

        result[name] = spec
        artifacts[name] = {
            "config_path": str(config_path),
            "config_file_sha256": config_hash,
            "config_canonical_sha256": observed_canonical,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_file_sha256": checkpoint_hash,
            "evaluation_path": str(eval_path),
            "evaluation_file_sha256": file_sha256(eval_path),
        }
        (
            provenance_path,
            provenance_file_hash,
            provenance_self_hash,
        ) = _validate_evaluation_provenance(
            root,
            plan,
            plan_path,
            name,
            spec,
            artifacts[name],
            expected_patient_ids=expected_patient_ids,
            expected_samples=expected_samples,
        )
        artifacts[name].update(
            {
                "evaluation_provenance_path": str(provenance_path),
                "evaluation_provenance_file_sha256": provenance_file_hash,
                "evaluation_provenance_sha256": provenance_self_hash,
            }
        )

    reference = result[VARIANT_REFERENCE]
    candidate = result[VARIANT_CANDIDATE]
    if reference.get("hard_all_null") is not True:
        raise ValueError("no_route plan must set hard_all_null=true")
    if candidate.get("hard_all_null") is not False:
        raise ValueError("h3_v2_native_null must set hard_all_null=false")
    if candidate.get("route_policy") != "h3_native_null":
        raise ValueError("H3-v2 candidate route_policy mismatch")
    if reference.get("route_policy") != "fixed_prior":
        raise ValueError("no_route route_policy must be fixed_prior")
    schedule_ref = _require_mapping(plan, "schedule", "experiment_plan")
    if reference.get("schedule_file_sha256") is not None:
        raise ValueError("no_route unexpectedly binds an H3-v2 schedule")
    if candidate.get("schedule_file_sha256") != schedule_ref.get(
        "file_sha256"
    ):
        raise ValueError("H3-v2 variant schedule SHA-256 mismatch")

    for name, parsed in parsed_configs.items():
        experiment_section = _require_mapping(
            parsed, "experiment", f"{name}.config"
        )
        training_section = _require_mapping(
            parsed, "training", f"{name}.config"
        )
        runtime_section = _require_mapping(parsed, "runtime", f"{name}.config")
        if experiment_section.get("seed") != training_seed:
            raise ValueError(f"{name} config training seed mismatch")
        if training_section.get("num_epochs") != epochs:
            raise ValueError(f"{name} config epoch budget mismatch")
        if (
            runtime_section.get("device") != "cuda"
            or runtime_section.get("require_cuda") is not True
        ):
            raise ValueError(f"{name} config lacks the formal CUDA gate")
    return result, artifacts


def _validate_eval(
    path: Path,
    expected_patients: Sequence[str],
    expected_samples: int,
    label: str,
) -> tuple[dict[str, Any], Mapping[str, Mapping[str, Any]]]:
    payload = _load_json(path)
    if _require_int(payload, "num_patients", label) != len(expected_patients):
        raise ValueError(f"{label} num_patients is not {len(expected_patients)}")
    if _require_int(payload, "num_samples", label) != expected_samples:
        raise ValueError(f"{label} num_samples is not {expected_samples}")
    raw_per_patient = _require_mapping(payload, "per_patient", label)
    per_patient: dict[str, Mapping[str, Any]] = {}
    for patient, metrics in raw_per_patient.items():
        if not isinstance(patient, str) or not isinstance(metrics, Mapping):
            raise ValueError(f"{label}.per_patient has an invalid entry")
        per_patient[patient] = metrics
    if sorted(per_patient) != list(expected_patients):
        missing = sorted(set(expected_patients) - set(per_patient))
        extra = sorted(set(per_patient) - set(expected_patients))
        raise ValueError(
            f"{label} patient set mismatch; missing={missing}, extra={extra}"
        )
    return payload, per_patient


def _metric_or_none(metrics: Mapping[str, Any], key: str) -> float | None:
    if key not in metrics:
        return None
    value = metrics[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Patient metric {key!r} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Patient metric {key!r} must be finite")
    return number


def _paired_values(
    reference: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    patients: Sequence[str],
    metric: str,
    *,
    require_all: bool,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    paired_patients: list[str] = []
    reference_values: list[float] = []
    candidate_values: list[float] = []
    for patient in patients:
        reference_value = _metric_or_none(reference[patient], metric)
        candidate_value = _metric_or_none(candidate[patient], metric)
        if (reference_value is None) != (candidate_value is None):
            raise ValueError(
                f"Metric presence mismatch for {metric}, patient {patient}"
            )
        if reference_value is None:
            if require_all:
                raise ValueError(
                    f"Required safety metric {metric} missing for patient {patient}"
                )
            continue
        paired_patients.append(patient)
        reference_values.append(reference_value)
        candidate_values.append(candidate_value)
    return (
        paired_patients,
        np.asarray(reference_values, dtype=np.float64),
        np.asarray(candidate_values, dtype=np.float64),
    )


def _bootstrap_indices(n: int, replicates: int, seed: int) -> np.ndarray:
    if n <= 0 or replicates <= 0:
        raise ValueError("Bootstrap requires positive n and replicates")
    return np.random.default_rng(seed).integers(
        0, n, size=(replicates, n), endpoint=False
    )


def _ci95(values: np.ndarray) -> list[float]:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Bootstrap distribution must be a finite vector")
    lower, upper = np.quantile(values, [0.025, 0.975])
    return [float(lower), float(upper)]


def _difference_statistic(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("Paired difference inputs must be equal-length vectors")
    indices = _bootstrap_indices(reference.size, replicates, seed)
    differences = candidate - reference
    bootstrap = differences[indices].mean(axis=1)
    return {
        "patients": int(reference.size),
        "reference_patient_mean": float(reference.mean()),
        "candidate_patient_mean": float(candidate.mean()),
        "candidate_minus_reference_mean": float(differences.mean()),
        "bootstrap_ci95": _ci95(bootstrap),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "ci_quantiles": [0.025, 0.975],
    }


def _ratio_statistic(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    epsilon: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("Paired ratio inputs must be equal-length vectors")
    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError("Ratio epsilon must be finite and positive")
    if np.any(reference < 0.0) or np.any(candidate < 0.0):
        raise ValueError("Ratio safety metric must be non-negative")
    indices = _bootstrap_indices(reference.size, replicates, seed)
    reference_bootstrap = reference[indices].mean(axis=1)
    candidate_bootstrap = candidate[indices].mean(axis=1)
    bootstrap = (candidate_bootstrap + epsilon) / (
        reference_bootstrap + epsilon
    )
    estimate = (float(candidate.mean()) + epsilon) / (
        float(reference.mean()) + epsilon
    )
    return {
        "patients": int(reference.size),
        "reference_patient_mean": float(reference.mean()),
        "candidate_patient_mean": float(candidate.mean()),
        "candidate_over_reference_ratio": float(estimate),
        "ratio_formula": "(candidate_patient_mean+epsilon)/(reference_patient_mean+epsilon)",
        "epsilon": epsilon,
        "bootstrap_ci95": _ci95(bootstrap),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "ci_quantiles": [0.025, 0.975],
    }


def _analyze_metrics(
    experiment: Mapping[str, Any],
    reference: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    patients: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    primary_spec = _require_mapping(
        experiment, "primary_gate", "protocol.exploratory_model_experiment"
    )
    primary_metric = _require_text(
        primary_spec,
        "metric",
        "protocol.exploratory_model_experiment.primary_gate",
    )
    if primary_spec.get("contrast") != "h3_v2_minus_no_route":
        raise ValueError("Primary contrast must be h3_v2_minus_no_route")
    if primary_spec.get("direction") != "lower":
        raise ValueError("Primary direction must be lower")
    minimum_patients = _require_int(
        primary_spec,
        "minimum_patients",
        "protocol.exploratory_model_experiment.primary_gate",
    )
    primary_threshold = _require_finite_number(
        primary_spec,
        "required_bootstrap_ci95_high_below",
        "protocol.exploratory_model_experiment.primary_gate",
    )
    replicates = _require_int(
        experiment,
        "bootstrap_replicates",
        "protocol.exploratory_model_experiment",
    )
    seed = _require_int(
        experiment,
        "bootstrap_seed",
        "protocol.exploratory_model_experiment",
    )
    if replicates <= 0:
        raise ValueError("Experiment bootstrap_replicates must be positive")

    primary_patients, primary_reference, primary_candidate = _paired_values(
        reference,
        candidate,
        patients,
        primary_metric,
        require_all=False,
    )
    if primary_reference.size:
        primary_result = _difference_statistic(
            primary_reference,
            primary_candidate,
            replicates=replicates,
            seed=seed,
        )
        primary_result["bootstrap_ci95_high_threshold"] = primary_threshold
        primary_result["minimum_patients"] = minimum_patients
        primary_result["metric"] = primary_metric
        primary_result["paired_patient_ids"] = primary_patients
        primary_result["passed"] = bool(
            primary_reference.size >= minimum_patients
            and primary_result["bootstrap_ci95"][1] < primary_threshold
        )
    else:
        primary_result = {
            "metric": primary_metric,
            "patients": 0,
            "paired_patient_ids": [],
            "minimum_patients": minimum_patients,
            "bootstrap_ci95_high_threshold": primary_threshold,
            "passed": False,
            "reason": "No patients had the preregistered small-lesion metric.",
        }

    safety_spec = _require_mapping(
        experiment, "safety_gates", "protocol.exploratory_model_experiment"
    )
    epsilon = _require_finite_number(
        safety_spec,
        "false_hotspot_ratio_epsilon",
        "protocol.exploratory_model_experiment.safety_gates",
    )
    safety_results: dict[str, Any] = {}
    safety_metrics: list[str] = []
    for gate_name, raw_threshold in safety_spec.items():
        if gate_name == "false_hotspot_ratio_epsilon":
            continue
        threshold = (
            float(raw_threshold)
            if isinstance(raw_threshold, (int, float))
            and not isinstance(raw_threshold, bool)
            and math.isfinite(float(raw_threshold))
            else None
        )
        if threshold is None:
            raise ValueError(f"Safety threshold {gate_name} must be finite")
        if gate_name.endswith("_maximum_delta"):
            metric = gate_name[: -len("_maximum_delta")]
            gate_kind = "maximum_delta"
        elif gate_name.endswith("_maximum_ratio"):
            metric = gate_name[: -len("_maximum_ratio")]
            gate_kind = "maximum_ratio"
        else:
            raise ValueError(f"Unrecognized preregistered safety gate: {gate_name}")
        metric_patients, metric_reference, metric_candidate = _paired_values(
            reference,
            candidate,
            patients,
            metric,
            require_all=True,
        )
        if metric_patients != list(patients):
            raise ValueError(f"Safety metric {metric} is not fully paired")
        if gate_kind == "maximum_delta":
            result = _difference_statistic(
                metric_reference,
                metric_candidate,
                replicates=replicates,
                seed=seed,
            )
            estimate = result["candidate_minus_reference_mean"]
            result["passed"] = bool(estimate <= threshold)
            result["gate_rule"] = (
                "candidate_minus_reference_mean <= preregistered_maximum_delta; "
                "bootstrap_ci95_is_descriptive"
            )
        else:
            result = _ratio_statistic(
                metric_reference,
                metric_candidate,
                epsilon=epsilon,
                replicates=replicates,
                seed=seed,
            )
            estimate = result["candidate_over_reference_ratio"]
            result["passed"] = bool(estimate <= threshold)
            result["gate_rule"] = (
                "candidate_over_reference_ratio <= "
                "preregistered_maximum_ratio; bootstrap_ci95_is_descriptive"
            )
        result.update(
            {
                "gate_name": gate_name,
                "metric": metric,
                "maximum_allowed": threshold,
            }
        )
        safety_results[gate_name] = result
        safety_metrics.append(metric)

    rows: list[dict[str, Any]] = []
    for patient in patients:
        row: dict[str, Any] = {
            "patient_id": patient,
            "partition": "validation",
            "difficult_patient": patient in HARD_PATIENTS,
        }
        metric_names = [primary_metric, *safety_metrics]
        for metric in metric_names:
            reference_value = _metric_or_none(reference[patient], metric)
            candidate_value = _metric_or_none(candidate[patient], metric)
            row[f"no_route__{metric}"] = (
                "" if reference_value is None else reference_value
            )
            row[f"h3_v2_native_null__{metric}"] = (
                "" if candidate_value is None else candidate_value
            )
            row[f"candidate_minus_reference__{metric}"] = (
                ""
                if reference_value is None or candidate_value is None
                else candidate_value - reference_value
            )
        rows.append(row)
    return primary_result, safety_results, rows


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("Patient difference CSV cannot be empty")
    columns = sorted({key for row in rows for key in row})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=columns,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _create_new_or_identical(
    decision_path: Path,
    decision_bytes: bytes,
    csv_path: Path,
    csv_bytes: bytes,
) -> str:
    decision_exists = decision_path.exists()
    csv_exists = csv_path.exists()
    if decision_exists != csv_exists:
        if decision_exists:
            if (
                not decision_path.is_file()
                or decision_path.read_bytes() != decision_bytes
            ):
                raise FileExistsError(
                    "Partial analysis decision differs; refusing recovery"
                )
            _write_atomic(csv_path, csv_bytes)
            return "RECOVERED_IDENTICAL_MISSING_CSV"
        if not csv_path.is_file() or csv_path.read_bytes() != csv_bytes:
            raise FileExistsError(
                "Partial patient CSV differs; refusing recovery"
            )
        _write_atomic(decision_path, decision_bytes)
        return "RECOVERED_IDENTICAL_MISSING_DECISION"
    if decision_exists:
        if not decision_path.is_file() or not csv_path.is_file():
            raise FileExistsError("Existing analysis output is not a regular file")
        if decision_path.read_bytes() != decision_bytes:
            raise FileExistsError(
                "Existing decision differs; refusing to overwrite sealed output"
            )
        if csv_path.read_bytes() != csv_bytes:
            raise FileExistsError(
                "Existing patient CSV differs; refusing to overwrite sealed output"
            )
        return "REUSED_IDENTICAL"
    _write_atomic(csv_path, csv_bytes)
    _write_atomic(decision_path, decision_bytes)
    return "CREATED_NEW"


def run(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    root = args.root.resolve()
    config_path = _resolve(root, args.protocol_config)
    config_file_hash = file_sha256(config_path)
    config, config_sha256, runtime_source_hashes = _validate_config(
        root, config_path
    )
    # Bind the source path without changing the hashed protocol object.
    config_with_source = dict(config)
    config_with_source["_source_path"] = str(config_path)

    plan_path = _resolve(root, args.experiment_plan)
    raw_plan = _load_json(plan_path)
    calibration_ref = _require_mapping(
        raw_plan, "calibration_decision", "experiment_plan"
    )
    plan_calibration_path = _resolve(
        root,
        _require_text(
            calibration_ref, "path", "experiment_plan.calibration_decision"
        ),
    )
    if args.calibration_decision is not None:
        requested_calibration_path = _resolve(root, args.calibration_decision)
        if requested_calibration_path != plan_calibration_path:
            raise ValueError(
                "--calibration-decision does not match the sealed experiment plan"
            )
    calibration_path = plan_calibration_path
    calibration_file_hash = file_sha256(calibration_path)
    calibration, calibration_checked = _validate_calibration(
        root, calibration_path, config_with_source, config_sha256
    )
    plan, plan_checked = _validate_plan(
        root,
        plan_path,
        config_path,
        config_file_hash,
        config_sha256,
        calibration_path,
        calibration_file_hash,
        calibration,
    )

    calibration_spec = _require_mapping(config, "calibration", "protocol")
    expected_timesteps = _require_int(
        calibration_spec, "num_train_timesteps", "protocol.calibration"
    )
    schedule_ref = _require_mapping(plan, "schedule", "experiment_plan")
    schedule_path = _resolve(
        root, _require_text(schedule_ref, "path", "experiment_plan.schedule")
    )
    schedule, schedule_file_hash, schedule_self_hash = _validate_schedule(
        schedule_path,
        schedule_ref.get("file_sha256"),
        schedule_ref.get("schedule_sha256"),
        config_sha256,
        expected_timesteps,
    )

    (
        partition,
        validation_patients,
        validation_samples,
        manifest_evidence,
    ) = _validate_manifests(root, config, plan)
    variant_specs, variant_artifacts = _validate_variants(
        root,
        plan,
        plan_path,
        expected_patient_ids=validation_patients,
        expected_samples=validation_samples,
    )

    reference_eval_path = Path(
        variant_artifacts[VARIANT_REFERENCE]["evaluation_path"]
    )
    candidate_eval_path = Path(
        variant_artifacts[VARIANT_CANDIDATE]["evaluation_path"]
    )
    if args.no_route_eval is not None:
        override = _resolve(root, args.no_route_eval)
        if override != reference_eval_path:
            raise ValueError("--no-route-eval differs from the sealed plan")
    if args.h3_v2_eval is not None:
        override = _resolve(root, args.h3_v2_eval)
        if override != candidate_eval_path:
            raise ValueError("--h3-v2-eval differs from the sealed plan")

    _, reference_patients = _validate_eval(
        reference_eval_path,
        validation_patients,
        validation_samples,
        "no_route_evaluation",
    )
    _, candidate_patients = _validate_eval(
        candidate_eval_path,
        validation_patients,
        validation_samples,
        "h3_v2_evaluation",
    )
    if set(reference_patients) != set(candidate_patients):
        raise ValueError("The two evaluation patient sets are not identical")

    experiment = _require_mapping(
        config, "exploratory_model_experiment", "protocol"
    )
    if experiment.get("evaluation_partition") != (
        "validation_31_patients_previously_exposed"
    ):
        raise ValueError("Unexpected exploratory evaluation partition")
    primary, safety, patient_rows = _analyze_metrics(
        experiment,
        reference_patients,
        candidate_patients,
        validation_patients,
    )
    safety_passed = all(result["passed"] for result in safety.values())
    supported = bool(primary["passed"] and safety_passed)
    decision_name = "SUPPORT" if supported else "NO_SUPPORT"
    wording_key = (
        "allowed_positive_wording"
        if supported
        else "allowed_negative_wording"
    )
    conclusion = _require_text(
        experiment, wording_key, "protocol.exploratory_model_experiment"
    )

    hard_patient_report: dict[str, Any] = {}
    all_report_metrics = [
        _require_text(
            _require_mapping(
                experiment,
                "primary_gate",
                "protocol.exploratory_model_experiment",
            ),
            "metric",
            "protocol.exploratory_model_experiment.primary_gate",
        )
    ]
    all_report_metrics.extend(result["metric"] for result in safety.values())
    for patient in HARD_PATIENTS:
        role = partition[patient]
        record: dict[str, Any] = {
            "partition": role,
            "evaluated_on_validation": role == "validation",
        }
        if role == "validation":
            record["no_route"] = {
                metric: _metric_or_none(reference_patients[patient], metric)
                for metric in all_report_metrics
            }
            record["h3_v2_native_null"] = {
                metric: _metric_or_none(candidate_patients[patient], metric)
                for metric in all_report_metrics
            }
        else:
            record["metrics"] = None
            record["reason"] = (
                "Not in the locked exposed validation partition; excluded from "
                "the evaluation contrast but explicitly reported."
            )
        hard_patient_report[patient] = record

    csv_payload = _csv_bytes(patient_rows)
    csv_hash = _sha256_bytes(csv_payload)
    output_dir = _resolve(root, args.output_dir)
    csv_path = output_dir / PATIENT_CSV_FILENAME
    decision_path = output_dir / DECISION_FILENAME
    decision: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_id": PIPELINE_ID,
        "protocol_pipeline_id": CALIBRATION_PIPELINE_ID,
        "execution_status": "COMPLETED",
        "decision": decision_name,
        "scientific_decision": decision_name,
        "scientific_conclusion": conclusion,
        "evaluation_label": "post_hoc_internal_exploratory_only",
        "validation_previously_exposed": True,
        "independent_validation_claimed": False,
        "model_experiment_complete": True,
        "input_artifacts": {
            "protocol_config": {
                "path": str(config_path),
                "file_sha256": config_file_hash,
                "config_sha256": config_sha256,
            },
            "runtime_source_hashes": runtime_source_hashes,
            "calibration_decision": {
                "path": str(calibration_path),
                "file_sha256": calibration_file_hash,
                "decision": "PASS",
                "checked_file_hashes": calibration_checked,
            },
            "frozen_schedule": {
                "path": str(schedule_path),
                "file_sha256": schedule_file_hash,
                "schedule_sha256": schedule_self_hash,
                "mapping": schedule.get("mapping"),
                "shallow_route_policy": schedule.get(
                    "shallow_route_policy"
                ),
            },
            "experiment_plan": {
                "path": str(plan_path),
                "file_sha256": file_sha256(plan_path),
                "plan_sha256": plan["plan_sha256"],
                "checked_file_hashes": plan_checked,
            },
            "variants": variant_artifacts,
        },
        "dataset_integrity": {
            **manifest_evidence,
            "required_validation_patients": len(validation_patients),
            "required_validation_samples": validation_samples,
            "paired_patient_ids": validation_patients,
            "no_route_patient_set_complete": True,
            "h3_v2_patient_set_complete": True,
            "patient_sets_identical": True,
        },
        "primary_gate": primary,
        "safety_gates": safety,
        "all_safety_gates_passed": safety_passed,
        "difficult_patients": hard_patient_report,
        "patient_differences": {
            "path": str(csv_path),
            "file_sha256": csv_hash,
            "rows": len(patient_rows),
            "unit": "patient",
        },
        "production_training_allowed": False,
        "production_activation_allowed": False,
        "h5_h6_allowed": False,
        "next_production_stage_allowed": False,
        "does_not_override_05B": True,
        "does_not_restore_h4_v1_or_h4_v2": True,
        "claim_boundary": (
            "Internal exploratory evidence on the already exposed development "
            "dataset only; no external, confirmatory, or production claim."
        ),
    }
    decision["decision_sha256"] = _canonical_self_hash(
        decision, "decision_sha256"
    )
    decision_payload = _json_bytes(decision)
    disposition = _create_new_or_identical(
        decision_path, decision_payload, csv_path, csv_payload
    )
    return decision, disposition


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root (default: inferred from this script).",
    )
    parser.add_argument(
        "--protocol-config",
        "--config",
        dest="protocol_config",
        type=Path,
        default=Path(
            "configs/h3_v2_full_timestep_native_null_v1.json"
        ),
        help="Frozen H3-v2 preregistration JSON.",
    )
    parser.add_argument(
        "--experiment-plan",
        type=Path,
        required=True,
        help="Sealed experiment_plan.json produced after calibration PASS.",
    )
    parser.add_argument(
        "--calibration-decision",
        type=Path,
        help=(
            "Optional explicit calibration decision; if provided it must equal "
            "the path sealed in the experiment plan."
        ),
    )
    parser.add_argument(
        "--no-route-eval",
        type=Path,
        help=(
            "Optional explicit no-route evaluation JSON; it must equal the "
            "path sealed in the experiment plan."
        ),
    )
    parser.add_argument(
        "--h3-v2-eval",
        type=Path,
        help=(
            "Optional explicit H3-v2 evaluation JSON; it must equal the path "
            "sealed in the experiment plan."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Create-new analysis directory for decision.json and "
            "patient_differences.csv."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        decision, disposition = run(args)
    except Exception as exc:
        print(
            f"H3-v2 experiment analysis FAILED CLOSED: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(
        json.dumps(
            {
                "pipeline_id": decision["pipeline_id"],
                "decision": decision["decision"],
                "decision_sha256": decision["decision_sha256"],
                "output_disposition": disposition,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
