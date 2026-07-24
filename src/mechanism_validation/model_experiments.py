"""Locked fixed-endpoint model experiments for H5/H6.

This module deliberately owns every formal training/evaluation choice used by
the downstream gates.  Variants differ only in predeclared router factors.
There is no best-checkpoint selection, early stopping, validation-driven
course change, or evaluation subset selection.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from src.data.lineage import (
    load_checkpoint_data_lineage,
    validate_checkpoint_data_lineage,
)
from src.mechanism_validation.common import (
    bootstrap_mean,
    canonical_json_sha256,
    file_sha256,
    sign_flip_p,
    write_json,
)
from src.model.config_utils import load_full_config, resolve_runtime_profile


FORMAL_SEED = 4242
EVALUATION_SEED = 42
DEFAULT_EPOCHS = 50
DEFAULT_MC_STEPS = 20
VARIANTS = (
    "null_reference",
    "full",
    "no_ct_support",
    "no_recoverability",
    "no_artifact_safety",
)


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be an object: {path}")
    return payload


def _set_nested(config: dict[str, Any], dotted: str, value: Any) -> None:
    cursor = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def _variant_overrides(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown formal variant {variant!r}")
    common = {
        "modules.residual_frequency.enabled": True,
        "modules.residual_frequency.mode": "spectral_evidence_router",
        "modules.residual_frequency.cross_level_router.enabled": True,
        "modules.residual_frequency.cross_level_router.hard_all_null": False,
        "modules.residual_frequency.cross_level_router.policy": "learned",
        "modules.residual_frequency.cross_level_router.fixed_prior": [
            0.05,
            0.05,
            0.90,
        ],
        "modules.residual_frequency.use_ct_reliability": False,
        "modules.residual_frequency.use_noise_release": True,
    }
    if variant == "null_reference":
        common[
            "modules.residual_frequency.cross_level_router.hard_all_null"
        ] = True
    elif variant == "no_ct_support":
        common[
            "modules.residual_frequency.ct_support_head.enabled"
        ] = False
    elif variant == "no_recoverability":
        common["modules.residual_frequency.use_noise_release"] = False
    elif variant == "no_artifact_safety":
        common[
            "modules.residual_frequency.cross_level_router.policy"
        ] = "learned_no_null"
        common[
            "modules.residual_frequency.cross_level_router.fixed_prior"
        ] = [0.5, 0.5]
    return common


def prepare_variant_config(
    *,
    root: Path,
    base_config: Path,
    cache_dir: Path,
    cache_lineage: Path,
    contract: Path,
    mechanism_manifest: Path,
    mean_checkpoint: Path,
    curriculum: Mapping[str, Any],
    variant: str,
    epochs: int,
    seed: int = FORMAL_SEED,
) -> dict[str, Any]:
    """Create a self-contained formal config with absolute data inputs."""

    if epochs <= 0:
        raise ValueError("Formal epochs must be positive")
    config = copy.deepcopy(load_full_config(str(base_config)))
    fixed = {
        "experiment.name": f"mechanism_v1_{variant}",
        "experiment.seed": int(seed),
        "data.cache_dir": str(cache_dir),
        "data.cache_lineage": str(cache_lineage),
        "data.dataset_contract": str(contract),
        "data.require_cache_lineage": True,
        "data.split_manifest": str(mechanism_manifest),
        "data.augment": True,
        "modules.conditional_mean.checkpoint": str(mean_checkpoint),
        "modules.conditional_mean.freeze": True,
        "modules.conditional_mean.detach_bridge": True,
        "training.num_epochs": int(epochs),
        "training.init_from": None,
        "training.resume_from": None,
        "runtime.eval_interval": int(epochs + 1),
        "runtime.sample_interval": int(epochs + 1),
        "runtime.save_interval": int(epochs),
        "runtime.best_checkpoint.enabled": False,
        "runtime.early_stopping.enabled": False,
        "runtime.tracked_sample_ids": [],
        "runtime.eval_seed": EVALUATION_SEED,
    }
    router_schedule = curriculum.get("router_epoch_schedule", {})
    fixed[
        "modules.residual_frequency.cross_level_router.native_warmup_epochs"
    ] = int(router_schedule.get("native_warmup_epochs", 0))
    fixed[
        "modules.residual_frequency.cross_level_router.routing_ramp_epochs"
    ] = int(router_schedule.get("routing_ramp_epochs", 0))
    loss_schedule = curriculum.get("loss_active_tau_max", {})
    for loss_name, tau_max in loss_schedule.items():
        fixed[f"losses.{loss_name}.active_tau_max"] = float(tau_max)
    ct_support = curriculum.get("ct_support_head", {})
    fixed.update(
        {
            "modules.residual_frequency.ct_support_head.enabled": bool(
                ct_support.get("enabled", False)
            ),
            "modules.residual_frequency.ct_support_head.selected_haar_band": str(
                ct_support.get("selected_haar_band", "l2_hh")
            ),
            "modules.residual_frequency.ct_support_head.response_direction": str(
                ct_support.get("response_direction", "-")
            ),
            "modules.residual_frequency.ct_support_head.support_only": bool(
                ct_support.get("support_only", True)
            ),
        }
    )
    for dotted, value in {**fixed, **_variant_overrides(variant)}.items():
        _set_nested(config, dotted, value)
    config.setdefault("formal_mechanism", {}).update(
        {
            "schema_version": 1,
            "variant": variant,
            "fixed_endpoint_epochs": int(epochs),
            "training_seed": int(seed),
            "evaluation_seed": EVALUATION_SEED,
            "checkpoint_policy": "exact final epoch EMA only",
            "validation_drives_training_or_course": False,
            "mechanism_manifest_sha256": file_sha256(mechanism_manifest),
            "mean_checkpoint_sha256": file_sha256(mean_checkpoint),
            "curriculum_sha256": canonical_json_sha256(dict(curriculum)),
        }
    )
    return config


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        yaml.safe_dump(
            dict(payload),
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run_streaming(command: Sequence[str], *, root: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            list(command),
            cwd=str(root),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        exit_code = process.wait()
    if exit_code:
        raise RuntimeError(
            f"Command failed with exit code {exit_code}: "
            f"{subprocess.list2cmdline(list(command))}"
        )


def _canonical_config_hash(config: Mapping[str, Any]) -> str:
    return canonical_json_sha256(dict(config))


def _validate_checkpoint(
    *,
    checkpoint_path: Path,
    expected_config: Mapping[str, Any],
    expected_lineage: Mapping[str, Any],
    expected_epoch: int,
    variant: str,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint is not a mapping: {checkpoint_path}")
    validate_checkpoint_data_lineage(
        checkpoint,
        expected_lineage,
        required=True,
        context=f"formal {variant} checkpoint {checkpoint_path}",
    )
    if int(checkpoint.get("epoch", -1)) != int(expected_epoch):
        raise ValueError(
            f"{variant} checkpoint epoch={checkpoint.get('epoch')!r}, "
            f"expected {expected_epoch}"
        )
    embedded_config = checkpoint.get("config")
    if not isinstance(embedded_config, dict):
        raise ValueError(f"{variant} checkpoint lacks embedded config")
    # train_v2 resolves the runtime profile before embedding. Formal model
    # experiments require CUDA, so this is identical to expected_config.
    expected_resolved = resolve_runtime_profile(dict(expected_config))
    if _canonical_config_hash(embedded_config) != _canonical_config_hash(
        expected_resolved
    ):
        raise ValueError(
            f"{variant} checkpoint embedded config differs from locked variant"
        )
    formal = embedded_config.get("formal_mechanism", {})
    if formal.get("variant") != variant:
        raise ValueError(f"{variant} checkpoint variant marker mismatch")
    return checkpoint


def ensure_variant(
    *,
    root: Path,
    work_dir: Path,
    base_config: Path,
    cache_dir: Path,
    cache_lineage: Path,
    contract: Path,
    mechanism_manifest: Path,
    mean_checkpoint: Path,
    curriculum: Mapping[str, Any],
    variant: str,
    epochs: int = DEFAULT_EPOCHS,
    mc_steps: int = DEFAULT_MC_STEPS,
    force: bool = False,
) -> dict[str, Any]:
    """Train/reuse one fixed endpoint and evaluate full calibration/validation."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Formal H5/H6 model experiments require CUDA; CPU smoke profiles "
            "change image size and are forbidden for formal gates"
        )
    config = prepare_variant_config(
        root=root,
        base_config=base_config,
        cache_dir=cache_dir,
        cache_lineage=cache_lineage,
        contract=contract,
        mechanism_manifest=mechanism_manifest,
        mean_checkpoint=mean_checkpoint,
        curriculum=curriculum,
        variant=variant,
        epochs=epochs,
    )
    config_path = work_dir / "configs" / f"{variant}.yaml"
    write_yaml(config_path, config)
    expected_lineage = load_checkpoint_data_lineage(config, root=root)
    if expected_lineage is None:
        raise RuntimeError("Formal model config did not resolve sealed lineage")

    experiment = str(config["experiment"]["name"])
    checkpoint_path = root / "checkpoints" / experiment / (
        f"ckpt_epoch{epochs:04d}.pt"
    )
    if force or not checkpoint_path.is_file():
        _run_streaming(
            [
                sys.executable,
                "scripts/train_v2.py",
                "--config",
                str(config_path),
            ],
            root=root,
            log_path=work_dir / "logs" / f"train_{variant}.log",
        )
    checkpoint = _validate_checkpoint(
        checkpoint_path=checkpoint_path,
        expected_config=config,
        expected_lineage=expected_lineage,
        expected_epoch=epochs,
        variant=variant,
    )

    evaluations: dict[str, Any] = {}
    checkpoint_sha = file_sha256(checkpoint_path)
    config_canonical_sha = _canonical_config_hash(config)
    for role, split in (("calibration", "val"), ("validation", "test")):
        output_path = work_dir / "evaluations" / variant / f"{role}.json"
        provenance_path = output_path.with_name(f"{role}_provenance.json")
        expected_provenance = {
            "schema_version": 1,
            "variant": variant,
            "role": role,
            "split": split,
            "checkpoint_sha256": checkpoint_sha,
            "config_canonical_sha256": config_canonical_sha,
            "weights": "ema",
            "evaluation_seed": EVALUATION_SEED,
            "mc_steps": int(mc_steps),
            "full_split": True,
        }
        reusable_evaluation = False
        if output_path.is_file() and provenance_path.is_file() and not force:
            try:
                observed_provenance = json.loads(
                    provenance_path.read_text(encoding="utf-8-sig")
                )
            except (OSError, json.JSONDecodeError):
                observed_provenance = None
            reusable_evaluation = observed_provenance == expected_provenance
        if not reusable_evaluation:
            _run_streaming(
                [
                    sys.executable,
                    "scripts/evaluate.py",
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--weights",
                    "ema",
                    "--split",
                    split,
                    "--output",
                    str(output_path),
                    "--seed",
                    str(EVALUATION_SEED),
                    "--mc-steps",
                    str(mc_steps),
                ],
                root=root,
                log_path=work_dir
                / "logs"
                / f"evaluate_{variant}_{role}.log",
            )
            write_json(provenance_path, expected_provenance)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if int(payload.get("num_patients", 0)) <= 0:
            raise ValueError(f"{variant} {role} evaluation has no patients")
        evaluations[role] = {
            "path": output_path.as_posix(),
            "sha256": file_sha256(output_path),
            "provenance": provenance_path.as_posix(),
            "provenance_sha256": file_sha256(provenance_path),
            "num_patients": int(payload["num_patients"]),
        }
    record = {
        "variant": variant,
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
            "canonical_sha256": config_canonical_sha,
        },
        "checkpoint": {
            "path": checkpoint_path.as_posix(),
            "sha256": checkpoint_sha,
            "epoch": int(checkpoint["epoch"]),
            "weights": "ema",
            "cache_metadata_sha256": expected_lineage[
                "cache_metadata_sha256"
            ],
        },
        "evaluations": evaluations,
        "fixed_endpoint": True,
        "validation_used_for_selection": False,
    }
    write_json(work_dir / "runs" / f"{variant}.json", record)
    return record


def load_evaluation(record: Mapping[str, Any], role: str) -> dict[str, Any]:
    path = Path(str(record["evaluations"][role]["path"]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(
        payload.get("per_patient"), dict
    ):
        raise ValueError(f"Invalid evaluation payload: {path}")
    return payload


def paired_patient_effects(
    left_eval: Mapping[str, Any],
    right_eval: Mapping[str, Any],
    *,
    metric: str,
    lower_is_better: bool,
) -> list[dict[str, Any]]:
    """Return positive-when-left-is-better effects for matched patients."""

    key = metric if metric.endswith("_mean") else f"{metric}_mean"
    left = left_eval.get("per_patient", {})
    right = right_eval.get("per_patient", {})
    patients = sorted(set(left) & set(right))
    rows = []
    for patient in patients:
        left_value = left[patient].get(key)
        right_value = right[patient].get(key)
        if not isinstance(left_value, (int, float)) or not isinstance(
            right_value, (int, float)
        ):
            continue
        if not np.isfinite(float(left_value)) or not np.isfinite(
            float(right_value)
        ):
            continue
        effect = (
            float(right_value) - float(left_value)
            if lower_is_better
            else float(left_value) - float(right_value)
        )
        rows.append(
            {
                "patient_id": patient,
                "metric": key,
                "left": float(left_value),
                "right": float(right_value),
                "effect_left_better": effect,
            }
        )
    return rows


def effect_statistics(
    effects: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int = 10_000,
) -> dict[str, Any]:
    values = [float(row["effect_left_better"]) for row in effects]
    result = bootstrap_mean(values, seed=seed, replicates=replicates)
    result["sign_flip_p"] = sign_flip_p(
        values,
        seed=seed + 1,
        replicates=replicates,
    )
    return result


def calibration_absolute_margin(
    effects: Sequence[Mapping[str, Any]],
    *,
    quantile: float = 0.95,
) -> float:
    values = np.asarray(
        [abs(float(row["effect_left_better"])) for row in effects],
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]
    if values.size < 5:
        raise RuntimeError("At least five calibration patients are required")
    return float(np.quantile(values, quantile))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_formal_context(
    *,
    root: Path,
    contract_path: Path,
    h2_decision_path: Path,
    curriculum_decision_path: Path,
    cache_dir: Path,
    cache_lineage: Path,
    base_config: Path,
) -> dict[str, Any]:
    """Resolve and cross-check the immutable inputs shared by H5/H6."""

    contract = json.loads(contract_path.read_text(encoding="utf-8-sig"))
    h2 = json.loads(h2_decision_path.read_text(encoding="utf-8-sig"))
    curriculum_decision = json.loads(
        curriculum_decision_path.read_text(encoding="utf-8-sig")
    )
    contract_sha = contract.get("contract_sha256")
    for label, payload in (
        ("H2", h2),
        ("curriculum", curriculum_decision),
    ):
        if payload.get("decision") != "PASS" or not payload.get(
            "next_stage_allowed"
        ):
            raise RuntimeError(f"{label} is not PASS")
        if payload.get("dataset_contract_sha256") != contract_sha:
            raise RuntimeError(f"{label} dataset contract mismatch")
    mechanism_manifest = h2_decision_path.parent / "mechanism_split_manifest.csv"
    analysis_spec_path = h2_decision_path.parent / "analysis_spec.json"
    analysis_spec = json.loads(
        analysis_spec_path.read_text(encoding="utf-8-sig")
    )
    declared_manifest = analysis_spec.get("derived_manifest", {})
    if file_sha256(mechanism_manifest) != declared_manifest.get("sha256"):
        raise RuntimeError("H2 mechanism manifest fingerprint mismatch")
    mean_checkpoint = Path(str(h2.get("downstream_mean_checkpoint", "")))
    if not mean_checkpoint.is_absolute():
        mean_checkpoint = resolve(root, mean_checkpoint)
    declared_excluded = h2.get("checkpoints", {}).get("excluded", {})
    if (
        not mean_checkpoint.is_file()
        or file_sha256(mean_checkpoint) != declared_excluded.get("sha256")
    ):
        raise RuntimeError("H2 downstream mean checkpoint fingerprint mismatch")
    curriculum_path = Path(
        str(curriculum_decision.get("curriculum", {}).get("path", ""))
    )
    if not curriculum_path.is_absolute():
        curriculum_path = resolve(root, curriculum_path)
    if (
        not curriculum_path.is_file()
        or file_sha256(curriculum_path)
        != curriculum_decision.get("curriculum", {}).get("sha256")
    ):
        raise RuntimeError("Frozen curriculum fingerprint mismatch")
    curriculum = json.loads(curriculum_path.read_text(encoding="utf-8-sig"))
    for required in (
        contract_path,
        cache_dir,
        cache_lineage,
        base_config,
        mechanism_manifest,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    return {
        "contract": contract,
        "contract_sha256": contract_sha,
        "h2": h2,
        "curriculum_decision": curriculum_decision,
        "curriculum": curriculum,
        "curriculum_path": curriculum_path,
        "mechanism_manifest": mechanism_manifest,
        "mean_checkpoint": mean_checkpoint,
        "cache_dir": cache_dir,
        "cache_lineage": cache_lineage,
        "base_config": base_config,
        "contract_path": contract_path,
    }
