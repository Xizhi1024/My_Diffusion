from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from pathlib import Path

import pytest
import torch
import yaml

from scripts import run_prior_anchored_router_100e as runner_module
from scripts.run_prior_anchored_router_100e import (
    DEFAULT_CONFIG,
    PIPELINE_ID,
    RunnerError,
    build_static_preflight,
    find_latest_resume_checkpoint,
    inspect_resume_checkpoint,
    materialize_run_config,
    normalize_repo_relative,
    prepare_fresh_run,
    select_latest_resumable_run,
    select_mean_checkpoint,
    validate_complete_metrics_jsonl,
    validate_prior_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _checkpoint_lineage() -> dict:
    lineage = {
        "lineage_type": "verified_png_tensor_cache",
        "manifest_semantic_sha256": "1" * 64,
        "raw_png_combined_sha256": "2" * 64,
        "preprocessing_config_sha256": "3" * 64,
        "dataset_contract_sha256": "4" * 64,
        "cache_payload_sha256": "5" * 64,
    }
    canonical = json.dumps(
        lineage,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    lineage["cache_metadata_sha256"] = hashlib.sha256(canonical).hexdigest()
    return lineage


def _write_current_lineage(root: Path) -> dict:
    contract = {
        "manifest": {"semantic_sha256": "1" * 64},
        "raw_png": {"combined_sha256": "2" * 64},
        "preprocessing_config_sha256": "3" * 64,
    }
    contract_body = json.dumps(
        contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    contract["contract_sha256"] = hashlib.sha256(contract_body).hexdigest()
    contract_path = root / "configs" / "dataset_contract_stage0a_v1.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    lineage = {
        "lineage_type": "verified_png_tensor_cache",
        "manifest_semantic_sha256": "1" * 64,
        "raw_png_combined_sha256": "2" * 64,
        "preprocessing_config_sha256": "3" * 64,
        "dataset_contract_sha256": contract["contract_sha256"],
        "cache_payload_sha256": "5" * 64,
    }
    lineage_body = json.dumps(
        lineage,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    lineage["cache_metadata_sha256"] = hashlib.sha256(
        lineage_body
    ).hexdigest()
    lineage_path = root / "cache" / "tensors_main" / "cache_lineage.json"
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    lineage_path.write_text(json.dumps(lineage), encoding="utf-8")
    return lineage


def _base_config() -> dict:
    payload = yaml.safe_load(
        (REPO_ROOT / DEFAULT_CONFIG).read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    return payload


def _materialized_config(root: Path, run_dir: Path) -> dict:
    return materialize_run_config(
        _base_config(),
        run_dir=run_dir,
        mean_checkpoint="checkpoints/mean.pt",
        prior_path=f"{run_dir.relative_to(root).as_posix()}/prior/prior.json",
        prior_sha256="a" * 64,
        root=root,
    )


def _write_checkpoint(
    path: Path,
    *,
    epoch: int,
    config: dict,
    lineage: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    generator_state = torch.Generator().get_state()
    torch.save(
        {
            "model": {},
            "optimizer": {},
            "scheduler": {},
            "ema": {},
            "epoch": epoch,
            "step": epoch * 7,
            "config": config,
            "data_lineage": lineage or _checkpoint_lineage(),
            "rng": {
                "schema_version": 2,
                "python_random": random.getstate(),
                "numpy": {
                    "bit_generator": "MT19937",
                    "state": torch.zeros(624, dtype=torch.int64),
                    "pos": 0,
                    "has_gauss": 0,
                    "cached_gaussian": 0.0,
                },
                "torch_cpu": generator_state.clone(),
                "torch_cuda_all": [generator_state.clone()],
                "loader_generators": {
                    "train_loader": generator_state.clone(),
                    "val_loader": generator_state.clone(),
                },
                "sampler_generators": {
                    "train_loader": generator_state.clone(),
                    "val_loader": None,
                },
            },
        },
        path,
    )


def test_preflight_is_static_and_does_not_require_cloud_files(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "configs" / "template.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        (REPO_ROOT / DEFAULT_CONFIG).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    result = build_static_preflight(
        config_path="configs/template.yaml",
        root=tmp_path,
    )

    assert result["decision"] == "STATIC_PREFLIGHT_PASS"
    assert result["cloud_runtime_started"] is False
    assert result["cloud_files_required_for_this_check"] is False
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    "value",
    [
        "/tmp/run",
        "C:/cloud/run",
        r"C:\cloud\run",
        r"\\server\share\run",
        "../outside",
        "results/../../outside",
    ],
)
def test_project_paths_reject_absolute_and_escaping_values(
    tmp_path: Path,
    value: str,
) -> None:
    with pytest.raises(ValueError):
        normalize_repo_relative(value, root=tmp_path)


def test_run_owned_config_uses_only_relative_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "runs" / "run-a"
    run_dir.mkdir(parents=True)

    config = _materialized_config(tmp_path, run_dir)

    assert config["training"]["num_epochs"] == 100
    assert config["training"]["checkpoint_dir"] == (
        "results/runs/run-a/checkpoints"
    )
    assert config["runtime"]["sample_dir"] == "results/runs/run-a/samples"
    assert config["runtime"]["training_metrics_jsonl"] == (
        "results/runs/run-a/training_metrics.jsonl"
    )
    assert config["runtime"]["dataloader_seed"] == 42
    router = config["modules"]["residual_frequency"]["cross_level_router"]
    assert router["policy"] == "prior_anchored_learned"
    assert router["prior_anchor_decay_end_epoch"] == 100
    assert router["h3_schedule_path"] == (
        "results/runs/run-a/prior/prior.json"
    )
    assert router["phase_observation_epochs"] == [10, 20, 30, 40, 100]
    assert not Path(config["training"]["checkpoint_dir"]).is_absolute()


def test_mean_checkpoint_selection_skips_non_excluded_candidate(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "checkpoints" / "invalid.pt"
    valid = tmp_path / "checkpoints" / "valid.pt"
    invalid.parent.mkdir(parents=True)
    torch.save({"mean_config": {"pathology_exclusion": {"enabled": False}}}, invalid)
    torch.save({"mean_config": {"pathology_exclusion": {"enabled": True}}}, valid)

    selected = select_mean_checkpoint(
        root=tmp_path,
        candidates=("checkpoints/invalid.pt", "checkpoints/valid.pt"),
    )

    assert selected == "checkpoints/valid.pt"


def test_resume_latest_selects_newest_incomplete_run(tmp_path: Path) -> None:
    runs = tmp_path / "results" / "runs"
    first = runs / "run-a"
    second = runs / "run-b"
    complete = runs / "run-c"
    for directory, status in (
        (first, "FAILED"),
        (second, "INTERRUPTED"),
        (complete, "COMPLETE"),
    ):
        directory.mkdir(parents=True)
        (directory / "resolved_config.yaml").write_text(
            "{}\n",
            encoding="utf-8",
        )
        (directory / "state.json").write_text(
            json.dumps({"status": status}),
            encoding="utf-8",
        )
    os.utime(first / "state.json", (10, 10))
    os.utime(second / "state.json", (20, 20))
    os.utime(complete / "state.json", (30, 30))

    selected = select_latest_resumable_run(
        "results/runs",
        root=tmp_path,
    )

    assert selected == "results/runs/run-b"


def test_fresh_launch_never_reuses_directory_with_checkpoint(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "results" / "runs" / "run-a"
    checkpoint = run_dir / "checkpoints" / "ckpt_epoch0010.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    with pytest.raises(RunnerError, match="Existing checkpoints"):
        prepare_fresh_run("results/runs/run-a", root=tmp_path)


def test_resume_does_not_silently_skip_corrupt_latest_checkpoint(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "results" / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    config = _materialized_config(tmp_path, run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    _write_checkpoint(
        checkpoint_dir / "ckpt_epoch0010.pt",
        epoch=10,
        config=config,
    )
    (checkpoint_dir / "ckpt_epoch0020.pt").write_bytes(b"not-a-checkpoint")

    with pytest.raises(RunnerError, match="unreadable"):
        find_latest_resume_checkpoint(
            checkpoint_dir,
            expected_config=config,
            verify_current_lineage=False,
        )


def test_resume_rejects_rng_tensor_that_cannot_be_restored(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "results" / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    config = _materialized_config(tmp_path, run_dir)
    checkpoint = run_dir / "checkpoints" / "ckpt_epoch0010.pt"
    _write_checkpoint(checkpoint, epoch=10, config=config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["rng"]["torch_cpu"] = torch.zeros(1, dtype=torch.uint8)
    torch.save(payload, checkpoint)

    with pytest.raises(RunnerError, match="not restorable"):
        find_latest_resume_checkpoint(
            checkpoint.parent,
            expected_config=config,
            verify_current_lineage=False,
        )


def test_resume_rejects_full_config_identity_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "results" / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    config = _materialized_config(tmp_path, run_dir)
    checkpoint = run_dir / "checkpoints" / "ckpt_epoch0010.pt"
    _write_checkpoint(checkpoint, epoch=10, config=config)
    changed = copy.deepcopy(config)
    changed["experiment"]["seed"] += 1

    with pytest.raises(RunnerError, match="configuration differs"):
        inspect_resume_checkpoint(
            checkpoint,
            expected_config=changed,
            root=tmp_path,
            verify_current_lineage=False,
        )


def test_resume_rejects_absolute_checkpoint_owned_resume_path(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "results" / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    config = _materialized_config(tmp_path, run_dir)
    checkpoint = run_dir / "checkpoints" / "ckpt_epoch0010.pt"
    _write_checkpoint(checkpoint, epoch=10, config=config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["config"]["training"]["resume_from"] = "C:/other/run/ckpt.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(RunnerError, match="repository-relative"):
        inspect_resume_checkpoint(
            checkpoint,
            expected_config=config,
            root=tmp_path,
            verify_current_lineage=False,
        )


def test_resume_rejects_lineage_that_differs_from_current_cache_metadata(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "results" / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    config = _materialized_config(tmp_path, run_dir)
    checkpoint_lineage = _write_current_lineage(tmp_path)
    checkpoint = run_dir / "checkpoints" / "ckpt_epoch0010.pt"
    _write_checkpoint(
        checkpoint,
        epoch=10,
        config=config,
        lineage=checkpoint_lineage,
    )
    current = dict(checkpoint_lineage)
    current["cache_payload_sha256"] = "9" * 64
    current.pop("cache_metadata_sha256")
    canonical = json.dumps(
        current,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    current["cache_metadata_sha256"] = hashlib.sha256(canonical).hexdigest()
    (
        tmp_path / "cache" / "tensors_main" / "cache_lineage.json"
    ).write_text(json.dumps(current), encoding="utf-8")

    with pytest.raises(RunnerError, match="differs from current cache"):
        inspect_resume_checkpoint(
            checkpoint,
            expected_config=config,
            root=tmp_path,
            verify_current_lineage=True,
        )


def test_resume_rejects_only_best_checkpoint_instead_of_restarting(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "results" / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    config = _materialized_config(tmp_path, run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    _write_checkpoint(
        checkpoint_dir / "ckpt_best_combined.pt",
        epoch=10,
        config=config,
    )

    with pytest.raises(RunnerError, match="refusing to restart"):
        find_latest_resume_checkpoint(
            checkpoint_dir,
            expected_config=config,
            verify_current_lineage=False,
        )


def test_resume_fails_closed_when_no_numbered_checkpoint_exists(
    tmp_path: Path,
) -> None:
    """Resume must never silently restart from epoch 0."""
    run_dir = tmp_path / "results" / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    config = _materialized_config(tmp_path, run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    # Fresh-run introspection: an empty dir is a legitimate "nothing yet".
    assert (
        find_latest_resume_checkpoint(
            checkpoint_dir,
            expected_config=config,
            require=False,
            verify_current_lineage=False,
        )
        is None
    )

    # Resume context: the same empty dir must fail closed, not restart.
    with pytest.raises(RunnerError, match="refusing to restart"):
        find_latest_resume_checkpoint(
            checkpoint_dir,
            expected_config=config,
            require=True,
            verify_current_lineage=False,
        )


def _write_metrics(path: Path, *, final_epoch: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        json.dumps(
            {
                "schema_version": 1,
                "epoch": epoch,
                "step": epoch * 7,
                "phase": runner_module._expected_router_phase(epoch),
                "train": {
                    "loss": 1.0 / epoch,
                    "frequency/route_native_mass": 0.4,
                    "frequency/route_shallow_mass": 0.1,
                    "frequency/route_null_mass": 0.5,
                },
                "eval": (
                    {"loss/total": 1.0 / epoch}
                    if epoch % 10 == 0
                    else None
                ),
                "validation": (
                    {"val/mae": 1.0 / epoch}
                    if epoch % 10 == 0
                    else None
                ),
            }
        )
        for epoch in range(1, final_epoch + 1)
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_complete_metrics_requires_exact_ordered_1_to_100(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "results" / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    config = _materialized_config(tmp_path, run_dir)
    metrics_path = run_dir / "training_metrics.jsonl"
    _write_metrics(metrics_path)

    result = validate_complete_metrics_jsonl(config, root=tmp_path)

    assert result["epochs"] == list(range(1, 101))
    assert result["path"] == metrics_path


def test_complete_metrics_rejects_checkpoint_only_completion(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "results" / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    config = _materialized_config(tmp_path, run_dir)
    _write_metrics(run_dir / "training_metrics.jsonl", final_epoch=99)

    with pytest.raises(RunnerError, match="exact ordered epoch ledger"):
        validate_complete_metrics_jsonl(config, root=tmp_path)


def test_complete_metrics_rejects_nonincreasing_steps(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "results" / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    config = _materialized_config(tmp_path, run_dir)
    metrics_path = run_dir / "training_metrics.jsonl"
    _write_metrics(metrics_path)
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[1]["step"] = rows[0]["step"]
    metrics_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RunnerError, match="increase strictly"):
        validate_complete_metrics_jsonl(config, root=tmp_path)


def test_complete_metrics_rejects_non_trainer_metric_values(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "results" / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    config = _materialized_config(tmp_path, run_dir)
    metrics_path = run_dir / "training_metrics.jsonl"
    _write_metrics(metrics_path)
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[2]["train"]["loss"] = ["not", "a", "trainer metric"]
    metrics_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RunnerError, match="finite number or null"):
        validate_complete_metrics_jsonl(config, root=tmp_path)


def test_complete_metrics_final_step_must_match_checkpoint(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "results" / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    config = _materialized_config(tmp_path, run_dir)
    _write_metrics(run_dir / "training_metrics.jsonl")

    with pytest.raises(RunnerError, match="does not match"):
        validate_complete_metrics_jsonl(
            config,
            root=tmp_path,
            expected_final_step=999,
        )


def test_fresh_setup_failure_is_persisted_in_runner_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_module, "_require_cuda", lambda: None)
    args = runner_module.parse_args(
        [
            "--config",
            "configs/missing.yaml",
            "--run-dir",
            "results/runs/run-a",
        ]
    )

    with pytest.raises(FileNotFoundError):
        runner_module.run_cloud(args, root=tmp_path)

    state = json.loads(
        (tmp_path / "results" / "runs" / "run-a" / "state.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["status"] == "FAILED"
    assert state["current_stage"] == "cloud_input_preflight"
    assert state["failure"].startswith("FileNotFoundError:")


def test_prior_artifact_requires_six_dense_curves(tmp_path: Path) -> None:
    path = tmp_path / "prior.json"
    path.write_text(
        json.dumps(
            {
                "pipeline_id": "H3_DIRECT_PNG_PRIOR_PREVIEW_V1",
                "decision": "PREVIEW_ONLY",
                "preview_native_active_mass": {
                    f"band_{index}": [0.5] * 1000 for index in range(6)
                },
            }
        ),
        encoding="utf-8",
    )

    payload = validate_prior_artifact(path)

    assert payload["decision"] == "PREVIEW_ONLY"
    assert PIPELINE_ID == "PRIOR_ANCHORED_ROUTER_100E_CLOUD_V1"
