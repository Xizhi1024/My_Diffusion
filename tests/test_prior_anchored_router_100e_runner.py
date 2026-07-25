from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import yaml

from scripts.run_prior_anchored_router_100e import (
    DEFAULT_CONFIG,
    PIPELINE_ID,
    RunnerError,
    build_static_preflight,
    find_latest_resume_checkpoint,
    materialize_run_config,
    normalize_repo_relative,
    prepare_fresh_run,
    select_latest_resumable_run,
    select_mean_checkpoint,
    validate_prior_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {},
            "optimizer": {},
            "scheduler": {},
            "ema": {},
            "epoch": epoch,
            "step": epoch * 7,
            "config": config,
            "data_lineage": {"cache_metadata_sha256": "b" * 64},
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
    router = config["modules"]["residual_frequency"]["cross_level_router"]
    assert router["policy"] == "prior_anchored_learned"
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
        )


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
