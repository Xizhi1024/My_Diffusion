from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts.run_prior_anchored_router_100e import (
    DEFAULT_CONFIG,
    materialize_run_config,
    sha256_file,
)
from scripts.summarize_prior_anchored_run import (
    phase_for_epoch,
    read_metrics_jsonl,
    summarize_run,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _base_config() -> dict:
    payload = yaml.safe_load(
        (REPO_ROOT / DEFAULT_CONFIG).read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    return payload


def _make_run(tmp_path: Path) -> tuple[Path, dict]:
    run_dir = tmp_path / "results" / "runs" / "run-a"
    prior_path = run_dir / "prior" / "prior.json"
    prior_path.parent.mkdir(parents=True)
    prior_path.write_text(
        json.dumps(
            {
                "pipeline_id": "H3_DIRECT_PNG_PRIOR_PREVIEW_V1",
                "decision": "PREVIEW_ONLY",
                "preview_only": True,
                "crossings": {},
            }
        ),
        encoding="utf-8",
    )
    config = materialize_run_config(
        _base_config(),
        run_dir=run_dir,
        mean_checkpoint="checkpoints/mean.pt",
        prior_path="results/runs/run-a/prior/prior.json",
        prior_sha256=sha256_file(prior_path),
        root=tmp_path,
    )
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "state.json").write_text(
        json.dumps({"status": "INTERRUPTED", "latest_epoch": 40}),
        encoding="utf-8",
    )
    return run_dir, config


def _write_checkpoint(
    run_dir: Path,
    config: dict,
    epoch: int,
) -> None:
    path = run_dir / "checkpoints" / f"ckpt_epoch{epoch:04d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {},
            "optimizer": {},
            "scheduler": {},
            "ema": {},
            "epoch": epoch,
            "step": epoch * 3,
            "config": config,
            "data_lineage": {"cache_metadata_sha256": "c" * 64},
        },
        path,
    )


def test_phase_boundaries_are_unambiguous() -> None:
    assert phase_for_epoch(0) == "not_started"
    assert phase_for_epoch(10) == "prior_frozen"
    assert phase_for_epoch(11) == "active_ramp"
    assert phase_for_epoch(20) == "active_ramp"
    assert phase_for_epoch(21) == "active_only_hold"
    assert phase_for_epoch(30) == "active_only_hold"
    assert phase_for_epoch(31) == "destination_ramp"
    assert phase_for_epoch(40) == "destination_ramp"
    assert phase_for_epoch(41) == "full_adaptive"
    assert phase_for_epoch(100) == "full_adaptive"


def test_partial_run_summary_reports_latest_and_missing_boundaries(
    tmp_path: Path,
) -> None:
    run_dir, config = _make_run(tmp_path)
    for epoch in (10, 20, 40):
        _write_checkpoint(run_dir, config, epoch)

    summary = summarize_run(
        "results/runs/run-a",
        root=tmp_path,
        write=False,
    )

    assert summary["decision"] == "INCOMPLETE"
    assert summary["checkpoints"]["valid_epochs"] == [10, 20, 40]
    assert summary["checkpoints"]["latest"]["epoch"] == 40
    assert summary["checkpoints"]["missing_phase_checkpoints"] == [30, 100]
    assert summary["prior"]["status"] == "PASS"
    assert summary["run_dir"] == "results/runs/run-a"
    assert summary["scientific_claim_allowed"] is False


def test_epoch_100_checkpoint_marks_run_summary_complete(tmp_path: Path) -> None:
    run_dir, config = _make_run(tmp_path)
    for epoch in (10, 20, 30, 40, 100):
        _write_checkpoint(run_dir, config, epoch)

    summary = summarize_run(
        "results/runs/run-a",
        root=tmp_path,
        write=False,
    )

    assert summary["decision"] == "COMPLETE"
    assert summary["checkpoints"]["final_checkpoint_valid"] is True
    assert summary["checkpoints"]["missing_phase_checkpoints"] == []


def test_metrics_jsonl_reports_duplicate_and_invalid_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "epoch": 10,
                        "loss/total": 1.0,
                        "frequency/route_active_progress": 0.0,
                    }
                ),
                json.dumps({"epoch": 10, "loss/total": 0.9}),
                "{invalid",
                json.dumps({"loss/total": 0.8}),
                json.dumps(
                    {
                        "epoch": 20,
                        "router/prior_anchor": 0.2,
                        "unrelated_tensor": [1, 2, 3],
                    }
                ),
            )
        ),
        encoding="utf-8",
    )

    result = read_metrics_jsonl(path)

    assert result["status"] == "PARTIAL"
    assert result["duplicate_epochs"] == [10]
    assert [row["line"] for row in result["invalid_lines"]] == [3, 4]
    assert result["latest"]["epoch"] == 20
    assert "unrelated_tensor" not in result["latest"]["metrics"]
    assert result["phase_observations"]["20"]["phase"] == "active_ramp"


def test_summary_write_is_atomic_and_relative(tmp_path: Path) -> None:
    run_dir, config = _make_run(tmp_path)
    _write_checkpoint(run_dir, config, 10)

    summarize_run(
        "results/runs/run-a",
        output="results/runs/run-a/observations/summary.json",
        root=tmp_path,
        write=True,
    )

    output = run_dir / "observations" / "summary.json"
    assert output.is_file()
    assert not output.with_name(".summary.json.tmp").exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["run_dir"] == "results/runs/run-a"


def test_summary_rejects_absolute_output_path(tmp_path: Path) -> None:
    run_dir, config = _make_run(tmp_path)
    _write_checkpoint(run_dir, config, 10)

    with pytest.raises(ValueError, match="Absolute project path"):
        summarize_run(
            "results/runs/run-a",
            output=tmp_path / "absolute.json",
            root=tmp_path,
            write=True,
        )
