from __future__ import annotations

import hashlib
import json
import random
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
    PHASE_OBSERVATION_EPOCHS,
    phase_for_epoch,
    read_metrics_jsonl,
    summarize_run,
)
from src.model.frequency.h3_native_null_schedule import (
    H3_V2_BAND_ORDER,
    H3_V2_ROUTE_ORDER,
)
from src.model.frequency.prior_anchor_schedule import (
    DIRECT_PNG_PREVIEW_PATH_POLICY,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _base_config() -> dict:
    payload = yaml.safe_load(
        (REPO_ROOT / DEFAULT_CONFIG).read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    return payload


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_cache_lineage_contract(root: Path) -> dict:
    contract_body = {
        "manifest": {"semantic_sha256": "1" * 64},
        "raw_png": {"combined_sha256": "2" * 64},
        "preprocessing_config_sha256": "3" * 64,
    }
    contract = {
        **contract_body,
        "contract_sha256": _canonical_sha256(contract_body),
    }
    contract_path = root / "configs" / "test_dataset_contract.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    lineage_body = {
        "lineage_type": "verified_png_tensor_cache",
        "manifest_semantic_sha256": (
            contract["manifest"]["semantic_sha256"]
        ),
        "raw_png_combined_sha256": (
            contract["raw_png"]["combined_sha256"]
        ),
        "preprocessing_config_sha256": (
            contract["preprocessing_config_sha256"]
        ),
        "dataset_contract_sha256": contract["contract_sha256"],
        "cache_payload_sha256": "4" * 64,
    }
    lineage = {
        **lineage_body,
        "cache_metadata_sha256": _canonical_sha256(lineage_body),
    }
    cache_dir = root / "cache" / "tensors_main"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "cache_lineage.json").write_text(
        json.dumps(lineage),
        encoding="utf-8",
    )
    return lineage


def _preview_payload(output_dir: str) -> dict:
    partition_hash = "b" * 64
    return {
        "schema_version": 1,
        "pipeline_id": "H3_DIRECT_PNG_PRIOR_PREVIEW_V1",
        "generated_at_utc": "2026-07-25T00:00:00+00:00",
        "decision": "PREVIEW_ONLY",
        "preview_only": True,
        "inference_schedule_allowed": False,
        "production_activation_allowed": False,
        "formal_h3_v2_claim_allowed": False,
        "reason": "unit-test preview",
        "paths": {
            "png_root": "Data/data",
            "manifest": "main_data/split_manifest.csv",
            "dataset_contract": "configs/dataset_contract_stage0a_v1.json",
            "mean_checkpoint": "checkpoints/mean.pt",
            "output_dir": output_dir,
        },
        "path_policy": DIRECT_PNG_PREVIEW_PATH_POLICY,
        "input_contract": {
            "source": "raw_png_direct",
            "cache_read": False,
            "cache_written": False,
            "manifest_file_sha256": "a" * 64,
            "mechanism_partition_sha256": partition_hash,
            "image_size": 192,
            "ct_normalization": "grayscale_uint8/127.5-1",
            "pet_normalization": "(255-grayscale_uint8)/127.5-1",
            "mask_normalization": "nearest_resize_then_uint8>127",
            "physical_split_policy": (
                "index_all_split_folders_by_sample_id"
            ),
            "raw_png_fingerprint": {
                "algorithm": "sha256",
                "canonicalization": "unit-test",
                "sha256": "c" * 64,
                "file_count": 18,
            },
        },
        "dataset_contract": {
            "contract_name": "ct_pet_png_dataset_contract",
            "contract_sha256": "d" * 64,
            "checks": {
                "raw_png_file_count": True,
                "raw_png_combined_sha256": True,
                "manifest_file_sha256": True,
            },
        },
        "checkpoint": {
            "file_sha256": "e" * 64,
            "epoch": 1,
            "mean_config": {"pathology_policy": "excluded"},
            "mechanism_partition_sha256": partition_hash,
            "has_data_lineage": True,
            "checkpoint_lineage_verified_for_current_png_run": False,
        },
        "quality_control": {"patients": 3},
        "analysis": {
            "analysis_seed": 17,
            "num_train_timesteps": 1000,
            "m_schedule": "linear",
            "sigma_scale": 1.0,
            "band_order": list(H3_V2_BAND_ORDER),
            "patient_weighting": "equal_patient_weight",
            "noise_identity": "unit-test",
            "recoverability": "unit-test",
            "active_transform": (
                "clip((recoverability-0.5)/0.5,0,1)"
            ),
            "shape_constraint": (
                "per-band isotonic non-increasing over timestep"
            ),
            "dense_curve_method": (
                "exact A=<x,x>, B=<e,e>, C=<x,e> sufficient-statistic "
                "evaluation; no timestep interpolation"
            ),
        },
        "route_mapping_preview": {
            "route_order": list(H3_V2_ROUTE_ORDER),
            "formula": "[a,0,1-a]",
            "shallow_probability": "structurally_zero",
        },
        "crossings": {},
        "preview_native_active_mass": {
            band: [0.5] * 1000 for band in H3_V2_BAND_ORDER
        },
        "outputs": {
            "curve_csv": f"{output_dir}/h3_prior_curves.csv",
            "patient_npz": (
                f"{output_dir}/h3_prior_patient_curves.npz"
            ),
            "plot_png": f"{output_dir}/h3_prior_curves.png",
        },
    }


def _write_state(
    run_dir: Path,
    config: dict,
    *,
    status: str,
    latest_epoch: int,
    current_stage: str | None,
) -> None:
    config_path = run_dir / "resolved_config.yaml"
    prior_path = run_dir / "prior" / "prior.json"
    metadata = config["prior_anchored_run"]
    state = {
        "pipeline_id": metadata["pipeline_id"],
        "run_dir": metadata["run_dir"],
        "status": status,
        "latest_epoch": latest_epoch,
        "current_stage": current_stage,
        "resolved_config": (
            f"{metadata['run_dir']}/resolved_config.yaml"
        ),
        "resolved_config_sha256": sha256_file(config_path),
        "prior_artifact": metadata["prior_artifact_path"],
        "prior_artifact_sha256": sha256_file(prior_path),
    }
    metrics_path = run_dir / "training_metrics.jsonl"
    if metrics_path.is_file():
        state["training_metrics"] = (
            f"{metadata['run_dir']}/training_metrics.jsonl"
        )
        state["training_metrics_sha256"] = sha256_file(metrics_path)
    (run_dir / "state.json").write_text(
        json.dumps(state),
        encoding="utf-8",
    )


def _make_run(tmp_path: Path) -> tuple[Path, dict]:
    _write_cache_lineage_contract(tmp_path)
    run_dir = tmp_path / "results" / "runs" / "run-a"
    prior_path = run_dir / "prior" / "prior.json"
    prior_path.parent.mkdir(parents=True)
    output_dir = prior_path.parent.relative_to(tmp_path).as_posix()
    prior_path.write_text(
        json.dumps(_preview_payload(output_dir)),
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
    config["data"].update(
        {
            "cache_dir": "cache/tensors_main",
            "cache_lineage": "cache/tensors_main/cache_lineage.json",
            "dataset_contract": "configs/test_dataset_contract.json",
        }
    )
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    _write_state(
        run_dir,
        config,
        status="INTERRUPTED",
        latest_epoch=40,
        current_stage="training",
    )
    return run_dir, config


def _write_checkpoint(
    run_dir: Path,
    config: dict,
    epoch: int,
) -> None:
    path = run_dir / "checkpoints" / f"ckpt_epoch{epoch:04d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    generator_state = torch.Generator(device="cpu").get_state()
    cuda_states = (
        [
            torch.Generator(device=f"cuda:{index}").get_state().cpu()
            for index in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [generator_state.clone()]
    )
    repository_root = run_dir
    run_relative = Path(config["prior_anchored_run"]["run_dir"])
    for _ in run_relative.parts:
        repository_root = repository_root.parent
    lineage_path = (
        repository_root / Path(config["data"]["cache_lineage"])
    )
    data_lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    torch.save(
        {
            "model": {},
            "optimizer": {},
            "scheduler": {},
            "ema": {},
            "epoch": epoch,
            "step": epoch * 3,
            "config": config,
            "data_lineage": data_lineage,
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
                "torch_cuda_all": cuda_states,
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


def _metrics_record(epoch: int) -> dict:
    evaluated = epoch % 10 == 0
    return {
        "schema_version": 1,
        "epoch": epoch,
        "step": epoch * 3,
        "phase": phase_for_epoch(epoch),
        "train": {
            "loss/total": 1.0 / epoch,
            "frequency/route_native_mass": 0.4,
            "frequency/route_shallow_mass": 0.1,
            "frequency/route_null_mass": 0.5,
        },
        "eval": {"loss/total": 1.5 / epoch} if evaluated else None,
        "validation": {"val/mae": 0.5 / epoch} if evaluated else None,
    }


def _write_complete_metrics(run_dir: Path) -> None:
    records = [_metrics_record(epoch) for epoch in range(1, 101)]
    (run_dir / "training_metrics.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def _replace_complete_metrics(
    run_dir: Path,
    config: dict,
    records: list[dict],
) -> None:
    (run_dir / "training_metrics.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    _write_state(
        run_dir,
        config,
        status="COMPLETE",
        latest_epoch=100,
        current_stage=None,
    )


def _complete_run(run_dir: Path, config: dict) -> None:
    for epoch in PHASE_OBSERVATION_EPOCHS:
        _write_checkpoint(run_dir, config, epoch)
    _write_complete_metrics(run_dir)
    _write_state(
        run_dir,
        config,
        status="COMPLETE",
        latest_epoch=100,
        current_stage=None,
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


def test_epoch_100_checkpoint_alone_does_not_mark_run_complete(
    tmp_path: Path,
) -> None:
    run_dir, config = _make_run(tmp_path)
    for epoch in (10, 20, 30, 40, 100):
        _write_checkpoint(run_dir, config, epoch)

    summary = summarize_run(
        "results/runs/run-a",
        root=tmp_path,
        write=False,
    )

    assert summary["decision"] == "INCOMPLETE"
    assert summary["checkpoints"]["final_checkpoint_valid"] is True
    assert summary["checkpoints"]["missing_phase_checkpoints"] == []
    assert summary["runner_state"]["status"] == "INTERRUPTED"
    assert summary["metrics_jsonl"]["status"] == "NOT_WRITTEN"


def test_complete_requires_every_run_contract(tmp_path: Path) -> None:
    run_dir, config = _make_run(tmp_path)
    _complete_run(run_dir, config)

    summary = summarize_run(
        "results/runs/run-a",
        root=tmp_path,
        write=False,
    )

    assert summary["decision"] == "COMPLETE"
    assert all(summary["completion_checks"].values())
    assert summary["runner_state"]["contract_valid"] is True
    assert (
        summary["runner_state"]["checks"]["training_metrics_path_matches"]
        is True
    )
    assert (
        summary["runner_state"]["checks"][
            "training_metrics_sha256_matches"
        ]
        is True
    )
    assert summary["prior"]["status"] == "PASS"
    assert summary["metrics_jsonl"]["status"] == "PASS"
    state = json.loads(
        (run_dir / "state.json").read_text(encoding="utf-8")
    )
    assert (
        summary["metrics_jsonl"]["sha256"]
        == state["training_metrics_sha256"]
    )
    assert summary["metrics_jsonl"]["epochs"] == list(range(1, 101))
    assert set(summary["metrics_jsonl"]["phase_observations"]) == {
        "10",
        "20",
        "30",
        "40",
        "100",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("status", "INTERRUPTED"),
        ("latest_epoch", 99),
        ("current_stage", "training"),
        ("resolved_config_sha256", "0" * 64),
        ("prior_artifact_sha256", "1" * 64),
        ("training_metrics", "results/runs/other/training_metrics.jsonl"),
        ("training_metrics_sha256", "2" * 64),
    ),
)
def test_invalid_runner_state_never_completes(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    run_dir, config = _make_run(tmp_path)
    _complete_run(run_dir, config)
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state[field] = value
    state_path.write_text(json.dumps(state), encoding="utf-8")

    summary = summarize_run(
        "results/runs/run-a",
        root=tmp_path,
        write=False,
    )

    assert summary["decision"] == "INCOMPLETE"
    assert summary["runner_state"]["contract_valid"] is False


@pytest.mark.parametrize(
    "missing_field",
    ("training_metrics", "training_metrics_sha256"),
)
def test_complete_state_requires_metrics_path_and_hash(
    tmp_path: Path,
    missing_field: str,
) -> None:
    run_dir, config = _make_run(tmp_path)
    _complete_run(run_dir, config)
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.pop(missing_field)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    summary = summarize_run(
        "results/runs/run-a",
        root=tmp_path,
        write=False,
    )

    assert summary["decision"] == "INCOMPLETE"
    checks = summary["runner_state"]["checks"]
    assert checks["training_metrics_fields_complete"] is False
    assert summary["runner_state"]["contract_valid"] is False


@pytest.mark.parametrize(
    "non_finite",
    (float("nan"), float("inf"), float("-inf")),
)
def test_metrics_jsonl_rejects_non_finite_json_constants(
    tmp_path: Path,
    non_finite: float,
) -> None:
    run_dir, config = _make_run(tmp_path)
    _complete_run(run_dir, config)
    metrics_path = run_dir / "training_metrics.jsonl"
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    records[49]["train"]["loss/total"] = non_finite
    metrics_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    # Refresh the state hash so this test isolates JSON parsing rather than
    # failing only because the ledger changed after finalization.
    _write_state(
        run_dir,
        config,
        status="COMPLETE",
        latest_epoch=100,
        current_stage=None,
    )

    summary = summarize_run(
        "results/runs/run-a",
        root=tmp_path,
        write=False,
    )

    assert summary["decision"] == "INCOMPLETE"
    assert summary["runner_state"]["contract_valid"] is True
    assert summary["metrics_jsonl"]["status"] == "PARTIAL"
    assert summary["metrics_jsonl"]["invalid_lines"][0]["line"] == 50
    assert "non-finite JSON value" in (
        summary["metrics_jsonl"]["invalid_lines"][0]["error"]
    )


def test_metrics_steps_must_increase_strictly(tmp_path: Path) -> None:
    run_dir, config = _make_run(tmp_path)
    _complete_run(run_dir, config)
    records = [_metrics_record(epoch) for epoch in range(1, 101)]
    records[49]["step"] = records[48]["step"]
    _replace_complete_metrics(run_dir, config, records)

    summary = summarize_run(
        "results/runs/run-a",
        root=tmp_path,
        write=False,
    )

    assert summary["decision"] == "INCOMPLETE"
    assert summary["metrics_jsonl"]["status"] == "PARTIAL"
    error = summary["metrics_jsonl"]["contract_errors"][0]
    assert error["line"] == 50
    assert "step must increase strictly" in error["errors"][0]


@pytest.mark.parametrize(
    ("epoch", "section", "value"),
    (
        (1, "train", ["not", "flat"]),
        (10, "eval", {"nested": 1.0}),
        (10, "validation", True),
    ),
)
def test_metric_sections_allow_only_flat_finite_scalars_or_null(
    tmp_path: Path,
    epoch: int,
    section: str,
    value: object,
) -> None:
    run_dir, config = _make_run(tmp_path)
    _complete_run(run_dir, config)
    records = [_metrics_record(item) for item in range(1, 101)]
    records[epoch - 1][section]["invalid"] = value
    _replace_complete_metrics(run_dir, config, records)

    summary = summarize_run(
        "results/runs/run-a",
        root=tmp_path,
        write=False,
    )

    assert summary["decision"] == "INCOMPLETE"
    error = summary["metrics_jsonl"]["contract_errors"][0]
    assert error["line"] == epoch
    assert "must be a finite number or null" in error["errors"][0]


@pytest.mark.parametrize("section", ("eval", "validation"))
def test_each_tenth_epoch_requires_eval_and_validation(
    tmp_path: Path,
    section: str,
) -> None:
    run_dir, config = _make_run(tmp_path)
    _complete_run(run_dir, config)
    records = [_metrics_record(epoch) for epoch in range(1, 101)]
    records[9][section] = None
    _replace_complete_metrics(run_dir, config, records)

    summary = summarize_run(
        "results/runs/run-a",
        root=tmp_path,
        write=False,
    )

    assert summary["decision"] == "INCOMPLETE"
    error = summary["metrics_jsonl"]["contract_errors"][0]
    assert error["line"] == 10
    assert (
        "epoch 10 lacks its configured validation observation"
        in error["errors"]
    )


@pytest.mark.parametrize("replacement", ("missing", None))
def test_phase_observations_require_canonical_route_masses(
    tmp_path: Path,
    replacement: str | None,
) -> None:
    run_dir, config = _make_run(tmp_path)
    _complete_run(run_dir, config)
    records = [_metrics_record(epoch) for epoch in range(1, 101)]
    key = "frequency/route_shallow_mass"
    if replacement == "missing":
        records[19]["train"].pop(key)
    else:
        records[19]["train"][key] = None
    _replace_complete_metrics(run_dir, config, records)

    summary = summarize_run(
        "results/runs/run-a",
        root=tmp_path,
        write=False,
    )

    assert summary["decision"] == "INCOMPLETE"
    error = summary["metrics_jsonl"]["contract_errors"][0]
    assert error["line"] == 20
    assert (
        "epoch 20 lacks canonical route-mass observations"
        in error["errors"]
    )
    assert "20" not in summary["metrics_jsonl"]["phase_observations"]


def test_final_metrics_step_must_match_final_checkpoint(
    tmp_path: Path,
) -> None:
    run_dir, config = _make_run(tmp_path)
    _complete_run(run_dir, config)
    records = [_metrics_record(epoch) for epoch in range(1, 101)]
    records[99]["step"] = 301
    _replace_complete_metrics(run_dir, config, records)

    summary = summarize_run(
        "results/runs/run-a",
        root=tmp_path,
        write=False,
    )

    assert summary["metrics_jsonl"]["status"] == "PASS"
    assert summary["metrics_jsonl"]["final_step"] == 301
    assert summary["checkpoints"]["latest"]["step"] == 300
    assert summary["completion_checks"][
        "metrics_final_step_matches_checkpoint"
    ] is False
    assert summary["decision"] == "INCOMPLETE"


def test_invalid_preview_contract_never_completes(tmp_path: Path) -> None:
    run_dir, config = _make_run(tmp_path)
    prior_path = run_dir / "prior" / "prior.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    prior["preview_only"] = False
    prior_path.write_text(json.dumps(prior), encoding="utf-8")
    new_hash = sha256_file(prior_path)
    config["prior_anchored_run"]["prior_artifact_sha256"] = new_hash
    router = config["modules"]["residual_frequency"]["cross_level_router"]
    router["h3_schedule_sha256"] = new_hash
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    _complete_run(run_dir, config)

    summary = summarize_run(
        "results/runs/run-a",
        root=tmp_path,
        write=False,
    )

    assert summary["decision"] == "INCOMPLETE"
    assert summary["prior"]["status"] == "INVALID"
    assert "preview_only must be True" in summary["prior"]["error"]


def test_configured_budget_must_be_exactly_100(tmp_path: Path) -> None:
    run_dir, config = _make_run(tmp_path)
    _complete_run(run_dir, config)
    config["training"]["num_epochs"] = 99
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    _write_state(
        run_dir,
        config,
        status="COMPLETE",
        latest_epoch=100,
        current_stage=None,
    )

    summary = summarize_run(
        "results/runs/run-a",
        root=tmp_path,
        write=False,
    )

    assert summary["decision"] == "INCOMPLETE"
    assert (
        summary["completion_checks"]["configuration_is_100_epochs"]
        is False
    )


def test_metrics_jsonl_reports_duplicate_and_invalid_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "schema_version": 1,
                        "epoch": 10,
                        "step": 30,
                        "phase": "prior_frozen",
                        "train": {"loss/total": 1.0},
                        "loss/total": 1.0,
                        "frequency/route_active_progress": 0.0,
                    }
                ),
                json.dumps({"epoch": 10, "loss/total": 0.9}),
                "{invalid",
                json.dumps({"loss/total": 0.8}),
                json.dumps(
                    {
                        "schema_version": 1,
                        "epoch": 20,
                        "step": 60,
                        "phase": "active_ramp",
                        "train": {
                            "loss/total": 0.8,
                            "frequency/route_native_mass": 0.4,
                            "frequency/route_shallow_mass": 0.1,
                            "frequency/route_null_mass": 0.5,
                        },
                        "eval": {"loss/total": 0.9},
                        "validation": {"val/mae": 0.2},
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


def test_metrics_contract_rejects_extra_epoch_and_incomplete_row(
    tmp_path: Path,
) -> None:
    run_dir, config = _make_run(tmp_path)
    _complete_run(run_dir, config)
    records = [_metrics_record(epoch) for epoch in range(1, 101)]
    records[39].pop("step")
    records[39].pop("phase")
    records[39]["train"] = {}
    lines = [json.dumps(record) for record in records]
    lines.extend(
        (
            json.dumps(_metrics_record(50)),
            json.dumps(_metrics_record(101)),
            "{invalid",
        )
    )
    metrics_path = run_dir / "training_metrics.jsonl"
    metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = read_metrics_jsonl(metrics_path)

    assert result["status"] == "PARTIAL"
    assert result["duplicate_epochs"] == [50]
    assert result["out_of_range_epochs"] == [101]
    assert result["invalid_lines"][0]["line"] == 103
    assert result["epoch_sequence_valid"] is False
    assert result["contract_errors"][0] == {
        "line": 40,
        "epoch": 40,
        "errors": [
            "step must be a non-negative integer",
            "phase must be 'destination_ramp' for epoch 40",
            "train must be a non-empty metrics object",
            "epoch 40 lacks canonical route-mass observations",
        ],
    }
    assert result["contract_errors"][1] == {
        "line": 101,
        "epoch": 50,
        "errors": [
            "step must increase strictly; found 150 after 300",
        ],
    }
    assert "40" not in result["phase_observations"]

    summary = summarize_run(
        "results/runs/run-a",
        root=tmp_path,
        write=False,
    )
    assert summary["decision"] == "INCOMPLETE"
    assert (
        summary["completion_checks"]["metrics_exactly_epochs_1_through_100"]
        is False
    )
    assert (
        summary["completion_checks"]["phase_observations_complete"]
        is False
    )


def test_metrics_jsonl_surfaces_trainer_nested_sections(tmp_path: Path) -> None:
    """The trainer writes ``{epoch, step, train:{...}, eval:{...}, validation:{...}}``.

    The summarizer must recurse into those sections; a top-level-only scan
    would hide every router metric in a real run.
    """
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "schema_version": 1,
                        "epoch": 10,
                        "step": 100,
                        "phase": "prior_frozen",
                        "train": {
                            "loss/total": 0.30,
                            "frequency/prior_anchor_active_progress": 0.1,
                            "frequency/prior_anchor_native_mass": 0.05,
                            "frequency/prior_anchor_null_mass": 0.90,
                            "frequency/route_native_mass": 0.05,
                            "frequency/route_shallow_mass": 0.05,
                            "frequency/route_null_mass": 0.90,
                            "grad/prior_active_final": 1.0e-3,
                            "perf/lr": 1.0e-4,
                        },
                        "eval": {"loss/total": 0.42},
                        "validation": {
                            "val/mae": 0.12,
                            "val/stripe_score": 0.5,
                            "val/lesion_peak_error_norm": 0.08,
                        },
                    }
                ),
                json.dumps(
                    {
                        "schema_version": 1,
                        "epoch": 20,
                        "step": 200,
                        "phase": "active_ramp",
                        "train": {
                            "loss/total": 0.20,
                            "frequency/prior_anchor_active_progress": 1.0,
                            "frequency/prior_anchor_native_mass": 0.50,
                            "frequency/prior_anchor_shallow_mass": 0.10,
                            "frequency/prior_anchor_null_mass": 0.40,
                            "frequency/route_native_mass": 0.50,
                            "frequency/route_shallow_mass": 0.10,
                            "frequency/route_null_mass": 0.40,
                            "grad/prior_active_final": 1.0e-3,
                            "perf/lr": 1.0e-4,
                        },
                        "eval": {"loss/total": 0.30},
                        "validation": {"val/mae": 0.09},
                    }
                ),
            )
        ),
        encoding="utf-8",
    )

    result = read_metrics_jsonl(path)

    # Two records out of the 100-epoch contract is intentionally PARTIAL:
    # missing epochs must never be reported as PASS.
    assert result["status"] == "PARTIAL"
    assert 1 in result["missing_epochs"]
    assert 11 in result["missing_epochs"]
    latest = result["latest"]["metrics"]
    assert latest["epoch"] == 20
    assert latest["step"] == 200
    # Router diagnostics nested under ``train/`` must be surfaced.
    assert latest["train/frequency/prior_anchor_active_progress"] == pytest.approx(1.0)
    assert latest["train/frequency/prior_anchor_native_mass"] == pytest.approx(0.50)
    assert latest["train/frequency/prior_anchor_null_mass"] == pytest.approx(0.40)
    assert latest["train/loss/total"] == pytest.approx(0.20)
    assert latest["train/grad/prior_active_final"] == pytest.approx(1.0e-3)
    assert latest["validation/val/mae"] == pytest.approx(0.09)
    # The phase observation at epoch 10 retains its nested prior metrics.
    epoch10 = result["phase_observations"]["10"]["metrics"]
    assert epoch10["train/frequency/prior_anchor_active_progress"] == pytest.approx(0.1)
    assert epoch10["validation/val/stripe_score"] == pytest.approx(0.5)


def test_metrics_jsonl_keeps_flat_records_backward_compatible(
    tmp_path: Path,
) -> None:
    """Legacy flat records (no train/eval/validation sections) still parse."""
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps(
            {
                "epoch": 10,
                "loss/total": 1.0,
                "frequency/route_active_progress": 0.0,
                "router/prior_anchor": 0.2,
            }
        ),
        encoding="utf-8",
    )

    result = read_metrics_jsonl(path)

    metrics = result["latest"]["metrics"]
    assert metrics["epoch"] == 10
    assert metrics["loss/total"] == pytest.approx(1.0)
    assert metrics["frequency/route_active_progress"] == pytest.approx(0.0)
    assert metrics["router/prior_anchor"] == pytest.approx(0.2)


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
