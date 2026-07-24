"""Contracts for standalone low-frequency PET mean pretraining."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import hashlib
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import FakeDataset
from src.model.frequency.haar import haar_dwt2
from src.model.mean_predictor import LowFrequencyPETPredictor


def _config():
    return {
        "experiment": {"seed": 42},
        "runtime": {"amp": False, "grad_clip_norm": 1.0},
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "lr_min": 1e-5,
        },
        "modules": {
            "conditional_mean": {
                "enabled": True,
                "levels": 2,
                "base_channels": 8,
                "charbonnier_eps": 1e-3,
            }
        },
    }


def _loaders():
    train = DataLoader(FakeDataset(4, image_size=32), batch_size=2, shuffle=False)
    val = DataLoader(FakeDataset(4, image_size=32), batch_size=2, shuffle=False)
    return train, val


def test_mean_target_is_exact_pet_ll2():
    from src.model.mean_pretraining import mean_target_ll2

    pet = torch.randn(2, 1, 32, 32)
    ll1, _ = haar_dwt2(pet)
    expected, _ = haar_dwt2(ll1)
    assert torch.equal(mean_target_ll2(pet), expected)


def test_mean_charbonnier_loss_is_finite_and_zero_error_equals_epsilon():
    from src.model.mean_pretraining import mean_charbonnier_loss

    target = torch.randn(2, 1, 8, 8)
    loss = mean_charbonnier_loss(target.clone(), target, epsilon=1e-3)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(1e-3)


def test_mean_pretrainer_updates_predictor_and_writes_versioned_checkpoints(tmp_path):
    from src.model.mean_pretraining import MeanPretrainer

    torch.manual_seed(0)
    predictor = LowFrequencyPETPredictor(base_channels=8)
    before = {name: value.clone() for name, value in predictor.state_dict().items()}
    train_loader, val_loader = _loaders()
    trainer = MeanPretrainer(
        predictor,
        train_loader,
        val_loader,
        _config(),
        device="cpu",
    )
    history = trainer.run(epochs=1, output_dir=tmp_path)

    assert len(history) == 1
    assert set(history[0]) == {"epoch", "train_loss", "val_loss", "learning_rate"}
    assert all(torch.isfinite(torch.tensor(value)) for value in history[0].values())
    assert any(
        not torch.equal(before[name], value)
        for name, value in predictor.state_dict().items()
    )

    best = torch.load(tmp_path / "mean_best.pt", map_location="cpu", weights_only=True)
    last = torch.load(tmp_path / "mean_last.pt", map_location="cpu", weights_only=True)
    assert set(best) == {"format_version", "model", "mean_config", "epoch", "val_loss"}
    assert best["format_version"] == 1
    assert best["epoch"] == 1
    assert torch.isfinite(torch.tensor(best["val_loss"]))
    assert last["format_version"] == 1
    saved_history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert saved_history == history


def test_mean_pretrainer_requires_validation_loader():
    from src.model.mean_pretraining import MeanPretrainer

    train_loader, _ = _loaders()
    with pytest.raises(ValueError, match="validation"):
        MeanPretrainer(
            LowFrequencyPETPredictor(base_channels=8),
            train_loader,
            None,
            _config(),
            device="cpu",
        )


def test_mean_pretraining_cli_runs_fake_data_and_writes_all_artifacts(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "mean_cli"
    command = [
        sys.executable,
        "scripts/pretrain_conditional_mean.py",
        "--config",
        "configs/experiments/slmf_png_residual_frequency.yaml",
        "--output-dir",
        str(output_dir),
        "--epochs",
        "1",
        "--seed",
        "42",
        "--override",
        "data.use_fake_data=true",
        "--override",
        "data.cache_dir=",
        "--override",
        "data.require_cache_lineage=false",
        "--override",
        "data.image_size=32",
        "--override",
        "data.batch_size=8",
        "--override",
        "data.val_batch_size=8",
        "--override",
        "runtime.num_workers=0",
        "--override",
        "runtime.amp=false",
        "--override",
        "modules.conditional_mean.base_channels=8",
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (output_dir / "mean_best.pt").is_file()
    assert (output_dir / "mean_last.pt").is_file()
    assert (output_dir / "history.json").is_file()
    assert (output_dir / "resolved_config.yaml").is_file()
    checkpoint = torch.load(output_dir / "mean_best.pt", weights_only=True)
    assert torch.isfinite(torch.tensor(checkpoint["val_loss"]))


def test_pathology_excluded_mean_loss_ignores_error_inside_support():
    from src.model.mean_pretraining import (
        pathology_excluded_mean_loss,
        mean_charbonnier_loss,
    )

    # LL2-resolution pred/target: zero error everywhere except 4 in-support pixels.
    target = torch.zeros(1, 1, 8, 8)
    pred = torch.zeros(1, 1, 8, 8)
    pred[0, 0, 0:2, 0:2] = 5.0
    # Pixel-space mask whose /4 pooling covers exactly those 4 LL2 pixels.
    mask = torch.zeros(1, 1, 32, 32)
    mask[0, 0, 0:8, 0:8] = 1.0
    eps = 1e-3

    excluded = pathology_excluded_mean_loss(
        pred, target, mask, epsilon=eps, guard_radius_px=0
    )
    # Background LL2 pixels have zero error → Charbonnier == eps; in-support ignored.
    assert excluded.item() == pytest.approx(eps, abs=1e-6)
    included = mean_charbonnier_loss(pred, target, eps)
    assert included.item() > excluded.item()


def test_pathology_excluded_mean_loss_rejects_bad_mask_shape():
    from src.model.mean_pretraining import pathology_excluded_mean_loss

    pred = torch.zeros(1, 1, 8, 8)
    bad_mask = torch.zeros(1, 2, 32, 32)  # 2 channels, not [B,1,H,W]
    with pytest.raises(ValueError, match="mask must have shape"):
        pathology_excluded_mean_loss(
            pred, pred, bad_mask, epsilon=1e-3, guard_radius_px=0
        )


def test_mean_pretrainer_exclusion_reads_mask_and_records_policy(tmp_path):
    from src.model.mean_pretraining import MeanPretrainer

    config = _config()
    config["modules"]["conditional_mean"]["pathology_exclusion"] = {
        "enabled": True,
        "guard_radius_px": 8,
    }
    torch.manual_seed(0)
    predictor = LowFrequencyPETPredictor(base_channels=8)
    train_loader, val_loader = _loaders()
    trainer = MeanPretrainer(predictor, train_loader, val_loader, config, device="cpu")
    history = trainer.run(epochs=1, output_dir=tmp_path)

    assert len(history) == 1
    assert torch.isfinite(torch.tensor(history[0]["val_loss"]))
    best = torch.load(tmp_path / "mean_best.pt", map_location="cpu", weights_only=True)
    # No new top-level checkpoint key: policy is captured inside mean_config.
    assert set(best) == {"format_version", "model", "mean_config", "epoch", "val_loss"}
    assert best["mean_config"]["pathology_exclusion"]["enabled"] is True
    assert best["mean_config"]["pathology_exclusion"]["guard_radius_px"] == 8


def test_mean_pretrainer_writes_fingerprint_sidecar_with_sha256_and_policy(tmp_path):
    from src.model.mean_pretraining import MeanPretrainer

    config = _config()
    config["modules"]["conditional_mean"]["pathology_exclusion"] = {
        "enabled": True,
        "guard_radius_px": 8,
    }
    torch.manual_seed(0)
    trainer = MeanPretrainer(
        LowFrequencyPETPredictor(base_channels=8),
        *_loaders(),
        config,
        device="cpu",
    )
    trainer.run(epochs=1, output_dir=tmp_path)

    sidecar_path = tmp_path / "mean_best.pt.fingerprint.json"
    assert sidecar_path.is_file()
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))

    # Policy is surfaced for audit without loading weights.
    assert payload["pathology_exclusion"] == {"enabled": True, "guard_radius_px": 8}
    # Checkpoint SHA-256 matches an independent hash of the .pt bytes.
    expected_sha = hashlib.sha256(
        (tmp_path / "mean_best.pt").read_bytes()
    ).hexdigest()
    assert payload["checkpoint_sha256"] == expected_sha
    assert payload["checkpoint_file"] == "mean_best.pt"
    # Fake-data run has no sealed cache lineage -> fingers must be absent but keyed.
    assert payload["lineage_present"] is False
    assert set(payload["lineage_fingerprints"]) == {
        "manifest_semantic_sha256",
        "raw_png_combined_sha256",
        "preprocessing_config_sha256",
        "dataset_contract_sha256",
        "cache_payload_sha256",
        "cache_metadata_sha256",
    }
    assert all(v is None for v in payload["lineage_fingerprints"].values())
    # Self-hash is tamper-evident.
    body = dict(payload)
    body.pop("fingerprint_sha256")
    from src.mechanism_validation.common import canonical_json_sha256

    assert payload["fingerprint_sha256"] == canonical_json_sha256(body)


def test_write_checkpoint_fingerprint_is_fail_closed_for_missing_file(tmp_path):
    from src.model.mean_pretraining import write_checkpoint_fingerprint

    with pytest.raises(FileNotFoundError):
        write_checkpoint_fingerprint(
            tmp_path / "absent.pt", {"mean_config": {}}, None
        )
