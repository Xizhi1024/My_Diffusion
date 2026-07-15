"""Contracts for standalone low-frequency PET mean pretraining."""

from __future__ import annotations

import json
import os
import subprocess
import sys
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
