"""Full-image mean anchor tests (DESIGN S11; PRD FR-5, P1 lock).

Incremental over tests/test_residual_frequency.py (comparison-prefix
checkpoint loading) and tests/test_mean_pretraining.py (freeze-script
contract): this file locks (a) the frozen full-image comparison mean has no
trainable parameters and stays frozen in train(), (b) the canonical
mean_weights_sha256 recipe (stability, discrimination, fp16/bf16 cast
semantics, key-order independence), and (c) the rc_brd fail-closed lock:
building a contracted model whose mean SHA does not match the loaded
checkpoint raises ValueError (DESIGN S3 v1.0b + S9.4).
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import uuid

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.rc_brd import RecoverabilityContract, mean_weights_sha256
from src.model.slmf_bbdm import SLMFBBDM

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
T = 100
IMAGE = 32


@pytest.fixture()
def artifact_dir() -> pathlib.Path:
    """Repo-anchored scratch dir (DSH sandbox cannot scandir tmp_path)."""
    run_dir = REPO_ROOT / ".t_dir" / "mean_anchor_tests" / uuid.uuid4().hex[:12]
    run_dir.mkdir(parents=True, exist_ok=True)
    yield run_dir
    shutil.rmtree(run_dir, ignore_errors=True)


def _comparison_checkpoint(path, base_channels: int = 8):
    """Pix2Pix-style checkpoint with generator.-prefixed mean weights."""
    from src.model.mean_predictor import FullImagePETPredictor
    torch.manual_seed(3)
    source = FullImagePETPredictor(base_channels=base_channels)
    state = source.state_dict()
    torch.save({"model": {**{f"generator.{k}": v for k, v in state.items()},
                          "discriminator.placeholder": torch.tensor(0.0)}}, path)
    return source, state


def _mean_anchor_cfg(checkpoint: str) -> dict:
    """Residual-bridge config with a frozen comparison-format mean."""
    return {
        "experiment": {"name": "mean_anchor", "seed": 42},
        "data": {"image_size": IMAGE},
        "runtime": {"eval_sampling_steps": 4},
        "model": {"objective": "pred_x0", "enable_heteroscedastic": False,
                  "self_conditioning": {"enabled": False},
                  "base_loss": {"min_snr_enabled": False}},
        "modules": {
            "gabor": {"enabled": False},
            "organ_prior": {"enabled": False},
            "hotspot_prior": {"enabled": False},
            "semantic_prior": {"enabled": False},
            "zero_adapter": {"enabled": False},
            "condition_dropout": {"enabled": False},
            "bbdm_bridge": {"enabled": True, "name": "bbdm_bridge",
                            "num_train_timesteps": T},
            "conditional_mean": {"enabled": True,
                                 "architecture": "full_image_unet",
                                 "base_channels": 8,
                                 "checkpoint": checkpoint,
                                 "checkpoint_format": "comparison",
                                 "freeze": True, "detach_bridge": True,
                                 "loss_weight": 0.0},
            "residual_bridge": {"enabled": True},
            "residual_frequency": {"enabled": False},
        },
        "losses": {},
    }


def test_frozen_full_image_mean_has_no_trainable_params(artifact_dir):
    ckpt = artifact_dir / "unet_comparison.pt"
    source, state = _comparison_checkpoint(ckpt)
    model = SLMFBBDM.from_config(_mean_anchor_cfg(str(ckpt)))

    assert model.mean_frozen is True
    params = list(model.mean_predictor.parameters())
    assert params and all(not p.requires_grad for p in params)
    # Loaded weights equal the comparison generator bit-for-bit.
    for key, value in state.items():
        assert torch.equal(model.mean_predictor.state_dict()[key], value)
    # train() must keep the frozen mean in eval mode (deterministic norms).
    model.train()
    assert model.mean_predictor.training is False


def test_mean_weights_sha256_is_stable_and_discriminating():
    from src.model.mean_predictor import FullImagePETPredictor
    torch.manual_seed(4)
    predictor = FullImagePETPredictor(base_channels=8)
    state = predictor.state_dict()

    # Same weights -> same SHA (deterministic, single canonical stream).
    assert mean_weights_sha256(state) == mean_weights_sha256(state)
    assert mean_weights_sha256(state) == mean_weights_sha256(dict(state))

    # Key-order independence: sorted-key iteration (DESIGN v1.0b recipe).
    reordered = dict(reversed(list(state.items())))
    assert mean_weights_sha256(reordered) == mean_weights_sha256(state)

    # Different weights -> different SHA (perturb one tensor).
    perturbed = {k: v.clone() for k, v in state.items()}
    key = next(iter(perturbed))
    perturbed[key] += 1.0
    assert mean_weights_sha256(perturbed) != mean_weights_sha256(state)

    # Key-name changes alter the digest (key bytes enter the stream).
    renamed = {f"x_{k}": v for k, v in state.items()}
    assert mean_weights_sha256(renamed) != mean_weights_sha256(state)


@pytest.mark.parametrize("half_dtype", [torch.float16, torch.bfloat16])
def test_mean_weights_sha256_cast_semantics(half_dtype):
    """fp16/bf16 checkpoints hash identically to their fp32 view.

    The v1.0b recipe casts floating tensors to float32 *before* hashing, so
    a half-precision mean checkpoint and its .float() image share one SHA
    (checkpoint dtype must not shift the contract anchor).
    """
    from src.model.mean_predictor import FullImagePETPredictor
    torch.manual_seed(5)
    state = FullImagePETPredictor(base_channels=8).state_dict()
    half = {k: v.to(half_dtype) for k, v in state.items()}
    half_view = {k: v.float() for k, v in half.items()}
    assert mean_weights_sha256(half) == mean_weights_sha256(half_view)
    # ... but precision loss vs the original fp32 weights is still visible.
    assert mean_weights_sha256(half) != mean_weights_sha256(state)


def _write_contract(path, mean_sha: str) -> None:
    groups = {"low": ["LL2"], "mid": ["LH2", "HL2", "HH2"],
              "high": ["LH1", "HL1", "HH1"]}
    payload = {
        "fold_id": "fold_0",
        "band_groups": groups,
        "log_snr_grid": [-10.0, 0.0, 10.0],
        "c_values": {"low": [0.2, 0.3, 0.4], "mid": [0.4, 0.5, 0.6],
                     "high": [0.6, 0.7, 0.8]},
        "mean_checkpoint_sha256": mean_sha,
        "b_active": ["low", "mid", "high"],
        "psd_floors": {"low": 1e-4, "mid": 2e-4, "high": 5e-4},
        "size_thresholds": {"small_lesion_q25": 12.0},
        "kappa_grid": [0.0, 0.25],
        "s_ref": 1.0,
        "support_mode": "floor_gated",
        "eta_max": 0.8,
        "floor_rho": 0.1,
    }
    contract = RecoverabilityContract.from_payload(payload)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({**contract.to_payload(),
                   "contract_sha256": contract.contract_sha256}, handle)


def test_contract_mean_sha_mismatch_fails_closed(artifact_dir):
    ckpt = artifact_dir / "unet_comparison.pt"
    _, state = _comparison_checkpoint(ckpt)
    contract = artifact_dir / "contract.json"

    # Wrong SHA -> ValueError at construction (fail-closed lock).
    _write_contract(contract, mean_sha="b" * 64)
    cfg = _mean_anchor_cfg(str(ckpt))
    cfg["modules"]["rc_brd"] = {"enabled": True, "num_timesteps": T,
                                "contract_path": str(contract)}
    with pytest.raises(ValueError, match="SHA"):
        SLMFBBDM.from_config(cfg)

    # Correct SHA (of the stripped, loaded comparison weights) -> builds.
    _write_contract(contract, mean_sha=mean_weights_sha256(state))
    cfg["modules"]["rc_brd"]["contract_path"] = str(contract)
    model = SLMFBBDM.from_config(cfg)
    assert model.rc_brd_contract is not None
    assert model.rc_brd_contract.mean_checkpoint_sha256 == mean_weights_sha256(state)
