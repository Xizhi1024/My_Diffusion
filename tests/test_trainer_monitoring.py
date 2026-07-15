"""Regression tests for deterministic, signed-range-safe trainer monitoring."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.trainer import Trainer, _compute_pet_sample_metrics, _stratified_indices


def test_signed_pet_metrics_do_not_select_zeroed_mask_background():
    pred = np.full((5, 5), -1.0, dtype=np.float32)
    target = np.full((5, 5), -1.0, dtype=np.float32)
    mask = np.zeros((5, 5), dtype=np.float32)
    mask[2, 2] = 1.0
    pred[2, 2] = -0.2
    target[2, 2] = 0.6
    pred[0, 0] = -0.6

    metrics = _compute_pet_sample_metrics(pred, target, mask)

    assert metrics is not None
    assert metrics["lesion_centroid_distance"] == pytest.approx(0.0)
    assert metrics["lesion_peak_error_norm"] == pytest.approx(0.4)
    assert metrics["outside_inside_peak_ratio"] == pytest.approx(0.5)
    assert metrics["failure"] == 0.0
    assert metrics["lesion_roi_l1"] == pytest.approx(0.4)


def test_signed_pet_metrics_skip_empty_mask():
    pred = np.zeros((4, 4), dtype=np.float32)
    target = np.zeros((4, 4), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=np.float32)

    assert _compute_pet_sample_metrics(pred, target, mask) is None


def test_stratified_indices_are_deterministic_unique_and_cover_endpoints():
    indices = _stratified_indices(total=30, count=16)

    assert indices == _stratified_indices(total=30, count=16)
    assert len(indices) == 16
    assert len(set(indices)) == 16
    assert indices[0] == 0
    assert indices[-1] == 29


def test_eval_sampling_is_repeatable_without_advancing_outer_rng():
    class RandomSampleModel:
        def sample(self, batch):
            return {"synthetic_pet": torch.randn_like(batch["ct"])}

    trainer = object.__new__(Trainer)
    trainer.device = "cpu"
    trainer.eval_seed = 123
    trainer.model = RandomSampleModel()
    batch = {"ct": torch.zeros(2, 1, 4, 4)}

    torch.manual_seed(999)
    before = torch.random.get_rng_state().clone()
    first = trainer._sample_with_eval_seed(batch)["synthetic_pet"]
    after = torch.random.get_rng_state().clone()
    second = trainer._sample_with_eval_seed(batch)["synthetic_pet"]

    assert torch.equal(before, after)
    assert torch.equal(first, second)


def test_checkpoint_selection_returns_one_shared_combined_improvement():
    trainer = object.__new__(Trainer)
    trainer.best_ckpts_enabled = False
    trainer.best_combined_alpha = 0.5
    trainer.best_stripe_penalty = 0.3
    trainer._best_combined_score = -1e9
    trainer._best_lesion_score = -1e9
    trainer._best_image_score = -1e9
    metrics = {
        "val/mae": 0.1,
        "val/ssim": 0.8,
        "val/stripe_score": 1.0,
        "val/lesion_peak_error_norm": 0.2,
        "val/lesion_centroid_distance": 2.0,
        "val/failure_rate": 0.0,
    }

    assert trainer._save_best_checkpoints(metrics) is True
    assert trainer._save_best_checkpoints(metrics) is False


def test_early_stopping_patience_is_measured_in_epochs():
    trainer = object.__new__(Trainer)
    trainer.early_stopping_enabled = True
    trainer.early_stopping_patience = 40
    trainer.early_stopping_min_epochs = 50
    trainer._last_combined_improvement_epoch = None
    trainer._epochs_since_improve = 0

    trainer.epoch_count = 50
    assert trainer._check_early_stopping(improved=True) is False
    trainer.epoch_count = 80
    assert trainer._check_early_stopping(improved=False) is False
    assert trainer._epochs_since_improve == 30
    trainer.epoch_count = 90
    assert trainer._check_early_stopping(improved=False) is True
    assert trainer._epochs_since_improve == 40


def test_early_stopping_improvement_resets_epoch_origin():
    trainer = object.__new__(Trainer)
    trainer.early_stopping_enabled = True
    trainer.early_stopping_patience = 40
    trainer.early_stopping_min_epochs = 0
    trainer._last_combined_improvement_epoch = 50
    trainer._epochs_since_improve = 30

    trainer.epoch_count = 80
    assert trainer._check_early_stopping(improved=True) is False
    assert trainer._last_combined_improvement_epoch == 80
    assert trainer._epochs_since_improve == 0
