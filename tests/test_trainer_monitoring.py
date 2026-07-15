"""Regression tests for deterministic, signed-range-safe trainer monitoring."""

import os
import sys

import numpy as np
import pytest
import torch
import yaml

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


def test_eval_rng_scope_is_repeatable_without_advancing_training_rng():
    trainer = object.__new__(Trainer)
    trainer.device = "cpu"
    trainer.eval_seed = 321

    torch.manual_seed(999)
    before = torch.random.get_rng_state().clone()
    with trainer._eval_rng_scope():
        first = torch.rand(8)
    after = torch.random.get_rng_state().clone()
    with trainer._eval_rng_scope():
        second = torch.rand(8)

    assert torch.equal(before, after)
    assert torch.equal(first, second)


def test_initial_tracked_batch_scan_does_not_advance_training_rng():
    class TwoSampleDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            mask = torch.zeros(1, 4, 4)
            mask.flatten()[: index + 1] = 1.0
            return {
                "ct": torch.zeros(1, 4, 4),
                "pet": torch.zeros(1, 4, 4),
                "mask": mask,
            }

    trainer = object.__new__(Trainer)
    trainer.device = "cpu"
    trainer.eval_seed = 42
    trainer.eval_num_samples = 2
    trainer.tracked_sample_ids = []
    trainer.val_loader = torch.utils.data.DataLoader(
        TwoSampleDataset(), batch_size=1, shuffle=False
    )
    trainer._tracked_batch = None
    trainer._tracked_meta = None

    torch.manual_seed(999)
    before = torch.random.get_rng_state().clone()
    trainer._initialize_tracked_batch()
    after = torch.random.get_rng_state().clone()

    assert torch.equal(before, after)
    assert trainer._tracked_batch is not None
    assert len({row["sample_id"] for row in trainer._tracked_meta}) == 2


def test_checkpoint_selection_returns_one_shared_combined_improvement():
    trainer = object.__new__(Trainer)
    trainer.best_ckpts_enabled = False
    trainer.best_combined_alpha = 0.5
    trainer.best_stripe_penalty = 0.3
    trainer._best_combined_score = -1e9
    trainer._best_lesion_score = -1e9
    trainer._best_image_score = -1e9
    trainer._last_combined_improvement_epoch = None
    trainer._epochs_since_improve = 0
    trainer.epoch_count = 50
    metrics = {
        "val/mae": 0.1,
        "val/ssim": 0.8,
        "val/stripe_score": 1.0,
        "val/lesion_peak_error_norm": 0.2,
        "val/lesion_centroid_distance": 2.0,
        "val/failure_rate": 0.0,
    }

    assert trainer._save_best_checkpoints(metrics) is True
    assert trainer._last_combined_improvement_epoch == 50
    assert trainer._save_best_checkpoints(metrics) is False


def test_all_best_checkpoints_capture_one_atomic_monitoring_snapshot(monkeypatch):
    class HasState:
        def state_dict(self):
            return {}

    trainer = object.__new__(Trainer)
    trainer.best_ckpts_enabled = True
    trainer.best_combined_alpha = 0.5
    trainer.best_stripe_penalty = 0.3
    trainer._best_combined_score = -1e9
    trainer._best_lesion_score = -1e9
    trainer._best_image_score = -1e9
    trainer._last_combined_improvement_epoch = None
    trainer._epochs_since_improve = 0
    trainer.epoch_count = 50
    trainer.step_count = 10
    trainer.config = {"experiment": {"name": "atomic-monitoring-test"}}
    trainer.model = HasState()
    trainer.optimizer = HasState()
    trainer.scheduler = HasState()
    trainer.ema = HasState()
    metrics = {
        "val/mae": 0.1,
        "val/ssim": 0.8,
        "val/stripe_score": 1.0,
        "val/lesion_peak_error_norm": 0.2,
        "val/lesion_centroid_distance": 2.0,
        "val/failure_rate": 0.0,
    }
    snapshots = []
    monkeypatch.setattr(os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        torch,
        "save",
        lambda payload, path: snapshots.append((path, dict(payload["monitoring"]))),
    )

    trainer._save_best_checkpoints(metrics)

    assert len(snapshots) == 3
    assert all(snapshot == snapshots[0][1] for _, snapshot in snapshots)
    assert snapshots[0][1] == trainer._monitoring_state_dict()


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


def test_png_baseline_declares_deterministic_eval_seed():
    with open("configs/experiments/slmf_png_baseline.yaml", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    assert cfg["runtime"]["eval_num_samples"] == 16
    assert cfg["runtime"]["eval_seed"] == cfg["experiment"]["seed"]


def test_evaluator_normalized_metrics_are_signed_range_safe():
    from scripts.evaluate import compute_normalized_lesion_metrics

    pred = np.full((1, 5, 5), -1.0, dtype=np.float32)
    target = np.full((1, 5, 5), -1.0, dtype=np.float32)
    mask = np.zeros((1, 5, 5), dtype=np.float32)
    organ = np.zeros((6, 5, 5), dtype=np.float32)
    mask[0, 2, 2] = 1.0
    pred[0, 2, 2] = -0.2
    target[0, 2, 2] = 0.6
    pred[0, 0, 0] = -0.6

    metrics = compute_normalized_lesion_metrics(pred, target, mask, organ)

    assert metrics["lesion_peak_error_norm"] == pytest.approx(0.4)
    assert metrics["lesion_mean_error_norm"] == pytest.approx(0.4)
    assert metrics["lesion_centroid_distance"] == pytest.approx(0.0)


def test_evaluator_can_overlay_old_checkpoint_ema_shadow():
    from scripts.evaluate import _select_checkpoint_state

    checkpoint = {
        "model": {
            "weight": torch.tensor([1.0]),
            "buffer": torch.tensor([3.0]),
        },
        "ema": {
            "shadow": {"weight": torch.tensor([2.0])},
            "step_count": 10,
            "decay": 0.999,
        },
    }

    ema_state, ema_source = _select_checkpoint_state(checkpoint, weights="ema")
    raw_state, raw_source = _select_checkpoint_state(checkpoint, weights="raw")

    assert ema_source == "ema.shadow"
    assert ema_state["weight"].item() == 2.0
    assert ema_state["buffer"].item() == 3.0
    assert raw_source == "model"
    assert raw_state["weight"].item() == 1.0


def test_evaluator_builds_fixed_stratified_lesion_subset():
    from scripts.evaluate import _build_stratified_subset

    class MaskAreaDataset:
        def __len__(self):
            return 30

        def __getitem__(self, index):
            mask = torch.zeros(1, 6, 6)
            mask.flatten()[: index + 1] = 1.0
            return {"mask": mask, "index": index}

    subset, indices = _build_stratified_subset(MaskAreaDataset(), count=16)

    assert len(subset) == 16
    assert len(indices) == 16
    assert len(set(indices)) == 16
    assert indices[0] == 0
    assert indices[-1] == 29


def test_evaluator_seed_repeats_torch_and_numpy_draws():
    from scripts.evaluate import _seed_evaluation

    _seed_evaluation(42)
    torch_first = torch.rand(4)
    numpy_first = np.random.rand(4)
    _seed_evaluation(42)
    torch_second = torch.rand(4)
    numpy_second = np.random.rand(4)

    assert torch.equal(torch_first, torch_second)
    assert np.array_equal(numpy_first, numpy_second)


def test_evaluator_failure_metrics_use_explicit_model_space_conversion():
    from scripts.evaluate import compute_failure_detection_metrics

    pred = np.zeros((1, 3, 3), dtype=np.float32)
    target = np.zeros((1, 3, 3), dtype=np.float32)
    mask = np.zeros((1, 3, 3), dtype=np.float32)
    mask[0, 1, 1] = 1.0
    target[0, 1, 1] = 1.0
    pred[0, 0, 0] = 0.2

    metrics = compute_failure_detection_metrics(
        pred,
        target,
        mask,
        model_space=True,
    )

    assert metrics["inside_peak"] == pytest.approx(0.5)
    assert metrics["outside_peak"] == pytest.approx(0.6)


def test_monitoring_state_round_trips_for_resumed_early_stopping():
    trainer = object.__new__(Trainer)
    trainer._best_combined_score = 1.25
    trainer._best_lesion_score = 0.75
    trainer._best_image_score = 1.5
    trainer._epochs_since_improve = 20
    trainer._last_combined_improvement_epoch = 100

    state = trainer._monitoring_state_dict()

    resumed = object.__new__(Trainer)
    resumed._best_combined_score = -1e9
    resumed._best_lesion_score = -1e9
    resumed._best_image_score = -1e9
    resumed._epochs_since_improve = 0
    resumed._last_combined_improvement_epoch = None
    resumed._load_monitoring_state({"monitoring": state})

    assert resumed._best_combined_score == pytest.approx(1.25)
    assert resumed._best_lesion_score == pytest.approx(0.75)
    assert resumed._best_image_score == pytest.approx(1.5)
    assert resumed._epochs_since_improve == 20
    assert resumed._last_combined_improvement_epoch == 100
