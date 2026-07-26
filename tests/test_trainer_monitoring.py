"""Regression tests for deterministic, signed-range-safe trainer monitoring."""

import json
import os
import sys

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.trainer import Trainer, _compute_pet_sample_metrics, _stratified_indices


def test_spectral_optimizer_groups_assign_learning_rates_and_no_decay():
    from torch import nn

    from src.model.trainer import _build_optimizer

    class _Preconditioner(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection_heads = nn.ModuleList([nn.Linear(2, 2)])
            self.l2_to_l1_projection = nn.Linear(2, 2)
            self.route_heads = nn.ModuleList([nn.Linear(2, 3)])
            self.no_null_route_heads = nn.ModuleList([nn.Linear(2, 2)])
            self.amplitude_heads = nn.ModuleList([nn.Linear(2, 1)])
            self.dct_descriptor = nn.Linear(2, 2)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(2, 2)
            self.residual_preconditioner = _Preconditioner()
            self.priors = nn.ModuleDict({"gabor": nn.Linear(2, 2)})

    model = _Model()
    optimizer = _build_optimizer(model, {
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "optimizer_groups": {
            "enabled": True,
            "projection_lr": 5e-4,
            "router_lr": 4e-4,
            "descriptor_lr": 2e-4,
            "no_decay_bias_and_offsets": True,
        },
    })

    by_parameter = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            assert id(parameter) not in by_parameter
            by_parameter[id(parameter)] = group
    assert len(by_parameter) == len(list(model.parameters()))
    assert by_parameter[id(model.backbone.weight)]["lr"] == pytest.approx(1e-4)
    assert by_parameter[id(model.residual_preconditioner.projection_heads[0].weight)]["lr"] == pytest.approx(5e-4)
    assert by_parameter[id(model.residual_preconditioner.route_heads[0].weight)]["lr"] == pytest.approx(4e-4)
    assert by_parameter[id(model.residual_preconditioner.dct_descriptor.weight)]["lr"] == pytest.approx(2e-4)
    assert by_parameter[id(model.priors["gabor"].weight)]["lr"] == pytest.approx(2e-4)
    assert by_parameter[id(model.backbone.bias)]["weight_decay"] == 0.0
    assert by_parameter[id(model.backbone.weight)]["weight_decay"] == pytest.approx(0.01)


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


def test_trainer_pet_metrics_include_robust_topq_peak():
    pred = np.zeros((4, 4), dtype=np.float32)
    target = np.zeros((4, 4), dtype=np.float32)
    mask = np.ones((4, 4), dtype=np.float32)
    pred.flat[-3:] = [0.6, 0.8, 1.0]
    target.flat[-3:] = [0.4, 0.6, 0.8]

    metrics = _compute_pet_sample_metrics(pred, target, mask)

    assert metrics is not None
    assert metrics["lesion_topq_peak_error_norm"] == pytest.approx(0.1)


def test_checkpoint_selection_penalizes_worse_topq_peak_error():
    trainer = object.__new__(Trainer)
    trainer.best_combined_alpha = 0.5
    trainer.best_stripe_penalty = 0.3
    common = {
        "val/mae": 0.05,
        "val/ssim": 0.95,
        "val/stripe_score": 1.0,
        "val/lesion_peak_error_norm": 0.05,
        "val/lesion_centroid_distance": 2.0,
        "val/failure_rate": 0.0,
    }

    good = trainer._model_selection_scores(
        {**common, "val/lesion_topq_peak_error_norm": 0.02}
    )
    bad = trainer._model_selection_scores(
        {**common, "val/lesion_topq_peak_error_norm": 0.20}
    )

    assert good[0] > bad[0]
    assert good[2] > bad[2]


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


def test_chunked_eval_sampling_never_exceeds_chunk_size():
    """A 64-sample tracked batch must reach model.sample in bounded chunks."""
    observed_batch_sizes: list[int] = []

    class RecordingModel:
        def sample(self, batch):
            observed_batch_sizes.append(int(batch["ct"].shape[0]))
            return {"synthetic_pet": torch.randn_like(batch["ct"])}

    trainer = object.__new__(Trainer)
    trainer.device = "cpu"
    trainer.eval_seed = 2026
    trainer.eval_sample_batch_size = 4
    trainer.model = RecordingModel()
    batch = {"ct": torch.zeros(10, 1, 4, 4)}

    result = trainer._sample_with_eval_seed(batch)

    # 10 samples / chunk 4 => chunks of (4, 4, 2); every call is bounded.
    assert observed_batch_sizes == [4, 4, 2]
    assert max(observed_batch_sizes) <= trainer.eval_sample_batch_size
    # The merged output preserves the full batch ordering.
    assert result["synthetic_pet"].shape[0] == 10


def test_chunked_eval_sampling_repeats_under_fixed_seed():
    """Chunked sampling inside one eval scope is deterministic across calls."""
    class RandomSampleModel:
        def sample(self, batch):
            return {"synthetic_pet": torch.randn_like(batch["ct"])}

    trainer = object.__new__(Trainer)
    trainer.device = "cpu"
    trainer.eval_seed = 7
    trainer.eval_sample_batch_size = 3
    trainer.model = RandomSampleModel()
    batch = {"ct": torch.zeros(9, 1, 4, 4)}

    first = trainer._sample_with_eval_seed(batch)["synthetic_pet"]
    second = trainer._sample_with_eval_seed(batch)["synthetic_pet"]

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
    replacements = []
    monkeypatch.setattr(os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        os,
        "replace",
        lambda source, destination: replacements.append((source, destination)),
    )
    monkeypatch.setattr(
        torch,
        "save",
        lambda payload, path: snapshots.append((path, dict(payload["monitoring"]))),
    )

    trainer._save_best_checkpoints(metrics)

    assert len(snapshots) == 3
    assert len(replacements) == 3
    assert all(source.endswith(".pt.tmp") for source, _ in replacements)
    assert all(not destination.endswith(".tmp") for _, destination in replacements)
    assert all(snapshot == snapshots[0][1] for _, snapshot in snapshots)
    assert snapshots[0][1] == trainer._monitoring_state_dict()


def test_sample_grid_failure_prevents_epoch_ledger_and_checkpoint_commit(
    tmp_path,
):
    """Optional sample artifacts must finish before final epoch commits."""

    class FakeModel:
        priors = {}
        loss_terms = {}

        def get_trainable_params(self):
            return 0

        def get_total_params(self):
            return 0

    class FakeEMA:
        def apply(self):
            return None

        def restore(self):
            return None

    trainer = object.__new__(Trainer)
    trainer.config = {"training": {"num_epochs": 1}}
    trainer.device = "cpu"
    trainer.amp_dtype = torch.float32
    trainer.torch_compile = False
    trainer.grad_accum = 1
    trainer.log_interval = 1
    trainer.eval_interval = 999
    trainer.sample_interval = 1
    trainer.save_interval = 1
    trainer.val_loader = None
    trainer.model = FakeModel()
    trainer.ema = FakeEMA()
    trainer.epoch_count = 0
    trainer.step_count = 0
    trainer._tracked_batch = {
        "ct": torch.zeros(1, 1, 4, 4),
        "pet": torch.zeros(1, 1, 4, 4),
    }
    trainer._tracked_meta = None
    trainer.metrics_jsonl = os.fspath(tmp_path / "metrics.jsonl")
    sample_calls = []
    checkpoint_calls = []

    def fake_train_epoch():
        trainer.epoch_count += 1
        return {
            "loss/total": 0.0,
            "perf/epoch_seconds": 0.0,
            "perf/lr": 0.0,
        }

    def fake_sample(batch):
        sample_calls.append(batch)
        return {"synthetic_pet": torch.zeros(1, 1, 4, 4)}

    def fail_grid(batch, synth):
        raise OSError("sample artifact disk failure")

    trainer.train_epoch = fake_train_epoch
    trainer._sample_with_eval_seed = fake_sample
    trainer._save_sample_grid = fail_grid
    trainer.save_checkpoint = lambda: checkpoint_calls.append("checkpoint")

    with pytest.raises(OSError, match="sample artifact disk failure"):
        trainer.run(num_epochs=1)

    assert len(sample_calls) == 1
    assert checkpoint_calls == []
    assert not (tmp_path / "metrics.jsonl").exists()


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


def test_evaluator_annotates_bottom_quantile_small_lesions():
    from scripts.evaluate import _annotate_small_lesion_metrics

    rows = [
        {
            "lesion_size": area,
            "lesion_topq_peak_error_norm": error,
            "lesion_topq_peak_signed_bias_norm": bias,
            "lesion_peak_underestimated": underestimated,
        }
        for area, error, bias, underestimated in (
            (4.0, 0.30, -0.25, 1.0),
            (8.0, 0.20, -0.10, 1.0),
            (16.0, 0.10, 0.05, 0.0),
            (32.0, 0.05, -0.01, 0.0),
        )
    ]

    info = _annotate_small_lesion_metrics(rows, quantile=0.25)

    assert info == {
        "small_lesion_quantile": 0.25,
        "small_lesion_underestimate_tolerance": 0.05,
        "small_lesion_area_threshold": 4.0,
        "small_lesion_sample_count": 1,
    }
    assert rows[0]["small_lesion_topq_peak_error_norm"] == pytest.approx(0.30)
    assert rows[0]["small_lesion_cold_bias_norm"] == pytest.approx(0.25)
    assert rows[0]["small_lesion_underestimate"] == pytest.approx(1.0)
    assert rows[1]["small_lesion"] == 0.0
    assert "small_lesion_topq_peak_error_norm" not in rows[1]


def test_small_lesion_underestimate_ignores_sub_tolerance_cold_bias():
    from scripts.evaluate import _annotate_small_lesion_metrics

    rows = [
        {
            "lesion_size": 4.0,
            "lesion_topq_peak_error_norm": 0.01,
            "lesion_topq_peak_signed_bias_norm": -0.01,
        }
    ]

    _annotate_small_lesion_metrics(
        rows, quantile=1.0, underestimate_tolerance=0.05
    )

    assert rows[0]["small_lesion_underestimate"] == 0.0


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


def test_evaluator_reports_prediction_target_and_excess_stripe_scores():
    from scripts.evaluate import compute_stripe_metrics

    target = np.zeros((1, 32, 32), dtype=np.float32)
    target[:, 8:24, 8:24] = 1.0
    pred = target.copy()
    pred[:, :, ::2] += 0.5
    metrics = compute_stripe_metrics(pred, target)

    assert set(metrics) == {"stripe_score", "target_stripe_score", "stripe_excess"}
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["stripe_excess"] == pytest.approx(
        metrics["stripe_score"] - metrics["target_stripe_score"]
    )


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


def test_resume_truncates_stale_metrics_beyond_checkpoint_epoch(tmp_path):
    """A crash between checkpoints must not yield duplicate epoch records."""
    metrics_path = tmp_path / "training_metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            json.dumps({"epoch": epoch, "train": {"loss/total": 1.0 / epoch}})
            for epoch in range(1, 13)
        ),
        encoding="utf-8",
    )

    trainer = object.__new__(Trainer)
    trainer.metrics_jsonl = os.fspath(metrics_path)
    trainer.epoch_count = 10
    trainer._truncate_metrics_jsonl_to_epoch()

    kept = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["epoch"] for row in kept] == list(range(1, 11))


def test_resume_leaves_metrics_intact_when_nothing_is_stale(tmp_path):
    metrics_path = tmp_path / "training_metrics.jsonl"
    payload = [
        {"epoch": epoch, "train": {"loss/total": 1.0 / epoch}}
        for epoch in range(1, 11)
    ]
    metrics_path.write_text(
        "\n".join(json.dumps(record) for record in payload),
        encoding="utf-8",
    )

    trainer = object.__new__(Trainer)
    trainer.metrics_jsonl = os.fspath(metrics_path)
    trainer.epoch_count = 10
    trainer._truncate_metrics_jsonl_to_epoch()

    kept = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["epoch"] for row in kept] == list(range(1, 11))
    # A no-op truncation must not leave a .tmp file behind.
    assert not (tmp_path / "training_metrics.jsonl.tmp").exists()


def test_resume_truncation_preserves_unparseable_lines(tmp_path):
    """Corrupt lines fail closed instead of being hidden by resume cleanup."""
    metrics_path = tmp_path / "training_metrics.jsonl"
    metrics_path.write_text(
        '{"epoch": 1}\nnot-json\n',
        encoding="utf-8",
    )
    trainer = object.__new__(Trainer)
    trainer.metrics_jsonl = os.fspath(metrics_path)
    trainer.epoch_count = 1

    with pytest.raises(RuntimeError, match="invalid JSON"):
        trainer._truncate_metrics_jsonl_to_epoch()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"epoch": 1}\n{"epoch": 1}\n', "duplicate epoch"),
        ('{"epoch": 0}\n', "epoch must be >= 1"),
        ('{"epoch": true}\n', "epoch must be an integer"),
        ('{"epoch": 1}\n\n', "blank line"),
    ],
)
def test_resume_metrics_contract_rejects_invalid_records(
    tmp_path,
    content,
    message,
):
    metrics_path = tmp_path / "training_metrics.jsonl"
    metrics_path.write_text(content, encoding="utf-8")
    trainer = object.__new__(Trainer)
    trainer.metrics_jsonl = os.fspath(metrics_path)
    trainer.epoch_count = 1

    with pytest.raises(RuntimeError, match=message):
        trainer._truncate_metrics_jsonl_to_epoch()


def test_resume_rejects_checkpoint_epoch_missing_from_metrics(tmp_path):
    metrics_path = tmp_path / "training_metrics.jsonl"
    metrics_path.write_text(
        "\n".join(json.dumps({"epoch": epoch}) for epoch in range(1, 10)),
        encoding="utf-8",
    )
    trainer = object.__new__(Trainer)
    trainer.metrics_jsonl = os.fspath(metrics_path)
    trainer.epoch_count = 10

    with pytest.raises(RuntimeError, match=r"does not exactly cover.*1\.\.10"):
        trainer._truncate_metrics_jsonl_to_epoch()


def test_resume_rejects_epoch_outside_configured_range(tmp_path):
    metrics_path = tmp_path / "training_metrics.jsonl"
    metrics_path.write_text(
        "\n".join(json.dumps({"epoch": epoch}) for epoch in range(1, 12)),
        encoding="utf-8",
    )
    trainer = object.__new__(Trainer)
    trainer.config = {"training": {"num_epochs": 10}}
    trainer.metrics_jsonl = os.fspath(metrics_path)
    trainer.epoch_count = 10

    with pytest.raises(RuntimeError, match="exceeds configured training range"):
        trainer._truncate_metrics_jsonl_to_epoch()


def test_checkpoint_rng_roundtrip_is_weights_only_safe_and_exact(tmp_path):
    """A real Trainer checkpoint restores all host and sampling RNG streams."""
    import random

    from src.data.dataset import build_dataloaders

    config = {
        "experiment": {"name": "rng-roundtrip", "seed": 37},
        "data": {
            "use_fake_data": True,
            "image_size": 8,
            "batch_size": 4,
            "val_batch_size": 4,
        },
        "runtime": {
            "amp": False,
            "channels_last": False,
            "num_workers": 0,
            "persistent_workers": False,
            "dataloader_seed": 37,
            "metrics_jsonl": os.fspath(tmp_path / "metrics.jsonl"),
        },
        "training": {
            "num_epochs": 3,
            "checkpoint_dir": os.fspath(tmp_path / "checkpoints"),
            "ema": {"decay": 0.9, "update_every": 1},
        },
    }
    train_loader, val_loader = build_dataloaders(
        config["data"],
        config["runtime"],
    )
    trainer = Trainer(
        torch.nn.Linear(2, 2),
        config,
        train_loader,
        val_loader,
        device="cpu",
    )
    trainer.epoch_count = 3
    trainer.step_count = 17
    # Mirror a real uninterrupted run: the fixed validation cohort is selected
    # before the first numbered checkpoint and is not itself serialized.
    trainer._initialize_tracked_batch()
    assert trainer._tracked_batch is not None
    expected_tracked_ct = trainer._tracked_batch["ct"].clone()
    (tmp_path / "metrics.jsonl").write_text(
        "\n".join(json.dumps({"epoch": epoch}) for epoch in range(1, 4)),
        encoding="utf-8",
    )

    random.seed(123)
    np.random.seed(456)
    torch.manual_seed(789)
    trainer.save_checkpoint()
    checkpoint_path = tmp_path / "checkpoints" / "ckpt_epoch0003.pt"
    assert not (tmp_path / "checkpoints" / "ckpt_epoch0003.pt.tmp").exists()

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    assert payload["rng"]["schema_version"] == 2
    assert torch.is_tensor(payload["rng"]["numpy"]["state"])
    assert payload["rng"]["numpy"]["state"].device.type == "cpu"
    saved_val_generator_state = payload["rng"]["loader_generators"]["val_loader"]
    expected_host_draws = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )
    expected_sampler_epoch = list(iter(trainer.train_loader.sampler))

    resumed_train, resumed_val = build_dataloaders(
        config["data"],
        config["runtime"],
    )
    resumed = Trainer(
        torch.nn.Linear(2, 2),
        config,
        resumed_train,
        resumed_val,
        device="cpu",
    )
    resumed.load_checkpoint(os.fspath(checkpoint_path))

    assert resumed._tracked_batch is not None
    assert torch.equal(resumed._tracked_batch["ct"], expected_tracked_ct)
    assert torch.equal(
        resumed.val_loader.generator.get_state(),
        saved_val_generator_state,
    )
    actual_host_draws = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )
    actual_sampler_epoch = list(iter(resumed.train_loader.sampler))
    assert actual_host_draws[0] == expected_host_draws[0]
    assert actual_host_draws[1] == expected_host_draws[1]
    assert torch.equal(actual_host_draws[2], expected_host_draws[2])
    assert actual_sampler_epoch == expected_sampler_epoch
    assert resumed.epoch_count == 3
    assert resumed.step_count == 17


def _prior_anchored_resume_config():
    checkpoint = "results/prior-run/checkpoints/ckpt_epoch0010.pt"
    return {
        "prior_anchored_run": {
            "pipeline_id": "PRIOR_ANCHORED_ROUTER_100E_CLOUD_V1",
        },
        "experiment": {"name": "prior-run", "seed": 42},
        "training": {
            "num_epochs": 100,
            "checkpoint_dir": "results/prior-run/checkpoints",
            "resume_from": checkpoint,
        },
        "model": {
            "base_loss": {
                "mse_weight": 1.0,
                "l1_weight": 1.0,
            },
        },
    }


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("experiment", "seed"), 43),
        (("model", "base_loss", "l1_weight"), 0.25),
    ],
)
def test_prior_anchored_resume_rejects_full_config_mismatch(
    monkeypatch,
    field_path,
    replacement,
):
    """Seed and loss changes cannot bypass the run-owned runner contract."""
    import copy

    current = _prior_anchored_resume_config()
    observed = copy.deepcopy(current)
    observed["training"]["resume_from"] = None
    target = observed
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = replacement

    trainer = object.__new__(Trainer)
    trainer.config = current
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {"config": observed},
    )

    with pytest.raises(RuntimeError, match="configuration differs"):
        trainer.load_checkpoint(current["training"]["resume_from"])


@pytest.mark.parametrize(
    ("source", "resume_from", "message"),
    [
        (
            "current",
            "D:/outside/ckpt_epoch0010.pt",
            "repository-relative",
        ),
        (
            "current",
            "results/other-run/checkpoints/ckpt_epoch0010.pt",
            "inside the current",
        ),
        (
            "checkpoint",
            "D:/outside/ckpt_epoch0005.pt",
            "repository-relative",
        ),
        (
            "checkpoint",
            "results/other-run/checkpoints/ckpt_epoch0005.pt",
            "inside the current",
        ),
        (
            "current",
            "results/prior-run/checkpoints/ckpt_best_combined.pt",
            "ckpt_epochNNNN",
        ),
    ],
)
def test_prior_anchored_resume_rejects_absolute_or_cross_run_pointer(
    monkeypatch,
    source,
    resume_from,
    message,
):
    """Both current and historical resume pointers are run-directory sealed."""
    import copy

    current = _prior_anchored_resume_config()
    observed = copy.deepcopy(current)
    observed["training"]["resume_from"] = None
    if source == "current":
        current["training"]["resume_from"] = resume_from
        load_path = resume_from
    else:
        observed["training"]["resume_from"] = resume_from
        load_path = current["training"]["resume_from"]

    trainer = object.__new__(Trainer)
    trainer.config = current
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {"config": observed},
    )

    with pytest.raises(RuntimeError, match=message):
        trainer.load_checkpoint(load_path)


def test_epoch_stamped_data_is_exact_across_persistent_worker_resume():
    """Persistent workers cannot perturb stateless epoch/sample augmentation."""
    from src.data.dataset import build_dataloaders

    data_cfg = {
        "use_fake_data": True,
        "image_size": 8,
        "batch_size": 4,
        "val_batch_size": 4,
    }
    runtime_cfg = {
        "num_workers": 1,
        "persistent_workers": True,
        "dataloader_seed": 91,
    }
    continuous_loader, continuous_val = build_dataloaders(
        data_cfg,
        runtime_cfg,
    )
    continuous = object.__new__(Trainer)
    continuous.train_loader = continuous_loader
    continuous.val_loader = continuous_val
    continuous.epoch_count = 0
    continuous._set_data_epoch()
    list(continuous_loader)  # consume epoch 0
    saved_rng = continuous._rng_state_dict()
    continuous.epoch_count = 1
    continuous._set_data_epoch()
    expected = torch.cat([batch["ct"] for batch in continuous_loader], dim=0)

    resumed_loader, resumed_val = build_dataloaders(data_cfg, runtime_cfg)
    resumed = object.__new__(Trainer)
    resumed.train_loader = resumed_loader
    resumed.val_loader = resumed_val
    resumed.epoch_count = 1
    resumed._load_rng_state({"rng": saved_rng})
    resumed._set_data_epoch()
    actual = torch.cat([batch["ct"] for batch in resumed_loader], dim=0)

    assert torch.equal(actual, expected)


def test_validation_metrics_keep_full_batch_on_cpu(monkeypatch):
    """Supplying sampled CPU output must not move the tracked batch to CUDA."""
    import src.model.trainer as trainer_module

    def _forbid_full_batch_transfer(*args, **kwargs):
        raise AssertionError("full tracked batch was transferred")

    monkeypatch.setattr(trainer_module, "_to_device", _forbid_full_batch_transfer)

    class EvalOnlyModel:
        def eval(self):
            return self

    trainer = object.__new__(Trainer)
    trainer.model = EvalOnlyModel()
    batch = {
        "pet": torch.zeros(4, 1, 8, 8),
        "mask": torch.ones(4, 1, 8, 8),
    }
    synth = torch.zeros_like(batch["pet"])

    metrics = trainer._compute_val_sample_metrics(batch, synth)

    assert metrics["val/mae"] == pytest.approx(0.0)
