"""Tests for the background-tail repair loss and sampled monitoring metric."""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.interfaces import ConditionBundle, LossContext
from src.model.loss_terms.nonlesion_body_lowpass import (
    NonLesionBodyLowpassLoss,
)
from src.model.trainer import _compute_body_background_sample_metrics


def _context(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    ct: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    tau: float = 0.1,
) -> LossContext:
    if ct is None:
        ct = torch.ones_like(target)
    if mask is None:
        mask = torch.zeros_like(target)
    return LossContext(
        model_pred=pred,
        loss_target=target,
        target_pet=target,
        pred_x0=pred,
        timesteps=torch.zeros(pred.shape[0], dtype=torch.long),
        tau=torch.full((pred.shape[0],), tau),
        batch={"ct": ct, "mask": mask},
        condition=ConditionBundle.empty(),
    )


def test_nonlesion_lowpass_loss_is_zero_for_exact_prediction():
    target = torch.linspace(-1.0, 1.0, 32).view(1, 1, 1, 32)
    target = target.expand(1, 1, 32, 32).contiguous()
    term = NonLesionBodyLowpassLoss(
        sigma=2.0,
        charbonnier_eps=0.0,
        weight=0.2,
    )

    loss, logs = term(_context(target.clone(), target))

    assert loss.item() == pytest.approx(0.0, abs=1e-8)
    assert logs["nonlesion_body_lowpass/lowpass_mae"].item() == pytest.approx(
        0.0,
        abs=1e-8,
    )
    assert logs["nonlesion_body_lowpass/valid_samples"].item() == 1.0


def test_nonlesion_lowpass_loss_excludes_dilated_lesion():
    target = torch.full((1, 1, 32, 32), -0.5)
    pred = target.clone()
    pred[:, :, 12:20, 12:20] = 1.0
    mask = torch.zeros_like(target)
    mask[:, :, 12:20, 12:20] = 1.0
    term = NonLesionBodyLowpassLoss(
        sigma=0.5,
        lesion_exclusion_radius=6,
        charbonnier_eps=0.0,
        weight=1.0,
    )

    loss, _ = term(_context(pred, target, mask=mask))

    assert loss.item() < 1.0e-5


def test_upper_uptake_tail_receives_more_weight_and_gradient():
    target = torch.full((1, 1, 32, 32), -0.8)
    target[:, :, 8:24, 8:24] = 0.6
    pred = target.clone()
    pred[:, :, 8:24, 8:24] = -0.2
    pred.requires_grad_(True)
    base = NonLesionBodyLowpassLoss(
        sigma=1.0,
        tail_weight=0.0,
        charbonnier_eps=0.0,
        weight=1.0,
    )
    weighted = NonLesionBodyLowpassLoss(
        sigma=1.0,
        tail_weight=3.0,
        tail_quantile=0.75,
        tail_temperature=0.02,
        charbonnier_eps=0.0,
        weight=1.0,
    )

    base_loss, _ = base(_context(pred, target))
    weighted_loss, logs = weighted(_context(pred, target))
    weighted_loss.backward()

    assert weighted_loss.item() > base_loss.item()
    assert logs["nonlesion_body_lowpass/tail_lowpass_mae"].item() > 0.0
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[:, :, 8:24, 8:24].abs().sum().item() > 0.0


def test_sampled_background_metric_matches_identical_pet():
    target = torch.full((32, 32), -0.7).numpy()
    target[8:24, 8:24] = 0.3
    ct = torch.ones((32, 32)).numpy()
    mask = torch.zeros((32, 32)).numpy()

    metrics = _compute_body_background_sample_metrics(
        target,
        target,
        ct,
        mask,
        lowpass_sigma=2.0,
        body_threshold=0.03,
        lesion_exclusion_radius=4,
    )

    assert metrics is not None
    assert metrics["body_nonlesion_lowpass_mae"] == pytest.approx(
        0.0,
        abs=1e-8,
    )
    assert metrics["body_nonlesion_mean_bias"] == pytest.approx(
        0.0,
        abs=1e-8,
    )
    assert metrics["body_nonlesion_q90_bias"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["body_nonlesion_q95_bias"] == pytest.approx(0.0, abs=1e-8)


def test_background_repair_config_is_optimizer_fresh_and_transfer_bounded():
    from src.model.config_utils import (
        load_full_config,
        validate_png_baseline_config,
    )

    config = load_full_config(
        "configs/experiments/slmf_png_background_tail_repair_e100.yaml"
    )
    validate_png_baseline_config(config)

    assert config["training"]["num_epochs"] == 50
    assert config["training"]["init_weights"] == "ema"
    assert config["training"]["resume_from"] is None
    assert config["runtime"]["eval_sampling_steps"] == 50
    assert config["runtime"]["save_interval"] == 999
    assert config["runtime"]["best_checkpoint"]["lightweight"] is True
    assert config["runtime"]["best_checkpoint"]["save_tags"] == [
        "best_background",
    ]
    assert config["runtime"]["best_checkpoint"]["background_constraints"] == {
        "max_failure_rate": pytest.approx(0.265625),
        "max_lesion_peak_error_norm": pytest.approx(0.1365),
        "max_lesion_topq_peak_error_norm": pytest.approx(0.1720),
    }
    frequency = config["modules"]["residual_frequency"]
    assert frequency["freeze"] is True
    assert (
        frequency["cross_level_router"]["destination_mode"] == "native_only"
    )
    repair = config["losses"]["nonlesion_body_lowpass"]
    assert repair["enabled"] is True
    assert repair["weight"] == pytest.approx(0.2)
    assert repair["tail_quantile"] == pytest.approx(0.75)
    assert repair["tail_weight"] == pytest.approx(2.0)
