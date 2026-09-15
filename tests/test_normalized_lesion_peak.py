from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _context(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    tau: torch.Tensor | None = None,
    model_pred: torch.Tensor | None = None,
):
    from src.model.interfaces import ConditionBundle, LossContext

    batch_size = pred.shape[0]
    return LossContext(
        model_pred=torch.zeros_like(pred) if model_pred is None else model_pred,
        loss_target=target,
        target_pet=target,
        pred_x0=pred,
        timesteps=torch.arange(batch_size),
        tau=torch.full((batch_size,), 0.1) if tau is None else tau,
        batch={"mask": mask},
        condition=ConditionBundle.empty(),
    )


def test_peak_loss_selects_prediction_and_target_topq_independently():
    from src.model.loss_terms.normalized_lesion_peak import (
        NormalizedLesionPeakLoss,
    )

    pred = torch.tensor([[[[0.9, 0.8, 0.1, 0.0]]]])
    target = torch.tensor([[[[0.7, 0.6, 1.0, 0.0]]]])
    mask = torch.ones_like(pred)
    term = NormalizedLesionPeakLoss(
        topk_percent=0.5,
        min_k=1,
        max_k=4,
        beta=0.01,
        active_tau_max=1.0,
        weight=1.0,
    )

    _, logs = term(_context(pred, target, mask))

    assert logs["normalized_lesion_peak/pred_peak"].item() == pytest.approx(0.85)
    assert logs["normalized_lesion_peak/target_peak"].item() == pytest.approx(0.85)
    assert logs["normalized_lesion_peak/signed_bias"].item() == pytest.approx(0.0)


def test_peak_loss_clamps_k_between_minimum_and_maximum():
    from src.model.loss_terms.normalized_lesion_peak import (
        NormalizedLesionPeakLoss,
    )

    pred = torch.arange(20, dtype=torch.float32).reshape(1, 1, 4, 5)
    target = torch.zeros_like(pred)
    mask = torch.ones_like(pred)
    term = NormalizedLesionPeakLoss(
        topk_percent=0.9,
        min_k=3,
        max_k=4,
        active_tau_max=1.0,
        weight=1.0,
    )

    _, logs = term(_context(pred, target, mask))

    assert logs["normalized_lesion_peak/k_mean"].item() == 4
    assert logs["normalized_lesion_peak/pred_peak"].item() == pytest.approx(17.5)


def test_empty_masks_are_excluded_without_diluting_valid_sample():
    from src.model.loss_terms.normalized_lesion_peak import (
        NormalizedLesionPeakLoss,
    )

    pred = torch.zeros(2, 1, 2, 2)
    target = torch.zeros_like(pred)
    pred[0, 0, 0, 0] = 1.0
    mask = torch.zeros_like(pred)
    mask[0] = 1.0
    term = NormalizedLesionPeakLoss(
        topk_percent=0.25,
        min_k=1,
        max_k=4,
        beta=0.1,
        active_tau_max=1.0,
        weight=1.0,
    )

    loss, logs = term(_context(pred, target, mask))

    assert logs["normalized_lesion_peak/valid_count"].item() == 1
    assert loss.item() == pytest.approx(0.95, abs=1e-4)


def test_tau_gate_normalizes_by_gate_mass_not_batch_size():
    from src.model.loss_terms.normalized_lesion_peak import (
        NormalizedLesionPeakLoss,
    )

    pred = torch.ones(2, 1, 1, 1)
    target = torch.zeros_like(pred)
    mask = torch.ones_like(pred)
    tau = torch.tensor([0.0, 1.0])
    term = NormalizedLesionPeakLoss(
        topk_percent=1.0,
        min_k=1,
        max_k=1,
        beta=0.1,
        active_tau_max=0.25,
        weight=1.0,
    )

    loss, logs = term(_context(pred, target, mask, tau=tau))

    # Audit L2: the single gated-in sample keeps its full nominal loss
    # (0.95 * sigmoid(7.5) ~= 0.9495) instead of being diluted by the
    # gated-out sample (old behaviour: batch-mean 0.475).
    assert loss.item() == pytest.approx(0.9495, abs=1e-3)
    assert logs["normalized_lesion_peak/gate_mean"].item() == pytest.approx(0.5, abs=5e-4)


def test_fully_gated_out_batch_keeps_temporal_gate_semantics():
    from src.model.loss_terms.normalized_lesion_peak import (
        NormalizedLesionPeakLoss,
    )

    pred = torch.ones(2, 1, 1, 1)
    target = torch.zeros_like(pred)
    mask = torch.ones_like(pred)
    tau = torch.tensor([1.0, 1.0])
    term = NormalizedLesionPeakLoss(
        topk_percent=1.0,
        min_k=1,
        max_k=1,
        beta=0.1,
        active_tau_max=0.25,
        weight=1.0,
    )

    loss, _ = term(_context(pred, target, mask, tau=tau))

    # Dividing by max(sum(gate), 1) must not resurrect full supervision when
    # every tau is above the gate threshold: the batch contributes ~0.
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_peak_loss_uses_pred_x0_and_backpropagates():
    from src.model.loss_terms.normalized_lesion_peak import (
        NormalizedLesionPeakLoss,
    )

    pred = torch.tensor([[[[0.8, 0.2]]]], requires_grad=True)
    target = torch.zeros_like(pred)
    mask = torch.ones_like(pred)
    term = NormalizedLesionPeakLoss(
        topk_percent=0.5,
        min_k=1,
        max_k=1,
        active_tau_max=1.0,
        weight=1.0,
    )

    loss, _ = term(
        _context(pred, target, mask, model_pred=torch.full_like(pred, 99.0))
    )
    loss.backward()

    assert pred.grad is not None
    assert pred.grad[0, 0, 0, 0].abs() > 0
    assert pred.grad[0, 0, 0, 1] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"topk_percent": 0.0},
        {"topk_percent": 1.1},
        {"min_k": 0},
        {"min_k": 4, "max_k": 3},
        {"beta": 0.0},
    ],
)
def test_peak_loss_rejects_invalid_configuration(kwargs):
    from src.model.loss_terms.normalized_lesion_peak import (
        NormalizedLesionPeakLoss,
    )

    with pytest.raises(ValueError):
        NormalizedLesionPeakLoss(**kwargs)
