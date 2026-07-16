"""Boundary-frequency supervision and evaluation contracts."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _square(shift: int = 0, size: int = 32) -> torch.Tensor:
    image = torch.zeros(1, 1, size, size)
    image[:, :, 10 + shift:22 + shift, 10:22] = 1.0
    return image


def _context(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    ct: torch.Tensor | None = None,
    organ_mask: torch.Tensor | None = None,
    model_pred: torch.Tensor | None = None,
    reliability: float = 1.0,
    gate_tv: torch.Tensor | None = None,
):
    from src.model.interfaces import ConditionBundle, LossContext

    batch = {
        "ct": target.clone() if ct is None else ct,
        "mask": torch.zeros_like(target) if mask is None else mask,
    }
    if organ_mask is not None:
        batch["organ_mask"] = organ_mask
    condition = ConditionBundle.empty()
    condition.scalars["frequency_noise_reliability"] = target.new_tensor(
        [reliability]
    )
    if gate_tv is not None:
        condition.scalars["frequency_gate_tv"] = gate_tv
    return LossContext(
        model_pred=torch.randn_like(target) if model_pred is None else model_pred,
        loss_target=target,
        target_pet=target,
        pred_x0=pred,
        timesteps=torch.tensor([10]),
        tau=torch.tensor([0.1]),
        batch=batch,
        condition=condition,
    )


def test_shifted_lesion_boundary_costs_more_than_aligned_prediction():
    from src.model.loss_terms.boundary_frequency import BoundaryFrequencyLoss

    target = _square()
    mask = _square()
    term = BoundaryFrequencyLoss(boundary_radius=1, weight=1.0)
    aligned, aligned_logs = term(_context(target.clone(), target, mask=mask))
    shifted, shifted_logs = term(_context(_square(shift=2), target, mask=mask))

    assert shifted > aligned
    assert shifted_logs["boundary_frequency/lesion_available"].item() == 1
    assert shifted_logs["boundary_frequency/lesion"] > aligned_logs[
        "boundary_frequency/lesion"
    ]


def test_boundary_loss_uses_predicted_x0_not_noisy_model_tensor():
    from src.model.loss_terms.boundary_frequency import BoundaryFrequencyLoss

    target = _square()
    pred = _square(shift=1)
    term = BoundaryFrequencyLoss(boundary_radius=1, weight=1.0)
    first, _ = term(
        _context(pred, target, mask=target, model_pred=torch.zeros_like(target))
    )
    second, _ = term(
        _context(pred, target, mask=target, model_pred=20 * torch.randn_like(target))
    )

    assert torch.allclose(first, second)


def test_empty_organ_is_unavailable_but_nonempty_organ_activates_boundary():
    from src.model.loss_terms.boundary_frequency import BoundaryFrequencyLoss

    target = _square()
    pred = _square(shift=1)
    empty = torch.zeros(1, 6, 32, 32)
    nonempty = empty.clone()
    nonempty[:, 2, 6:26, 6:26] = 1.0
    term = BoundaryFrequencyLoss(boundary_radius=1, weight=1.0)

    _, empty_logs = term(_context(pred, target, organ_mask=empty))
    _, present_logs = term(_context(pred, target, organ_mask=nonempty))

    assert empty_logs["boundary_frequency/organ_available"].item() == 0
    assert empty_logs["boundary_frequency/organ"].item() == 0
    assert present_logs["boundary_frequency/organ_available"].item() == 1


def test_ct_only_edge_is_not_treated_as_pet_anatomy_consensus():
    from src.model.loss_terms.boundary_frequency import BoundaryFrequencyLoss

    blank_pet = torch.zeros(1, 1, 32, 32)
    ct_edge = _square()
    pred_edge = _square(shift=2)
    term = BoundaryFrequencyLoss(boundary_radius=1, weight=1.0)
    _, ct_only = term(_context(pred_edge, blank_pet, ct=ct_edge))
    _, consensus = term(_context(pred_edge, ct_edge, ct=ct_edge))

    assert ct_only["boundary_frequency/anatomy"].item() < 1e-6
    assert consensus["boundary_frequency/anatomy"] > ct_only[
        "boundary_frequency/anatomy"
    ]


def test_bridge_reliability_suppresses_boundary_supervision_at_high_noise():
    from src.model.loss_terms.boundary_frequency import BoundaryFrequencyLoss

    target = _square()
    pred = _square(shift=2)
    term = BoundaryFrequencyLoss(boundary_radius=1, weight=1.0)
    low_reliability, low_logs = term(
        _context(pred, target, mask=target, reliability=0.01)
    )
    high_reliability, high_logs = term(
        _context(pred, target, mask=target, reliability=1.0)
    )

    assert low_reliability < high_reliability
    assert low_logs["boundary_frequency/gate_mean"] < high_logs[
        "boundary_frequency/gate_mean"
    ]


def test_frequency_gate_tv_loss_consumes_differentiable_router_scalar():
    from src.model.loss_terms.frequency_gate_tv import FrequencyGateTVLoss

    gate_tv = torch.tensor(0.4, requires_grad=True)
    ctx = _context(
        _square(),
        _square(),
        gate_tv=gate_tv,
    )
    loss, logs = FrequencyGateTVLoss(weight=0.1)(ctx)

    assert torch.allclose(loss, torch.tensor(0.04))
    assert logs["frequency_gate_tv/available"].item() == 1
    loss.backward()
    assert gate_tv.grad is not None and gate_tv.grad.item() > 0


def _square_np(shift: int = 0, size: int = 32) -> np.ndarray:
    image = np.zeros((1, size, size), dtype=np.float32)
    image[:, 10 + shift:22 + shift, 10:22] = 1.0
    return image


def test_boundary_metrics_reward_aligned_lesion_and_anatomy_edges():
    from scripts.evaluate import compute_boundary_metrics

    target = _square_np()
    mask = _square_np()
    ct = _square_np()
    empty_organs = np.zeros((6, 32, 32), dtype=np.float32)
    aligned = compute_boundary_metrics(target, target, ct, mask, empty_organs)
    shifted = compute_boundary_metrics(
        _square_np(shift=2), target, ct, mask, empty_organs
    )

    assert aligned["lesion_boundary_gradient_mae_norm"] < shifted[
        "lesion_boundary_gradient_mae_norm"
    ]
    assert aligned["anatomy_edge_gradient_mae_norm"] < shifted[
        "anatomy_edge_gradient_mae_norm"
    ]
    assert np.isnan(aligned["organ_boundary_gradient_mae_norm"])


def test_directional_spectrum_compares_prediction_to_target_not_isotropy():
    from scripts.evaluate import compute_directional_spectrum_error

    target = np.zeros((1, 32, 32), dtype=np.float32)
    target[:, :, 14:18] = 1.0
    artifact = np.zeros_like(target)
    artifact[:, 14:18, :] = 1.0

    assert compute_directional_spectrum_error(target, target) == 0.0
    assert compute_directional_spectrum_error(artifact, target) > 0.01
