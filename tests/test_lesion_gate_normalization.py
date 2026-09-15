"""Audit L2: gated lesion losses normalize by gate mass, not batch size.

lesion_roi_l1 / topk_lesion / normalized_lesion_peak previously divided the
gate-weighted per-sample losses by the batch size (or appended-sample count),
so the effective weight of the lesion supervision tracked the gate's batch
share. They now divide by max(sum(gate), 1): active samples keep full nominal
weight while a fully gated-out batch still contributes ~0.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# sigmoid(30 * (0.25 - tau)) at tau = 0.0, sharpness 30 (default).
GATE_IN = torch.sigmoid(torch.tensor(30.0 * 0.25)).item()   # ~= 0.9994483


def _context(pred, target, mask, tau):
    from src.model.interfaces import ConditionBundle, LossContext

    return LossContext(
        model_pred=torch.zeros_like(pred),
        loss_target=target,
        target_pet=target,
        pred_x0=pred,
        timesteps=torch.arange(pred.shape[0]),
        tau=tau,
        batch={"mask": mask},
        condition=ConditionBundle.empty(),
    )


def _single_pixel_lesion_batch():
    pred = torch.zeros(2, 1, 4, 4)
    target = torch.zeros_like(pred)
    mask = torch.zeros_like(pred)
    for i in range(2):
        target[i, 0, 0, 0] = 1.0
        mask[i, 0, 0, 0] = 1.0
    return pred, target, mask


def test_lesion_roi_l1_gate_mass_normalization():
    from src.model.loss_terms.lesion_roi import LesionROIL1Loss

    pred, target, mask = _single_pixel_lesion_batch()
    term = LesionROIL1Loss(
        dilate_radius=0,
        beta=0.0,
        active_tau_max=0.25,
        weight=1.0,
    )
    tau = torch.tensor([0.0, 1.0])

    loss, _ = term(_context(pred, target, mask, tau))

    # Old behaviour: (1.0*GATE_IN + 1.0*GATE_OUT) / 2 ~= 0.4997.
    # New: divided by max(sum(gate), 1) => full weight for the active sample.
    assert loss.item() == pytest_approx(GATE_IN, abs=1e-3)


def test_lesion_roi_l1_excludes_empty_mask_samples_from_gate_mass():
    from src.model.loss_terms.lesion_roi import LesionROIL1Loss

    pred, target, mask = _single_pixel_lesion_batch()
    mask[1] = 0.0  # second sample: negative slice, no lesion
    term = LesionROIL1Loss(
        dilate_radius=0,
        beta=0.0,
        active_tau_max=0.25,
        weight=1.0,
    )
    tau = torch.tensor([0.0, 0.0])  # both gates fully open

    loss, _ = term(_context(pred, target, mask, tau))

    # The empty-mask sample contributes no gate mass: the active sample's
    # loss keeps its full gate value (per_sample 1.0 * gate ~= 0.9994).
    # Old behaviour: batch-mean ~= 0.4997.
    assert loss.item() == pytest_approx(GATE_IN, abs=1e-3)


def test_lesion_roi_l1_fully_gated_out_is_zero():
    from src.model.loss_terms.lesion_roi import LesionROIL1Loss

    pred, target, mask = _single_pixel_lesion_batch()
    term = LesionROIL1Loss(
        dilate_radius=0,
        beta=0.0,
        active_tau_max=0.25,
        weight=1.0,
    )
    tau = torch.tensor([1.0, 1.0])

    loss, _ = term(_context(pred, target, mask, tau))

    assert loss.item() < 1e-6


def test_topk_lesion_gate_mass_normalization():
    from src.model.loss_terms.topk import TopKLesionLoss

    pred, target, mask = _single_pixel_lesion_batch()
    term = TopKLesionLoss(
        topk_percent=0.01,
        focal_gamma=2.0,
        active_tau_max=0.25,
        weight=1.0,
    )
    tau = torch.tensor([0.0, 1.0])

    loss, _ = term(_context(pred, target, mask, tau))

    # Per sample: error=1.0, focal=(1-exp(-1))^2 ~= 0.399576.
    focal = (1.0 - pow(2.718281828459045, -1.0)) ** 2
    expected = focal * GATE_IN  # / max(sum(gate), 1) = / 1
    assert loss.item() == pytest_approx(expected, abs=1e-3)


def pytest_approx(value, abs=1e-6):
    import pytest

    return pytest.approx(value, abs=abs)
