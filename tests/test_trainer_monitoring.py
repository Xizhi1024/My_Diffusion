"""Regression tests for deterministic, signed-range-safe trainer monitoring."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.trainer import _compute_pet_sample_metrics


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
