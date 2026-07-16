from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _model_space(unit: np.ndarray) -> np.ndarray:
    return 2.0 * unit.astype(np.float32) - 1.0


def test_normalized_peak_metrics_report_signed_and_topq_errors():
    from scripts.evaluate import compute_normalized_lesion_metrics

    pred_unit = np.zeros((1, 5, 5), dtype=np.float32)
    target_unit = np.zeros_like(pred_unit)
    mask = np.zeros_like(pred_unit)
    organ = np.zeros((6, 5, 5), dtype=np.float32)
    mask[0, 1:4, 1:4] = 1.0
    pred_unit[0, 1:4, 1:4] = np.array(
        [[0.2, 0.3, 0.4], [0.5, 0.9, 0.8], [0.1, 0.7, 0.6]],
        dtype=np.float32,
    )
    target_unit[0, 1:4, 1:4] = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.8, 0.7], [0.0, 0.6, 0.5]],
        dtype=np.float32,
    )

    metrics = compute_normalized_lesion_metrics(
        _model_space(pred_unit),
        _model_space(target_unit),
        mask,
        organ,
        topk_percent=0.10,
        min_k=3,
        max_k=16,
    )

    assert metrics["lesion_peak_signed_bias_norm"] == pytest.approx(0.1)
    assert metrics["lesion_peak_error_norm"] == pytest.approx(0.1)
    assert metrics["lesion_topq_peak_pred_norm"] == pytest.approx(0.8)
    assert metrics["lesion_topq_peak_target_norm"] == pytest.approx(0.7)
    assert metrics["lesion_topq_peak_signed_bias_norm"] == pytest.approx(0.1)
    assert metrics["lesion_topq_peak_error_norm"] == pytest.approx(0.1)
    assert metrics["lesion_peak_overestimated"] == 1.0
    assert metrics["lesion_peak_underestimated"] == 0.0
    assert metrics["lesion_topq_k"] == 3.0
    assert metrics["lesion_size"] == 9.0


def test_core_ring_metrics_locate_center_peak_and_preserve_target_relation():
    from scripts.evaluate import compute_normalized_lesion_metrics

    pred_unit = np.zeros((1, 7, 7), dtype=np.float32)
    target_unit = np.zeros_like(pred_unit)
    mask = np.zeros_like(pred_unit)
    organ = np.zeros((6, 7, 7), dtype=np.float32)
    mask[0, 2:5, 2:5] = 1.0
    pred_unit[0, 3, 3] = 0.9
    target_unit[0, 3, 3] = 0.8
    pred_unit[0, 2:5, 2:5] += 0.2
    target_unit[0, 2:5, 2:5] += 0.2

    metrics = compute_normalized_lesion_metrics(
        _model_space(pred_unit),
        _model_space(target_unit),
        mask,
        organ,
        topk_percent=0.10,
        min_k=1,
        max_k=16,
    )

    assert metrics["lesion_core_fallback"] == 0.0
    assert metrics["lesion_peak_to_boundary_distance"] == pytest.approx(1.0)
    assert metrics["lesion_core_topq_pred_norm"] > metrics["lesion_ring_topq_pred_norm"]
    assert metrics["lesion_ring_core_ratio_pred"] < 1.0


def test_single_pixel_lesion_uses_core_fallback():
    from scripts.evaluate import compute_normalized_lesion_metrics

    pred = np.full((1, 5, 5), -1.0, dtype=np.float32)
    target = pred.copy()
    mask = np.zeros_like(pred)
    organ = np.zeros((6, 5, 5), dtype=np.float32)
    mask[0, 2, 2] = 1.0
    pred[0, 2, 2] = 0.6
    target[0, 2, 2] = 0.4

    metrics = compute_normalized_lesion_metrics(pred, target, mask, organ)

    assert metrics["lesion_core_fallback"] == 1.0
    assert metrics["lesion_core_topq_pred_norm"] == pytest.approx(0.8)
    assert metrics["lesion_peak_to_boundary_distance"] == pytest.approx(0.0)


def test_empty_lesion_returns_nan_for_new_peak_diagnostics():
    from scripts.evaluate import compute_normalized_lesion_metrics

    image = np.zeros((1, 5, 5), dtype=np.float32)
    mask = np.zeros_like(image)
    organ = np.zeros((6, 5, 5), dtype=np.float32)

    metrics = compute_normalized_lesion_metrics(image, image, mask, organ)

    assert np.isnan(metrics["lesion_topq_peak_error_norm"])
    assert np.isnan(metrics["lesion_core_topq_pred_norm"])
    assert np.isnan(metrics["lesion_peak_to_boundary_distance"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"topk_percent": 0.0},
        {"topk_percent": 1.1},
        {"min_k": 0},
        {"min_k": 4, "max_k": 3},
    ],
)
def test_peak_diagnostics_reject_invalid_topq_configuration(kwargs):
    from scripts.evaluate import compute_normalized_lesion_metrics

    image = np.zeros((1, 5, 5), dtype=np.float32)
    mask = np.ones_like(image)
    organ = np.zeros((6, 5, 5), dtype=np.float32)

    with pytest.raises(ValueError):
        compute_normalized_lesion_metrics(image, image, mask, organ, **kwargs)
