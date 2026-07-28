from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.eval_router_background_suite import (
    _load_fixed_destination,
    compute_body_background_metrics,
    compute_global_hotspot_metrics,
)


def test_fixed_destination_is_read_from_requested_epoch_train_block(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training_metrics.jsonl"
    train = {}
    expected = {
        "l2": [0.20, 0.30, 0.40],
        "l1": [0.05, 0.06, 0.07],
    }
    for level, values in expected.items():
        for band, value in zip(("lh", "hl", "hh"), values):
            train[
                f"frequency/prior_anchor_{level}_{band}_conditional_shallow"
            ] = value
    path.write_text(
        json.dumps({"epoch": 99, "train": {}}) + "\n"
        + json.dumps({"epoch": 100, "train": train})
        + "\n",
        encoding="utf-8",
    )

    assert _load_fixed_destination(path, 100) == expected
    with pytest.raises(RuntimeError, match="Epoch 101"):
        _load_fixed_destination(path, 101)


def test_body_background_metrics_separate_low_and_high_frequency_error() -> None:
    size = 32
    yy, xx = np.mgrid[:size, :size]
    body = ((yy - 16) ** 2 + (xx - 16) ** 2) <= 11**2
    ct = np.full((1, size, size), -1.0, dtype=np.float32)
    ct[0, body] = 0.2
    target = np.full_like(ct, -1.0)
    target[0, body] = 0.2 + 0.2 * np.sin(xx[body] / 2.0)
    mean = np.full_like(ct, -1.0)
    mean[0, body] = -0.2
    pred = mean.copy()
    residual = pred - mean
    lesion = np.zeros_like(ct)
    lesion[0, 15:18, 15:18] = 1.0

    metrics = compute_body_background_metrics(
        pred,
        target,
        mean,
        residual,
        ct,
        lesion,
        lowpass_sigma=2.0,
        body_threshold=0.03,
        lesion_exclusion_radius=2,
    )

    assert metrics["body_fraction"] > 0.0
    assert metrics["body_nonlesion_fraction"] > 0.0
    assert metrics["body_nonlesion_mean_bias"] < 0.0
    assert metrics["body_nonlesion_lowpass_mae"] > 0.0
    assert metrics["body_nonlesion_highpass_energy_ratio"] < 1.0


def test_global_hotspot_metric_uses_global_not_mask_conditioned_peak() -> None:
    pred = np.full((1, 16, 16), -1.0, dtype=np.float32)
    lesion = np.zeros_like(pred)
    lesion[0, 7:9, 7:9] = 1.0
    pred[0, 7, 7] = 0.5
    pred[0, 1, 1] = 1.0

    metrics = compute_global_hotspot_metrics(
        pred,
        lesion,
        hit_radius=1,
    )

    assert metrics["global_hotspot_hit"] == 0.0
    assert metrics["global_hotspot_distance_to_lesion"] > 0.0
