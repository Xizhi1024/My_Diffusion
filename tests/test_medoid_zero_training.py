from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_medoid_zero_training import (
    _haar_dwt2,
    component_mechanism_rows,
    medoid_indices,
    trajectory_dispersion,
    within_patient_slope,
)


def test_medoid_indices_selects_existing_central_candidate_and_breaks_ties() -> None:
    stack = np.asarray(
        [
            [[[0.0, 0.0]]],
            [[[1.0, 1.0]]],
            [[[2.0, 2.0]]],
            [[[10.0, 10.0]]],
        ],
        dtype=np.float32,
    )
    assert medoid_indices(stack).tolist() == [1]

    tied = np.asarray(
        [
            [[[0.0]]],
            [[[0.0]]],
            [[[1.0]]],
            [[[1.0]]],
        ],
        dtype=np.float32,
    )
    assert medoid_indices(tied).tolist() == [0]


def test_trajectory_dispersion_separates_lesion_and_full_image() -> None:
    stack = np.zeros((4, 1, 1, 2, 2), dtype=np.float32)
    stack[:, 0, 0, 0, 0] = np.asarray([0.0, 1.0, 2.0, 3.0])
    mask = np.zeros((1, 1, 2, 2), dtype=np.float32)
    mask[0, 0, 0, 0] = 1.0
    result = trajectory_dispersion(stack, mask).iloc[0]
    assert result["lesion_pixel_std"] > result["full_pixel_std"]
    assert result["lesion_pairwise_l1"] > result["full_pairwise_l1"]


def test_within_patient_slope_recovers_patient_intercept_controlled_effect() -> None:
    frame = pd.DataFrame(
        {
            "patient": ["a", "a", "a", "b", "b", "b"],
            "x": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "y": [10.0, 12.0, 14.0, -5.0, -3.0, -1.0],
        }
    )
    result = within_patient_slope(
        frame,
        x="x",
        y="y",
        patient="patient",
        replicates=200,
        rng=np.random.default_rng(7),
    )
    assert result["estimate"] == pytest.approx(2.0)
    assert result["ci95_low"] == pytest.approx(2.0)
    assert result["ci95_high"] == pytest.approx(2.0)


def test_numpy_haar_preserves_energy() -> None:
    rng = np.random.default_rng(11)
    image = rng.normal(size=(3, 1, 8, 8))
    ll, details = _haar_dwt2(image)
    transformed_energy = np.square(ll).sum()
    transformed_energy += sum(np.square(band).sum() for band in details)
    assert transformed_energy == pytest.approx(np.square(image).sum())


def test_component_mechanism_rows_reports_each_component() -> None:
    masks = np.zeros((1, 1, 8, 8), dtype=np.float32)
    masks[0, 0, 1:3, 1:3] = 1.0
    masks[0, 0, 5:7, 5:7] = 1.0
    gradients = np.ones_like(masks, dtype=np.float64)
    residual = np.zeros_like(masks, dtype=np.float32)
    residual[masks > 0.5] = 1.0
    cohort = pd.DataFrame(
        [{"patient_id": "001", "sample_id": "001001", "slice_id": 1}]
    )
    rows = component_mechanism_rows(
        masks=masks,
        gradient_maps=gradients,
        target_residual=residual,
        cohort=cohort,
    )
    assert len(rows) == 2
    assert {row["component_area"] for row in rows} == {4}
    assert all(row["lesion_gradient_mass"] == pytest.approx(4.0) for row in rows)
    assert all(
        0.0 <= row["target_residual_high_fraction"] <= 1.0 for row in rows
    )
