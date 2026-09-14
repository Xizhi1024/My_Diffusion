from __future__ import annotations

from pathlib import Path

import torch

from scripts.validate_h2_pathology_excluded_residual import (
    _decision_from_metrics,
    pathology_excluded_mean_loss,
    pathology_included_mean_loss,
)
from src.mechanism_validation.common import (
    partition_counts,
    patient_partition,
    read_manifest,
)


def test_formal_partition_reproduces_h1_patient_counts() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = read_manifest(root / "main_data" / "split_manifest.csv")
    counts = partition_counts(rows, patient_partition(rows))
    assert {role: value["patients"] for role, value in counts.items()} == {
        "mechanism_train": 99,
        "calibration": 25,
        "validation": 31,
    }


def test_excluded_loss_ignores_dilated_pathology_support() -> None:
    target = torch.zeros(1, 1, 8, 8)
    prediction = target.clone()
    prediction[:, :, 3:5, 3:5] = 10.0
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 12:20, 12:20] = 1.0

    included = pathology_included_mean_loss(
        prediction,
        target,
        epsilon=1e-3,
    )
    excluded = pathology_excluded_mean_loss(
        prediction,
        target,
        mask,
        epsilon=1e-3,
        guard_radius_px=0,
    )
    assert included > 0.5
    assert excluded < 0.01


def test_zero_mask_makes_included_and_excluded_losses_equal() -> None:
    generator = torch.Generator().manual_seed(7)
    target = torch.randn(2, 1, 8, 8, generator=generator)
    prediction = torch.randn(2, 1, 8, 8, generator=generator)
    mask = torch.zeros(2, 1, 32, 32)
    included = pathology_included_mean_loss(
        prediction,
        target,
        epsilon=1e-3,
    )
    excluded = pathology_excluded_mean_loss(
        prediction,
        target,
        mask,
        epsilon=1e-3,
        guard_radius_px=8,
    )
    torch.testing.assert_close(included, excluded)


def test_h2_gate_uses_patient_paired_validation_and_calibration_margin() -> None:
    rows = []
    for index in range(25):
        rows.append(
            {
                "partition": "calibration",
                "patient_id": f"c{index}",
                "excluded_residual_enrichment": 0.2,
                "included_residual_enrichment": 0.1,
                "excluded_boundary_gradient_error": 0.1005,
                "included_boundary_gradient_error": 0.1,
            }
        )
    for index in range(31):
        rows.append(
            {
                "partition": "validation",
                "patient_id": f"v{index}",
                "excluded_residual_enrichment": 0.25,
                "included_residual_enrichment": 0.1,
                "excluded_boundary_gradient_error": 0.1001,
                "included_boundary_gradient_error": 0.1,
            }
        )
    evidence, thresholds = _decision_from_metrics(rows)
    assert evidence["decision"] == "PASS"
    assert thresholds["boundary_noninferiority_margin"] >= 0.0005
