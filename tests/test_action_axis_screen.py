from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.eval_action_axis_screen import (
    VARIANTS,
    _configure_variant,
    _medoid,
    _metric_rows_equal,
    _parse_seeds,
    _parse_variants,
    build_patient_balanced_subset,
)


class _Dataset:
    def __init__(self) -> None:
        self.entries = []
        self.samples = []
        areas = (0, 1, 4, 16)
        for patient in range(4):
            for slice_index, area in enumerate(areas):
                mask = torch.zeros(1, 8, 8)
                if area:
                    width = int(area**0.5)
                    mask[:, :width, :width] = 1.0
                sample_id = f"{patient:03d}_{slice_index:02d}"
                self.entries.append(
                    SimpleNamespace(
                        patient_id=f"{patient:03d}",
                        sample_id=sample_id,
                        slice_id=str(slice_index),
                    )
                )
                self.samples.append({"mask": mask})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


class _Router:
    def __init__(self) -> None:
        self.destination = "learned"
        self.actions = None
        self.frequency = "full"
        self.detail_gain = 1.0
        self.ll_gain = 1.0
        self.minimum = None
        self.maximum = None

    def set_inference_destination_intervention(self, mode):
        self.destination = mode

    def set_inference_route_action_intervention(
        self,
        *,
        actions_l2=None,
        actions_l1=None,
    ):
        self.actions = (
            None
            if actions_l2 is None
            else (tuple(actions_l2), tuple(actions_l1))
        )

    def set_inference_frequency_intervention(self, mode):
        self.frequency = mode

    def set_inference_frequency_gain(self, *, detail_gain, ll_gain):
        self.detail_gain = detail_gain
        self.ll_gain = ll_gain

    def set_inference_frequency_window(self, *, logsnr_min, logsnr_max):
        self.minimum = logsnr_min
        self.maximum = logsnr_max

    def inference_destination_intervention(self):
        return {"mode": self.destination}

    def inference_route_action_intervention(self):
        return {"actions": self.actions}

    def inference_frequency_intervention(self):
        return {
            "mode": self.frequency,
            "detail_gain": self.detail_gain,
            "ll_gain": self.ll_gain,
            "logsnr_min": self.minimum,
            "logsnr_max": self.maximum,
        }


class _PositiveOnlyDataset(_Dataset):
    def __init__(self) -> None:
        super().__init__()
        for sample in self.samples:
            if float(sample["mask"].sum()) == 0.0:
                sample["mask"][:, :1, :1] = 1.0


def test_patient_balanced_subset_has_cap_and_required_strata() -> None:
    subset, cohort = build_patient_balanced_subset(
        _Dataset(),
        count=8,
        max_per_patient=2,
    )
    assert len(subset) == 8
    counts = {}
    for row in cohort:
        counts[row["patient_id"]] = counts.get(row["patient_id"], 0) + 1
    assert max(counts.values()) <= 2
    assert {
        "small_lesion",
        "large_lesion",
        "background",
    }.issubset({row["stratum"] for row in cohort})


def test_positive_only_cohort_requires_explicit_override() -> None:
    with pytest.raises(ValueError, match="background"):
        build_patient_balanced_subset(
            _PositiveOnlyDataset(),
            count=8,
            max_per_patient=2,
        )


def test_positive_only_cohort_preserves_lesion_strata_and_cap() -> None:
    subset, cohort = build_patient_balanced_subset(
        _PositiveOnlyDataset(),
        count=8,
        max_per_patient=2,
        allow_positive_only_cohort=True,
    )
    assert len(subset) == 8
    counts = {}
    for row in cohort:
        counts[row["patient_id"]] = counts.get(row["patient_id"], 0) + 1
    assert max(counts.values()) <= 2
    strata = {row["stratum"] for row in cohort}
    assert {"small_lesion", "large_lesion"}.issubset(strata)
    assert "background" not in strata


def test_all_validation_ignores_sampling_cap_and_preserves_order() -> None:
    dataset = _Dataset()
    subset, cohort = build_patient_balanced_subset(
        dataset,
        count=8,
        max_per_patient=2,
        all_validation=True,
    )
    assert len(subset) == len(dataset)
    assert [row["index"] for row in cohort] == list(range(len(dataset)))
    counts = {}
    for row in cohort:
        counts[row["patient_id"]] = counts.get(row["patient_id"], 0) + 1
    assert max(counts.values()) > 2


def test_all_nine_variants_reset_and_apply_exact_axes() -> None:
    router = _Router()
    descriptions = {
        variant: _configure_variant(router, variant)
        for variant in VARIANTS
    }
    assert descriptions["B1"]["frequency"]["mode"] == "all_frequency_off"
    assert descriptions["B4"]["route_action"]["actions"][0] == (
        "native",
        "native",
        "native",
    )
    assert descriptions["B5"]["route_action"]["actions"][0] == (
        "shallow",
        "shallow",
        "shallow",
    )
    assert descriptions["B6"]["frequency"]["detail_gain"] == 0.5
    assert descriptions["B7"]["frequency"]["detail_gain"] == 1.5
    assert descriptions["B8"]["frequency"]["logsnr_min"] == 0.0


def test_medoid_uses_generated_image_distances_only() -> None:
    stack = np.asarray([[[[[0.0]]]], [[[[1.0]]]], [[[[10.0]]]]])
    result = _medoid(stack)
    np.testing.assert_array_equal(result, np.asarray([[[[1.0]]]]))


def test_metric_row_exact_comparison_treats_nan_as_equal() -> None:
    assert _metric_rows_equal(
        [{"sample": "x", "value": float("nan")}],
        [{"sample": "x", "value": float("nan")}],
    )
    assert not _metric_rows_equal(
        [{"sample": "x", "value": 1.0}],
        [{"sample": "x", "value": 2.0}],
    )


def test_seed_parser_requires_four_distinct_values() -> None:
    assert _parse_seeds("42,43,44,45") == [42, 43, 44, 45]


def test_variant_parser_supports_b0_only_and_rejects_missing_baseline() -> None:
    assert _parse_variants("B0") == ["B0"]
    with pytest.raises(ValueError, match="include B0"):
        _parse_variants("B4,B5")
    with pytest.raises(ValueError, match="Unknown"):
        _parse_variants("B0,B9")
