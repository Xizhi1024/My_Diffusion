from __future__ import annotations

import pytest
import torch

from src.mechanism_validation.magnification import (
    CropBox,
    backproject_crop,
    bootstrap_patient_balanced_slope,
    couple_noise_to_crop,
    crop_resize,
    lesion_crop_box,
    patient_balanced_slope,
)
from src.model.slmf_bbdm import SLMFBBDM


def _model() -> SLMFBBDM:
    model = SLMFBBDM(
        image_size=32,
        objective="pred_x0",
        enable_heteroscedastic=False,
        sample_scheduler="ddim",
        eval_sampling_steps=2,
    )
    return model.eval()


def _batch() -> dict[str, torch.Tensor]:
    return {
        "ct": torch.linspace(-1.0, 1.0, 32 * 32).reshape(1, 1, 32, 32),
        "mask": torch.zeros(1, 1, 32, 32),
        "organ_mask": torch.zeros(1, 6, 32, 32),
        "mu_map": torch.zeros(1, 1, 32, 32),
    }


def test_explicit_initial_noise_makes_sampling_reproducible() -> None:
    model = _model()
    batch = _batch()
    noise = torch.randn_like(batch["ct"])

    first = model.sample(batch, num_steps=2, initial_noise=noise)["synthetic_pet"]
    torch.manual_seed(987654)
    second = model.sample(batch, num_steps=2, initial_noise=noise)["synthetic_pet"]

    assert torch.equal(first, second)


def test_explicit_initial_noise_bypasses_internal_draw(monkeypatch) -> None:
    model = _model()
    batch = _batch()
    noise = torch.zeros_like(batch["ct"])

    def fail_internal_draw(*args, **kwargs):
        raise AssertionError("torch.randn_like must not run for explicit noise")

    monkeypatch.setattr(torch, "randn_like", fail_internal_draw)
    result = model.sample(batch, num_steps=1, initial_noise=noise)

    assert torch.isfinite(result["synthetic_pet"]).all()


@pytest.mark.parametrize(
    "noise,match",
    [
        (torch.zeros(1, 1, 16, 16), "shape"),
        (torch.zeros(1, 1, 32, 32, dtype=torch.float64), "dtype"),
        (torch.empty(1, 1, 32, 32, device="meta"), "device"),
    ],
)
def test_explicit_initial_noise_rejects_incompatible_tensor(
    noise: torch.Tensor,
    match: str,
) -> None:
    model = _model()

    with pytest.raises(ValueError, match=match):
        model.sample(_batch(), num_steps=1, initial_noise=noise)


def test_omitted_initial_noise_preserves_random_sampling(monkeypatch) -> None:
    model = _model()
    batch = _batch()
    sentinel = torch.full_like(batch["ct"], 0.125)
    calls = 0

    def fixed_internal_draw(reference: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        assert reference is batch["ct"]
        return sentinel

    monkeypatch.setattr(torch, "randn_like", fixed_internal_draw)
    result = model.sample(batch, num_steps=1)

    assert calls == 1
    assert torch.isfinite(result["synthetic_pet"]).all()


def test_lesion_crop_box_is_centered_and_deterministic() -> None:
    mask = torch.zeros(1, 32, 32)
    mask[:, 10:14, 20:24] = 1.0

    box = lesion_crop_box(mask, crop_size=8)

    assert box == CropBox(top=8, left=18, size=8)
    assert lesion_crop_box(mask, crop_size=8) == box


def test_lesion_crop_box_shifts_at_boundary_without_truncation() -> None:
    mask = torch.zeros(1, 16, 16)
    mask[:, 0:3, 14:16] = 1.0

    box = lesion_crop_box(mask, crop_size=8)

    assert box == CropBox(top=0, left=8, size=8)
    cropped = mask[:, box.top : box.bottom, box.left : box.right]
    assert cropped.sum() == mask.sum()


def test_lesion_crop_box_rejects_empty_or_oversized_lesion() -> None:
    with pytest.raises(ValueError, match="empty"):
        lesion_crop_box(torch.zeros(1, 16, 16), crop_size=8)

    mask = torch.zeros(1, 16, 16)
    mask[:, 2:12, 4:6] = 1.0
    with pytest.raises(ValueError, match="does not fit"):
        lesion_crop_box(mask, crop_size=8)


def test_crop_resize_and_backprojection_preserve_constant_region() -> None:
    image = torch.zeros(1, 1, 16, 16)
    image[:, :, 4:12, 4:12] = 0.75
    box = CropBox(top=4, left=4, size=8)

    zoom = crop_resize(image, box, output_size=16, mode="bilinear")
    restored = backproject_crop(
        zoom,
        box,
        canvas_size=(16, 16),
        mode="bilinear",
    )

    assert zoom.shape == (1, 1, 16, 16)
    assert torch.allclose(restored[:, :, 4:12, 4:12], image[:, :, 4:12, 4:12])
    assert restored[:, :, :4].count_nonzero() == 0


def test_nearest_mask_resize_remains_binary() -> None:
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, 6:9, 7:10] = 1.0
    box = CropBox(top=4, left=4, size=8)

    zoom = crop_resize(mask, box, output_size=32, mode="nearest")

    assert set(torch.unique(zoom).tolist()) == {0.0, 1.0}


def test_coupled_noise_is_standardized_and_anatomically_indexed() -> None:
    full_noise = torch.arange(16 * 16, dtype=torch.float32).reshape(1, 1, 16, 16)
    box = CropBox(top=4, left=6, size=8)

    coupled = couple_noise_to_crop(full_noise, box, output_size=16)
    raw_zoom = crop_resize(full_noise, box, output_size=16, mode="bilinear")

    assert coupled.shape == full_noise.shape
    assert torch.allclose(coupled.mean(dim=(-2, -1)), torch.zeros(1, 1), atol=1e-6)
    assert torch.allclose(
        coupled.std(dim=(-2, -1), unbiased=False),
        torch.ones(1, 1),
        atol=1e-6,
    )
    assert torch.equal(coupled.argsort(dim=-1), raw_zoom.argsort(dim=-1))


def test_patient_balanced_slope_is_invariant_to_slice_duplication() -> None:
    base_patients = ["a", "b", "c"]
    base_x = [0.0, 1.0, 3.0]
    base_y = [0.0, 4.0, 3.0]
    duplicated_patients = ["a", *(["b"] * 20), "c"]
    duplicated_x = [0.0, *([1.0] * 20), 3.0]
    duplicated_y = [0.0, *([4.0] * 20), 3.0]

    base = patient_balanced_slope(base_patients, base_x, base_y)
    duplicated = patient_balanced_slope(
        duplicated_patients,
        duplicated_x,
        duplicated_y,
    )

    assert duplicated == pytest.approx(base)


def test_patient_bootstrap_is_deterministic_and_keeps_linear_effect() -> None:
    patients = ["a", "a", "b", "b", "c", "c", "d", "d"]
    x = [0.0, 0.2, 1.0, 1.2, 2.0, 2.2, 3.0, 3.2]
    y = [2.0 + 3.0 * value for value in x]

    first = bootstrap_patient_balanced_slope(
        patients,
        x,
        y,
        seed=17,
        replicates=500,
        x_reference=1.0,
    )
    second = bootstrap_patient_balanced_slope(
        patients,
        x,
        y,
        seed=17,
        replicates=500,
        x_reference=1.0,
    )

    assert first == second
    assert first.slope == pytest.approx(3.0)
    assert first.slope_ci95_low == pytest.approx(3.0)
    assert first.slope_ci95_high == pytest.approx(3.0)
    assert first.reference_prediction == pytest.approx(5.0)
    assert first.valid_replicates > 0
