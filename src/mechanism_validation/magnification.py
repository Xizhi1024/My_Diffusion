"""Pure geometry, noise, and statistics for lesion magnification audits.

This module deliberately has no filesystem, checkpoint, model, or CLI
dependencies.  The same primitives can therefore be reused by a frozen
mechanism audit and, only after that audit passes, by a training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CropBox:
    """Integer square crop in top/left/size form."""

    top: int
    left: int
    size: int

    def __post_init__(self) -> None:
        if self.top < 0 or self.left < 0:
            raise ValueError("crop top and left must be non-negative")
        if self.size <= 0:
            raise ValueError("crop size must be positive")

    @property
    def bottom(self) -> int:
        return self.top + self.size

    @property
    def right(self) -> int:
        return self.left + self.size


@dataclass(frozen=True)
class MagnificationView:
    """Declared relationship between a source crop and model input grid."""

    box: CropBox
    output_size: int

    def __post_init__(self) -> None:
        if self.output_size <= 0:
            raise ValueError("output_size must be positive")

    @property
    def factor(self) -> float:
        return float(self.output_size / self.box.size)


@dataclass(frozen=True)
class PatientBootstrapResult:
    """Patient-resampled linear-fit estimate and percentile intervals."""

    patients: int
    requested_replicates: int
    valid_replicates: int
    intercept: float
    slope: float
    slope_ci95_low: float
    slope_ci95_high: float
    x_reference: float
    reference_prediction: float
    reference_ci95_low: float
    reference_ci95_high: float


def lesion_crop_box(mask: torch.Tensor, crop_size: int) -> CropBox:
    """Return a fixed square crop containing the complete non-empty lesion."""

    if crop_size <= 0:
        raise ValueError("crop_size must be positive")
    if mask.ndim < 2:
        raise ValueError("mask must have at least two spatial dimensions")

    height, width = (int(mask.shape[-2]), int(mask.shape[-1]))
    if crop_size > min(height, width):
        raise ValueError(
            f"crop_size={crop_size} exceeds image bounds {(height, width)}"
        )

    lesion = mask > 0.5
    lesion_2d = lesion.reshape(-1, height, width).any(dim=0)
    coordinates = torch.nonzero(lesion_2d, as_tuple=False)
    if coordinates.numel() == 0:
        raise ValueError("lesion mask is empty")

    y_min = int(coordinates[:, 0].min().item())
    y_max = int(coordinates[:, 0].max().item())
    x_min = int(coordinates[:, 1].min().item())
    x_max = int(coordinates[:, 1].max().item())
    lesion_height = y_max - y_min + 1
    lesion_width = x_max - x_min + 1
    if lesion_height > crop_size or lesion_width > crop_size:
        raise ValueError(
            "lesion bounding box does not fit requested crop: "
            f"{(lesion_height, lesion_width)} > {crop_size}"
        )

    center_y = (y_min + y_max + 1) // 2
    center_x = (x_min + x_max + 1) // 2
    top = min(max(center_y - crop_size // 2, 0), height - crop_size)
    left = min(max(center_x - crop_size // 2, 0), width - crop_size)
    box = CropBox(top=top, left=left, size=crop_size)
    if not (
        box.top <= y_min
        and y_max < box.bottom
        and box.left <= x_min
        and x_max < box.right
    ):
        raise RuntimeError("computed crop unexpectedly truncates the lesion")
    return box


def _as_4d(tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
    original_dimensions = tensor.ndim
    if original_dimensions == 2:
        return tensor[None, None], original_dimensions
    if original_dimensions == 3:
        return tensor[None], original_dimensions
    if original_dimensions == 4:
        return tensor, original_dimensions
    raise ValueError(
        "tensor must have shape [H,W], [C,H,W], or [B,C,H,W]"
    )


def _restore_dimensions(tensor: torch.Tensor, dimensions: int) -> torch.Tensor:
    if dimensions == 2:
        return tensor[0, 0]
    if dimensions == 3:
        return tensor[0]
    return tensor


def _interpolate(
    tensor: torch.Tensor,
    *,
    size: tuple[int, int],
    mode: str,
) -> torch.Tensor:
    if mode == "nearest":
        return F.interpolate(tensor, size=size, mode=mode)
    if mode != "bilinear":
        raise ValueError("mode must be 'bilinear' or 'nearest'")
    return F.interpolate(
        tensor,
        size=size,
        mode=mode,
        align_corners=False,
    )


def crop_resize(
    tensor: torch.Tensor,
    box: CropBox,
    *,
    output_size: int,
    mode: str,
) -> torch.Tensor:
    """Crop a declared square and resize it to a square output grid."""

    if output_size <= 0:
        raise ValueError("output_size must be positive")
    tensor_4d, dimensions = _as_4d(tensor)
    height, width = tensor_4d.shape[-2:]
    if box.bottom > height or box.right > width:
        raise ValueError(
            f"crop {box} exceeds tensor bounds {(height, width)}"
        )
    cropped = tensor_4d[..., box.top : box.bottom, box.left : box.right]
    resized = _interpolate(
        cropped,
        size=(output_size, output_size),
        mode=mode,
    )
    return _restore_dimensions(resized, dimensions)


def backproject_crop(
    zoom_tensor: torch.Tensor,
    box: CropBox,
    *,
    canvas_size: tuple[int, int],
    mode: str,
    base: torch.Tensor | None = None,
) -> torch.Tensor:
    """Resize a zoom result to its source crop and insert it on a canvas."""

    zoom_4d, dimensions = _as_4d(zoom_tensor)
    height, width = (int(canvas_size[0]), int(canvas_size[1]))
    if height <= 0 or width <= 0:
        raise ValueError("canvas_size must be positive")
    if box.bottom > height or box.right > width:
        raise ValueError(
            f"crop {box} exceeds canvas bounds {(height, width)}"
        )

    patch = _interpolate(
        zoom_4d,
        size=(box.size, box.size),
        mode=mode,
    )
    if base is None:
        canvas = zoom_4d.new_zeros(
            zoom_4d.shape[0],
            zoom_4d.shape[1],
            height,
            width,
        )
    else:
        base_4d, _ = _as_4d(base)
        expected = (
            zoom_4d.shape[0],
            zoom_4d.shape[1],
            height,
            width,
        )
        if tuple(base_4d.shape) != expected:
            raise ValueError(
                f"base shape must be {expected}, got {tuple(base_4d.shape)}"
            )
        canvas = base_4d.clone()
    canvas[
        ...,
        box.top : box.bottom,
        box.left : box.right,
    ] = patch
    return _restore_dimensions(canvas, dimensions)


def couple_noise_to_crop(
    full_noise: torch.Tensor,
    box: CropBox,
    *,
    output_size: int,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Create anatomically indexed, marginally standardized zoom noise."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    transformed = crop_resize(
        full_noise,
        box,
        output_size=output_size,
        mode="bilinear",
    )
    transformed_4d, dimensions = _as_4d(transformed)
    mean = transformed_4d.mean(dim=(-2, -1), keepdim=True)
    standard_deviation = transformed_4d.std(
        dim=(-2, -1),
        keepdim=True,
        unbiased=False,
    )
    if bool((standard_deviation <= epsilon).any()):
        raise ValueError("transformed noise has near-zero spatial variance")
    standardized = (transformed_4d - mean) / standard_deviation
    return _restore_dimensions(standardized, dimensions)


def _linear_inputs(
    patient_ids: Sequence[object],
    x: Sequence[float],
    y: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patients = np.asarray(patient_ids, dtype=object)
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if not (patients.ndim == x_values.ndim == y_values.ndim == 1):
        raise ValueError("patient_ids, x, and y must be one-dimensional")
    if not (len(patients) == len(x_values) == len(y_values)):
        raise ValueError("patient_ids, x, and y must have equal length")
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    patients = patients[finite]
    x_values = x_values[finite]
    y_values = y_values[finite]
    if x_values.size < 2:
        raise ValueError("at least two finite observations are required")
    if len(set(patients.tolist())) < 2:
        raise ValueError("at least two patients are required")
    return patients, x_values, y_values


def _patient_balanced_linear_fit(
    patient_ids: Sequence[object],
    x: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float]:
    patients, x_values, y_values = _linear_inputs(patient_ids, x, y)
    counts: dict[object, int] = {}
    for patient in patients:
        counts[patient] = counts.get(patient, 0) + 1
    weights = np.asarray(
        [1.0 / counts[patient] for patient in patients],
        dtype=np.float64,
    )
    weight_sum = float(weights.sum())
    x_mean = float(np.sum(weights * x_values) / weight_sum)
    y_mean = float(np.sum(weights * y_values) / weight_sum)
    centered_x = x_values - x_mean
    denominator = float(np.sum(weights * centered_x * centered_x))
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("x has no patient-balanced variance")
    slope = float(
        np.sum(weights * centered_x * (y_values - y_mean)) / denominator
    )
    intercept = float(y_mean - slope * x_mean)
    return intercept, slope


def patient_balanced_slope(
    patient_ids: Sequence[object],
    x: Sequence[float],
    y: Sequence[float],
) -> float:
    """Fit a slope while giving every patient total weight one."""

    _, slope = _patient_balanced_linear_fit(patient_ids, x, y)
    return slope


def bootstrap_patient_balanced_slope(
    patient_ids: Sequence[object],
    x: Sequence[float],
    y: Sequence[float],
    *,
    seed: int,
    replicates: int,
    x_reference: float | None = None,
) -> PatientBootstrapResult:
    """Patient-cluster bootstrap for slope and a reference prediction."""

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    patients, x_values, y_values = _linear_inputs(patient_ids, x, y)
    unique_patients = list(dict.fromkeys(patients.tolist()))
    intercept, slope = _patient_balanced_linear_fit(
        patients,
        x_values,
        y_values,
    )
    if x_reference is None:
        patient_x_means = [
            float(x_values[patients == patient].mean())
            for patient in unique_patients
        ]
        x_reference = float(np.mean(patient_x_means))
    if not np.isfinite(x_reference):
        raise ValueError("x_reference must be finite")

    groups = {
        patient: np.flatnonzero(patients == patient)
        for patient in unique_patients
    }
    rng = np.random.default_rng(seed)
    slope_draws: list[float] = []
    reference_draws: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(
            unique_patients,
            size=len(unique_patients),
            replace=True,
        )
        sampled_x: list[np.ndarray] = []
        sampled_y: list[np.ndarray] = []
        sampled_units: list[np.ndarray] = []
        for unit_index, patient in enumerate(sampled.tolist()):
            indices = groups[patient]
            sampled_x.append(x_values[indices])
            sampled_y.append(y_values[indices])
            sampled_units.append(
                np.full(indices.size, unit_index, dtype=np.int64)
            )
        try:
            draw_intercept, draw_slope = _patient_balanced_linear_fit(
                np.concatenate(sampled_units),
                np.concatenate(sampled_x),
                np.concatenate(sampled_y),
            )
        except ValueError:
            continue
        slope_draws.append(draw_slope)
        reference_draws.append(
            draw_intercept + draw_slope * float(x_reference)
        )
    if not slope_draws:
        raise ValueError("all patient bootstrap draws were degenerate")

    slope_low, slope_high = np.quantile(slope_draws, (0.025, 0.975))
    reference_low, reference_high = np.quantile(
        reference_draws,
        (0.025, 0.975),
    )
    return PatientBootstrapResult(
        patients=len(unique_patients),
        requested_replicates=replicates,
        valid_replicates=len(slope_draws),
        intercept=intercept,
        slope=slope,
        slope_ci95_low=float(slope_low),
        slope_ci95_high=float(slope_high),
        x_reference=float(x_reference),
        reference_prediction=float(
            intercept + slope * float(x_reference)
        ),
        reference_ci95_low=float(reference_low),
        reference_ci95_high=float(reference_high),
    )

