"""Low-frequency PET supervision inside the CT body and outside lesions.

The target failure mode is not a uniform global darkness: validation medians
are already close while the upper physiological-uptake quantiles are too cold.
This term therefore combines a body-wide low-pass reconstruction loss with a
soft extra weight above each sample's target low-pass quantile.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ..interfaces import LossContext, LossTerm
from .base import smooth_tau_gate


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    mask = (mask > 0.5).to(dtype=mask.dtype)
    if radius <= 0:
        return mask
    width = 2 * int(radius) + 1
    return F.max_pool2d(mask, kernel_size=width, stride=1, padding=radius)


def _binary_closing(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Torch-only binary closing suitable for a fixed, non-gradient mask."""
    if radius <= 0:
        return (mask > 0.5).to(dtype=mask.dtype)
    dilated = _dilate(mask, radius)
    width = 2 * int(radius) + 1
    eroded = 1.0 - F.max_pool2d(
        1.0 - dilated,
        kernel_size=width,
        stride=1,
        padding=radius,
    )
    return (eroded > 0.5).to(dtype=mask.dtype)


class NonLesionBodyLowpassLoss(LossTerm):
    """Match the non-lesion physiological uptake field at low frequency.

    Inputs are expected in the repository's model range ``[-1, 1]``.  PET and
    CT are converted to unit scale internally.  The CT-derived body mask and
    target-derived tail weights are supervision-only and are never model
    inputs, so they introduce no inference-time mask or PET leakage.
    """

    name = "nonlesion_body_lowpass"

    def __init__(
        self,
        sigma: float = 4.0,
        body_threshold: float = 0.03,
        body_closing_radius: int = 2,
        lesion_exclusion_radius: int = 8,
        tail_quantile: float = 0.75,
        tail_weight: float = 2.0,
        tail_temperature: float = 0.02,
        charbonnier_eps: float = 1.0e-3,
        active_tau_max: float = 0.70,
        enabled: bool = True,
        weight: float = 0.2,
    ):
        super().__init__(enabled=enabled, weight=weight)
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        if not 0.0 <= body_threshold <= 1.0:
            raise ValueError("body_threshold must be in [0, 1]")
        if body_closing_radius < 0:
            raise ValueError("body_closing_radius must be >= 0")
        if lesion_exclusion_radius < 0:
            raise ValueError("lesion_exclusion_radius must be >= 0")
        if not 0.0 < tail_quantile < 1.0:
            raise ValueError("tail_quantile must be in (0, 1)")
        if tail_weight < 0:
            raise ValueError("tail_weight must be >= 0")
        if tail_temperature <= 0:
            raise ValueError("tail_temperature must be > 0")
        if charbonnier_eps < 0:
            raise ValueError("charbonnier_eps must be >= 0")
        if not 0.0 < active_tau_max <= 1.0:
            raise ValueError("active_tau_max must be in (0, 1]")

        self.sigma = float(sigma)
        self.body_threshold = float(body_threshold)
        self.body_closing_radius = int(body_closing_radius)
        self.lesion_exclusion_radius = int(lesion_exclusion_radius)
        self.tail_quantile = float(tail_quantile)
        self.tail_weight = float(tail_weight)
        self.tail_temperature = float(tail_temperature)
        self.charbonnier_eps = float(charbonnier_eps)
        self.active_tau_max = float(active_tau_max)

        radius = max(1, int(math.ceil(3.0 * self.sigma)))
        coordinates = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (coordinates / self.sigma).square())
        kernel = kernel / kernel.sum()
        # This derived constant must not change checkpoint compatibility when
        # the loss is added to a fine-tuning config.
        self.register_buffer(
            "_gaussian_kernel",
            kernel,
            persistent=False,
        )

    def _gaussian_blur(self, image: torch.Tensor) -> torch.Tensor:
        channels = int(image.shape[1])
        kernel = self._gaussian_kernel.to(device=image.device, dtype=image.dtype)
        radius = int(kernel.numel() // 2)
        horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
        vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
        blurred = F.conv2d(
            F.pad(image, (radius, radius, 0, 0), mode="replicate"),
            horizontal,
            groups=channels,
        )
        return F.conv2d(
            F.pad(blurred, (0, 0, radius, radius), mode="replicate"),
            vertical,
            groups=channels,
        )

    @staticmethod
    def _unit_interval(image: torch.Tensor, *, clamp: bool) -> torch.Tensor:
        converted = (image.float() + 1.0) * 0.5
        return converted.clamp(0.0, 1.0) if clamp else converted

    def forward(
        self,
        ctx: LossContext,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        target = ctx.target_pet
        device = target.device
        zero = target.new_zeros(())
        if not self.enabled:
            return zero, {
                f"{self.name}/enabled": zero,
                f"{self.name}/loss": zero,
            }

        ct = ctx.batch.get("ct")
        if ct is None:
            raise ValueError(f"{self.name} requires batch['ct']")
        pred = ctx.pred_x0 if ctx.pred_x0 is not None else ctx.model_pred

        # Target/CT are fixed supervision.  Prediction remains unclamped so an
        # out-of-range prediction still receives a gradient back toward target.
        target_unit = self._unit_interval(target, clamp=True)
        pred_unit = self._unit_interval(pred, clamp=False)
        ct_unit = self._unit_interval(
            ct.to(device=device, dtype=target.dtype),
            clamp=True,
        )
        target_low = self._gaussian_blur(target_unit)
        pred_low = self._gaussian_blur(pred_unit)

        with torch.no_grad():
            body = _binary_closing(
                (ct_unit > self.body_threshold).to(dtype=target_low.dtype),
                self.body_closing_radius,
            )
            lesion = ctx.batch.get("mask")
            if lesion is None:
                excluded = torch.zeros_like(body)
            else:
                excluded = _dilate(
                    lesion.to(device=device, dtype=target_low.dtype),
                    self.lesion_exclusion_radius,
                )
            region = body * (1.0 - excluded).clamp(0.0, 1.0)

        absolute_error = (pred_low - target_low).abs()
        if self.charbonnier_eps > 0:
            eps = self.charbonnier_eps
            point_error = torch.sqrt(
                (pred_low - target_low).square() + eps * eps
            ) - eps
        else:
            point_error = absolute_error

        per_sample_losses = []
        per_sample_mae = []
        per_sample_tail_mae = []
        per_sample_bias = []
        thresholds = []
        valid = []
        for index in range(pred_low.shape[0]):
            selected = region[index] > 0.5
            if not bool(selected.any()):
                per_sample_losses.append(zero)
                per_sample_mae.append(zero)
                per_sample_tail_mae.append(zero)
                per_sample_bias.append(zero)
                thresholds.append(zero)
                valid.append(False)
                continue

            threshold = torch.quantile(
                target_low[index][selected].detach().float(),
                self.tail_quantile,
            ).to(target_low)
            tail_soft = torch.sigmoid(
                (target_low[index] - threshold) / self.tail_temperature
            )
            weights = region[index] * (1.0 + self.tail_weight * tail_soft)
            denom = weights.sum().clamp_min(1.0)
            base_denom = region[index].sum().clamp_min(1.0)
            hard_tail = region[index] * (target_low[index] >= threshold).to(
                dtype=region.dtype
            )
            tail_denom = hard_tail.sum().clamp_min(1.0)

            per_sample_losses.append((point_error[index] * weights).sum() / denom)
            per_sample_mae.append(
                (absolute_error[index] * region[index]).sum() / base_denom
            )
            per_sample_tail_mae.append(
                (absolute_error[index] * hard_tail).sum() / tail_denom
            )
            per_sample_bias.append(
                ((pred_low[index] - target_low[index]) * region[index]).sum()
                / base_denom
            )
            thresholds.append(threshold)
            valid.append(True)

        valid_tensor = torch.tensor(
            valid,
            device=device,
            dtype=target_low.dtype,
        )
        valid_denom = valid_tensor.sum().clamp_min(1.0)
        loss_vector = torch.stack(per_sample_losses)
        gate = smooth_tau_gate(
            ctx.tau,
            max_tau=self.active_tau_max,
        ).to(device=device, dtype=target_low.dtype)
        loss = (loss_vector * gate * valid_tensor).sum() / valid_denom

        def _valid_mean(values: list[torch.Tensor]) -> torch.Tensor:
            stacked = torch.stack(values)
            return (stacked * valid_tensor).sum() / valid_denom

        return loss * self.weight, {
            f"{self.name}/loss": loss.detach(),
            f"{self.name}/lowpass_mae": _valid_mean(per_sample_mae).detach(),
            f"{self.name}/tail_lowpass_mae": _valid_mean(
                per_sample_tail_mae
            ).detach(),
            f"{self.name}/signed_bias": _valid_mean(per_sample_bias).detach(),
            f"{self.name}/target_tail_threshold": _valid_mean(
                thresholds
            ).detach(),
            f"{self.name}/gate_mean": gate.mean().detach(),
            f"{self.name}/body_nonlesion_pixels": region.sum().detach(),
            f"{self.name}/valid_samples": valid_tensor.sum().detach(),
            f"{self.name}/enabled": target.new_tensor(1.0),
        }
