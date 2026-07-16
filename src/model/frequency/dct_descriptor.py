"""Small selected-frequency DCT descriptor for Haar detail energy maps."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


_SELECTED_8X8 = (
    (0, 1), (1, 0), (1, 1), (0, 2),
    (2, 0), (1, 2), (2, 1), (2, 2),
    (0, 3), (3, 0), (1, 3), (3, 1),
)


def _dct_basis(size: int, frequencies: tuple[tuple[int, int], ...]) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32)
    rows = []
    for u, v in frequencies:
        cu = math.sqrt(1 / size) if u == 0 else math.sqrt(2 / size)
        cv = math.sqrt(1 / size) if v == 0 else math.sqrt(2 / size)
        basis_u = cu * torch.cos(math.pi * (2 * coords + 1) * u / (2 * size))
        basis_v = cv * torch.cos(math.pi * (2 * coords + 1) * v / (2 * size))
        rows.append(torch.outer(basis_u, basis_v))
    return torch.stack(rows, dim=0)


class SelectedDCTDescriptor(nn.Module):
    def __init__(self, pooled_size: int = 8, selected_frequencies: int = 12) -> None:
        super().__init__()
        if pooled_size != 8:
            raise ValueError("V5 selected DCT requires pooled_size=8")
        if not 1 <= selected_frequencies <= len(_SELECTED_8X8):
            raise ValueError("selected_frequencies must be in [1, 12]")
        frequencies = _SELECTED_8X8[:selected_frequencies]
        self.pooled_size = pooled_size
        self.selected_frequencies = selected_frequencies
        self.register_buffer("basis", _dct_basis(pooled_size, frequencies))

        prior = torch.tensor(
            [1.0, 1.0, 1.2, 1.2, 1.2, 1.4, 1.4, 1.4, 1.1, 1.1, 1.0, 1.0],
            dtype=torch.float32,
        )[:selected_frequencies]
        prior = prior / prior.sum()
        self.register_buffer("prior_weights", prior)
        self.weight_offsets = nn.Parameter(torch.zeros(selected_frequencies))

    def frequency_weights(self) -> torch.Tensor:
        logits = self.prior_weights.clamp_min(1e-8).log() + self.weight_offsets
        return torch.softmax(logits, dim=0)

    def forward(
        self, details: torch.Tensor
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if details.ndim != 4 or details.shape[1] != 3:
            raise ValueError("DCT descriptor expects [B,3,H,W] Haar details")
        energy = torch.log1p(details.abs())
        pooled = F.adaptive_avg_pool2d(
            energy, output_size=(self.pooled_size, self.pooled_size)
        )
        pooled_scale = pooled.abs().mean(dim=(-2, -1)).unsqueeze(-1)
        pooled = pooled - pooled.mean(dim=(-2, -1), keepdim=True)
        basis = self.basis.to(device=details.device, dtype=details.dtype)
        coefficients = torch.einsum("bchw,khw->bck", pooled, basis).abs()
        coefficient_sum = coefficients.sum(dim=-1, keepdim=True)
        dtype_info = torch.finfo(coefficients.dtype)
        normalization_floor = pooled_scale * math.sqrt(dtype_info.eps)
        denominator = torch.maximum(coefficient_sum, normalization_floor)
        coefficients = coefficients / denominator.clamp_min(dtype_info.tiny)
        weights = self.frequency_weights().to(details)
        descriptor = coefficients * weights.view(1, 1, -1)
        return descriptor, {
            "frequency_weights": weights,
            "weight_offset_energy": self.weight_offsets.square().mean(),
        }
