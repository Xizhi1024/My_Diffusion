"""Bridge-aware multiband preconditioning for residual BBDM skip features."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .haar import haar_dwt2, haar_idwt2


class _ZeroProjection(nn.Module):
    """Small projection whose final convolution makes initialization an exact no-op."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        hidden = min(32, max(8, out_channels // 4))
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.SiLU(),
        )
        self.final = nn.Conv2d(hidden, out_channels, 1)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.features(x))


class _IndependentBandGates(nn.Module):
    def __init__(self, hidden_channels: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 3),
        )
        nn.init.normal_(self.net[-1].weight, std=0.02)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([-0.4, 0.0, 0.4]))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features))


class ResidualFrequencyPreconditioner(nn.Module):
    """Turn a noisy residual into level-matched, zero-init decoder injections."""

    def __init__(
        self,
        output_channels: Sequence[int] = (256, 256, 128, 64),
        band_scales: Sequence[float] = (1.0, 0.5, 0.25),
        inject_wavelet: bool = True,
        use_gabor_gate: bool = False,
        gabor_orientations: int = 8,
        gate_strength: float = 0.1,
    ):
        super().__init__()
        if len(output_channels) != 4:
            raise ValueError("output_channels must contain deep-to-shallow L3/L2/L1/L0 widths")
        if len(band_scales) != 3 or any(float(x) <= 0 for x in band_scales):
            raise ValueError("band_scales must contain three positive low/mid/high values")
        if not 0 <= gate_strength <= 0.5:
            raise ValueError("gate_strength must be in [0, 0.5]")

        self.output_channels = tuple(int(x) for x in output_channels)
        self.register_buffer("band_scales", torch.tensor(band_scales, dtype=torch.float32))
        self.inject_wavelet = bool(inject_wavelet)
        self.use_gabor_gate = bool(use_gabor_gate)
        self.gate_strength = float(gate_strength)
        self.band_gates = _IndependentBandGates()
        self.gabor_gate = nn.Conv2d(gabor_orientations, 1, 1)
        nn.init.zeros_(self.gabor_gate.weight)
        nn.init.zeros_(self.gabor_gate.bias)
        self.projection_heads = nn.ModuleList(
            [
                _ZeroProjection(1, self.output_channels[0]),
                _ZeroProjection(3, self.output_channels[1]),
                _ZeroProjection(3, self.output_channels[2]),
                _ZeroProjection(3, self.output_channels[3]),
            ]
        )

    def _gate_features(self, timesteps: torch.Tensor, schedule) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(schedule, "m_t") or not hasattr(schedule, "sigma_t"):
            raise ValueError("ResidualFrequencyPreconditioner requires a Brownian bridge schedule")
        tau = schedule.get_tau(timesteps).to(dtype=torch.float32)
        m = schedule.m_t[timesteps].to(device=timesteps.device, dtype=torch.float32)
        sigma = schedule.sigma_t[timesteps].to(device=timesteps.device, dtype=torch.float32)
        progress = 1.0 - m
        scales = self.band_scales.to(device=timesteps.device)
        signal_power = (progress[:, None] * scales[None, :]).square()
        noise_power = sigma[:, None].square().clamp_min(1e-8)
        log_snr = torch.log(signal_power.clamp_min(1e-8) / noise_power).clamp(-20.0, 20.0)
        features = torch.cat([tau[:, None], progress[:, None], log_snr], dim=1)
        return features, log_snr

    def gabor_factor(
        self,
        orientation: Optional[torch.Tensor],
        size: tuple[int, int],
    ) -> torch.Tensor | float:
        if not self.use_gabor_gate or orientation is None:
            return 1.0
        resized = F.interpolate(orientation, size=size, mode="bilinear", align_corners=False)
        return 1.0 + self.gate_strength * torch.tanh(self.gabor_gate(resized))

    def modulate_residual(
        self,
        residual: torch.Tensor,
        gabor_orientation: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Gate only level-1 Haar details and preserve the low-pass coefficient."""
        if not self.use_gabor_gate or gabor_orientation is None:
            return residual
        ll1, details = haar_dwt2(residual)
        factor = self.gabor_factor(gabor_orientation, ll1.shape[-2:])
        gated_details = tuple(detail * factor for detail in details)
        return haar_idwt2(ll1, gated_details)  # type: ignore[arg-type]

    def forward(
        self,
        noisy_residual: torch.Tensor,
        timesteps: torch.Tensor,
        schedule,
        gabor_orientation: Optional[torch.Tensor] = None,
    ) -> tuple[list[torch.Tensor], Dict[str, torch.Tensor]]:
        ll1, detail1 = haar_dwt2(noisy_residual)
        ll2, detail2 = haar_dwt2(ll1)
        low = F.interpolate(ll2, scale_factor=0.5, mode="bilinear", align_corners=False)
        middle = torch.cat(detail2, dim=1)
        high = torch.cat(detail1, dim=1)
        high_l0 = F.interpolate(high, scale_factor=2.0, mode="bilinear", align_corners=False)

        low = low / self.band_scales[0].to(low)
        middle = middle / self.band_scales[1].to(middle)
        high = high / self.band_scales[2].to(high)
        high_l0 = high_l0 / self.band_scales[2].to(high_l0)

        gate_features, log_snr = self._gate_features(timesteps, schedule)
        gates = self.band_gates(gate_features).to(dtype=noisy_residual.dtype)
        diagnostics = {
            "band_gates": gates,
            "band_log_snr": log_snr,
        }
        if not self.inject_wavelet:
            return [], diagnostics
        high = high * self.gabor_factor(gabor_orientation, high.shape[-2:])
        high_l0 = high_l0 * self.gabor_factor(gabor_orientation, high_l0.shape[-2:])

        raw_bands = (low, middle, high, high_l0)
        gate_columns = (0, 1, 2, 2)
        injections = []
        for head, band, gate_column in zip(self.projection_heads, raw_bands, gate_columns):
            gate = gates[:, gate_column]
            while gate.ndim < band.ndim:
                gate = gate.unsqueeze(-1)
            injections.append(head(band * gate))

        return injections, diagnostics
