"""Boundary-reliable subband injection for residual Brownian bridges.

The module is deliberately decoder-only.  It reads the current bridge state,
CT, timestep, and optional phase-insensitive Gabor orientation energy, but it
never rewrites the diffusion state.  Only gated Haar detail residuals are
projected into decoder skips.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .haar import HaarDetails, haar_dwt2, haar_idwt2


class _ZeroProjection(nn.Module):
    """Small residual head whose final convolution starts at exact zero."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
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


class _ContentGate(nn.Module):
    """Two-layer global-statistics gate for three Haar detail bands."""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, statistics: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(statistics))


def _stack_details(details: HaarDetails) -> torch.Tensor:
    return torch.cat(details, dim=1)


def _split_details(details: torch.Tensor) -> HaarDetails:
    if details.shape[1] != 3:
        raise ValueError("A one-channel Haar detail tensor must have three channels")
    return details[:, 0:1], details[:, 1:2], details[:, 2:3]


def _total_variation(gate: torch.Tensor) -> torch.Tensor:
    dy = (gate[..., 1:, :] - gate[..., :-1, :]).abs().mean()
    dx = (gate[..., :, 1:] - gate[..., :, :-1]).abs().mean()
    return dx + dy


class BoundaryReliableFrequencyInjector(nn.Module):
    """Inject reliability-filtered L2/L1/L0 high-frequency skip residuals.

    Decoder order is deep to shallow: L3, L2, L1, L0.  L3 is always exact
    zero.  L0 is reconstructed as ``IDWT(0, LH1, HL1, HH1)`` so low-frequency
    CT/PET content cannot be duplicated into the shallow skip.
    """

    inject_wavelet = True

    def __init__(
        self,
        output_channels: Sequence[int] = (256, 256, 128, 64),
        band_scales: Sequence[float] = (0.5, 0.25),
        use_noise_release: bool = True,
        use_ct_reliability: bool = True,
        use_subband_gates: bool = True,
        use_directional_reliability: bool = False,
        gabor_orientations: int = 8,
        gate_max: float = 0.25,
        snr_center: float = 0.0,
        snr_temperature: float = 2.0,
        cross_temperature: float = 1.0,
        content_hidden_channels: int = 16,
    ) -> None:
        super().__init__()
        if len(output_channels) != 4:
            raise ValueError("output_channels must contain L3/L2/L1/L0 widths")
        if len(band_scales) != 2 or any(float(scale) <= 0 for scale in band_scales):
            raise ValueError("band_scales must contain positive L2/L1 values")
        if not 0.0 < gate_max <= 0.5:
            raise ValueError("gate_max must be in (0, 0.5]")
        if snr_temperature <= 0:
            raise ValueError("snr_temperature must be positive")
        if cross_temperature <= 0:
            raise ValueError("cross_temperature must be positive")
        if content_hidden_channels <= 0:
            raise ValueError("content_hidden_channels must be positive")
        if gabor_orientations <= 0:
            raise ValueError("gabor_orientations must be positive")

        self.output_channels = tuple(int(channel) for channel in output_channels)
        self.use_noise_release = bool(use_noise_release)
        self.use_ct_reliability = bool(use_ct_reliability)
        self.use_subband_gates = bool(use_subband_gates)
        self.use_directional_reliability = bool(use_directional_reliability)
        self.gabor_orientations = int(gabor_orientations)
        self.gate_max = float(gate_max)
        self.snr_center = float(snr_center)
        self.snr_temperature = float(snr_temperature)
        self.cross_temperature = float(cross_temperature)
        self.register_buffer(
            "band_scales", torch.tensor(tuple(band_scales), dtype=torch.float32)
        )

        # Native levels L2/L1 plus pure high-frequency reconstruction at L0.
        self.projection_heads = nn.ModuleList(
            (
                _ZeroProjection(3, self.output_channels[1]),
                _ZeroProjection(3, self.output_channels[2]),
                _ZeroProjection(1, self.output_channels[3]),
            )
        )
        self.content_gates = nn.ModuleList(
            _ContentGate(content_hidden_channels) for _ in range(2)
        )

    @staticmethod
    def decompose(image: torch.Tensor) -> tuple[torch.Tensor, HaarDetails, HaarDetails]:
        """Return LL1, native level-1 details, and native level-2 details."""
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("Boundary-reliable Haar analysis expects [B,1,H,W]")
        ll1, details1 = haar_dwt2(image)
        _, details2 = haar_dwt2(ll1)
        return ll1, details1, details2

    @staticmethod
    def reconstruct_l0(details1: HaarDetails) -> torch.Tensor:
        """Subband-aligned inverse reconstruction of a pure HF increment."""
        return haar_idwt2(torch.zeros_like(details1[0]), details1)

    def noise_reliability(self, timesteps: torch.Tensor, schedule) -> torch.Tensor:
        """Return analytic bridge reliability for native levels [L2, L1]."""
        if not hasattr(schedule, "m_t") or not hasattr(schedule, "sigma_t"):
            raise ValueError(
                "BoundaryReliableFrequencyInjector requires a Brownian bridge schedule"
            )
        if not self.use_noise_release:
            return torch.ones(
                timesteps.shape[0], 2, device=timesteps.device, dtype=torch.float32
            )
        m = schedule.m_t[timesteps].to(device=timesteps.device, dtype=torch.float32)
        sigma = schedule.sigma_t[timesteps].to(
            device=timesteps.device, dtype=torch.float32
        )
        signal = (1.0 - m)[:, None] * self.band_scales.to(timesteps.device)[None]
        log_snr = torch.log(
            signal.square().clamp_min(1e-8)
            / sigma[:, None].square().clamp_min(1e-8)
        ).clamp(-20.0, 20.0)
        return torch.sigmoid(
            (log_snr - self.snr_center) / self.snr_temperature
        )

    @staticmethod
    def _normalized_local_energy(details: HaarDetails) -> torch.Tensor:
        energy = F.avg_pool2d(
            _stack_details(details).abs(), kernel_size=3, stride=1, padding=1
        )
        scale = energy.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return torch.tanh(energy / scale)

    def cross_modal_reliability(
        self,
        residual_details: HaarDetails,
        ct_details: HaarDetails,
    ) -> torch.Tensor:
        """Compare normalized local energy, never signed cross-modal values."""
        residual_energy = self._normalized_local_energy(residual_details)
        if not self.use_ct_reliability:
            return torch.ones_like(residual_energy)
        ct_energy = self._normalized_local_energy(ct_details)
        distance = (residual_energy - ct_energy).abs()
        return torch.exp(-distance / self.cross_temperature).clamp(0.0, 1.0)

    def directional_reliability(
        self,
        orientation_energy: Optional[torch.Tensor],
        size: tuple[int, int],
    ) -> torch.Tensor:
        """Map non-negative Gabor orientation energy to LH/HL/HH reliability."""
        if orientation_energy is None:
            raise ValueError("orientation_energy is required for this direct call")
        if orientation_energy.ndim != 4:
            raise ValueError("Gabor orientation energy must have shape [B,O,H,W]")
        if orientation_energy.shape[1] != self.gabor_orientations:
            raise ValueError(
                "Gabor orientation channel count does not match gabor_orientations"
            )
        resized = F.interpolate(
            orientation_energy.abs(), size=size, mode="bilinear", align_corners=False
        )
        angles = torch.arange(
            self.gabor_orientations, device=resized.device, dtype=resized.dtype
        ) * (math.pi / self.gabor_orientations)
        weights = torch.stack(
            (
                angles.cos().square(),
                angles.sin().square(),
                torch.sin(2.0 * angles).square(),
            ),
            dim=0,
        )
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        directional = torch.einsum("bohw,ko->bkhw", resized, weights)
        maximum = directional.amax(dim=1, keepdim=True)
        normalized = directional / maximum.clamp_min(1e-6)
        reliability = 0.5 + 0.5 * normalized
        no_energy = maximum <= 1e-6
        return torch.where(no_energy.expand_as(reliability), torch.ones_like(reliability), reliability)

    def _content_reliability(
        self,
        level_index: int,
        residual_details: HaarDetails,
        ct_details: HaarDetails,
        noise_gate: torch.Tensor,
    ) -> torch.Tensor:
        residual_stats = _stack_details(residual_details).abs().mean(dim=(-2, -1))
        ct_stats = _stack_details(ct_details).abs().mean(dim=(-2, -1))
        statistics = torch.cat((residual_stats, ct_stats, noise_gate[:, None]), dim=1)
        gates = self.content_gates[level_index](statistics)
        if not self.use_subband_gates:
            gates = gates.mean(dim=1, keepdim=True).expand(-1, 3)
        return gates

    def _level_gates(
        self,
        level_index: int,
        residual_details: HaarDetails,
        ct_details: HaarDetails,
        noise_gate: torch.Tensor,
        gabor_orientation: Optional[torch.Tensor],
    ) -> torch.Tensor:
        cross = self.cross_modal_reliability(residual_details, ct_details)
        content = self._content_reliability(
            level_index, residual_details, ct_details, noise_gate
        )[:, :, None, None]
        if self.use_directional_reliability and gabor_orientation is not None:
            direction = self.directional_reliability(
                gabor_orientation, residual_details[0].shape[-2:]
            )
        else:
            direction = torch.ones_like(cross)

        if not self.use_subband_gates:
            cross = cross.mean(dim=1, keepdim=True).expand_as(cross)
            direction = direction.mean(dim=1, keepdim=True).expand_as(direction)

        gate = self.gate_max * noise_gate[:, None, None, None]
        gate = gate * cross * content * direction
        return gate.clamp(0.0, self.gate_max)

    def forward(
        self,
        noisy_state: torch.Tensor,
        timesteps: torch.Tensor,
        schedule,
        ct: torch.Tensor,
        gabor_orientation: Optional[torch.Tensor] = None,
    ) -> tuple[list[torch.Tensor], Dict[str, torch.Tensor]]:
        if noisy_state.shape != ct.shape:
            raise ValueError("noisy state and CT must have identical [B,1,H,W] shapes")
        if noisy_state.shape[-2] % 8 or noisy_state.shape[-1] % 8:
            raise ValueError("spatial dimensions must be divisible by eight")

        _, residual_details1, residual_details2 = self.decompose(noisy_state)
        _, ct_details1, ct_details2 = self.decompose(ct)
        noise = self.noise_reliability(timesteps, schedule).to(noisy_state)

        # Native order follows [L2, L1].
        gates_l2 = self._level_gates(
            0, residual_details2, ct_details2, noise[:, 0], gabor_orientation
        )
        gates_l1 = self._level_gates(
            1, residual_details1, ct_details1, noise[:, 1], gabor_orientation
        )

        scale_l2 = self.band_scales[0].to(noisy_state)
        scale_l1 = self.band_scales[1].to(noisy_state)
        bounded_l2 = torch.tanh(_stack_details(residual_details2) / scale_l2)
        bounded_l1 = torch.tanh(_stack_details(residual_details1) / scale_l1)
        gated_l2 = bounded_l2 * gates_l2
        gated_l1 = bounded_l1 * gates_l1
        l0_source = self.reconstruct_l0(_split_details(gated_l1))

        batch, _, height, width = noisy_state.shape
        l3 = noisy_state.new_zeros(
            batch,
            self.output_channels[0],
            height // 8,
            width // 8,
        )
        injections = [
            l3,
            self.projection_heads[0](gated_l2),
            self.projection_heads[1](gated_l1),
            self.projection_heads[2](l0_source),
        ]
        return injections, {
            "gates_l2": gates_l2,
            "gates_l1": gates_l1,
            "noise_reliability": noise,
            "gate_tv": 0.5 * (_total_variation(gates_l2) + _total_variation(gates_l1)),
        }
