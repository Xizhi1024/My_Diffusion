"""Phase-stable complex Gabor descriptor for local directional statistics.

The bank is a Cartesian ``scales x orientations`` grid. Cosine/sine
quadrature responses are combined into amplitude before they are exposed to
the rest of the model, preventing the raw carrier phase from creating stripe
textures in decoder features.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces import ConditionBundle, PriorModule


class GaborPrior(PriorModule):
    name = "gabor"

    def __init__(
        self,
        filters: Optional[int] = None,
        kernel_size: int = 15,
        enabled: bool = True,
        scales: int = 4,
        orientations: int = 8,
        min_frequency: float = 0.06,
        max_frequency: float = 0.28,
        parameter_delta: float = 0.25,
    ):
        super().__init__(enabled=enabled)
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        if filters is not None and filters != scales * orientations:
            if filters % orientations == 0:
                scales = filters // orientations
            else:
                scales, orientations = 1, filters
        if scales < 1 or orientations < 1:
            raise ValueError("scales and orientations must be positive")

        self.scales = int(scales)
        self.orientations = int(orientations)
        self.filters = self.scales * self.orientations
        self.kernel_size = int(kernel_size)
        self.parameter_delta = float(parameter_delta)

        frequencies = torch.logspace(
            math.log10(min_frequency), math.log10(max_frequency), self.scales
        )
        orientation_angles = torch.arange(self.orientations, dtype=torch.float32)
        orientation_angles = orientation_angles * (math.pi / self.orientations)
        base_frequency = frequencies[:, None].expand(-1, self.orientations).reshape(-1)
        base_theta = orientation_angles[None, :].expand(self.scales, -1).reshape(-1)
        base_sigma = torch.full((self.filters,), kernel_size / 5.0)

        # Non-persistent: legacy checkpoints did not contain these grid buffers.
        self.register_buffer("base_frequency", base_frequency, persistent=False)
        self.register_buffer("base_theta", base_theta, persistent=False)
        self.register_buffer("base_sigma", base_sigma, persistent=False)
        # Historical attribute names remain trainable for checkpoint/API
        # compatibility, but now represent bounded deviations from the grid.
        self.log_frequency = nn.Parameter(torch.zeros(self.filters))
        self.theta_raw = nn.Parameter(torch.zeros(self.filters))
        self.log_sigma = nn.Parameter(torch.zeros(self.filters))
        self.gamma_raw = nn.Parameter(torch.zeros(self.filters))
        # Kept only as a frozen state-dict compatibility key. Quadrature
        # amplitude makes an explicit carrier phase unnecessary.
        self.phase = nn.Parameter(torch.zeros(self.filters), requires_grad=False)

        coords = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("grid_x", xx, persistent=False)
        self.register_buffer("grid_y", yy, persistent=False)

    def _bounded_parameters(self, detach: bool = False) -> Dict[str, torch.Tensor]:
        raw_frequency = self.log_frequency.detach() if detach else self.log_frequency
        raw_theta = self.theta_raw.detach() if detach else self.theta_raw
        raw_sigma = self.log_sigma.detach() if detach else self.log_sigma
        raw_gamma = self.gamma_raw.detach() if detach else self.gamma_raw

        log_ratio = math.log1p(self.parameter_delta)
        frequency = self.base_frequency * torch.exp(log_ratio * torch.tanh(raw_frequency))
        # Keep every learned orientation within one quarter of its grid cell,
        # so neighbouring filters cannot cross or collapse onto each other.
        theta = self.base_theta + (math.pi / (4 * self.orientations)) * torch.tanh(raw_theta)
        sigma = self.base_sigma * torch.exp(log_ratio * torch.tanh(raw_sigma))
        sigma = sigma.clamp(1.0, self.kernel_size / 2)
        gamma = 1.0 + 0.75 * torch.tanh(raw_gamma)
        return {
            "frequency": frequency,
            "theta": theta,
            "sigma": sigma,
            "gamma": gamma,
        }

    def parameter_offset_energy(self) -> torch.Tensor:
        trainable = (
            self.log_frequency,
            self.theta_raw,
            self.log_sigma,
            self.gamma_raw,
        )
        return torch.stack(
            [parameter.square().mean() for parameter in trainable]
        ).mean()

    def _build_quadrature_kernels(
        self,
        dtype: torch.dtype,
        device: torch.device,
        detach_parameters: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return L2-normalized cosine/sine kernels shaped ``[F,1,K,K]``."""
        x = self.grid_x.to(device=device, dtype=dtype).unsqueeze(0)
        y = self.grid_y.to(device=device, dtype=dtype).unsqueeze(0)
        params = self._bounded_parameters(detach=detach_parameters)
        frequency = params["frequency"].to(device=device, dtype=dtype).view(-1, 1, 1)
        theta = params["theta"].to(device=device, dtype=dtype).view(-1, 1, 1)
        sigma = params["sigma"].to(device=device, dtype=dtype).view(-1, 1, 1)
        gamma = params["gamma"].to(device=device, dtype=dtype).view(-1, 1, 1)

        x_theta = x * theta.cos() + y * theta.sin()
        y_theta = -x * theta.sin() + y * theta.cos()
        envelope = torch.exp(
            -(x_theta.square() + gamma.square() * y_theta.square()) / (2 * sigma.square())
        )
        phase = 2 * math.pi * frequency * x_theta
        real = envelope * phase.cos()
        imag = envelope * phase.sin()

        def normalize(kernels: torch.Tensor) -> torch.Tensor:
            kernels = kernels - kernels.mean(dim=(1, 2), keepdim=True)
            norm = kernels.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-6)
            return (kernels / norm).unsqueeze(1)

        return normalize(real), normalize(imag)

    def _build_kernels(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Backward-compatible real-kernel accessor used by existing tests/tools."""
        real, _ = self._build_quadrature_kernels(dtype, device)
        return real

    def describe(
        self,
        image: torch.Tensor,
        detach_parameters: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Describe an image using normalized amplitude and orientation statistics."""
        real, imag = self._build_quadrature_kernels(
            image.dtype, image.device, detach_parameters=detach_parameters
        )
        padding = self.kernel_size // 2
        response_real = F.conv2d(image, real, padding=padding)
        response_imag = F.conv2d(image, imag, padding=padding)
        amplitude = torch.sqrt(response_real.square() + response_imag.square() + 1e-8)
        rms = amplitude.square().mean(dim=(2, 3), keepdim=True).sqrt().clamp_min(1e-6)
        amplitude = amplitude / rms

        b, _, h, w = amplitude.shape
        by_scale = amplitude.view(b, self.scales, self.orientations, h, w)
        orientation = by_scale.mean(dim=1)
        energy = orientation.mean(dim=1, keepdim=True)

        angles = torch.arange(self.orientations, device=image.device, dtype=image.dtype)
        angles = angles * (math.pi / self.orientations)
        cos2 = torch.cos(2 * angles).view(1, -1, 1, 1)
        sin2 = torch.sin(2 * angles).view(1, -1, 1, 1)
        orientation_sum = orientation.sum(dim=1, keepdim=True).clamp_min(1e-6)
        vector_x = (orientation * cos2).sum(dim=1, keepdim=True)
        vector_y = (orientation * sin2).sum(dim=1, keepdim=True)
        anisotropy = torch.sqrt(vector_x.square() + vector_y.square() + 1e-8) / orientation_sum
        anisotropy = anisotropy.clamp(0.0, 1.0)

        return {
            "gabor_feat": amplitude,
            "gabor_orientation": orientation,
            "gabor_energy": energy,
            "gabor_anisotropy": anisotropy,
        }

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        timesteps: torch.Tensor,
        partial_bundle=None,
    ) -> ConditionBundle:
        if not self.enabled:
            return ConditionBundle(logs={f"{self.name}/enabled": False})
        return ConditionBundle(
            maps=self.describe(batch["ct"]),
            logs={f"{self.name}/enabled": True},
        )
