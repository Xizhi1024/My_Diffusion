"""Gabor Bank prior – learnable Gabor filters for frequency-domain texture analysis.

Produces:
  maps.gabor_feat   – [B, filters, H, W]  Gabor response maps
  maps.gabor_energy – [B, 1, H, W]        normalised high-frequency energy

Used by:
  - Zero-Conv Adapter (shallow layers L0-L1) for lesion edge preservation
  - Scale-adaptive noise scheduler (E_gabor ↑ → σ_high ↓)
  - Focal Frequency Loss late-step weighting
"""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces import ConditionBundle, PriorModule


class GaborPrior(PriorModule):
    name = "gabor"

    def __init__(self, filters: int = 32, kernel_size: int = 15, enabled: bool = True):
        super().__init__(enabled=enabled)
        self.filters = filters
        self.kernel_size = kernel_size

        # Learnable Gabor parameters.  Softplus/sigmoid transforms in
        # _build_kernels keep them in physically meaningful ranges.
        self.log_frequency = nn.Parameter(torch.linspace(math.log(0.06), math.log(0.28), filters))
        self.theta_raw = nn.Parameter(torch.linspace(0.0, math.pi, filters))
        self.log_sigma = nn.Parameter(torch.full((filters,), math.log(kernel_size / 5.0)))
        self.gamma_raw = nn.Parameter(torch.zeros(filters))
        self.phase = nn.Parameter(torch.zeros(filters))

        coords = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("grid_x", xx, persistent=False)
        self.register_buffer("grid_y", yy, persistent=False)

    def _build_kernels(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Return normalised Gabor kernels shaped [filters, 1, K, K]."""
        x = self.grid_x.to(device=device, dtype=dtype).unsqueeze(0)
        y = self.grid_y.to(device=device, dtype=dtype).unsqueeze(0)

        freq = self.log_frequency.exp().to(device=device, dtype=dtype).view(-1, 1, 1)
        theta = self.theta_raw.to(device=device, dtype=dtype).view(-1, 1, 1)
        sigma = self.log_sigma.exp().clamp_min(1e-3).to(device=device, dtype=dtype).view(-1, 1, 1)
        gamma = (0.25 + 1.75 * torch.sigmoid(self.gamma_raw)).to(device=device, dtype=dtype).view(-1, 1, 1)
        phase = self.phase.to(device=device, dtype=dtype).view(-1, 1, 1)

        x_theta = x * theta.cos() + y * theta.sin()
        y_theta = -x * theta.sin() + y * theta.cos()
        envelope = torch.exp(-(x_theta.square() + gamma.square() * y_theta.square()) / (2 * sigma.square()))
        carrier = torch.cos(2 * math.pi * freq * x_theta + phase)
        kernels = envelope * carrier

        kernels = kernels - kernels.mean(dim=(1, 2), keepdim=True)
        kernels = kernels / kernels.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-6)
        return kernels.unsqueeze(1)

    def forward(self, batch: Dict[str, torch.Tensor], timesteps: torch.Tensor, partial_bundle=None) -> ConditionBundle:
        if not self.enabled:
            return ConditionBundle(logs={f"{self.name}/enabled": False})

        ct = batch["ct"]
        kernels = self._build_kernels(ct.dtype, ct.device)
        feat = F.conv2d(ct, kernels, padding=self.kernel_size // 2)  # [B, filters, H, W]
        energy = torch.sqrt(feat.square().sum(dim=1, keepdim=True).clamp_min(1e-12))
        # Normalise per-sample so energy map is comparable across CT scans
        energy = energy / (energy.amax(dim=(2, 3), keepdim=True).clamp_min(1e-8))

        return ConditionBundle(
            maps={"gabor_feat": feat, "gabor_energy": energy},
            logs={f"{self.name}/enabled": True},
        )
