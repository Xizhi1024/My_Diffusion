"""Noise schedule base implementations – DDPM and BBDM bridge."""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..interfaces import ConditionBundle, NoiseSchedule


# ---------------------------------------------------------------------------
# Helper: sigmoid / linear beta schedules
# ---------------------------------------------------------------------------

def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule as in improved DDPM."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.02)


def _linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


# ---------------------------------------------------------------------------
# DDPM
# ---------------------------------------------------------------------------

class DDPMNoiseSchedule(NoiseSchedule):
    """Standard DDPM forward process: q(x_t | x_0) = N(sqrt(ᾱ_t)·x_0, (1-ᾱ_t)·I)."""

    name = "ddpm"

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.num_train_timesteps = num_train_timesteps

        if beta_schedule == "cosine":
            betas = _cosine_beta_schedule(num_train_timesteps)
        else:
            betas = _linear_beta_schedule(num_train_timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)

        # For x0-pred: sqrt(ᾱ_t) and sqrt(1-ᾱ_t)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt(), persistent=False)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt(), persistent=False)

    def add_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        condition: ConditionBundle,
    ) -> torch.Tensor:
        """q(x_t | x_0) = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε"""
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].to(x0.device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].to(x0.device)

        # Reshape for broadcasting: [B] → [B, 1, 1, 1]
        while sqrt_alpha.dim() < x0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

    def get_tau(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Return τ = t/T ∈ [0, 1]."""
        return timesteps.float() / self.num_train_timesteps


# ---------------------------------------------------------------------------
# BBDM Bridge
# ---------------------------------------------------------------------------

class BBDMBridgeSchedule(NoiseSchedule):
    """Brownian Bridge Diffusion: CT → PET bridge.

    Forward: x_t = m_t·CT + (1-m_t)·PET + σ_t·ε

    where:
      m_t = t/T  (linear bridge: 0→1 as t goes 0→T)
      At t=0: x_0 = PET (pure target, no noise)
      At t=T: x_T ≈ CT + noise (source, max noise)
      Reverse (sampling): CT+noise → PET

    CT is an *endpoint constraint*, not just a condition.
    """

    name = "bbdm_bridge"

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        m_schedule: str = "linear",
        sigma_scale: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.num_train_timesteps = num_train_timesteps
        self.m_schedule = m_schedule
        self.sigma_scale = sigma_scale

        # Pre-compute m_t and σ_t for each timestep
        t = torch.arange(num_train_timesteps)
        if m_schedule == "linear":
            m_t = t / num_train_timesteps
        elif m_schedule == "cosine":
            m_t = 0.5 * (1 - torch.cos(t / num_train_timesteps * math.pi))
        else:
            raise ValueError(f"Unknown m_schedule: {m_schedule}")

        sigma_t = sigma_scale * torch.sqrt(2 * m_t * (1 - m_t)).clamp_min(1e-8)

        self.register_buffer("m_t", m_t, persistent=False)
        self.register_buffer("sigma_t", sigma_t, persistent=False)

    def add_noise(
        self,
        x0: torch.Tensor,       # PET (target)
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        condition: ConditionBundle,
        x_source: Optional[torch.Tensor] = None,  # CT (source)
    ) -> torch.Tensor:
        """x_t = m_t·CT + (1-m_t)·PET + σ_t·ε

        At t=0 (m=0): x_0 = PET   (pure target)
        At t=T (m≈1): x_T ≈ CT + noise  (source + noise)
        Reverse process denoises from CT toward PET.
        """
        if x_source is None:
            x_source = condition.get_map("ct") if condition else None
        if x_source is None:
            raise ValueError("BBDM requires source image (CT) as x_source argument")

        m = self.m_t[timesteps].to(x0.device)
        sigma = self.sigma_t[timesteps].to(x0.device)

        while m.dim() < x0.dim():
            m = m.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)

        mean = m * x_source + (1 - m) * x0
        return mean + sigma * noise

    def get_tau(self, timesteps: torch.Tensor) -> torch.Tensor:
        return timesteps.float() / self.num_train_timesteps

    def get_m(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Bridge coefficient: 0 = target (PET), 1 = source (CT)."""
        return self.m_t[timesteps]

    def predict_x0_from_xt(
        self,
        x_t: torch.Tensor,
        x_source: torch.Tensor,
        pred_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Recover predicted PET: solve x_t = m·CT + (1-m)·PET + σ·ε for PET."""
        m = self.m_t[timesteps].to(x_t.device)
        sigma = self.sigma_t[timesteps].to(x_t.device)
        while m.dim() < x_t.dim():
            m = m.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        return (x_t - m * x_source - sigma * pred_noise) / (1 - m).clamp_min(1e-8)
