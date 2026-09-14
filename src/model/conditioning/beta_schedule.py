"""Time-varying beta schedules for condition injection strength.

Controls HOW MUCH each condition type influences the UNet at each timestep.
Early steps (τ≈1): global structure guidance dominates.
Late steps  (τ≈0): fine details (Gabor, hotspot) dominate.
"""

import math
from typing import Dict, Callable

import torch


def local_beta(tau: torch.Tensor, gamma: float = 1.5) -> torch.Tensor:
    """(1 - τ)^γ – rises toward late steps, good for Gabor/fine details."""
    return (1.0 - tau).clamp(0.0, 1.0).pow(gamma)


def global_beta(tau: torch.Tensor, floor: float = 0.3, slope: float = 0.7) -> torch.Tensor:
    """floor + slope·τ – strong early, weaker late. For global semantics."""
    return (floor + slope * tau).clamp(0.0, 1.0)


def spatial_beta(tau: torch.Tensor, floor: float = 0.6, slope: float = 0.4) -> torch.Tensor:
    """Nearly constant, slight rise toward late steps. For organ/CT spatial."""
    return (floor + slope * (1.0 - tau)).clamp(0.0, 1.0)


def semantic_beta(tau: torch.Tensor) -> torch.Tensor:
    """sin(π·τ) – peaks at τ=0.5. Semantic context most useful mid-denoising."""
    return torch.sin(math.pi * tau).clamp(0.0, 1.0)


# Pre-built schedules that accept (tau_tensor) -> weight_tensor
beta_schedules: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "local": local_beta,
    "global": global_beta,
    "spatial": spatial_beta,
    "semantic": semantic_beta,
    "identity": lambda tau: torch.ones_like(tau),
}


def multi_level_betas(
    tau: torch.Tensor,
    levels: int = 5,
) -> torch.Tensor:
    """Return [B, levels] beta matrix for spatial-temporal coupling (vectorised).

    τ=1.0  τ=0.6  τ=0.25  τ=0
    Bottleneck (L4):  1.0    0.5    0.1     0
    L3:               0.6    1.0    0.8     0.3
    L2:               0.2    0.8    1.0     0.5
    L1:               0      0.3    0.8     1.0
    L0:               0      0      0.6     1.0
    """
    B = tau.shape[0]
    t_anchors = torch.tensor([1.0, 0.6, 0.25, 0.0], device=tau.device, dtype=tau.dtype)
    anchor_vals = torch.tensor([
        [1.0, 0.5, 0.1, 0.0],
        [0.6, 1.0, 0.8, 0.3],
        [0.2, 0.8, 1.0, 0.5],
        [0.0, 0.3, 0.8, 1.0],
        [0.0, 0.0, 0.6, 1.0],
    ], device=tau.device, dtype=tau.dtype)[:levels]

    # For each tau, find the bracketing anchor indices [B]
    tau_clamped = tau.clamp(t_anchors[-1], t_anchors[0])
    # right anchor index: smallest anchor >= tau
    right_idx = torch.searchsorted(-t_anchors, -tau_clamped, right=False)
    left_idx = (right_idx - 1).clamp_min(0)

    t_left = t_anchors[left_idx]   # [B]
    t_right = t_anchors[right_idx]  # [B]
    denom = (t_left - t_right).clamp_min(1e-8)
    frac = ((tau_clamped - t_right) / denom).unsqueeze(1)  # [B, 1]

    val_left = anchor_vals[:, left_idx].T    # [B, levels]
    val_right = anchor_vals[:, right_idx].T  # [B, levels]
    betas = frac * val_left + (1 - frac) * val_right  # [B, levels]

    return betas
