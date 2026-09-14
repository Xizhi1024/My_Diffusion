"""Base loss utilities – NoOp (disabled) loss and tau gating."""

from typing import Dict

import torch
import torch.nn as nn

from ..interfaces import LossContext, LossTerm


class DisabledLossTerm(LossTerm):
    """NoOp loss - returns stable zero logs when a loss is disabled."""

    name = "disabled"

    def __init__(self, name: str = "disabled", weight: float = 1.0):
        super().__init__(enabled=False, weight=weight)
        self.name = name

    def forward(self, ctx: LossContext) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = torch.tensor(0.0, device=ctx.target_pet.device)
        return (
            zero,
            {
                f"{self.name}/enabled": zero,
                f"{self.name}/loss": zero,
            },
        )


def tau_gate(
    tau: torch.Tensor,
    max_tau: float = 1.0,
    min_tau: float = 0.0,
) -> torch.Tensor:
    """Return per-element gate ∈ [0, 1] based on τ position.

    Gate = 1 when min_tau ≤ τ ≤ max_tau, 0 otherwise (hard threshold).
    For smooth gating, use sigmoid-based approach (see specific loss terms).
    """
    gate = torch.ones_like(tau, dtype=torch.float32)
    gate = gate * (tau <= max_tau).float()
    gate = gate * (tau >= min_tau).float()
    return gate


def smooth_tau_gate(
    tau: torch.Tensor,
    max_tau: float = 0.25,
    sharpness: float = 30.0,
) -> torch.Tensor:
    """Gate = 1 when τ < max_tau (late denoising steps), 0 otherwise.

    τ = t/T ∈ [0, 1].  τ close to 0 = late step (fine details).
    τ close to 1 = early step (coarse structure).

    Gate rises as τ decreases below max_tau.
    """
    return torch.sigmoid(sharpness * (max_tau - tau))
