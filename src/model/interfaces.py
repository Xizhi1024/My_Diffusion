"""Unified interfaces for all pluggable modules.

Every optimisation module must conform to one of these base classes.
Modules that are disabled return NoOp-equivalent outputs so the main
training loop contains zero conditional branches for ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

TensorDict = Dict[str, torch.Tensor]


@dataclass
class ConditionBundle:
    """Unified container for all conditioning signals.

    Three kinds of condition by injection target:
      maps    – spatial feature maps  → ControlNet / Zero-Conv Adapter (skip levels)
      tokens  – 1D semantic tokens    → Cross-Attention (bottleneck)
      scalars – global scalars        → FiLM / AdaGN (all layers)
    """

    maps: TensorDict = field(default_factory=dict)
    tokens: TensorDict = field(default_factory=dict)
    scalars: TensorDict = field(default_factory=dict)
    logs: Dict[str, Any] = field(default_factory=dict)

    def get_map(self, name: str) -> Optional[torch.Tensor]:
        return self.maps.get(name)

    def get_token(self, name: str) -> Optional[torch.Tensor]:
        return self.tokens.get(name)

    def merge(self, other: ConditionBundle) -> ConditionBundle:
        self.maps.update(other.maps)
        self.tokens.update(other.tokens)
        self.scalars.update(other.scalars)
        self.logs.update(other.logs)
        return self

    def copy(self) -> ConditionBundle:
        """Return a shallow copy so callers can mutate maps/tokens safely."""
        return ConditionBundle(
            maps=dict(self.maps),
            tokens=dict(self.tokens),
            scalars=dict(self.scalars),
            logs=dict(self.logs),
        )

    @classmethod
    def empty(cls) -> ConditionBundle:
        return cls()


@dataclass
class LossContext:
    """Everything a LossTerm needs to compute its scalar loss value."""

    model_pred: torch.Tensor
    loss_target: torch.Tensor
    target_pet: torch.Tensor
    pred_x0: Optional[torch.Tensor]
    timesteps: torch.Tensor
    tau: torch.Tensor
    batch: Dict[str, torch.Tensor]
    condition: ConditionBundle
    pred_logvar: Optional[torch.Tensor] = None
    pred_residual: Optional[torch.Tensor] = None
    target_residual: Optional[torch.Tensor] = None
    mean_pet: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class PriorModule(nn.Module):
    """Produces conditioning signals from raw batch data.

    Disabled → returns ConditionBundle with empty tensors + log entry.

    ``partial_bundle`` carries features from priors that ran earlier in the
    pipeline (e.g. Gabor features available to HotspotPrior).  Priors that
    don't need cross-prior features can ignore this argument.
    """

    name: str = "prior"

    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        timesteps: torch.Tensor,
        partial_bundle: Optional[ConditionBundle] = None,
    ) -> ConditionBundle:
        raise NotImplementedError


class ConditionAdapter(nn.Module):
    """Transforms noisy_x + raw_condition + priors into UNet input channel."""

    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled

    def forward(
        self,
        noisy_x: torch.Tensor,
        raw_condition: torch.Tensor,
        condition: ConditionBundle,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class NoiseSchedule(nn.Module):
    """Forward diffusion noising strategy (DDPM, BBDM, scale-adaptive, ...)."""

    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled

    def add_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        condition: ConditionBundle,
    ) -> torch.Tensor:
        raise NotImplementedError


class LossTerm(nn.Module):
    """Single loss contribution with time-varying activation gate."""

    name: str = "loss"
    weight: float = 1.0

    def __init__(self, enabled: bool = True, weight: float = 1.0):
        super().__init__()
        self.enabled = enabled
        self.weight = weight

    def forward(self, ctx: LossContext) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns (scalar_loss, log_dict)."""
        raise NotImplementedError
