"""Conservative spectral routing components for Haar detail packets."""

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .boundary_reliable import (
    BoundaryReliableFrequencyInjector,
    _split_details,
    _stack_details,
    _total_variation,
    _ZeroProjection,
)
from .dct_descriptor import SelectedDCTDescriptor


class BoundedAmplitudeHead(nn.Module):
    """Predict a trust amplitude within fixed limits and start at zero."""

    def __init__(
        self,
        input_features: int,
        hidden_channels: int,
        minimum: float = -0.05,
        maximum: float = 0.10,
    ) -> None:
        super().__init__()
        if not minimum < 0 < maximum:
            raise ValueError("minimum and maximum must satisfy minimum < 0 < maximum")

        self.minimum = minimum
        self.maximum = maximum
        self.features = nn.Sequential(
            nn.Linear(input_features, hidden_channels),
            nn.SiLU(),
        )
        self.final = nn.Linear(hidden_channels, 1)

        neutral = (0.0 - minimum) / (maximum - minimum)
        neutral_logit = math.log(neutral / (1.0 - neutral))
        with torch.no_grad():
            self.final.weight.zero_()
            self.final.bias.fill_(neutral_logit)

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        normalized = torch.sigmoid(self.final(self.features(evidence))).squeeze(-1)
        amplitude = self.minimum + (self.maximum - self.minimum) * normalized
        return amplitude.clamp(min=self.minimum, max=self.maximum)


class ConservativeRouteHead(nn.Module):
    """Route one packet to native, adjacent-shallow, or null."""

    def __init__(
        self,
        input_features: int,
        hidden_channels: int,
        initial_null_probability: float = 0.90,
    ) -> None:
        super().__init__()
        if not 0.5 < initial_null_probability < 1.0:
            raise ValueError("initial_null_probability must be between 0.5 and 1")

        self.features = nn.Sequential(
            nn.Linear(input_features, hidden_channels),
            nn.SiLU(),
        )
        self.final = nn.Linear(hidden_channels, 3)

        residual_probability = (1.0 - initial_null_probability) / 2.0
        prior = torch.tensor(
            [
                residual_probability,
                residual_probability,
                initial_null_probability,
            ],
            dtype=self.final.bias.dtype,
            device=self.final.bias.device,
        )
        with torch.no_grad():
            self.final.weight.zero_()
            self.final.bias.copy_(prior.log())

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.final(self.features(evidence)), dim=-1)
