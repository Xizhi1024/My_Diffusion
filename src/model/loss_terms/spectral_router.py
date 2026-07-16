"""Non-image regularization for spectral evidence routing."""

from __future__ import annotations

from typing import Dict

import torch

from ..interfaces import LossContext, LossTerm


class SpectralRouterRegularizationLoss(LossTerm):
    name = "spectral_router_regularization"

    def __init__(
        self,
        enabled: bool = True,
        weight: float = 1.0,
        temporal_weight: float = 1e-4,
        dct_weight: float = 1e-4,
        gabor_weight: float = 1e-4,
    ) -> None:
        super().__init__(enabled=enabled, weight=weight)
        self.temporal_weight = float(temporal_weight)
        self.dct_weight = float(dct_weight)
        self.gabor_weight = float(gabor_weight)

    def forward(self, ctx: LossContext) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = ctx.target_pet.new_zeros(())
        if not self.enabled:
            return zero, {
                f"{self.name}/enabled": zero,
                f"{self.name}/loss": zero,
            }

        temporal = ctx.condition.scalars.get(
            "spectral_route_temporal_smoothness", zero
        ).mean()
        dct = ctx.condition.scalars.get("spectral_dct_weight_offset", zero).mean()
        gabor = ctx.condition.scalars.get(
            "spectral_gabor_parameter_offset", zero
        ).mean()
        raw = (
            self.temporal_weight * temporal
            + self.dct_weight * dct
            + self.gabor_weight * gabor
        )
        return self.weight * raw, {
            f"{self.name}/enabled": torch.ones_like(zero),
            f"{self.name}/temporal": temporal.detach(),
            f"{self.name}/dct": dct.detach(),
            f"{self.name}/gabor": gabor.detach(),
            f"{self.name}/loss": raw.detach(),
        }
