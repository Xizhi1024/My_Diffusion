"""Weak smoothness regularizer for new spatial frequency gates."""

from __future__ import annotations

from typing import Dict

import torch

from ..interfaces import LossContext, LossTerm


class FrequencyGateTVLoss(LossTerm):
    name = "frequency_gate_tv"

    def __init__(self, enabled: bool = True, weight: float = 0.001) -> None:
        super().__init__(enabled=enabled, weight=weight)

    def forward(self, ctx: LossContext) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = ctx.target_pet.new_zeros(())
        gate_tv = ctx.condition.scalars.get("frequency_gate_tv")
        if not self.enabled or gate_tv is None:
            return zero, {
                f"{self.name}/enabled": torch.ones_like(zero) if self.enabled else zero,
                f"{self.name}/available": zero,
                f"{self.name}/loss": zero,
            }
        raw = gate_tv.mean()
        return raw * self.weight, {
            f"{self.name}/enabled": torch.ones_like(zero),
            f"{self.name}/available": torch.ones_like(zero),
            f"{self.name}/loss": raw.detach(),
        }
