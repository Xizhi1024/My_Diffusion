"""NoOp prior – returns empty ConditionBundle when module is disabled."""

from typing import Dict

import torch

from ..interfaces import ConditionBundle, PriorModule


class NoOpPrior(PriorModule):
    name = "noop"

    def __init__(self, **kwargs):
        super().__init__(enabled=False)

    def forward(self, batch: Dict[str, torch.Tensor], timesteps: torch.Tensor, partial_bundle=None) -> ConditionBundle:
        return ConditionBundle(logs={f"{self.name}/enabled": False})
