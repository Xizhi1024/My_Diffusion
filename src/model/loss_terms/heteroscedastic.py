"""Heteroscedastic NLL Loss – uncertainty-aware weighting.

Model outputs both mean and log-variance.  NLL loss lets the model learn
which regions are inherently uncertain → automatically down-weights those
regions in the L2 term, improving calibration.
"""

from typing import Dict

import torch
import torch.nn as nn

from ..interfaces import LossContext, LossTerm


class HeteroscedasticNLLLoss(LossTerm):
    name = "heteroscedastic_nll"

    def __init__(
        self,
        logvar_min: float = -6.0,
        logvar_max: float = 2.0,
        enabled: bool = True,
        weight: float = 1.0,
    ):
        super().__init__(enabled=enabled, weight=weight)
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

    def forward(self, ctx: LossContext) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.enabled or ctx.pred_logvar is None:
            return (
                torch.tensor(0.0, device=ctx.target_pet.device),
                {f"{self.name}/enabled": torch.tensor(0.0 if (not self.enabled or ctx.pred_logvar is None) else 1.0)},
            )

        logvar = ctx.pred_logvar.clamp(self.logvar_min, self.logvar_max)
        pred = ctx.pred_x0 if ctx.pred_x0 is not None else ctx.model_pred
        target = ctx.target_pet

        # NLL = 0.5 * (exp(-logvar) * (pred-target)² + logvar)
        precision = torch.exp(-logvar)
        mse = (pred - target) ** 2
        loss = 0.5 * (precision * mse + logvar).mean()

        return loss * self.weight, {
            f"{self.name}/loss": loss.detach(),
            f"{self.name}/logvar_mean": logvar.mean().detach(),
            f"{self.name}/enabled": torch.tensor(1.0),
        }
