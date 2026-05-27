"""2D PatchNCE loss for CT-PET spatial binding."""

from typing import Dict

import torch
import torch.nn.functional as F

from ..interfaces import LossContext, LossTerm
from .base import smooth_tau_gate


class PatchNCELoss(LossTerm):
    name = "patch_nce"

    def __init__(
        self,
        patch_size: int = 3,
        num_patches: int = 256,
        temperature: float = 0.07,
        active_tau_max: float = 0.6,
        enabled: bool = True,
        weight: float = 0.1,
    ):
        super().__init__(enabled=enabled, weight=weight)
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.temperature = temperature
        self.active_tau_max = active_tau_max

    def _patches(self, x: torch.Tensor) -> torch.Tensor:
        patches = F.unfold(x, kernel_size=self.patch_size, padding=self.patch_size // 2)
        return patches.transpose(1, 2)

    def forward(self, ctx: LossContext) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.enabled:
            return (
                torch.tensor(0.0, device=ctx.target_pet.device),
                {f"{self.name}/enabled": torch.tensor(0.0, device=ctx.target_pet.device)},
            )

        gate = smooth_tau_gate(ctx.tau, max_tau=self.active_tau_max)

        ct = ctx.batch["ct"]
        pet = ctx.pred_x0 if ctx.pred_x0 is not None else ctx.model_pred
        ct_patches = F.normalize(self._patches(ct), dim=-1)
        pet_patches = F.normalize(self._patches(pet), dim=-1)

        B, N, _ = ct_patches.shape
        sample_n = min(self.num_patches, N)
        if sample_n < N:
            idx = torch.randperm(N, device=ct.device)[:sample_n]
            ct_patches = ct_patches[:, idx]
            pet_patches = pet_patches[:, idx]
            N = sample_n

        labels = torch.arange(N, device=ct.device)
        losses = []
        for b in range(B):
            logits = ct_patches[b] @ pet_patches[b].transpose(0, 1)
            logits = logits / self.temperature
            losses.append(F.cross_entropy(logits, labels))
        loss = torch.stack(losses).mean() * gate.mean()

        return loss * self.weight, {
            f"{self.name}/loss": loss.detach(),
            f"{self.name}/gate_mean": gate.mean().detach(),
            f"{self.name}/enabled": torch.tensor(1.0, device=ct.device),
        }
