"""Organ prior encoder -- fixed anatomical mask conditioning.

Input:
  m_organs       [B, 6, H, W]   organ one-hot masks (TotalSegmentator, frozen)
  organ_distance [B, 6, H, W]   signed distance transform per organ (optional)
  mu_map         [B, 1, H, W]   CT->511keV attenuation map (physical, frozen)

Output spatial feature maps at 3 scales for Zero-Conv Adapter injection.
The organ prior reduces false positives in bladder/rectum/bone regions.
When organ_distance is absent in the batch, a zero tensor is substituted.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..interfaces import ConditionBundle, PriorModule


class OrganPrior(PriorModule):
    name = "organ_prior"

    def __init__(
        self,
        organ_channels: int = 6,
        distance_channels: int = 6,
        mu_map_channels: int = 1,
        out_channels: tuple = (16, 32, 64),
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        in_ch = organ_channels + distance_channels + mu_map_channels  # 13
        c1, c2, c3 = out_channels

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, padding=1),
            nn.SiLU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, stride=2, padding=1),
            nn.SiLU(),
        )

    def _build_input(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        organ = batch.get("organ_mask")
        dist = batch.get("organ_distance")
        mu = batch.get("mu_map")
        B, _, H, W = batch["ct"].shape
        device = batch["ct"].device
        dtype = batch["ct"].dtype

        if organ is None:
            organ = torch.zeros(B, 6, H, W, device=device, dtype=dtype)
        if dist is None:
            dist = torch.zeros(B, 6, H, W, device=device, dtype=dtype)
        if mu is None:
            mu = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
        return torch.cat([organ, dist, mu], dim=1)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        timesteps: torch.Tensor,
        partial_bundle: Optional[ConditionBundle] = None,
    ) -> ConditionBundle:
        if not self.enabled:
            return ConditionBundle(logs={f"{self.name}/enabled": False})

        x = self._build_input(batch)
        p1 = self.stage1(x)       # [B, 16, H, W]
        p2 = self.stage2(p1)      # [B, 32, H/2, W/2]
        p3 = self.stage3(p2)      # [B, 64, H/4, W/4]

        return ConditionBundle(
            maps={"organ_feat_1": p1, "organ_feat_2": p2, "organ_feat_3": p3},
            logs={f"{self.name}/enabled": True},
        )
