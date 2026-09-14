"""Tiny Hotspot Prior Net -- lightweight CT->lesion-candidate network.

Input (per docs v2.0):
  x_ct          [B, 1, H, W]   CT image
  f_gabor       [B, 1, H, W]   Gabor high-frequency energy (from GaborPrior)
  (organ features optionally fused via partial_bundle)

Supervised on: lesion mask (hard), PET high-uptake threshold (soft),
and distance transform (continuous).  Outputs a soft attention map
that guides the diffusion process toward lesion regions.

Parameter budget: < 0.5M
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces import ConditionBundle, PriorModule


class _TinyUNet(nn.Module):
    """Minimal U-Net for hotspot prediction (< 0.5M params)."""

    def __init__(self, in_ch: int = 1, base: int = 16):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.SiLU(),
        )
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.SiLU(),
        )
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base * 4, base * 4, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base * 4, base * 4, 3, padding=1), nn.SiLU(),
        )
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base * 4 + base * 2, base * 2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.SiLU(),
        )
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base * 2 + base, base, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.SiLU(),
        )
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.silu(self.down1(e1)))
        b = self.bottleneck(F.silu(self.down2(e2)))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))


class HotspotPrior(PriorModule):
    name = "hotspot_prior"

    def __init__(
        self,
        base_channels: int = 16,
        params_max: int = 500_000,
        use_gabor_energy: bool = True,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.use_gabor_energy = use_gabor_energy
        # 2 channels when Gabor energy is available, else 1 (CT only)
        in_ch = 2 if use_gabor_energy else 1
        self.net = _TinyUNet(in_ch=in_ch, base=base_channels)
        total = sum(p.numel() for p in self.net.parameters())
        if total > params_max:
            print(f"[HotspotPrior] WARNING: {total} params exceeds budget {params_max}")

    def _build_input(
        self,
        batch: Dict[str, torch.Tensor],
        partial_bundle: Optional[ConditionBundle] = None,
    ) -> torch.Tensor:
        ct = batch["ct"]  # [B, 1, H, W]
        inputs = [ct]

        if self.use_gabor_energy:
            gabor_energy = None
            if partial_bundle is not None:
                gabor_energy = partial_bundle.get_map("gabor_energy")
            if gabor_energy is not None:
                if gabor_energy.shape[2:] != ct.shape[2:]:
                    gabor_energy = F.interpolate(
                        gabor_energy, size=ct.shape[2:], mode="bilinear", align_corners=False
                    )
                inputs.append(gabor_energy)
            else:
                # Zero-channel when Gabor not yet computed (still 2-ch input)
                inputs.append(torch.zeros_like(ct))

        return torch.cat(inputs, dim=1)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        timesteps: torch.Tensor,
        partial_bundle: Optional[ConditionBundle] = None,
    ) -> ConditionBundle:
        if not self.enabled:
            return ConditionBundle(logs={f"{self.name}/enabled": False})

        x = self._build_input(batch, partial_bundle)
        hotspot_map = self.net(x)  # [B, 1, H, W]

        return ConditionBundle(
            maps={"hotspot_prior": hotspot_map},
            logs={f"{self.name}/enabled": True},
        )
