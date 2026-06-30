"""Zero-Conv Adapter – lightweight alternative to full ControlNet.

Instead of copying the entire UNet encoder (~3M params), each skip level
gets a single 1×1 zero-initialised convolution (~5K params total).
Condition is injected as:

    h_l = h_l + β_l(t) · ZeroConv_l(cond_l)

where cond_l concatenates CT features, organ prior, Gabor features,
and hotspot prior at level l.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..interfaces import ConditionBundle, ConditionAdapter
from .beta_schedule import multi_level_betas, global_beta


def _zero_module(module: nn.Module) -> nn.Module:
    """Zero-initialise all parameters so training starts from identity."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ZeroConvAdapter(ConditionAdapter):
    """Lightweight injector with per-level zero-conv + time-varying beta."""

    name = "zero_adapter"

    def __init__(
        self,
        ct_channels: List[int] = (64, 128, 256, 256),
        organ_channels: List[int] = (16, 32, 64),
        gabor_channels: int = 32,
        hotspot_channels: int = 1,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        # Total condition channels per level:
        #   L0 (192²): ct[0]=64 + gabor=32 + hotspot=1 = 97
        #   L1 (96²):  ct[1]=128 + organ[0]=16 = 144
        #   L2 (48²):  ct[2]=256 + organ[1]=32 = 288
        #   L3 (24²):  ct[3]=256 + organ[2]=64 = 320
        self.ct_channels = ct_channels
        self.organ_channels = organ_channels
        self.gabor_channels = gabor_channels
        self.hotspot_channels = hotspot_channels

        # Per-level 1×1 zero-convs
        self.zero_convs = nn.ModuleList([
            _zero_module(nn.Conv2d(ct_channels[0] + gabor_channels + hotspot_channels, ct_channels[0], 1)),
            _zero_module(nn.Conv2d(ct_channels[1] + organ_channels[0], ct_channels[1], 1)),
            _zero_module(nn.Conv2d(ct_channels[2] + organ_channels[1], ct_channels[2], 1)),
            _zero_module(nn.Conv2d(ct_channels[3] + organ_channels[2], ct_channels[3], 1)),
        ])

    def _build_cond_per_level(
        self,
        ct_feats: List[torch.Tensor],
        organ_feats: List[Optional[torch.Tensor]],
        gabor_feat: Optional[torch.Tensor],
        hotspot_prior: Optional[torch.Tensor],
        hw_list: List[int],
    ) -> List[torch.Tensor]:
        """Assemble condition tensor for each UNet resolution level."""

        def _resize(t: torch.Tensor, hw: int) -> torch.Tensor:
            if t.shape[2] != hw:
                return nn.functional.interpolate(t, size=hw, mode="bilinear", align_corners=False)
            return t

        conds = []

        # L0: CT0 + Gabor + Hotspot
        c0_parts = [ct_feats[0]]
        if gabor_feat is not None:
            c0_parts.append(_resize(gabor_feat, hw_list[0]))
        else:
            c0_parts.append(torch.zeros(
                ct_feats[0].shape[0], self.gabor_channels, hw_list[0], hw_list[0],
                device=ct_feats[0].device, dtype=ct_feats[0].dtype
            ))
        if hotspot_prior is not None:
            c0_parts.append(_resize(hotspot_prior, hw_list[0]))
        else:
            c0_parts.append(torch.zeros(
                ct_feats[0].shape[0], self.hotspot_channels, hw_list[0], hw_list[0],
                device=ct_feats[0].device, dtype=ct_feats[0].dtype
            ))
        conds.append(torch.cat(c0_parts, dim=1))

        # L1-L3: CT + Organ
        for i in range(1, 4):
            c_parts = [ct_feats[i]]
            organ_feat = organ_feats[i - 1] if organ_feats is not None and len(organ_feats) >= i else None
            if organ_feat is not None:
                c_parts.append(_resize(organ_feat, hw_list[i]))
            else:
                c_parts.append(torch.zeros(
                    ct_feats[i].shape[0], self.organ_channels[i - 1],
                    hw_list[i], hw_list[i],
                    device=ct_feats[i].device, dtype=ct_feats[i].dtype
                ))
            conds.append(torch.cat(c_parts, dim=1))

        return conds

    def forward(
        self,
        noisy_x: torch.Tensor,
        raw_condition: torch.Tensor,
        condition: ConditionBundle,
        timesteps: torch.Tensor,
        ct_feats: Optional[List[torch.Tensor]] = None,
        hw_list: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Returns model input tensor [B, 1, H, W] (noisy_x concat ct)."""
        if not self.enabled:
            return torch.cat([noisy_x, raw_condition], dim=1)

        # Minimal default: just concat CT
        return torch.cat([noisy_x, raw_condition], dim=1)

    def get_zero_conv_outputs(
        self,
        ct_feats: List[torch.Tensor],
        organ_feats: List[Optional[torch.Tensor]],
        gabor_feat: Optional[torch.Tensor],
        hotspot_prior: Optional[torch.Tensor],
        hw_list: List[int],
        timesteps: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Return per-level zero-conv outputs for skip-connection injection.

        Applies time-varying beta modulation when tau is provided:
          - Late steps (τ→0): Gabor/local beta → 1 (fine details)
          - Early steps (τ→1): global/spatial beta → 1 (structure)
        """
        if not self.enabled:
            return [torch.zeros_like(f) for f in ct_feats]

        conds = self._build_cond_per_level(ct_feats, organ_feats, gabor_feat, hotspot_prior, hw_list)
        outputs = [zc(c) for zc, c in zip(self.zero_convs, conds)]

        # Apply per-level time-varying beta from the full 5-level matrix.
        # Reversed: adapter outputs [L3→L0], decoder consumes [L0→L3].
        if tau is not None:
            betas = multi_level_betas(tau, levels=5)  # [B, 5]: L4(bn)→L0
            # betas columns: [bottleneck, L3, L2, L1, L0]
            # adapter outputs 4 levels [L3, L2, L1, L0] (reversed to [L0, L1, L2, L3])
            # Match: betas[:, 1:] = [L3, L2, L1, L0] → reverse → [L0, L1, L2, L3]
            level_betas = betas[:, 1:].flip(1)  # [B, 4], L0→L3
            for i in range(len(outputs)):
                b = level_betas[:, i]  # [B]
                while b.dim() < outputs[i].dim():
                    b = b.unsqueeze(-1)
                outputs[i] = outputs[i] * b

        return outputs


class RawConcatAdapter(ConditionAdapter):
    """Fallback: just concat noisy_x and raw_condition along channel dim."""

    name = "raw_concat"

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)

    def forward(
        self,
        noisy_x: torch.Tensor,
        raw_condition: torch.Tensor,
        condition: ConditionBundle,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([noisy_x, raw_condition], dim=1)

    def get_zero_conv_outputs(
        self,
        ct_feats=None, organ_feats=None, gabor_feat=None,
        hotspot_prior=None, hw_list=None, timesteps=None, tau=None,
    ) -> list:
        return []
