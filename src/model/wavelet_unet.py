"""Haar-wavelet scale transitions for the BBDM denoiser."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from .bbdm_unet import _CrossAttention, _ResBlock, _TimeEmbedding
from .frequency.haar import haar_dwt2, haar_idwt2


def _group_count(channels: int, maximum: int = 32) -> int:
    """Return the largest useful GroupNorm group count that divides channels."""
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _validate_transition(in_channels: int, out_channels: int, kernel_size: int) -> None:
    if in_channels <= 0 or out_channels <= 0:
        raise ValueError("wavelet transition channels must be positive")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("mix_kernel_size must be a positive odd integer")


class WaveletDownsample(nn.Module):
    """Downsample with Haar DWT followed by learned cross-subband mixing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mix_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        _validate_transition(in_channels, out_channels, mix_kernel_size)
        padding = mix_kernel_size // 2
        self.mix = nn.Sequential(
            nn.Conv2d(4 * in_channels, out_channels, 1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                mix_kernel_size,
                padding=padding,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ll, details = haar_dwt2(x)
        return self.mix(torch.cat((ll, *details), dim=1))


class WaveletUpsample(nn.Module):
    """Predict four subbands and reconstruct the finer scale with Haar IWT."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mix_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        _validate_transition(in_channels, out_channels, mix_kernel_size)
        expanded_channels = 4 * out_channels
        padding = mix_kernel_size // 2
        self.expand = nn.Sequential(
            nn.Conv2d(
                in_channels,
                expanded_channels,
                mix_kernel_size,
                padding=padding,
            ),
            nn.GroupNorm(_group_count(expanded_channels), expanded_channels),
            nn.SiLU(),
            nn.Conv2d(expanded_channels, expanded_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh = self.expand(x).chunk(4, dim=1)
        return haar_idwt2(ll, (lh, hl, hh))


class WaveletBBDMUNet(nn.Module):
    """BBDM U-Net with Haar DWT/IWT at every backbone scale transition."""

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        channel_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        time_dim: int = 256,
        dropout: float = 0.0,
        enable_heteroscedastic: bool = True,
        ca_kv_dim: int = 64,
        ca_num_heads: int = 4,
        meta_dim: int = 0,
        mix_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if len(channel_mult) < 2:
            raise ValueError("channel_mult must contain at least two resolution levels")
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be positive")

        self.enable_heteroscedastic = bool(enable_heteroscedastic)
        self.num_scale_transitions = len(channel_mult) - 1
        self.time_emb = _TimeEmbedding(time_dim)

        self.meta_proj: Optional[nn.Module] = None
        if meta_dim > 0:
            self.meta_proj = nn.Sequential(
                nn.Linear(meta_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )

        channels = [base_channels * int(multiplier) for multiplier in channel_mult]
        if any(channel <= 0 for channel in channels):
            raise ValueError("base_channels and channel_mult must produce positive widths")

        self.head = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index, channel in enumerate(channels):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        _ResBlock(channel, channel, time_dim, dropout)
                        for _ in range(num_res_blocks)
                    ]
                )
            )
            if index < len(channels) - 1:
                self.downsamples.append(
                    WaveletDownsample(
                        channel,
                        channels[index + 1],
                        mix_kernel_size=mix_kernel_size,
                    )
                )

        bottleneck_channels = channels[-1]
        self.bn_block1 = _ResBlock(
            bottleneck_channels, bottleneck_channels, time_dim, dropout
        )
        self.bn_block2 = _ResBlock(
            bottleneck_channels, bottleneck_channels, time_dim, dropout
        )
        self.cross_attn = _CrossAttention(
            bottleneck_channels, ca_kv_dim, ca_num_heads
        )

        reverse_channels = list(reversed(channels))
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for index, channel in enumerate(reverse_channels):
            blocks = nn.ModuleList(
                [_ResBlock(2 * channel, channel, time_dim, dropout)]
            )
            blocks.extend(
                _ResBlock(channel, channel, time_dim, dropout)
                for _ in range(num_res_blocks - 1)
            )
            self.up_blocks.append(blocks)
            if index < len(reverse_channels) - 1:
                self.upsamples.append(
                    WaveletUpsample(
                        channel,
                        reverse_channels[index + 1],
                        mix_kernel_size=mix_kernel_size,
                    )
                )

        output_channels = channels[0]
        self.out_norm = nn.GroupNorm(
            _group_count(output_channels), output_channels
        )
        self.out_channels = 2 if enable_heteroscedastic else 1
        self.out_conv = nn.Conv2d(output_channels, self.out_channels, 1)

    def _validate_spatial_shape(self, x: torch.Tensor) -> None:
        divisor = 2 ** self.num_scale_transitions
        height, width = x.shape[-2:]
        if height % divisor or width % divisor:
            raise ValueError(
                "WaveletBBDMUNet input dimensions must be divisible by "
                f"{divisor}, got {(height, width)}"
            )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context_tokens: Optional[torch.Tensor] = None,
        skip_injections: Optional[List[torch.Tensor]] = None,
        meta: Optional[torch.Tensor] = None,
        ca_beta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_spatial_shape(x)
        if skip_injections is not None and len(skip_injections) != len(self.up_blocks):
            raise ValueError(
                "skip_injections must match the number of wavelet decoder levels"
            )

        time_embedding = self.time_emb(timesteps)
        if meta is not None and self.meta_proj is not None:
            time_embedding = time_embedding + self.meta_proj(meta)

        hidden = self.head(x)
        skips: List[torch.Tensor] = []
        for index, blocks in enumerate(self.down_blocks):
            for block in blocks:
                hidden = block(hidden, time_embedding)
            skips.append(hidden)
            if index < len(self.downsamples):
                hidden = self.downsamples[index](hidden)

        hidden = self.bn_block1(hidden, time_embedding)
        hidden = self.bn_block2(hidden, time_embedding)
        if context_tokens is not None:
            attended = self.cross_attn(hidden, context_tokens)
            if ca_beta is not None:
                beta = ca_beta
                while beta.dim() < hidden.dim():
                    beta = beta.unsqueeze(-1)
                hidden = hidden + beta * (attended - hidden)
            else:
                hidden = attended

        for index, blocks in enumerate(self.up_blocks):
            if index > 0:
                hidden = self.upsamples[index - 1](hidden)
            skip = skips.pop()
            if skip_injections is not None:
                injection = skip_injections[index]
                if injection.shape != skip.shape:
                    raise ValueError(
                        "wavelet skip injection shape mismatch at decoder level "
                        f"{index}: expected {tuple(skip.shape)}, got {tuple(injection.shape)}"
                    )
                skip = skip + injection
            hidden = torch.cat((hidden, skip), dim=1)
            for block in blocks:
                hidden = block(hidden, time_embedding)

        return self.out_conv(torch.nn.functional.silu(self.out_norm(hidden)))
