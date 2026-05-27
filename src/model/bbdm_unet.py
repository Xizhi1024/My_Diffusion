"""Residual U-Net with bottleneck Cross-Attention for BBDM.

Simple architecture:
  - Encoder: conv head → 4 levels of (ResBlocks + Downsample)
  - Bottleneck: 2× ResBlock + CrossAttention
  - Decoder: 4 levels of (Upsample + concat skip + ResBlocks)
  - Output: GroupNorm → SiLU → Conv2d(→2 if heteroscedastic, else 1)
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.emb_dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.emb_dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return self.mlp(torch.cat([emb.sin(), emb.cos()], dim=1))


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch * 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale_shift = self.time_proj(F.silu(t_emb))
        scale, shift = scale_shift.chunk(2, dim=1)
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(-1); shift = shift.unsqueeze(-1)
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = self.norm2(h) * (1 + scale) + shift
        h = F.silu(h); h = self.dropout(h); h = self.conv2(h)
        return h + self.skip(x)


class _CrossAttention(nn.Module):
    """Flash Attention via F.scaled_dot_product_attention."""

    def __init__(self, query_dim: int, kv_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(kv_dim, query_dim)
        self.to_v = nn.Linear(kv_dim, query_dim)
        self.to_out = nn.Linear(query_dim, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, H * W).permute(0, 2, 1)
        q = self.to_q(x_flat).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.to_k(context).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.to_v(context).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(B, H * W, C)
        return self.to_out(out).permute(0, 2, 1).reshape(B, C, H, W) + x


def _zero_conv(in_ch: int, out_ch: int) -> nn.Conv2d:
    c = nn.Conv2d(in_ch, out_ch, 1)
    nn.init.zeros_(c.weight); nn.init.zeros_(c.bias)
    return c


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class BBDMUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        channel_mult: tuple = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        time_dim: int = 256,
        dropout: float = 0.0,
        enable_heteroscedastic: bool = True,
        ca_kv_dim: int = 64,
        ca_num_heads: int = 4,
        meta_dim: int = 0,
    ):
        super().__init__()
        self.enable_heteroscedastic = enable_heteroscedastic
        self.time_emb = _TimeEmbedding(time_dim)

        self.meta_proj = None
        if meta_dim > 0:
            self.meta_proj = nn.Sequential(
                nn.Linear(meta_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim),
            )

        chs = [base_channels * m for m in channel_mult]  # e.g. [64, 128, 256, 256]

        # ---- Encoder ----
        self.head = nn.Conv2d(in_channels, chs[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i, ch in enumerate(chs):
            blocks = nn.ModuleList([_ResBlock(ch, ch, time_dim, dropout) for _ in range(num_res_blocks)])
            self.down_blocks.append(blocks)
            if i < len(chs) - 1:
                self.downsamples.append(nn.Conv2d(ch, chs[i + 1], 3, stride=2, padding=1))

        # ---- Bottleneck ----
        bn_ch = chs[-1]
        self.bn_block1 = _ResBlock(bn_ch, bn_ch, time_dim, dropout)
        self.bn_block2 = _ResBlock(bn_ch, bn_ch, time_dim, dropout)
        self.cross_attn = _CrossAttention(bn_ch, ca_kv_dim, ca_num_heads)

        # ---- Decoder ----
        rev_chs = list(reversed(chs))
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, ch in enumerate(rev_chs):
            blocks = nn.ModuleList()
            blocks.append(_ResBlock(ch * 2, ch, time_dim, dropout))  # 2*ch → ch
            for _ in range(num_res_blocks - 1):
                blocks.append(_ResBlock(ch, ch, time_dim, dropout))   # ch → ch
            self.up_blocks.append(blocks)
            if i < len(rev_chs) - 1:
                self.upsamples.append(nn.Conv2d(ch, rev_chs[i + 1], 3, padding=1))

        # ---- Output ----
        out_ch = chs[0]
        self.out_norm = nn.GroupNorm(min(32, out_ch), out_ch)
        self.out_channels = 2 if enable_heteroscedastic else 1
        self.out_conv = nn.Conv2d(out_ch, self.out_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context_tokens: Optional[torch.Tensor] = None,
        skip_injections: Optional[List[torch.Tensor]] = None,
        meta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_emb = self.time_emb(timesteps)
        if meta is not None and self.meta_proj is not None:
            t_emb = t_emb + self.meta_proj(meta)

        # ---- Encoder ----
        h = self.head(x)
        skips = []

        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, t_emb)
            skips.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        # ---- Bottleneck ----
        h = self.bn_block1(h, t_emb)
        h = self.bn_block2(h, t_emb)
        if context_tokens is not None:
            h = self.cross_attn(h, context_tokens)

        # ---- Decoder ----
        for i, blocks in enumerate(self.up_blocks):
            if i > 0:
                h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
                h = self.upsamples[i - 1](h)
            skip = skips.pop()
            # Apply zero-conv injection to skip
            if skip_injections is not None and i < len(skip_injections):
                inj = skip_injections[i]
                if inj.shape[2:] != skip.shape[2:]:
                    inj = F.interpolate(inj, size=skip.shape[2:], mode="bilinear", align_corners=False)
                skip = skip + inj
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, t_emb)

        # ---- Output ----
        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_conv(h)
