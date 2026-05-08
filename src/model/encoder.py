"""
MSRD-Style Dual-Stream Condition Encoder for CT-to-PET Diffusion Model.

This module implements:
1. MultiScaleStem: Multi-scale feature extraction from raw CT images.
2. DualStreamCTEncoder: Parallel CNN + Transformer streams for local/global feature fusion.

Reference: Inspired by MSRD-UNet and TransUNet architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# =============================================================================
# Helper Modules
# =============================================================================

def _resolve_group_count(channels: int, preferred_groups: int = 8) -> int:
    """Find a valid GroupNorm group count that divides channels."""
    preferred_groups = max(1, min(preferred_groups, channels))
    for g in range(preferred_groups, 0, -1):
        if channels % g == 0:
            return g
    return 1

class ResnetBlock2D(nn.Module):
    """
    Standard ResNet-style block with skip connection.
    
    Architecture: Conv -> Norm -> Act -> Conv -> Norm -> Add(skip) -> Act
    """
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        groups_in = _resolve_group_count(in_channels, groups)
        groups_out = _resolve_group_count(out_channels, groups)

        self.norm1 = nn.GroupNorm(num_groups=groups_in, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=groups_out, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        
        # Skip connection (1x1 conv if channel mismatch)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        
        return x + residual


class EfficientChannelAttention(nn.Module):
    """
    Efficient Channel-wise Self-Attention (Restormer-style).
    
    Instead of spatial attention O((H*W)^2), this computes attention over channels O(C^2),
    which is much more memory efficient for high-resolution images.
    
    Reference: Restormer (CVPR 2022) - "Restormer: Efficient Transformer for High-Resolution Image Restoration"
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        
        # Depth-wise conv for local context (like in Restormer)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) spatial feature map
        Returns:
            (B, C, H, W) attended feature map
        """
        B, C, H, W = x.shape
        
        # Pre-norm (reshape for LayerNorm)
        x_norm = rearrange(x, 'b c h w -> b (h w) c')
        x_norm = self.norm(x_norm)
        x_norm = rearrange(x_norm, 'b (h w) c -> b c h w', h=H, w=W)
        
        # QKV projection
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention: (B, C, H, W) -> (B, heads, head_dim, H*W)
        q = rearrange(q, 'b (heads d) h w -> b heads d (h w)', heads=self.num_heads)
        k = rearrange(k, 'b (heads d) h w -> b heads d (h w)', heads=self.num_heads)
        v = rearrange(v, 'b (heads d) h w -> b heads d (h w)', heads=self.num_heads)
        
        # Channel-wise attention: attention over spatial dimension, not H*W x H*W
        # This is the key difference: attn shape is (B, heads, d, d) instead of (B, heads, HW, HW)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, d, d)
        attn = attn.softmax(dim=-1)
        
        # Apply attention
        out = attn @ v  # (B, heads, d, HW)
        out = rearrange(out, 'b heads d (h w) -> b (heads d) h w', h=H, w=W)
        
        # Local context enhancement
        out = self.dwconv(out)
        
        # Project and residual
        out = self.proj(out)
        out = out + x
        
        return out


class LocalWindowAttention2D(nn.Module):
    """
    Local window self-attention.

    Each non-overlapping window attends only within itself, emphasizing local texture
    and boundaries while keeping memory usage bounded.
    """
    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 8):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        groups = _resolve_group_count(dim, 8)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ws = self.window_size

        pad_h = (ws - (h % ws)) % ws
        pad_w = (ws - (w % ws)) % ws
        residual = x

        x_norm = self.norm(x)
        if pad_h > 0 or pad_w > 0:
            x_norm = F.pad(x_norm, (0, pad_w, 0, pad_h), mode='replicate')

        _, _, hp, wp = x_norm.shape
        q, k, v = self.qkv(x_norm).chunk(3, dim=1)

        q = rearrange(
            q,
            'b (heads d) (nh ws1) (nw ws2) -> (b nh nw) heads (ws1 ws2) d',
            heads=self.num_heads,
            ws1=ws,
            ws2=ws
        )
        k = rearrange(
            k,
            'b (heads d) (nh ws1) (nw ws2) -> (b nh nw) heads (ws1 ws2) d',
            heads=self.num_heads,
            ws1=ws,
            ws2=ws
        )
        v = rearrange(
            v,
            'b (heads d) (nh ws1) (nw ws2) -> (b nh nw) heads (ws1 ws2) d',
            heads=self.num_heads,
            ws1=ws,
            ws2=ws
        )

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = rearrange(
            out,
            '(b nh nw) heads (ws1 ws2) d -> b (heads d) (nh ws1) (nw ws2)',
            b=b,
            nh=hp // ws,
            nw=wp // ws,
            ws1=ws,
            ws2=ws
        )
        out = self.proj(out)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return out + residual


class GlobalContextAttention2D(nn.Module):
    """
    Global attention with pooled keys/values.

    Full-resolution queries attend to globally pooled context tokens, modeling
    long-range dependencies with controlled cost.
    """
    def __init__(self, dim: int, num_heads: int = 4, pool_size: int = 16):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.pool_size = pool_size
        self.scale = self.head_dim ** -0.5

        groups = _resolve_group_count(dim, 8)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.to_kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x

        x_norm = self.norm(x)
        pooled = F.adaptive_avg_pool2d(x_norm, (self.pool_size, self.pool_size))

        q = self.to_q(x_norm)
        k, v = self.to_kv(pooled).chunk(2, dim=1)

        q = rearrange(q, 'b (heads d) h w -> b heads (h w) d', heads=self.num_heads)
        k = rearrange(k, 'b (heads d) hp wp -> b heads (hp wp) d', heads=self.num_heads)
        v = rearrange(v, 'b (heads d) hp wp -> b heads (hp wp) d', heads=self.num_heads)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = rearrange(out, 'b heads (h w) d -> b (heads d) h w', h=h, w=w)
        out = self.proj(out)

        return out + residual


# =============================================================================
# Main Modules
# =============================================================================

class MultiScaleStem(nn.Module):
    """
    Multi-Scale Stem for extracting features at different receptive field sizes.
    
    Architecture (inspired by MSRD-Net and Inception):
        Path A: Conv3x3
        Path B: Conv3x3 -> ReLU -> Conv7x7
        Fusion: PathA + PathB -> Conv1x1 -> GroupNorm -> SiLU
    
    Args:
        in_channels: Number of input channels (default: 1 for CT)
        base_channels: Number of output channels (default: 64)
        groups: Number of groups for GroupNorm (default: 8)
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 64, groups: int = 8):
        super().__init__()
        
        # Path A: Fine-grained local features (3x3 kernel)
        self.path_a = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Path B: Larger receptive field (3x3 -> 7x7)
        self.path_b_conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.path_b_act = nn.ReLU(inplace=True)
        self.path_b_conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=7, padding=3)
        
        # Fusion: Project and normalize
        self.fusion_conv = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        norm_groups = _resolve_group_count(base_channels, groups)
        self.norm = nn.GroupNorm(num_groups=norm_groups, num_channels=base_channels)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 1, H, W)
        Returns:
            Multi-scale features of shape (B, base_channels, H, W)
        """
        # Path A: Direct 3x3 convolution
        out_a = self.path_a(x)
        
        # Path B: 3x3 -> ReLU -> 7x7 (captures larger context)
        out_b = self.path_b_conv1(x)
        out_b = self.path_b_act(out_b)
        out_b = self.path_b_conv2(out_b)
        
        # Fusion: Element-wise addition
        fused = out_a + out_b
        
        # Refinement: 1x1 projection + normalization + activation
        out = self.fusion_conv(fused)
        out = self.norm(out)
        out = self.act(out)
        
        return out


class DualStreamCTEncoder(nn.Module):
    """
    Dual-Stream Condition Encoder for CT images.
    
    Captures both:
        - Local Texture (CNN Stream): Sharp edges, boundaries, fine details
        - Global Anatomy (Transformer Stream): Long-range dependencies, organ context
    
    Architecture:
        Input -> MultiScaleStem -> [CNN Stream | Transformer Stream] -> Concat -> Conv1x1 -> Output
    
    Args:
        in_channels: Input channels (default: 1 for CT grayscale)
        base_channels: Base feature channels (default: 64)
        out_channels: Output channels to match U-Net (default: 320 for SD)
        num_heads: Number of attention heads (default: 8)
        num_res_blocks: Number of ResNet blocks in CNN stream (default: 3)
        groups: GroupNorm groups (default: 8)
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        out_channels: int = 320,
        num_heads: int = 8,
        num_res_blocks: int = 3,
        groups: int = 8
    ):
        super().__init__()
        
        # Stage 1: Multi-Scale Stem
        self.stem = MultiScaleStem(in_channels=in_channels, base_channels=base_channels, groups=groups)
        
        # Stream 1: CNN Path (Local Texture)
        self.cnn_stream = nn.Sequential(*[
            ResnetBlock2D(in_channels=base_channels, out_channels=base_channels, groups=groups)
            for _ in range(num_res_blocks)
        ])
        
        # Stream 2: Transformer Path (Global Anatomy)
        self.transformer_stream = EfficientChannelAttention(dim=base_channels, num_heads=num_heads)
        
        # Fusion: Concatenate and reduce channels
        # After concat: base_channels * 2 -> out_channels
        self.fusion_conv = nn.Conv2d(base_channels * 2, out_channels, kernel_size=1)
        self.fusion_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.fusion_act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: CT image tensor of shape (B, 1, H, W)
        Returns:
            Condition features of shape (B, out_channels, H, W)
        """
        # Multi-Scale Stem: (B, 1, H, W) -> (B, base_channels, H, W)
        stem_features = self.stem(x)
        
        # Stream 1 (CNN): Local texture features
        cnn_out = self.cnn_stream(stem_features)
        
        # Stream 2 (Transformer): Global context features
        transformer_out = self.transformer_stream(stem_features)
        
        # Fusion: Concatenate along channel dimension
        fused = torch.cat([cnn_out, transformer_out], dim=1)
        
        # Output projection
        out = self.fusion_conv(fused)
        out = self.fusion_norm(out)
        out = self.fusion_act(out)
        
        return out


class TrueDualStreamAttentionEncoder(nn.Module):
    """
    True dual-stream condition encoder with parallel local/global attention.

    Stream A (local): CNN + local-window attention for lesion-level details.
    Stream B (global): CNN + global-context attention for anatomy consistency.
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        out_channels: int = 64,
        num_heads: int = 4,
        groups: int = 8,
        local_window_size: int = 8,
        global_pool_size: int = 16,
    ):
        super().__init__()

        self.stem = MultiScaleStem(
            in_channels=in_channels,
            base_channels=base_channels,
            groups=groups
        )

        self.local_stream = nn.Sequential(
            ResnetBlock2D(base_channels, base_channels, groups=groups),
            LocalWindowAttention2D(
                dim=base_channels,
                num_heads=num_heads,
                window_size=local_window_size,
            ),
            ResnetBlock2D(base_channels, base_channels, groups=groups),
        )

        self.global_stream = nn.Sequential(
            ResnetBlock2D(base_channels, base_channels, groups=groups),
            GlobalContextAttention2D(
                dim=base_channels,
                num_heads=num_heads,
                pool_size=global_pool_size,
            ),
            ResnetBlock2D(base_channels, base_channels, groups=groups),
        )

        norm_groups = _resolve_group_count(out_channels, groups)
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels * 2, out_channels, kernel_size=1),
            nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.stem(x)
        local_feat = self.local_stream(stem)
        global_feat = self.global_stream(stem)
        fused = torch.cat([local_feat, global_feat], dim=1)
        return self.fusion(fused)


# =============================================================================
# Testing / Demo
# =============================================================================

if __name__ == "__main__":
    # Test the encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    encoder = DualStreamCTEncoder(
        in_channels=1,
        base_channels=64,
        out_channels=320,
        num_heads=8,
        num_res_blocks=3
    ).to(device)
    
    # Create dummy CT input: (Batch=2, Channels=1, H=256, W=256)
    dummy_ct = torch.randn(2, 1, 256, 256).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = encoder(dummy_ct)
    
    print(f"Input shape:  {dummy_ct.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in encoder.parameters()):,}")
