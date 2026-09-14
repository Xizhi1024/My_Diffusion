"""Tiny PET Segmenter — lightweight PET→lesion relaxed segmentation.

Three-stage training protocol (v2.0 doc 九):
  Stage 1: PET → Gaussian-smoothed mask (σ=6px), L1 loss
  Stage 2: PET → signed distance transform of mask, L1 loss
  Stage 3: PET → hard binary mask, Dice + Focal loss

After training, frozen segmenter provides L_seg for BBDM:
  L_seg = L1(seg(syn_PET), seg(real_PET))   (distance-transform consistency)

Architecture: tiny UNet (< 0.3M params), PET 1ch in → 1ch out.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(size: int = 13, sigma: float = 6.0) -> torch.Tensor:
    """2D Gaussian kernel for heatmap generation (stage 1)."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    k = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    return k.view(1, 1, size, size)


def mask_to_heatmap(mask: torch.Tensor, kernel_size: int = 13, sigma: float = 6.0) -> torch.Tensor:
    """Convert binary lesion mask → Gaussian-smoothed heatmap."""
    kernel = _gaussian_kernel(kernel_size, sigma).to(mask.device, mask.dtype)
    pad = kernel_size // 2
    heatmap = F.conv2d(
        F.pad(mask, (pad, pad, pad, pad), mode="replicate"),
        kernel,
    )
    heatmap = heatmap / heatmap.amax(dim=(2, 3), keepdim=True).clamp_min(1e-8)
    return heatmap


# ---------------------------------------------------------------------------
# Tiny UNet
# ---------------------------------------------------------------------------

class _TinySegUNet(nn.Module):
    """Minimal U-Net for PET→segmentation.  ~0.25M params with base=16."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 16):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.SiLU(),
        )
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)   # /2
        self.enc2 = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.SiLU(),
        )
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)  # /4
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
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.silu(self.down1(e1)))
        b = self.bottleneck(F.silu(self.down2(e2)))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


# ---------------------------------------------------------------------------
# Main segmenter
# ---------------------------------------------------------------------------

class TinySegmenter(nn.Module):
    """PET → relaxed lesion segmentation with configurable stage.

    Stage 1 (heatmap):    sigmoid output → Gaussian-smoothed lesion probability
    Stage 2 (relaxed):    tanh output → signed-distance-like continuous map
    Stage 3 (hard):       sigmoid output → binary lesion mask
    """

    name = "tiny_segmenter"

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        stage: int = 2,             # 1=heatmap, 2=relaxed(dist), 3=hard
        gaussian_sigma: float = 6.0,
        enabled: bool = True,
    ):
        super().__init__()
        self.enabled = enabled
        self.stage = stage
        self.gaussian_sigma = gaussian_sigma
        self.net = _TinySegUNet(in_ch=in_channels, out_ch=1, base=base_channels)

        total = sum(p.numel() for p in self.net.parameters())
        if total > 500_000:
            print(f"[TinySegmenter] WARNING: {total:,} params exceeds 500K budget")

    def forward(self, pet: torch.Tensor) -> torch.Tensor:
        """pet: [B, 1, H, W] (normalised [-1, 1] or physical units).
        Returns [B, 1, H, W] continuous lesion map.
        """
        if not self.enabled:
            return torch.zeros_like(pet)
        raw = self.net(pet)
        if self.stage == 1:
            # Gaussian-smoothed heatmap: sigmoid → [0, 1]
            return torch.sigmoid(raw)
        elif self.stage == 3:
            # Hard mask: sigmoid
            return torch.sigmoid(raw)
        else:
            # Stage 2 (default): distance-transform-like continuous output
            # tanh gives [-1, 1] with smooth transitions, good for relaxed supervision
            return torch.tanh(raw)

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Stage-specific target generation
    # ------------------------------------------------------------------

    @staticmethod
    def build_target(
        mask: torch.Tensor,
        stage: int = 2,
        gaussian_sigma: float = 6.0,
    ) -> torch.Tensor:
        """Build training target for a given stage from binary lesion mask.

        Args:
            mask: [B, 1, H, W] binary (0/1) lesion mask
            stage: 1=heatmap, 2=relaxed(dist-like), 3=hard
            gaussian_sigma: sigma for stage 1 heatmap

        Returns:
            target: [B, 1, H, W] in appropriate range for stage
        """
        if stage == 1:
            return mask_to_heatmap(mask, sigma=gaussian_sigma)
        elif stage == 3:
            return mask  # already binary
        else:
            # Stage 2: pseudo-distance transform (signed, continuous)
            return _pseudo_distance_transform(mask)


def _pseudo_distance_transform(mask: torch.Tensor) -> torch.Tensor:
    """Approximate signed distance transform via distance to mask boundary.

    Positive inside lesion, negative outside, zero at boundary.
    This is a target-generation helper, so it is exact via SciPy when available
    and falls back to a smoothed heatmap when SciPy is missing.
    """
    try:
        import numpy as np
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        heat = mask_to_heatmap(mask, kernel_size=13, sigma=4.0)
        return (2.0 * heat - 1.0).clamp(-1.0, 1.0)

    mask_np = (mask.detach().cpu().numpy() > 0.5).astype(np.bool_)
    out = np.empty(mask_np.shape, dtype=np.float32)
    for idx in np.ndindex(mask_np.shape[:2]):
        m = mask_np[idx]
        d_in = distance_transform_edt(m).astype(np.float32)
        d_out = distance_transform_edt(~m).astype(np.float32)
        signed = d_in - d_out
        max_abs = max(float(np.abs(signed).max()), 1.0)
        out[idx] = np.tanh(signed / max_abs).astype(np.float32)
    return torch.from_numpy(out).to(device=mask.device, dtype=mask.dtype)


# --------------------------------------------------------------------------
# Training helper
# --------------------------------------------------------------------------

def segmenter_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    stage: int = 2,
    dice_weight: float = 0.5,
    focal_weight: float = 0.5,
    focal_gamma: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute segmenter training loss for the current stage.

    Args:
        pred: segmenter output [B, 1, H, W]
        target: stage-appropriate target [B, 1, H, W]
        mask: binary lesion mask [B, 1, H, W] (for stage 3 Dice)
        stage: current stage (1/2/3)

    Returns:
        (loss, logs)
    """
    if stage <= 2:
        # L1 loss for relaxed targets
        l1 = F.l1_loss(pred, target)
        logs = {"seg/l1": l1.detach()}
        return l1, logs
    else:
        # Stage 3: Dice + Focal on hard masks
        pred_prob = torch.sigmoid(pred) if mask is not None else pred
        if mask is None:
            return F.l1_loss(pred, target), {"seg/l1": torch.tensor(0.0)}

        # Dice
        smooth = 1.0
        intersection = (pred_prob * mask).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
        dice = 1.0 - ((2.0 * intersection + smooth) / (union + smooth)).mean()

        # Focal. BCE is unsafe while CUDA autocast is enabled, even if inputs
        # are explicitly cast to float32.
        autocast_device = pred_prob.device.type if pred_prob.device.type in {"cuda", "cpu"} else "cpu"
        with torch.amp.autocast(autocast_device, enabled=False):
            pred_f = pred_prob.float()
            mask_f = mask.float()
            bce = F.binary_cross_entropy(pred_f, mask_f, reduction="none")
            pt = torch.where(mask_f > 0.5, pred_f, 1.0 - pred_f)
        focal = ((1.0 - pt) ** focal_gamma * bce).mean()

        loss = dice_weight * dice + focal_weight * focal
        logs = {"seg/dice": dice.detach(), "seg/focal": focal.detach()}
        return loss, logs
