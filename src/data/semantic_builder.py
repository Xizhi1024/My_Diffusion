"""Semantic Token Builder: frozen backbone feature extraction for CT slices.

CT HU -> 3-window fake RGB -> frozen backbone -> per-slice tokens -> .npz cache
Tokens are consumed by SemanticPrior at training time (injected via Cross-Attention).

Backends:
  - radimagenet: RadImageNet pretrained ResNet50 (recommended, medical domain,
                  requires ``pip install timm`` + model download ~100 MB)
  - random:      Random projection baseline, no pretrained weights needed,
                  deterministic per run (fixed seed=42).
  - dinov2:      DINOv2 ViT-S/14 (Meta, general vision).
                  **NOT VALIDATED — requires model download + potential code adaptation.**
                  The ``torch.hub.load('facebookresearch/dinov2', ...)`` path is
                  officially maintained but was never tested in this pipeline.
                  Use for ablation comparison against RadImageNet only after
                  verifying the model loads correctly in your environment.
                  Download size: ~85 MB (ViT-S/14).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Three-window fake RGB generation
# ---------------------------------------------------------------------------

_CT_WINDOWS = {
    "soft_tissue":   {"width": 350,  "level": 40},
    "fat_soft":      {"width": 400,  "level": -50},
    "bone":          {"width": 1500, "level": 500},
}


def ct_hu_to_3win_rgb(
    ct_hu: np.ndarray,             # [H, W] in HU
    windows: Optional[Dict[str, Dict[str, float]]] = None,
) -> np.ndarray:
    """Convert CT HU to 3-channel RGB via windowing.

    Returns [H, W, 3] float32 in [0, 1].
    """
    if windows is None:
        windows = _CT_WINDOWS

    rgb = np.zeros((*ct_hu.shape, 3), dtype=np.float32)
    for i, (name, w) in enumerate(windows.items()):
        width = w["width"]
        level = w["level"]
        low = level - width / 2
        high = level + width / 2
        channel = np.clip((ct_hu - low) / (high - low), 0.0, 1.0)
        rgb[..., i] = channel.astype(np.float32)
    return rgb


def ct_norm_to_3win_rgb(
    ct_norm: np.ndarray,           # [H, W] in [-1, 1]
    ct_hu_min: float = -150.0,
    ct_hu_max: float = 250.0,
) -> np.ndarray:
    """Convert normalised CT [-1, 1] -> 3-window RGB via intermediate HU."""
    # [-1, 1] -> [0, 1] -> HU
    ct_01 = (ct_norm + 1.0) / 2.0
    ct_hu = ct_01 * (ct_hu_max - ct_hu_min) + ct_hu_min
    return ct_hu_to_3win_rgb(ct_hu)


# ---------------------------------------------------------------------------
# Backbone wrappers
# ---------------------------------------------------------------------------

class _BackboneWrapper(nn.Module):
    """Abstract frozen backbone -> token extractor."""

    def __init__(self, token_dim: int, num_tokens: int):
        super().__init__()
        self.token_dim = token_dim
        self.num_tokens = num_tokens

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb: [B, 3, H, W] -> tokens: [B, num_tokens, token_dim]"""
        raise NotImplementedError


class _RadImageNetBackbone(_BackboneWrapper):
    """RadImageNet pretrained ResNet50 -> global average pool -> project."""

    def __init__(self, token_dim: int = 64, num_tokens: int = 4):
        super().__init__(token_dim, num_tokens)
        try:
            import timm
            self.encoder = timm.create_model("resnet50.radimagenet_irh", pretrained=True, num_classes=0)
        except Exception:
            print("[SemanticBuilder] WARNING: RadImageNet model not available via timm.")
            print("  Install: pip install timm")
            print("  Falling back to random projection.")
            self.encoder = None

        feat_dim = 2048  # ResNet50 global pool
        self.token_proj = nn.Linear(feat_dim, token_dim * num_tokens)

        # Freeze encoder
        if self.encoder is not None:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        B = rgb.shape[0]
        if self.encoder is not None:
            with torch.no_grad():
                feat = self.encoder(rgb)  # [B, 2048]
        else:
            # Random fallback: use positional-statistical features
            feat = torch.cat([
                rgb.mean(dim=(2, 3)),
                rgb.std(dim=(2, 3)),
                rgb.amax(dim=(2, 3)),
                rgb.amin(dim=(2, 3)),
            ], dim=1)  # [B, 12]
            # Pad/expand to match feat_dim
            if feat.shape[1] < 2048:
                pad = torch.zeros(B, 2048 - feat.shape[1], device=feat.device, dtype=feat.dtype)
                feat = torch.cat([feat, pad], dim=1)

        tokens = self.token_proj(feat)  # [B, token_dim * num_tokens]
        return tokens.view(B, self.num_tokens, self.token_dim)


class _DINOv2Backbone(_BackboneWrapper):
    """DINOv2 ViT-S/14 -> CLS token -> project to multiple tokens."""

    def __init__(self, token_dim: int = 64, num_tokens: int = 4):
        super().__init__(token_dim, num_tokens)
        try:
            self.encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        except Exception:
            print("[SemanticBuilder] WARNING: DINOv2 not available via torch.hub.")
            print("  Falling back to random projection.")
            self.encoder = None

        feat_dim = 384  # ViT-S/14
        self.token_proj = nn.Linear(feat_dim, token_dim * num_tokens)

        if self.encoder is not None:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        B = rgb.shape[0]
        if self.encoder is not None:
            with torch.no_grad():
                feat = self.encoder(rgb)  # [B, 384]
        else:
            feat = torch.cat([
                rgb.mean(dim=(2, 3)),
                rgb.std(dim=(2, 3)),
            ], dim=1)
            if feat.shape[1] < 384:
                pad = torch.zeros(B, 384 - feat.shape[1], device=feat.device, dtype=feat.dtype)
                feat = torch.cat([feat, pad], dim=1)

        tokens = self.token_proj(feat)
        return tokens.view(B, self.num_tokens, self.token_dim)


class _RandomBackbone(_BackboneWrapper):
    """Deterministic random projection baseline (no pretrained weights needed)."""

    def __init__(self, token_dim: int = 64, num_tokens: int = 4, seed: int = 42):
        super().__init__(token_dim, num_tokens)
        rng = np.random.RandomState(seed)
        H_out, W_out = 7, 7
        self.proj = nn.Linear(H_out * W_out * 3, token_dim * num_tokens)
        # Initialise with fixed random weights
        with torch.no_grad():
            proj_w = rng.randn(token_dim * num_tokens, H_out * W_out * 3).astype(np.float32)
            self.proj.weight.copy_(torch.from_numpy(proj_w * 0.02))
            self.proj.bias.zero_()
        for p in self.proj.parameters():
            p.requires_grad = False

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        B = rgb.shape[0]
        # Average pool to 7x7 then flatten
        x = F.adaptive_avg_pool2d(rgb, (7, 7))
        x = x.reshape(B, -1)
        tokens = self.proj(x)
        return tokens.view(B, self.num_tokens, self.token_dim)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class SemanticTokenBuilder:
    """Offline pipeline: CT -> 3-window RGB -> frozen backbone -> .npz tokens.

    Usage::

        builder = SemanticTokenBuilder(backend="radimagenet", token_dim=64, num_tokens=4)
        builder.process_cache(cache_dir)
    """

    def __init__(
        self,
        backend: str = "radimagenet",
        token_dim: int = 64,
        num_tokens: int = 4,
        device: str = "cpu",
        image_size: int = 192,
    ):
        self.backend_name = backend
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.device = device
        self.image_size = image_size

        if backend == "radimagenet":
            self.model = _RadImageNetBackbone(token_dim, num_tokens)
        elif backend == "dinov2":
            self.model = _DINOv2Backbone(token_dim, num_tokens)
        elif backend == "random":
            self.model = _RandomBackbone(token_dim, num_tokens)
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose radimagenet, dinov2, or random.")

        self.model = self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def _extract_tokens(self, rgb_batch: torch.Tensor) -> np.ndarray:
        """rgb_batch: [B, 3, H, W] -> tokens: [B, num_tokens, token_dim] np.float32."""
        rgb_batch = rgb_batch.to(self.device)
        tokens = self.model(rgb_batch)
        return tokens.cpu().numpy().astype(np.float32)

    def process_cache(
        self,
        cache_dir: Path,
        batch_size: int = 16,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Read .npz files, compute semantic_tokens, write back to each .npz.

        Returns stats dict.
        """
        cache_dir = Path(cache_dir)
        npz_files = sorted(cache_dir.glob("*.npz"))
        if not npz_files:
            raise ValueError(f"No .npz files in {cache_dir}")

        stats: Dict[str, Any] = {
            "total": len(npz_files), "updated": 0, "skipped": 0,
            "missing_ct_hu_range": 0, "errors": [],
        }

        # Process in batches for GPU efficiency
        for start in range(0, len(npz_files), batch_size):
            batch_paths = npz_files[start:start + batch_size]

            # Load CT data and convert to 3-window RGB
            rgb_list = []
            metas = []
            valid_indices = []

            for i, npz_path in enumerate(batch_paths):
                try:
                    data = np.load(npz_path)
                    # Handle ct shape: [H,W], [1,H,W], or [1,1,H,W]
                    ct_raw = data["ct"]
                    ct_norm = np.squeeze(ct_raw)  # removes all singleton dims
                    if ct_norm.ndim != 2:
                        stats.setdefault("errors", []).append(
                            f"{npz_path.name}: unexpected ct shape {ct_raw.shape}")
                        stats["skipped"] += 1
                        continue

                    # Extract scale_meta for HU range
                    scale_meta = {}
                    if "scale_meta_json" in data:
                        try:
                            json_bytes = bytes(data["scale_meta_json"].tolist())
                            scale_meta = json.loads(json_bytes.decode("utf-8"))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            pass

                    ct_hu_min = scale_meta.get("ct_hu_min", -150.0)
                    ct_hu_max = scale_meta.get("ct_hu_max", 250.0)

                    rgb = ct_norm_to_3win_rgb(ct_norm, ct_hu_min, ct_hu_max)
                    rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()  # [3, H, W]
                    rgb = F.interpolate(
                        rgb.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
                    ).squeeze(0)

                    rgb_list.append(rgb)
                    metas.append(scale_meta)
                    valid_indices.append(i)

                except Exception as exc:
                    stats.setdefault("errors", []).append(f"{npz_path.name}: {exc}")
                    stats["skipped"] += 1

            if not rgb_list:
                continue

            # Batch extract tokens
            rgb_batch = torch.stack(rgb_list, dim=0)  # [B, 3, 224, 224]
            tokens_batch = self._extract_tokens(rgb_batch)  # [B, num_tokens, token_dim]

            # Write tokens back to each .npz — use valid_indices for correct alignment
            for j, orig_idx in enumerate(valid_indices):
                npz_path = batch_paths[orig_idx]
                try:
                    existing = dict(np.load(npz_path))
                    existing["semantic_tokens"] = tokens_batch[j]
                    np.savez_compressed(str(npz_path), **existing)
                    stats["updated"] += 1
                except Exception as exc:
                    stats.setdefault("errors", []).append(f"{npz_path.name}: {exc}")
                    stats["skipped"] += 1

            if start % (batch_size * 5) == 0:
                print(f"  Processed {min(start + batch_size, len(npz_files))}/{len(npz_files)} files")

        return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Semantic token builder for SLMF-BBDM")
    ap.add_argument("--cache-dir", required=True, type=Path, help="Path to .npz cache directory")
    ap.add_argument("--backend", default="radimagenet", choices=["radimagenet", "dinov2", "random"])
    ap.add_argument("--token-dim", type=int, default=64)
    ap.add_argument("--num-tokens", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--gpu", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", help="Check availability, no processing")
    args = ap.parse_args(argv)

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    print(f"[SemanticBuilder] backend={args.backend} token_dim={args.token_dim} num_tokens={args.num_tokens}")
    print(f"  device={device}")

    if args.dry_run:
        npz_files = sorted(args.cache_dir.glob("*.npz"))
        print(f"  {len(npz_files)} .npz files in {args.cache_dir}")
        # Check if backend is available
        try:
            builder = SemanticTokenBuilder(
                backend=args.backend,
                token_dim=args.token_dim,
                num_tokens=args.num_tokens,
                device=device,
            )
            print(f"  Backend '{args.backend}': OK")
        except Exception as exc:
            print(f"  Backend '{args.backend}': FAILED — {exc}")
        return 0

    builder = SemanticTokenBuilder(
        backend=args.backend,
        token_dim=args.token_dim,
        num_tokens=args.num_tokens,
        device=device,
    )

    stats = builder.process_cache(args.cache_dir, batch_size=args.batch_size)

    print(f"\nDone. {stats['updated']} updated, {stats['skipped']} skipped, "
          f"{len(stats.get('errors', []))} errors")
    if stats.get("errors"):
        for err in stats["errors"][:5]:
            print(f"  - {err}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
