"""Semantic prior – frozen visual features as global context tokens.

Original plan: DINOv2 / RadImageNet pretrained backbone extracts semantic tokens
from CT (via 3-window fake RGB), injected into UNet bottleneck Cross-Attention
to provide global anatomical context ("what body region / tissue type is this slice").

Current state:
  - RadImageNet ResNet50: implemented in semantic_builder.py (recommended for medical)
  - DINOv2 ViT-S/14:    code path exists but NOT validated — requires model download
                          (~100 MB) and may need code adaptation (torch.hub vs local).
                          Se actualmente como ablation option vs RadImageNet.
  - Random projection:   baseline, no pretrained weights needed

Alternatives worth considering before committing to DINOv2:
  - Learnable per-position embeddings (no backbone, just learn from data)
  - BiomedCLIP / PMC-CLIP (medical vision-language, better domain match)
  - Organ segmentation statistics (already partially covered by OrganPrior)
  - Simply drop — 4 tokens may be too few to carry useful information for a
    192×192 single-slice task where OrganPrior already provides anatomy.

Tokens are precomputed offline and stored in .npz cache.  At training time
SemanticPrior just reads cached tokens or emits null tokens.
"""

from typing import Dict

import torch
import torch.nn as nn

from ..interfaces import ConditionBundle, PriorModule


class SemanticPrior(PriorModule):
    name = "semantic_prior"

    def __init__(
        self,
        mode: str = "cached_tokens",
        token_dim: int = 64,
        num_tokens: int = 4,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.mode = mode
        self.token_dim = token_dim
        self.num_tokens = num_tokens

        self.null_tokens = nn.Parameter(torch.zeros(1, num_tokens, token_dim))

        # Project cached features or act as deterministic null tokens.
        if mode == "cached_tokens":
            self.proj = nn.Linear(token_dim, token_dim)

    def forward(self, batch: Dict[str, torch.Tensor], timesteps: torch.Tensor, partial_bundle=None) -> ConditionBundle:
        if not self.enabled:
            return ConditionBundle(logs={f"{self.name}/enabled": False})

        B = batch["ct"].shape[0]

        if self.mode == "cached_tokens" and "semantic_tokens" in batch:
            tokens = self.proj(batch["semantic_tokens"])  # [B, N, D]
        elif self.mode == "cached_tokens":
            null = self.null_tokens.expand(B, -1, -1).to(device=batch["ct"].device)
            tokens = self.proj(null)
        else:
            tokens = self.null_tokens.expand(B, -1, -1)

        return ConditionBundle(
            tokens={"semantic": tokens},
            logs={f"{self.name}/enabled": True},
        )
