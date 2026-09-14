"""Condition dropout – randomly replace conditions with null during training.

Improves robustness to missing conditions and enables weak CFG at inference.
Each condition is independently dropped with probability p.
"""

import random

import torch

from ..interfaces import ConditionBundle


class ConditionDropout:
    """Stateless dropout controller applied to ConditionBundle before UNet injection."""

    def __init__(
        self,
        p_organ: float = 0.1,
        p_hotspot: float = 0.1,
        p_semantic: float = 0.1,
        p_meta: float = 0.1,
        p_gabor: float = 0.1,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.p_organ = p_organ
        self.p_hotspot = p_hotspot
        self.p_semantic = p_semantic
        self.p_meta = p_meta
        self.p_gabor = p_gabor

    def apply(self, condition: ConditionBundle, training: bool = True) -> ConditionBundle:
        """Zero out condition entries with configured probabilities."""
        if not training or not self.enabled:
            return condition

        # Avoid mutating the original bundle; other consumers may still need it.
        condition = condition.copy()

        if random.random() < self.p_organ:
            for k in list(condition.maps.keys()):
                if "organ" in k:
                    condition.maps[k] = torch.zeros_like(condition.maps[k])

        if random.random() < self.p_gabor:
            for k in list(condition.maps.keys()):
                if "gabor" in k:
                    condition.maps[k] = torch.zeros_like(condition.maps[k])

        if random.random() < self.p_hotspot:
            if "hotspot_prior" in condition.maps:
                condition.maps["hotspot_prior"] = torch.zeros_like(condition.maps["hotspot_prior"])

        if random.random() < self.p_semantic:
            for k in list(condition.tokens.keys()):
                if "semantic" in k:
                    condition.tokens[k] = torch.zeros_like(condition.tokens[k])

        if random.random() < self.p_meta:
            for k in list(condition.scalars.keys()):
                condition.scalars[k] = torch.zeros_like(condition.scalars[k])

        return condition
