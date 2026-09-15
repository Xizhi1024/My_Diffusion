"""EMA (Exponential Moving Average) for stable inference weights."""

import torch
import torch.nn as nn


class EMA:
    """Exponential Moving Average with optional interval-based updates.

    Args:
        model: the training model
        decay: EMA decay rate (0.999 = slow, 0.995 = fast)
        update_every: only update every N optimizer steps (None = every step)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, update_every: int = 10):
        self.model = model
        self.decay = decay
        self.update_every = update_every
        self.step_count = 0
        self.shadow: dict = {}
        self._backup: dict = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self):
        self.step_count += 1
        if self.step_count % self.update_every != 0:
            return
        untracked_trainable = []
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
                elif param.requires_grad:
                    # Audit P1-4: a parameter that became trainable AFTER this
                    # EMA was constructed (e.g. resume with a different freeze
                    # set) is silently never averaged and never swapped in by
                    # apply(). Surface it once instead of running a partial EMA.
                    untracked_trainable.append(name)
        if untracked_trainable and not getattr(self, "_untracked_warned", False):
            self._untracked_warned = True
            print(
                f"[EMA] WARNING: {len(untracked_trainable)} trainable parameter(s) "
                f"are not tracked by the EMA shadow (e.g. {untracked_trainable[:3]}). "
                "They were probably unfrozen after EMA construction (resume with a "
                "changed freeze set); eval/apply() will mix raw weights for them."
            )

    def apply(self):
        """Save current training weights and copy EMA weights into model."""
        self._backup = {}
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore the training weights saved by apply()."""
        for name, param in self.model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self) -> dict:
        return {
            "shadow": self.shadow,
            "step_count": self.step_count,
            "decay": self.decay,
            "update_every": self.update_every,
        }

    def load_state_dict(self, state: dict):
        shadow = state["shadow"]
        # Audit P1-4: refuse mismatched key sets instead of silently becoming
        # a no-op (update/apply both skip names missing from the shadow).
        model_keys = {name for name, _ in self.model.named_parameters()}
        stale = [k for k in shadow if k not in model_keys]
        if stale:
            raise RuntimeError(
                f"EMA shadow contains {len(stale)} parameter name(s) absent from "
                f"the model (first: {stale[:3]}). This typically means the "
                "checkpoint was saved under a different module wrapping (e.g. "
                "torch.compile '_orig_mod.' prefix) or architecture. Refusing "
                "to silently disable EMA tracking."
            )
        self.shadow = shadow
        self.step_count = state["step_count"]
        # Audit T1 (reviewer F1): decay/update_every are run configuration,
        # not restorable state. Letting a checkpoint's decay overwrite the
        # configured value silently reverted the EMA cadence when resuming an
        # old run under a new config (e.g. stale decay=0.999 overriding the
        # configured 0.995). Keep the configured values and warn on mismatch.
        checkpoint_decay = state.get("decay")
        if (
            checkpoint_decay is not None
            and float(checkpoint_decay) != float(self.decay)
        ):
            print(
                f"[EMA] checkpoint decay {checkpoint_decay} differs from "
                f"configured {self.decay}; keeping configured value"
            )
