"""Lightweight registry – no external dependency.

Every pluggable module type (Prior, Adapter, NoiseSchedule, LossTerm)
gets its own Registry instance.  Modules are registered by name and
instantiated by passing a config dict that includes a ``name`` key.
"""

from __future__ import annotations

from typing import Any, Callable, Dict


class RegistryError(Exception):
    pass


class Registry:
    """Typed module registry."""

    def __init__(self, kind: str = "module"):
        self._items: Dict[str, Callable[..., Any]] = {}
        self._kind = kind

    def register(self, name: str, cls: Callable[..., Any]) -> None:
        if name in self._items:
            raise RegistryError(f"{self._kind} '{name}' already registered")
        self._items[name] = cls

    def build(self, cfg: Dict[str, Any]) -> Any:
        """Instantiate a registered class from a config dict.

        ``cfg`` must contain a ``"name"`` key matching a registered entry.
        The remaining keys are forwarded as kwargs to the constructor.
        """
        cfg = dict(cfg)
        name = cfg.pop("name")
        if name not in self._items:
            raise RegistryError(
                f"Unknown {self._kind} '{name}'. Available: {list(self._items.keys())}"
            )
        return self._items[name](**cfg)

    def names(self):
        return list(self._items.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __repr__(self) -> str:
        return f"Registry({self._kind}, items={list(self._items.keys())})"


# Global registries
prior_registry = Registry("prior")
adapter_registry = Registry("adapter")
noise_registry = Registry("noise_schedule")
loss_registry = Registry("loss_term")
