"""Standalone training utilities for the low-frequency PET mean predictor."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .frequency.haar import haar_dwt2
from .mean_predictor import LowFrequencyPETPredictor


def mean_target_ll2(pet: torch.Tensor) -> torch.Tensor:
    """Return the level-2 orthonormal Haar low-pass PET coefficient."""
    ll1, _ = haar_dwt2(pet)
    ll2, _ = haar_dwt2(ll1)
    return ll2


def mean_charbonnier_loss(
    pred_ll2: torch.Tensor,
    target_ll2: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    if epsilon <= 0:
        raise ValueError("Charbonnier epsilon must be positive")
    return torch.sqrt((pred_ll2 - target_ll2).square() + epsilon ** 2).mean()


class MeanPretrainer:
    """Train and select only :class:`LowFrequencyPETPredictor`."""

    def __init__(
        self,
        predictor: LowFrequencyPETPredictor,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        if val_loader is None:
            raise ValueError("Conditional mean pretraining requires a validation loader")
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.predictor = predictor.to(self.device)

        mean_cfg = config.get("modules", {}).get("conditional_mean", {})
        self.mean_config = dict(mean_cfg)
        self.epsilon = float(mean_cfg.get("charbonnier_eps", 1e-3))
        training_cfg = config.get("training", {})
        self.learning_rate = float(training_cfg.get("learning_rate", 1e-4))
        self.lr_min = float(training_cfg.get("lr_min", 1e-6))
        weight_decay = float(training_cfg.get("weight_decay", 0.01))
        self.optimizer = torch.optim.AdamW(
            self.predictor.parameters(),
            lr=self.learning_rate,
            weight_decay=weight_decay,
        )
        run_cfg = config.get("runtime", {})
        self.grad_clip_norm = float(run_cfg.get("grad_clip_norm", 1.0))
        self.amp_enabled = bool(run_cfg.get("amp", True)) and self.device.type == "cuda"
        if self.amp_enabled and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.amp_enabled and self.amp_dtype == torch.float16,
        )

    def _batch_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        ct = batch["ct"].to(self.device, non_blocking=True)
        pet = batch["pet"].to(self.device, non_blocking=True)
        with torch.amp.autocast(
            self.device.type,
            enabled=self.amp_enabled,
            dtype=self.amp_dtype,
        ):
            pred_ll2 = self.predictor(ct)["ll2"]
            target_ll2 = mean_target_ll2(pet)
            return mean_charbonnier_loss(pred_ll2, target_ll2, self.epsilon)

    def train_epoch(self) -> float:
        self.predictor.train()
        total = 0.0
        count = 0
        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._batch_loss(batch)
            self.scaler.scale(loss).backward()
            if self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.predictor.parameters(), self.grad_clip_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += float(loss.detach().item())
            count += 1
        if count == 0:
            raise ValueError("Conditional mean training loader is empty")
        return total / count

    @torch.no_grad()
    def validate(self) -> float:
        self.predictor.eval()
        total = 0.0
        count = 0
        for batch in self.val_loader:
            loss = self._batch_loss(batch)
            total += float(loss.detach().item())
            count += 1
        if count == 0:
            raise ValueError("Conditional mean validation loader is empty")
        value = total / count
        if not math.isfinite(value):
            raise FloatingPointError(f"Non-finite conditional mean validation loss: {value}")
        return value

    def _checkpoint(self, epoch: int, val_loss: float) -> Dict[str, Any]:
        return {
            "format_version": 1,
            "model": self.predictor.state_dict(),
            "mean_config": self.mean_config,
            "epoch": int(epoch),
            "val_loss": float(val_loss),
        }

    def run(self, epochs: int, output_dir: str | Path) -> list[dict[str, float]]:
        if epochs < 1:
            raise ValueError("Conditional mean pretraining epochs must be positive")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=self.lr_min,
        )
        best_val = float("inf")
        history: list[dict[str, float]] = []
        for epoch_index in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            epoch = epoch_index + 1
            row = {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            }
            history.append(row)
            checkpoint = self._checkpoint(epoch, val_loss)
            torch.save(checkpoint, output_path / "mean_last.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(checkpoint, output_path / "mean_best.pt")
            scheduler.step()
            with (output_path / "history.json").open("w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2, ensure_ascii=False)
            print(
                f"Mean epoch {epoch:3d}/{epochs} | "
                f"train={train_loss:.6f} | val={val_loss:.6f}"
            )
        return history
