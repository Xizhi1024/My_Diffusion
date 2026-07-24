"""Training loop for SLMF-BBDM — performance-optimised.

Covers the "common speed killers" checklist:
  1. AMP bf16 (not fp16 — no grad scaler needed on H100/B200)
  2. torch.compile with reduce-overhead
  3. Flash Attention (sdpa or fa2) in CrossAttention blocks
  4. DataLoader with pin_memory + persistent_workers + prefetch_factor
  5. Batched logging — .item() only every log_interval steps, not every step
  6. channels_last + TF32 enabled by default
  7. Fused AdamW when available
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from src.data.lineage import (
    attach_data_lineage,
    load_checkpoint_data_lineage,
    validate_checkpoint_data_lineage,
)

from .slmf_bbdm import SLMFBBDM
from .ema import EMA


def _detect_dtype() -> torch.dtype:
    """Return best AMP dtype: bf16 > fp16 > fp32."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _fused_adamw(params, lr: float, wd: float) -> torch.optim.AdamW:
    """Use fused AdamW when available (much faster on CUDA)."""
    try:
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, fused=True)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


class Trainer:
    def __init__(
        self,
        model: SLMFBBDM,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[str] = None,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.data_lineage = load_checkpoint_data_lineage(config)

        run_cfg = config.get("runtime", {})
        self.amp = run_cfg.get("amp", True)
        self.amp_dtype = _detect_dtype() if self.amp else torch.float32
        self.channels_last = run_cfg.get("channels_last", True)
        self.torch_compile = run_cfg.get("torch_compile", False)
        self.grad_accum = run_cfg.get("gradient_accumulate_every", 2)
        self.grad_clip_norm = run_cfg.get("grad_clip_norm", 1.0)
        self.log_interval = run_cfg.get("log_interval", 10)
        self.eval_interval = run_cfg.get("eval_interval", 50)
        self.sample_interval = run_cfg.get("sample_interval", 50)
        self.save_interval = run_cfg.get("save_interval", 50)

        # Device + TF32
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Model setup
        if self.channels_last and device == "cuda":
            model = model.to(memory_format=torch.channels_last)
        if self.torch_compile and device == "cuda":
            model = torch.compile(model, mode="reduce-overhead")

        self.model = model.to(device)

        # Optimizer (fused if available)
        lr = config.get("training", {}).get("learning_rate", 1e-4)
        wd = config.get("training", {}).get("weight_decay", 0.01)
        self.optimizer = _fused_adamw(model.parameters(), lr=lr, wd=wd)

        # EMA
        ema_cfg = config.get("training", {}).get("ema", {})
        self.ema = EMA(
            model,
            decay=ema_cfg.get("decay", 0.999),
            update_every=ema_cfg.get("update_every", 10),
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("training", {}).get("num_epochs", 1000),
            eta_min=config.get("training", {}).get("lr_min", 1e-6),
        )

        self.step_count = 0
        self.accum_count = 0
        self.epoch_count = 0
        self.start_time = None

    def _apply_optimizer_step(self) -> None:
        """Apply one optimiser/EMA update and clear accumulated gradients."""
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.ema.update()
        self.accum_count = 0

    # ------------------------------------------------------------------
    # EMA context manager
    # ------------------------------------------------------------------

    @contextmanager
    def ema_scope(self):
        """Temporarily apply EMA weights, restore on exit."""
        self.ema.apply()
        try:
            yield
        finally:
            self.ema.restore()

    # ------------------------------------------------------------------
    # Training step (minimal Python overhead)
    # ------------------------------------------------------------------

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Optional[Dict[str, float]]:
        """One training step. Returns logs only every log_interval steps."""
        self.model.train()

        # Non-blocking host→device transfer
        batch = {
            k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        use_amp = self.amp and self.device == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=self.amp_dtype):
            loss, logs = self.model(batch)

        loss = loss / self.grad_accum
        loss.backward()
        self.accum_count += 1

        if self.accum_count == self.grad_accum:
            self._apply_optimizer_step()

        self.step_count += 1

        # Only detach + item() on logging interval
        if self.step_count % self.log_interval != 0:
            return None

        log_dict: Dict[str, float] = {}
        for k, v in logs.items():
            log_dict[k] = v.item() if torch.is_tensor(v) else float(v)
        log_dict["loss/total"] = loss.item() * self.grad_accum

        return log_dict

    @torch.no_grad()
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        batch = {
            k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        use_amp = self.amp and self.device == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=self.amp_dtype):
            loss, logs = self.model(batch)
        return {k: (v.item() if torch.is_tensor(v) else float(v)) for k, v in logs.items()}

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    def train_epoch(self) -> Dict[str, float]:
        epoch_start = time.time()
        epoch_logs: Dict[str, List[float]] = {}
        batch_count = 0

        for batch in self.train_loader:
            step_logs = self.train_step(batch)
            batch_count += 1
            if step_logs is not None:
                for k, v in step_logs.items():
                    epoch_logs.setdefault(k, []).append(v)

        if self.accum_count > 0:
            self._apply_optimizer_step()

        self.epoch_count += 1
        self.scheduler.step()

        if not epoch_logs:
            return {"perf/epoch_seconds": time.time() - epoch_start}

        avg_logs = {k: sum(v) / len(v) for k, v in epoch_logs.items()}
        avg_logs["perf/epoch_seconds"] = time.time() - epoch_start
        avg_logs["perf/lr"] = self.scheduler.get_last_lr()[0]

        if self.device == "cuda":
            avg_logs["perf/gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2
            torch.cuda.reset_peak_memory_stats()

        return avg_logs

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def run(self, num_epochs: Optional[int] = None) -> None:
        num_epochs = num_epochs or self.config.get("training", {}).get("num_epochs", 1000)
        self.start_time = time.time()

        print(f"\n{'='*60}")
        print(f"SLMF-BBDM Training")
        print(f"Device: {self.device} | AMP: {self.amp_dtype} | Compile: {self.torch_compile}")
        print(f"Trainable: {self.model.get_trainable_params():,} | Total: {self.model.get_total_params():,}")
        print(f"Grad Accum: {self.grad_accum} | Log interval: {self.log_interval}")
        print(f"Priors: {[n for n, p in self.model.priors.items() if p.enabled]}")
        print(f"Losses: {[n for n, loss in self.model.loss_terms.items() if loss.enabled]}")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            train_logs = self.train_epoch()

            total = train_logs.get("loss/total", 0)
            elapsed = time.time() - self.start_time
            epoch_s = train_logs.get("perf/epoch_seconds", 0)
            lr = train_logs.get("perf/lr", 0)
            print(
                f"Epoch {self.epoch_count:4d}/{num_epochs} | "
                f"Loss: {total:.4f} | "
                f"Time: {elapsed:.0f}s ({epoch_s:.1f}s/ep) | "
                f"LR: {lr:.2e}"
            )

            # Evaluation (with EMA)
            if self.val_loader is not None and self.epoch_count % self.eval_interval == 0:
                with self.ema_scope():
                    eval_logs = [self.eval_step(b) for b in self.val_loader]
                k0 = list(eval_logs[0].keys())
                avg_eval = {k: sum(d.get(k, 0) for d in eval_logs) / len(eval_logs) for k in k0}
                print(f"  Eval  Loss: {avg_eval.get('loss/total', 0):.4f}")

            # Checkpoint
            if self.epoch_count % self.save_interval == 0:
                self.save_checkpoint()

            # Sampling
            if self.val_loader is not None and self.epoch_count % self.sample_interval == 0:
                with self.ema_scope():
                    val_batch = next(iter(self.val_loader))
                    val_batch = {
                        k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                        for k, v in val_batch.items()
                    }
                    sample_result = self.model.sample(val_batch)
                synth_pet = sample_result["synthetic_pet"]
                print(f"  Sample PET range: [{synth_pet.min().item():.4f}, {synth_pet.max().item():.4f}]")
                self._save_sample_grid(val_batch, synth_pet)

        print(f"\nTraining complete. Total: {time.time() - self.start_time:.0f}s")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, tag: Optional[str] = None):
        exp_name = self.config.get("experiment", {}).get("name", "slmf_bbdm")
        save_dir = os.path.join("checkpoints", exp_name)
        os.makedirs(save_dir, exist_ok=True)
        fname = f"ckpt_epoch{self.epoch_count:04d}.pt" if tag is None else f"ckpt_{tag}.pt"
        path = os.path.join(save_dir, fname)
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "ema": self.ema.state_dict(),
            "epoch": self.epoch_count,
            "step": self.step_count,
            "config": self.config,
        }
        torch.save(
            attach_data_lineage(
                checkpoint, getattr(self, "data_lineage", None)
            ),
            path,
        )
        print(f"  Saved → {path}")

    def _save_sample_grid(self, batch: dict, synth_pet: torch.Tensor) -> None:
        """Save a CT / target PET / pred PET / mask grid for visual monitoring."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [sample grid skipped: matplotlib not available]")
            return

        exp_name = self.config.get("experiment", {}).get("name", "slmf_bbdm")
        sample_dir = os.path.join("outputs", "samples", exp_name)
        os.makedirs(sample_dir, exist_ok=True)

        n = min(4, synth_pet.shape[0])
        target_pet = batch.get("pet")
        has_mask = "mask" in batch
        cols = 4 if has_mask else 3
        fig, axes = plt.subplots(n, cols, figsize=(4 * cols, 4 * n))
        if n == 1:
            axes = axes[None, :]

        for i in range(n):
            axes[i, 0].imshow(batch["ct"][i, 0].cpu().numpy(), cmap="gray")
            axes[i, 0].set_title("CT")
            axes[i, 1].imshow(target_pet[i, 0].cpu().numpy(), cmap="hot", vmin=-1, vmax=1)
            axes[i, 1].set_title("Target PET")
            axes[i, 2].imshow(synth_pet[i, 0].cpu().numpy(), cmap="hot", vmin=-1, vmax=1)
            axes[i, 2].set_title("Pred PET")
            if has_mask:
                axes[i, 3].imshow(batch["mask"][i, 0].cpu().numpy(), cmap="gray")
                axes[i, 3].set_title("Lesion mask")
            for ax in axes[i]:
                ax.axis("off")

        fig.suptitle(f"Epoch {self.epoch_count}")
        fig.tight_layout()
        out_path = os.path.join(sample_dir, f"epoch_{self.epoch_count:04d}.png")
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        print(f"  Sample grid → {out_path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        validate_checkpoint_data_lineage(
            checkpoint,
            self.data_lineage,
            required=bool(
                self.config.get("data", {}).get("require_cache_lineage", False)
            ),
            context=f"resume checkpoint {path}",
        )
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.ema.load_state_dict(checkpoint["ema"])
        self.epoch_count = checkpoint["epoch"]
        self.step_count = checkpoint["step"]
        self.accum_count = 0
        print(f"Loaded checkpoint from {path} (epoch {self.epoch_count})")
