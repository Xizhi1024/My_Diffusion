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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .slmf_bbdm import SLMFBBDM
from .ema import EMA


def _stripe_score(pred_np: np.ndarray) -> float:
    """Directional-gradient anisotropy: max directional energy / mean.

    High score → strong directional bias (stripes).  Isotropic texture → ~1.
    pred_np: [H, W] ndarray in model output space.
    """
    if pred_np.ndim == 3:
        pred_np = pred_np[0]
    # Sobel gradients
    kx = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    ky = kx.T
    from scipy.ndimage import correlate
    gx = correlate(pred_np, kx, mode="reflect")
    gy = correlate(pred_np, ky, mode="reflect")
    n_dirs = 8
    energies = np.empty(n_dirs, dtype=np.float64)
    for i in range(n_dirs):
        theta = i * np.pi / n_dirs
        dg = gx * np.cos(theta) + gy * np.sin(theta)
        energies[i] = float((dg * dg).sum())
    return float(energies.max() / max(energies.mean(), 1e-8))


def _to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


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

        # ---- Training monitoring: tracked samples, best ckpt, early stop ----
        bc_cfg = run_cfg.get("best_checkpoint", {}) or {}
        es_cfg = run_cfg.get("early_stopping", {}) or {}
        self.best_ckpts_enabled = bool(bc_cfg.get("enabled", True))
        self.best_combined_alpha = float(bc_cfg.get("combined_alpha", 0.5))
        self.best_stripe_penalty = float(bc_cfg.get("stripe_penalty", 0.3))

        self.early_stopping_enabled = bool(es_cfg.get("enabled", False))
        self.early_stopping_patience = int(es_cfg.get("patience", 40))
        self.early_stopping_min_epochs = int(es_cfg.get("min_epochs", 0))
        self._best_combined_score: float = -1e9
        self._best_lesion_score: float = -1e9
        self._best_image_score: float = -1e9
        self._epochs_since_improve: int = 0

        self.tracked_sample_ids: List[str] = list(run_cfg.get("tracked_sample_ids", []) or [])
        self._tracked_batch: Optional[Dict[str, Any]] = None
        self._tracked_meta: Optional[List[Dict[str, Any]]] = None

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

        # Backwards-compat: a few unit tests build Trainer via object.__new__
        # without calling __init__.  Ensure the monitoring attributes exist.
        if not hasattr(self, "_tracked_batch"):
            self._tracked_batch = None
            self._tracked_meta = None
            self.best_ckpts_enabled = False
            self.early_stopping_enabled = False

        print(f"\n{'='*60}")
        print(f"SLMF-BBDM Training")
        print(f"Device: {self.device} | AMP: {self.amp_dtype} | Compile: {self.torch_compile}")
        print(f"Trainable: {self.model.get_trainable_params():,} | Total: {self.model.get_total_params():,}")
        print(f"Grad Accum: {self.grad_accum} | Log interval: {self.log_interval}")
        print(f"Priors: {[n for n, p in self.model.priors.items() if p.enabled]}")
        print(f"Losses: {[n for n, loss in self.model.loss_terms.items() if loss.enabled]}")
        print(f"{'='*60}\n")

        while self.epoch_count < num_epochs:
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

            # Select fixed tracked samples once (lazy — val_loader must exist)
            if self._tracked_batch is None and self.val_loader is not None:
                with self.ema_scope():
                    pass  # no EMA needed for selection; keep symmetry
                self._select_tracked_batch()

            # Evaluation (with EMA) + sample-based monitoring + model selection
            should_stop = False
            if self.val_loader is not None and self.epoch_count % self.eval_interval == 0:
                with self.ema_scope():
                    eval_logs = [self.eval_step(b) for b in self.val_loader]
                k0 = list(eval_logs[0].keys())
                avg_eval = {k: sum(d.get(k, 0) for d in eval_logs) / len(eval_logs) for k in k0}
                print(f"  Eval  Loss: {avg_eval.get('loss/total', 0):.4f}")

                if self._tracked_batch is not None:
                    with self.ema_scope():
                        val_metrics = self._compute_val_sample_metrics(self._tracked_batch)
                    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
                    self._save_best_checkpoints(val_metrics)
                    _, _, combined = self._model_selection_scores(val_metrics)
                    if self._check_early_stopping(combined):
                        should_stop = True

            # Checkpoint
            if self.epoch_count % self.save_interval == 0:
                self.save_checkpoint()

            # Sampling — fixed tracked samples (not next(iter(val_loader)))
            if self._tracked_batch is not None and self.epoch_count % self.sample_interval == 0:
                with self.ema_scope():
                    sample_result = self.model.sample(_to_device(self._tracked_batch, self.device))
                synth_pet = sample_result["synthetic_pet"]
                print(f"  Sample PET range: [{synth_pet.min().item():.4f}, {synth_pet.max().item():.4f}]")
                self._save_sample_grid(self._tracked_batch, synth_pet)

            if should_stop:
                break

        print(f"\nTraining complete. Total: {time.time() - self.start_time:.0f}s")

    # ------------------------------------------------------------------
    # Validation & model selection
    # ------------------------------------------------------------------

    def _select_tracked_batch(self) -> None:
        """Scan val_loader once and cache a fixed batch of small/medium/large
        lesion samples for visual + metric tracking across epochs.

        If ``tracked_sample_ids`` is set in config, only those samples are kept.
        Otherwise we pick 2 small / 2 medium / 2 large (by mask area).
        """
        if self.val_loader is None:
            return
        candidates: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        wanted = set(self.tracked_sample_ids)
        for batch in self.val_loader:
            masks = batch.get("mask")
            if masks is None:
                continue
            B = masks.shape[0]
            for i in range(B):
                area = float(masks[i].sum().item())
                sample = {
                    k: (v[i:i + 1].clone() if torch.is_tensor(v) else ([v[i]] if isinstance(v, list) else v))
                    for k, v in batch.items()
                }
                # patient / sample id for display
                meta = batch.get("meta")
                pid = ""
                if isinstance(meta, list) and i < len(meta):
                    pid = str(meta[i].get("patient_id", "")) if isinstance(meta[i], dict) else ""
                elif isinstance(meta, dict):
                    mv = meta.get("patient_id")
                    if isinstance(mv, list) and i < len(mv):
                        pid = str(mv[i])
                sid = f"{pid or 's'}_{i}"
                if wanted and sid not in wanted and pid not in wanted:
                    continue
                candidates.append((area, sample, {"sample_id": sid, "patient_id": pid}))

        if not candidates:
            print("  [tracked samples] no candidates found; skipping fixed tracking")
            return

        candidates.sort(key=lambda c: c[0])
        n = len(candidates)
        if not self.tracked_sample_ids:
            idx_sets = [0, n // 3, 2 * n // 3]
            picks: List[int] = []
            for s in idx_sets:
                picks.extend(range(s, min(s + 2, n)))
            picks = sorted(set(picks))
        else:
            picks = list(range(n))

        merged: Dict[str, List[Any]] = {}
        meta_list: List[Dict[str, Any]] = []
        for p in picks:
            _, sample, meta = candidates[p]
            meta_list.append(meta)
            for k, v in sample.items():
                merged.setdefault(k, []).append(v)
        tracked: Dict[str, Any] = {}
        for k, vs in merged.items():
            if vs and torch.is_tensor(vs[0]):
                tracked[k] = torch.cat(vs, dim=0)
            elif vs and isinstance(vs[0], dict):
                tracked[k] = vs
            else:
                tracked[k] = vs
        self._tracked_batch = tracked
        self._tracked_meta = meta_list
        print(f"  [tracked samples] {len(meta_list)} fixed samples: "
              + ", ".join(m["sample_id"] for m in meta_list))

    @torch.no_grad()
    def _compute_val_sample_metrics(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Run sampling on a fixed batch and compute monitoring metrics."""
        self.model.eval()
        batch = _to_device(batch, self.device)
        sample_out = self.model.sample(batch)
        synth = sample_out["synthetic_pet"]  # [B,1,H,W]
        target = batch["pet"]
        mask = batch.get("mask")

        metrics: Dict[str, float] = {}
        mae_vals, ssim_vals, stripe_vals = [], [], []
        peak_err_vals, centroid_vals, oir_vals = [], [], []
        failure_count = 0
        B = synth.shape[0]
        for i in range(B):
            p = synth[i, 0].float().cpu().numpy()
            t = target[i, 0].float().cpu().numpy()
            mae_vals.append(float(np.abs(p - t).mean()))
            stripe_vals.append(_stripe_score(p))
            # SSIM (lazy import to avoid hard dep at module load)
            try:
                from skimage.metrics import structural_similarity as _ssim
                ssim_vals.append(float(_ssim(t, p, data_range=2.0)))
            except Exception:
                pass
            if mask is not None:
                mi = mask[i, 0].float().cpu().numpy()
                if mi.sum() > 0:
                    pred_in = float((p * mi).max())
                    tgt_in = float((t * mi).max())
                    peak_err_vals.append(abs(pred_in - tgt_in))
                    # centroid distance (pred peak vs mask centroid)
                    ys, xs = np.nonzero(mi)
                    if len(xs) > 0:
                        cy, cx = ys.mean(), xs.mean()
                        pm = p * mi
                        py, px = np.unravel_index(np.argmax(pm), pm.shape)
                        centroid_vals.append(float(np.hypot(py - cy, px - cx)))
                    # outside/inside peak ratio
                    outside = p * (1.0 - mi)
                    in_peak = max(pred_in, 1e-6)
                    out_peak = float(outside.max())
                    oir_vals.append(out_peak / in_peak)
                    if out_peak > pred_in:
                        failure_count += 1

        def _mean(vals):
            return float(np.mean(vals)) if vals else float("nan")

        metrics["val/mae"] = _mean(mae_vals)
        metrics["val/ssim"] = _mean(ssim_vals)
        metrics["val/stripe_score"] = _mean(stripe_vals)
        metrics["val/lesion_peak_error_norm"] = _mean(peak_err_vals)
        metrics["val/lesion_centroid_distance"] = _mean(centroid_vals)
        metrics["val/outside_inside_peak_ratio"] = _mean(oir_vals)
        metrics["val/failure_rate"] = float(failure_count) / max(B, 1)
        # lesion_roi_l1 (normalised) — dense PET supervision proxy inside mask
        if mask is not None:
            mi_all = mask.float()
            num = (synth - target).abs() * mi_all
            den = mi_all.sum().clamp_min(1.0)
            metrics["val/lesion_roi_l1"] = float(num.sum().item() / den.item())
        return metrics

    def _model_selection_scores(self, metrics: Dict[str, float]) -> Tuple[float, float, float]:
        """Return (lesion_score, image_score, combined) where higher is better.

        lesion_score rewards low peak error + low centroid distance + low failure.
        image_score rewards high SSIM + low MAE + low stripe.
        combined blends both with a stripe penalty.
        """
        def _g(k):
            v = metrics.get(k, float("nan"))
            return v if v == v else 0.0  # NaN→0

        peak = _g("val/lesion_peak_error_norm")
        centroid = _g("val/lesion_centroid_distance")
        fail = _g("val/failure_rate")
        mae = _g("val/mae")
        ssim = _g("val/ssim")
        stripe = _g("val/stripe_score")

        lesion_score = -(peak + 0.02 * centroid + fail)
        image_score = ssim - mae - 0.1 * max(stripe - 1.0, 0.0)
        combined = (
            self.best_combined_alpha * lesion_score
            + (1.0 - self.best_combined_alpha) * image_score
            - self.best_stripe_penalty * max(stripe - 1.0, 0.0)
        )
        return lesion_score, image_score, combined

    def _save_best_checkpoints(self, metrics: Dict[str, float]) -> None:
        if not self.best_ckpts_enabled:
            return
        lesion, image, combined = self._model_selection_scores(metrics)
        exp_name = self.config.get("experiment", {}).get("name", "slmf_bbdm")
        save_dir = os.path.join("checkpoints", exp_name)
        os.makedirs(save_dir, exist_ok=True)

        def _save(tag: str, score: float, best_key: str) -> None:
            best = getattr(self, best_key, -1e9)
            if score > best:
                setattr(self, best_key, score)
                path = os.path.join(save_dir, f"ckpt_{tag}.pt")
                torch.save({
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "ema": self.ema.state_dict(),
                    "epoch": self.epoch_count,
                    "step": self.step_count,
                    "score": score,
                    "metrics": metrics,
                    "config": self.config,
                }, path)
                print(f"  ★ new best {tag} (score={score:.4f}) → {path}")

        _save("best_lesion", lesion, "_best_lesion_score")
        _save("best_image", image, "_best_image_score")
        _save("best_combined", combined, "_best_combined_score")

    def _check_early_stopping(self, combined: float) -> bool:
        """Return True if training should stop."""
        if not self.early_stopping_enabled:
            return False
        if self.epoch_count < self.early_stopping_min_epochs:
            return False
        if combined > self._best_combined_score:
            self._best_combined_score = combined
            self._epochs_since_improve = 0
        else:
            self._epochs_since_improve += 1
            if self._epochs_since_improve >= self.early_stopping_patience:
                print(
                    f"  Early stopping: no improvement for "
                    f"{self._epochs_since_improve} epochs (patience="
                    f"{self.early_stopping_patience})."
                )
                return True
        return False

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
        torch.save(checkpoint, path)
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
            sid = ""
            if self._tracked_meta and i < len(self._tracked_meta):
                sid = self._tracked_meta[i].get("sample_id", "")
            axes[i, 0].imshow(batch["ct"][i, 0].cpu().numpy(), cmap="gray")
            axes[i, 0].set_title(f"CT\n{sid}" if sid else "CT")
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
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.ema.load_state_dict(checkpoint["ema"])
        self.epoch_count = checkpoint["epoch"]
        self.step_count = checkpoint["step"]
        self.accum_count = 0
        print(f"Loaded checkpoint from {path} (epoch {self.epoch_count})")
