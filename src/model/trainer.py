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


def _to_unit_interval(array: np.ndarray) -> np.ndarray:
    """Convert model-space PET values from [-1, 1] to clipped [0, 1]."""
    return np.clip((array.astype(np.float32) + 1.0) * 0.5, 0.0, 1.0)


def _compute_pet_sample_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    topk_percent: float = 0.10,
    min_k: int = 3,
    max_k: int = 16,
    area_quantiles: Optional[tuple[float, float]] = None,
) -> Optional[Dict[str, float]]:
    """Compute lesion metrics without zero-filled masked-array artefacts.

    Args:
        area_quantiles: Optional (Q33, Q67) lesion-area thresholds in pixels.
            When provided, adds ``small_lesion_*`` metric variants.  Thresholds
            should be fixed once on the validation set ground-truth masks and
            never recomputed per experiment.
    """
    if not 0.0 < topk_percent <= 1.0:
        raise ValueError("topk_percent must be in (0, 1]")
    if min_k < 1:
        raise ValueError("min_k must be at least 1")
    if max_k < min_k:
        raise ValueError("max_k must be greater than or equal to min_k")

    valid = mask > 0.5
    if not np.any(valid):
        return None

    pred_unit = _to_unit_interval(pred)
    target_unit = _to_unit_interval(target)
    outside = ~valid
    pred_in_peak = float(pred_unit[valid].max())
    target_in_peak = float(target_unit[valid].max())
    out_peak = float(pred_unit[outside].max()) if np.any(outside) else 0.0
    lesion_size = int(valid.sum())
    topq_k = min(
        max(int(np.ceil(lesion_size * topk_percent)), min_k),
        max_k,
        lesion_size,
    )
    pred_values = pred_unit[valid]
    target_values = target_unit[valid]
    pred_topq = float(np.partition(pred_values, -topq_k)[-topq_k:].mean())
    target_topq = float(np.partition(target_values, -topq_k)[-topq_k:].mean())
    peak_bias = pred_in_peak - target_in_peak
    topq_bias = pred_topq - target_topq

    ys, xs = np.nonzero(valid)
    peak_index = int(np.argmax(pred_unit[valid]))
    py, px = float(ys[peak_index]), float(xs[peak_index])
    cy, cx = float(ys.mean()), float(xs.mean())

    return {
        "lesion_peak_error_norm": abs(pred_in_peak - target_in_peak),
        "lesion_peak_signed_bias_norm": peak_bias,
        "lesion_topq_peak_pred_norm": pred_topq,
        "lesion_topq_peak_target_norm": target_topq,
        "lesion_topq_peak_signed_bias_norm": topq_bias,
        "lesion_topq_peak_error_norm": abs(topq_bias),
        "lesion_peak_overestimated": float(peak_bias > 0.0),
        "lesion_peak_underestimated": float(peak_bias < 0.0),
        "lesion_topq_k": float(topq_k),
        "lesion_size": float(lesion_size),
        "lesion_centroid_distance": float(np.hypot(py - cy, px - cx)),
        "outside_inside_peak_ratio": out_peak / max(pred_in_peak, 1e-6),
        "failure": float(out_peak > pred_in_peak),
        "lesion_roi_l1": float(np.abs(pred_unit[valid] - target_unit[valid]).mean()),
        # Small-lesion stratification
        **_small_lesion_variants(
            topq_bias, peak_bias, out_peak, pred_in_peak,
            lesion_size, area_quantiles,
        ),
    }


def _small_lesion_variants(
    topq_bias: float,
    peak_bias: float,
    out_peak: float,
    pred_in_peak: float,
    lesion_size: int,
    area_quantiles: Optional[tuple[float, float]],
    underestimate_tolerance: float = 0.05,
) -> Dict[str, float]:
    """Compute small-lesion specific metrics when area_quantiles is set."""
    if area_quantiles is None:
        return {}
    q33, q67 = area_quantiles
    is_small = lesion_size <= q33
    is_medium = bool(q33 < lesion_size <= q67)
    is_large = lesion_size > q67

    return {
        "lesion_size_category": float(
            0.0 if is_small else (1.0 if is_medium else 2.0)
        ),
        "small_lesion_topq_peak_error_norm": (
            abs(topq_bias) if is_small else 0.0
        ),
        "small_lesion_signed_bias_norm": (
            topq_bias if is_small else 0.0
        ),
        "small_lesion_underestimate": (
            float(topq_bias < -underestimate_tolerance) if is_small else 0.0
        ),
        "small_lesion_failure": (
            float(out_peak > pred_in_peak) if is_small else 0.0
        ),
        "small_lesion_count": 1.0 if is_small else 0.0,
        "medium_lesion_count": 1.0 if is_medium else 0.0,
        "large_lesion_count": 1.0 if is_large else 0.0,
    }


def _stratified_indices(total: int, count: int) -> List[int]:
    """Return deterministic, near-quantile indices including both endpoints."""
    if total <= 0 or count <= 0:
        return []
    count = min(total, count)
    return np.rint(np.linspace(0, total - 1, num=count)).astype(int).tolist()


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


def _spectral_parameter_group(name: str) -> str:
    """Classify trainable parameters for spectral-router learning rates."""
    if "residual_preconditioner.projection_heads" in name or (
        "residual_preconditioner.l2_to_l1_projection" in name
    ):
        return "projection"
    if any(
        token in name
        for token in (
            "residual_preconditioner.route_heads",
            "residual_preconditioner.no_null_route_heads",
            "residual_preconditioner.amplitude_heads",
        )
    ):
        return "router"
    if (
        "residual_preconditioner.dct_descriptor" in name
        or "priors.gabor" in name
    ):
        return "descriptor"
    return "base"


def _build_optimizer(model, training_cfg: Dict[str, Any]) -> torch.optim.AdamW:
    base_lr = float(training_cfg.get("learning_rate", 1e-4))
    base_wd = float(training_cfg.get("weight_decay", 0.01))
    group_cfg = training_cfg.get("optimizer_groups", {}) or {}
    if not bool(group_cfg.get("enabled", False)):
        return _fused_adamw(model.parameters(), lr=base_lr, wd=base_wd)

    learning_rates = {
        "base": base_lr,
        "projection": float(group_cfg.get("projection_lr", 5e-4)),
        "router": float(group_cfg.get("router_lr", 5e-4)),
        "descriptor": float(group_cfg.get("descriptor_lr", 2e-4)),
    }
    split_no_decay = bool(group_cfg.get("no_decay_bias_and_offsets", True))
    buckets: Dict[tuple[str, bool], List[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        kind = _spectral_parameter_group(name)
        no_decay = split_no_decay and (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or "offset" in name
            or "logit" in name
        )
        buckets.setdefault((kind, no_decay), []).append(parameter)

    groups = []
    for (kind, no_decay), parameters in sorted(buckets.items()):
        groups.append({
            "params": parameters,
            "lr": learning_rates[kind],
            "weight_decay": 0.0 if no_decay else base_wd,
            "group_name": f"{kind}_{'no_decay' if no_decay else 'decay'}",
        })
    return _fused_adamw(groups, lr=base_lr, wd=base_wd)


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
        self.eval_num_samples = int(run_cfg.get("eval_num_samples", 16))
        self.eval_seed = int(
            run_cfg.get("eval_seed", config.get("experiment", {}).get("seed", 42))
        )
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

        # Optimizer (fused if available).  V6 can give the delayed spectral
        # branch a larger LR without changing legacy configurations.
        training_cfg = config.get("training", {})
        self.optimizer = _build_optimizer(model, training_cfg)

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
        self.gradient_diagnostics = bool(
            run_cfg.get("gradient_diagnostics", False)
        )
        self._last_gradient_logs: Dict[str, torch.Tensor] = {}

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
        self._last_combined_improvement_epoch: Optional[int] = None

        self.tracked_sample_ids: List[str] = list(run_cfg.get("tracked_sample_ids", []) or [])
        self._tracked_batch: Optional[Dict[str, Any]] = None
        self._tracked_meta: Optional[List[Dict[str, Any]]] = None

    def _apply_optimizer_step(self) -> None:
        """Apply one optimiser/EMA update and clear accumulated gradients."""
        if getattr(self, "gradient_diagnostics", False):
            self._last_gradient_logs = self._collect_spectral_gradient_logs()
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.ema.update()
        self.accum_count = 0

    def _collect_spectral_gradient_logs(self) -> Dict[str, torch.Tensor]:
        groups = {
            "grad/projection_final": ((
                "residual_preconditioner.projection_heads",
                "residual_preconditioner.l2_to_l1_projection",
            ), True),
            "grad/route_final": ((
                "residual_preconditioner.route_heads",
                "residual_preconditioner.no_null_route_heads",
            ), True),
            "grad/amplitude_final": ((
                "residual_preconditioner.amplitude_heads",
            ), True),
            "grad/descriptor": ((
                "residual_preconditioner.dct_descriptor",
                "priors.gabor",
            ), False),
        }
        logs: Dict[str, torch.Tensor] = {}
        named_parameters = list(self.model.named_parameters())
        for key, (tokens, final_only) in groups.items():
            total = None
            count = 0
            for name, parameter in named_parameters:
                if (
                    parameter.grad is None
                    or not any(token in name for token in tokens)
                    or (final_only and ".final." not in name)
                ):
                    continue
                grad = parameter.grad.detach().float()
                value = grad.square().sum()
                total = value if total is None else total + value
                count += grad.numel()
            if total is not None and count > 0:
                logs[key] = (total / count).sqrt().detach()
        return logs

    def _set_spectral_router_epoch(self) -> None:
        model = getattr(self.model, "_orig_mod", self.model)
        preconditioner = getattr(model, "residual_preconditioner", None)
        setter = getattr(preconditioner, "set_training_epoch", None)
        if callable(setter):
            setter(self.epoch_count)

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
        for k, v in getattr(self, "_last_gradient_logs", {}).items():
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
        self._set_spectral_router_epoch()
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
            tracked_sample_result = None

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
            self._initialize_tracked_batch()

            # Evaluation (with EMA) + sample-based monitoring + model selection
            should_stop = False
            if self.val_loader is not None and self.epoch_count % self.eval_interval == 0:
                val_metrics = None
                combined_improved = False
                with self.ema_scope(), self._eval_rng_scope():
                    eval_logs = [self.eval_step(b) for b in self.val_loader]
                    if self._tracked_batch is not None:
                        tracked_sample_result = self._sample_with_eval_seed(self._tracked_batch)
                        val_metrics = self._compute_val_sample_metrics(
                            self._tracked_batch,
                            tracked_sample_result["synthetic_pet"],
                        )
                        # Scores use EMA weights, so best checkpoints must store them too.
                        combined_improved = self._save_best_checkpoints(val_metrics)
                k0 = list(eval_logs[0].keys())
                avg_eval = {k: sum(d.get(k, 0) for d in eval_logs) / len(eval_logs) for k in k0}
                print(f"  Eval  Loss: {avg_eval.get('loss/total', 0):.4f}")
                print(
                    "  Eval components: "
                    f"base={avg_eval.get('loss/base_diffusion', float('nan')):.4f}  "
                    f"roi={avg_eval.get('loss/lesion_roi_l1/loss', float('nan')):.4f}  "
                    f"topk={avg_eval.get('loss/topk_lesion/loss', float('nan')):.4f}  "
                    f"ranking={avg_eval.get('loss/outside_peak_ranking/loss', float('nan')):.4f}"
                )

                if val_metrics is not None:
                    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
                    if self._check_early_stopping(combined_improved):
                        should_stop = True

            # Checkpoint
            if self.epoch_count % self.save_interval == 0:
                self.save_checkpoint()

            # Sampling — fixed tracked samples (not next(iter(val_loader)))
            if self._tracked_batch is not None and self.epoch_count % self.sample_interval == 0:
                if tracked_sample_result is None:
                    with self.ema_scope():
                        tracked_sample_result = self._sample_with_eval_seed(self._tracked_batch)
                synth_pet = tracked_sample_result["synthetic_pet"]
                print(f"  Sample PET range: [{synth_pet.min().item():.4f}, {synth_pet.max().item():.4f}]")
                self._save_sample_grid(self._tracked_batch, synth_pet)

            if should_stop:
                break

        print(f"\nTraining complete. Total: {time.time() - self.start_time:.0f}s")

    # ------------------------------------------------------------------
    # Validation & model selection
    # ------------------------------------------------------------------

    def _initialize_tracked_batch(self) -> None:
        """Lazily select validation samples without advancing training RNG."""
        if self._tracked_batch is not None or self.val_loader is None:
            return
        with self._eval_rng_scope():
            self._select_tracked_batch()

    def _select_tracked_batch(self) -> None:
        """Scan val_loader once and cache a fixed lesion-area-stratified batch.

        If ``tracked_sample_ids`` is set in config, only those samples are kept.
        Otherwise we pick ``eval_num_samples`` lesion-area quantiles.
        """
        if self.val_loader is None:
            return
        candidates: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        wanted = set(self.tracked_sample_ids)
        global_index = 0
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
                # Patient / sample ID for display and explicit sample tracking.
                meta = batch.get("meta")
                sample_meta: Dict[str, Any] = {}
                if isinstance(meta, list) and i < len(meta):
                    if isinstance(meta[i], dict):
                        sample_meta = meta[i]
                elif isinstance(meta, dict):
                    for key, value in meta.items():
                        if torch.is_tensor(value):
                            if value.ndim == 0:
                                sample_meta[key] = value.item()
                            elif i < len(value):
                                item = value[i]
                                sample_meta[key] = item.item() if item.numel() == 1 else item
                        elif isinstance(value, (list, tuple)) and i < len(value):
                            sample_meta[key] = value[i]
                        else:
                            sample_meta[key] = value
                patient_value = sample_meta.get("patient_id")
                pid = "" if patient_value is None else str(patient_value)
                source_value = next(
                    (
                        sample_meta[key]
                        for key in ("sample_id", "slice_id", "filename")
                        if sample_meta.get(key) is not None
                        and str(sample_meta.get(key)) != ""
                    ),
                    global_index,
                )
                source_id = str(source_value)
                sid = f"{pid}_{source_id}" if pid else f"s_{source_id}"
                global_index += 1
                if wanted and sid not in wanted and pid not in wanted:
                    continue
                candidates.append((area, sample, {"sample_id": sid, "patient_id": pid}))

        if not candidates:
            print("  [tracked samples] no candidates found; skipping fixed tracking")
            return

        candidates.sort(key=lambda c: c[0])
        n = len(candidates)
        if not self.tracked_sample_ids:
            picks = _stratified_indices(n, self.eval_num_samples)
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
        print(f"  [fixed validation samples] {len(meta_list)} samples: "
              + ", ".join(m["sample_id"] for m in meta_list))

    @contextmanager
    def _eval_rng_scope(self):
        """Isolate every validation random draw from the training RNG state."""
        cuda_devices: List[int] = []
        device = torch.device(self.device)
        if device.type == "cuda":
            cuda_devices = [
                device.index if device.index is not None else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.eval_seed)
            yield

    @torch.no_grad()
    def _sample_with_eval_seed(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Sample with fixed noise without advancing the training RNG state."""
        with self._eval_rng_scope():
            return self.model.sample(_to_device(batch, self.device))

    @torch.no_grad()
    def _compute_val_sample_metrics(
        self,
        batch: Dict[str, Any],
        synth: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Run sampling on a fixed batch and compute monitoring metrics."""
        self.model.eval()
        batch = _to_device(batch, self.device)
        if synth is None:
            synth = self._sample_with_eval_seed(batch)["synthetic_pet"]
        target = batch["pet"]
        mask = batch.get("mask")

        metrics: Dict[str, float] = {}
        mae_vals, ssim_vals, stripe_vals = [], [], []
        peak_err_vals, topq_peak_err_vals = [], []
        centroid_vals, oir_vals, roi_l1_vals = [], [], []
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
                lesion_metrics = _compute_pet_sample_metrics(p, t, mi)
                if lesion_metrics is not None:
                    peak_err_vals.append(lesion_metrics["lesion_peak_error_norm"])
                    topq_peak_err_vals.append(
                        lesion_metrics["lesion_topq_peak_error_norm"]
                    )
                    centroid_vals.append(lesion_metrics["lesion_centroid_distance"])
                    oir_vals.append(lesion_metrics["outside_inside_peak_ratio"])
                    roi_l1_vals.append(lesion_metrics["lesion_roi_l1"])
                    failure_count += int(lesion_metrics["failure"])

        def _mean(vals):
            return float(np.mean(vals)) if vals else float("nan")

        metrics["val/mae"] = _mean(mae_vals)
        metrics["val/ssim"] = _mean(ssim_vals)
        metrics["val/stripe_score"] = _mean(stripe_vals)
        metrics["val/lesion_peak_error_norm"] = _mean(peak_err_vals)
        metrics["val/lesion_topq_peak_error_norm"] = _mean(topq_peak_err_vals)
        metrics["val/lesion_centroid_distance"] = _mean(centroid_vals)
        metrics["val/outside_inside_peak_ratio"] = _mean(oir_vals)
        metrics["val/failure_rate"] = float(failure_count) / max(len(peak_err_vals), 1)
        metrics["val/lesion_roi_l1"] = _mean(roi_l1_vals)
        metrics["val/lesion_sample_count"] = float(len(peak_err_vals))
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
        topq = metrics.get("val/lesion_topq_peak_error_norm", peak)
        topq = topq if topq == topq else peak
        centroid = _g("val/lesion_centroid_distance")
        fail = _g("val/failure_rate")
        mae = _g("val/mae")
        ssim = _g("val/ssim")
        stripe = _g("val/stripe_score")

        lesion_score = -(topq + 0.5 * peak + 0.02 * centroid + 0.5 * fail)
        image_score = ssim - mae - 0.1 * max(stripe - 1.0, 0.0)
        combined = (
            self.best_combined_alpha * lesion_score
            + (1.0 - self.best_combined_alpha) * image_score
            - self.best_stripe_penalty * max(stripe - 1.0, 0.0)
        )
        return lesion_score, image_score, combined

    def _monitoring_state_dict(self) -> Dict[str, Any]:
        """Return checkpointable model-selection and early-stop state."""
        return {
            "best_combined_score": self._best_combined_score,
            "best_lesion_score": self._best_lesion_score,
            "best_image_score": self._best_image_score,
            "epochs_since_improve": self._epochs_since_improve,
            "last_combined_improvement_epoch": self._last_combined_improvement_epoch,
        }

    def _load_monitoring_state(self, checkpoint: Dict[str, Any]) -> None:
        """Restore monitoring state when present; accept older checkpoints."""
        state = checkpoint.get("monitoring")
        if not isinstance(state, dict):
            return
        self._best_combined_score = float(
            state.get("best_combined_score", self._best_combined_score)
        )
        self._best_lesion_score = float(
            state.get("best_lesion_score", self._best_lesion_score)
        )
        self._best_image_score = float(
            state.get("best_image_score", self._best_image_score)
        )
        self._epochs_since_improve = int(
            state.get("epochs_since_improve", self._epochs_since_improve)
        )
        last_epoch = state.get(
            "last_combined_improvement_epoch",
            self._last_combined_improvement_epoch,
        )
        self._last_combined_improvement_epoch = (
            int(last_epoch) if last_epoch is not None else None
        )

    def _save_best_checkpoints(self, metrics: Dict[str, float]) -> bool:
        lesion, image, combined = self._model_selection_scores(metrics)
        improvements = {
            "best_lesion": lesion > self._best_lesion_score,
            "best_image": image > self._best_image_score,
            "best_combined": combined > self._best_combined_score,
        }
        combined_improved = improvements["best_combined"]
        if combined_improved:
            self._last_combined_improvement_epoch = self.epoch_count
            self._epochs_since_improve = 0
        elif self._last_combined_improvement_epoch is not None:
            self._epochs_since_improve = (
                self.epoch_count - self._last_combined_improvement_epoch
            )

        # Update all scores before writing any file so every checkpoint saved
        # in this evaluation carries the same complete monitoring snapshot.
        if improvements["best_lesion"]:
            self._best_lesion_score = lesion
        if improvements["best_image"]:
            self._best_image_score = image
        if combined_improved:
            self._best_combined_score = combined

        save_dir = ""
        if self.best_ckpts_enabled:
            exp_name = self.config.get("experiment", {}).get("name", "slmf_bbdm")
            save_dir = os.path.join("checkpoints", exp_name)
            os.makedirs(save_dir, exist_ok=True)

        def _save(tag: str, score: float) -> None:
            if not self.best_ckpts_enabled or not improvements[tag]:
                return
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
                "monitoring": self._monitoring_state_dict(),
            }, path)
            print(f"  ★ new best {tag} (score={score:.4f}) → {path}")

        _save("best_lesion", lesion)
        _save("best_image", image)
        _save("best_combined", combined)
        return combined_improved

    def _check_early_stopping(self, improved: bool) -> bool:
        """Return True if training should stop."""
        if not self.early_stopping_enabled:
            return False
        if improved or self._last_combined_improvement_epoch is None:
            self._last_combined_improvement_epoch = self.epoch_count
            self._epochs_since_improve = 0
            return False

        self._epochs_since_improve = self.epoch_count - self._last_combined_improvement_epoch
        if self.epoch_count < self.early_stopping_min_epochs:
            return False
        if self._epochs_since_improve >= self.early_stopping_patience:
            print(
                f"  Early stopping: no combined-score improvement for "
                f"{self._epochs_since_improve} epochs "
                f"(patience={self.early_stopping_patience})."
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
            "monitoring": self._monitoring_state_dict(),
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
        self._load_monitoring_state(checkpoint)
        self.accum_count = 0
        self._set_spectral_router_epoch()
        print(f"Loaded checkpoint from {path} (epoch {self.epoch_count})")
