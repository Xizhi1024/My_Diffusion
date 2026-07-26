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

import copy
import json
import math
import os
import random
import re
import time
from contextlib import contextmanager
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.lineage import (
    attach_data_lineage,
    load_checkpoint_data_lineage,
    validate_checkpoint_data_lineage,
)

from .slmf_bbdm import SLMFBBDM
from .ema import EMA


_PRIOR_ANCHORED_PIPELINE_ID = "PRIOR_ANCHORED_ROUTER_100E_CLOUD_V1"
_NUMBERED_CHECKPOINT_NAME = re.compile(r"^ckpt_epoch\d{4}\.pt$")


def _strict_repo_relative_path(value: Any, *, label: str) -> str:
    """Validate one persisted path without resolving it against local state."""
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty repository-relative path")
    if "\\" in value:
        raise RuntimeError(f"{label} must use repository-relative POSIX syntax")
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise RuntimeError(f"{label} must be repository-relative")
    if PureWindowsPath(value).drive:
        raise RuntimeError(f"{label} must not contain a Windows drive")
    segments = value.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        raise RuntimeError(
            f"{label} must not contain empty, '.' or '..' path segments"
        )
    return PurePosixPath(value).as_posix()


def _validate_prior_anchored_resume_identity(
    *,
    current_config: Mapping[str, Any],
    checkpoint_config: Any,
    checkpoint_path: str,
) -> None:
    """Fail closed for the run-owned exploratory 100-epoch pipeline."""
    prior_run = current_config.get("prior_anchored_run")
    if (
        not isinstance(prior_run, Mapping)
        or prior_run.get("pipeline_id") != _PRIOR_ANCHORED_PIPELINE_ID
    ):
        return
    if not isinstance(checkpoint_config, Mapping):
        raise RuntimeError("Resume checkpoint config is not a mapping")

    current_training = current_config.get("training")
    observed_training = checkpoint_config.get("training")
    if not isinstance(current_training, Mapping) or not isinstance(
        observed_training, Mapping
    ):
        raise RuntimeError(
            "Prior-anchored resume requires training config in both configs"
        )

    checkpoint_dir = _strict_repo_relative_path(
        current_training.get("checkpoint_dir"),
        label="training.checkpoint_dir",
    )

    def _validate_pointer(value: Any, *, label: str, allow_null: bool) -> Optional[str]:
        if value is None:
            if allow_null:
                return None
            raise RuntimeError(f"{label} must name the checkpoint being resumed")
        normalized = _strict_repo_relative_path(value, label=label)
        pointer = PurePosixPath(normalized)
        if (
            pointer.parent.as_posix() != checkpoint_dir
            or _NUMBERED_CHECKPOINT_NAME.fullmatch(pointer.name) is None
        ):
            raise RuntimeError(
                f"{label} must name ckpt_epochNNNN.pt inside the current "
                "training.checkpoint_dir"
            )
        return normalized

    current_pointer = _validate_pointer(
        current_training.get("resume_from"),
        label="current training.resume_from",
        allow_null=False,
    )
    _validate_pointer(
        observed_training.get("resume_from"),
        label="checkpoint training.resume_from",
        allow_null=True,
    )
    actual_pointer = _validate_pointer(
        checkpoint_path,
        label="load_checkpoint path",
        allow_null=False,
    )
    if actual_pointer != current_pointer:
        raise RuntimeError(
            "load_checkpoint path differs from current training.resume_from"
        )

    current_identity = copy.deepcopy(dict(current_config))
    observed_identity = copy.deepcopy(dict(checkpoint_config))
    for identity in (current_identity, observed_identity):
        training = identity.get("training")
        if isinstance(training, dict):
            training["resume_from"] = None
    if observed_identity != current_identity:
        raise RuntimeError(
            "Resume checkpoint configuration differs from the current "
            "prior-anchored run config"
        )


def resolve_checkpoint_dir(config: Dict[str, Any]) -> str:
    """Return the configured checkpoint directory without changing legacy runs.

    Strict audit runners can set ``training.checkpoint_dir`` to keep every
    mutable training artifact inside a run-owned directory.  Existing configs
    continue to use ``checkpoints/<experiment.name>``.
    """
    explicit = config.get("training", {}).get("checkpoint_dir")
    if explicit is not None:
        explicit = os.fspath(explicit).strip()
        if not explicit:
            raise ValueError("training.checkpoint_dir must not be empty")
        return explicit
    exp_name = config.get("experiment", {}).get("name", "slmf_bbdm")
    return os.path.join("checkpoints", exp_name)


def resolve_sample_dir(config: Dict[str, Any]) -> str:
    """Return a run-owned sample directory when one is explicitly configured."""
    explicit = config.get("runtime", {}).get("sample_dir")
    if explicit is not None:
        explicit = os.fspath(explicit).strip()
        if not explicit:
            raise ValueError("runtime.sample_dir must not be empty")
        return explicit
    exp_name = config.get("experiment", {}).get("name", "slmf_bbdm")
    return os.path.join("outputs", "samples", exp_name)


def _atomic_torch_save(payload: Dict[str, Any], path: str) -> None:
    """Write a torch payload without exposing a partially written checkpoint."""
    temporary = path + ".tmp"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass
        raise


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


def _chunk_batch(batch: Mapping[str, Any], chunk_size: int) -> List[Dict[str, Any]]:
    """Split a tracked-sample batch into chunks no larger than ``chunk_size``.

    Tensor values are sliced along dim 0; list values are sliced element-wise;
    scalar / other values are replicated unchanged so each chunk is a valid
    ``model.sample`` input.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    batch_size = 0
    for value in batch.values():
        if torch.is_tensor(value) and value.ndim >= 1:
            batch_size = max(batch_size, int(value.shape[0]))
        elif isinstance(value, list) and value:
            batch_size = max(batch_size, len(value))
    if batch_size == 0:
        return [dict(batch)]
    chunks: List[Dict[str, Any]] = []
    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        chunk: Dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == batch_size:
                chunk[key] = value[start:end]
            elif isinstance(value, list) and len(value) == batch_size:
                chunk[key] = value[start:end]
            else:
                chunk[key] = value
        chunks.append(chunk)
    return chunks


def _cat_sampled(key: str, parts: List[Dict[str, Any]]) -> Any:
    """Concatenate a sampled-output key across chunks, preserving list values."""
    values = [part[key] for part in parts]
    tensors = [v for v in values if torch.is_tensor(v)]
    if len(tensors) == len(values):
        return torch.cat(tensors, dim=0)
    # Mixed / list outputs: fall back to a flat list.
    merged: List[Any] = []
    for value in values:
        if isinstance(value, list):
            merged.extend(value)
        else:
            merged.append(value)
    return merged


def _sampled_output_to_cpu(output: Dict[str, Any]) -> Dict[str, Any]:
    """Detach sampled tensors and release each GPU chunk immediately."""
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in output.items()
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
            "residual_preconditioner.prior_active_heads",
            "residual_preconditioner.prior_destination_heads",
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
        self.eval_num_samples = int(run_cfg.get("eval_num_samples", 16))
        self.eval_seed = int(
            run_cfg.get("eval_seed", config.get("experiment", {}).get("seed", 42))
        )
        # Chunked validation sampling: the tracked-sample batch (up to
        # eval_num_samples) is fed to model.sample in chunks no larger than
        # eval_sample_batch_size so peak GPU memory stays bounded.  Default
        # caps at the configured validation batch size.
        val_batch_size = int(config.get("data", {}).get("val_batch_size", 1))
        default_chunk = max(1, min(val_batch_size, self.eval_num_samples))
        requested_chunk = int(run_cfg.get("eval_sample_batch_size", default_chunk))
        self.eval_sample_batch_size = max(1, min(requested_chunk, self.eval_num_samples))
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
        self._resumed_elapsed = 0.0
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
        self.metrics_jsonl = os.fspath(
            run_cfg.get(
                "metrics_jsonl",
                run_cfg.get(
                    "training_metrics_jsonl",
                    os.path.join(
                        resolve_checkpoint_dir(config),
                        "training_metrics.jsonl",
                    ),
                ),
            )
        )

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
                "residual_preconditioner.prior_active_heads",
                "residual_preconditioner.prior_destination_heads",
            ), True),
            "grad/prior_active_final": ((
                "residual_preconditioner.prior_active_heads",
            ), True),
            "grad/prior_destination_final": ((
                "residual_preconditioner.prior_destination_heads",
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

    def _set_data_epoch(self) -> None:
        """Stamp the next training epoch on an epoch-aware sampler."""
        sampler = getattr(self.train_loader, "sampler", None)
        setter = getattr(sampler, "set_epoch", None)
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
        self._set_data_epoch()
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

    @staticmethod
    def _finite_metric_mapping(
        values: Optional[Dict[str, float]],
    ) -> Optional[Dict[str, Optional[float]]]:
        if values is None:
            return None
        normalized: Dict[str, Optional[float]] = {}
        for key, value in values.items():
            numeric = float(value)
            normalized[key] = numeric if math.isfinite(numeric) else None
        return normalized

    def _router_phase_label(self) -> Optional[str]:
        """Return the configured router phase label for the current epoch, if any."""
        model = getattr(self.model, "_orig_mod", self.model)
        for attr in ("residual_preconditioner", "preconditioner"):
            preconditioner = getattr(model, attr, None)
            label_fn = getattr(preconditioner, "phase_label", None)
            if callable(label_fn):
                # epoch_count is 1-based (number of completed epochs); the
                # router expects the 0-based index of the epoch just trained.
                return str(label_fn(max(self.epoch_count - 1, 0)))
        return None

    def _append_epoch_metrics(
        self,
        *,
        train_logs: Dict[str, float],
        eval_logs: Optional[Dict[str, float]],
        validation_metrics: Optional[Dict[str, float]],
    ) -> None:
        """Atomically commit exactly one monitoring record for this epoch."""
        metrics_path = os.path.normpath(self.metrics_jsonl)
        parent = os.path.dirname(metrics_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        record: Dict[str, Any] = {
            "schema_version": 1,
            "epoch": int(self.epoch_count),
            "step": int(self.step_count),
            # phase is stamped at the top level so the summarizer does not
            # need to re-derive it from the epoch number.
            "phase": self._router_phase_label(),
            "train": self._finite_metric_mapping(train_logs),
            "eval": self._finite_metric_mapping(eval_logs),
            "validation": self._finite_metric_mapping(validation_metrics),
        }
        existing = self._validated_metrics_jsonl(metrics_path)
        expected = list(range(1, int(self.epoch_count)))
        observed = [epoch for epoch, _ in existing]
        if observed != expected:
            raise RuntimeError(
                "Metrics JSONL is not an exact completed-epoch prefix before "
                f"writing epoch {self.epoch_count}: expected {expected}, "
                f"found {observed}"
            )

        serialized = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        temporary = metrics_path + ".tmp"
        with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
            for _, line in existing:
                handle.write(line)
                handle.write("\n")
            handle.write(
                serialized
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, metrics_path)

    def _validated_metrics_jsonl(
        self,
        path: str,
    ) -> List[Tuple[int, str]]:
        """Parse JSONL strictly and return ordered ``(epoch, line)`` rows.

        Corrupt, duplicate, non-positive, out-of-order, and configured-range
        violations are errors.  Resume cleanup may remove a valid future tail,
        but it must never hide corruption.
        """
        if not os.path.exists(path):
            return []
        configured_max = None
        config = getattr(self, "config", None)
        if isinstance(config, dict):
            raw_max = config.get("training", {}).get("num_epochs")
            if raw_max is not None:
                configured_max = int(raw_max)

        rows: List[Tuple[int, str]] = []
        previous_epoch = 0
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                stripped = raw.strip()
                if not stripped:
                    raise RuntimeError(
                        f"Metrics JSONL contains a blank line at {line_number}"
                    )
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "Metrics JSONL contains invalid JSON at line "
                        f"{line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise RuntimeError(
                        "Metrics JSONL record must be an object at line "
                        f"{line_number}"
                    )
                epoch = payload.get("epoch")
                if isinstance(epoch, bool) or not isinstance(epoch, int):
                    raise RuntimeError(
                        "Metrics JSONL epoch must be an integer at line "
                        f"{line_number}"
                    )
                if epoch < 1:
                    raise RuntimeError(
                        f"Metrics JSONL epoch must be >= 1, found {epoch}"
                    )
                if configured_max is not None and epoch > configured_max:
                    raise RuntimeError(
                        "Metrics JSONL epoch exceeds configured training range: "
                        f"{epoch} > {configured_max}"
                    )
                if epoch <= previous_epoch:
                    detail = "duplicate" if epoch == previous_epoch else "out of order"
                    raise RuntimeError(
                        f"Metrics JSONL contains {detail} epoch {epoch}"
                    )
                previous_epoch = epoch
                rows.append((epoch, stripped))
        return rows

    def _truncate_metrics_jsonl_to_epoch(self) -> None:
        """Drop a valid future tail and require an exact checkpoint prefix."""
        path = getattr(self, "metrics_jsonl", "")
        if not path:
            if self.epoch_count > 0:
                raise RuntimeError(
                    "Cannot resume: checkpoint has completed epochs but no "
                    "metrics JSONL path is configured"
                )
            return

        normalized = os.path.normpath(path)
        rows = self._validated_metrics_jsonl(normalized)
        kept = [(epoch, line) for epoch, line in rows if epoch <= self.epoch_count]
        observed = [epoch for epoch, _ in kept]
        expected = list(range(1, int(self.epoch_count) + 1))
        if observed != expected:
            raise RuntimeError(
                "Cannot resume: metrics JSONL does not exactly cover the "
                f"checkpoint prefix 1..{self.epoch_count}; found {observed}"
            )

        dropped = len(rows) - len(kept)
        if dropped == 0:
            return
        temporary = normalized + ".tmp"
        with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
            for _, line in kept:
                handle.write(line)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, normalized)
        print(
            "  Resumed metrics JSONL truncated to epoch "
            f"{self.epoch_count} (dropped {dropped} valid future "
            f"record{'s' if dropped != 1 else ''})."
        )


    @staticmethod
    def _print_prior_anchor_summary(train_logs: Dict[str, float]) -> None:
        active = train_logs.get("frequency/prior_anchor_active_progress")
        if active is None:
            return
        destination = train_logs.get(
            "frequency/prior_anchor_destination_progress",
            float("nan"),
        )
        prior_mae = train_logs.get(
            "frequency/prior_anchor_prior_active_mae",
            float("nan"),
        )
        shallow = train_logs.get(
            "frequency/prior_anchor_shallow_mass",
            float("nan"),
        )
        monotonic = train_logs.get(
            "frequency/prior_anchor_monotonic_violation",
            float("nan"),
        )
        print(
            "  Router "
            f"active_release={active:.2f} "
            f"destination_release={destination:.2f} "
            f"prior_mae={prior_mae:.4f} "
            f"shallow={shallow:.4f} "
            f"mono={monotonic:.6f}"
        )

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def run(self, num_epochs: Optional[int] = None) -> None:
        num_epochs = num_epochs or self.config.get("training", {}).get("num_epochs", 1000)
        self.start_time = time.time()
        monitoring_contract_enabled = hasattr(self, "metrics_jsonl")

        # Backwards-compat: a few unit tests build Trainer via object.__new__
        # without calling __init__.  Ensure the monitoring attributes exist.
        if not hasattr(self, "_tracked_batch"):
            self._tracked_batch = None
            self._tracked_meta = None
            self.best_ckpts_enabled = False
            self.early_stopping_enabled = False
        if not hasattr(self, "metrics_jsonl"):
            self.metrics_jsonl = os.path.join(
                resolve_checkpoint_dir(self.config),
                "training_metrics.jsonl",
            )
        # Monitoring records carry step_count; ensure it exists for the
        # object.__new__ path used by a few unit tests.
        if not hasattr(self, "step_count"):
            self.step_count = 0
        if not hasattr(self, "_resumed_elapsed"):
            self._resumed_elapsed = 0.0

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
            avg_eval = None
            val_metrics = None

            total = train_logs.get("loss/total", 0)
            elapsed = time.time() - self.start_time + self._resumed_elapsed
            epoch_s = train_logs.get("perf/epoch_seconds", 0)
            lr = train_logs.get("perf/lr", 0)
            print(
                f"Epoch {self.epoch_count:4d}/{num_epochs} | "
                f"Loss: {total:.4f} | "
                f"Time: {elapsed:.0f}s ({epoch_s:.1f}s/ep) | "
                f"LR: {lr:.2e}"
            )
            self._print_prior_anchor_summary(train_logs)

            # Select fixed tracked samples once (lazy — val_loader must exist)
            self._initialize_tracked_batch()

            # Evaluation (with EMA) + sample-based monitoring + model selection
            should_stop = False
            if self.val_loader is not None and self.epoch_count % self.eval_interval == 0:
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

            # Sampling — fixed tracked samples (not next(iter(val_loader))).
            # This non-critical artifact may still fail (matplotlib, disk,
            # permissions), so finish it before committing the epoch ledger and
            # numbered checkpoint.  Once the checkpoint exists, no optional
            # post-commit work may turn a completed run into a false FAILED.
            if self._tracked_batch is not None and self.epoch_count % self.sample_interval == 0:
                if tracked_sample_result is None:
                    with self.ema_scope():
                        tracked_sample_result = self._sample_with_eval_seed(self._tracked_batch)
                synth_pet = tracked_sample_result["synthetic_pet"]
                print(f"  Sample PET range: [{synth_pet.min().item():.4f}, {synth_pet.max().item():.4f}]")
                self._save_sample_grid(self._tracked_batch, synth_pet)

            # Commit the epoch record before its checkpoint.  If checkpoint
            # writing then fails, resume from the previous checkpoint safely
            # truncates this valid future metrics tail.  The inverse ordering
            # can create a checkpoint whose own epoch has no metrics record.
            if monitoring_contract_enabled:
                self._append_epoch_metrics(
                    train_logs=train_logs,
                    eval_logs=avg_eval,
                    validation_metrics=val_metrics,
                )

            # The numbered checkpoint is the final fallible operation for this
            # epoch.  A successful final checkpoint therefore cannot be
            # followed by an optional artifact failure.
            if self.epoch_count % self.save_interval == 0:
                self.save_checkpoint()

            if should_stop:
                break

        print(f"\nTraining complete. Total: {time.time() - self.start_time + self._resumed_elapsed:.0f}s")

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
    def _sample_with_eval_seed(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Sample with fixed noise without advancing the training RNG state.

        The batch is fed to ``model.sample`` in chunks of
        ``eval_sample_batch_size`` so a 64-sample tracked batch never reaches
        the sampler as a single forward pass.  All chunks run inside the same
        eval RNG scope, so the per-sample noise trajectory is identical to a
        single seeded pass for the same total ordering.
        """
        chunk_size = max(1, int(getattr(self, "eval_sample_batch_size", 1)))
        with self._eval_rng_scope():
            chunks = _chunk_batch(batch, chunk_size)
            parts: List[Dict[str, Any]] = []
            for chunk in chunks:
                sampled = self.model.sample(_to_device(chunk, self.device))
                parts.append(_sampled_output_to_cpu(sampled))
            if len(parts) == 1:
                return parts[0]
            merged: Dict[str, Any] = {}
            for key in parts[0]:
                merged[key] = _cat_sampled(key, parts)
            return merged

    @torch.no_grad()
    def _compute_val_sample_metrics(
        self,
        batch: Dict[str, Any],
        synth: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Run sampling on a fixed batch and compute monitoring metrics."""
        self.model.eval()
        if synth is None:
            synth = self._sample_with_eval_seed(batch)["synthetic_pet"]
        # Sampling already transfers one bounded chunk at a time and returns
        # CPU tensors.  Keep the full tracked target/mask batch on CPU so metric
        # computation never creates a second batch-sized GPU allocation.
        synth = synth.detach().cpu()
        target = batch["pet"].detach().cpu()
        raw_mask = batch.get("mask")
        mask = raw_mask.detach().cpu() if torch.is_tensor(raw_mask) else raw_mask

        # Small-lesion stratification: fix the area threshold from THIS batch's
        # ground-truth masks so the small-lesion underestimate rate is
        # well-defined even without a precomputed cohort threshold.
        area_quantiles = self._batch_area_quantiles(mask)

        metrics: Dict[str, float] = {}
        mae_vals, mse_vals, ssim_vals, stripe_vals, psnr_vals = [], [], [], [], []
        peak_err_vals, topq_peak_err_vals = [], []
        centroid_vals, oir_vals, roi_l1_vals = [], [], []
        failure_count = 0
        small_under_count = 0
        small_total = 0
        B = synth.shape[0]
        for i in range(B):
            p = synth[i, 0].float().cpu().numpy()
            t = target[i, 0].float().cpu().numpy()
            diff = p - t
            mae_vals.append(float(np.abs(diff).mean()))
            mse_vals.append(float(np.mean(diff * diff)))
            # PSNR in dB over a [-1, 1] data range; guard divide-by-zero.
            _mse = max(float(np.mean(diff * diff)), 1e-12)
            psnr_vals.append(float(10.0 * np.log10(4.0 / _mse)))
            stripe_vals.append(_stripe_score(p))
            # SSIM (lazy import to avoid hard dep at module load)
            try:
                from skimage.metrics import structural_similarity as _ssim
                ssim_vals.append(float(_ssim(t, p, data_range=2.0)))
            except Exception:
                pass
            if mask is not None:
                mi = mask[i, 0].float().cpu().numpy()
                lesion_metrics = _compute_pet_sample_metrics(
                    p, t, mi, area_quantiles=area_quantiles
                )
                if lesion_metrics is not None:
                    peak_err_vals.append(lesion_metrics["lesion_peak_error_norm"])
                    topq_peak_err_vals.append(
                        lesion_metrics["lesion_topq_peak_error_norm"]
                    )
                    centroid_vals.append(lesion_metrics["lesion_centroid_distance"])
                    oir_vals.append(lesion_metrics["outside_inside_peak_ratio"])
                    roi_l1_vals.append(lesion_metrics["lesion_roi_l1"])
                    failure_count += int(lesion_metrics["failure"])
                    small_total += int(lesion_metrics.get("small_lesion_count", 0.0))
                    small_under_count += int(
                        lesion_metrics.get("small_lesion_underestimate", 0.0)
                    )

        def _mean(vals):
            return float(np.mean(vals)) if vals else float("nan")

        metrics["val/mae"] = _mean(mae_vals)
        metrics["val/mse"] = _mean(mse_vals)
        metrics["val/psnr"] = _mean(psnr_vals)
        metrics["val/ssim"] = _mean(ssim_vals)
        metrics["val/stripe_score"] = _mean(stripe_vals)
        metrics["val/lesion_peak_error_norm"] = _mean(peak_err_vals)
        metrics["val/lesion_topq_peak_error_norm"] = _mean(topq_peak_err_vals)
        metrics["val/lesion_centroid_distance"] = _mean(centroid_vals)
        metrics["val/outside_inside_peak_ratio"] = _mean(oir_vals)
        # false_hotspot_proxy: fraction of lesion-bearing samples where the
        # outside peak exceeds the in-lesion peak (same definition as failure).
        metrics["val/false_hotspot_proxy"] = float(failure_count) / max(
            len(peak_err_vals), 1
        )
        metrics["val/failure_rate"] = float(failure_count) / max(
            len(peak_err_vals), 1
        )
        metrics["val/lesion_roi_l1"] = _mean(roi_l1_vals)
        metrics["val/lesion_sample_count"] = float(len(peak_err_vals))
        metrics["val/small_lesion_underestimate"] = (
            float(small_under_count) / max(small_total, 1)
            if small_total > 0
            else float("nan")
        )
        metrics["val/small_lesion_sample_count"] = float(small_total)
        return metrics

    @staticmethod
    def _batch_area_quantiles(
        mask: Optional[torch.Tensor],
    ) -> Optional[tuple[float, float]]:
        """Return (Q33, Q67) of per-sample lesion area for small/medium/large cuts.

        Computed once per validation batch from ground-truth masks so the
        small-lesion underestimate rate has a deterministic threshold even
        without a precomputed cohort reference.  Returns None when no sample
        carries a lesion.
        """
        if mask is None:
            return None
        areas = mask.float().reshape(mask.shape[0], -1).sum(dim=1).cpu().numpy()
        positive = [float(a) for a in areas if a > 0.0]
        if len(positive) < 2:
            return None
        q33 = float(np.quantile(positive, 1.0 / 3.0))
        q67 = float(np.quantile(positive, 2.0 / 3.0))
        if q33 <= 0.0:
            return None
        return (q33, q67)


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
            "resumed_elapsed": (
                time.time() - self.start_time
                if self.start_time is not None
                else 0.0
            ),
        }

    def _rng_state_dict(self) -> Dict[str, Any]:
        """Capture Python / NumPy / Torch / CUDA / DataLoader RNG state.

        Schema v2 intentionally contains only primitives, containers, and CPU
        tensors accepted by ``torch.load(weights_only=True)``.  In particular,
        NumPy's ndarray state is encoded as a tensor instead of pickling
        ``numpy._core.multiarray._reconstruct``.
        """
        np_state = np.random.get_state()
        states: Dict[str, Any] = {
            "schema_version": 2,
            "python_random": random.getstate(),
            "numpy": {
                "bit_generator": str(np_state[0]),
                "state": torch.from_numpy(
                    np_state[1].astype(np.int64, copy=True)
                ).cpu(),
                "pos": int(np_state[2]),
                "has_gauss": int(np_state[3]),
                "cached_gaussian": float(np_state[4]),
            },
            "torch_cpu": torch.get_rng_state().detach().cpu(),
        }
        if torch.cuda.is_available():
            states["torch_cuda_all"] = [
                value.detach().cpu().clone()
                for value in torch.cuda.get_rng_state_all()
            ]
        loader_states: Dict[str, Any] = {}
        sampler_states: Dict[str, Any] = {}
        for name in ("train_loader", "val_loader"):
            loader = getattr(self, name, None)
            generator = getattr(loader, "generator", None)
            if generator is not None:
                loader_states[name] = generator.get_state().detach().cpu().clone()
            else:
                loader_states[name] = None
            sampler = getattr(loader, "sampler", None)
            sampler_generator = getattr(sampler, "generator", None)
            if sampler_generator is not None:
                sampler_states[name] = (
                    sampler_generator.get_state().detach().cpu().clone()
                )
            else:
                sampler_states[name] = None
        states["loader_generators"] = loader_states
        states["sampler_generators"] = sampler_states
        return states

    def _load_rng_state(self, checkpoint: Dict[str, Any]) -> None:
        """Restore schema-v2 RNG state, failing closed on an unsafe resume."""
        rng = checkpoint.get("rng")
        if not isinstance(rng, dict):
            raise RuntimeError(
                "Resume checkpoint lacks RNG state; exact continuation cannot "
                "be guaranteed"
            )
        if rng.get("schema_version") != 2:
            raise RuntimeError(
                "Resume checkpoint RNG schema is unsupported; expected "
                "schema_version=2"
            )
        py_state = rng.get("python_random")
        if py_state is None:
            raise RuntimeError("Resume checkpoint lacks Python RNG state")
        try:
            random.setstate(py_state)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Resume checkpoint Python RNG state is invalid") from exc

        np_state = rng.get("numpy")
        if not isinstance(np_state, dict):
            raise RuntimeError("Resume checkpoint lacks NumPy RNG state")
        required_numpy = {
            "bit_generator",
            "state",
            "pos",
            "has_gauss",
            "cached_gaussian",
        }
        missing_numpy = sorted(required_numpy.difference(np_state))
        if missing_numpy:
            raise RuntimeError(
                f"Resume checkpoint NumPy RNG state lacks {missing_numpy}"
            )
        np_array = np_state["state"]
        if not torch.is_tensor(np_array):
            raise RuntimeError("Resume checkpoint NumPy RNG array is not a tensor")
        try:
            np.random.set_state(
                (
                    str(np_state["bit_generator"]),
                    np_array.detach().cpu().numpy().astype(np.uint32, copy=True),
                    int(np_state["pos"]),
                    int(np_state["has_gauss"]),
                    float(np_state["cached_gaussian"]),
                )
            )
        except (ValueError, TypeError) as exc:
            raise RuntimeError("Resume checkpoint NumPy RNG state is invalid") from exc

        cpu_state = rng.get("torch_cpu")
        if not torch.is_tensor(cpu_state):
            raise RuntimeError("Resume checkpoint lacks Torch CPU RNG state")
        try:
            torch.set_rng_state(cpu_state.detach().cpu())
        except (TypeError, RuntimeError) as exc:
            raise RuntimeError("Resume checkpoint Torch CPU RNG state is invalid") from exc

        cuda_states = rng.get("torch_cuda_all")
        resume_device = torch.device(getattr(self, "device", "cpu"))
        if resume_device.type == "cuda" and torch.cuda.is_available():
            if (
                not isinstance(cuda_states, list)
                or len(cuda_states) != torch.cuda.device_count()
                or not all(torch.is_tensor(state) for state in cuda_states)
            ):
                raise RuntimeError(
                    "Resume checkpoint lacks complete Torch CUDA RNG state"
                )
            try:
                torch.cuda.set_rng_state_all(
                    [state.detach().cpu() for state in cuda_states]
                )
            except (AttributeError, TypeError, RuntimeError) as exc:
                raise RuntimeError(
                    "Resume checkpoint Torch CUDA RNG state is invalid"
                ) from exc

        loader_states = rng.get("loader_generators")
        sampler_states = rng.get("sampler_generators")
        if not isinstance(loader_states, dict) or not isinstance(
            sampler_states, dict
        ):
            raise RuntimeError(
                "Resume checkpoint lacks DataLoader/sampler RNG state"
            )
        for name in ("train_loader", "val_loader"):
            loader = getattr(self, name, None)
            if loader is None:
                continue
            generator = getattr(loader, "generator", None)
            saved_loader_state = loader_states.get(name)
            if generator is None or not torch.is_tensor(saved_loader_state):
                raise RuntimeError(
                    f"Exact resume requires an explicit {name} generator"
                )
            try:
                generator.set_state(saved_loader_state.detach().cpu())
            except (TypeError, RuntimeError) as exc:
                raise RuntimeError(
                    f"Resume checkpoint {name} generator state is invalid"
                ) from exc

            sampler = getattr(loader, "sampler", None)
            sampler_generator = getattr(sampler, "generator", None)
            saved_sampler_state = sampler_states.get(name)
            # Validation is sequential and has no sampler RNG.  The shuffled
            # training sampler must always be explicit and checkpointed.
            if name == "train_loader":
                if sampler_generator is None or not torch.is_tensor(
                    saved_sampler_state
                ):
                    raise RuntimeError(
                        "Exact resume requires an explicit train sampler "
                        "generator"
                    )
                try:
                    sampler_generator.set_state(
                        saved_sampler_state.detach().cpu()
                    )
                except (TypeError, RuntimeError) as exc:
                    raise RuntimeError(
                        "Resume checkpoint train sampler generator state is "
                        "invalid"
                    ) from exc

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
        resumed_elapsed = state.get("resumed_elapsed")
        self._resumed_elapsed = (
            float(resumed_elapsed) if resumed_elapsed is not None else 0.0
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
            save_dir = resolve_checkpoint_dir(self.config)
            os.makedirs(save_dir, exist_ok=True)

        def _save(tag: str, score: float) -> None:
            if not self.best_ckpts_enabled or not improvements[tag]:
                return
            path = os.path.join(save_dir, f"ckpt_{tag}.pt")
            checkpoint = {
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
            }
            _atomic_torch_save(
                attach_data_lineage(
                    checkpoint, getattr(self, "data_lineage", None)
                ),
                path,
            )
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
        save_dir = resolve_checkpoint_dir(self.config)
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
            "rng": self._rng_state_dict(),
        }
        _atomic_torch_save(
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

        sample_dir = resolve_sample_dir(self.config)
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
        # Load on CPU so CPU RNG tensors are never remapped to CUDA.  Module and
        # optimizer loaders copy/cast their tensors to the live parameter
        # devices below.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError("Resume checkpoint payload is not a mapping")
        _validate_prior_anchored_resume_identity(
            current_config=self.config,
            checkpoint_config=checkpoint.get("config"),
            checkpoint_path=os.fspath(path),
        )
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
        parameter_devices = {
            name: parameter.device
            for name, parameter in self.model.named_parameters()
        }
        self.ema.shadow = {
            name: value.to(parameter_devices.get(name, self.device))
            for name, value in self.ema.shadow.items()
        }
        self.epoch_count = checkpoint["epoch"]
        self.step_count = checkpoint["step"]
        self._load_monitoring_state(checkpoint)
        # The fixed tracked batch is intentionally not serialized.  Rebuild it
        # while the fresh validation loader is still at its initial state, then
        # restore the checkpoint RNG below.  Otherwise run() would perform this
        # scan after RNG restoration and advance the val-loader generator beyond
        # the uninterrupted trajectory.  build_dataloaders always constructs the
        # validation CachedDataset with augment=False, so this scan is
        # deterministic and carries no hidden worker augmentation state.
        self._initialize_tracked_batch()
        self._load_rng_state(checkpoint)
        self.accum_count = 0
        self._set_spectral_router_epoch()
        self._truncate_metrics_jsonl_to_epoch()
        print(f"Loaded checkpoint from {path} (epoch {self.epoch_count})")
