"""Lesion-aware perceptual ``x0`` feature supervision (PFM-inspired).

Supervises the model's recovered clean PET ``pred_x0`` against a **frozen**
PET lesion encoder.  The encoder is trained on training/calibration patients
only, then frozen; the target branch runs under ``no_grad`` and the prediction
branch must keep gradients flowing back to ``pred_x0``.

This is NOT a faithful re-implementation of Perceptual Flow Matching.  It is a
PFM-inspired supervision-space ablation on top of the existing Brownian-bridge
residual diffusion: the BBDM forward/reverse equations are unchanged.

Fail-closed policy:
  * A real encoder checkpoint is required unless ``encoder_kind == "random"``
    (only the P4_FEAT_RANDOM negative control may use a random frozen encoder).
  * ``require_checkpoint_lineage`` demands a ``_lineage`` dict embedded in the
    checkpoint so a stale or cross-patient encoder is rejected at load time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces import LossContext, LossTerm
from .base import smooth_tau_gate

# Multiscale feature layers produced by the encoder's forward_features.
FEATURE_LAYER_NAMES = ("full", "half", "quarter")

# Minimum gates for the frozen encoder lineage (design doc hard gate #3):
# overall lesion recall and small-lesion-quartile recall must both clear 0.70.
MIN_LESION_RECALL = 0.70
MIN_SMALL_LESION_RECALL = 0.70


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def _dilate_binary(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilation by max-pooling, preserving [B,1,H,W]."""
    binary = (mask > 0.5).float()
    if radius <= 0:
        return binary
    k = 2 * radius + 1
    return F.max_pool2d(binary, kernel_size=k, stride=1, padding=radius)


def build_feature_layer(
    in_channels: int,
    out_channels: int,
    target_scale: int,
) -> nn.Sequential:
    """One feature block; ``target_scale`` is 1 (full), 2 (half), 4 (quarter).

    Each block downsamples to its own target resolution relative to the input
    (not cumulatively): half reaches 1/2 and quarter reaches 1/4 of the input,
    so the ``quarter`` feature map is truly 1/4, not 1/8.
    """
    kernel = 3 if target_scale <= 2 else 5
    padding = kernel // 2
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.SiLU(),
    ]
    down_steps = {1: 0, 2: 1, 4: 2}[target_scale]
    for _ in range(down_steps):
        layers.append(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        )
        layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class PETFeatureEncoder(nn.Module):
    """Small multi-scale PET→feature encoder with an explicit forward_features.

    Exposes ``forward_features`` returning at least ``full``, ``half`` and
    ``quarter`` spatial scales (relative to the input) so the loss never relies
    on fragile global hooks.  Each scale is a separate branch from the input,
    so the downsampling is not accumulated across scales.
    """

    name = "pet_feature_encoder"

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        feature_layers: Sequence[str] = FEATURE_LAYER_NAMES,
    ):
        super().__init__()
        layer_plan = {
            "full": (base_channels, 1),
            "half": (base_channels * 2, 2),
            "quarter": (base_channels * 4, 4),
        }
        blocks: Dict[str, nn.Module] = {}
        for name in FEATURE_LAYER_NAMES:
            out_channels, target_scale = layer_plan[name]
            # "half" collides with nn.Module.half(), so block keys are
            # prefixed; the public feature dict still uses full/half/quarter.
            blocks[f"layer_{name}"] = build_feature_layer(
                in_channels, out_channels, target_scale
            )
        self.blocks = nn.ModuleDict(blocks)
        self.feature_layers = tuple(feature_layers)
        self.scale_factors = {
            name: {1: 1, 2: 2, 4: 4}[layer_plan[name][1]]
            for name in FEATURE_LAYER_NAMES
        }

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return a dict of spatial feature maps, one per configured layer."""
        features: Dict[str, torch.Tensor] = {}
        for name in FEATURE_LAYER_NAMES:
            features[name] = self.blocks[f"layer_{name}"](x)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)["quarter"]

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _load_sidecar_lineage(checkpoint_path: Path) -> Dict[str, Any]:
    """Load the lineage sidecar JSON next to the encoder checkpoint.

    The canonical lineage lives in ``<checkpoint>.lineage.json`` written by the
    pretraining script.  This is preferred over an embedded ``_lineage`` field
    because the embedded dict is not independently verifiable.
    """
    sidecar = checkpoint_path.with_name(checkpoint_path.name + ".lineage.json")
    if not sidecar.is_file():
        return {}
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Lineage sidecar is not a JSON object: {sidecar}")
    return payload


def build_encoder(
    *,
    checkpoint: Optional[str] = None,
    checkpoint_dir: str = "checkpoints",
    encoder_kind: str = "pretrained",
    in_channels: int = 1,
    base_channels: int = 16,
    feature_layers: Sequence[str] = FEATURE_LAYER_NAMES,
) -> Tuple[PETFeatureEncoder, Dict[str, Any]]:
    """Construct and freeze a PET feature encoder with strict lineage.

    ``encoder_kind == "random"`` is the explicit P4_FEAT_RANDOM negative
    control: a randomly initialised, then frozen, encoder with no checkpoint.
    Any other ``encoder_kind`` requires an on-disk checkpoint whose lineage is
    validated (sidecar file preferred over the embedded ``_lineage`` dict).
    """
    encoder = PETFeatureEncoder(
        in_channels=in_channels,
        base_channels=base_channels,
        feature_layers=feature_layers,
    )
    lineage: Dict[str, Any] = {}
    if encoder_kind == "random":
        # Random negative control: freeze immediately, no checkpoint, no lineage.
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
        return encoder, {"encoder_kind": "random", "checkpoint": None}

    if not checkpoint:
        raise ValueError(
            "perceptual_x0 encoder_kind != 'random' requires a 'checkpoint' "
            "path; refusing to silently fall back to random weights"
        )
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        # The experiment plan stores ``checkpoint`` as a repo-root-relative path
        # (e.g. "checkpoints/pet_feature_encoder_v1/encoder_best.pt"), matching
        # the runner's audit in run_perceptual_x0_ablation.py:_encoder_checkpoint_for
        # which resolves ``root / checkpoint``.  train_v2.py runs with cwd=root,
        # so a relative Path is already root-relative here.  Fall back to
        # ``checkpoint_dir / checkpoint`` only when the root-relative path is
        # absent, preserving the legacy checkpoint_dir-relative convention.
        if not checkpoint_path.is_file():
            checkpoint_path = Path(checkpoint_dir) / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"perceptual_x0 encoder checkpoint not found: {checkpoint_path}"
        )
    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state, dict) or "state_dict" not in state:
        raise ValueError(
            f"perceptual_x0 encoder checkpoint must contain 'state_dict': "
            f"{checkpoint_path}"
        )
    encoder.load_state_dict(state["state_dict"])
    if "base_channels" in state:
        if int(state["base_channels"]) != base_channels:
            raise ValueError(
                "perceptual_x0 encoder checkpoint base_channels mismatch: "
                f"state={state['base_channels']}, config={base_channels}"
            )
    # Lineage: sidecar file wins; embedded _lineage is a fallback.
    lineage = _load_sidecar_lineage(checkpoint_path) or dict(
        state.get("_lineage") or {}
    )
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    return encoder, {
        "encoder_kind": encoder_kind,
        "checkpoint": checkpoint_path.as_posix(),
        "sha256": _file_sha256(checkpoint_path),
        "lineage": lineage,
    }


def _file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _stable_channel_normalize(features: torch.Tensor) -> torch.Tensor:
    """Per-channel standardisation with a stable scale and NaN/Inf guards.

    Each channel is normalised by max(per-channel std, eps).  Empty or constant
    channels produce a zero map instead of dividing by zero or emitting NaNs.
    """
    eps = 1e-8
    scale = features.std(dim=(2, 3), keepdim=True)
    scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)
    scale = scale.clamp_min(eps)
    normalized = features / scale
    return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def _charbonnier(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    diff = (a - b).square()
    return (diff + eps * eps).sqrt() - eps


class LesionAwarePerceptualX0Loss(LossTerm):
    """Multiscale frozen-feature distance on clean PET ``x0`` predictions.

    Supports global and lesion-balanced region modes.  In lesion-balanced mode
    the lesion (dilated) region and the non-lesion body are averaged separately
    and combined with explicit weights so small lesions are not drowned out by
    the whole-image mean.
    """

    name = "perceptual_x0"

    def __init__(
        self,
        *,
        enabled: bool = True,
        weight: float = 1.0,
        checkpoint: Optional[str] = None,
        checkpoint_dir: str = "checkpoints",
        encoder_kind: str = "pretrained",
        in_channels: int = 1,
        base_channels: int = 16,
        feature_layers: Sequence[str] = FEATURE_LAYER_NAMES,
        layer_weights: Optional[Sequence[float]] = None,
        distance: str = "charbonnier",
        charbonnier_eps: float = 1e-3,
        region_mode: str = "global",
        lesion_weight: float = 4.0,
        background_weight: float = 1.0,
        dilate_radius: int = 3,
        timestep_weighting: str = "uniform",
        require_checkpoint_lineage: bool = False,
    ):
        super().__init__(enabled=enabled, weight=weight)
        if not 0.0 < charbonnier_eps:
            raise ValueError("charbonnier_eps must be positive")
        if distance not in ("charbonnier", "l1"):
            raise ValueError(f"Unknown distance {distance!r}")
        if region_mode not in ("global", "lesion_balanced"):
            raise ValueError(f"Unknown region_mode {region_mode!r}")
        if timestep_weighting not in ("uniform",):
            raise ValueError(
                f"Unsupported timestep_weighting {timestep_weighting!r}; "
                "v1 only supports 'uniform'"
            )
        self.distance = distance
        self.charbonnier_eps = charbonnier_eps
        self.region_mode = region_mode
        self.lesion_weight = float(lesion_weight)
        self.background_weight = float(background_weight)
        self.dilate_radius = int(dilate_radius)
        self.timestep_weighting = timestep_weighting
        self.require_checkpoint_lineage = bool(require_checkpoint_lineage)

        resolved_layers = tuple(feature_layers)
        unknown = set(resolved_layers) - set(FEATURE_LAYER_NAMES)
        if unknown:
            raise ValueError(f"Unknown feature_layers {sorted(unknown)}")
        self.feature_layers = resolved_layers
        self.layer_weights = dict(
            zip(
                FEATURE_LAYER_NAMES,
                layer_weights if layer_weights else [1.0, 0.5, 0.25],
            )
        )

        # The encoder lives on this module so that its parameters are owned by
        # the loss term, but all of them are requires_grad=False so neither
        # _fused_adamw nor the grouped optimizer will ever see them.
        self.encoder, self.encoder_meta = build_encoder(
            checkpoint=checkpoint,
            checkpoint_dir=checkpoint_dir,
            encoder_kind=encoder_kind,
            in_channels=in_channels,
            base_channels=base_channels,
            feature_layers=feature_layers,
        )
        if self.require_checkpoint_lineage:
            self._validate_encoder_lineage()

    def _validate_encoder_lineage(self) -> None:
        if self.encoder_meta.get("encoder_kind") == "random":
            return  # negative control has no lineage by design
        lineage = self.encoder_meta.get("lineage") or {}
        checkpoint_path = Path(self.encoder_meta.get("checkpoint", ""))

        def require(name: str, field: str, present: bool, detail: str) -> None:
            if not present:
                raise ValueError(
                    f"perceptual_x0 encoder lineage is missing {field!r} "
                    f"({detail}); refusing to run with unverified weights"
                )

        # 1. Hash must match the actual checkpoint file.
        declared_sha = str(lineage.get("checkpoint_sha256") or "")
        actual_sha = self.encoder_meta.get("sha256") or ""
        require(
            "hash", "checkpoint_sha256",
            bool(declared_sha) and bool(actual_sha) and declared_sha == actual_sha,
            f"declared={declared_sha!r}, actual={actual_sha!r}, path={checkpoint_path}",
        )

        # 2. Training-split provenance (patients only, no test/validation overlap).
        train_split = lineage.get("train_patient_split")
        require("split", "train_patient_split", isinstance(train_split, dict) and bool(train_split), "must be a patient->split map")
        if isinstance(train_split, dict):
            for patient, split in train_split.items():
                if split not in ("train", "calibration"):
                    raise ValueError(
                        f"perceptual_x0 encoder lineage lists patient {patient!r} "
                        f"in split {split!r}; encoder must train only on "
                        "train/calibration patients, never validation"
                    )

        # 3. Recall gates: overall and small-lesion-quartile recall >= 0.70.
        eval_metrics = lineage.get("eval_metrics")
        require("metrics", "eval_metrics", isinstance(eval_metrics, dict) and bool(eval_metrics), "must record lesion recall")
        if isinstance(eval_metrics, dict):
            recall = eval_metrics.get("lesion_recall")
            small_recall = eval_metrics.get("small_lesion_recall")
            require(
                "recall", "eval_metrics.lesion_recall",
                isinstance(recall, (int, float)) and float(recall) >= MIN_LESION_RECALL,
                f"lesion_recall={recall!r} must be >= {MIN_LESION_RECALL}",
            )
            require(
                "small_recall", "eval_metrics.small_lesion_recall",
                isinstance(small_recall, (int, float)) and float(small_recall) >= MIN_SMALL_LESION_RECALL,
                f"small_lesion_recall={small_recall!r} must be >= {MIN_SMALL_LESION_RECALL}",
            )

    def _downsample_mask(
        self,
        mask: torch.Tensor,
        spatial: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Downsample a [B,1,H,W] binary mask to a feature map's spatial size."""
        if mask.shape[-2:] == spatial[-2:]:
            return mask.to(device=device, dtype=dtype)
        pooled = F.avg_pool2d(
            mask.to(device=device, dtype=dtype),
            kernel_size=2,
            stride=2,
        )
        # Round to binary: any touched region counts.
        binary = (pooled > 0).to(dtype=dtype)
        return F.interpolate(
            binary,
            size=spatial[-2:],
            mode="nearest",
        )

    def _region_mean(
        self,
        diff: torch.Tensor,
        region: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Per-sample mean of ``diff`` over binary ``region`` [B,1,H,W].

        Empty regions return zero for that sample instead of NaN, so empty or
        tiny lesion masks are fail-safe.
        """
        diff = diff.to(dtype=torch.float32)
        if region.shape[1] == 1 and region.shape[1] != diff.shape[1]:
            # Broadcast the [B,1,H,W] mask across feature channels.
            region = region.expand_as(diff)
        flat = diff.flatten(1)
        region_flat = region.to(dtype=torch.float32).flatten(1)
        counts = region_flat.sum(dim=1).clamp_min(eps)
        return (flat * region_flat).sum(dim=1) / counts

    def _per_layer_region_loss(
        self,
        pred_feat: torch.Tensor,
        target_feat: torch.Tensor,
        lesion: Optional[torch.Tensor],
        scale_factor: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute per-sample loss for one feature scale.

        The lesion mask is dilated at the feature scale using a radius scaled to
        the input resolution (``dilate_radius / scale_factor``), so the three
        scales correspond to the same original-image dilation radius.

        Returns (per_sample_loss [B], logs).  Gradient is only allowed through
        the pred branch; the target branch was already computed under no_grad.
        """
        pred_n = _stable_channel_normalize(pred_feat)
        target_n = _stable_channel_normalize(target_feat)
        if self.distance == "charbonnier":
            distance = _charbonnier(pred_n, target_n, self.charbonnier_eps)
        else:
            distance = (pred_n - target_n).abs()

        layer_logs: Dict[str, torch.Tensor] = {}
        if self.region_mode == "global" or lesion is None:
            per_sample = distance.mean(dim=(1, 2, 3))
            layer_logs["lesion_weight"] = torch.tensor(0.0, device=pred_feat.device)
            layer_logs["background_weight"] = torch.tensor(0.0, device=pred_feat.device)
            return per_sample, layer_logs

        spatial = pred_feat.shape
        device = pred_feat.device
        dtype = pred_feat.dtype
        lesion_map = self._downsample_mask(lesion, spatial, device, dtype)
        radius = max(1, round(self.dilate_radius / scale_factor))
        if radius > 0:
            lesion_map = _dilate_binary(lesion_map, radius)
        background = 1.0 - lesion_map

        lesion_diff = self._region_mean(distance, lesion_map)
        background_diff = self._region_mean(distance, background)

        if self.lesion_weight > 0:
            lesion_term = self.lesion_weight * lesion_diff
        else:
            lesion_term = torch.zeros_like(lesion_diff)
        if self.background_weight > 0:
            background_term = self.background_weight * background_diff
        else:
            background_term = torch.zeros_like(background_diff)
        per_sample = lesion_term + background_term
        layer_logs["lesion_weight"] = torch.as_tensor(
            self.lesion_weight, device=device
        )
        layer_logs["background_weight"] = torch.as_tensor(
            self.background_weight, device=device
        )
        return per_sample, layer_logs

    def forward(
        self,
        ctx: LossContext,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = ctx.target_pet.device
        zero = torch.tensor(0.0, device=device)
        if not self.enabled:
            return zero, {f"{self.name}/enabled": zero}

        pred_pet = ctx.pred_x0 if ctx.pred_x0 is not None else ctx.model_pred
        if pred_pet.shape != ctx.target_pet.shape:
            raise ValueError(
                f"perceptual_x0 expects pred_x0 shape {ctx.target_pet.shape}, "
                f"got {tuple(pred_pet.shape)}"
            )

        with torch.no_grad():
            target_features = self.encoder.forward_features(
                ctx.target_pet.float()
            )
        # Prediction branch: gradients must flow into pred_x0.
        pred_features = self.encoder.forward_features(pred_pet.float())

        lesion = ctx.batch.get("mask")
        if lesion is not None:
            lesion = lesion.to(device=device)

        per_sample = torch.zeros(
            pred_pet.shape[0], device=device, dtype=torch.float32
        )
        layer_logs: Dict[str, torch.Tensor] = {}
        for layer in self.feature_layers:
            scale_factor = self.encoder.scale_factors[layer]
            layer_loss, logs = self._per_layer_region_loss(
                pred_features[layer],
                target_features[layer],
                lesion,
                scale_factor,
            )
            per_sample = per_sample + float(self.layer_weights[layer]) * layer_loss
            layer_logs[f"{self.name}/layer_{layer}"] = layer_loss.detach().mean()
            for key, value in logs.items():
                layer_logs[f"{self.name}/layer_{layer}/{key}"] = value

        if self.timestep_weighting == "uniform":
            # Contract-specified uniform weighting: every timestep contributes
            # equally.  (No sigmoid tau gate; a plain mean is used.)
            loss = per_sample.mean()
        else:  # pragma: no cover - __init__ restricts to "uniform"
            gate = smooth_tau_gate(ctx.tau, max_tau=1.0).to(device=device)
            loss = (per_sample * gate).mean()

        if not torch.isfinite(loss):
            # Fail closed rather than silently masking a training fault: a
            # non-finite perceptual loss is a contract violation, not a case
            # to paper over with a zero.
            raise RuntimeError(
                f"perceptual_x0 loss is not finite: {float(loss)!r}"
            )

        encoder_params = sum(
            int(param.numel()) for param in self.encoder.parameters()
        )
        trainable_encoder_params = sum(
            int(param.numel())
            for param in self.encoder.parameters()
            if param.requires_grad
        )
        logs: Dict[str, torch.Tensor] = {
            f"{self.name}/loss": loss.detach(),
            f"{self.name}/region_mode": torch.as_tensor(
                0.0 if self.region_mode == "global" else 1.0, device=device
            ),
            f"{self.name}/encoder_kind": torch.as_tensor(
                0.0 if self.encoder_meta.get("encoder_kind") == "random" else 1.0,
                device=device,
            ),
            f"{self.name}/encoder_total_params": torch.as_tensor(
                encoder_params, dtype=torch.float32, device=device
            ),
            f"{self.name}/encoder_trainable_params": torch.as_tensor(
                trainable_encoder_params, dtype=torch.float32, device=device
            ),
            f"{self.name}/enabled": torch.tensor(1.0, device=device),
        }
        logs.update(layer_logs)
        return loss * self.weight, logs
