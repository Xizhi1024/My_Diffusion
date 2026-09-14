"""SLMF-BBDM: Small-Lesion Metabolic-Fidelity Brownian Bridge Diffusion Model.

Core model class that wires together:
  - Prior modules (Gabor, Organ, Hotspot, Semantic)
  - Condition adapter (Zero-Conv or raw concat)
  - Noise schedule (DDPM, BBDM bridge, or scale-adaptive)
  - UNet backbone
  - Pluggable loss stack with time-varying activation

Design principle: every optimisation module has an `enabled` flag.
Disabled modules follow NoOp paths – zero new code branches in training.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .interfaces import ConditionBundle, LossContext
from .conditioning.adapter import ZeroConvAdapter, RawConcatAdapter
from .conditioning.beta_schedule import multi_level_betas
from .conditioning.dropout import ConditionDropout
from .bbdm_unet import BBDMUNet


_META_KEYS = ("uptake_min", "weight_kg", "age_years", "thickness_mm", "z_mm")
# Default normalisation ranges: (low, high) → [-1, 1]
_META_NORM = {
    "uptake_min":    (30.0,  120.0),   # FDG injection-to-scan delay (min)
    "weight_kg":     (40.0,  150.0),   # Patient weight (kg)
    "age_years":     (20.0,   90.0),   # Patient age (years)
    "thickness_mm":  ( 0.5,    5.0),   # Slice thickness (mm)
    "z_mm":          (-120.0, 120.0),  # Axial slice location (mm)
}


def _meta_to_tensor(meta_batch, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Extract metadata from batch and normalise to [-1, 1] tensor [B, D].

    ``meta_batch`` can be: a list of per-sample dicts, a dict of lists (default
    DataLoader collation), or a single dict (broadcast to all B).
    Missing keys are filled with 0.0 (centre of normalised range).
    """
    # Normalise to per-sample list of dicts
    if isinstance(meta_batch, list):
        meta_list = meta_batch[:B]
    elif isinstance(meta_batch, dict):
        # Heuristic: if first value has len==B, it's collated → transpose
        first_val = next(iter(meta_batch.values()), None)
        is_collated = isinstance(first_val, (list, tuple)) and len(first_val) == B
        if is_collated:
            meta_list = [{k: meta_batch[k][i] for k in meta_batch} for i in range(B)]
        else:
            meta_list = [meta_batch] * B
    else:
        meta_list = [{}] * B

    vec = torch.zeros(B, len(_META_KEYS), device=device, dtype=dtype)
    for i in range(B):
        m = meta_list[i] if i < len(meta_list) else {}
        for j, key in enumerate(_META_KEYS):
            val = m.get(key)
            if val is None:
                continue  # leave as 0.0
            lo, hi = _META_NORM[key]
            # Normalise: (val - lo) / (hi - lo) → [0, 1] → 2*x - 1 → [-1, 1]
            x = (float(val) - lo) / max(hi - lo, 1e-8)
            vec[i, j] = 2.0 * x - 1.0
    return vec


_NOISE_HAS_XSOURCE: dict = {}  # Weak cache keyed by schedule type
_PRIOR_MODULE_NAMES = ("gabor", "organ_prior", "hotspot_prior", "semantic_prior")


def resolve_noise_config(modules_cfg: dict) -> dict:
    """Find the noise schedule config from modules dict.

    Searches for a key whose value is a dict containing ``"name"`` with a
    recognised noise schedule name.  Falls back to ``bbdm_bridge``.
    """
    noise_keys = {"scale_adaptive_noise", "bbdm_bridge", "noise"}
    for key in noise_keys:
        cfg = modules_cfg.get(key)
        if isinstance(cfg, dict) and "name" in cfg:
            return cfg
    return {"name": "bbdm_bridge"}


def _add_noise(schedule, x0, noise, timesteps, condition, x_source=None):
    """Call schedule.add_noise with or without x_source depending on signature."""
    schedule_type = type(schedule)
    if schedule_type not in _NOISE_HAS_XSOURCE:
        import inspect
        try:
            sig = inspect.signature(schedule.add_noise)
            _NOISE_HAS_XSOURCE[schedule_type] = "x_source" in sig.parameters
        except (ValueError, TypeError):
            _NOISE_HAS_XSOURCE[schedule_type] = False
    if _NOISE_HAS_XSOURCE[schedule_type]:
        return schedule.add_noise(x0, noise, timesteps, condition, x_source=x_source)
    return schedule.add_noise(x0, noise, timesteps, condition)


def _get_alpha_cumprod(schedule, timesteps):
    """Get alpha_cumprod from a noise schedule (DDPM or ScaleAdaptive)."""
    if hasattr(schedule, 'alphas_cumprod'):
        return schedule.alphas_cumprod[timesteps]
    if hasattr(schedule, 'base_sigma'):
        sigma = schedule.base_sigma[timesteps]
        return 1.0 / (1.0 + sigma ** 2)
    raise AttributeError(f"Noise schedule {type(schedule).__name__} has no alphas_cumprod or base_sigma")


def _image_gradient_l1_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample L1 difference between spatial gradients."""
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    loss_x = (pred_dx - target_dx).abs().flatten(1).mean(dim=1)
    loss_y = (pred_dy - target_dy).abs().flatten(1).mean(dim=1)
    return 0.5 * (loss_x + loss_y)


# ---------------------------------------------------------------------------
# CT Encoder (lightweight, multi-scale)
# ---------------------------------------------------------------------------

class _CTEncoder(nn.Module):
    """Produces 4 multi-scale feature maps [c1, c2, c3, c4] from CT input."""

    def __init__(
        self,
        in_channels: int = 1,
        base_ch: int = 64,
        out_channels: tuple = (64, 128, 256, 256),
    ):
        super().__init__()
        c1, c2, c3, c4 = out_channels

        # Multi-Scale Stem
        self.stem_3 = nn.Conv2d(in_channels, base_ch // 2, 3, padding=1)
        self.stem_7 = nn.Conv2d(in_channels, base_ch // 2, 7, padding=3)
        self.stem_fuse = nn.Conv2d(base_ch, c1, 1)

        # Down blocks
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c1, c2, 4, stride=2, padding=1),
            nn.SiLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c2, c3, 4, stride=2, padding=1),
            nn.SiLU(),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(c3, c3, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c3, c4, 4, stride=2, padding=1),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        s3 = self.stem_3(x)
        s7 = self.stem_7(x)
        c1 = self.stem_fuse(torch.cat([s3, s7], dim=1))   # [B, 64, H, W]
        c2 = self.down1(c1)                                # [B, 128, H/2, W/2]
        c3 = self.down2(c2)                                # [B, 256, H/4, W/4]
        c4 = self.down3(c3)                                # [B, 256, H/8, W/8]
        return [c1, c2, c3, c4]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SLMFBBDM(nn.Module):
    """Small-Lesion Metabolic-Fidelity Brownian Bridge Diffusion Model.

    Usage::

        model = SLMFBBDM.from_config(config)
        loss, logs = model(batch)
    """

    def __init__(
        self,
        image_size: int = 192,
        objective: str = "pred_x0",
        enable_heteroscedastic: bool = True,
        heteroscedastic_logvar_min: float = -6.0,
        heteroscedastic_logvar_max: float = 2.0,
        # Module configurations (raw dicts from YAML)
        ct_encoder_config: Optional[Dict[str, Any]] = None,
        prior_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        adapter_config: Optional[Dict[str, Any]] = None,
        noise_config: Optional[Dict[str, Any]] = None,
        loss_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        condition_dropout_config: Optional[Dict[str, Any]] = None,
        base_loss_config: Optional[Dict[str, Any]] = None,
        self_conditioning_config: Optional[Dict[str, Any]] = None,
        meta_config: Optional[Dict[str, Any]] = None,
        segmenter_config: Optional[Dict[str, Any]] = None,
        # Inference
        sample_scheduler: str = "ddim",
        eval_sampling_steps: int = 20,
    ):
        super().__init__()
        self.image_size = image_size
        if objective != "pred_x0":
            raise ValueError(f"SLMF-BBDM only supports objective='pred_x0', got '{objective}'")
        self.enable_heteroscedastic = enable_heteroscedastic
        self.heteroscedastic_logvar_min = heteroscedastic_logvar_min
        self.heteroscedastic_logvar_max = heteroscedastic_logvar_max
        self.sample_scheduler = sample_scheduler
        self.eval_sampling_steps = eval_sampling_steps

        base_cfg = base_loss_config or {}
        self.base_mse_weight = base_cfg.get("mse_weight", 1.0)
        self.base_l1_weight = base_cfg.get("l1_weight", 1.0)
        self.base_gradient_weight = base_cfg.get("gradient_weight", 0.1)
        self.min_snr_enabled = base_cfg.get("min_snr_enabled", True)
        self.min_snr_gamma = base_cfg.get("min_snr_gamma", 5.0)

        sc_cfg = self_conditioning_config or {}
        self.self_conditioning = sc_cfg.get("enabled", False)
        self.self_conditioning_prob = sc_cfg.get("probability", 0.5)

        # ---- CT Encoder ----
        # Output channels must match UNet skip channels: [64, 128, 256, 256]
        ct_cfg = ct_encoder_config or {}
        self.ct_encoder = _CTEncoder(
            in_channels=ct_cfg.get("in_channels", 1),
            base_ch=ct_cfg.get("base_ch", 64),
            out_channels=(64, 128, 256, 256),
        )

        # ---- Prior modules ----
        prior_cfgs = prior_configs or {}
        self.priors = nn.ModuleDict()
        for name, cfg in prior_cfgs.items():
            if cfg.get("enabled", False):
                self.priors[name] = self._build_prior(name, cfg)
            else:
                from .priors.noop import NoOpPrior
                self.priors[name] = NoOpPrior()
                self.priors[name].name = name

        # ---- Condition adapter ----
        adapter_cfg = adapter_config or {}
        self.zero_adapter_enabled = adapter_cfg.get("enabled", True)
        if self.zero_adapter_enabled:
            self.adapter = ZeroConvAdapter(enabled=True)
        else:
            self.adapter = RawConcatAdapter(enabled=False)

        # ---- Noise schedule ----
        noise_cfg = noise_config or {"name": "bbdm_bridge"}
        self.noise_schedule = self._build_noise(noise_cfg)

        # ---- Metadata FiLM ----
        meta_cfg = meta_config or {}
        self.meta_enabled = meta_cfg.get("enabled", False)
        self.meta_keys = meta_cfg.get("keys", list(_META_KEYS))
        meta_dim = len(self.meta_keys) if self.meta_enabled else 0

        # ---- Tiny PET Segmenter (frozen, for L_seg) ----
        seg_cfg = segmenter_config or {}
        self.segmenter_enabled = seg_cfg.get("enabled", False)
        self.segmenter: Optional[nn.Module] = None
        if self.segmenter_enabled:
            from .segmenter import TinySegmenter
            self.segmenter = TinySegmenter(
                in_channels=seg_cfg.get("in_channels", 1),
                base_channels=seg_cfg.get("base_channels", 16),
                stage=seg_cfg.get("stage", 2),
                gaussian_sigma=seg_cfg.get("gaussian_sigma", 6.0),
                enabled=True,
            )
            # Load pretrained weights if provided
            ckpt_path = seg_cfg.get("checkpoint")
            if ckpt_path is not None:
                state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                self.segmenter.load_state_dict(state)
                print(f"[SLMF-BBDM] Loaded frozen segmenter from {ckpt_path}")
            # Freeze — segmenter is an oracle, not a trainable part of BBDM
            for p in self.segmenter.parameters():
                p.requires_grad = False
            self.segmenter.eval()

        # ---- UNet ----
        # Input is [noisy_x, ct] plus optional previous x0 estimate.
        self.unet = BBDMUNet(
            in_channels=3 if self.self_conditioning else 2,
            enable_heteroscedastic=enable_heteroscedastic,
            ca_kv_dim=64,
            meta_dim=meta_dim,
        )

        # ---- Loss stack ----
        loss_cfgs = loss_configs or {}
        self.loss_terms = nn.ModuleDict()
        for name, cfg in loss_cfgs.items():
            self.loss_terms[name] = self._build_loss(name, cfg)

        # ---- Condition dropout ----
        drop_cfg = condition_dropout_config or {}
        self.condition_dropout = ConditionDropout(
            p_organ=drop_cfg.get("p_organ", 0.1),
            p_hotspot=drop_cfg.get("p_hotspot", 0.1),
            p_semantic=drop_cfg.get("p_semantic", 0.1),
            p_meta=drop_cfg.get("p_meta", 0.1),
            p_gabor=drop_cfg.get("p_gabor", 0.1),
            enabled=drop_cfg.get("enabled", False),
        )

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def _build_prior(self, name: str, cfg: Dict[str, Any]) -> PriorModule:
        from .priors.noop import NoOpPrior

        enabled = cfg.get("enabled", True)
        if not enabled:
            return NoOpPrior()

        if name == "gabor":
            from .priors.gabor import GaborPrior
            return GaborPrior(filters=cfg.get("filters", 32), enabled=enabled)
        elif name == "organ_prior":
            from .priors.organ import OrganPrior
            return OrganPrior(
                organ_channels=cfg.get("organ_channels", 6),
                mu_map_channels=cfg.get("mu_map_channels", 1),
                out_channels=tuple(cfg.get("out_channels", [16, 32, 64])),
                enabled=enabled,
            )
        elif name == "hotspot_prior":
            from .priors.hotspot import HotspotPrior
            return HotspotPrior(
                base_channels=cfg.get("base_channels", 16),
                params_max=cfg.get("params_max", 500_000),
                enabled=enabled,
            )
        elif name == "semantic_prior":
            from .priors.semantic import SemanticPrior
            return SemanticPrior(
                mode=cfg.get("mode", "cached_tokens"),
                token_dim=cfg.get("token_dim", 64),
                num_tokens=cfg.get("num_tokens", 4),
                enabled=enabled,
            )
        else:
            return NoOpPrior()

    def _build_noise(self, cfg: Dict[str, Any]) -> NoiseSchedule:
        name = cfg.get("name", "bbdm_bridge")
        enabled = cfg.get("enabled", True)
        if name == "ddpm":
            from .noise.base import DDPMNoiseSchedule
            return DDPMNoiseSchedule(
                num_train_timesteps=cfg.get("num_train_timesteps", 1000),
                enabled=enabled,
            )
        elif name == "bbdm_bridge":
            from .noise.base import BBDMBridgeSchedule
            return BBDMBridgeSchedule(
                num_train_timesteps=cfg.get("num_train_timesteps", 1000),
                m_schedule=cfg.get("m_schedule", "linear"),
                enabled=enabled,
            )
        elif name == "scale_adaptive":
            from .noise.scale_adaptive import ScaleAdaptiveNoise
            return ScaleAdaptiveNoise(
                num_train_timesteps=cfg.get("num_train_timesteps", 1000),
                low_sigma_mult=cfg.get("low_sigma_mult", 1.0),
                mid_sigma_mult=cfg.get("mid_sigma_mult", 0.75),
                high_sigma_mult=cfg.get("high_sigma_mult", 0.45),
                use_gabor_energy=cfg.get("use_gabor_energy", True),
                bridge_mode=cfg.get("bridge_mode", False),
                enabled=enabled,
            )
        else:
            raise ValueError(f"Unknown noise schedule: {name}")

    def _build_loss(self, name: str, cfg: Dict[str, Any]) -> LossTerm:
        enabled = cfg.get("enabled", True)
        weight = cfg.get("weight", 1.0)
        if not enabled:
            from .loss_terms.base import DisabledLossTerm
            return DisabledLossTerm(name=name, weight=weight)
        if name == "base_diffusion":
            raise ValueError(
                "Loss 'base_diffusion' is always on and handled in SLMFBBDM.forward; "
                "remove it from the configurable loss list."
            )
        elif name == "topk_lesion":
            from .loss_terms.topk import TopKLesionLoss
            return TopKLesionLoss(
                topk_percent=cfg.get("topk_percent", 0.01),
                active_tau_max=cfg.get("active_tau_max", 0.25),
                enabled=enabled, weight=weight,
            )
        elif name == "focal_frequency":
            from .loss_terms.frequency import FocalFrequencyLoss
            return FocalFrequencyLoss(
                alpha=cfg.get("alpha", 1.0),
                active_tau_max=cfg.get("active_tau_max", 0.4),
                enabled=enabled, weight=weight,
            )
        elif name == "roi_suv":
            from .loss_terms.roi_suv import ROISUVLoss
            return ROISUVLoss(
                active_tau_max=cfg.get("active_tau_max", 0.25),
                enabled=enabled, weight=weight,
            )
        elif name == "false_hotspot":
            from .loss_terms.false_hotspot import FalseHotspotLoss
            return FalseHotspotLoss(
                active_tau_max=cfg.get("active_tau_max", 0.3),
                enabled=enabled, weight=weight,
            )
        elif name == "heteroscedastic_nll":
            from .loss_terms.heteroscedastic import HeteroscedasticNLLLoss
            return HeteroscedasticNLLLoss(
                logvar_min=self.heteroscedastic_logvar_min,
                logvar_max=self.heteroscedastic_logvar_max,
                enabled=enabled, weight=weight,
            )
        elif name == "hotspot_prior":
            from .loss_terms.hotspot import HotspotPriorLoss
            return HotspotPriorLoss(
                active_tau_min=cfg.get("active_tau_min", 0.25),
                active_tau_max=cfg.get("active_tau_max", 0.75),
                focal_gamma=cfg.get("focal_gamma", 2.0),
                dice_weight=cfg.get("dice_weight", 1.0),
                focal_weight=cfg.get("focal_weight", 1.0),
                distance_weight=cfg.get("distance_weight", 0.25),
                pet_threshold_quantile=cfg.get("pet_threshold_quantile", 0.95),
                enabled=enabled, weight=weight,
            )
        elif name == "organ_consistency":
            from .loss_terms.organ_consistency import OrganConsistencyLoss
            return OrganConsistencyLoss(
                active_tau_min=cfg.get("active_tau_min", 0.25),
                cold_weight=cfg.get("cold_weight", 0.02),
                enabled=enabled, weight=weight,
            )
        elif name == "segmenter_consistency":
            from .loss_terms.segmenter_consistency import SegmenterConsistencyLoss
            return SegmenterConsistencyLoss(
                segmenter=self.segmenter,
                active_tau_max=cfg.get("active_tau_max", 0.3),
                enabled=enabled, weight=weight,
            )
        elif name == "patch_nce":
            from .loss_terms.patch_nce import PatchNCELoss
            return PatchNCELoss(
                patch_size=cfg.get("patch_size", 3),
                num_patches=cfg.get("num_patches", 256),
                temperature=cfg.get("temperature", 0.07),
                active_tau_max=cfg.get("active_tau_max", 0.6),
                enabled=enabled, weight=weight,
            )
        else:
            from .loss_terms.base import DisabledLossTerm
            return DisabledLossTerm(name=name, weight=weight)

    def _model_input(
        self,
        noisy_x: torch.Tensor,
        x_source: torch.Tensor,
        self_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.self_conditioning:
            return torch.cat([noisy_x, x_source], dim=1)
        if self_cond is None:
            self_cond = torch.zeros_like(noisy_x)
        return torch.cat([noisy_x, x_source, self_cond], dim=1)

    def _min_snr_weight(self, timesteps: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if not self.min_snr_enabled:
            return torch.ones(timesteps.shape[0], device=ref.device, dtype=ref.dtype)

        if hasattr(self.noise_schedule, "alphas_cumprod"):
            alpha = self.noise_schedule.alphas_cumprod[timesteps].to(device=ref.device, dtype=ref.dtype)
            snr = alpha / (1.0 - alpha).clamp_min(1e-8)
        elif hasattr(self.noise_schedule, "m_t") and hasattr(self.noise_schedule, "sigma_t"):
            m = self.noise_schedule.m_t[timesteps].to(device=ref.device, dtype=ref.dtype)
            sigma = self.noise_schedule.sigma_t[timesteps].to(device=ref.device, dtype=ref.dtype)
            snr = (1.0 - m).square() / sigma.square().clamp_min(1e-8)
        else:
            return torch.ones(timesteps.shape[0], device=ref.device, dtype=ref.dtype)

        gamma = torch.as_tensor(self.min_snr_gamma, device=ref.device, dtype=ref.dtype)
        return torch.minimum(snr, gamma) / snr.clamp_min(1e-8)

    def _base_reconstruction_loss(
        self,
        pred_x0: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        tau: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        reduce_dims = tuple(range(1, pred_x0.dim()))

        # ---- τ-dependent stage weights ----
        # early (τ→1): focus on global intensity → high MSE, low gradient
        # late  (τ→0): focus on fine edges       → low MSE, high gradient
        # Smooth transitions via sigmoid gates.
        tau_f = tau.float()
        w_mse  = 0.5 + 1.5 * torch.sigmoid(10.0 * (tau_f - 0.43))    # 2.0→0.5
        w_l1   = torch.ones_like(tau_f)                                 # 1.0 constant
        w_grad = 0.2 * torch.sigmoid(10.0 * (0.25 - tau_f))            # 0→0.2
        mse = (pred_x0 - target).square().mean(dim=reduce_dims)
        l1 = (pred_x0 - target).abs().mean(dim=reduce_dims)
        grad = _image_gradient_l1_per_sample(pred_x0, target)
        per_sample = (
            self.base_mse_weight * w_mse * mse
            + self.base_l1_weight * w_l1 * l1
            + self.base_gradient_weight * w_grad * grad
        )
        min_snr = self._min_snr_weight(timesteps, pred_x0)
        loss = (per_sample * min_snr).mean()
        return loss, {
            "loss/base_mse": mse.mean().detach(),
            "loss/base_l1": l1.mean().detach(),
            "loss/base_gradient": grad.mean().detach(),
            "loss/min_snr_weight": min_snr.mean().detach(),
            "loss/tau_mse_weight": w_mse.mean().detach(),
            "loss/tau_grad_weight": w_grad.mean().detach(),
        }

    # ------------------------------------------------------------------
    # Condition bundle assembly
    # ------------------------------------------------------------------

    def build_condition_bundle(
        self,
        batch: Dict[str, torch.Tensor],
        timesteps: torch.Tensor,
    ) -> ConditionBundle:
        """Run CT Encoder + all enabled prior modules, merge into one bundle."""
        bundle = ConditionBundle.empty()
        bundle.maps["ct"] = batch["ct"]

        # CT Encoder → multi-scale features (always run, needed for adapter)
        ct_feats = self.ct_encoder(batch["ct"])
        for i, f in enumerate(ct_feats):
            bundle.maps[f"ct_feat_{i}"] = f

        for name, prior in self.priors.items():
            prior_bundle = prior(batch, timesteps, partial_bundle=bundle)
            bundle.merge(prior_bundle)

        return bundle

    def _build_skip_injections(
        self,
        condition: ConditionBundle,
        timesteps: Optional[torch.Tensor] = None,
        hw_list: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """Build Zero-Conv adapter outputs for UNet skip connections.

        When adapter is disabled, returns empty injections (no-op).
        Applies time-varying beta modulation from the adapter.
        """
        if not self.zero_adapter_enabled:
            return []

        ct_feats = [condition.maps[f"ct_feat_{i}"] for i in range(4)]

        organ_feats = [
            condition.maps.get("organ_feat_1"),
            condition.maps.get("organ_feat_2"),
            condition.maps.get("organ_feat_3"),
            None,  # L3 uses nearest organ feat
        ]
        gabor_feat = condition.maps.get("gabor_feat")
        hotspot = condition.maps.get("hotspot_prior")

        if hw_list is None:
            hw_list = [ct_feats[0].shape[2], ct_feats[0].shape[2] // 2,
                       ct_feats[0].shape[2] // 4, ct_feats[0].shape[2] // 8]

        tau = self.noise_schedule.get_tau(timesteps) if timesteps is not None else None
        injections = self.adapter.get_zero_conv_outputs(
            ct_feats, organ_feats, gabor_feat, hotspot, hw_list, tau=tau,
        )
        # Reverse: adapter outputs [shallow→deep], decoder consumes [deep→shallow]
        return list(reversed(injections)) if injections else []

    # ------------------------------------------------------------------
    # Forward pass (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        timesteps: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """One training step.

        Returns (total_loss, log_dict).
        """
        device = batch["ct"].device
        B = batch["ct"].shape[0]
        x0 = batch["pet"]         # target PET
        x_source = batch["ct"]    # source CT
        noise = torch.randn_like(x0)

        # Sample timesteps if not given
        if timesteps is None:
            T = self.noise_schedule.num_train_timesteps
            timesteps = torch.randint(0, T, (B,), device=device)

        # 1. Build condition bundle
        condition = self.build_condition_bundle(batch, timesteps)

        # 2. Forward diffusion (add noise)
        # BBDM bridge needs x_source; DDPM ignores it
        noisy_x = _add_noise(self.noise_schedule, x0, noise, timesteps, condition, x_source)

        # 3. Condition dropout (training only) — MUST happen BEFORE adapter reads conditions
        condition = self.condition_dropout.apply(condition, training=self.training)

        # 4. Build skip injections from adapter (Zero-Conv or NoOp)
        #    Dropout already applied → adapter sees zeroed-out conditions for dropped modules
        skip_injections = self._build_skip_injections(condition, timesteps=timesteps)

        # 4.5 Build metadata tensor + Cross-Attn beta
        meta_tensor = None
        if self.meta_enabled:
            meta_tensor = _meta_to_tensor(batch.get("meta"), B, device, x0.dtype)

        # Cross-Attention time-varying beta: strongest early (τ→1), fades late (τ→0)
        tau = self.noise_schedule.get_tau(timesteps)
        ca_beta = multi_level_betas(tau, levels=5)[:, 0]  # bottleneck column

        semantic_tokens = condition.tokens.get("semantic")
        self_cond = None
        if self.self_conditioning:
            self_cond = torch.zeros_like(x0)
            use_self_cond = self.training and torch.rand((), device=device) < self.self_conditioning_prob
            if use_self_cond:
                with torch.no_grad():
                    sc_input = self._model_input(noisy_x, x_source, self_cond)
                    sc_output = self.unet(
                        sc_input,
                        timesteps,
                        context_tokens=semantic_tokens,
                        skip_injections=skip_injections if skip_injections else None,
                        meta=meta_tensor,
                        ca_beta=ca_beta,
                    )
                    self_cond = sc_output[:, :1].detach()

        # 5. UNet forward
        model_input = self._model_input(noisy_x, x_source, self_cond)
        output = self.unet(model_input, timesteps, context_tokens=semantic_tokens,
                          skip_injections=skip_injections if skip_injections else None,
                          meta=meta_tensor,
                          ca_beta=ca_beta)

        # 5. Split output
        if self.enable_heteroscedastic:
            pred_x0, pred_logvar = output[:, :1], output[:, 1:]
            pred_logvar = pred_logvar.clamp(self.heteroscedastic_logvar_min, self.heteroscedastic_logvar_max)
        else:
            pred_x0, pred_logvar = output[:, :1], None

        # 6. Compute losses
        tau = self.noise_schedule.get_tau(timesteps)
        ctx = LossContext(
            model_pred=pred_x0,
            loss_target=x0,
            target_pet=x0,
            pred_x0=pred_x0,
            timesteps=timesteps,
            tau=tau,
            batch=batch,
            condition=condition,
            pred_logvar=pred_logvar,
        )

        total_loss = torch.tensor(0.0, device=device)
        logs: Dict[str, torch.Tensor] = {}

        # Base diffusion/reconstruction loss (always on)
        base_loss, base_logs = self._base_reconstruction_loss(pred_x0, x0, timesteps, tau)
        total_loss = total_loss + base_loss
        logs["loss/base_diffusion"] = base_loss.detach()
        logs.update(base_logs)

        # Pluggable loss terms
        for name, term in self.loss_terms.items():
            loss_val, loss_logs = term(ctx)
            total_loss = total_loss + loss_val
            for k, v in loss_logs.items():
                logs[f"loss/{k}"] = v.detach() if torch.is_tensor(v) else v

        logs["loss/total"] = total_loss.detach()

        # Module status logs
        for name in self.priors:
            logs[f"module/{name}"] = torch.tensor(
                1.0 if self.priors[name].enabled else 0.0, device=device
            )
        logs["module/zero_adapter"] = torch.tensor(1.0 if self.zero_adapter_enabled else 0.0, device=device)
        logs["module/condition_dropout"] = torch.tensor(
            1.0 if self.condition_dropout.enabled else 0.0, device=device
        )
        logs["module/self_conditioning"] = torch.tensor(1.0 if self.self_conditioning else 0.0, device=device)
        logs["module/metadata_film"] = torch.tensor(1.0 if self.meta_enabled else 0.0, device=device)
        logs["module/segmenter"] = torch.tensor(1.0 if self.segmenter_enabled else 0.0, device=device)
        logs["module/scale_adaptive_noise"] = torch.tensor(
            1.0 if getattr(self.noise_schedule, "name", "") == "scale_adaptive_noise"
            and self.noise_schedule.enabled else 0.0,
            device=device,
        )

        return total_loss, logs

    # ------------------------------------------------------------------
    # Sampling (inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        batch: Dict[str, torch.Tensor],
        num_steps: Optional[int] = None,
        progress: bool = False,
        cfg_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """DDIM-style sampling with optional weak Classifier-Free Guidance.

        Args:
            cfg_scale: CFG scale [1.0, 1.5].  1.0 = no CFG.
                       Higher = stronger condition influence.
                       Keep ≤1.5 to avoid hallucinating lesions.
        """
        if self.sample_scheduler != "ddim":
            import warnings
            warnings.warn(
                f"sample_scheduler='{self.sample_scheduler}' is not implemented; falling back to DDIM."
            )

        steps = num_steps or self.eval_sampling_steps
        device = batch["ct"].device
        B = batch["ct"].shape[0]
        x_source = batch["ct"]

        # Build metadata tensor once (shared across all denoising steps)
        meta_tensor = None
        if self.meta_enabled:
            meta_tensor = _meta_to_tensor(batch.get("meta"), B, device, x_source.dtype)

        T = self.noise_schedule.num_train_timesteps
        ddim_timesteps = torch.linspace(T - 1, 0, steps, device=device, dtype=torch.long)
        is_bbdm = hasattr(self.noise_schedule, 'm_t')
        has_custom_reverse = hasattr(self.noise_schedule, "step_from_prediction")

        # ---- Initial state ----
        noise = torch.randn_like(x_source)
        timesteps_T = torch.full((B,), T - 1, device=device, dtype=torch.long)
        condition = self.build_condition_bundle(batch, timesteps_T)
        skip_inj = self._build_skip_injections(condition, timesteps=timesteps_T)

        if is_bbdm:
            # x_T = m_T·CT + (1-m_T)·0 + σ_T·ε ≈ CT + noise
            x_t = _add_noise(self.noise_schedule, torch.zeros_like(x_source),
                            noise, timesteps_T, condition, x_source)
        elif has_custom_reverse:
            # Custom schedules may define a non-isotropic forward prior state.
            x_t = self.noise_schedule.add_noise(torch.zeros_like(x_source), noise, timesteps_T, condition)
        else:
            # Standard: x_T = noise
            x_t = noise
        self_cond = torch.zeros_like(x_t) if self.self_conditioning else None

        step_range = range(len(ddim_timesteps))
        if progress:
            from tqdm import tqdm
            step_range = tqdm(step_range, desc="Sampling")

        for i in step_range:
            t = ddim_timesteps[i]
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            condition = self.build_condition_bundle(batch, t_batch)
            skip_inj = self._build_skip_injections(condition, timesteps=t_batch)
            semantic_tokens = condition.tokens.get("semantic")
            model_input = self._model_input(x_t, x_source, self_cond)
            ca_beta_step = multi_level_betas(t_batch.float() / self.noise_schedule.num_train_timesteps, levels=5)[:, 0]
            output = self.unet(model_input, t_batch, context_tokens=semantic_tokens,
                              skip_injections=skip_inj if skip_inj else None,
                              meta=meta_tensor,
                              ca_beta=ca_beta_step)

            pred_x0 = output[:, :1]

            # ---- Weak CFG: blend cond + uncond predictions ----
            if cfg_scale > 1.0:
                # Build null conditions (all zero)
                skip_inj_null = [torch.zeros_like(s) for s in skip_inj] if skip_inj else None
                null_output = self.unet(model_input, t_batch,
                                       context_tokens=None,
                                       skip_injections=skip_inj_null,
                                       meta=torch.zeros_like(meta_tensor) if meta_tensor is not None else None,
                                       ca_beta=torch.zeros_like(ca_beta_step))
                pred_x0_uncond = null_output[:, :1]
                pred_x0 = pred_x0_uncond + cfg_scale * (pred_x0 - pred_x0_uncond)

            if self.self_conditioning:
                self_cond = pred_x0.detach()

            # DDIM step to next timestep
            if i < len(ddim_timesteps) - 1:
                t_next = ddim_timesteps[i + 1]
                t_next_batch = torch.full_like(t_batch, t_next)
                if has_custom_reverse:
                    x_t = self.noise_schedule.step_from_prediction(
                        x_t,
                        pred_x0,
                        t_batch,
                        t_next_batch,
                        condition,
                        x_source=x_source,
                    )
                elif is_bbdm:
                    # BBDM: x_{t-1} = m_{t-1}·CT + (1-m_{t-1})·pred_x0
                    # As t→0: m→0, x→pred_x0=PET ✓
                    m_next = self.noise_schedule.m_t[t_next_batch]
                    while m_next.dim() < x_t.dim():
                        m_next = m_next.unsqueeze(-1)
                    x_t = m_next * x_source + (1 - m_next) * pred_x0
                else:
                    # Standard DDIM
                    alpha_t = _get_alpha_cumprod(self.noise_schedule, t_batch)
                    alpha_next = _get_alpha_cumprod(self.noise_schedule, t_next_batch)
                    while alpha_t.dim() < x_t.dim():
                        alpha_t = alpha_t.unsqueeze(-1)
                        alpha_next = alpha_next.unsqueeze(-1)
                    eps_pred = (x_t - alpha_t.sqrt() * pred_x0) / (1 - alpha_t).sqrt().clamp_min(1e-8)
                    x_t = alpha_next.sqrt() * pred_x0 + (1 - alpha_next).sqrt() * eps_pred

        result = {"synthetic_pet": x_t}
        if self.enable_heteroscedastic:
            result["logvar"] = output[:, 1:].clamp(
                self.heteroscedastic_logvar_min, self.heteroscedastic_logvar_max)
        if "hotspot_prior" in condition.maps:
            result["hotspot_prior"] = condition.maps["hotspot_prior"]

        return result

    # ------------------------------------------------------------------
    # MC sampling (epistemic uncertainty)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_mc(
        self,
        batch: Dict[str, torch.Tensor],
        n_samples: int = 20,
        num_steps: Optional[int] = None,
        progress: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Monte Carlo sampling: n_samples independent forward passes.

        Returns:
            synthetic_pet       [B, 1, H, W]  mean prediction
            epistemic_var       [B, 1, H, W]  variance across MC samples
            aleatoric_logvar    [B, 1, H, W]  heteroscedastic log-variance (if enabled)
            confidence_map      [B, 1, H, W]  combined confidence (0=low, 1=high)
            samples             [n, B, 1, H, W]  all individual samples
        """
        all_samples: List[torch.Tensor] = []
        all_logvars: List[torch.Tensor] = []

        step_range = range(n_samples)
        if progress:
            from tqdm import tqdm
            step_range = tqdm(step_range, desc="MC Sampling")

        for _ in step_range:
            result = self.sample(batch, num_steps=num_steps, progress=False)
            all_samples.append(result["synthetic_pet"])
            if "logvar" in result:
                all_logvars.append(result["logvar"])

        samples = torch.stack(all_samples, dim=0)  # [N, B, 1, H, W]
        mean = samples.mean(dim=0)                  # [B, 1, H, W]
        epistemic_var = samples.var(dim=0)          # [B, 1, H, W]

        output: Dict[str, torch.Tensor] = {
            "synthetic_pet": mean,
            "epistemic_var": epistemic_var,
            "samples": samples,
        }

        # Combine aleatoric + epistemic for total confidence
        if all_logvars:
            aleatoric_logvar = torch.stack(all_logvars, dim=0).mean(dim=0)
            aleatoric_var = aleatoric_logvar.exp()
            total_var = epistemic_var + aleatoric_var

            # Confidence map: inverse of normalised total uncertainty
            var_max = total_var.amax(dim=(2, 3), keepdim=True).clamp_min(1e-8)
            confidence = 1.0 - (total_var / var_max)

            output["aleatoric_logvar"] = aleatoric_logvar
            output["aleatoric_var"] = aleatoric_var
            output["total_var"] = total_var
            output["confidence_map"] = confidence

        return output

    # ------------------------------------------------------------------
    # Config-driven construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> SLMFBBDM:
        model_cfg = config.get("model", {})
        modules_cfg = config.get("modules", {})
        data_cfg = config.get("data", {})
        loss_cfg = config.get("losses", {})
        prior_cfgs = {name: modules_cfg.get(name, {}) for name in _PRIOR_MODULE_NAMES}

        return cls(
            image_size=data_cfg.get("image_size", 192),
            objective=model_cfg.get("objective", "pred_x0"),
            enable_heteroscedastic=model_cfg.get("enable_heteroscedastic", True),
            heteroscedastic_logvar_min=model_cfg.get("heteroscedastic_logvar_min", -6.0),
            heteroscedastic_logvar_max=model_cfg.get("heteroscedastic_logvar_max", 2.0),
            prior_configs=prior_cfgs,
            adapter_config=modules_cfg.get("zero_adapter", {}),
            noise_config=resolve_noise_config(modules_cfg),
            loss_configs=loss_cfg,
            condition_dropout_config=modules_cfg.get("condition_dropout", {}),
            base_loss_config=model_cfg.get("base_loss", {}),
            meta_config=model_cfg.get("metadata", config.get("metadata", {})),
            segmenter_config=model_cfg.get("segmenter", config.get("segmenter", {})),
            self_conditioning_config=model_cfg.get("self_conditioning", {}),
            sample_scheduler=model_cfg.get("sample_scheduler", "ddim"),
            eval_sampling_steps=config.get("runtime", {}).get("eval_sampling_steps", 20),
        )

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
