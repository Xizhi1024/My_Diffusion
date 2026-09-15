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

from pathlib import Path
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
    noise_keys = ("scale_adaptive_noise", "bbdm_bridge", "noise")
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


def _bbdm_ddim_step(
    schedule,
    x_t: torch.Tensor,
    pred_x0: torch.Tensor,
    x_source: torch.Tensor,
    timesteps: torch.Tensor,
    next_timesteps: torch.Tensor,
) -> torch.Tensor:
    """Deterministic BBDM step that preserves the inferred noise trajectory.

    The forward bridge is ``x_t = m_t*x_source + (1-m_t)*x_0 + sigma_t*eps``.
    Reusing the inferred ``eps`` at the next timestep keeps reverse states on
    the noisy distribution seen during training.  Dropping that term makes
    every state after the first reverse step nearly clean, which is especially
    harmful to frequency-dependent conditioning.
    """
    m_t = schedule.m_t[timesteps].to(device=x_t.device, dtype=x_t.dtype)
    m_next = schedule.m_t[next_timesteps].to(device=x_t.device, dtype=x_t.dtype)
    sigma_t = schedule.sigma_t[timesteps].to(device=x_t.device, dtype=x_t.dtype)
    sigma_next = schedule.sigma_t[next_timesteps].to(
        device=x_t.device, dtype=x_t.dtype
    )
    while m_t.dim() < x_t.dim():
        m_t = m_t.unsqueeze(-1)
        m_next = m_next.unsqueeze(-1)
        sigma_t = sigma_t.unsqueeze(-1)
        sigma_next = sigma_next.unsqueeze(-1)

    eps_hat = (
        x_t - m_t * x_source - (1.0 - m_t) * pred_x0
    ) / sigma_t.clamp_min(1e-6)
    return (
        m_next * x_source
        + (1.0 - m_next) * pred_x0
        + sigma_next * eps_hat
    )


def _rc_brd_base_log_snr(u: torch.Tensor, sigma_bridge: float,
                         lambda_min: float, lambda_max: float) -> torch.Tensor:
    """λ₀(u)=clip(log((1−u)/(ν²·u)),λmin,λmax)，ν²=2·sigma_bridge².

    DESIGN §4 v1.0f clock-lookup convention ([计划] v1.5 §3.6; the v1.4
    linear λ warp is deprecated).  Shared by the schedule's warped clock and
    the integration-side runtime c lookup so the two cannot drift apart.
    Endpoints are clamped before the log (λ₀ diverges at u∈{0,1}).
    """
    nu_sq = 2.0 * float(sigma_bridge) ** 2
    u = u.clamp(1e-12, 1.0 - 1e-12)
    return torch.log((1.0 - u) / (nu_sq * u)).clamp(float(lambda_min), float(lambda_max))


def _rc_brd_group_shares(band_groups: Dict[str, Any]) -> Dict[str, float]:
    """Per-group Haar-coefficient shares n_g/N (DESIGN §8 v1.0g; AUDIT 5 §3.4).

    The mean corruption density rho-bar must be averaged over *Haar
    coefficients*, not over groups: a level-2 band holds 1 unit of
    coefficients and a level-1 band holds 4 units (4^(2-level)), so the
    canonical 3-group split yields LL2:mid:high = 1:3:12 out of 16 — derived
    from the band names listed in wavelet.band_groups (trailing digit =
    level), never hardcoded.
    """
    units: Dict[str, float] = {}
    total = 0.0
    for group, bands in band_groups.items():
        group_units = 0.0
        for band in bands:
            name = str(band)
            if len(name) < 2 or not name[-1].isdigit() or int(name[-1]) not in (1, 2):
                raise ValueError(
                    f"band name {name!r} lacks a Haar-level suffix in (1, 2); "
                    "shares are derived from the canonical band layout")
            group_units += float(4 ** (2 - int(name[-1])))
        units[str(group)] = group_units
        total += group_units
    if total <= 0.0:
        raise ValueError("band_groups must list at least one band")
    return {group: value / total for group, value in units.items()}


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
        initialization_seed: Optional[int] = None,
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
        conditional_mean_config: Optional[Dict[str, Any]] = None,
        residual_bridge_config: Optional[Dict[str, Any]] = None,
        residual_frequency_config: Optional[Dict[str, Any]] = None,
        rc_brd_config: Optional[Dict[str, Any]] = None,
        rc_brd_expected_fold: Optional[str] = None,
        wavelet_unet_config: Optional[Dict[str, Any]] = None,
        meta_config: Optional[Dict[str, Any]] = None,
        segmenter_config: Optional[Dict[str, Any]] = None,
        expected_data_lineage: Optional[Dict[str, Any]] = None,
        require_checkpoint_lineage: bool = False,
        # Inference
        sample_scheduler: str = "ddim",
        eval_sampling_steps: int = 20,
    ):
        super().__init__()
        self.image_size = image_size
        self.initialization_seed = initialization_seed
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
        self._min_snr_reference_mean: Optional[torch.Tensor] = None

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

        # ---- Gabor routing ----
        # Real, independently-switchable routes.  These are the *only* knobs that
        # decide where Gabor features flow.  The YAML fields below are read here
        # (not just declared) and every consumer (adapter / noise / hotspot /
        # loss) must consult the corresponding route.
        gabor_cfg = prior_cfgs.get("gabor", {})
        self.gabor_routes: Dict[str, bool] = {
            "enabled":          bool(gabor_cfg.get("enabled", False)),
            "inject_adapter":   bool(gabor_cfg.get("inject_adapter", False)),
            "use_for_noise":    bool(gabor_cfg.get("use_for_noise", False)),
            "use_for_hotspot":  bool(gabor_cfg.get("use_for_hotspot", False)),
            "use_for_loss":     bool(gabor_cfg.get("use_for_loss", False)),
        }

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

        # ---- Conditional-mean residual bridge ----
        mean_cfg = conditional_mean_config or {}
        residual_cfg = residual_bridge_config or {}
        frequency_cfg = residual_frequency_config or {}
        self.conditional_mean_enabled = bool(mean_cfg.get("enabled", False))
        self.residual_bridge_enabled = bool(residual_cfg.get("enabled", False))
        self.residual_frequency_enabled = bool(frequency_cfg.get("enabled", False))
        self.residual_frequency_frozen = bool(frequency_cfg.get("freeze", False))
        self.residual_frequency_mode = str(frequency_cfg.get("mode", "legacy"))
        dct_descriptor_cfg = frequency_cfg.get("dct_descriptor", {})
        gabor_descriptor_cfg = frequency_cfg.get("gabor_descriptor", {})
        cross_level_router_cfg = frequency_cfg.get("cross_level_router", {})
        ct_support_cfg = frequency_cfg.get("ct_support_head", {})
        self.mean_checkpoint = mean_cfg.get("checkpoint")
        self.mean_architecture = str(
            mean_cfg.get("architecture", "low_frequency")
        ).strip().lower()
        self.mean_checkpoint_state_prefix = str(
            mean_cfg.get("checkpoint_state_prefix", "")
        )
        self.mean_checkpoint_format = str(
            mean_cfg.get("checkpoint_format", "mean")
        ).strip().lower()
        self.mean_frozen = bool(mean_cfg.get("freeze", False))
        self.mean_detach_bridge = bool(mean_cfg.get("detach_bridge", True))
        self.mean_loss_weight = float(mean_cfg.get("loss_weight", 1.0))
        self.mean_charbonnier_eps = float(mean_cfg.get("charbonnier_eps", 1e-3))

        if self.mean_checkpoint_format not in {"mean", "full_model", "comparison"}:
            raise ValueError(
                "modules.conditional_mean.checkpoint_format must be "
                "'mean', 'full_model', or 'comparison'"
            )
        if self.conditional_mean_enabled and self.mean_frozen and not self.mean_checkpoint:
            raise ValueError(
                "modules.conditional_mean.freeze=true requires a pretrained checkpoint"
            )
        if self.residual_bridge_enabled and not self.conditional_mean_enabled:
            raise ValueError("modules.residual_bridge requires modules.conditional_mean.enabled=true")
        if self.residual_bridge_enabled and getattr(self.noise_schedule, "name", "") != "bbdm_bridge":
            raise ValueError("modules.residual_bridge requires the bbdm_bridge noise schedule")
        if self.residual_frequency_enabled and not self.residual_bridge_enabled:
            raise ValueError("modules.residual_frequency requires modules.residual_bridge.enabled=true")
        if self.residual_frequency_mode not in {
            "legacy",
            "boundary_reliable",
            "spectral_evidence_router",
        }:
            raise ValueError(
                "modules.residual_frequency.mode must be 'legacy', "
                "'boundary_reliable', or 'spectral_evidence_router'"
            )
        use_gabor_gate = bool(frequency_cfg.get("use_gabor_gate", False))
        use_directional_reliability = bool(
            frequency_cfg.get("use_directional_reliability", False)
        )
        if (
            (use_gabor_gate and self.residual_frequency_mode == "legacy")
            or (
                use_directional_reliability
                and self.residual_frequency_mode == "boundary_reliable"
            )
        ) and not self.gabor_routes.get("enabled", False):
            raise ValueError("Residual-frequency Gabor gating requires modules.gabor.enabled=true")
        if (
            self.residual_frequency_enabled
            and self.residual_frequency_mode == "spectral_evidence_router"
            and bool(gabor_descriptor_cfg.get("enabled", True))
            and not self.gabor_routes.get("enabled", False)
        ):
            raise ValueError(
                "Gabor evidence requires modules.gabor.enabled=true"
            )
        configured_losses = loss_configs or {}
        residual_wavelet_cfg = configured_losses.get("residual_wavelet", {})
        if residual_wavelet_cfg.get("enabled", False) and not self.residual_bridge_enabled:
            raise ValueError("losses.residual_wavelet requires modules.residual_bridge.enabled=true")
        gabor_loss_cfg = configured_losses.get("gabor_consistency", {})
        if gabor_loss_cfg.get("enabled", False) and not self.gabor_routes.get("use_for_loss", False):
            raise ValueError(
                "losses.gabor_consistency requires modules.gabor.use_for_loss=true"
            )
        route_utility_cfg = configured_losses.get(
            "route_utility_supervision",
            {},
        )
        if (
            route_utility_cfg.get("enabled", False)
            and not bool(
                cross_level_router_cfg.get(
                    "spatial_destination_enabled",
                    False,
                )
            )
        ):
            raise ValueError(
                "losses.route_utility_supervision requires "
                "cross_level_router.spatial_destination_enabled=true"
            )

        # Prefix-stripped mean state: RC-BRD's fail-closed SHA anchor (§3 v1.0b, §9.4).
        self._rc_brd_mean_state: Optional[Dict[str, torch.Tensor]] = None
        self.mean_predictor: Optional[nn.Module] = None
        if self.conditional_mean_enabled:
            from .mean_predictor import build_mean_predictor
            self.mean_predictor = build_mean_predictor(mean_cfg)
            if self.mean_checkpoint:
                checkpoint_path = Path(self.mean_checkpoint)
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(
                        f"Conditional-mean checkpoint not found: {checkpoint_path}"
                    )
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
                if not isinstance(checkpoint, dict):
                    raise ValueError("Conditional-mean checkpoint must contain a mapping")
                from src.data.lineage import validate_checkpoint_data_lineage
                validate_checkpoint_data_lineage(
                    checkpoint,
                    expected_data_lineage,
                    required=require_checkpoint_lineage,
                    context=f"conditional-mean checkpoint {checkpoint_path}",
                )
                checkpoint_state = checkpoint.get("model")
                if not isinstance(checkpoint_state, dict):
                    raise ValueError(
                        "Conditional-mean checkpoint is missing the 'model' state dict"
                    )
                if self.mean_checkpoint_format == "mean":
                    if checkpoint.get("format_version") not in (1, 2):
                        raise ValueError(
                            "Unsupported conditional-mean checkpoint format_version: "
                            f"{checkpoint.get('format_version')!r}; expected 1 or 2"
                        )
                    mean_state = checkpoint_state
                elif self.mean_checkpoint_format == "full_model":
                    prefix = "mean_predictor."
                    mean_state = {
                        key[len(prefix):]: value
                        for key, value in checkpoint_state.items()
                        if key.startswith(prefix)
                    }
                    if not mean_state:
                        raise ValueError(
                            "Full-model checkpoint contains no 'mean_predictor.*' tensors"
                        )
                else:
                    mean_state = checkpoint_state
                    if not self.mean_checkpoint_state_prefix:
                        self.mean_checkpoint_state_prefix = "generator."
                if self.mean_checkpoint_state_prefix:
                    prefix = self.mean_checkpoint_state_prefix
                    mean_state = {
                        key[len(prefix):]: value
                        for key, value in mean_state.items()
                        if key.startswith(prefix)
                    }
                    if not mean_state:
                        raise ValueError(
                            "Conditional-mean checkpoint contains no tensors with "
                            f"checkpoint_state_prefix={prefix!r}"
                        )
                self.mean_predictor.load_state_dict(mean_state, strict=True)
                self._rc_brd_mean_state = mean_state
                print(
                    "[SLMF-BBDM] Loaded conditional mean "
                    f"({self.mean_checkpoint_format}) from {checkpoint_path}"
                )
            if self.mean_frozen:
                for parameter in self.mean_predictor.parameters():
                    parameter.requires_grad_(False)
                self.mean_predictor.eval()

        self.residual_output_scale = float(residual_cfg.get("output_scale", 1.0))
        if not 0.0 <= self.residual_output_scale <= 1.0:
            raise ValueError("modules.residual_bridge.output_scale must be in [0, 1]")
        self.residual_output_clip = bool(residual_cfg.get("clip_output", False))

        self.residual_preconditioner: Optional[nn.Module] = None
        if self.residual_frequency_enabled:
            gabor_orientations = frequency_cfg.get(
                "gabor_orientations", gabor_cfg.get("orientations", 8)
            )
            if self.residual_frequency_mode == "legacy":
                from .frequency.residual_preconditioner import ResidualFrequencyPreconditioner

                self.residual_preconditioner = ResidualFrequencyPreconditioner(
                    output_channels=tuple(frequency_cfg.get("output_channels", [256, 256, 128, 64])),
                    band_scales=tuple(frequency_cfg.get("band_scales", [1.0, 0.5, 0.25])),
                    inject_wavelet=frequency_cfg.get("inject_wavelet", True),
                    use_gabor_gate=use_gabor_gate,
                    gabor_orientations=gabor_orientations,
                    gate_strength=frequency_cfg.get("gate_strength", 0.1),
                    state_modulation=frequency_cfg.get("state_modulation", True),
                )
            elif self.residual_frequency_mode == "boundary_reliable":
                from .frequency.boundary_reliable import (
                    BoundaryReliableFrequencyInjector,
                )

                self.residual_preconditioner = BoundaryReliableFrequencyInjector(
                    output_channels=tuple(frequency_cfg.get("output_channels", [256, 256, 128, 64])),
                    band_scales=tuple(frequency_cfg.get("band_scales", [0.5, 0.25])),
                    use_noise_release=frequency_cfg.get("use_noise_release", True),
                    use_ct_reliability=frequency_cfg.get("use_ct_reliability", True),
                    use_content_reliability=frequency_cfg.get(
                        "use_content_reliability", True
                    ),
                    use_subband_gates=frequency_cfg.get("use_subband_gates", True),
                    use_directional_reliability=use_directional_reliability,
                    gabor_orientations=gabor_orientations,
                    gate_max=frequency_cfg.get("gate_max", 0.25),
                    snr_center=frequency_cfg.get("snr_center", 0.0),
                    snr_temperature=frequency_cfg.get("snr_temperature", 2.0),
                    cross_temperature=frequency_cfg.get("cross_temperature", 1.0),
                    content_hidden_channels=frequency_cfg.get(
                        "content_hidden_channels", 16
                    ),
                    ct_reliability_floors=(
                        frequency_cfg.get("ct_reliability_floor_l2", 0.0),
                        frequency_cfg.get("ct_reliability_floor_l1", 0.0),
                    ),
                    use_gabor_agreement=frequency_cfg.get(
                        "use_gabor_agreement", False
                    ),
                    gabor_agreement_alpha=frequency_cfg.get(
                        "gabor_agreement_alpha", 0.10
                    ),
                    gabor_agreement_l2=frequency_cfg.get(
                        "gabor_agreement_l2", False
                    ),
                    gabor_agreement_l1=frequency_cfg.get(
                        "gabor_agreement_l1", True
                    ),
                    detach_gabor_descriptor=frequency_cfg.get(
                        "detach_gabor_descriptor", True
                    ),
                )
            elif self.residual_frequency_mode == "spectral_evidence_router":
                from .frequency.spectral_router import (
                    SpectralEvidenceFrequencyRouter,
                )

                dct_enabled = bool(dct_descriptor_cfg.get("enabled", True))
                if dct_enabled and (
                    dct_descriptor_cfg.get("pooled_size", 8) != 8
                    or dct_descriptor_cfg.get("selected_frequencies", 12) != 12
                ):
                    raise ValueError(
                        "V5 selected DCT requires pooled_size=8 "
                        "and selected_frequencies=12"
                    )
                band_scales = tuple(
                    frequency_cfg.get("band_scales", [0.5, 0.25])[-2:]
                )
                self.residual_preconditioner = SpectralEvidenceFrequencyRouter(
                    output_channels=tuple(
                        frequency_cfg.get(
                            "output_channels", [256, 256, 128, 64]
                        )
                    ),
                    band_scales=band_scales,
                    use_noise_release=frequency_cfg.get("use_noise_release", True),
                    use_ct_reliability=frequency_cfg.get(
                        "use_ct_reliability", True
                    ),
                    use_content_reliability=frequency_cfg.get(
                        "use_content_reliability", True
                    ),
                    use_subband_gates=frequency_cfg.get(
                        "use_subband_gates", True
                    ),
                    gabor_orientations=gabor_orientations,
                    gate_max=frequency_cfg.get("gate_max", 0.25),
                    snr_center=frequency_cfg.get("snr_center", 0.0),
                    snr_temperature=frequency_cfg.get("snr_temperature", 2.0),
                    cross_temperature=frequency_cfg.get("cross_temperature", 1.0),
                    content_hidden_channels=frequency_cfg.get(
                        "content_hidden_channels", 16
                    ),
                    ct_reliability_floors=(
                        frequency_cfg.get("ct_reliability_floor_l2", 0.25),
                        frequency_cfg.get("ct_reliability_floor_l1", 0.50),
                    ),
                    hidden_channels=cross_level_router_cfg.get(
                        "hidden_channels",
                        frequency_cfg.get("router_hidden_channels", 32),
                    ),
                    dct_enabled=dct_enabled,
                    gabor_enabled=gabor_descriptor_cfg.get("enabled", True),
                    cross_level_enabled=cross_level_router_cfg.get(
                        "enabled", True
                    ),
                    hard_all_null=cross_level_router_cfg.get(
                        "hard_all_null", False
                    ),
                    initial_null_probability=cross_level_router_cfg.get(
                        "initial_null_probability", 0.90
                    ),
                    route_policy=cross_level_router_cfg.get(
                        "policy", "learned"
                    ),
                    fixed_prior=tuple(cross_level_router_cfg.get(
                        "fixed_prior", [0.05, 0.05, 0.90]
                    )),
                    native_warmup_epochs=cross_level_router_cfg.get(
                        "native_warmup_epochs", 0
                    ),
                    routing_ramp_epochs=cross_level_router_cfg.get(
                        "routing_ramp_epochs", 0
                    ),
                    ct_support_enabled=ct_support_cfg.get(
                        "enabled", False
                    ),
                    ct_support_band=ct_support_cfg.get(
                        "selected_haar_band", "l2_hh"
                    ),
                    ct_support_direction=ct_support_cfg.get(
                        "response_direction", "-"
                    ),
                    ct_support_only=ct_support_cfg.get(
                        "support_only", False
                    ),
                    amplitude_delta_min=frequency_cfg.get(
                        "amplitude_delta_min", -0.05
                    ),
                    amplitude_delta_max=frequency_cfg.get(
                        "amplitude_delta_max", 0.10
                    ),
                    uncertainty_aware_router_enabled=cross_level_router_cfg.get(
                        "uncertainty_aware_enabled", False
                    ),
                    uncertainty_aware_confidence_threshold=cross_level_router_cfg.get(
                        "uncertainty_aware_confidence_threshold", None
                    ),
                    h3_schedule_path=cross_level_router_cfg.get(
                        "h3_schedule_path", None
                    ),
                    h3_schedule_sha256=cross_level_router_cfg.get(
                        "h3_schedule_sha256", None
                    ),
                    h3_schedule_source=cross_level_router_cfg.get(
                        "h3_schedule_source", "formal_h3_v2"
                    ),
                    h3_repository_root=cross_level_router_cfg.get(
                        "h3_repository_root", "."
                    ),
                    h3_allow_unverified_preview_lineage=cross_level_router_cfg.get(
                        "h3_allow_unverified_preview_lineage", False
                    ),
                    h3_num_train_timesteps=int(
                        self.noise_schedule.num_train_timesteps
                    ),
                    prior_warmup_epochs=cross_level_router_cfg.get(
                        "prior_warmup_epochs", 10
                    ),
                    prior_active_ramp_epochs=cross_level_router_cfg.get(
                        "prior_active_ramp_epochs", 10
                    ),
                    prior_destination_warmup_epochs=cross_level_router_cfg.get(
                        "prior_destination_warmup_epochs", 30
                    ),
                    prior_destination_ramp_epochs=cross_level_router_cfg.get(
                        "prior_destination_ramp_epochs", 10
                    ),
                    prior_anchor_decay_end_epoch=cross_level_router_cfg.get(
                        "prior_anchor_decay_end_epoch", 100
                    ),
                    prior_anchor_final_scale=cross_level_router_cfg.get(
                        "prior_anchor_final_scale", 0.10
                    ),
                    active_logit_delta_max=cross_level_router_cfg.get(
                        "active_logit_delta_max", 2.0
                    ),
                    initial_destination_native_probability=(
                        cross_level_router_cfg.get(
                            "initial_destination_native_probability", 0.95
                        )
                    ),
                    shallow_projection_init_scale=cross_level_router_cfg.get(
                        "shallow_projection_init_scale", 0.01
                    ),
                    destination_bootstrap_probability=(
                        cross_level_router_cfg.get(
                            "destination_bootstrap_probability",
                            0.0,
                        )
                    ),
                    destination_probability_floor=(
                        cross_level_router_cfg.get(
                            "destination_probability_floor",
                            0.0,
                        )
                    ),
                    destination_probability_ceiling=(
                        cross_level_router_cfg.get(
                            "destination_probability_ceiling",
                            1.0,
                        )
                    ),
                    spatial_destination_enabled=(
                        cross_level_router_cfg.get(
                            "spatial_destination_enabled",
                            False,
                        )
                    ),
                    spatial_destination_hidden_channels=(
                        cross_level_router_cfg.get(
                            "spatial_destination_hidden_channels",
                            16,
                        )
                    ),
                    spatial_destination_delta_max=(
                        cross_level_router_cfg.get(
                            "spatial_destination_delta_max",
                            3.0,
                        )
                    ),
                    low_frequency_background_enabled=(
                        frequency_cfg.get(
                            "low_frequency_background_enabled",
                            False,
                        )
                    ),
                    low_frequency_gate_max=frequency_cfg.get(
                        "low_frequency_gate_max",
                        0.10,
                    ),
                    low_frequency_projection_init_scale=(
                        frequency_cfg.get(
                            "low_frequency_projection_init_scale",
                            0.005,
                        )
                    ),
                )
            else:
                raise ValueError(
                    "modules.residual_frequency.mode must be 'legacy', "
                    "'boundary_reliable', or 'spectral_evidence_router'"
                )
        destination_mode = str(
            cross_level_router_cfg.get("destination_mode", "learned")
        ).strip().lower()
        if destination_mode != "learned":
            if (
                self.residual_preconditioner is None
                or self.residual_frequency_mode != "spectral_evidence_router"
            ):
                raise ValueError(
                    "modules.residual_frequency.cross_level_router."
                    "destination_mode requires the spectral evidence router"
                )
            self.residual_preconditioner.set_inference_destination_intervention(
                destination_mode
            )
        self.residual_frequency_destination_mode = destination_mode
        if self.residual_frequency_frozen:
            if self.residual_preconditioner is None:
                raise ValueError(
                    "modules.residual_frequency.freeze=true requires "
                    "modules.residual_frequency.enabled=true"
                )
            for parameter in self.residual_preconditioner.parameters():
                parameter.requires_grad_(False)
            self.residual_preconditioner.eval()
        self._last_frequency_diagnostics: Dict[str, torch.Tensor] = {}

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
        wavelet_cfg = wavelet_unet_config or {}
        self.wavelet_unet_enabled = bool(wavelet_cfg.get("enabled", False))
        unet_class = BBDMUNet
        unet_kwargs = {
            "in_channels": 3 if self.self_conditioning else 2,
            "enable_heteroscedastic": enable_heteroscedastic,
            "ca_kv_dim": 64,
            "meta_dim": meta_dim,
        }
        if self.wavelet_unet_enabled:
            from .wavelet_unet import WaveletBBDMUNet

            unet_class = WaveletBBDMUNet
            unet_kwargs["mix_kernel_size"] = int(
                wavelet_cfg.get("mix_kernel_size", 3)
            )
        if self.initialization_seed is None:
            self.unet = unet_class(**unet_kwargs)
        else:
            # Optional modules are constructed before the U-Net and therefore
            # consume different amounts of RNG state across ablation variants.
            # Isolate U-Net construction so its initialization remains paired.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.initialization_seed)
                self.unet = unet_class(**unet_kwargs)

        # ---- Loss stack ----
        loss_cfgs = loss_configs or {}
        self.loss_terms = nn.ModuleDict()
        self._training_epoch_index = 0
        self.loss_epoch_ramps: Dict[str, tuple[int, int]] = {}
        self.loss_epoch_windows: Dict[str, tuple[int, int]] = {}
        for name, cfg in loss_cfgs.items():
            self.loss_terms[name] = self._build_loss(name, cfg)
            ramp = cfg.get("epoch_warmup")
            if ramp is not None:
                if (
                    not isinstance(ramp, (list, tuple))
                    or len(ramp) != 2
                ):
                    raise ValueError(
                        f"losses.{name}.epoch_warmup must be [start, end]"
                    )
                start, end = int(ramp[0]), int(ramp[1])
                if start < 1 or end < start:
                    raise ValueError(
                        f"losses.{name}.epoch_warmup must satisfy "
                        "1 <= start <= end"
                    )
                self.loss_epoch_ramps[name] = (start, end)
            window = cfg.get("epoch_window")
            if window is not None:
                if (
                    not isinstance(window, (list, tuple))
                    or len(window) != 2
                ):
                    raise ValueError(
                        f"losses.{name}.epoch_window must be [start, end]"
                    )
                start, end = int(window[0]), int(window[1])
                if start < 1 or end < start:
                    raise ValueError(
                        f"losses.{name}.epoch_window must satisfy "
                        "1 <= start <= end"
                    )
                self.loss_epoch_windows[name] = (start, end)

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

        # ---- RC-BRD bandwise dispatch modules (DESIGN §9) ----
        # Built last so legacy modules' RNG draws stay identical to an
        # rc_brd-free build (FR-5.1); enabled=false constructs nothing (§9.6).
        self._build_rc_brd(rc_brd_config or {}, rc_brd_expected_fold)

    def train(self, mode: bool = True) -> SLMFBBDM:
        """Set training mode while keeping frozen auxiliary modules deterministic."""
        super().train(mode)
        if self.mean_frozen and self.mean_predictor is not None:
            self.mean_predictor.eval()
        if (
            self.residual_frequency_frozen
            and self.residual_preconditioner is not None
        ):
            self.residual_preconditioner.eval()
        return self

    def set_training_epoch(self, epoch_index: int) -> None:
        """Set the 0-based epoch used by true epoch-wise loss curricula."""

        self._training_epoch_index = max(int(epoch_index), 0)

    def _loss_epoch_scale(self, name: str) -> float:
        epoch = self._training_epoch_index + 1
        window = self.loss_epoch_windows.get(name)
        if window is not None and not window[0] <= epoch <= window[1]:
            return 0.0
        ramp = self.loss_epoch_ramps.get(name)
        if ramp is None:
            return 1.0
        start, end = ramp
        if epoch < start:
            return 0.0
        if end == start or epoch >= end:
            return 1.0
        return float(epoch - start) / float(end - start)

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
            return GaborPrior(
                filters=cfg.get("filters"),
                scales=cfg.get("scales", 4),
                orientations=cfg.get("orientations", 8),
                kernel_size=cfg.get("kernel_size", 15),
                parameter_delta=cfg.get("parameter_delta", 0.25),
                enabled=enabled,
            )
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
            # Gabor energy reaches the hotspot net only when Gabor is enabled
            # AND routed to hotspot.  hotspot_prior.use_gabor_energy can refine
            # this but can never bypass a closed route (use_for_hotspot=False
            # forces CT-only input).
            route_active = (
                self.gabor_routes.get("enabled", False)
                and self.gabor_routes.get("use_for_hotspot", False)
            )
            use_gabor = route_active and bool(cfg.get("use_gabor_energy", True))
            return HotspotPrior(
                base_channels=cfg.get("base_channels", 16),
                params_max=cfg.get("params_max", 500_000),
                use_gabor_energy=use_gabor,
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
            # Gabor energy modulates the high-frequency noise band only when
            # Gabor is enabled AND routed to noise.  A closed route forces the
            # schedule to ignore gabor_energy even if it is present in the
            # ConditionBundle.
            route_active = (
                self.gabor_routes.get("enabled", False)
                and self.gabor_routes.get("use_for_noise", False)
            )
            use_gabor = route_active and bool(cfg.get("use_gabor_energy", True))
            return ScaleAdaptiveNoise(
                num_train_timesteps=cfg.get("num_train_timesteps", 1000),
                low_sigma_mult=cfg.get("low_sigma_mult", 1.0),
                mid_sigma_mult=cfg.get("mid_sigma_mult", 0.75),
                high_sigma_mult=cfg.get("high_sigma_mult", 0.45),
                use_gabor_energy=use_gabor,
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
        elif name == "normalized_lesion_peak":
            from .loss_terms.normalized_lesion_peak import NormalizedLesionPeakLoss
            return NormalizedLesionPeakLoss(
                topk_percent=cfg.get("topk_percent", 0.10),
                min_k=cfg.get("min_k", 3),
                max_k=cfg.get("max_k", 16),
                beta=cfg.get("beta", 0.02),
                active_tau_max=cfg.get("active_tau_max", 0.25),
                cold_weight=cfg.get("cold_weight", 1.0),
                cold_tolerance=cfg.get("cold_tolerance", 0.02),
                enabled=enabled,
                weight=weight,
            )
        elif name == "focal_frequency":
            from .loss_terms.frequency import FocalFrequencyLoss
            return FocalFrequencyLoss(
                alpha=cfg.get("alpha", 1.0),
                active_tau_max=cfg.get("active_tau_max", 0.4),
                enabled=enabled, weight=weight,
            )
        elif name == "residual_wavelet":
            from .loss_terms.residual_frequency import ResidualWaveletLoss
            return ResidualWaveletLoss(
                lesion_weight=cfg.get("lesion_weight", 4.0),
                band_weights=tuple(cfg.get("band_weights", [0.5, 1.0, 1.5])),
                epsilon=cfg.get("epsilon", 1e-3),
                active_tau_max=cfg.get("active_tau_max", 0.7),
                enabled=enabled,
                weight=weight,
            )
        elif name == "gabor_consistency":
            from .loss_terms.residual_frequency import GaborConsistencyLoss
            return GaborConsistencyLoss(
                lesion_weight=cfg.get("lesion_weight", 3.0),
                orientation_weight=cfg.get("orientation_weight", 0.1),
                epsilon=cfg.get("epsilon", 1e-6),
                active_tau_max=cfg.get("active_tau_max", 0.7),
                enabled=enabled,
                weight=weight,
            )
        elif name == "spectral_router_regularization":
            from .loss_terms.spectral_router import (
                SpectralRouterRegularizationLoss,
            )
            return SpectralRouterRegularizationLoss(
                enabled=enabled,
                weight=weight,
                temporal_weight=cfg.get("temporal_weight", 1e-4),
                dct_weight=cfg.get("dct_weight", 1e-4),
                gabor_weight=cfg.get("gabor_weight", 1e-4),
                active_mass_weight=cfg.get("active_mass_weight", 0.0),
                active_mass_floor=cfg.get("active_mass_floor", 0.25),
                prior_anchor_weight=cfg.get("prior_anchor_weight", 0.0),
                monotonic_weight=cfg.get("monotonic_weight", 0.0),
                curvature_weight=cfg.get("curvature_weight", 0.0),
                budget_weight=cfg.get("budget_weight", 0.0),
                shallow_weight=cfg.get(
                    "shallow_cost_weight",
                    cfg.get("shallow_weight", 0.0),
                ),
            )
        elif name == "boundary_frequency":
            from .loss_terms.boundary_frequency import BoundaryFrequencyLoss
            return BoundaryFrequencyLoss(
                lesion_weight=cfg.get("lesion_weight", 1.0),
                anatomy_weight=cfg.get("anatomy_weight", 0.5),
                organ_weight=cfg.get("organ_weight", 0.5),
                wavelet_weight=cfg.get("wavelet_weight", 0.25),
                boundary_radius=cfg.get("boundary_radius", 2),
                epsilon=cfg.get("epsilon", 1e-3),
                active_tau_max=cfg.get("active_tau_max", 0.7),
                enabled=enabled,
                weight=weight,
            )
        elif name == "frequency_gate_tv":
            from .loss_terms.frequency_gate_tv import FrequencyGateTVLoss
            return FrequencyGateTVLoss(enabled=enabled, weight=weight)
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
        elif name == "lesion_roi_l1":
            from .loss_terms.lesion_roi import LesionROIL1Loss
            return LesionROIL1Loss(
                dilate_radius=cfg.get("dilate_radius", 3),
                beta=cfg.get("beta", 0.05),
                active_tau_max=cfg.get("active_tau_max", 0.7),
                enabled=enabled, weight=weight,
            )
        elif name == "nonlesion_body_lowpass":
            from .loss_terms.nonlesion_body_lowpass import (
                NonLesionBodyLowpassLoss,
            )
            return NonLesionBodyLowpassLoss(
                sigma=cfg.get("sigma", 4.0),
                body_threshold=cfg.get("body_threshold", 0.03),
                body_closing_radius=cfg.get("body_closing_radius", 2),
                lesion_exclusion_radius=cfg.get(
                    "lesion_exclusion_radius", 8
                ),
                tail_quantile=cfg.get("tail_quantile", 0.75),
                tail_weight=cfg.get("tail_weight", 2.0),
                tail_temperature=cfg.get("tail_temperature", 0.02),
                charbonnier_eps=cfg.get("charbonnier_eps", 1.0e-3),
                active_tau_max=cfg.get("active_tau_max", 0.70),
                enabled=enabled,
                weight=weight,
            )
        elif name == "route_utility_supervision":
            from .loss_terms.route_utility_supervision import (
                RouteUtilitySupervisionLoss,
            )
            return RouteUtilitySupervisionLoss(
                lesion_dilate_radius=cfg.get(
                    "lesion_dilate_radius",
                    3,
                ),
                spatial_weight=cfg.get("spatial_weight", 1.0),
                global_weight=cfg.get("global_weight", 0.25),
                spatial_tv_weight=cfg.get(
                    "spatial_tv_weight",
                    1.0e-3,
                ),
                positive_weight=cfg.get("positive_weight", 4.0),
                background_target=cfg.get("background_target", 0.02),
                lesion_target=cfg.get("lesion_target", 0.90),
                global_target_min=cfg.get("global_target_min", 0.05),
                global_target_max=cfg.get("global_target_max", 0.55),
                active_tau_max=cfg.get("active_tau_max", 0.70),
                enabled=enabled,
                weight=weight,
            )
        elif name == "outside_peak_ranking":
            from .loss_terms.lesion_roi import OutsidePeakRankingLoss
            return OutsidePeakRankingLoss(
                margin=cfg.get("margin", 0.05),
                inside_radius=cfg.get("inside_radius", 3),
                outside_radius=cfg.get("outside_radius", 8),
                topk_percent=cfg.get("topk_percent", 0.01),
                active_tau_max=cfg.get("active_tau_max", 0.65),
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
                target_mode=cfg.get("target_mode", "mask_only"),
                local_uptake_radius=cfg.get("local_uptake_radius", 8),
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
        elif name == "perceptual_x0":
            from .loss_terms.perceptual_x0 import LesionAwarePerceptualX0Loss
            return LesionAwarePerceptualX0Loss(
                enabled=enabled,
                weight=weight,
                checkpoint=cfg.get("checkpoint"),
                checkpoint_dir=cfg.get("checkpoint_dir", "checkpoints"),
                encoder_kind=cfg.get("encoder_kind", "pretrained"),
                in_channels=cfg.get("in_channels", 1),
                base_channels=cfg.get("base_channels", 16),
                feature_layers=cfg.get(
                    "feature_layers", ["full", "half", "quarter"]
                ),
                layer_weights=cfg.get(
                    "layer_weights", [1.0, 0.5, 0.25]
                ),
                distance=cfg.get("distance", "charbonnier"),
                charbonnier_eps=cfg.get("charbonnier_eps", 1e-3),
                region_mode=cfg.get("region_mode", "global"),
                lesion_weight=cfg.get("lesion_weight", 4.0),
                background_weight=cfg.get("background_weight", 1.0),
                dilate_radius=cfg.get("dilate_radius", 3),
                timestep_weighting=cfg.get("timestep_weighting", "uniform"),
                require_checkpoint_lineage=cfg.get(
                    "require_checkpoint_lineage", False
                ),
            )
        else:
            from .loss_terms.base import DisabledLossTerm
            return DisabledLossTerm(name=name, weight=weight)

    # ------------------------------------------------------------------
    # RC-BRD construction + bandwise dispatch (DESIGN §8/§9; PRD FR-5).
    # §12.4: scheduling calls only — bandwise math lives in src/model/rc_brd.
    # ------------------------------------------------------------------

    def _build_rc_brd(self, rc_cfg: Dict[str, Any], expected_fold: Optional[str]) -> None:
        """Parse 'modules.rc_brd'; build the bandwise schedule/specialist.
        Fail-closed (DESIGN §8; PRD §3/§4): bad enums, missing/inconsistent
        contracts, SHA/fold mismatches, κ≠0 without contract → ValueError.
        """
        self.rc_brd_enabled = bool(rc_cfg.get("enabled", False))
        self.rc_brd_schedule: Optional[nn.Module] = None
        self.rc_brd_head: Optional[nn.Module] = None
        self.rc_brd_ct_proj: Optional[nn.Module] = None
        self.rc_brd_contract: Optional[Any] = None
        self.rc_brd_readout_mode, self.rc_brd_mc_samples = "mc_mean", 8  # C6
        self.rc_brd_sampling_mode = "endpoint_ddim"  # v1.0g (DESIGN §8)
        self.rc_brd_density_match = False     # A3b logSNR density matching
        self.rc_brd_loss_weighting = "uniform"  # A8 per-band Min-SNR-γ
        self.rc_brd_min_snr_gamma = 5.0
        self._rc_brd_density_contract: Optional[Any] = None  # pre-transform
        self._rc_brd_density_draws = 0
        if not self.rc_brd_enabled:
            return
        if not self.residual_bridge_enabled:
            raise ValueError("modules.rc_brd requires modules.residual_bridge.enabled=true")
        if self.residual_frequency_enabled:
            # v1.0f guard ③: the legacy learned router must never co-run.
            raise ValueError("modules.rc_brd is mutually exclusive with "
                             "modules.residual_frequency.enabled=true")
        if self.image_size % 4:
            raise ValueError("modules.rc_brd requires image_size divisible by 4 (two-level Haar)")
        num_t = int(rc_cfg.get("num_timesteps", self.noise_schedule.num_train_timesteps))
        if num_t != int(self.noise_schedule.num_train_timesteps):
            raise ValueError(f"modules.rc_brd.num_timesteps ({num_t}) must equal the bbdm_bridge "
                             f"num_train_timesteps ({self.noise_schedule.num_train_timesteps})")
        kappa = float(rc_cfg.get("kappa", 0.0))
        contract = None
        contract_path = str(rc_cfg.get("contract_path", "") or "")
        if contract_path:
            if not Path(contract_path).is_file():
                raise ValueError(f"modules.rc_brd.contract_path not found: {contract_path}")
            if self._rc_brd_mean_state is None:
                raise ValueError("modules.rc_brd contract validation requires a loaded "
                                 "conditional-mean checkpoint (mean SHA anchor, §9.4)")
            from .rc_brd import ContractViolationError, RecoverabilityContract, mean_weights_sha256
            fold = str(rc_cfg.get("contract_fold", "") or "") or expected_fold
            try:
                contract = RecoverabilityContract.load(
                    contract_path, expected_fold=fold or None,
                    expected_mean_sha=mean_weights_sha256(self._rc_brd_mean_state))
            except ContractViolationError as exc:
                raise ValueError(f"rc_brd contract failed fail-closed validation: {exc}") from exc
            declared = rc_cfg.get("contract", {}) or {}
            if "support_mode" in declared and str(declared["support_mode"]) != contract.support_mode:
                raise ValueError("modules.rc_brd.contract.support_mode disagrees with the contract artifact")
            for key in ("eta_max", "floor_rho"):
                if key in declared and float(declared[key]) != float(getattr(contract, key)):
                    raise ValueError(f"modules.rc_brd.contract.{key} disagrees with the contract artifact")
        elif kappa != 0.0:
            # κ=0 may run contract-free (strict D1, DESIGN §4); a time-changed
            # clock requires the frozen contract.
            raise ValueError("modules.rc_brd.kappa != 0 requires contract_path")
        from .rc_brd import (A3_VARIANTS, CLOCK_MODES, CONTRACT_TRANSFORMS,
                             BandwiseBridgeSchedule, BandwiseScheduleConfig,
                             apply_contract_transform, band_groups)
        forward_mode = str(rc_cfg.get("forward_mode", "bridge_time_changed"))
        endpoint_mode = str(rc_cfg.get("endpoint_mode", "zeros"))
        # v2 band clock (DESIGN_RC_BRD_clock_v2): base_snr = v1 power-blind
        # control B; band_snr = self-consistent band-SNR axis (needs a v2
        # contract with band_powers - the schedule guard raises otherwise).
        clock_mode = str(rc_cfg.get("clock_mode", "base_snr"))
        if clock_mode not in CLOCK_MODES:
            raise ValueError(f"modules.rc_brd.clock_mode must be one of "
                             f"{CLOCK_MODES}, got {clock_mode!r}")
        if forward_mode == "vp_bandwise" and endpoint_mode == "ct_minus_mean":
            # v1.0f guard ①: the sampling path cannot carry ct_minus_mean in vp.
            raise ValueError("modules.rc_brd forward_mode=vp_bandwise cannot be combined "
                             "with endpoint_mode=ct_minus_mean (v1.0f guard)")
        if contract is not None and kappa not in tuple(float(k) for k in contract.kappa_grid):
            # v1.0f guard ②: κ must be pre-registered (κ=0 w/o contract skips).
            raise ValueError(f"modules.rc_brd.kappa={kappa} is not in the contract "
                             f"kappa_grid {tuple(contract.kappa_grid)}")
        transform = str(rc_cfg.get("contract_transform", "none") or "none")
        a3_variant = str(rc_cfg.get("a3_variant", "") or "")
        density_match = str(rc_cfg.get("density_match", "") or "")
        self.rc_brd_loss_weighting = str(rc_cfg.get("loss_weighting", "uniform") or "uniform")
        self.rc_brd_min_snr_gamma = float(rc_cfg.get("min_snr_gamma", 5.0))
        if transform not in CONTRACT_TRANSFORMS:
            raise ValueError(f"modules.rc_brd.contract_transform must be one of "
                             f"{CONTRACT_TRANSFORMS}, got {transform!r}")
        if a3_variant and a3_variant not in A3_VARIANTS:
            raise ValueError(f"modules.rc_brd.a3_variant must be one of {A3_VARIANTS} "
                             "or empty, got {a3_variant!r}")
        if (a3_variant, density_match) not in {("", ""), ("budget_matched", ""),
                                               ("density_matched", "logsnr")}:
            # C8 pairing (DESIGN §8 v1.0f): budget_matched→density 空，
            # density_matched→density_match="logsnr"（反向亦禁）。
            raise ValueError(f"modules.rc_brd a3_variant={a3_variant!r}/density_match="
                             f"{density_match!r} is not a legal C8 pairing")
        if self.rc_brd_loss_weighting not in ("uniform", "per_band_min_snr"):
            raise ValueError("modules.rc_brd.loss_weighting must be 'uniform' or "
                             f"'per_band_min_snr', got {self.rc_brd_loss_weighting!r}")
        if not self.rc_brd_min_snr_gamma > 0.0:
            raise ValueError("modules.rc_brd.min_snr_gamma must be > 0")
        self.rc_brd_density_match = density_match == "logsnr"
        if (transform != "none" or self.rc_brd_density_match) and contract is None:
            raise ValueError("modules.rc_brd.contract_transform/density_match require "
                             "contract_path (no contract was loaded)")
        self._rc_brd_density_contract = contract  # original, pre-transform (A3b 密度)
        if transform != "none":
            # A3/A4/A5 contract-layer transform, applied after fail-closed load.
            contract = apply_contract_transform(contract, transform)
        groups = band_groups(int(rc_cfg.get("band_groups", 3)))  # ValueError unless 3|7
        if contract is not None and {g: list(b) for g, b in contract.band_groups.items()} != groups:
            raise ValueError("modules.rc_brd.band_groups disagrees with the contract band_groups")
        self.rc_brd_schedule = BandwiseBridgeSchedule(
            BandwiseScheduleConfig(
                num_timesteps=num_t, kappa=kappa,
                forward_mode=forward_mode,
                sigma_bridge=float(rc_cfg.get("sigma_bridge", 1.0)),
                lambda_min=float(rc_cfg.get("lambda_min", -10.0)),
                lambda_max=float(rc_cfg.get("lambda_max", 10.0)),
                endpoint_mode=endpoint_mode,
                clock_mode=clock_mode,
            ),
            contract,
        )
        self.rc_brd_contract = contract
        specialist_cfg = rc_cfg.get("specialist", {}) or {}
        allow_unverified = bool(rc_cfg.get("allow_unverified_tokens", False))  # v1.0g
        if specialist_cfg.get("enabled", True):
            from .rc_brd import BoundedSpecialistHead, SpecialistConfig
            channels = int(specialist_cfg.get("channels", 64))
            self.rc_brd_head = BoundedSpecialistHead(SpecialistConfig(
                a_max=float(specialist_cfg.get("a_max", 0.25)),
                d_max=float(specialist_cfg.get("d_max", 0.10)),  # v1.0g tanh 幅度界
                gate_init=float(specialist_cfg.get("gate_init", 0.1)),
                channels=channels),
                allow_unverified_tokens=allow_unverified)
            # CT condition stream for the head ([B,64,H/2,W/2], DESIGN §9.2):
            # 2× average-pooled CT → trainable 1×1 conv (specialist-owned).
            self.rc_brd_ct_proj = nn.Conv2d(1, channels, kernel_size=1)
        readout_cfg = rc_cfg.get("readout", {}) or {}
        self.rc_brd_readout_mode = str(readout_cfg.get("mode", "mc_mean"))
        if self.rc_brd_readout_mode not in ("mc_mean", "deterministic_head"):
            raise ValueError("modules.rc_brd.readout.mode must be 'mc_mean' or 'deterministic_head' (C6)")
        # deterministic_head (C6): v1's head has no stochastic component; the
        # switch is validated now, reserved for [审计] §4.2 δ_det+δ_sto.
        self.rc_brd_mc_samples = int(readout_cfg.get("mc_samples", 8))
        if self.rc_brd_mc_samples < 1:
            raise ValueError("modules.rc_brd.readout.mc_samples must be >= 1")
        # v1.0g sampling readout (DESIGN §8): endpoint_ddim (default; random
        # initial state + deterministic steps) | ancestral_mc (per-step
        # posterior sampling; the MC-mean paper claim requires this mode).
        self.rc_brd_sampling_mode = str(rc_cfg.get("sampling_mode", "endpoint_ddim"))
        if self.rc_brd_sampling_mode not in ("endpoint_ddim", "ancestral_mc"):
            raise ValueError(
                "modules.rc_brd.sampling_mode must be 'endpoint_ddim' or "
                f"'ancestral_mc' (v1.0g), got {self.rc_brd_sampling_mode!r}")

    def _rc_brd_endpoint_bands(
        self, ct: torch.Tensor, bridge_mean: torch.Tensor,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Endpoint e_r = W(CT−μ) in band coordinates (C7; [计划] §3.6).

        endpoint_mode='zeros' → None (e_r=0); image-coordinate μ/CT are never
        handed to the schedule (rc_brd.schedule rejects them, FR-3.4).
        """
        if self.rc_brd_schedule.config.endpoint_mode != "ct_minus_mean":
            return None
        from .rc_brd import haar_forward2
        return haar_forward2(ct - bridge_mean)

    def _rc_brd_c_effective(
        self, timesteps: torch.Tensor, reference: torch.Tensor,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Band-level c_run at λ₀(t) (DESIGN §4 v1.0f; [审计] §3.1/§3.3/§5).

        Active groups interpolate/clip/η-shrink the contract at the v1.0f
        λ₀(u)=clip(log((1−u)/(ν²u)),λmin,λmax) lookup; floor_gated then applies
        the runtime floor c_run=floor_rho+(1−floor_rho)·c̃ ([审计] §5;
        stratified_mixture is the identity).  Inactive groups keep the
        explicit 0 (head switch-off semantics, batch 1B); no contract → None
        (calibration form c=1).  Group c is band-expanded via
        contract.expand_group_to_bands (DESIGN §3).
        """
        contract = self.rc_brd_contract
        if contract is None or self.rc_brd_head is None:
            return None
        zero = torch.zeros(timesteps.shape[0], dtype=torch.float64,
                           device=reference.device)
        active = set(contract.b_active)
        c_groups = {}
        for group in contract.band_groups:
            if group not in active:
                c_groups[group] = zero
                continue
            # v2 (audit defect (b) fix): query on the clock's own axis.
            # base_snr keeps the shared lambda0(t/T) for every group (v1
            # semantics); band_snr uses lambda_g(m_g(t)) with the frozen band
            # power.  Single source: schedule.clock_query_log_snr - the head
            # gating cannot drift from the clock.
            log_snr = self.rc_brd_schedule.clock_query_log_snr(
                group, timesteps).to(device=reference.device, dtype=torch.float64)
            c_groups[group] = contract.effective_c(group, log_snr)
        if contract.support_mode == "floor_gated":  # v1.0f guard ④ ([审计] §5)
            floor = float(contract.floor_rho)
            c_groups = {g: (floor + (1.0 - floor) * c if g in active else c)
                        for g, c in c_groups.items()}  # inactive keeps explicit 0
        from .rc_brd import expand_group_to_bands
        return expand_group_to_bands(c_groups, contract.band_groups)

    def _rc_brd_apply_specialist(
        self, pred_z0: torch.Tensor, timesteps: torch.Tensor, ct: torch.Tensor,
    ) -> torch.Tensor:
        """ẑ0_base (image domain) → bands → bounded specialist → image domain.

        [计划] §3.7 / DESIGN §9.2: the head is exactly the identity at init
        (zero-initialised Δ; v1.0g tanh(0)=0), so κ=0 stays numerically
        identical to D1. v1.0g provenance (AUDIT 5 GT-leak X): the ct_proj
        output is wrapped in a CTFeatureToken signed by the head session
        before entering the head — bare tensors never cross this boundary.
        """
        import torch.nn.functional as F
        from .rc_brd import haar_forward2, haar_inverse2
        ct_feat = self.rc_brd_ct_proj(F.avg_pool2d(ct, kernel_size=2))
        ct_token = self.rc_brd_head.issue_ct_token(ct_feat)  # v1.0g provenance
        out_bands = self.rc_brd_head(haar_forward2(pred_z0), ct_token,
                                     self._rc_brd_c_effective(timesteps, pred_z0))
        return haar_inverse2(out_bands)

    def _rc_brd_density_weights(self) -> torch.Tensor:
        """A3b mean-corruption-rate timestep weights ρ̄ on the u=t/T grid.

        ρ̄(u)=Σ_g (n_g/N)·exp{κ(2c̃_g(λ₀(u))−1)} (DESIGN §8 v1.0g), averaged
        over Haar *coefficients* (n_g/N from _rc_brd_group_shares — LL2:mid:
        high = 1:3:12 /16; AUDIT 5 §3.4: the total corruption budget must not
        be a plain 3-group mean), c̃ from the *original* (pre-transform)
        contract — A3b keeps the contracted arm's sampling density while
        running flat c — and λ₀ per DESIGN §4 v1.0f. Normalised to sum 1 over
        t=0..T−1 (torch.multinomial ready).
        """
        cfg = self.rc_brd_schedule.config
        contract = self._rc_brd_density_contract
        t_grid = torch.arange(cfg.num_timesteps, dtype=torch.int64)
        shares = _rc_brd_group_shares(contract.band_groups)  # n_g/N (v1.0g)
        rho = torch.zeros(cfg.num_timesteps, dtype=torch.float64)
        for group, share in shares.items():
            # v2: same single-source query axis as the clock (base_snr keeps
            # the shared lambda0; band_snr uses lambda_g(m_g(t))).  band_powers
            # are invariant under the A3/A4/A5 contract transforms.
            lam = self.rc_brd_schedule.clock_query_log_snr(
                group, t_grid).to(torch.float64)
            rho = rho + share * torch.exp(
                cfg.kappa * (2.0 * contract.effective_c(group, lam).to(torch.float64) - 1.0))
        return rho / rho.sum()

    def _rc_brd_density_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Draw timesteps ~ ρ̄ via multinomial with an explicit generator.

        The generator is seeded from the model's initialization_seed (fallback
        0) plus a draw counter, so the sequence is reproducible run-to-run yet
        advances call-to-call (DESIGN §8 v1.0f "seed 显式").
        """
        seed = int(self.initialization_seed or 0) + 7919 * self._rc_brd_density_draws
        self._rc_brd_density_draws += 1
        generator = torch.Generator().manual_seed(seed)
        draw = torch.multinomial(self._rc_brd_density_weights(), batch_size,
                                 replacement=True, generator=generator)
        return draw.to(device=device, dtype=torch.long)

    def _rc_brd_base_loss(self, pred: torch.Tensor, target: torch.Tensor,
                          timesteps: torch.Tensor,
                          tau: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """A8 per-band Min-SNR-γ base loss (DESIGN §8 v1.0f; [地图] §10.1 A8).

        Replaces the scalar min-SNR weight of _base_reconstruction_loss with
        the bandwise clamp min{SNR_b(t), γ}, SNR_b=ᾱ_b/(1−ᾱ_b) from
        schedule.alpha_hat (bridge: ᾱ=1−m, the documented diagnostic proxy).
        The MSE term is computed per Haar band and size-weighted (orthonormal
        Parseval: Σ_b (n_b/N)·mse_b == image MSE; uniform weights therefore
        reproduce the unweighted MSE exactly).  L1/gradient terms and the
        τ-stage weights follow _base_reconstruction_loss unchanged.
        """
        from .rc_brd import BAND_NAMES, haar_forward2
        pred_bands, target_bands = haar_forward2(pred), haar_forward2(target)
        contract = self.rc_brd_contract
        if contract is not None:
            band_to_group = {b: g for g, bands in contract.band_groups.items() for b in bands}
        else:
            band_to_group = {b: b for b in BAND_NAMES}
        gamma = torch.tensor(self.rc_brd_min_snr_gamma, dtype=torch.float64)
        n_total = float(pred.shape[-1] * pred.shape[-2])
        reduce_dims = tuple(range(1, pred.dim()))
        weighted = torch.zeros(pred.shape[0], dtype=torch.float64, device=pred.device)
        weight_norm = torch.zeros(pred.shape[0], dtype=torch.float64, device=pred.device)
        mse_sum = torch.zeros(pred.shape[0], dtype=torch.float64, device=pred.device)
        band_logs: Dict[str, torch.Tensor] = {}
        for band in BAND_NAMES:
            alpha = self.rc_brd_schedule.alpha_hat(
                band_to_group[band], timesteps).to(dtype=torch.float64)
            snr = alpha / (1.0 - alpha).clamp_min(1e-8)  # ᾱ/(1−ᾱ)
            w_b = torch.minimum(snr, gamma)              # min{SNR_b(t), γ}
            mse_b = (pred_bands[band].double() - target_bands[band].double()).square()
            mse_b = mse_b.mean(dim=reduce_dims)
            share = float(pred_bands[band][0].numel()) / n_total  # n_b/N (Parseval)
            weighted = weighted + share * w_b * mse_b
            weight_norm = weight_norm + share * w_b
            mse_sum = mse_sum + share * mse_b
            band_logs[f"loss/rc_brd_band_weight_{band}"] = w_b.mean().to(pred.dtype).detach()
        mse = (weighted / weight_norm.clamp_min(1e-8)).to(pred.dtype)  # mean-weight 1
        l1 = (pred - target).abs().mean(dim=reduce_dims)
        grad = _image_gradient_l1_per_sample(pred, target)
        tau_f = tau.float()
        w_mse = 0.5 + 1.5 * torch.sigmoid(10.0 * (tau_f - 0.43))   # τ-stage weights
        w_l1 = torch.ones_like(tau_f)
        w_grad = 0.2 * torch.sigmoid(10.0 * (0.25 - tau_f))
        per_sample = (self.base_mse_weight * w_mse * mse
                      + self.base_l1_weight * w_l1 * l1
                      + self.base_gradient_weight * w_grad * grad)
        return per_sample.mean(), {
            "loss/base_mse": mse.mean().detach(),
            "loss/base_l1": l1.mean().detach(),
            "loss/base_gradient": grad.mean().detach(),
            "loss/min_snr_weight": (weight_norm / 1.0).mean().detach().to(pred.dtype),
            "loss/tau_mse_weight": w_mse.mean().detach(),
            "loss/tau_grad_weight": w_grad.mean().detach(),
            **band_logs,
        }

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

    def _compose_residual_output(
        self,
        mean_pet: torch.Tensor,
        pred_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the pre-calibrated safe residual correction to a frozen mean.

        ``output_scale=0`` is an exact identity fallback to the deterministic
        mean model (provided output clipping is disabled).  The diffusion loss
        still supervises the unscaled residual; image-space losses and reported
        predictions see the same scaled correction used at inference.
        """

        correction = self.residual_output_scale * pred_residual
        synthetic_pet = mean_pet + correction
        if self.residual_output_clip:
            synthetic_pet = synthetic_pet.clamp(-1.0, 1.0)
        return synthetic_pet, correction

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

        # pred_x0 objective: weight = min(SNR, γ), not epsilon-prediction weight
        weights = torch.minimum(snr, gamma)

        # Normalize by the full-schedule reference mean so the average base loss
        # magnitude stays approximately unchanged.  Compute once and cache.
        if self._min_snr_reference_mean is None or self._min_snr_reference_mean.device != ref.device:
            full_indices = torch.arange(
                self.noise_schedule.num_train_timesteps,
                device=ref.device,
            )
            if hasattr(self.noise_schedule, "alphas_cumprod"):
                full_alpha = self.noise_schedule.alphas_cumprod[full_indices].to(
                    device=ref.device, dtype=ref.dtype
                )
                full_snr = full_alpha / (1.0 - full_alpha).clamp_min(1e-8)
            else:
                full_m = self.noise_schedule.m_t[full_indices].to(
                    device=ref.device, dtype=ref.dtype
                )
                full_sigma = self.noise_schedule.sigma_t[full_indices].to(
                    device=ref.device, dtype=ref.dtype
                )
                full_snr = (1.0 - full_m).square() / full_sigma.square().clamp_min(1e-8)
            full_weights = torch.minimum(full_snr, gamma)
            self._min_snr_reference_mean = full_weights.mean().detach()

        weights = weights / self._min_snr_reference_mean.clamp_min(1e-8)
        return weights

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

    def _build_frequency_injections(
        self,
        condition: ConditionBundle,
        timesteps: Optional[torch.Tensor],
        noisy_residual: Optional[torch.Tensor],
        router_confidence: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        self._last_frequency_diagnostics = {}
        if not self.residual_frequency_enabled:
            return []
        if timesteps is None or noisy_residual is None or self.residual_preconditioner is None:
            raise ValueError("Residual-frequency injection requires noisy_residual and timesteps")
        if self.residual_frequency_mode == "spectral_evidence_router":
            injections, diagnostics = self.residual_preconditioner(
                noisy_residual,
                timesteps,
                self.noise_schedule,
                condition.maps["ct"],
                gabor_feat=condition.maps.get("gabor_feat"),
                gabor_orientation=condition.maps.get("gabor_orientation"),
                gabor_anisotropy=condition.maps.get("gabor_anisotropy"),
                router_confidence=router_confidence,
            )
            self._last_frequency_diagnostics = diagnostics
            for level in (2, 1):
                spatial_key = (
                    f"route_spatial_conditional_shallow_l{level}"
                )
                spatial_route = diagnostics.get(spatial_key)
                if spatial_route is not None:
                    condition.maps[
                        f"spectral_route_spatial_shallow_l{level}"
                    ] = spatial_route
            condition.scalars["frequency_gate_tv"] = diagnostics["gate_tv"]
            condition.scalars["spectral_route_temporal_smoothness"] = diagnostics[
                "route_temporal_smoothness"
            ]
            condition.scalars["spectral_route_active_mass"] = 0.5 * (
                diagnostics["route_l2_active_mass"]
                + diagnostics["route_l1_active_mass"]
            )
            effective_policy = getattr(
                self.residual_preconditioner, "_effective_policy", ""
            )
            condition.scalars["spectral_route_is_learned"] = (
                noisy_residual.new_ones(())
                if effective_policy
                in {
                    "learned",
                    "learned_no_null",
                    "prior_anchored_learned",
                }
                else noisy_residual.new_zeros(())
            )
            is_prior_anchored = (
                effective_policy == "prior_anchored_learned"
            )
            condition.scalars["spectral_route_is_prior_anchored"] = (
                noisy_residual.new_tensor(float(is_prior_anchored))
            )
            if is_prior_anchored:
                scalar_mapping = {
                    "spectral_route_prior_active": "route_prior_active",
                    "spectral_route_active": "route_active",
                    "spectral_route_active_delta": "route_active_delta",
                    "spectral_route_active_next": "route_active_next",
                    "spectral_route_delta_prev": "route_delta_prev",
                    "spectral_route_delta_next": "route_delta_next",
                    "spectral_route_shallow_probability": (
                        "route_shallow_probability"
                    ),
                    "spectral_route_conditional_shallow": (
                        "route_conditional_shallow"
                    ),
                    "spectral_route_active_phase": "route_active_progress",
                    "spectral_route_destination_phase": (
                        "route_destination_progress"
                    ),
                    "spectral_route_anchor_scale": "route_anchor_scale",
                    "spectral_route_has_next": "route_has_next",
                    "spectral_route_has_prev": "route_has_prev",
                }
                for scalar_key, diagnostic_key in scalar_mapping.items():
                    condition.scalars[scalar_key] = diagnostics[diagnostic_key]
            condition.scalars["spectral_dct_weight_offset"] = diagnostics[
                "dct_weight_offset"
            ]
            gabor_prior = self.priors["gabor"] if "gabor" in self.priors else None
            condition.scalars["spectral_gabor_parameter_offset"] = (
                gabor_prior.parameter_offset_energy()
                if gabor_prior is not None
                and hasattr(gabor_prior, "parameter_offset_energy")
                else noisy_residual.new_zeros(())
            )
        elif self.residual_frequency_mode == "boundary_reliable":
            frequency_kwargs = {
                "gabor_orientation": condition.maps.get("gabor_orientation"),
            }
            if self.residual_preconditioner.use_gabor_agreement:
                frequency_kwargs["gabor_anisotropy"] = condition.maps.get(
                    "gabor_anisotropy"
                )
            injections, diagnostics = self.residual_preconditioner(
                noisy_residual,
                timesteps,
                self.noise_schedule,
                condition.maps["ct"],
                **frequency_kwargs,
            )
            self._last_frequency_diagnostics = diagnostics
            condition.scalars["frequency_gate_tv"] = diagnostics["gate_tv"]
            condition.scalars["frequency_noise_reliability"] = diagnostics[
                "noise_reliability"
            ].mean(dim=1)
        else:
            injections, diagnostics = self.residual_preconditioner(
                noisy_residual,
                timesteps,
                self.noise_schedule,
                condition.maps.get("gabor_orientation"),
            )
            self._last_frequency_diagnostics = diagnostics
        return injections

    def _build_adapter_injections(
        self,
        condition: ConditionBundle,
        timesteps: Optional[torch.Tensor] = None,
        hw_list: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """Build only the existing CT/organ/Gabor/hotspot adapter branch."""
        if not self.zero_adapter_enabled:
            return []

        ct_feats = [condition.maps[f"ct_feat_{i}"] for i in range(4)]
        organ_feats = [
            condition.maps.get("organ_feat_1"),
            condition.maps.get("organ_feat_2"),
            condition.maps.get("organ_feat_3"),
            None,
        ]
        if self.gabor_routes.get("enabled", False) and self.gabor_routes.get("inject_adapter", False):
            gabor_feat = condition.maps.get("gabor_feat")
        else:
            gabor_feat = None
        hotspot = condition.maps.get("hotspot_prior")
        if hw_list is None:
            hw_list = [
                ct_feats[0].shape[2],
                ct_feats[0].shape[2] // 2,
                ct_feats[0].shape[2] // 4,
                ct_feats[0].shape[2] // 8,
            ]
        tau = self.noise_schedule.get_tau(timesteps) if timesteps is not None else None
        injections = self.adapter.get_zero_conv_outputs(
            ct_feats,
            organ_feats,
            gabor_feat,
            hotspot,
            hw_list,
            tau=tau,
        )
        return list(reversed(injections)) if injections else []

    def _build_skip_injections(
        self,
        condition: ConditionBundle,
        timesteps: Optional[torch.Tensor] = None,
        hw_list: Optional[List[int]] = None,
        noisy_residual: Optional[torch.Tensor] = None,
        router_confidence: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Build Zero-Conv adapter outputs for UNet skip connections.

        When adapter is disabled, returns empty injections (no-op).
        Applies time-varying beta modulation from the adapter.
        """
        adapter_injections = self._build_adapter_injections(
            condition,
            timesteps=timesteps,
            hw_list=hw_list,
        )
        frequency_injections = self._build_frequency_injections(
            condition, timesteps, noisy_residual, router_confidence=router_confidence
        )
        if adapter_injections and frequency_injections:
            if len(adapter_injections) != len(frequency_injections):
                raise RuntimeError("Adapter and residual-frequency skip levels do not match")
            return [a + f for a, f in zip(adapter_injections, frequency_injections)]
        return adapter_injections or frequency_injections

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
            if self.rc_brd_enabled and self.rc_brd_density_match:
                # A3b (DESIGN §8 v1.0f): timestep density matched to ρ̄.
                timesteps = self._rc_brd_density_timesteps(B, device)
            else:
                T = self.noise_schedule.num_train_timesteps
                timesteps = torch.randint(0, T, (B,), device=device)

        mean_output: Optional[Dict[str, torch.Tensor]] = None
        mean_pet: Optional[torch.Tensor] = None
        model_target = x0
        bridge_source = x_source
        if self.residual_bridge_enabled:
            if self.mean_predictor is None:
                raise RuntimeError("Residual bridge was enabled without a mean predictor")
            mean_output = self.mean_predictor(x_source)
            mean_pet = mean_output["mean_pet"]
            bridge_mean = mean_pet.detach() if self.mean_detach_bridge else mean_pet
            model_target = x0 - bridge_mean
            bridge_source = torch.zeros_like(x_source)

        # 1. Build condition bundle
        condition = self.build_condition_bundle(batch, timesteps)
        # 2. Forward diffusion (add noise)
        # BBDM bridge needs x_source; DDPM ignores it
        if self.rc_brd_enabled:
            # DESIGN §9.2: r0 → W(r0) → q_marginal (ε explicit) → W⁻¹; the UNet
            # input form and the r0 (pred_x0) target stay unchanged.
            from .rc_brd import haar_forward2, haar_inverse2
            z_t_bands, _ = self.rc_brd_schedule.q_marginal(
                haar_forward2(model_target),
                timesteps,
                eps_bands=haar_forward2(noise),
                endpoint_bands=self._rc_brd_endpoint_bands(x_source, bridge_mean),
            )
            noisy_x = haar_inverse2(z_t_bands)
        else:
            noisy_x = _add_noise(
                self.noise_schedule,
                model_target,
                noise,
                timesteps,
                condition,
                bridge_source,
            )

        # 3. Condition dropout (training only) — MUST happen BEFORE adapter reads conditions
        condition = self.condition_dropout.apply(condition, training=self.training)
        denoiser_state = noisy_x
        if (
            self.residual_frequency_enabled
            and self.residual_preconditioner is not None
            and self.residual_frequency_mode == "legacy"
            and not self.residual_preconditioner.inject_wavelet
            and self.residual_preconditioner.state_modulation
        ):
            denoiser_state = self.residual_preconditioner.modulate_residual(
                noisy_x, condition.maps.get("gabor_orientation")
            )

        # 4. Build skip injections from adapter (Zero-Conv or NoOp)
        #    Dropout already applied → adapter sees zeroed-out conditions for dropped modules
        skip_injections = self._build_skip_injections(
            condition,
            timesteps=timesteps,
            noisy_residual=noisy_x if self.residual_bridge_enabled else None,
            router_confidence=batch.get("router_confidence"),
        )

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
            self_cond = torch.zeros_like(model_target)
            use_self_cond = self.training and torch.rand((), device=device) < self.self_conditioning_prob
            if use_self_cond:
                with torch.no_grad():
                    sc_input = self._model_input(denoiser_state, x_source, self_cond)
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
        model_input = self._model_input(denoiser_state, x_source, self_cond)
        output = self.unet(model_input, timesteps, context_tokens=semantic_tokens,
                          skip_injections=skip_injections if skip_injections else None,
                          meta=meta_tensor,
                          ca_beta=ca_beta)

        # 5. Split output
        if self.enable_heteroscedastic:
            pred_model, pred_logvar = output[:, :1], output[:, 1:]
            pred_logvar = pred_logvar.clamp(self.heteroscedastic_logvar_min, self.heteroscedastic_logvar_max)
        else:
            pred_model, pred_logvar = output[:, :1], None

        if self.residual_bridge_enabled:
            if mean_pet is None:
                raise RuntimeError("Residual bridge did not produce a conditional mean")
            if self.rc_brd_head is not None:
                # DESIGN §9.2: bounded specialist increment on the base ẑ0 bands.
                pred_model = self._rc_brd_apply_specialist(pred_model, timesteps, x_source)
            reconstruction_mean = mean_pet.detach() if self.mean_detach_bridge else mean_pet
            pred_x0, _ = self._compose_residual_output(reconstruction_mean, pred_model)
        else:
            pred_x0 = pred_model

        if self.gabor_routes.get("enabled", False) and self.gabor_routes.get("use_for_loss", False):
            gabor_prior = self.priors["gabor"]
            if not hasattr(gabor_prior, "describe"):
                raise RuntimeError("Gabor loss route requires a descriptor-capable Gabor prior")
            pred_descriptor = gabor_prior.describe(pred_x0, detach_parameters=True)
            with torch.no_grad():
                target_descriptor = gabor_prior.describe(x0, detach_parameters=True)
            condition.maps.update({
                "gabor_pred_feat": pred_descriptor["gabor_feat"],
                "gabor_target_feat": target_descriptor["gabor_feat"],
                "gabor_pred_orientation": pred_descriptor["gabor_orientation"],
                "gabor_target_orientation": target_descriptor["gabor_orientation"],
            })

        # 6. Compute losses
        tau = self.noise_schedule.get_tau(timesteps)
        ctx = LossContext(
            model_pred=pred_model,
            loss_target=model_target,
            target_pet=x0,
            pred_x0=pred_x0,
            timesteps=timesteps,
            tau=tau,
            batch=batch,
            condition=condition,
            pred_logvar=pred_logvar,
            pred_residual=pred_model if self.residual_bridge_enabled else None,
            target_residual=model_target if self.residual_bridge_enabled else None,
            mean_pet=mean_pet,
        )

        total_loss = torch.tensor(0.0, device=device)
        logs: Dict[str, torch.Tensor] = {}

        # Base diffusion/reconstruction loss (always on)
        if self.rc_brd_enabled and self.rc_brd_loss_weighting == "per_band_min_snr":
            # A8 (DESIGN §8 v1.0f): bandwise min{SNR_b(t), γ} weighting.
            base_loss, base_logs = self._rc_brd_base_loss(
                pred_model, model_target, timesteps, tau)
        else:
            base_loss, base_logs = self._base_reconstruction_loss(
                pred_model, model_target, timesteps, tau
            )
        total_loss = total_loss + base_loss
        logs["loss/base_diffusion"] = base_loss.detach()
        if self.residual_bridge_enabled:
            logs["module/residual_output_scale"] = x0.new_tensor(
                self.residual_output_scale
            )
        logs.update(base_logs)

        if self.residual_bridge_enabled and mean_output is not None:
            from .frequency.haar import haar_dwt2

            target_ll1, _ = haar_dwt2(x0)
            target_ll2, _ = haar_dwt2(target_ll1)
            mean_error = mean_output["ll2"] - target_ll2
            mean_lowpass = torch.sqrt(
                mean_error.square() + self.mean_charbonnier_eps ** 2
            ).mean()
            weighted_mean = self.mean_loss_weight * mean_lowpass
            total_loss = total_loss + weighted_mean
            logs["loss/mean_lowpass"] = mean_lowpass.detach()
            logs["loss/mean_lowpass_weighted"] = weighted_mean.detach()

        # Pluggable loss terms
        for name, term in self.loss_terms.items():
            epoch_scale = self._loss_epoch_scale(name)
            if epoch_scale <= 0.0:
                logs[f"loss/{name}/enabled"] = x0.new_tensor(
                    float(term.enabled)
                )
                logs[f"loss/{name}/epoch_scale"] = x0.new_zeros(())
                continue
            loss_val, loss_logs = term(ctx)
            total_loss = total_loss + epoch_scale * loss_val
            for k, v in loss_logs.items():
                logs[f"loss/{k}"] = v.detach() if torch.is_tensor(v) else v
            logs[f"loss/{name}/epoch_scale"] = x0.new_tensor(
                epoch_scale
            )

        logs["loss/total"] = total_loss.detach()

        # Module status logs
        for name in self.priors:
            logs[f"module/{name}"] = torch.tensor(
                1.0 if self.priors[name].enabled else 0.0, device=device
            )
        # Gabor routing status (independent of whether the prior itself ran)
        for route, flag in self.gabor_routes.items():
            logs[f"module/gabor_{route}"] = torch.tensor(1.0 if flag else 0.0, device=device)
        logs["module/zero_adapter"] = torch.tensor(1.0 if self.zero_adapter_enabled else 0.0, device=device)
        logs["module/condition_dropout"] = torch.tensor(
            1.0 if self.condition_dropout.enabled else 0.0, device=device
        )
        logs["module/self_conditioning"] = torch.tensor(1.0 if self.self_conditioning else 0.0, device=device)
        logs["module/metadata_film"] = torch.tensor(1.0 if self.meta_enabled else 0.0, device=device)
        logs["module/segmenter"] = torch.tensor(1.0 if self.segmenter_enabled else 0.0, device=device)
        logs["module/conditional_mean"] = torch.tensor(
            1.0 if self.conditional_mean_enabled else 0.0, device=device
        )
        logs["module/conditional_mean_frozen"] = torch.tensor(
            1.0 if self.mean_frozen else 0.0, device=device
        )
        logs["module/residual_bridge"] = torch.tensor(
            1.0 if self.residual_bridge_enabled else 0.0, device=device
        )
        logs["module/residual_frequency"] = torch.tensor(
            1.0 if self.residual_frequency_enabled else 0.0, device=device
        )
        logs["module/residual_frequency_frozen"] = torch.tensor(
            1.0 if self.residual_frequency_frozen else 0.0, device=device
        )
        if self.rc_brd_enabled:
            # Only when rc_brd is on: the disabled log key set must stay
            # identical to the pre-integration baseline (FR-5.1).
            logs["module/rc_brd"] = torch.tensor(1.0, device=device)
            logs["module/rc_brd_specialist"] = torch.tensor(
                1.0 if self.rc_brd_head is not None else 0.0, device=device
            )
        logs["module/residual_frequency_native_only"] = torch.tensor(
            1.0
            if self.residual_frequency_destination_mode == "native_only"
            else 0.0,
            device=device,
        )
        if self.residual_frequency_mode == "boundary_reliable":
            for level, key in ((2, "gates_l2"), (1, "gates_l1")):
                gates = self._last_frequency_diagnostics.get(key)
                if gates is not None:
                    for index, band in enumerate(("lh", "hl", "hh")):
                        logs[f"frequency/gate_l{level}_{band}"] = (
                            gates[:, index].mean().detach()
                        )
            gate_tv = self._last_frequency_diagnostics.get("gate_tv")
            if gate_tv is not None:
                logs["frequency/gate_tv"] = gate_tv.detach()
        elif self.residual_frequency_mode == "spectral_evidence_router":
            destinations = ("native", "shallow", "null")
            for level in (2, 1):
                routes = self._last_frequency_diagnostics.get(f"routes_l{level}")
                if routes is not None:
                    n_routes = routes.shape[-1]
                    for index in range(min(n_routes, len(destinations))):
                        logs[f"frequency/route_l{level}_{destinations[index]}"] = (
                            routes[..., index].mean().detach()
                        )
                gates = self._last_frequency_diagnostics.get(f"gates_l{level}")
                if gates is not None:
                    for index, band in enumerate(("lh", "hl", "hh")):
                        logs[f"frequency/gate_l{level}_{band}"] = (
                            gates[:, index].mean().detach()
                        )
            # Per-level route diagnostics (entropy, active_mass, null quantiles)
            for level in (2, 1):
                for suffix in (
                    "active_mass",
                    "entropy",
                    "null_mean",
                    "null_p10",
                    "null_p50",
                    "null_p90",
                ):
                    key = f"route_l{level}_{suffix}"
                    val = self._last_frequency_diagnostics.get(key)
                    if val is not None:
                        logs[f"frequency/{key}"] = (
                            val.detach() if isinstance(val, torch.Tensor) else val
                        )
            routing_progress = self._last_frequency_diagnostics.get(
                "route_routing_progress"
            )
            if routing_progress is not None:
                logs["frequency/route_routing_progress"] = (
                    routing_progress.detach()
                )
            # Stable cross-policy names used by paired monitoring.  These are
            # present for native-only, fixed, learned, and prior-anchored
            # routers alike; prior-specific aliases below remain for backward
            # compatibility with existing C-run summaries.
            for diagnostic_key in (
                "route_native_mass",
                "route_shallow_mass",
                "route_null_mass",
            ):
                value = self._last_frequency_diagnostics.get(diagnostic_key)
                if value is not None:
                    logs[f"frequency/{diagnostic_key}"] = (
                        value.detach()
                        if isinstance(value, torch.Tensor)
                        else value
                    )
            if (
                getattr(
                    self.residual_preconditioner,
                    "_effective_policy",
                    "",
                )
                == "prior_anchored_learned"
            ):
                for diagnostic_key, log_key in (
                    ("route_active_progress", "active_progress"),
                    ("route_destination_progress", "destination_progress"),
                    ("route_anchor_scale", "anchor_scale"),
                    ("route_prior_active_mae", "prior_active_mae"),
                    (
                        "route_active_delta_abs_mean",
                        "active_delta_abs_mean",
                    ),
                    ("route_active_delta_abs_max", "active_delta_abs_max"),
                    (
                        "route_monotonic_violation",
                        "monotonic_violation",
                    ),
                    (
                        "route_monotonic_violation_fraction",
                        "monotonic_violation_fraction",
                    ),
                    ("route_shallow_mass", "shallow_mass"),
                    ("route_native_mass", "native_mass"),
                    ("route_null_mass", "null_mass"),
                    ("route_prior_active_mean", "prior_active_mean"),
                    ("route_active_mean", "active_mean"),
                ):
                    value = self._last_frequency_diagnostics.get(
                        diagnostic_key
                    )
                    if value is not None:
                        logs[f"frequency/prior_anchor_{log_key}"] = (
                            value.detach()
                            if isinstance(value, torch.Tensor)
                            else value
                        )
                level_names = ("l2", "l1")
                band_names = ("lh", "hl", "hh")
                for diagnostic_key, log_stem in (
                    ("route_prior_active", "prior"),
                    ("route_active", "active"),
                    ("route_active_delta", "delta"),
                    ("route_shallow_probability", "shallow"),
                    ("route_conditional_shallow", "conditional_shallow"),
                ):
                    value = self._last_frequency_diagnostics.get(
                        diagnostic_key
                    )
                    if value is None:
                        continue
                    for level_index, level_name in enumerate(level_names):
                        for band_index, band_name in enumerate(band_names):
                            logs[
                                "frequency/prior_anchor_"
                                f"{level_name}_{band_name}_{log_stem}"
                            ] = value[
                                :, level_index, band_index
                            ].mean().detach()
            # Uncertainty-aware selector (V2-05) diagnostics, surfaced only when
            # the selector ran (flag ON + router_confidence supplied).
            for ua_key in (
                "router_active_fraction",
                "router_confidence_threshold",
            ):
                ua_val = self._last_frequency_diagnostics.get(ua_key)
                if ua_val is not None:
                    logs[f"frequency/{ua_key}"] = (
                        ua_val.detach()
                        if isinstance(ua_val, torch.Tensor)
                        else ua_val
                    )
            for ua_key in (
                "router_confidence",
                "router_active",
                "router_abstained",
            ):
                ua_val = self._last_frequency_diagnostics.get(ua_key)
                if ua_val is not None and isinstance(ua_val, torch.Tensor):
                    logs[f"frequency/{ua_key}_mean"] = (
                        ua_val.float().mean().detach()
                    )
            # Injection RMS per level
            for lvl in (0, 1, 2, 3):
                rms_val = self._last_frequency_diagnostics.get(f"injection/l{lvl}_rms")
                if rms_val is not None:
                    logs[f"frequency/injection_l{lvl}_rms"] = rms_val.detach()
            for diagnostic_key, log_key in (
                ("effective/native_l2_rms", "effective_native_l2_rms"),
                ("effective/native_l1_rms", "effective_native_l1_rms"),
                (
                    "effective/shallow_l2_to_l1_rms",
                    "effective_shallow_l2_to_l1_rms",
                ),
                (
                    "effective/shallow_l1_to_l0_rms",
                    "effective_shallow_l1_to_l0_rms",
                ),
                (
                    "effective/low_frequency_l1_rms",
                    "effective_low_frequency_l1_rms",
                ),
                ("route_spatial_l2_std", "route_spatial_l2_std"),
                ("route_spatial_l1_std", "route_spatial_l1_std"),
            ):
                value = self._last_frequency_diagnostics.get(diagnostic_key)
                if value is not None:
                    logs[f"frequency/{log_key}"] = value.detach()
            # Gate TV
            gate_tv = self._last_frequency_diagnostics.get("gate_tv")
            if gate_tv is not None:
                logs["frequency/gate_tv"] = gate_tv.detach()
            # Policy
            policy = self._last_frequency_diagnostics.get("route_policy")
            if policy is not None:
                logs["frequency/route_policy"] = (
                    policy if isinstance(policy, (int, float))
                    else 0.0
                )
        logs["module/wavelet_unet"] = torch.tensor(
            1.0 if self.wavelet_unet_enabled else 0.0, device=device
        )
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
        initial_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """DDIM-style sampling with optional weak Classifier-Free Guidance.

        Args:
            cfg_scale: CFG scale [1.0, 1.5].  1.0 = no CFG.
                       Higher = stronger condition influence.
                       Keep ≤1.5 to avoid hallucinating lesions.
            initial_noise: Optional caller-supplied initial diffusion field.
                It must exactly match the CT tensor's shape, dtype, and device.
                When omitted, sampling draws a fresh standard-normal field.
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
        mean_pet: Optional[torch.Tensor] = None
        bridge_source = x_source
        if self.residual_bridge_enabled:
            if self.mean_predictor is None:
                raise RuntimeError("Residual bridge was enabled without a mean predictor")
            mean_pet = self.mean_predictor(x_source)["mean_pet"]
            bridge_source = torch.zeros_like(x_source)

        # Build metadata tensor once (shared across all denoising steps)
        meta_tensor = None
        if self.meta_enabled:
            meta_tensor = _meta_to_tensor(batch.get("meta"), B, device, x_source.dtype)

        T = self.noise_schedule.num_train_timesteps
        ddim_timesteps = torch.linspace(T - 1, 0, steps, device=device, dtype=torch.long)
        is_bbdm = hasattr(self.noise_schedule, 'm_t')
        has_custom_reverse = hasattr(self.noise_schedule, "step_from_prediction")

        # ---- Initial state ----
        if initial_noise is None:
            noise = torch.randn_like(x_source)
        else:
            if initial_noise.shape != x_source.shape:
                raise ValueError(
                    "initial_noise shape must match batch['ct']: "
                    f"{tuple(initial_noise.shape)} != {tuple(x_source.shape)}"
                )
            if initial_noise.device != x_source.device:
                raise ValueError(
                    "initial_noise device must match batch['ct']: "
                    f"{initial_noise.device} != {x_source.device}"
                )
            if initial_noise.dtype != x_source.dtype:
                raise ValueError(
                    "initial_noise dtype must match batch['ct']: "
                    f"{initial_noise.dtype} != {x_source.dtype}"
                )
            noise = initial_noise
        timesteps_T = torch.full((B,), T - 1, device=device, dtype=torch.long)
        condition = self.build_condition_bundle(batch, timesteps_T)
        # Constant across steps: residual endpoint in band coordinates (C7).
        rc_brd_endpoint = (self._rc_brd_endpoint_bands(x_source, mean_pet)
                           if self.rc_brd_enabled else None)
        if self.rc_brd_enabled:
            # DESIGN §9.3: z_T via q_marginal on zero r0 (bridge: μ_T+σ_T·ε;
            # vp: √(1−ᾱ)·ε), then back to the image domain.
            from .rc_brd import haar_forward2, haar_inverse2
            z_t_bands, _ = self.rc_brd_schedule.q_marginal(
                haar_forward2(torch.zeros_like(x_source)), timesteps_T,
                eps_bands=haar_forward2(noise), endpoint_bands=rc_brd_endpoint)
            x_t = haar_inverse2(z_t_bands)
        elif is_bbdm:
            # x_T = m_T·CT + (1-m_T)·0 + σ_T·ε ≈ CT + noise
            x_t = _add_noise(self.noise_schedule, torch.zeros_like(x_source),
                            noise, timesteps_T, condition, bridge_source)
        elif has_custom_reverse:
            # Custom schedules may define a non-isotropic forward prior state.
            x_t = self.noise_schedule.add_noise(torch.zeros_like(x_source), noise, timesteps_T, condition)
        else:
            # Standard: x_T = noise
            x_t = noise
        self_cond = torch.zeros_like(x_t) if self.self_conditioning else None

        # Track the terminal clean estimate.  The loop below updates x_t at every
        # step except the last; the last step's pred_x0 is the model's final
        # denoised output and must be what we return (not the previous x_t which
        # still carries schedule noise).
        final_pred_x0: Optional[torch.Tensor] = None
        final_output: Optional[torch.Tensor] = None

        step_range = range(len(ddim_timesteps))
        if progress:
            from tqdm import tqdm
            step_range = tqdm(step_range, desc="Sampling")

        for i in step_range:
            t = ddim_timesteps[i]
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            condition = self.build_condition_bundle(batch, t_batch)
            denoiser_state = x_t
            if (
                self.residual_frequency_enabled
                and self.residual_preconditioner is not None
                and self.residual_frequency_mode == "legacy"
                and not self.residual_preconditioner.inject_wavelet
                and self.residual_preconditioner.state_modulation
            ):
                denoiser_state = self.residual_preconditioner.modulate_residual(
                    x_t, condition.maps.get("gabor_orientation")
                )
            skip_inj = self._build_skip_injections(
                condition,
                timesteps=t_batch,
                noisy_residual=x_t if self.residual_bridge_enabled else None,
                router_confidence=batch.get("router_confidence"),
            )
            semantic_tokens = condition.tokens.get("semantic")
            model_input = self._model_input(denoiser_state, x_source, self_cond)
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

            if self.rc_brd_head is not None:
                # Same specialist read as the training path (DESIGN §9.2).
                pred_x0 = self._rc_brd_apply_specialist(pred_x0, t_batch, x_source)

            # DDIM step to next timestep
            if i < len(ddim_timesteps) - 1:
                t_next = ddim_timesteps[i + 1]
                t_next_batch = torch.full_like(t_batch, t_next)
                if self.rc_brd_enabled:
                    # DESIGN §9.3 (v1.0g dual readout): endpoint_ddim keeps the
                    # deterministic eps-hat reuse rule (κ=0 matches
                    # _bbdm_ddim_step within fp32 tolerance); ancestral_mc
                    # draws a fresh per-step xi and passes it via the noise
                    # argument so step_from_prediction samples z_{t_prev}
                    # from the true bandwise posterior (mean+sqrt(var)*xi,
                    # AUDIT 5 §3.3 X).
                    from .rc_brd import haar_forward2, haar_inverse2
                    step_noise = None
                    if self.rc_brd_sampling_mode == "ancestral_mc":
                        step_noise = haar_forward2(torch.randn_like(x_t))
                    z_next = self.rc_brd_schedule.step_from_prediction(
                        haar_forward2(pred_x0), haar_forward2(x_t),
                        t_next_batch, t_batch, noise=step_noise,
                        endpoint_bands=rc_brd_endpoint)
                    x_t = haar_inverse2(z_next)
                elif has_custom_reverse:
                    x_t = self.noise_schedule.step_from_prediction(
                        x_t,
                        pred_x0,
                        t_batch,
                        t_next_batch,
                        condition,
                        x_source=bridge_source,
                    )
                elif is_bbdm:
                    # BBDM: x_{t-1} = m_{t-1}·CT + (1-m_{t-1})·pred_x0
                    # As t→0: m→0, x→pred_x0=PET ✓
                    x_t = _bbdm_ddim_step(
                        self.noise_schedule,
                        x_t,
                        pred_x0,
                        bridge_source,
                        t_batch,
                        t_next_batch,
                    )
                else:
                    # Standard DDIM
                    alpha_t = _get_alpha_cumprod(self.noise_schedule, t_batch)
                    alpha_next = _get_alpha_cumprod(self.noise_schedule, t_next_batch)
                    while alpha_t.dim() < x_t.dim():
                        alpha_t = alpha_t.unsqueeze(-1)
                        alpha_next = alpha_next.unsqueeze(-1)
                    eps_pred = (x_t - alpha_t.sqrt() * pred_x0) / (1 - alpha_t).sqrt().clamp_min(1e-8)
                    x_t = alpha_next.sqrt() * pred_x0 + (1 - alpha_next).sqrt() * eps_pred
            else:
                # Terminal step (t≈0): pred_x0 is the final clean estimate and
                # must be returned directly.  x_t at this point still carries
                # residual schedule noise, so we do NOT return it.
                final_pred_x0 = pred_x0
                final_output = output

        # Prefer the final-step pred_x0.  Fall back to x_t only if the loop did
        # not execute (defensive — steps is always ≥1 in practice).
        final_model_prediction = final_pred_x0 if final_pred_x0 is not None else x_t
        if self.residual_bridge_enabled:
            if mean_pet is None:
                raise RuntimeError("Residual sampling did not produce a conditional mean")
            synthetic_pet, applied_correction = self._compose_residual_output(
                mean_pet,
                final_model_prediction,
            )
            result = {
                "synthetic_pet": synthetic_pet,
                "mean_pet": mean_pet,
                "pred_residual": applied_correction,
                "raw_pred_residual": final_model_prediction,
            }
        else:
            synthetic_pet = final_model_prediction
            result = {"synthetic_pet": synthetic_pet}
        if self.enable_heteroscedastic and final_output is not None:
            # logvar must come from the same terminal model output as pred_x0.
            result["logvar"] = final_output[:, 1:].clamp(
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
        n_samples: Optional[int] = None,
        num_steps: Optional[int] = None,
        progress: bool = False,
        aggregate: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        """Monte Carlo sampling: n_samples independent forward passes.

        Returns:
            synthetic_pet       [B, 1, H, W]  configured mean/median point estimate
            epistemic_var       [B, 1, H, W]  variance across MC samples
            aleatoric_logvar    [B, 1, H, W]  heteroscedastic log-variance (if enabled)
            confidence_map      [B, 1, H, W]  combined confidence (0=low, 1=high)
            samples             [n, B, 1, H, W]  all individual samples
        """
        if aggregate not in {"mean", "median"}:
            raise ValueError("aggregate must be 'mean' or 'median'")
        if n_samples is None:
            # C6 (PRD §3): rc_brd mc_mean pins K=readout.mc_samples; the legacy
            # default stays 20 when rc_brd is disabled.
            n_samples = self.rc_brd_mc_samples if self.rc_brd_enabled else 20
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
        point_estimate = (
            mean
            if aggregate == "mean"
            else torch.quantile(samples.float(), 0.5, dim=0).to(samples.dtype)
        )
        epistemic_var = samples.var(dim=0)          # [B, 1, H, W]

        output: Dict[str, torch.Tensor] = {
            "synthetic_pet": point_estimate,
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
        require_checkpoint_lineage = bool(
            data_cfg.get("require_cache_lineage", False)
        )
        expected_data_lineage = None
        if require_checkpoint_lineage:
            from src.data.lineage import load_checkpoint_data_lineage
            expected_data_lineage = load_checkpoint_data_lineage(config)

        return cls(
            image_size=data_cfg.get("image_size", 192),
            objective=model_cfg.get("objective", "pred_x0"),
            initialization_seed=model_cfg.get("initialization_seed"),
            enable_heteroscedastic=model_cfg.get("enable_heteroscedastic", True),
            heteroscedastic_logvar_min=model_cfg.get("heteroscedastic_logvar_min", -6.0),
            heteroscedastic_logvar_max=model_cfg.get("heteroscedastic_logvar_max", 2.0),
            prior_configs=prior_cfgs,
            adapter_config=modules_cfg.get("zero_adapter", {}),
            noise_config=resolve_noise_config(modules_cfg),
            loss_configs=loss_cfg,
            condition_dropout_config=modules_cfg.get("condition_dropout", {}),
            base_loss_config=model_cfg.get("base_loss", {}),
            conditional_mean_config=modules_cfg.get("conditional_mean", {}),
            residual_bridge_config=modules_cfg.get("residual_bridge", {}),
            residual_frequency_config=modules_cfg.get("residual_frequency", {}),
            rc_brd_config=modules_cfg.get("rc_brd", {}),
            rc_brd_expected_fold=config.get("runtime", {}).get("rc_brd_fold"),
            wavelet_unet_config=modules_cfg.get("wavelet_unet", {}),
            meta_config=model_cfg.get("metadata", config.get("metadata", {})),
            segmenter_config=model_cfg.get("segmenter", config.get("segmenter", {})),
            expected_data_lineage=expected_data_lineage,
            require_checkpoint_lineage=require_checkpoint_lineage,
            self_conditioning_config=model_cfg.get("self_conditioning", {}),
            sample_scheduler=model_cfg.get("sample_scheduler", "ddim"),
            eval_sampling_steps=config.get("runtime", {}).get("eval_sampling_steps", 20),
        )

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
