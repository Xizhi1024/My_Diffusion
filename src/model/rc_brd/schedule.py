"""Bandwise forward marginal / posterior / sampling schedule for RC-BRD.

Implements DESIGN §4 (docs/prd/DESIGN_RC_BRD_v1.md):

* `bridge_time_changed` — the recoverability-contracted time-changed residual
  Brownian bridge of [审计] §3.1–3.3, with the σ alignment mandated by DESIGN
  §4: the audit's σ_b² equals 2·sigma_bridge², so κ=0 reproduces the scalar
  `BBDMBridgeSchedule(sigma_scale=1.0)` path elementwise within fp32
  tolerance (rtol 1e-5 / atol 1e-6).
* `vp_bandwise` — the v1.4 bandwise VP forward of [计划] §3.6 (λ warp → ᾱ →
  VP marginal), kept as an explicit ablation switch (C1).  PRD v1.0.2
  non-equivalence: at κ=0 only the clock degenerates (m≡u) — the VP marginal
  √ᾱ·z0+√(1−ᾱ)·ε never equals the bridge marginal, so this arm is an
  exploration switch, never a scalar-D1 twin (its posterior/transition raise
  NotImplementedError and its sampling step is deterministic-DDIM only).

Numerics: m/λ/ᾱ are computed internally in float64 and cast back to the
input dtype; a numerically non-monotone clock raises ValueError (DESIGN §4).

The optional `contract` is duck-typed at runtime: the required surface is
`effective_c(group, log_snr_tensor) -> Tensor` plus attributes `eta_max`,
`band_groups`, `log_snr_grid`, `b_active`.  The static annotation references
`RecoverabilityContract` under TYPE_CHECKING only — `contract.py` lands in a
later batch and is never imported at runtime.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .wavelet import BAND_NAMES

if TYPE_CHECKING:  # pragma: no cover - static-only import (batch 1B file)
    from .contract import RecoverabilityContract

# C1 ([审计] §1.2): bridge marginal is the mainline; v1.4 VP stays as an arm.
FORWARD_MODES = ("bridge_time_changed", "vp_bandwise")
# C7 ([计划] §3.6): residual endpoint coordinate convention.
ENDPOINT_MODES = ("zeros", "ct_minus_mean")
# DESIGN §4: the warped clock is trapezoid-integrated on a fine u-grid with
# >= 4·T nodes; 4 subdivisions per timestep make every u = t/T a grid node.
_CLOCK_GRID_SUBDIVISIONS = 4
# σ guard in the deterministic bridge step; mirrors clamp_min(1e-6) of
# src/model/slmf_bbdm.py::_bbdm_ddim_step (DESIGN §4 step_from_prediction).
# v1.0g: numeric fallback for NON-degenerate steps only — degenerate
# m_t∈{0,1} steps take the explicit posterior branch instead (AUDIT 5 §3.3).
_SIGMA_DIV_EPS = 1e-6
# [DESIGN §4 v1.0g / AUDIT 5 §3.3] tolerance of the degenerate-step branch:
# m_t within this distance of {0, 1} makes ε̂=(z_t−μ_t)/σ_t non-invertible.
_M_ENDPOINT_TOL = 1e-12
# Duck-typed runtime surface required from a recoverability contract.
_CONTRACT_ATTRIBUTES = ("effective_c", "eta_max", "band_groups", "log_snr_grid", "b_active")


def _expand(values: torch.Tensor, ndim: int) -> torch.Tensor:
    """Reshape a [B] tensor to broadcast against an [B,1,h,w] band tensor."""
    return values.view(-1, *([1] * (ndim - 1)))


def _validate_contract_duck_type(contract: object) -> None:
    """Fail-closed duck-type check (schedule must not import contract.py)."""
    missing = [name for name in _CONTRACT_ATTRIBUTES if not hasattr(contract, name)]
    if missing:
        raise ValueError(
            f"contract must provide the duck-typed surface {_CONTRACT_ATTRIBUTES} (missing: {missing}); "
            "schedule relies on duck typing and never imports contract.py at runtime"
        )
    if not callable(getattr(contract, "effective_c")):
        raise ValueError("contract.effective_c must be callable: (group, log_snr) -> Tensor")


@dataclass(frozen=True)
class BandwiseScheduleConfig:
    """Immutable config (DESIGN §4). Defaults reproduce scalar D1 at κ=0;
    enum/range violations raise ValueError at construction (fail-closed)."""

    num_timesteps: int = 1000
    forward_mode: str = "bridge_time_changed"
    kappa: float = 0.0
    sigma_bridge: float = 1.0          # σ_b (bridge mode); [审计] §3.2 σ_b²=2σ²
    lambda_min: float = -10.0          # vp mode λ endpoint (max noise)
    lambda_max: float = 10.0           # vp mode λ endpoint (clean signal)
    endpoint_mode: str = "zeros"

    def __post_init__(self) -> None:
        if isinstance(self.num_timesteps, bool) or not isinstance(self.num_timesteps, int):
            raise ValueError(f"num_timesteps must be a positive int, got {self.num_timesteps!r}")
        if self.num_timesteps < 1:
            raise ValueError(f"num_timesteps must be >= 1, got {self.num_timesteps}")
        if self.forward_mode not in FORWARD_MODES:
            raise ValueError(f"forward_mode must be one of {FORWARD_MODES}, got {self.forward_mode!r}")
        if self.endpoint_mode not in ENDPOINT_MODES:
            raise ValueError(f"endpoint_mode must be one of {ENDPOINT_MODES}, got {self.endpoint_mode!r}")
        # [DESIGN §4 v1.0g / AUDIT 5 §3.3] finite-domain validation: NaN/±inf
        # would silently poison the clock integral and posterior variances.
        for name in ("kappa", "sigma_bridge", "lambda_min", "lambda_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
        try:
            steps_finite = math.isfinite(self.num_timesteps)
        except OverflowError:  # int too large to view as a float
            steps_finite = False
        if not steps_finite:
            raise ValueError(f"num_timesteps must be finite, got {self.num_timesteps!r}")
        if not self.sigma_bridge > 0:
            raise ValueError(f"sigma_bridge must be > 0, got {self.sigma_bridge}")
        if not self.lambda_min < self.lambda_max:
            raise ValueError(f"lambda_min must be < lambda_max, got ({self.lambda_min}, {self.lambda_max})")


class BandwiseBridgeSchedule(nn.Module):
    """Bandwise forward/posterior/sampling schedule (DESIGN §4).

    `contract=None` with `kappa=0` yields the pure scalar residual bridge
    (D1): every group shares m(u)=u and σ_t=σ_b·√(2m(1−m)) — bridge arm
    only; the vp arm shares the identity clock at κ=0 but keeps its own
    non-equivalent marginal (PRD v1.0.2).
    """

    def __init__(
        self,
        config: BandwiseScheduleConfig,
        contract: "RecoverabilityContract | None" = None,
    ) -> None:
        """Validate config/contract coupling; κ≠0 without contract → ValueError."""
        super().__init__()
        if not isinstance(config, BandwiseScheduleConfig):
            raise TypeError(f"config must be BandwiseScheduleConfig, got {type(config)!r}")
        if contract is not None:
            _validate_contract_duck_type(contract)
        if config.kappa != 0.0 and contract is None:
            raise ValueError("kappa != 0 requires a contract; contract=None only valid for kappa=0")
        self.config = config
        self._contract = contract
        if contract is None:
            self._band_to_group: dict[str, str] | None = None
        else:
            self._band_to_group = {
                band: group for group, band_list in contract.band_groups.items() for band in band_list
            }
        # Instance-level memo of computed clocks (never a module-level global).
        self._m_cache: dict[str, torch.Tensor] = {}

    # ---- clock / m sequence ------------------------------------------------

    def _compute_m_sequence(self, group: str) -> torch.Tensor:
        """Integrate the warped clock on the fine u-grid (float64, CPU)."""
        num_t = self.config.num_timesteps
        if self._contract is None or self.config.kappa == 0.0:
            # κ=0 → exact identity u = t/T ([审计] §3.2), contract irrelevant.
            return torch.arange(num_t + 1, dtype=torch.float64) / num_t
        # [审计] §3.1: ρ=exp{κ(2c̃(u)−1)}; [审计] §3.3 keeps ρ in e^{±κη_max}.
        # [计划] v1.5 §3.6 / DESIGN v1.0f lookup: λ₀(u)=log((1−u)/(ν²·u)) with
        # base clock m₀(u)=u, ν²=2·sigma_bridge²; ±inf endpoints clip to
        # [λ_min, λ_max].  The v1.4 linear λ warp is deprecated.
        nodes = num_t * _CLOCK_GRID_SUBDIVISIONS
        u = torch.arange(nodes + 1, dtype=torch.float64) / nodes
        nu_squared = 2.0 * self.config.sigma_bridge ** 2
        log_snr = torch.log((1.0 - u) / (nu_squared * u)).clamp(
            self.config.lambda_min, self.config.lambda_max
        )
        c_tilde = self._contract.effective_c(group, log_snr).to(dtype=torch.float64)
        if c_tilde.shape != u.shape:
            raise ValueError(
                f"contract.effective_c returned shape {tuple(c_tilde.shape)}, "
                f"expected {tuple(u.shape)} for group {group!r}"
            )
        rho = torch.exp(self.config.kappa * (2.0 * c_tilde - 1.0))
        increments = 0.5 * (rho[:-1] + rho[1:]) / nodes  # trapezoid rule
        cumulative = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(increments, dim=0)])
        total = cumulative[-1]
        if not torch.isfinite(total) or total <= 0.0:
            raise ValueError(
                f"clock integral for group {group!r} is not positive/finite; "
                "kappa is numerically out of range"
            )
        m_fine = cumulative / total  # pins m[0]=0, m[end]=1 exactly
        m = m_fine[::_CLOCK_GRID_SUBDIVISIONS]
        diffs = m[1:] - m[:-1]
        if not bool(torch.isfinite(m).all()) or not bool((diffs > 0).all()):
            raise ValueError(
                f"clock m_sequence for group {group!r} is not strictly monotone "
                "(numerical violation); decrease |kappa| or check the contract"
            )
        return m

    def m_sequence(self, group: str) -> torch.Tensor:
        """Warped clock m_b at u=t/T, t=0..T → [T+1] float64 CPU tensor.

        bridge: m_b(u)=∫₀^uρ/∫₀^1ρ, ρ=exp{κ(2c̃_b(u)−1)} ([审计] §3.1),
        trapezoid-integrated on a fine grid (≥4·T nodes); vp: A_b(u), the
        same integral ([计划] §3.6).  c̃ lookup per [计划] v1.5 §3.6 /
        DESIGN v1.0f: λ₀(u)=log((1−u)/(ν²·u)), ν²=2σ², clipped to
        [λ_min, λ_max].  m[0]=0, m[T]=1 exactly; monotonicity is verified
        numerically (ValueError on violation).  kappa==0 → exact identity
        clock m≡u in BOTH modes (only the bridge arm then degenerates to the
        scalar D1 path — PRD v1.0.2); contract may be None.  Unknown group
        (contract bound) → KeyError.
        """
        if self._contract is not None and group not in self._contract.band_groups:
            raise KeyError(group)
        cached = self._m_cache.get(group)
        if cached is None:
            cached = self._compute_m_sequence(group)
            self._m_cache[group] = cached
        return cached.clone()

    # ---- shared validation helpers ------------------------------------------

    def _validate_timesteps(self, t: torch.Tensor, name: str) -> torch.Tensor:
        """Validate a [B] int64 timestep tensor with values in [0, T]."""
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(t)!r}")
        if t.dtype != torch.int64:
            raise ValueError(f"{name} must have dtype int64, got {t.dtype}")
        if t.ndim != 1:
            raise ValueError(f"{name} must be 1-D with shape [B], got {tuple(t.shape)}")
        if bool((t < 0).any()) or bool((t > self.config.num_timesteps).any()):
            raise ValueError(f"{name} values must lie in [0, {self.config.num_timesteps}]")
        return t

    def _validate_band_dict(
        self, bands: dict, name: str, *, batch_size: int | None = None, keys=None,
    ) -> None:
        """Validate a band dict of [B,1,h,w] tensors keyed by canonical names."""
        if not isinstance(bands, dict):
            raise ValueError(f"{name} must be a dict[str, torch.Tensor], got {type(bands)!r}")
        if not bands:
            raise ValueError(f"{name} must contain at least one band")
        unknown = sorted(band for band in bands if band not in BAND_NAMES)
        if unknown:
            raise ValueError(f"{name} contains unknown band names {unknown}; expected subsets of {BAND_NAMES}")
        for band, value in bands.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"{name}[{band!r}] must be a torch.Tensor, got {type(value)!r}")
            if value.ndim != 4 or value.shape[1] != 1:
                raise ValueError(f"{name}[{band!r}] must be [B,1,h,w], got shape {tuple(value.shape)}")
            if batch_size is not None and value.shape[0] != batch_size:
                raise ValueError(f"{name}[{band!r}] batch size {value.shape[0]} != timestep batch {batch_size}")
        if keys is not None:
            missing = sorted(set(keys) - set(bands))
            extra = sorted(set(bands) - set(keys))
            if missing or extra:
                raise ValueError(f"{name} keys must match {sorted(set(keys))} (missing {missing}, extra {extra})")

    def _group_of_band(self, band: str) -> str:
        """Map a canonical band name to its contract group (identity if None)."""
        if band not in BAND_NAMES:
            raise ValueError(f"unknown band name {band!r}; expected one of {BAND_NAMES}")
        if self._band_to_group is not None:
            if band not in self._band_to_group:
                raise ValueError(f"band {band!r} not covered by contract band_groups {sorted(self._contract.band_groups)}")
            return self._band_to_group[band]
        return band

    def _resolve_endpoints(
        self, z0_bands: dict[str, torch.Tensor], endpoint_bands
    ) -> dict[str, torch.Tensor]:
        """Return band → float64 endpoint e^b (DESIGN §4; [计划] §3.6).

        'zeros' → e=0.  'ct_minus_mean' → e must be explicit band-coordinate
        tensors (haar_forward2 of CT−μ).  Missing keys / shape mismatches /
        image-domain (4× upsampled) tensors raise ValueError: image-coordinate
        μ/CT in the residual endpoint is forbidden ([计划] §3.6).
        """
        if self.config.endpoint_mode == "zeros":
            return {b: torch.zeros_like(z, dtype=torch.float64) for b, z in z0_bands.items()}
        if endpoint_bands is None:
            raise ValueError(
                "endpoint_mode='ct_minus_mean' requires explicit endpoint_bands "
                "(residual endpoint e_r = CT−μ transformed to Haar band coordinates)"
            )
        missing = [band for band in z0_bands if band not in endpoint_bands]
        if missing:
            raise ValueError(f"endpoint_bands is missing band keys: {sorted(missing)}")
        resolved: dict[str, torch.Tensor] = {}
        for band, z0 in z0_bands.items():
            endpoint = endpoint_bands[band]
            if not isinstance(endpoint, torch.Tensor) or endpoint.ndim != 4:
                raise ValueError(f"endpoint_bands[{band!r}] must be a [B,1,h,w] tensor")
            if endpoint.shape != z0.shape:
                if (
                    endpoint.shape[0] == z0.shape[0]
                    and endpoint.shape[-2] == 4 * z0.shape[-2]
                    and endpoint.shape[-1] == 4 * z0.shape[-1]
                ):
                    raise ValueError(
                        f"endpoint_bands[{band!r}] has image-domain resolution {tuple(endpoint.shape)} "
                        "(4× the band grid): image-coordinate μ/CT tensors are forbidden as "
                        "residual endpoints ([计划] §3.6); transform with haar_forward2 first"
                    )
                raise ValueError(
                    f"endpoint_bands[{band!r}] shape {tuple(endpoint.shape)} != band shape {tuple(z0.shape)}"
                )
            resolved[band] = endpoint.to(dtype=torch.float64)
        return resolved

    def _alpha_hat_values(self, m: torch.Tensor) -> torch.Tensor:
        """ᾱ_b = σ(λ_max+(λ_min−λ_max)·m) ([计划] §3.6 warp), float64."""
        log_snr = self.config.lambda_max + (self.config.lambda_min - self.config.lambda_max) * m
        return torch.sigmoid(log_snr)

    # ---- forward marginal ----------------------------------------------------

    def q_marginal(
        self,
        z0_bands: dict[str, torch.Tensor],
        t: torch.Tensor,
        eps_bands: dict[str, torch.Tensor] | None = None,
        endpoint_bands: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Bandwise forward marginal; returns (z_t bands, target bands).

        t: [B] int64 in [0, T].  bridge ([审计] §3.2, σ-aligned):
        mean=(1−m_t)z0+m_t·e, std=σ_b·√(2·m_t(1−m_t)), z_t=mean+std·ε;
        target=ẑ0-target=z0 (pred_x0).  vp ([计划] §3.6):
        ᾱ_b(u)=σ(λ_max+(λ_min−λ_max)·m_b(u)), z_t=√ᾱ·z0+√(1−ᾱ)·ε; endpoints
        unused (the VP marginal has no second fixed endpoint).  PRD v1.0.2:
        at κ=0 the vp clock is m≡u but its marginal stays ≠ the bridge
        marginal — a non-equivalent exploration arm by design.
        endpoint_mode='zeros' → e=0; 'ct_minus_mean' → e=endpoint_bands[band]
        (band-keyed; missing keys/shape violations raise ValueError).
        eps_bands=None draws fresh standard noise from the torch global RNG
        (pass eps explicitly for reproducibility).
        """
        t = self._validate_timesteps(t, "t")
        self._validate_band_dict(z0_bands, "z0_bands", batch_size=t.shape[0])
        if eps_bands is not None:
            self._validate_band_dict(
                eps_bands, "eps_bands", batch_size=t.shape[0], keys=z0_bands.keys()
            )
        bridge = self.config.forward_mode == "bridge_time_changed"
        endpoints = self._resolve_endpoints(z0_bands, endpoint_bands) if bridge else {}
        z_t_out: dict[str, torch.Tensor] = {}
        for band, z0 in z0_bands.items():
            group = self._group_of_band(band)
            m_t = self.m_sequence(group).to(device=t.device)[t]
            z0_64 = z0.to(dtype=torch.float64)
            if eps_bands is not None:
                eps_64 = eps_bands[band].to(dtype=torch.float64)
            else:
                eps_64 = torch.randn_like(z0, dtype=torch.float64)
            if bridge:
                # [审计] §3.2 marginal with σ_b²=2·sigma_bridge².
                m_view = _expand(m_t, z0_64.ndim)
                mean = (1.0 - m_view) * z0_64 + m_view * endpoints[band]
                var = (2.0 * m_t * (1.0 - m_t)).clamp_min(0.0)
                std = self.config.sigma_bridge * torch.sqrt(var)
                z_t_64 = mean + _expand(std, z0_64.ndim) * eps_64
            else:
                alpha = self._alpha_hat_values(m_t)
                sqrt_a = torch.sqrt(alpha)
                sqrt_1ma = torch.sqrt((1.0 - alpha).clamp_min(0.0))
                z_t_64 = _expand(sqrt_a, z0_64.ndim) * z0_64 + _expand(sqrt_1ma, z0_64.ndim) * eps_64
            z_t_out[band] = z_t_64.to(dtype=z0.dtype)
        return z_t_out, dict(z0_bands)

    # ---- posterior / sampling step --------------------------------------------

    def posterior(
        self,
        z0_bands: dict[str, torch.Tensor],
        z_t_bands: dict[str, torch.Tensor],
        t_prev: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Closed-form bridge posterior q(z_{t_prev} | z_t, z0) → (mean, var).

        [审计] §3.2 (σ-aligned): μ=μ_s+(m_s/m_t)(z_t−μ_t),
        var=2σ²·m_s(m_t−m_s)/m_t, μ_j=(1−m_j)z0+m_j·e.  The endpoint e cancels
        analytically, so no endpoint input is needed.  vp mode raises
        NotImplementedError (C1).  t_prev==t → variance 0, mean z_t.
        Numerically negative variances clamp to 0 with a RuntimeWarning.
        step_from_prediction(noise=...) consumes this closed form as its
        ancestral sampling branch (DESIGN §4 v1.0g).
        Returns per-band tensors in the dtype of z_t_bands.
        """
        if self.config.forward_mode == "vp_bandwise":
            raise NotImplementedError("bridge posterior is not available in forward_mode='vp_bandwise' (C1)")
        t_prev = self._validate_timesteps(t_prev, "t_prev")
        t = self._validate_timesteps(t, "t")
        if t_prev.shape != t.shape:
            raise ValueError(f"t_prev {tuple(t_prev.shape)} and t {tuple(t.shape)} must match")
        self._validate_band_dict(z0_bands, "z0_bands", batch_size=t.shape[0])
        self._validate_band_dict(
            z_t_bands, "z_t_bands", batch_size=t.shape[0], keys=z0_bands.keys()
        )
        if bool((t_prev > t).any()):
            raise ValueError("posterior requires t_prev <= t for every sample")
        same = t_prev == t
        sigma2 = self.config.sigma_bridge ** 2
        mean_out: dict[str, torch.Tensor] = {}
        var_out: dict[str, torch.Tensor] = {}
        for band, z0 in z0_bands.items():
            z_t = z_t_bands[band]
            group = self._group_of_band(band)
            m_seq = self.m_sequence(group).to(device=t.device)
            m_s, m_t = m_seq[t_prev], m_seq[t]
            safe_m_t = torch.where(same, torch.ones_like(m_t), m_t)
            ratio = m_s / safe_m_t  # [审计] §3.2 m_s/m_t
            var = 2.0 * sigma2 * m_s * (m_t - m_s) / safe_m_t
            if bool((var < 0).any()):
                warnings.warn(
                    f"posterior variance for band {band!r} was numerically negative; clamped to 0",
                    RuntimeWarning,
                    stacklevel=2,
                )
                var = var.clamp_min(0.0)
            var = torch.where(same, torch.zeros_like(var), var)
            z0_64 = z0.to(dtype=torch.float64)
            zt_64 = z_t.to(dtype=torch.float64)
            mean = (1.0 - _expand(ratio, z0_64.ndim)) * z0_64 + _expand(ratio, zt_64.ndim) * zt_64
            mean = torch.where(_expand(same, zt_64.ndim), zt_64, mean)  # t_prev==t → z_t
            mean_out[band] = mean.to(dtype=z_t.dtype)
            var_out[band] = var.to(dtype=z_t.dtype)
        return mean_out, var_out

    def step_from_prediction(
        self,
        z0_pred_bands: dict[str, torch.Tensor],
        z_t_bands: dict[str, torch.Tensor],
        t_prev: torch.Tensor,
        t: torch.Tensor,
        noise: dict | None = None,
        endpoint_bands: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Bandwise sampling step (bridge arm; DESIGN §4 v1.0g dual semantics).

        - noise=None (default, endpoint_ddim readout) — deterministic ε̂-reuse
          rule: μ_j^b=m_j·e^b+(1−m_j)·ẑ0^b with e^b=0 for endpoint_mode='zeros'
          (endpoint_bands then None) and e^b=endpoint_bands[band] for
          'ct_minus_mean' (then required; band-coordinate tensors only,
          image-domain μ/CT raises via _resolve_endpoints);
          ε̂^b=(z_t^b−μ_t^b)/σ_t^b with σ_t^b=σ_b·√(2m_t(1−m_t)),
          z_{t_prev}^b=μ_{t_prev}^b+σ_{t_prev}^b·ε̂^b.  κ=0 matches
          _bbdm_ddim_step within fp32 tolerance (existing equivalence
          preserved).  Degenerate start m_t∈{0,1} (tolerance
          _M_ENDPOINT_TOL) makes ε̂ non-invertible — the step returns the
          closed-form posterior mean instead (explicit v1.0g branch); σ_t's
          clamp_min is a numeric fallback for NON-degenerate steps only and
          never masks the degenerate division.
        - noise≠None (ancestral_mc readout) — correct ancestral posterior
          sampling z_{t_prev}^b=posterior_mean+√posterior_var·noise^b with
          the closed form of posterior() ([审计] §3.2); the retired
          ε̂-reuse+σ_{t_prev}·noise stacking is gone (AUDIT 5 §3.3 X item).
          Degenerate m_t needs no special case here (the closed form's only
          division, by m_t, is guarded by the t_prev==t branch of posterior).
        t_prev==0 → ẑ0 pinned (both modes).  vp arm: deterministic DDIM-style
        step only; noise≠None → NotImplementedError (PRD v1.0.2-2).
        """
        t_prev = self._validate_timesteps(t_prev, "t_prev")
        t = self._validate_timesteps(t, "t")
        if t_prev.shape != t.shape:
            raise ValueError(f"t_prev {tuple(t_prev.shape)} and t {tuple(t.shape)} must match")
        if bool((t_prev > t).any()):
            raise ValueError("step_from_prediction requires t_prev <= t")
        self._validate_band_dict(z0_pred_bands, "z0_pred_bands", batch_size=t.shape[0])
        self._validate_band_dict(
            z_t_bands, "z_t_bands", batch_size=t.shape[0], keys=z0_pred_bands.keys()
        )
        if noise is not None:
            self._validate_band_dict(
                noise, "noise", batch_size=t.shape[0], keys=z0_pred_bands.keys()
            )
        bridge = self.config.forward_mode == "bridge_time_changed"
        if not bridge and noise is not None:
            # PRD v1.0.2-2: the vp arm is a non-equivalent exploration switch
            # and carries no stochastic ancestral sampler.
            raise NotImplementedError(
                "step_from_prediction(noise=...) is bridge-only: the vp_bandwise "
                "arm supports the deterministic DDIM-style step only (PRD v1.0.2-2)"
            )
        # [DESIGN v1.0a] endpoint coupling: 'zeros' forbids endpoint_bands;
        # 'ct_minus_mean' requires validated band-coordinate endpoint_bands.
        # Validation applies to the deterministic and ancestral paths alike.
        if bridge:
            if self.config.endpoint_mode == "zeros":
                if endpoint_bands is not None:
                    raise ValueError(
                        "step_from_prediction got endpoint_bands while "
                        "endpoint_mode='zeros'; pass endpoint_bands=None"
                    )
            else:
                self._resolve_endpoints(z0_pred_bands, endpoint_bands)
        elif endpoint_bands is not None:
            raise ValueError("endpoint_bands is unused in vp_bandwise mode; pass None")
        if bridge and noise is not None:
            # [DESIGN §4 v1.0g] ancestral draw: z_s=μ_post+√var_post·ξ with
            # the closed form of posterior() ([审计] §3.2; endpoint e cancels).
            mean_bands, var_bands = self.posterior(z0_pred_bands, z_t_bands, t_prev, t)
            out: dict[str, torch.Tensor] = {}
            for band, z0_pred in z0_pred_bands.items():
                mean_64 = mean_bands[band].to(dtype=torch.float64)
                # posterior() variance is per-sample [B]; expand for broadcast.
                std_64 = torch.sqrt(var_bands[band].to(dtype=torch.float64).clamp_min(0.0))
                z_prev = mean_64 + _expand(std_64, noise[band].ndim) * noise[band].to(dtype=torch.float64)
                pinned = _expand(t_prev == 0, z_prev.ndim)
                z_prev = torch.where(pinned, z0_pred.to(dtype=torch.float64), z_prev)
                out[band] = z_prev.to(dtype=z_t_bands[band].dtype)
            return out
        if bridge:
            endpoints = (
                {b: torch.zeros_like(z, dtype=torch.float64) for b, z in z0_pred_bands.items()}
                if self.config.endpoint_mode == "zeros"
                else {b: e.to(dtype=torch.float64) for b, e in endpoint_bands.items()}
            )
        out = {}
        for band, z0_pred in z0_pred_bands.items():
            z_t = z_t_bands[band]
            group = self._group_of_band(band)
            m_seq = self.m_sequence(group).to(device=t.device)
            m_t, m_p = m_seq[t], m_seq[t_prev]
            z0_64 = z0_pred.to(dtype=torch.float64)
            zt_64 = z_t.to(dtype=torch.float64)
            if bridge:
                e_64 = endpoints[band]
                m_t_view = _expand(m_t, zt_64.ndim)
                m_p_view = _expand(m_p, zt_64.ndim)
                # [DESIGN §4 v1.0g] degenerate start m_t∈{0,1} (within
                # _M_ENDPOINT_TOL): ε̂ not invertible → posterior-mean fallback
                # below; the σ_t clamp guards non-degenerate steps only.
                degenerate = (m_t <= _M_ENDPOINT_TOL) | (m_t >= 1.0 - _M_ENDPOINT_TOL)
                sigma_t = (
                    self.config.sigma_bridge * torch.sqrt((2.0 * m_t * (1.0 - m_t)).clamp_min(0.0))
                ).clamp_min(_SIGMA_DIV_EPS)
                sigma_p = self.config.sigma_bridge * torch.sqrt((2.0 * m_p * (1.0 - m_p)).clamp_min(0.0))
                mu_t = m_t_view * e_64 + (1.0 - m_t_view) * z0_64  # μ_j=m_j·e+(1−m_j)ẑ0
                mu_p = m_p_view * e_64 + (1.0 - m_p_view) * z0_64
                eps_hat = (zt_64 - mu_t) / _expand(sigma_t, zt_64.ndim)
                z_prev = mu_p + _expand(sigma_p, zt_64.ndim) * eps_hat
                # posterior-mean fallback, same closed form and t_prev==t
                # guard as posterior(): ((m_t−m_p)/m_t)·ẑ0+(m_p/m_t)·z_t.
                same = t_prev == t
                safe_m_t = torch.where(same, torch.ones_like(m_t), m_t)
                ratio = m_p / safe_m_t  # posterior coefficient m_s/m_t ([审计] §3.2)
                p_mean = (1.0 - _expand(ratio, zt_64.ndim)) * z0_64 + _expand(ratio, zt_64.ndim) * zt_64
                p_mean = torch.where(_expand(same, zt_64.ndim), zt_64, p_mean)
                z_prev = torch.where(_expand(degenerate, zt_64.ndim), p_mean, z_prev)
            else:
                a_t = self._alpha_hat_values(m_t)
                a_p = self._alpha_hat_values(m_p)
                eps_den = torch.sqrt((1.0 - a_t).clamp_min(_SIGMA_DIV_EPS ** 2))
                eps_hat = (zt_64 - _expand(torch.sqrt(a_t), zt_64.ndim) * z0_64) / _expand(eps_den, zt_64.ndim)
                z_prev = _expand(torch.sqrt(a_p), zt_64.ndim) * z0_64 + _expand(
                    torch.sqrt((1.0 - a_p).clamp_min(0.0)), zt_64.ndim
                ) * eps_hat
            pinned = _expand(t_prev == 0, zt_64.ndim)
            z_prev = torch.where(pinned, z0_64, z_prev)
            out[band] = z_prev.to(dtype=z_t.dtype)
        return out

    def forward_transition(
        self,
        z_s_bands: dict[str, torch.Tensor],
        t_s: torch.Tensor,
        t: torch.Tensor,
        endpoint_bands: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Closed-form forward transition q(z_t | z_s, z0, e) → (mean, var).

        [审计] §3.2 / DESIGN §4 v1.0g (σ-aligned), s<t:
        mean = μ_t + ((1−m_t)/(1−m_s))·(z_s−μ_s),
        var  = 2σ²·(m_t−m_s)(1−m_t)/(1−m_s),  μ_j=(1−m_j)z0+m_j·e.
        z0 cancels analytically (bridge Markov property in m-time):
        mean = gain·z_s + (1−gain)·e with gain=(1−m_t)/(1−m_s), so only the
        endpoint coordinate is consumed and the endpoint semantics are
        identical to q_marginal ('zeros' → e=0; 'ct_minus_mean' requires
        validated band-coordinate endpoint_bands).  Training-side Markov
        consistency checks (z_s~q(z_s|z0) → transition → empirical marginal
        vs q_marginal) rely on this closed form.
        t_s==t → variance 0, mean z_s; t_s>t → ValueError.  vp mode raises
        NotImplementedError (C1 / PRD v1.0.2 non-equivalence).  Returns
        per-band tensors in the dtype of z_s_bands.
        """
        if self.config.forward_mode == "vp_bandwise":
            raise NotImplementedError(
                "forward_transition is bridge-only: q(z_t|z_s) of the vp_bandwise "
                "arm is out of scope (C1 / PRD v1.0.2)"
            )
        t_s = self._validate_timesteps(t_s, "t_s")
        t = self._validate_timesteps(t, "t")
        if t_s.shape != t.shape:
            raise ValueError(f"t_s {tuple(t_s.shape)} and t {tuple(t.shape)} must match")
        if bool((t_s > t).any()):
            raise ValueError("forward_transition requires t_s <= t for every sample")
        self._validate_band_dict(z_s_bands, "z_s_bands", batch_size=t.shape[0])
        endpoints = self._resolve_endpoints(z_s_bands, endpoint_bands)
        same = t_s == t
        sigma2 = self.config.sigma_bridge ** 2
        mean_out: dict[str, torch.Tensor] = {}
        var_out: dict[str, torch.Tensor] = {}
        for band, z_s in z_s_bands.items():
            group = self._group_of_band(band)
            m_seq = self.m_sequence(group).to(device=t.device)
            m_s, m_t = m_seq[t_s], m_seq[t]
            # t_s==T (m_s==1) forces t==t_s, so the same-guard keeps every
            # non-identity division by (1−m_s) strictly positive here.
            safe_one_minus = torch.where(same, torch.ones_like(m_s), 1.0 - m_s)
            gain = (1.0 - m_t) / safe_one_minus            # [审计] §3.2
            forward_mass = (m_t - m_s) / safe_one_minus    # = 1 − gain
            var = 2.0 * sigma2 * forward_mass * (1.0 - m_t)
            var = torch.where(same, torch.zeros_like(var), var.clamp_min(0.0))
            zs_64 = z_s.to(dtype=torch.float64)
            mean = _expand(gain, zs_64.ndim) * zs_64 + _expand(forward_mass, zs_64.ndim) * endpoints[band]
            mean = torch.where(_expand(same, zs_64.ndim), zs_64, mean)
            mean_out[band] = mean.to(dtype=z_s.dtype)
            var_out[band] = var.to(dtype=z_s.dtype)
        return mean_out, var_out

    def alpha_hat(self, group: str, t: torch.Tensor) -> torch.Tensor:
        """Cumulative ᾱ_b(t) for a group at timesteps t ([B]).

        vp: ᾱ_b(t)=σ(λ_max+(λ_min−λ_max)·m_b(t/T)) ([计划] §3.6).  bridge:
        returns 1−m_t — a *diagnostic* bridge SNR proxy, NOT a VP ᾱ
        (semantic difference documented per DESIGN §4).  Computed in float64;
        cast to t.dtype when t is floating, else to the default dtype.
        """
        t = self._validate_timesteps(t, "t")
        m_t = self.m_sequence(group).to(device=t.device)[t]
        if self.config.forward_mode == "vp_bandwise":
            values = self._alpha_hat_values(m_t)
        else:
            values = 1.0 - m_t
        target_dtype = t.dtype if t.dtype.is_floating_point else torch.get_default_dtype()
        return values.to(dtype=target_dtype)
