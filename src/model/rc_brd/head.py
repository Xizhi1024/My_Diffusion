"""RC-BRD bounded specialist head (FR-4, DESIGN §5; v1.0g AUDIT 5 fixes).

Reverse-side specialist that adds a bounded, zero-initialised correction on
top of the full-rank base prediction (DESIGN §9.2; AUDIT 5 §6 minimal-arch
corrected form):

    z0_hat^b = z0_hat_base^b + c^b * a_b * d_max * tanh(Δ_b)   # [计划] §3.7 + §6

- per-band gate a_b = a_max * sigmoid(h_b) with a_max frozen up front
  ([审计] §5/§6), so |a_b| <= a_max element-wise;
- v1.0g tanh magnitude bound (AUDIT 5 §3.3 "bounded specialist 名过强"):
  the correction is absolutely bounded, |z0_hat - base| <= |c|·a_max·d_max
  element-wise, because tanh ∈ (−1, 1). Zero-initialised Δ keeps the exact
  init identity (tanh(0) = 0; [计划] §3.7 zero-init);
- v1.0g CT provenance (AUDIT 5 GT-leak X item): condition_feat must arrive
  as a CTFeatureToken issued by the model's internal rc_brd_ct_proj via
  head.issue_ct_token; bare tensors are rejected by default (shape checks
  cannot prove provenance). allow_unverified_tokens=True is an explicit
  calibration/testing escape hatch that warns once.
- background-energy penalty λ_bg·‖(1-M_soft)⊙Δz_special‖₁ ([审计] §5/§8) is
  exposed as background_delta_energy; it is a training-time loss option and is
  not wired into any trainer by default (FR-4.5).

Import policy (DESIGN §1 / batch 1B): this module must not import contract
(c_effective arrives as a plain band-name -> Tensor dict). The band layout is
imported from wavelet as the single source (AUDIT_4 P2; the parallel-batch
local copy was retired once batch 1 landed).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelet import BAND_NAMES as _BAND_NAMES  # DESIGN §2, single source (AUDIT_4 P2)
_BAND_LEVEL: dict[str, int] = {  # Haar level of each band (two-level decomposition)
    "LL2": 2, "LH2": 2, "HL2": 2, "HH2": 2, "LH1": 1, "HL1": 1, "HH1": 1,
}
_VALID_SCALES = (1, 2)  # last two decoder scales ([审计] §6: specialist 只加在最后两个 scales)
_HIDDEN_CHANNELS = 16   # small per-band trunk: specialist 参数增量控制在 5–10% ([审计] §6)
_C_RANGE_EPS = 1e-6     # c~ ∈ [0,1] ([审计] §3.3); ε 只容忍浮点误差，域外 fail-closed


@dataclass(frozen=True)
class SpecialistConfig:
    """Configuration of the bounded specialist head (DESIGN §5, verbatim fields).

    a_max:      frozen upper bound of the per-band gate (|a_b| <= a_max).
    d_max:      v1.0g tanh magnitude bound; the absolute correction bound is
                a_max·d_max (AUDIT 5 §3.3/§6; DESIGN §5).
    gate_init:  initial value of sigmoid(h_b) (small non-zero; init gate =
                a_max * gate_init).
    channels:   channel count of the CT condition feature stream feeding the
                specialist (the last two decoder scales' channels).
    apply_scales: decoder scales the specialist attaches to; bands whose Haar
                level is not listed pass through unchanged.
    """

    a_max: float = 0.25            # 预冻结上界
    d_max: float = 0.10            # v1.0g tanh 幅度上界（绝对 bound = a_max·d_max）
    gate_init: float = 0.1         # σ(h) 初始值（小非零）
    channels: int = 64             # 最后两个 decoder scales 的通道
    apply_scales: tuple[int, ...] = (1, 2)   # 最后两个 decoder scale 索引


def _checked_config(config: SpecialistConfig) -> None:
    """Fail-closed config validation (unknown/invalid values -> ValueError)."""
    if not isinstance(config, SpecialistConfig):
        raise ValueError("config must be a SpecialistConfig instance")
    scales = tuple(config.apply_scales)
    if not scales:
        raise ValueError("apply_scales must be non-empty")
    if len(set(scales)) != len(scales):
        raise ValueError(f"apply_scales must not repeat entries, got {scales}")
    if any(isinstance(s, bool) or not isinstance(s, int) or s not in _VALID_SCALES for s in scales):
        raise ValueError(
            f"apply_scales entries must be integers within {_VALID_SCALES} "
            f"(last two decoder scales), got {scales}")
    if not (math.isfinite(config.a_max) and config.a_max > 0.0):
        raise ValueError(f"a_max must be finite and > 0, got {config.a_max!r}")
    if not (math.isfinite(config.d_max) and config.d_max > 0.0):
        raise ValueError(f"d_max must be finite and > 0, got {config.d_max!r}")
    if not (0.0 < config.gate_init < 1.0):
        raise ValueError(f"gate_init must lie in (0,1) (sigmoid initial value), got {config.gate_init!r}")
    if isinstance(config.channels, bool) or not isinstance(config.channels, int)             or config.channels < 1:
        raise ValueError(f"channels must be a positive int, got {config.channels!r}")


class CTFeatureToken:
    """v1.0g provenance wrapper (DESIGN §5; AUDIT 5 GT-leak X item).

    Wraps the CT condition feature tensor together with an opaque session
    token issued by the model's internal rc_brd_ct_proj pipeline (via
    BoundedSpecialistHead.issue_ct_token). Shape checks alone cannot prove a
    tensor's origin, so the head refuses bare tensors by default: a GT-derived
    tensor of exactly the right shape still fails the provenance check.
    The token is compared by object identity against the issuing head's
    session secret, so a token from another head instance is rejected too.
    """

    __slots__ = ("tensor", "token")

    def __init__(self, tensor: torch.Tensor, token: object) -> None:
        if not isinstance(tensor, torch.Tensor) or tensor.dim() != 4:
            shape = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else type(tensor)
            raise ValueError(f"CTFeatureToken tensor must be [B,C,h,w], got {shape}")
        if token is None:
            raise ValueError("CTFeatureToken token must be an opaque session object, not None")
        object.__setattr__(self, "tensor", tensor)
        object.__setattr__(self, "token", token)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("CTFeatureToken is immutable (provenance wrapper)")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("CTFeatureToken is immutable (provenance wrapper)")


class _BandDeltaHead(nn.Module):
    """Per-band Δ head: Conv1x1 → SiLU → zero-init Conv3x3 ([计划] §3.7).

    Output channels equal the band channels (=1 for [B,1,h,w] Haar bands).
    The final conv starts at exactly zero so the specialist is an exact
    identity contribution at init.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = nn.SiLU()
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_conv(self.act(self.in_proj(x)))


class BoundedSpecialistHead(nn.Module):
    """Bounded per-band specialist head: ẑ0 = ẑ0_base + c·a·d_max·tanh(Δ) ([计划] §3.7 + AUDIT 5 §6).

    Every band gets an independent scalar gate a_b = a_max·σ(h_b) and an
    independent zero-initialised Δ head; bands whose Haar level is not in
    config.apply_scales pass through untouched. head does not know about
    band *groups*: group-level c must be expanded with
    contract.expand_group_to_bands before calling forward.

    v1.0g: the increment is absolutely bounded (|output−base| ≤ |c|·a_max·d_max)
    via the tanh magnitude cap, and condition_feat must be a CTFeatureToken
    issued by this head's session (issue_ct_token). allow_unverified_tokens
    (ctor kwarg, default False) is the explicit calibration/testing escape
    hatch for bare tensors and warns once when used.
    """

    def __init__(self, config: SpecialistConfig, *,
                 allow_unverified_tokens: bool = False):
        super().__init__()
        _checked_config(config)
        self.config = config
        self.allow_unverified_tokens = bool(allow_unverified_tokens)
        self._unverified_warned = False
        # Opaque per-instance session secret; identity-compared in forward so
        # only tensors wrapped through this head's issue_ct_token are trusted.
        self._ct_session_token = object()
        gate_logit = math.log(config.gate_init / (1.0 - config.gate_init))
        self._gates = nn.ParameterDict()
        self._deltas = nn.ModuleDict()
        for band in _BAND_NAMES:
            if _BAND_LEVEL[band] in set(config.apply_scales):
                self._gates[band] = nn.Parameter(torch.tensor(float(gate_logit)))
                self._deltas[band] = _BandDeltaHead(
                    1 + config.channels, _HIDDEN_CHANNELS, out_channels=1)

    def issue_ct_token(self, tensor: torch.Tensor) -> CTFeatureToken:
        """Wrap a CT feature tensor from the model's rc_brd_ct_proj (v1.0g).

        The model-side integration calls this right after its internal
        ct_proj projection; the returned token carries this head's session
        secret and is the only condition_feat form accepted by default.
        """
        return CTFeatureToken(tensor, self._ct_session_token)

    # ---- input validation helpers (fail-closed, DESIGN §5) ----

    def _checked_bands(self, base_z0_bands) -> dict[str, torch.Tensor]:
        if not isinstance(base_z0_bands, dict) or not base_z0_bands:
            raise ValueError(
                "base_z0_bands must be a non-empty dict of band name -> [B,1,h,w] tensor")
        unknown = sorted(set(base_z0_bands) - set(_BAND_NAMES))
        if unknown:
            raise ValueError(
                f"unknown band names in base_z0_bands: {unknown}; expected a subset of {_BAND_NAMES}")
        batch_size = None
        for band, tensor in base_z0_bands.items():
            if not isinstance(tensor, torch.Tensor) or tensor.dim() != 4 or tensor.shape[1] != 1:
                shape = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else type(tensor)
                raise ValueError(f"band {band!r} must be a [B,1,h,w] tensor, got {shape}")
            if batch_size is None:
                batch_size = tensor.shape[0]
            elif tensor.shape[0] != batch_size:
                raise ValueError(
                    f"batch dimension mismatch across bands: {batch_size} vs "
                    f"{tensor.shape[0]} at band {band!r}")
        return base_z0_bands

    def _checked_condition_tensor(self, condition_feat) -> torch.Tensor:
        """Unwrap the condition stream with the v1.0g provenance check.

        Accepted: a CTFeatureToken whose token is this head's session secret
        (issued via issue_ct_token by the model's rc_brd_ct_proj pipeline).
        Bare tensors raise ValueError unless the head was constructed with
        allow_unverified_tokens=True (explicit calibration/testing escape
        hatch; warns once per head instance). Shape/channel/resolution checks
        are kept but are *secondary* — provenance cannot come from a shape.
        """
        if isinstance(condition_feat, CTFeatureToken):
            if condition_feat.token is not self._ct_session_token:
                raise ValueError(
                    "CTFeatureToken was not issued by this head's session "
                    "(token identity mismatch); re-issue it via "
                    "head.issue_ct_token on the owning model's ct_proj output")
            return condition_feat.tensor
        if isinstance(condition_feat, torch.Tensor):
            if not self.allow_unverified_tokens:
                raise ValueError(
                    "condition_feat must be a CTFeatureToken issued by the model's "
                    "internal rc_brd_ct_proj (head.issue_ct_token); bare tensors are "
                    "rejected because shape checks cannot prove provenance (v1.0g, "
                    "AUDIT 5 GT-leak X item). Construct the head with "
                    "allow_unverified_tokens=True only for calibration/tests")
            if not self._unverified_warned:
                self._unverified_warned = True
                warnings.warn(
                    "BoundedSpecialistHead accepted a bare condition tensor via "
                    "allow_unverified_tokens=True (calibration/testing escape hatch; "
                    "production configs must keep it disabled)",
                    UserWarning, stacklevel=2)
            return condition_feat
        kind = type(condition_feat).__name__
        raise ValueError(
            f"condition_feat must be None, a CTFeatureToken or a [B,C,h,w] "
            f"tensor, got {kind}")

    def _band_conditions(self, condition_feat, bands) -> dict[str, torch.Tensor]:
        """Validate the CT condition stream and align it to each band scale.

        GT-leak fail-closed (DESIGN §5 v1.0g): condition_feat must be a
        CTFeatureToken from the model's ct_proj (provenance), its channel
        count must equal SpecialistConfig.channels, and its spatial resolution
        must equal the finest processed band's resolution (the CT feature
        stream scale). Full-resolution tensors such as GT masks raise
        ValueError. Coarser (level-2) bands receive a deterministic 2×
        average-pooled copy of the same stream.
        """
        if condition_feat is None:
            return {}
        condition_tensor = self._checked_condition_tensor(condition_feat)
        if condition_tensor.shape[1] != self.config.channels:
            raise ValueError(
                f"condition_feat must have channels={self.config.channels} "
                f"(SpecialistConfig.channels), got {condition_tensor.shape[1]}")
        processed = [band for band in bands if band in self._deltas]
        if not processed:
            return {}
        reference = min(processed, key=lambda band: _BAND_LEVEL[band])
        ref_h, ref_w = bands[reference].shape[-2:]
        if tuple(condition_tensor.shape[-2:]) != (ref_h, ref_w):
            raise ValueError(
                f"condition_feat spatial resolution {tuple(condition_tensor.shape[-2:])} must "
                f"equal the base bands' resolution {(ref_h, ref_w)} (CT feature stream); "
                "full-resolution inputs such as GT masks are rejected (no-GT-leak fail-closed)")
        aligned = {}
        for band in processed:
            height, width = bands[band].shape[-2:]
            if ref_h % height or ref_w % width or ref_h // height != ref_w // width:
                raise ValueError(
                    f"band {band!r} resolution {(height, width)} is not an integer "
                    f"downsampling of the reference {(ref_h, ref_w)}")
            factor = ref_h // height
            aligned[band] = condition_tensor if factor == 1 else F.avg_pool2d(
                condition_tensor, kernel_size=factor)
        return aligned

    def _checked_c(self, c_effective, bands) -> "dict[str, torch.Tensor] | None":
        """Validate band-level c tensors; return band -> [B,1,1,1] float tensors or None."""
        if c_effective is None:
            return None
        if not isinstance(c_effective, dict):
            raise ValueError("c_effective must be None or a dict of band name -> tensor")
        unknown = sorted(set(c_effective) - set(bands))
        if unknown:
            raise ValueError(
                f"c_effective contains unknown band names {unknown}; expected a subset "
                f"of {sorted(bands)}")
        specialist_bands = [band for band in self._deltas if band in bands]
        missing = sorted(set(specialist_bands) - set(c_effective))
        if missing:
            raise ValueError(
                f"c_effective is missing specialist bands {missing}; pass an explicit 0 "
                "tensor to switch a band's specialist off")
        batch_size = next(iter(bands.values())).shape[0]
        prepared = {}
        for band, value in c_effective.items():
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            if tensor.dim() > 1:
                raise ValueError(
                    f"c_effective[{band!r}] must be a scalar or [B] tensor, got shape "
                    f"{tuple(tensor.shape)}")
            if tensor.dim() == 1 and tensor.shape[0] != batch_size:
                raise ValueError(
                    f"c_effective[{band!r}] has length {tensor.shape[0]} but the batch "
                    f"size is {batch_size}")
            values = tensor.to(dtype=torch.float64)
            if not torch.isfinite(values).all():
                raise ValueError(f"c_effective[{band!r}] contains non-finite values")
            if values.min() < -_C_RANGE_EPS or values.max() > 1.0 + _C_RANGE_EPS:
                raise ValueError(
                    f"c_effective[{band!r}] values must lie in [0, 1] (+{_C_RANGE_EPS} float "
                    f"tolerance); got min={values.min().item()}, max={values.max().item()}")
            prepared[band] = values.clamp_(0.0, 1.0).to(dtype=torch.float32).reshape(-1, 1, 1, 1)
        return prepared

    def _delta_dtype(self) -> torch.dtype:
        if not self._deltas:
            return torch.float32
        first = next(iter(self._deltas.values()))
        return first.in_proj.weight.dtype

    def _stack_inputs(self, base: torch.Tensor, cond_band) -> torch.Tensor:
        dtype = self._delta_dtype()
        if cond_band is None:
            cond = torch.zeros(
                base.shape[0], self.config.channels, base.shape[2], base.shape[3],
                device=base.device, dtype=dtype)
        else:
            cond = cond_band.to(device=base.device, dtype=dtype)
        return torch.cat([base.to(dtype=dtype), cond], dim=1)

    # ---- public API (DESIGN §5, signatures verbatim) ----

    def forward(self, base_z0_bands: dict[str, torch.Tensor],
                condition_feat: "CTFeatureToken | torch.Tensor | None",
                c_effective: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor]:
        """Apply the bounded specialist increment band by band ([计划] §3.7).

        Key semantics: base_z0_bands / c_effective / output are all band-name ->
        Tensor dicts (group-level c must be expanded beforehand with
        contract.expand_group_to_bands; the head never sees groups).

        - v1.0g bounded form (AUDIT 5 §3.3/§6): ẑ0 = base + c·a_b·d_max·tanh(Δ_b)
          — the correction is *absolutely* bounded element-wise,
          |output − base| ≤ |c|·a_max·d_max (tanh ∈ (−1,1), gate ≤ a_max);
        - init: Δ = 0 (zero-init last conv) and tanh(0) = 0, so the output
          equals base exactly (bitwise);
        - c_effective=None is treated as c = 1 (base+Δ ceiling form, used for
          calibration);
        - v1.0g provenance: condition_feat must be a CTFeatureToken issued by
          this head (bare tensors → ValueError unless
          allow_unverified_tokens=True, which warns once); see
          _checked_condition_tensor.
        """
        bands = self._checked_bands(base_z0_bands)
        conditions = self._band_conditions(condition_feat, bands)
        c_map = self._checked_c(c_effective, bands)
        d_max = float(self.config.d_max)
        outputs: dict[str, torch.Tensor] = {}
        for band, base in bands.items():
            delta_head = self._deltas[band] if band in self._deltas else None
            if delta_head is None:
                outputs[band] = base  # scale not covered: base stays fully on ([计划] §3.7)
                continue
            feats = self._stack_inputs(base, conditions.get(band))
            delta = delta_head(feats).to(dtype=base.dtype)
            a_b = (self.config.a_max * torch.sigmoid(self._gates[band]))  # a_b = a_max·σ(h_b)
            a_b = a_b.to(dtype=base.dtype)
            if c_map is None or band not in c_map:
                c_b = torch.ones(1, 1, 1, 1, device=base.device, dtype=base.dtype)
            else:
                c_b = c_map[band].to(device=base.device, dtype=base.dtype)
            # ẑ0 = ẑ0_base + c·a·d_max·tanh(Δ) — [计划] §3.7 + AUDIT 5 §6 修正量有界化
            outputs[band] = base + c_b * a_b * d_max * torch.tanh(delta)
        return outputs

    def gate_values(self) -> dict[str, float]:
        """Current per-band gate values a_b = a_max·σ(h_b) ([计划] §3.7; 诊断/telemetry)."""
        values = {}
        for band, gate in self._gates.items():
            sigmoid = torch.sigmoid(gate.detach().to(dtype=torch.float64))
            values[band] = float(self.config.a_max * sigmoid)
        return values


def background_delta_energy(delta_bands: dict[str, torch.Tensor],
                            soft_mask_inv: torch.Tensor) -> torch.Tensor:
    """L1 background energy of the specialist increments (FR-4.5; [审计] §5/§8).

    Returns Σ_b ‖soft_mask_inv ⊙ Δ_b‖₁, i.e. the summed absolute specialist
    energy leaking outside the (soft) lesion support:
    λ_bg·‖(1−M_soft)⊙Δz_special‖₁. soft_mask_inv = 1 − M_soft. Spatial
    alignment of soft_mask_inv with each band's resolution is the caller's
    duty; a resolution mismatch fails closed with ValueError. Optional
    training-time loss only - never wired into the default trainer.
    """
    if not isinstance(delta_bands, dict) or not delta_bands:
        raise ValueError("delta_bands must be a non-empty dict of band name -> tensor")
    if not isinstance(soft_mask_inv, torch.Tensor):
        raise ValueError("soft_mask_inv must be a torch tensor (1 − M_soft)")
    total = None
    for band, delta in delta_bands.items():
        if tuple(soft_mask_inv.shape[-2:]) != tuple(delta.shape[-2:]):
            raise ValueError(
                f"soft_mask_inv resolution {tuple(soft_mask_inv.shape[-2:])} does not match "
                f"band {band!r} resolution {tuple(delta.shape[-2:])}; alignment is the "
                "caller's responsibility and mismatches fail closed")
        term = (soft_mask_inv.to(dtype=delta.dtype) * delta).abs().sum()
        total = term if total is None else total + term
    return total
