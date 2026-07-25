"""Conservative spectral routing components for Haar detail packets."""

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .boundary_reliable import (
    BoundaryReliableFrequencyInjector,
    _split_details,
    _stack_details,
    _total_variation,
)
from .dct_descriptor import SelectedDCTDescriptor
from .h3_native_null_schedule import (
    load_h3_native_null_schedule,
    native_null_routes,
)
from .prior_anchor_schedule import load_prior_anchor_schedule


class BoundedAmplitudeHead(nn.Module):
    """Predict a trust amplitude within fixed limits and start at zero."""

    def __init__(
        self,
        input_features: int,
        hidden_channels: int,
        minimum: float = -0.05,
        maximum: float = 0.10,
    ) -> None:
        super().__init__()
        if not minimum < 0 < maximum:
            raise ValueError("minimum and maximum must satisfy minimum < 0 < maximum")

        self.minimum = minimum
        self.maximum = maximum
        self.features = nn.Sequential(
            nn.Linear(input_features, hidden_channels),
            nn.SiLU(),
        )
        self.final = nn.Linear(hidden_channels, 1)

        neutral = (0.0 - minimum) / (maximum - minimum)
        neutral_logit = math.log(neutral / (1.0 - neutral))
        with torch.no_grad():
            self.final.weight.zero_()
            self.final.bias.fill_(neutral_logit)

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        normalized = torch.sigmoid(self.final(self.features(evidence))).squeeze(-1)
        amplitude = self.minimum + (self.maximum - self.minimum) * normalized
        return amplitude.clamp(min=self.minimum, max=self.maximum)


class ConservativeRouteHead(nn.Module):
    """Route one packet to native, adjacent-shallow, or null."""

    def __init__(
        self,
        input_features: int,
        hidden_channels: int,
        initial_null_probability: float = 0.90,
    ) -> None:
        super().__init__()
        if not 0.5 <= initial_null_probability < 1.0:
            raise ValueError(
                "initial_null_probability must be in the interval [0.5, 1)"
            )

        self.features = nn.Sequential(
            nn.Linear(input_features, hidden_channels),
            nn.SiLU(),
        )
        self.final = nn.Linear(hidden_channels, 3)

        residual_probability = (1.0 - initial_null_probability) / 2.0
        prior = torch.tensor(
            [
                residual_probability,
                residual_probability,
                initial_null_probability,
            ],
            dtype=self.final.bias.dtype,
            device=self.final.bias.device,
        )
        with torch.no_grad():
            self.final.weight.zero_()
            self.final.bias.copy_(prior.log())

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.final(self.features(evidence)), dim=-1)


class NoNullRouteHead(nn.Module):
    """Two-way softmax router without a null sink (native / shallow only)."""

    def __init__(
        self,
        input_features: int,
        hidden_channels: int,
        initial_native_probability: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_native_probability < 1.0:
            raise ValueError("initial_native_probability must be between 0 and 1")

        self.features = nn.Sequential(
            nn.Linear(input_features, hidden_channels),
            nn.SiLU(),
        )
        self.final = nn.Linear(hidden_channels, 2)

        prior = torch.tensor(
            [
                initial_native_probability,
                1.0 - initial_native_probability,
            ],
            dtype=self.final.bias.dtype,
            device=self.final.bias.device,
        )
        with torch.no_grad():
            self.final.weight.zero_()
            self.final.bias.copy_(prior.log())

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.final(self.features(evidence)), dim=-1)


class BoundedActiveLogitDeltaHead(nn.Module):
    """Predict a bounded log-odds correction and start at exact zero."""

    def __init__(
        self,
        input_features: int,
        hidden_channels: int,
        maximum_absolute_delta: float = 2.0,
    ) -> None:
        super().__init__()
        if maximum_absolute_delta <= 0:
            raise ValueError("maximum_absolute_delta must be positive")
        self.maximum_absolute_delta = float(maximum_absolute_delta)
        self.features = nn.Sequential(
            nn.Linear(input_features, hidden_channels),
            nn.SiLU(),
        )
        self.final = nn.Linear(hidden_channels, 1)
        with torch.no_grad():
            self.final.weight.zero_()
            self.final.bias.zero_()

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        raw = self.final(self.features(evidence)).squeeze(-1)
        return self.maximum_absolute_delta * torch.tanh(raw)


class UncertaintyAwareRouteSelector(nn.Module):
    """Abstain to a frozen H3 route schedule when evidence is uncertain.

    This selector is intentionally not wired into the production router while
    H4-v2 is unconfirmed.  It defines the frozen, unit-testable interface that
    H5 may activate only after a formal zero-overlap H4-v2 PASS.
    """

    def __init__(self, confidence_threshold: float) -> None:
        super().__init__()
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.register_buffer(
            "_confidence_threshold",
            torch.tensor(float(confidence_threshold), dtype=torch.float32),
            persistent=True,
        )

    @property
    def confidence_threshold(self) -> float:
        return float(self._confidence_threshold.item())

    def forward(
        self,
        evidence_routes: torch.Tensor,
        evidence_confidence: torch.Tensor,
        h3_fixed_routes: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if evidence_routes.shape != h3_fixed_routes.shape:
            raise ValueError(
                "evidence_routes and h3_fixed_routes must have identical shapes"
            )
        if evidence_routes.ndim < 2:
            raise ValueError("route tensors must include a route dimension")
        expected_confidence_shape = evidence_routes.shape[:-1]
        if evidence_confidence.shape == (*expected_confidence_shape, 1):
            evidence_confidence = evidence_confidence.squeeze(-1)
        if evidence_confidence.shape != expected_confidence_shape:
            raise ValueError(
                "evidence_confidence must match all non-route dimensions"
            )
        if not torch.isfinite(evidence_confidence).all():
            raise ValueError("evidence_confidence must be finite")
        confidence = evidence_confidence.to(
            device=evidence_routes.device,
            dtype=evidence_routes.dtype,
        ).clamp(0.0, 1.0)
        threshold = self._confidence_threshold.to(
            device=evidence_routes.device,
            dtype=evidence_routes.dtype,
        )
        active = confidence >= threshold
        selected = torch.where(
            active.unsqueeze(-1),
            evidence_routes,
            h3_fixed_routes.to(evidence_routes),
        )
        return selected, {
            "router_confidence": confidence,
            "router_active": active.to(evidence_routes.dtype),
            "router_abstained": (~active).to(evidence_routes.dtype),
            "router_active_fraction": active.float().mean(),
        }


class BiasFreeZeroProjection(nn.Module):
    """Zero-initialized projection that stays strictly zero for zero input.

    Unlike ``_ZeroProjection``, this module has **no bias** in any convolution,
    so P(0) = 0 is guaranteed by construction even after training.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden = min(32, max(8, out_channels // 4))
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.SiLU(),
        )
        self.final = nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.final.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.features(x))


class SpectralEvidenceFrequencyRouter(BoundaryReliableFrequencyInjector):
    """Route reliability-gated Haar packets using fixed-width spectral evidence.

    Route policies
    --------------
    ``native_only``    – [1,0,0] frozen, no cross-level routing (pure native).
    ``fixed_prior``    – frozen softmax prior, e.g. [0.05,0.05,0.90].
    ``h3_native_null`` – frozen full-timestep [native,0,null] schedule.
    ``prior_anchored_learned`` – H3 active prior + bounded adaptive correction.
    ``learned``        – 3-way softmax learned from evidence (native/shallow/null).
    ``learned_no_null``– 2-way softmax (native/shallow), no null option.
    ``legacy_off``     – (deprecated) cross_level_enabled=False → implicit [1,0,0]/[1,1,0].
    """

    _ROUTE_POLICIES = frozenset({
        "native_only",
        "fixed_prior",
        "h3_native_null",
        "prior_anchored_learned",
        "learned",
        "learned_no_null",
        "legacy_off",
    })

    evidence_features = 48  # 12(r_dct) + 12(ct_dct) + 12(diff) + 12(scalar)
    dct_features = 36  # when shared across all 3 bands; per-band is 12
    dct_per_band = 12
    scalar_features = 12
    timestep_feature_index = dct_features + 3  # kept for backward compat
    log_snr_feature_index = dct_features + 4

    def __init__(
        self,
        output_channels: Sequence[int] = (256, 256, 128, 64),
        band_scales: Sequence[float] = (0.5, 0.25),
        ct_reliability_floors: Sequence[float] = (0.25, 0.50),
        gabor_orientations: int = 8,
        hidden_channels: int = 32,
        dct_enabled: bool = True,
        gabor_enabled: bool = True,
        cross_level_enabled: bool = True,
        hard_all_null: bool = False,
        initial_null_probability: float = 0.90,
        amplitude_delta_min: float = -0.05,
        amplitude_delta_max: float = 0.10,
        route_policy: str = "learned",
        fixed_prior: Sequence[float] = (0.05, 0.05, 0.90),
        native_warmup_epochs: int = 0,
        routing_ramp_epochs: int = 0,
        ct_support_enabled: bool = False,
        ct_support_band: str = "l2_hh",
        ct_support_direction: str = "-",
        ct_support_only: bool = False,
        uncertainty_aware_router_enabled: bool = False,
        uncertainty_aware_confidence_threshold: Optional[float] = None,
        h3_schedule_path: Optional[str] = None,
        h3_schedule_sha256: Optional[str] = None,
        h3_schedule_source: str = "formal_h3_v2",
        h3_repository_root: Optional[str] = None,
        h3_allow_unverified_preview_lineage: bool = False,
        h3_num_train_timesteps: int = 1000,
        prior_warmup_epochs: int = 10,
        prior_active_ramp_epochs: int = 10,
        prior_destination_warmup_epochs: int = 30,
        prior_destination_ramp_epochs: int = 10,
        prior_anchor_decay_end_epoch: int = 100,
        prior_anchor_final_scale: float = 0.10,
        active_logit_delta_max: float = 2.0,
        initial_destination_native_probability: float = 0.95,
        **base_kwargs,
    ) -> None:
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if route_policy not in self._ROUTE_POLICIES:
            raise ValueError(
                f"route_policy must be one of {sorted(self._ROUTE_POLICIES)}, "
                f"got {route_policy!r}"
            )
        if len(fixed_prior) not in (2, 3):
            raise ValueError(
                f"fixed_prior must have 2 or 3 entries, got {len(fixed_prior)}"
            )
        prior_sum = sum(fixed_prior)
        if abs(prior_sum - 1.0) > 1e-6:
            raise ValueError(
                f"fixed_prior must sum to 1.0, got {prior_sum}"
            )
        if native_warmup_epochs < 0 or routing_ramp_epochs < 0:
            raise ValueError("routing warmup and ramp epochs must be non-negative")
        if min(
            prior_warmup_epochs,
            prior_active_ramp_epochs,
            prior_destination_warmup_epochs,
            prior_destination_ramp_epochs,
            prior_anchor_decay_end_epoch,
        ) < 0:
            raise ValueError("prior-anchored phase epochs must be non-negative")
        if prior_active_ramp_epochs == 0 or prior_destination_ramp_epochs == 0:
            raise ValueError("prior-anchored ramp epochs must be positive")
        if (
            prior_destination_warmup_epochs
            < prior_warmup_epochs + prior_active_ramp_epochs
        ):
            raise ValueError(
                "prior_destination_warmup_epochs must not precede the end of "
                "the active-correction ramp"
            )
        if prior_anchor_decay_end_epoch <= prior_warmup_epochs:
            raise ValueError(
                "prior_anchor_decay_end_epoch must be after prior_warmup_epochs"
            )
        if not 0.0 < prior_anchor_final_scale <= 1.0:
            raise ValueError("prior_anchor_final_scale must be in (0, 1]")
        if active_logit_delta_max <= 0:
            raise ValueError("active_logit_delta_max must be positive")
        if not 0.5 < initial_destination_native_probability < 1.0:
            raise ValueError(
                "initial_destination_native_probability must be in (0.5, 1)"
            )
        if ct_support_band not in {"l2_lh", "l2_hl", "l2_hh"}:
            raise ValueError("ct_support_band must be an L2 Haar detail band")
        if ct_support_direction not in {"+", "-"}:
            raise ValueError("ct_support_direction must be '+' or '-'")
        base_kwargs.pop("use_directional_reliability", None)
        base_kwargs.pop("use_gabor_agreement", None)
        if ct_support_only:
            # In formal H1 mode CT is a spatial support field only.  It must
            # not enter residual/CT agreement amplitudes from the base class.
            base_kwargs["use_ct_reliability"] = False
        super().__init__(
            output_channels=output_channels,
            band_scales=band_scales,
            ct_reliability_floors=ct_reliability_floors,
            gabor_orientations=gabor_orientations,
            use_directional_reliability=False,
            use_gabor_agreement=False,
            **base_kwargs,
        )
        self.dct_enabled = bool(dct_enabled)
        self.gabor_enabled = bool(gabor_enabled)
        self.cross_level_enabled = bool(cross_level_enabled)
        self.hard_all_null = bool(hard_all_null)
        self.route_policy = route_policy
        self.fixed_prior = tuple(float(p) for p in fixed_prior)
        self.native_warmup_epochs = int(native_warmup_epochs)
        self.routing_ramp_epochs = int(routing_ramp_epochs)
        self.ct_support_enabled = bool(ct_support_enabled)
        self.ct_support_band = str(ct_support_band)
        self.ct_support_direction = str(ct_support_direction)
        self.ct_support_only = bool(ct_support_only)
        self.h3_schedule_path = h3_schedule_path
        self.h3_schedule_sha256 = h3_schedule_sha256
        self.h3_schedule_source = str(h3_schedule_source)
        self.h3_repository_root = h3_repository_root
        self.h3_allow_unverified_preview_lineage = bool(
            h3_allow_unverified_preview_lineage
        )
        self.prior_warmup_epochs = int(prior_warmup_epochs)
        self.prior_active_ramp_epochs = int(prior_active_ramp_epochs)
        self.prior_destination_warmup_epochs = int(
            prior_destination_warmup_epochs
        )
        self.prior_destination_ramp_epochs = int(prior_destination_ramp_epochs)
        self.prior_anchor_decay_end_epoch = int(prior_anchor_decay_end_epoch)
        self.prior_anchor_final_scale = float(prior_anchor_final_scale)
        self.active_logit_delta_max = float(active_logit_delta_max)
        self.initial_destination_native_probability = float(
            initial_destination_native_probability
        )

        # Resolve effective policy when legacy flags are used
        effective_policy = self._resolve_policy()
        self._effective_policy = effective_policy
        self._num_routes = 2 if effective_policy == "learned_no_null" else 3
        initial_progress = (
            0.0
            if effective_policy in {"learned", "learned_no_null"}
            and self.native_warmup_epochs > 0
            else 1.0
        )
        self.register_buffer(
            "_routing_progress",
            torch.tensor(initial_progress, dtype=torch.float32),
            persistent=False,
        )

        self.dct_descriptor = (
            SelectedDCTDescriptor(pooled_size=8, selected_frequencies=12)
            if self.dct_enabled
            else None
        )
        self.amplitude_heads = nn.ModuleList(
            BoundedAmplitudeHead(
                self.evidence_features,
                hidden_channels,
                minimum=amplitude_delta_min,
                maximum=amplitude_delta_max,
            )
            for _ in range(2)
        )
        self.route_heads = nn.ModuleList(
            ConservativeRouteHead(
                self.evidence_features,
                hidden_channels,
                initial_null_probability=initial_null_probability,
            )
            for _ in range(2)
        )
        self.no_null_route_heads = nn.ModuleList(
            NoNullRouteHead(
                self.evidence_features,
                hidden_channels,
                initial_native_probability=self.fixed_prior[0]
                / (self.fixed_prior[0] + self.fixed_prior[1])
                if (self.fixed_prior[0] + self.fixed_prior[1]) > 0
                else 0.5,
            )
            for _ in range(2)
        )
        self.prior_active_heads = nn.ModuleList()
        self.prior_destination_heads = nn.ModuleList()
        if effective_policy == "prior_anchored_learned":
            self.prior_active_heads.extend(
                BoundedActiveLogitDeltaHead(
                    self.evidence_features,
                    hidden_channels,
                    maximum_absolute_delta=self.active_logit_delta_max,
                )
                for _ in range(2)
            )
            self.prior_destination_heads.extend(
                NoNullRouteHead(
                    self.evidence_features,
                    hidden_channels,
                    initial_native_probability=(
                        self.initial_destination_native_probability
                    ),
                )
                for _ in range(2)
            )

        # Replace base-class _ZeroProjection heads with bias-free variants.
        # The base class stores them as self.projection_heads[0..2]; we rebuild.
        self.projection_heads = nn.ModuleList([
            BiasFreeZeroProjection(3, self.output_channels[1]),
            BiasFreeZeroProjection(3, self.output_channels[2]),
            BiasFreeZeroProjection(1, self.output_channels[3]),
        ])
        self.l2_to_l1_projection = BiasFreeZeroProjection(1, self.output_channels[2])

        # Build fixed-route buffers for non-learned policies
        self.register_buffer(
            "_fixed_routes_l2",
            torch.tensor(self._fixed_route_values(), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_fixed_routes_l1",
            torch.tensor(self._fixed_route_values(), dtype=torch.float32),
            persistent=False,
        )
        self._h3_schedule_metadata: Optional[Dict[str, object]] = None
        if effective_policy in {"h3_native_null", "prior_anchored_learned"}:
            if not h3_schedule_path or not h3_schedule_sha256:
                raise ValueError(
                    f"route_policy={effective_policy!r} requires both "
                    "h3_schedule_path and h3_schedule_sha256"
                )
            if effective_policy == "h3_native_null":
                active_mass, schedule_metadata = load_h3_native_null_schedule(
                    h3_schedule_path,
                    expected_file_sha256=h3_schedule_sha256,
                    expected_num_train_timesteps=int(h3_num_train_timesteps),
                )
            else:
                active_mass, schedule_metadata = load_prior_anchor_schedule(
                    h3_schedule_path,
                    schedule_source=self.h3_schedule_source,
                    expected_file_sha256=h3_schedule_sha256,
                    expected_num_train_timesteps=int(h3_num_train_timesteps),
                    repository_root=self.h3_repository_root,
                    allow_unverified_preview_lineage=(
                        self.h3_allow_unverified_preview_lineage
                    ),
                )
            self.register_buffer(
                "_h3_native_active_mass",
                active_mass,
                persistent=True,
            )
            self._h3_schedule_metadata = dict(schedule_metadata)
            frozen_modules = (
                self.amplitude_heads,
                self.route_heads,
                self.no_null_route_heads,
            )
            # The fixed policy freezes every evidence head.  The anchored
            # policy keeps only its two explicitly separated heads trainable.
            for module in frozen_modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        elif h3_schedule_path is not None or h3_schedule_sha256 is not None:
            raise ValueError(
                "h3_schedule_path/h3_schedule_sha256 are only valid with "
                "route_policy='h3_native_null' or "
                "'prior_anchored_learned'"
            )

        if effective_policy == "prior_anchored_learned":
            self.register_buffer(
                "_prior_active_progress",
                torch.tensor(0.0, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "_prior_destination_progress",
                torch.tensor(0.0, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "_prior_anchor_scale",
                torch.tensor(1.0, dtype=torch.float32),
                persistent=True,
            )

        # Uncertainty-aware route selector (H4-v2 interface).  OFF by default;
        # when enabled it abstains element-wise to a frozen fixed-policy route
        # whenever the externally-supplied per-(level, band) confidence is below
        # the frozen threshold.  Confidence is NEVER fabricated inside forward:
        # it must come from a leakage-free context-aware inference path (explicit
        # patient_id/slice_id adjacent-slice provider + frozen population stats),
        # supplied via the ``router_confidence`` kwarg.  Without that path the
        # selector stays disabled, so production behaviour is unchanged.
        self.uncertainty_aware_router_enabled = bool(
            uncertainty_aware_router_enabled
        )
        self._uncertainty_aware_selector: Optional[UncertaintyAwareRouteSelector] = None
        if self.uncertainty_aware_router_enabled:
            if effective_policy in {"h3_native_null", "prior_anchored_learned"}:
                raise ValueError(
                    f"{effective_policy} cannot be combined with the "
                    "uncertainty-aware H4-v2 selector"
                )
            if self._effective_policy not in {"learned", "learned_no_null"}:
                raise ValueError(
                    "uncertainty_aware_router_enabled requires a learned route "
                    "policy (got "
                    f"{self._effective_policy!r}); a fixed policy has nothing "
                    "to abstain from"
                )
            if uncertainty_aware_confidence_threshold is None:
                raise ValueError(
                    "uncertainty_aware_router_enabled requires "
                    "uncertainty_aware_confidence_threshold in [0, 1]"
                )
            self._uncertainty_aware_selector = UncertaintyAwareRouteSelector(
                confidence_threshold=float(uncertainty_aware_confidence_threshold)
            )
            # Frozen-by-construction abstention target: the configured fixed
            # policy route vector, shared across both levels.  This is the
            # router-level analog of "abstain to the H3 fixed schedule"; it is
            # deterministic and independent of any data, mask, or PET target.
            self.register_buffer(
                "_ua_fallback_route",
                torch.tensor(self._fixed_route_values(), dtype=torch.float32),
                persistent=False,
            )

    @property
    def routing_progress(self) -> float:
        return float(self._routing_progress.item())

    def set_routing_progress(self, progress: float) -> None:
        """Blend learned routes in gradually from the native-only baseline."""
        value = min(max(float(progress), 0.0), 1.0)
        self._routing_progress.fill_(value)

    def set_training_epoch(self, epoch: int) -> None:
        """Apply the configured native warm-up and learned-route ramp."""
        if self._effective_policy == "prior_anchored_learned":
            epoch = max(int(epoch), 0)
            active_progress = self._phase_progress(
                epoch,
                start_epoch=self.prior_warmup_epochs,
                ramp_epochs=self.prior_active_ramp_epochs,
            )
            destination_progress = self._phase_progress(
                epoch,
                start_epoch=self.prior_destination_warmup_epochs,
                ramp_epochs=self.prior_destination_ramp_epochs,
            )
            decay_fraction = min(
                max(
                    (
                        epoch - self.prior_warmup_epochs
                    )
                    / (
                        self.prior_anchor_decay_end_epoch
                        - self.prior_warmup_epochs
                    ),
                    0.0,
                ),
                1.0,
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * decay_fraction))
            anchor_scale = self.prior_anchor_final_scale + (
                1.0 - self.prior_anchor_final_scale
            ) * cosine
            self._prior_active_progress.fill_(active_progress)
            self._prior_destination_progress.fill_(destination_progress)
            self._prior_anchor_scale.fill_(anchor_scale)
            return
        if self._effective_policy not in {"learned", "learned_no_null"}:
            return
        epoch = max(int(epoch), 0)
        if epoch < self.native_warmup_epochs:
            self.set_routing_progress(0.0)
            return
        if self.routing_ramp_epochs <= 0:
            self.set_routing_progress(1.0)
            return
        progress = (epoch - self.native_warmup_epochs + 1) / self.routing_ramp_epochs
        self.set_routing_progress(progress)

    @staticmethod
    def _phase_progress(
        epoch: int,
        *,
        start_epoch: int,
        ramp_epochs: int,
    ) -> float:
        if epoch < start_epoch:
            return 0.0
        return min(max((epoch - start_epoch + 1) / ramp_epochs, 0.0), 1.0)

    def _blend_with_native(self, routes: torch.Tensor) -> torch.Tensor:
        if self._effective_policy not in {"learned", "learned_no_null"}:
            return routes
        progress_value = self.routing_progress
        if progress_value >= 1.0:
            return routes
        progress = self._routing_progress.to(device=routes.device, dtype=routes.dtype)
        native = torch.zeros_like(routes)
        native[..., 0] = 1.0
        if progress_value <= 0.0:
            return native
        return native + progress * (routes - native)

    def _resolve_policy(self) -> str:
        """Determine the effective route policy from legacy + new configuration."""
        # Hard-null always wins
        if self.hard_all_null:
            return "native_only"  # treated as zero injections downstream
        # Explicit policy takes precedence
        if self.route_policy != "learned":
            return self.route_policy
        # Legacy cross_level_enabled=False maps to legacy_off
        if not self.cross_level_enabled:
            return "legacy_off"
        return "learned"

    def _fixed_route_values(self) -> list[float]:
        """Return the frozen route vector for this policy."""
        policy = self._effective_policy
        if policy == "native_only":
            # 3-way: [1,0,0]
            return [1.0, 0.0, 0.0]
        if policy == "fixed_prior":
            return list(self.fixed_prior)
        if policy == "h3_native_null":
            # Dynamic rows are looked up in _acquire_routes.  This null vector
            # is only a safe construction-time placeholder.
            return [0.0, 0.0, 1.0]
        if policy == "prior_anchored_learned":
            # Dynamic rows are anchored to the H3 table at runtime.
            return [0.0, 0.0, 1.0]
        if policy == "learned_no_null":
            # 2-way: native/shallow split of the non-null budget
            # Use the first two entries of fixed_prior, renormalized
            n, s = self.fixed_prior[:2]
            total = n + s
            if total <= 0:
                return [0.5, 0.5]
            return [n / total, s / total]
        # learned / legacy_off: not used as fixed
        return [1.0, 0.0, 0.0]  # safe default

    def _lookup_prior_active(
        self,
        *,
        level_index: int,
        timestep: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        active_mass = getattr(self, "_h3_native_active_mass", None)
        if active_mass is None:
            raise RuntimeError("prior-anchored H3 schedule buffer is missing")
        if level_index not in (0, 1):
            raise ValueError("level_index must identify L2 (0) or L1 (1)")
        table = active_mass[level_index].transpose(0, 1)
        prior = table.to(device=timestep.device)[timestep.long()]
        return prior.to(device=reference.device, dtype=torch.float32)

    def _retime_prior_evidence(
        self,
        evidence: torch.Tensor,
        timestep: torch.Tensor,
        schedule,
        *,
        level_index: int,
        observed_band_abs_mean: torch.Tensor,
    ) -> torch.Tensor:
        """Update every explicitly timestep-dependent evidence feature."""
        retimed = evidence.clone()
        normalized_timestep, normalized_log_snr = self._time_features(
            timestep,
            schedule,
            level_index,
            evidence,
        )
        retimed[:, :, self.timestep_feature_index] = (
            normalized_timestep[:, None].expand(-1, 3)
        )
        retimed[:, :, self.log_snr_feature_index] = (
            normalized_log_snr[:, None].expand(-1, 3)
        )
        noise = self.noise_reliability(timestep, schedule).to(evidence)
        retimed[:, :, self.dct_features + 5] = noise[:, level_index, None].expand(
            -1, 3
        )
        if self.use_noise_release:
            sigma = schedule.sigma_t[timestep].to(
                device=evidence.device,
                dtype=evidence.dtype,
            )[:, None]
            expected_abs_noise = sigma * math.sqrt(2.0 / math.pi)
            log_ratio = torch.log(
                (observed_band_abs_mean.to(evidence) + 1e-6)
                / (expected_abs_noise + 1e-6)
            )
            calibrated = torch.tanh(log_ratio / 5.0)
        else:
            calibrated = torch.zeros_like(observed_band_abs_mean).to(evidence)
        retimed[:, :, self.dct_features + 6] = calibrated
        return retimed

    def _prior_anchored_route_at(
        self,
        *,
        level_index: int,
        evidence: torch.Tensor,
        reference: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        prior = self._lookup_prior_active(
            level_index=level_index,
            timestep=timestep,
            reference=reference,
        )
        active_progress_value = float(self._prior_active_progress.item())
        destination_progress_value = float(
            self._prior_destination_progress.item()
        )

        # Do not call a closed branch: this gives grad=None, not merely a zero
        # gradient, and prevents optimizer state from drifting before release.
        if active_progress_value <= 0.0:
            effective_delta = torch.zeros_like(prior)
            active = prior
        else:
            raw_delta = self.prior_active_heads[level_index](evidence).float()
            active_progress = self._prior_active_progress.to(
                device=raw_delta.device,
                dtype=torch.float32,
            )
            effective_delta = active_progress * raw_delta
            odds_multiplier = torch.exp(
                effective_delta.clamp(
                    -self.active_logit_delta_max,
                    self.active_logit_delta_max,
                )
            )
            numerator = prior * odds_multiplier
            denominator = (1.0 - prior) + numerator
            active = numerator / denominator.clamp_min(1e-12)
            active = torch.where(prior <= 0.0, torch.zeros_like(active), active)
            active = torch.where(prior >= 1.0, torch.ones_like(active), active)

        if destination_progress_value <= 0.0:
            conditional_shallow = torch.zeros_like(active)
        else:
            learned_destination = self.prior_destination_heads[level_index](
                evidence
            ).float()
            destination_progress = self._prior_destination_progress.to(
                device=learned_destination.device,
                dtype=torch.float32,
            )
            conditional_shallow = (
                destination_progress * learned_destination[..., 1]
            )
        conditional_native = 1.0 - conditional_shallow
        native = active * conditional_native
        shallow = active * conditional_shallow
        routes = torch.stack((native, shallow, 1.0 - active), dim=-1)
        return routes.to(reference), {
            "prior_active": prior,
            "active": active,
            "active_delta": effective_delta,
            "conditional_shallow": conditional_shallow,
            "shallow_probability": shallow,
        }

    def _acquire_routes(
        self,
        level_index: int,
        evidence: torch.Tensor,
        reference: torch.Tensor,
        timestep: torch.Tensor,
        schedule,
        *,
        observed_band_abs_mean: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ):
        """Return (routes, temporal_smoothness) for one level."""
        policy = self._effective_policy
        batch = evidence.shape[0]
        device = evidence.device

        def _result(
            routes: torch.Tensor,
            temporal: torch.Tensor,
            diagnostics: Optional[Dict[str, torch.Tensor]] = None,
        ):
            if return_diagnostics:
                return routes, temporal, diagnostics or {}
            return routes, temporal

        if policy in ("native_only", "fixed_prior"):
            # Fixed routes: expand batch and band dimensions
            fixed = (
                self._fixed_routes_l2
                if level_index == 0
                else self._fixed_routes_l1
            )
            routes = fixed.view(1, 1, -1).expand(batch, 3, self._num_routes).to(device)
            return _result(routes, reference.new_zeros(()))

        if policy == "h3_native_null":
            active_mass = getattr(self, "_h3_native_active_mass", None)
            if active_mass is None:
                raise RuntimeError("H3-v2 schedule buffer is missing")
            routes = native_null_routes(
                active_mass,
                timestep,
                level_index=level_index,
                reference=reference,
            )
            next_timestep = (timestep + 1).clamp_max(
                int(schedule.num_train_timesteps) - 1
            )
            next_routes = native_null_routes(
                active_mass,
                next_timestep,
                level_index=level_index,
                reference=reference,
            )
            temporal_smoothness = (routes - next_routes).abs().mean()
            return _result(routes, temporal_smoothness)

        if policy == "prior_anchored_learned":
            if observed_band_abs_mean is None:
                raise ValueError(
                    "prior-anchored routing requires observed band amplitudes "
                    "for exact timestep retiming"
                )
            routes, current = self._prior_anchored_route_at(
                level_index=level_index,
                evidence=evidence,
                reference=reference,
                timestep=timestep,
            )
            final_timestep = int(schedule.num_train_timesteps) - 1
            next_timestep = (timestep + 1).clamp_max(final_timestep)
            previous_timestep = (timestep - 1).clamp_min(0)
            next_evidence = self._retime_prior_evidence(
                evidence,
                next_timestep,
                schedule,
                level_index=level_index,
                observed_band_abs_mean=observed_band_abs_mean,
            )
            previous_evidence = self._retime_prior_evidence(
                evidence,
                previous_timestep,
                schedule,
                level_index=level_index,
                observed_band_abs_mean=observed_band_abs_mean,
            )
            next_routes, following = self._prior_anchored_route_at(
                level_index=level_index,
                evidence=next_evidence,
                reference=reference,
                timestep=next_timestep,
            )
            _, previous = self._prior_anchored_route_at(
                level_index=level_index,
                evidence=previous_evidence,
                reference=reference,
                timestep=previous_timestep,
            )
            prior_routes = torch.stack(
                (
                    current["prior_active"],
                    torch.zeros_like(current["prior_active"]),
                    1.0 - current["prior_active"],
                ),
                dim=-1,
            ).to(routes)
            next_prior_routes = torch.stack(
                (
                    following["prior_active"],
                    torch.zeros_like(following["prior_active"]),
                    1.0 - following["prior_active"],
                ),
                dim=-1,
            ).to(next_routes)
            temporal_smoothness = (
                (routes - prior_routes) - (next_routes - next_prior_routes)
            ).abs().mean()
            diagnostics = {
                **current,
                "active_next": following["active"],
                "delta_next": following["active_delta"],
                "delta_prev": previous["active_delta"],
                "has_next": timestep < final_timestep,
                "has_prev": timestep > 0,
            }
            return _result(routes, temporal_smoothness, diagnostics)

        if policy == "legacy_off":
            # Old implicit behaviour: L2=[1,0,0], L1=[1,1,0]
            if level_index == 0:
                vec = reference.new_tensor([1.0, 0.0, 0.0])
            else:
                vec = reference.new_tensor([1.0, 1.0, 0.0])
            routes = vec.view(1, 1, 3).expand(batch, 3, 3)
            return _result(routes, reference.new_zeros(()))

        if policy == "learned_no_null":
            # Two-way learnable router: native vs shallow, no null sink
            routes = self._blend_with_native(
                self.no_null_route_heads[level_index](evidence)
            )
            next_timestep = (timestep + 1).clamp_max(
                int(schedule.num_train_timesteps) - 1
            )
            next_evidence = evidence.clone()
            next_evidence[:, :, self.timestep_feature_index], next_evidence[
                :, :, self.log_snr_feature_index
            ] = self._expanded_next_time_features(
                next_timestep, schedule, level_index, reference
            )
            next_routes = self._blend_with_native(
                self.no_null_route_heads[level_index](next_evidence)
            )
            temporal_smoothness = (routes - next_routes).abs().mean()
            return _result(routes, temporal_smoothness)

        # Learned 3-way policy: compute from evidence
        routes = self._blend_with_native(self.route_heads[level_index](evidence))
        next_timestep = (timestep + 1).clamp_max(
            int(schedule.num_train_timesteps) - 1
        )
        next_evidence = evidence.clone()
        next_evidence[:, :, self.timestep_feature_index], next_evidence[
            :, :, self.log_snr_feature_index
        ] = self._expanded_next_time_features(
            next_timestep, schedule, level_index, reference
        )
        next_routes = self._blend_with_native(
            self.route_heads[level_index](next_evidence)
        )
        temporal_smoothness = (routes - next_routes).abs().mean()
        return _result(routes, temporal_smoothness)

    @staticmethod
    def _validate_inputs(current_residual: torch.Tensor, ct: torch.Tensor) -> None:
        if current_residual.shape != ct.shape:
            raise ValueError("current residual and CT must have identical [B,1,H,W] shapes")
        if current_residual.ndim != 4 or current_residual.shape[1] != 1:
            raise ValueError("spectral routing expects [B,1,H,W] residual and CT")
        if current_residual.shape[-2] % 8 or current_residual.shape[-1] % 8:
            raise ValueError("spatial dimensions must be divisible by eight")

    @staticmethod
    def _band_distribution(details: torch.Tensor) -> torch.Tensor:
        energy = details.abs().mean(dim=(-2, -1))
        return energy / energy.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _time_features(
        self,
        timesteps: torch.Tensor,
        schedule,
        level_index: int,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        steps = int(schedule.num_train_timesteps)
        normalized_timestep = (
            timesteps.to(device=reference.device, dtype=torch.float32)
            / max(steps - 1, 1)
        ).clamp(0.0, 1.0)
        m = schedule.m_t[timesteps].to(device=reference.device, dtype=torch.float32)
        sigma = schedule.sigma_t[timesteps].to(
            device=reference.device, dtype=torch.float32
        )
        signal = (1.0 - m) * self.band_scales[level_index].to(
            device=reference.device, dtype=torch.float32
        )
        log_snr = torch.log(
            signal.square().clamp_min(1e-8) / sigma.square().clamp_min(1e-8)
        ).clamp(-20.0, 20.0)
        normalized_log_snr = (log_snr / 20.0).clamp(-1.0, 1.0)
        if not self.use_noise_release:
            normalized_log_snr = torch.zeros_like(normalized_log_snr)
        return normalized_timestep.to(reference), normalized_log_snr.to(reference)

    def _noise_calibrated_band_evidence(
        self,
        residual: torch.Tensor,
        timesteps: torch.Tensor,
        schedule,
    ) -> torch.Tensor:
        """Observed band energy relative to analytic pure bridge noise."""

        if not self.use_noise_release:
            return residual.new_zeros(residual.shape[0], 3)
        observed = residual.abs().mean(dim=(-2, -1))
        sigma = schedule.sigma_t[timesteps].to(
            device=residual.device,
            dtype=residual.dtype,
        )[:, None]
        expected_abs_noise = sigma * math.sqrt(2.0 / math.pi)
        log_ratio = torch.log(
            (observed + 1e-6) / (expected_abs_noise + 1e-6)
        )
        return torch.tanh(log_ratio / 5.0)

    def _dct_evidence(
        self, details: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = details.shape[0]
        if self.dct_descriptor is None:
            return (
                details.new_zeros(batch, 3, 12),
                details.new_zeros(()),
                details.new_zeros(12),
            )
        descriptor, diagnostics = self.dct_descriptor(details)
        return (
            descriptor,
            diagnostics["weight_offset_energy"],
            diagnostics["frequency_weights"],
        )

    def _gabor_evidence(
        self,
        residual_distribution: torch.Tensor,
        dct: torch.Tensor,
        size: tuple[int, int],
        gabor_feat: Optional[torch.Tensor],
        gabor_orientation: Optional[torch.Tensor],
        gabor_anisotropy: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = residual_distribution.shape[0]
        zero = residual_distribution.new_zeros(batch, 3)
        scale_energy = zero
        orientation_energy = zero
        anisotropy_energy = zero
        gabor_haar_agreement = zero
        gabor_dct_agreement = zero
        if not self.gabor_enabled:
            return (
                scale_energy,
                orientation_energy,
                anisotropy_energy,
                gabor_haar_agreement,
                gabor_dct_agreement,
            )

        if gabor_feat is not None:
            if gabor_feat.ndim != 4 or gabor_feat.shape[0] != batch:
                raise ValueError("Gabor scale features must have shape [B,C,H,W]")
            gabor_feat = gabor_feat.to(
                device=residual_distribution.device,
                dtype=residual_distribution.dtype,
            )
            scale = torch.tanh(torch.log1p(gabor_feat.abs().mean(dim=(1, 2, 3))))
            scale_energy = scale[:, None].expand(-1, 3)

        if gabor_anisotropy is not None:
            if gabor_anisotropy.ndim != 4 or gabor_anisotropy.shape[:2] != (batch, 1):
                raise ValueError("Gabor anisotropy must have shape [B,1,H,W]")
            gabor_anisotropy = gabor_anisotropy.to(
                device=residual_distribution.device,
                dtype=residual_distribution.dtype,
            )
            anisotropy = gabor_anisotropy.abs().mean(dim=(1, 2, 3)).clamp(0.0, 1.0)
            anisotropy_energy = anisotropy[:, None].expand(-1, 3)

        if gabor_orientation is not None:
            if (
                gabor_orientation.ndim != 4
                or gabor_orientation.shape[0] != batch
                or gabor_orientation.shape[1] != self.gabor_orientations
            ):
                raise ValueError("Gabor orientation energy has incompatible shape")
            gabor_orientation = gabor_orientation.to(
                device=residual_distribution.device,
                dtype=residual_distribution.dtype,
            )
            oriented = self._orientation_band_energy(gabor_orientation, size)
            orientation_energy = oriented.mean(dim=(-2, -1))
            orientation_energy = orientation_energy / orientation_energy.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-6)
            gabor_haar_agreement = (
                1.0 - (orientation_energy - residual_distribution).abs()
            ).clamp(0.0, 1.0)
            if self.dct_enabled:
                dct_distribution = dct.sum(dim=-1)
                dct_distribution = dct_distribution / dct_distribution.sum(
                    dim=1, keepdim=True
                ).clamp_min(1e-6)
                gabor_dct_agreement = (
                    1.0 - (orientation_energy - dct_distribution).abs()
                ).clamp(0.0, 1.0)

        return (
            scale_energy,
            orientation_energy,
            anisotropy_energy,
            gabor_haar_agreement,
            gabor_dct_agreement,
        )

    def _level_evidence(
        self,
        level_index: int,
        residual_details,
        ct_details,
        base_gate: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        schedule,
        gabor_feat: Optional[torch.Tensor],
        gabor_orientation: Optional[torch.Tensor],
        gabor_anisotropy: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        residual = _stack_details(residual_details)
        ct = _stack_details(ct_details)

        # Per-band DCT: residual, CT, and |residual - CT| difference
        dct_r, dct_offset, dct_weights = self._dct_evidence(residual)
        dct_c, _, _ = self._dct_evidence(ct)
        dct_diff = (dct_r - dct_c).abs()

        residual_distribution = self._band_distribution(residual)
        ct_distribution = self._band_distribution(ct)
        haar_agreement = (1.0 - (residual_distribution - ct_distribution).abs()).clamp(
            0.0, 1.0
        )
        if self.ct_support_only:
            # Preserve the fixed 48-feature layout while ensuring CT cannot
            # determine learned PET band amplitudes/routes.  CT remains
            # available only through the spatial support multiplier below.
            dct_c = torch.zeros_like(dct_c)
            dct_diff = torch.zeros_like(dct_diff)
            ct_distribution = torch.zeros_like(ct_distribution)
            haar_agreement = torch.zeros_like(haar_agreement)
        normalized_timestep, normalized_log_snr = self._time_features(
            timesteps, schedule, level_index, residual
        )
        noise_calibrated_evidence = self._noise_calibrated_band_evidence(
            residual,
            timesteps,
            schedule,
        )
        (
            gabor_global_energy,
            gabor_orientation_energy,
            gabor_anisotropy_energy,
            gabor_haar_agreement,
            gabor_dct_agreement,
        ) = self._gabor_evidence(
            residual_distribution,
            dct_r,
            residual.shape[-2:],
            gabor_feat,
            gabor_orientation,
            gabor_anisotropy,
        )
        scalars = torch.stack(
            (
                residual_distribution,
                ct_distribution,
                haar_agreement,
                normalized_timestep[:, None].expand(-1, 3),
                normalized_log_snr[:, None].expand(-1, 3),
                noise[:, None].expand(-1, 3),
                noise_calibrated_evidence,
                gabor_global_energy,
                gabor_orientation_energy,
                gabor_anisotropy_energy,
                gabor_haar_agreement,
                gabor_dct_agreement,
            ),
            dim=-1,
        )
        # Evidence layout: [B, 3, 48]
        #   dct_r   (12) — residual DCT per band
        #   dct_c   (12) — CT DCT per band
        #   dct_diff(12) — |residual - CT| per band
        #   scalars (12) — band distribution ×2 + agreement + time + noise + ...
        evidence = torch.cat(
            (dct_r, dct_c, dct_diff, scalars),
            dim=-1,
        )
        return evidence, {
            "dct_weight_offset": dct_offset,
            "dct_frequency_weights": dct_weights,
            "gabor_haar_agreement": gabor_haar_agreement,
            "gabor_dct_agreement": gabor_dct_agreement,
            "noise_calibrated_band_evidence": noise_calibrated_evidence,
            "observed_band_abs_mean": residual.abs().mean(dim=(-2, -1)),
        }

    def _ct_support_field(
        self,
        ct_details_l2,
    ) -> torch.Tensor:
        """H1-selected bounded CT spatial support, shared across PET bands."""

        details = _stack_details(ct_details_l2).abs()
        band_index = {"l2_lh": 0, "l2_hl": 1, "l2_hh": 2}[
            self.ct_support_band
        ]
        energy = F.avg_pool2d(
            details[:, band_index : band_index + 1],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        flattened = energy.flatten(1)
        location = flattened.median(dim=1).values[:, None, None, None]
        deviation = (energy - location).abs().flatten(1).median(dim=1).values
        scale = (1.4826 * deviation).clamp_min(1e-6)[:, None, None, None]
        signed = (energy - location) / scale
        if self.ct_support_direction == "-":
            signed = -signed
        return torch.sigmoid(signed).clamp(0.0, 1.0)

    def _zero_result(
        self, current_residual: torch.Tensor
    ) -> tuple[list[torch.Tensor], Dict[str, torch.Tensor]]:
        batch, _, height, width = current_residual.shape
        injections = [
            current_residual.new_zeros(
                batch, self.output_channels[0], height // 8, width // 8
            ),
            current_residual.new_zeros(
                batch, self.output_channels[1], height // 4, width // 4
            ),
            current_residual.new_zeros(
                batch, self.output_channels[2], height // 2, width // 2
            ),
            current_residual.new_zeros(batch, self.output_channels[3], height, width),
        ]
        n_routes = 2 if self._effective_policy == "learned_no_null" else 3
        routes = current_residual.new_zeros(batch, 3, n_routes)
        routes[..., -1] = 1.0 if n_routes == 3 else 0.0
        diagnostics = {
            "gates_l2": current_residual.new_zeros(batch, 3, height // 4, width // 4),
            "gates_l1": current_residual.new_zeros(batch, 3, height // 2, width // 2),
            "routes_l2": routes,
            "routes_l1": routes,
            "noise_reliability": current_residual.new_zeros(batch, 2),
            "gate_tv": current_residual.new_zeros(()),
            "route_temporal_smoothness": current_residual.new_zeros(()),
            "dct_weight_offset": current_residual.new_zeros(()),
            "dct_frequency_weights": current_residual.new_zeros(12),
            "gabor_haar_agreement": current_residual.new_zeros(batch, 2, 3),
            "gabor_dct_agreement": current_residual.new_zeros(batch, 2, 3),
            "route_policy": self._effective_policy,
            # Per-level zero diagnostics (all null → entropy=0, active_mass=0)
            "route_l2_active_mass": current_residual.new_zeros(()),
            "route_l2_entropy": current_residual.new_zeros(()),
            "route_l1_active_mass": current_residual.new_zeros(()),
            "route_l1_entropy": current_residual.new_zeros(()),
        }
        if self._num_routes == 3:
            one = current_residual.new_ones(())
            diagnostics.update({
                "route_l2_null_mean": one,
                "route_l2_null_p10": one,
                "route_l2_null_p50": one,
                "route_l2_null_p90": one,
                "route_l1_null_mean": one,
                "route_l1_null_p10": one,
                "route_l1_null_p50": one,
                "route_l1_null_p90": one,
            })
        # Injection RMS — all zero for hard-null
        for lvl in range(4):
            diagnostics[f"injection/l{lvl}_rms"] = current_residual.new_zeros(())
        if self._effective_policy == "legacy_off":
            diagnostics.update(self._legacy_route_diagnostics(current_residual))
        return injections, diagnostics

    @staticmethod
    def _legacy_route_diagnostics(reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Legacy implicit route display for backward compatibility."""
        batch = reference.shape[0]
        l2 = reference.new_tensor([1.0, 0.0, 0.0]).view(1, 1, 3).expand(batch, 3, 3)
        l1 = reference.new_tensor([1.0, 1.0, 0.0]).view(1, 1, 3).expand(batch, 3, 3)
        return {
            "independent_route_weights_l2": l2,
            "independent_route_weights_l1": l1,
        }

    def _route_diagnostics(
        self,
        routes_l2: torch.Tensor,
        routes_l1: torch.Tensor,
        reference: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-level monitoring statistics for route distributions.

        Returns diagnostics keyed by level so collapse in one layer
        (e.g. L1 null→100%) is visible independently of the other.
        """
        eps = 1e-8
        result: Dict[str, torch.Tensor] = {}
        for level, routes in ((2, routes_l2), (1, routes_l1)):
            flat = routes.flatten(0, 1)  # [B*3, ...]
            prefix = f"route_l{level}"

            null_prob = flat[..., -1] if self._num_routes == 3 else torch.zeros_like(flat[..., 0])
            native_prob = flat[..., 0]
            shallow_prob = flat[..., 1] if flat.shape[-1] >= 2 else torch.zeros_like(native_prob)
            active_mass = (native_prob + shallow_prob).mean()

            # Entropy
            entropy = -(flat * (flat + eps).log()).sum(dim=-1).mean()

            result[f"{prefix}_active_mass"] = active_mass
            result[f"{prefix}_entropy"] = entropy

            if self._num_routes == 3:
                sorted_null = null_prob.sort().values
                n = sorted_null.numel()
                result[f"{prefix}_null_mean"] = null_prob.mean()
                result[f"{prefix}_null_p10"] = (
                    sorted_null[int(0.10 * (n - 1))] if n > 0 else reference.new_zeros(())
                )
                result[f"{prefix}_null_p50"] = (
                    sorted_null[int(0.50 * (n - 1))] if n > 0 else reference.new_zeros(())
                )
                result[f"{prefix}_null_p90"] = (
                    sorted_null[int(0.90 * (n - 1))] if n > 0 else reference.new_zeros(())
                )

        return result

    def _apply_uncertainty_aware_selection(
        self,
        *,
        routes_l2: torch.Tensor,
        routes_l1: torch.Tensor,
        router_confidence: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Apply the H4-v2 selector, or return ``None`` when it is disabled.

        Fail-closed contract: when the selector is enabled, a ``router_confidence``
        tensor of shape ``[B, 2, 3]`` (level × band) MUST be supplied by a
        leakage-free context-aware caller.  The router never invents confidence
        internally, so production forward (which does not supply it) must keep
        the flag off.
        """

        selector = self._uncertainty_aware_selector
        if selector is None:
            return None
        if router_confidence is None:
            raise ValueError(
                "uncertainty_aware_router_enabled=true requires a "
                "router_confidence tensor of shape [B, 2, 3] (level, band); "
                "the router never fabricates confidence internally"
            )
        if router_confidence.ndim != 3:
            raise ValueError("router_confidence must have shape [B, 2, 3]")
        batch = routes_l2.shape[0]
        expected = (batch, 2, 3)
        if tuple(router_confidence.shape) != expected:
            raise ValueError(
                f"router_confidence must have shape {expected}, got "
                f"{tuple(router_confidence.shape)}"
            )
        if not torch.isfinite(router_confidence).all():
            raise ValueError("router_confidence must be finite")

        fallback = self._ua_fallback_route.to(
            device=reference.device, dtype=routes_l2.dtype
        )
        fallback_l2 = fallback.view(1, 1, -1).expand_as(routes_l2)
        fallback_l1 = fallback.view(1, 1, -1).expand_as(routes_l1)

        confidence = router_confidence.to(
            device=reference.device, dtype=routes_l2.dtype
        )
        selected_l2, diag_l2 = selector(routes_l2, confidence[:, 0, :], fallback_l2)
        selected_l1, diag_l1 = selector(routes_l1, confidence[:, 1, :], fallback_l1)

        active = torch.stack(
            (diag_l2["router_active"], diag_l1["router_active"]), dim=1
        )
        abstained = torch.stack(
            (diag_l2["router_abstained"], diag_l1["router_abstained"]), dim=1
        )
        return {
            "router_confidence": confidence,
            "router_active": active,
            "router_abstained": abstained,
            "router_active_fraction": active.float().mean(),
            "router_confidence_threshold": selector._confidence_threshold.to(
                device=reference.device, dtype=routes_l2.dtype
            ),
            "router_selected_routes_l2": selected_l2,
            "router_selected_routes_l1": selected_l1,
            "router_fallback_routes_l2": fallback_l2,
            "router_fallback_routes_l1": fallback_l1,
        }

    def forward(
        self,
        current_residual: torch.Tensor,
        timestep: torch.Tensor,
        schedule,
        ct: torch.Tensor,
        gabor_orientation: Optional[torch.Tensor] = None,
        gabor_anisotropy: Optional[torch.Tensor] = None,
        gabor_feat: Optional[torch.Tensor] = None,
        lesion_score: Optional[torch.Tensor] = None,
        topq_mask: Optional[torch.Tensor] = None,
        router_confidence: Optional[torch.Tensor] = None,
    ) -> tuple[list[torch.Tensor], Dict[str, torch.Tensor]]:
        self._validate_inputs(current_residual, ct)
        if timestep.ndim != 1 or timestep.shape[0] != current_residual.shape[0]:
            raise ValueError("timestep must have shape [B]")
        if self.hard_all_null:
            if self._uncertainty_aware_selector is not None:
                raise ValueError(
                    "uncertainty_aware_router_enabled is incompatible with "
                    "hard_all_null (no learned routes to abstain from)"
                )
            return self._zero_result(current_residual)

        # Training-only masks are accepted for integration compatibility but never routed.
        _ = lesion_score, topq_mask
        _, residual_details1, residual_details2 = self.decompose(current_residual)
        _, ct_details1, ct_details2 = self.decompose(ct)
        noise = self.noise_reliability(timestep, schedule).to(current_residual)
        gates_l2 = self._level_gates(
            0,
            residual_details2,
            ct_details2,
            noise[:, 0],
            None,
            None,
        )
        gates_l1 = self._level_gates(
            1,
            residual_details1,
            ct_details1,
            noise[:, 1],
            None,
            None,
        )
        if self.ct_support_enabled:
            support_l2 = self._ct_support_field(ct_details2)
            support_l1 = F.interpolate(
                support_l2,
                size=gates_l1.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            floor_l2 = self.ct_reliability_floors[0].to(gates_l2)
            floor_l1 = self.ct_reliability_floors[1].to(gates_l1)
            support_l2 = self.apply_reliability_floor(
                support_l2, floor_l2
            )
            support_l1 = self.apply_reliability_floor(
                support_l1, floor_l1
            )
        else:
            support_l2 = gates_l2.new_ones(
                gates_l2.shape[0], 1, *gates_l2.shape[-2:]
            )
            support_l1 = gates_l1.new_ones(
                gates_l1.shape[0], 1, *gates_l1.shape[-2:]
            )
        evidence_l2, evidence_diagnostics_l2 = self._level_evidence(
            0,
            residual_details2,
            ct_details2,
            gates_l2,
            noise[:, 0],
            timestep,
            schedule,
            gabor_feat,
            gabor_orientation,
            gabor_anisotropy,
        )
        evidence_l1, evidence_diagnostics_l1 = self._level_evidence(
            1,
            residual_details1,
            ct_details1,
            gates_l1,
            noise[:, 1],
            timestep,
            schedule,
            gabor_feat,
            gabor_orientation,
            gabor_anisotropy,
        )
        # Apply CT only after constructing learnable amplitude/route evidence:
        # it controls spatial support but cannot determine PET band amplitude.
        gates_l2 = gates_l2 * support_l2
        gates_l1 = gates_l1 * support_l1

        if self._effective_policy in {
            "h3_native_null",
            "prior_anchored_learned",
        }:
            # H3-v2 is an evidence-free fixed schedule.  The route mass is the
            # only learned-mechanism intervention; evidence-dependent amplitude
            # heads are structurally bypassed.
            delta_l2 = evidence_l2.new_zeros(
                evidence_l2.shape[0], evidence_l2.shape[1], 1, 1
            )
            delta_l1 = evidence_l1.new_zeros(
                evidence_l1.shape[0], evidence_l1.shape[1], 1, 1
            )
        else:
            delta_l2 = self.amplitude_heads[0](evidence_l2)[:, :, None, None]
            delta_l1 = self.amplitude_heads[1](evidence_l1)[:, :, None, None]
        amplitude_l2 = (gates_l2 * (1.0 + delta_l2)).clamp(0.0, self.gate_max)
        amplitude_l1 = (gates_l1 * (1.0 + delta_l1)).clamp(0.0, self.gate_max)

        diagnostics: Dict[str, torch.Tensor] = {}
        routes_l2, temporal_l2, prior_diagnostics_l2 = self._acquire_routes(
            0,
            evidence_l2,
            current_residual,
            timestep,
            schedule,
            observed_band_abs_mean=evidence_diagnostics_l2[
                "observed_band_abs_mean"
            ],
            return_diagnostics=True,
        )
        routes_l1, temporal_l1, prior_diagnostics_l1 = self._acquire_routes(
            1,
            evidence_l1,
            current_residual,
            timestep,
            schedule,
            observed_band_abs_mean=evidence_diagnostics_l1[
                "observed_band_abs_mean"
            ],
            return_diagnostics=True,
        )
        temporal_smoothness = 0.5 * (temporal_l2 + temporal_l1)

        # Uncertainty-aware abstention (H4-v2 interface).  Replaces routes in
        # place when enabled; low-confidence (level, band) entries become exactly
        # the frozen fixed-policy fallback, so gradient flows only through the
        # active evidence routes.  Mask / PET target never enter this path.
        selector_diagnostics = self._apply_uncertainty_aware_selection(
            routes_l2=routes_l2,
            routes_l1=routes_l1,
            router_confidence=router_confidence,
            reference=current_residual,
        )
        if selector_diagnostics is not None:
            routes_l2 = selector_diagnostics["router_selected_routes_l2"]
            routes_l1 = selector_diagnostics["router_selected_routes_l1"]
            diagnostics.update(selector_diagnostics)

        # Update route diagnostics
        diagnostics.update(
            self._route_diagnostics(routes_l2, routes_l1, current_residual)
        )
        if self._effective_policy == "prior_anchored_learned":
            diagnostics["route_routing_progress"] = (
                self._prior_destination_progress.to(
                    device=current_residual.device,
                    dtype=current_residual.dtype,
                )
            )
            diagnostics["route_active_progress"] = (
                self._prior_active_progress.to(
                    device=current_residual.device,
                    dtype=current_residual.dtype,
                )
            )
            diagnostics["route_destination_progress"] = (
                self._prior_destination_progress.to(
                    device=current_residual.device,
                    dtype=current_residual.dtype,
                )
            )
            diagnostics["route_anchor_scale"] = self._prior_anchor_scale.to(
                device=current_residual.device,
                dtype=current_residual.dtype,
            )
            for key in (
                "prior_active",
                "active",
                "active_delta",
                "active_next",
                "delta_prev",
                "delta_next",
                "conditional_shallow",
                "shallow_probability",
            ):
                diagnostics[f"route_{key}"] = torch.stack(
                    (
                        prior_diagnostics_l2[key],
                        prior_diagnostics_l1[key],
                    ),
                    dim=1,
                )
            diagnostics["route_has_next"] = prior_diagnostics_l2["has_next"]
            diagnostics["route_has_prev"] = prior_diagnostics_l2["has_prev"]
            active = diagnostics["route_active"].float()
            prior_active = diagnostics["route_prior_active"].float()
            active_next = diagnostics["route_active_next"].float()
            has_next = diagnostics["route_has_next"][:, None, None]
            monotonic_excess = torch.where(
                has_next,
                torch.relu(active_next - active),
                torch.zeros_like(active),
            )
            diagnostics["route_prior_active_mae"] = (
                active - prior_active
            ).abs().mean()
            diagnostics["route_active_delta_abs_mean"] = diagnostics[
                "route_active_delta"
            ].float().abs().mean()
            diagnostics["route_active_delta_abs_max"] = diagnostics[
                "route_active_delta"
            ].float().abs().amax()
            diagnostics["route_monotonic_violation"] = monotonic_excess.mean()
            diagnostics["route_monotonic_violation_fraction"] = (
                monotonic_excess > 1e-6
            ).float().mean()
            diagnostics["route_shallow_mass"] = diagnostics[
                "route_shallow_probability"
            ].float().mean()
        else:
            diagnostics["route_routing_progress"] = self._routing_progress.to(
                device=current_residual.device,
                dtype=current_residual.dtype,
            )

        scale_l2 = self.band_scales[0].to(current_residual)
        scale_l1 = self.band_scales[1].to(current_residual)
        gated_l2 = (
            torch.tanh(_stack_details(residual_details2) / scale_l2) * amplitude_l2
        )
        gated_l1 = (
            torch.tanh(_stack_details(residual_details1) / scale_l1) * amplitude_l1
        )

        # Route application: flexible number of routes
        native_l2 = gated_l2 * routes_l2[..., 0, None, None]
        shallow_l2 = gated_l2 * routes_l2[..., 1, None, None]
        native_l1 = gated_l1 * routes_l1[..., 0, None, None]
        shallow_l1 = gated_l1 * routes_l1[..., 1, None, None]

        batch, _, height, width = current_residual.shape
        l3 = current_residual.new_zeros(
            batch, self.output_channels[0], height // 8, width // 8
        )
        l2 = self.projection_heads[0](native_l2)
        l1 = self.projection_heads[1](native_l1)
        l1 = l1 + self.l2_to_l1_projection(
            self.reconstruct_l0(_split_details(shallow_l2))
        )
        l0 = self.projection_heads[2](
            self.reconstruct_l0(_split_details(shallow_l1))
        )

        # Injection RMS per level — scale-invariant per-element energy
        # so L0/L1/L2 are comparable despite different resolutions and channels.
        injections = [l3, l2, l1, l0]
        injection_norms = {}
        for idx, inj in enumerate(injections):
            rms = inj.float().square().flatten(1).mean(dim=1).sqrt().mean().detach()
            injection_norms[f"injection/l{idx}_rms"] = rms

        diagnostics.update(
            {
                "gates_l2": amplitude_l2,
                "gates_l1": amplitude_l1,
                "ct_support_l2": support_l2,
                "ct_support_l1": support_l1,
                "routes_l2": routes_l2,
                "routes_l1": routes_l1,
                "noise_reliability": noise,
                "gate_tv": 0.5
                * (_total_variation(amplitude_l2) + _total_variation(amplitude_l1)),
                "route_temporal_smoothness": temporal_smoothness,
                "dct_weight_offset": 0.5
                * (
                    evidence_diagnostics_l2["dct_weight_offset"]
                    + evidence_diagnostics_l1["dct_weight_offset"]
                ),
                "dct_frequency_weights": evidence_diagnostics_l2[
                    "dct_frequency_weights"
                ],
                "gabor_haar_agreement": torch.stack(
                    (
                        evidence_diagnostics_l2["gabor_haar_agreement"],
                        evidence_diagnostics_l1["gabor_haar_agreement"],
                    ),
                    dim=1,
                ),
                "gabor_dct_agreement": torch.stack(
                    (
                        evidence_diagnostics_l2["gabor_dct_agreement"],
                        evidence_diagnostics_l1["gabor_dct_agreement"],
                    ),
                    dim=1,
                ),
                "noise_calibrated_band_evidence": torch.stack(
                    (
                        evidence_diagnostics_l2[
                            "noise_calibrated_band_evidence"
                        ],
                        evidence_diagnostics_l1[
                            "noise_calibrated_band_evidence"
                        ],
                    ),
                    dim=1,
                ),
                **injection_norms,
            }
        )
        return [l3, l2, l1, l0], diagnostics

    def _expanded_next_time_features(
        self,
        next_timestep: torch.Tensor,
        schedule,
        level_index: int,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_timestep, normalized_log_snr = self._time_features(
            next_timestep, schedule, level_index, reference
        )
        return (
            normalized_timestep[:, None].expand(-1, 3),
            normalized_log_snr[:, None].expand(-1, 3),
        )
