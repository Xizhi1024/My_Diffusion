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
    _ZeroProjection,
)
from .dct_descriptor import SelectedDCTDescriptor


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
        if not 0.5 < initial_null_probability < 1.0:
            raise ValueError("initial_null_probability must be between 0.5 and 1")

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


class SpectralEvidenceFrequencyRouter(BoundaryReliableFrequencyInjector):
    """Route reliability-gated Haar packets using fixed-width spectral evidence.

    Route policies
    --------------
    ``native_only``    – [1,0,0] frozen, no cross-level routing (pure native).
    ``fixed_prior``    – frozen softmax prior, e.g. [0.05,0.05,0.90].
    ``learned``        – 3-way softmax learned from evidence (native/shallow/null).
    ``learned_no_null``– 2-way softmax (native/shallow), no null option.
    ``legacy_off``     – (deprecated) cross_level_enabled=False → implicit [1,0,0]/[1,1,0].
    """

    _ROUTE_POLICIES = frozenset({
        "native_only",
        "fixed_prior",
        "learned",
        "learned_no_null",
        "legacy_off",
    })

    evidence_features = 48
    dct_features = 36
    scalar_features = 12
    timestep_feature_index = dct_features + 3
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
        base_kwargs.pop("use_directional_reliability", None)
        base_kwargs.pop("use_gabor_agreement", None)
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

        # Resolve effective policy when legacy flags are used
        effective_policy = self._resolve_policy()
        self._effective_policy = effective_policy
        self._num_routes = 2 if effective_policy == "learned_no_null" else 3

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
        self.l2_to_l1_projection = _ZeroProjection(1, self.output_channels[2])

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

    def _acquire_routes(
        self,
        level_index: int,
        evidence: torch.Tensor,
        reference: torch.Tensor,
        timestep: torch.Tensor,
        schedule,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (routes, temporal_smoothness) for one level."""
        policy = self._effective_policy
        batch = evidence.shape[0]
        device = evidence.device

        if policy in ("native_only", "fixed_prior", "learned_no_null"):
            # Fixed routes: expand batch and band dimensions
            fixed = (
                self._fixed_routes_l2
                if level_index == 0
                else self._fixed_routes_l1
            )
            routes = fixed.view(1, 1, -1).expand(batch, 3, self._num_routes).to(device)
            return routes, reference.new_zeros(())

        if policy == "legacy_off":
            # Old implicit behaviour: L2=[1,0,0], L1=[1,1,0]
            if level_index == 0:
                vec = reference.new_tensor([1.0, 0.0, 0.0])
            else:
                vec = reference.new_tensor([1.0, 1.0, 0.0])
            routes = vec.view(1, 1, 3).expand(batch, 3, 3)
            return routes, reference.new_zeros(())

        # Learned policy: compute from evidence
        routes = self.route_heads[level_index](evidence)
        next_timestep = (timestep + 1).clamp_max(
            int(schedule.num_train_timesteps) - 1
        )
        next_evidence = evidence.clone()
        next_evidence[:, :, self.timestep_feature_index], next_evidence[
            :, :, self.log_snr_feature_index
        ] = self._expanded_next_time_features(
            next_timestep, schedule, level_index, reference
        )
        temporal_smoothness = (
            routes - self.route_heads[level_index](next_evidence)
        ).abs().mean()
        return routes, temporal_smoothness

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
        return normalized_timestep.to(reference), normalized_log_snr.to(reference)

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
        dct, dct_offset, dct_weights = self._dct_evidence(residual)
        residual_distribution = self._band_distribution(residual)
        ct_distribution = self._band_distribution(ct)
        haar_agreement = (1.0 - (residual_distribution - ct_distribution).abs()).clamp(
            0.0, 1.0
        )
        normalized_timestep, normalized_log_snr = self._time_features(
            timesteps, schedule, level_index, residual
        )
        base_reliability = (
            base_gate.mean(dim=(-2, -1)) / self.gate_max
        ).clamp(0.0, 1.0)
        (
            gabor_scale,
            gabor_orientation_energy,
            gabor_anisotropy_energy,
            gabor_haar_agreement,
            gabor_dct_agreement,
        ) = self._gabor_evidence(
            residual_distribution,
            dct,
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
                base_reliability,
                gabor_scale,
                gabor_orientation_energy,
                gabor_anisotropy_energy,
                gabor_haar_agreement,
                gabor_dct_agreement,
            ),
            dim=-1,
        )
        shared_dct = dct.flatten(1)[:, None, :].expand(-1, 3, -1)
        evidence = torch.cat((shared_dct, scalars), dim=-1)
        return evidence, {
            "dct_weight_offset": dct_offset,
            "dct_frequency_weights": dct_weights,
            "gabor_haar_agreement": gabor_haar_agreement,
            "gabor_dct_agreement": gabor_dct_agreement,
        }

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
            "route_null_mean": current_residual.new_zeros(()),
            "route_entropy": current_residual.new_zeros(()),
            "route_active_mass": current_residual.new_zeros(()),
        }
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
        """Compute monitoring statistics for route distributions."""
        all_routes = torch.cat([routes_l2.flatten(0, 1), routes_l1.flatten(0, 1)], dim=0)
        null_prob = all_routes[..., -1] if self._num_routes == 3 else torch.zeros_like(all_routes[..., 0])
        native_prob = all_routes[..., 0]
        shallow_prob = all_routes[..., 1] if all_routes.shape[-1] >= 2 else torch.zeros_like(native_prob)

        # Null statistics (only meaningful for 3-way)
        sorted_null = null_prob.sort().values
        n = sorted_null.numel()
        p10 = sorted_null[int(0.10 * (n - 1))] if n > 0 else reference.new_zeros(())
        p50 = sorted_null[int(0.50 * (n - 1))] if n > 0 else reference.new_zeros(())
        p90 = sorted_null[int(0.90 * (n - 1))] if n > 0 else reference.new_zeros(())
        null_mean = null_prob.mean()

        # Entropy
        eps = 1e-8
        entropy = -(all_routes * (all_routes + eps).log()).sum(dim=-1).mean()

        # Active mass: P(native) + P(shallow) = 1 - P(null)
        active_mass = (native_prob + shallow_prob).mean()

        return {
            "route_null_mean": null_mean,
            "route_null_p10": p10,
            "route_null_p50": p50,
            "route_null_p90": p90,
            "route_entropy": entropy,
            "route_active_mass": active_mass,
            "route_policy": self._effective_policy,
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
    ) -> tuple[list[torch.Tensor], Dict[str, torch.Tensor]]:
        self._validate_inputs(current_residual, ct)
        if timestep.ndim != 1 or timestep.shape[0] != current_residual.shape[0]:
            raise ValueError("timestep must have shape [B]")
        if self.hard_all_null:
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

        delta_l2 = self.amplitude_heads[0](evidence_l2)[:, :, None, None]
        delta_l1 = self.amplitude_heads[1](evidence_l1)[:, :, None, None]
        amplitude_l2 = (gates_l2 * (1.0 + delta_l2)).clamp(0.0, self.gate_max)
        amplitude_l1 = (gates_l1 * (1.0 + delta_l1)).clamp(0.0, self.gate_max)

        diagnostics: Dict[str, torch.Tensor] = {}
        routes_l2, temporal_l2 = self._acquire_routes(
            0, evidence_l2, current_residual, timestep, schedule
        )
        routes_l1, temporal_l1 = self._acquire_routes(
            1, evidence_l1, current_residual, timestep, schedule
        )
        temporal_smoothness = 0.5 * (temporal_l2 + temporal_l1)

        # Update route diagnostics
        diagnostics.update(
            self._route_diagnostics(routes_l2, routes_l1, current_residual)
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

        # Injection norms for monitoring
        injections = [l3, l2, l1, l0]
        injection_norms = {
            f"injection/l{idx}_norm": inj.norm(dim=tuple(range(1, inj.dim()))).mean().detach()
            for idx, inj in enumerate(injections)
        }

        diagnostics.update(
            {
                "gates_l2": amplitude_l2,
                "gates_l1": amplitude_l1,
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
