"""Non-image regularization for spectral evidence routing.

The prior-anchored terms intentionally consume router diagnostics from the
``ConditionBundle`` instead of reaching back into the router.  This keeps the
loss usable with the legacy fixed/learned policies and makes each optional
term a no-op when its required diagnostics are absent.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ..interfaces import LossContext, LossTerm


class SpectralRouterRegularizationLoss(LossTerm):
    name = "spectral_router_regularization"

    def __init__(
        self,
        enabled: bool = True,
        weight: float = 1.0,
        temporal_weight: float = 1e-4,
        dct_weight: float = 1e-4,
        gabor_weight: float = 1e-4,
        active_mass_weight: float = 0.0,
        active_mass_floor: float = 0.25,
        prior_anchor_weight: float = 0.0,
        monotonic_weight: float = 0.0,
        curvature_weight: float = 0.0,
        budget_weight: float = 0.0,
        shallow_weight: float = 0.0,
    ) -> None:
        super().__init__(enabled=enabled, weight=weight)
        self.temporal_weight = float(temporal_weight)
        self.dct_weight = float(dct_weight)
        self.gabor_weight = float(gabor_weight)
        self.active_mass_weight = float(active_mass_weight)
        self.active_mass_floor = float(active_mass_floor)
        self.prior_anchor_weight = float(prior_anchor_weight)
        self.monotonic_weight = float(monotonic_weight)
        self.curvature_weight = float(curvature_weight)
        self.budget_weight = float(budget_weight)
        self.shallow_weight = float(shallow_weight)
        if self.active_mass_weight < 0:
            raise ValueError("active_mass_weight must be non-negative")
        if not 0.0 <= self.active_mass_floor <= 1.0:
            raise ValueError("active_mass_floor must be in [0, 1]")
        for name in (
            "prior_anchor_weight",
            "monotonic_weight",
            "curvature_weight",
            "budget_weight",
            "shallow_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @staticmethod
    def _finite_float(
        value: object,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Move a scalar diagnostic to the loss device and sanitize in fp32."""

        if isinstance(value, torch.Tensor):
            tensor = value.to(device=reference.device, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(
                value,
                device=reference.device,
                dtype=torch.float32,
            )
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    @classmethod
    def _get(
        cls,
        ctx: LossContext,
        key: str,
        reference: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        value = ctx.condition.scalars.get(key)
        if value is None:
            return None
        return cls._finite_float(value, reference)

    @staticmethod
    def _mean_or_zero(
        value: Optional[torch.Tensor],
        zero: torch.Tensor,
    ) -> torch.Tensor:
        if value is None or value.numel() == 0:
            return zero
        return value.mean()

    @staticmethod
    def _route_per_sample(
        value: torch.Tensor,
        key: str,
        batch_size: int,
    ) -> torch.Tensor:
        """Reduce a ``[B, 6]`` or ``[B, 2, 3]`` route tensor per sample."""

        if value.ndim not in (2, 3) or value.shape[0] != batch_size:
            raise ValueError(
                f"{key} must have shape [B, 6] or [B, 2, 3]; "
                f"got {tuple(value.shape)} for batch size {batch_size}"
            )
        flattened = value.reshape(batch_size, -1)
        if flattened.shape[1] != 6:
            raise ValueError(
                f"{key} must contain exactly 6 route bands per sample; "
                f"got shape {tuple(value.shape)}"
            )
        return flattened.mean(dim=1)

    @classmethod
    def _batch_gate(
        cls,
        value: Optional[torch.Tensor],
        *,
        default: float,
        key: str,
        batch_size: int,
        reference: torch.Tensor,
        upper_bound: Optional[float] = 1.0,
    ) -> torch.Tensor:
        """Return a scalar-or-[B] gate as a finite fp32 vector."""

        if value is None:
            gate = reference.new_full((batch_size,), float(default))
        elif value.numel() == 1:
            gate = value.reshape(1).expand(batch_size)
        elif value.ndim == 1 and value.shape[0] == batch_size:
            gate = value
        else:
            raise ValueError(
                f"{key} must be scalar or shape [B]; got {tuple(value.shape)} "
                f"for batch size {batch_size}"
            )
        gate = gate.clamp_min(0.0)
        if upper_bound is not None:
            gate = gate.clamp_max(upper_bound)
        return gate

    @staticmethod
    def _masked_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
        zero: torch.Tensor,
    ) -> torch.Tensor:
        denominator = mask.sum()
        if not bool((denominator > 0).item()):
            return zero
        return (values * mask).sum() / denominator

    @classmethod
    def _route_pair(
        cls,
        ctx: LossContext,
        first_key: str,
        second_key: str,
        reference: torch.Tensor,
        batch_size: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        first = cls._get(ctx, first_key, reference)
        second = cls._get(ctx, second_key, reference)
        if first is None or second is None:
            return None
        cls._route_per_sample(first, first_key, batch_size)
        cls._route_per_sample(second, second_key, batch_size)
        if first.shape != second.shape:
            raise ValueError(
                f"{first_key} and {second_key} must have matching shapes; "
                f"got {tuple(first.shape)} and {tuple(second.shape)}"
            )
        return first, second

    def forward(self, ctx: LossContext) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Router regularization is deliberately evaluated in fp32 even under
        # mixed precision because logit corrections and schedule masks are
        # small global values.
        reference = ctx.target_pet.new_zeros((), dtype=torch.float32)
        zero = reference
        if not self.enabled:
            return zero, {
                f"{self.name}/enabled": zero,
                f"{self.name}/loss": zero,
            }

        batch_size = int(ctx.target_pet.shape[0])
        temporal = self._mean_or_zero(
            self._get(
                ctx,
                "spectral_route_temporal_smoothness",
                reference,
            ),
            zero,
        )
        dct = self._mean_or_zero(
            self._get(ctx, "spectral_dct_weight_offset", reference),
            zero,
        )
        gabor = self._mean_or_zero(
            self._get(ctx, "spectral_gabor_parameter_offset", reference),
            zero,
        )
        active_mass = self._mean_or_zero(
            self._get(ctx, "spectral_route_active_mass", reference),
            zero,
        )
        is_learned = self._mean_or_zero(
            self._get(ctx, "spectral_route_is_learned", reference),
            zero,
        ).clamp(0.0, 1.0)
        is_prior_anchored_tensor = self._get(
            ctx,
            "spectral_route_is_prior_anchored",
            reference,
        )
        is_prior_anchored = self._mean_or_zero(
            is_prior_anchored_tensor,
            zero,
        ).clamp(0.0, 1.0)
        active_mass_penalty_raw = (
            torch.relu(active_mass.new_tensor(self.active_mass_floor) - active_mass)
            .square()
            * is_learned
        )
        # A prior-anchored router is already protected by the explicit budget
        # and anchor terms.  The legacy active-mass floor would otherwise fight
        # valid low-availability regions in the recoverability prior.
        active_mass_penalty = active_mass_penalty_raw * (1.0 - is_prior_anchored)

        prior_gate = self._batch_gate(
            is_prior_anchored_tensor,
            default=0.0,
            key="spectral_route_is_prior_anchored",
            batch_size=batch_size,
            reference=reference,
        )
        active_phase = self._batch_gate(
            self._get(ctx, "spectral_route_active_phase", reference),
            default=1.0,
            key="spectral_route_active_phase",
            batch_size=batch_size,
            reference=reference,
        )
        destination_phase = self._batch_gate(
            self._get(ctx, "spectral_route_destination_phase", reference),
            default=1.0,
            key="spectral_route_destination_phase",
            batch_size=batch_size,
            reference=reference,
        )
        anchor_scale = self._batch_gate(
            self._get(ctx, "spectral_route_anchor_scale", reference),
            default=1.0,
            key="spectral_route_anchor_scale",
            batch_size=batch_size,
            reference=reference,
            upper_bound=None,
        )

        prior_anchor = zero
        prior_anchor_effective = zero
        active_delta = self._get(ctx, "spectral_route_active_delta", reference)
        if active_delta is not None:
            delta_per_sample = self._route_per_sample(
                active_delta.square(),
                "spectral_route_active_delta",
                batch_size,
            )
            prior_anchor = delta_per_sample.mean()
            prior_anchor_effective = (
                delta_per_sample * prior_gate * anchor_scale
            ).mean()

        monotonic = zero
        monotonic_effective = zero
        active_pair = self._route_pair(
            ctx,
            "spectral_route_active",
            "spectral_route_active_next",
            reference,
            batch_size,
        )
        if active_pair is not None:
            active, active_next = active_pair
            monotonic_per_sample = self._route_per_sample(
                torch.relu(active_next - active),
                "spectral_route_active_next",
                batch_size,
            )
            has_next = self._batch_gate(
                self._get(ctx, "spectral_route_has_next", reference),
                default=1.0,
                key="spectral_route_has_next",
                batch_size=batch_size,
                reference=reference,
            )
            monotonic = self._masked_mean(monotonic_per_sample, has_next, zero)
            monotonic_effective = self._masked_mean(
                monotonic_per_sample * prior_gate * active_phase,
                has_next,
                zero,
            )

        curvature = zero
        curvature_effective = zero
        delta_prev = self._get(ctx, "spectral_route_delta_prev", reference)
        delta_next = self._get(ctx, "spectral_route_delta_next", reference)
        if (
            active_delta is not None
            and delta_prev is not None
            and delta_next is not None
        ):
            self._route_per_sample(
                delta_prev,
                "spectral_route_delta_prev",
                batch_size,
            )
            self._route_per_sample(
                delta_next,
                "spectral_route_delta_next",
                batch_size,
            )
            if (
                delta_prev.shape != active_delta.shape
                or delta_next.shape != active_delta.shape
            ):
                raise ValueError(
                    "spectral_route_delta_prev, spectral_route_active_delta, "
                    "and spectral_route_delta_next must have matching shapes"
                )
            curvature_per_sample = self._route_per_sample(
                (delta_next - 2.0 * active_delta + delta_prev).abs(),
                "spectral_route_active_delta",
                batch_size,
            )
            has_prev = self._batch_gate(
                self._get(ctx, "spectral_route_has_prev", reference),
                default=1.0,
                key="spectral_route_has_prev",
                batch_size=batch_size,
                reference=reference,
            )
            has_next = self._batch_gate(
                self._get(ctx, "spectral_route_has_next", reference),
                default=1.0,
                key="spectral_route_has_next",
                batch_size=batch_size,
                reference=reference,
            )
            curvature_mask = has_prev * has_next
            curvature = self._masked_mean(
                curvature_per_sample,
                curvature_mask,
                zero,
            )
            curvature_effective = self._masked_mean(
                curvature_per_sample * prior_gate * active_phase,
                curvature_mask,
                zero,
            )

        budget = zero
        budget_effective = zero
        prior_pair = self._route_pair(
            ctx,
            "spectral_route_active",
            "spectral_route_prior_active",
            reference,
            batch_size,
        )
        if prior_pair is not None:
            active, prior_active = prior_pair
            active_mean = self._route_per_sample(
                active,
                "spectral_route_active",
                batch_size,
            )
            prior_mean = self._route_per_sample(
                prior_active,
                "spectral_route_prior_active",
                batch_size,
            )
            budget_per_sample = (active_mean - prior_mean).square()
            budget = budget_per_sample.mean()
            budget_effective = (
                budget_per_sample * prior_gate * active_phase
            ).mean()

        shallow = zero
        shallow_effective = zero
        shallow_probability = self._get(
            ctx,
            "spectral_route_shallow_probability",
            reference,
        )
        if shallow_probability is not None:
            shallow_per_sample = self._route_per_sample(
                shallow_probability,
                "spectral_route_shallow_probability",
                batch_size,
            )
            shallow = shallow_per_sample.mean()
            shallow_effective = (
                shallow_per_sample * prior_gate * destination_phase
            ).mean()

        raw = (
            self.temporal_weight * temporal
            + self.dct_weight * dct
            + self.gabor_weight * gabor
            + self.active_mass_weight * active_mass_penalty
            + self.prior_anchor_weight * prior_anchor_effective
            + self.monotonic_weight * monotonic_effective
            + self.curvature_weight * curvature_effective
            + self.budget_weight * budget_effective
            + self.shallow_weight * shallow_effective
        )
        raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        return self.weight * raw, {
            f"{self.name}/enabled": torch.ones_like(zero),
            f"{self.name}/temporal": temporal.detach(),
            f"{self.name}/dct": dct.detach(),
            f"{self.name}/gabor": gabor.detach(),
            f"{self.name}/active_mass": active_mass.detach(),
            f"{self.name}/active_mass_penalty_raw": active_mass_penalty_raw.detach(),
            f"{self.name}/active_mass_penalty": active_mass_penalty.detach(),
            f"{self.name}/is_prior_anchored": is_prior_anchored.detach(),
            f"{self.name}/active_phase": active_phase.mean().detach(),
            f"{self.name}/destination_phase": destination_phase.mean().detach(),
            f"{self.name}/anchor_scale": anchor_scale.mean().detach(),
            f"{self.name}/prior_anchor": prior_anchor.detach(),
            f"{self.name}/prior_anchor_effective": prior_anchor_effective.detach(),
            f"{self.name}/monotonic": monotonic.detach(),
            f"{self.name}/monotonic_effective": monotonic_effective.detach(),
            f"{self.name}/curvature": curvature.detach(),
            f"{self.name}/curvature_effective": curvature_effective.detach(),
            f"{self.name}/budget": budget.detach(),
            f"{self.name}/budget_effective": budget_effective.detach(),
            f"{self.name}/shallow": shallow.detach(),
            f"{self.name}/shallow_effective": shallow_effective.detach(),
            f"{self.name}/loss": raw.detach(),
        }
