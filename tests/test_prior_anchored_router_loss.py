from __future__ import annotations

import pytest
import torch

from src.model.interfaces import ConditionBundle, LossContext
from src.model.loss_terms.spectral_router import (
    SpectralRouterRegularizationLoss,
)


def _context(
    scalars: dict[str, torch.Tensor],
    *,
    batch_size: int = 2,
    dtype: torch.dtype = torch.float32,
) -> LossContext:
    target = torch.zeros(batch_size, 1, 4, 4, dtype=dtype)
    return LossContext(
        model_pred=target,
        loss_target=target,
        target_pet=target,
        pred_x0=target,
        timesteps=torch.zeros(batch_size, dtype=torch.long),
        tau=torch.zeros(batch_size),
        batch={},
        condition=ConditionBundle(scalars=scalars),
    )


def test_legacy_terms_are_unchanged_when_new_inputs_are_absent() -> None:
    scalars = {
        "spectral_route_temporal_smoothness": torch.tensor([0.2, 0.4]),
        "spectral_dct_weight_offset": torch.tensor(0.3),
        "spectral_gabor_parameter_offset": torch.tensor(0.5),
        "spectral_route_active_mass": torch.tensor(0.1),
        "spectral_route_is_learned": torch.tensor(1.0),
    }
    term = SpectralRouterRegularizationLoss(
        weight=2.0,
        temporal_weight=3.0,
        dct_weight=4.0,
        gabor_weight=5.0,
        active_mass_weight=6.0,
        active_mass_floor=0.3,
    )

    loss, logs = term(_context(scalars))

    raw_expected = (
        3.0 * 0.3
        + 4.0 * 0.3
        + 5.0 * 0.5
        + 6.0 * (0.3 - 0.1) ** 2
    )
    assert loss.item() == pytest.approx(2.0 * raw_expected)
    assert logs[f"{term.name}/loss"].item() == pytest.approx(raw_expected)
    assert logs[f"{term.name}/prior_anchor"].item() == 0.0
    assert logs[f"{term.name}/monotonic"].item() == 0.0
    assert logs[f"{term.name}/curvature"].item() == 0.0
    assert logs[f"{term.name}/budget"].item() == 0.0
    assert logs[f"{term.name}/shallow"].item() == 0.0


def test_prior_anchored_terms_apply_phase_scales_and_boundary_masks() -> None:
    active_delta = torch.stack(
        (torch.ones(6), torch.full((6,), 2.0)),
    ).requires_grad_()
    active = torch.stack(
        (torch.full((6,), 0.2), torch.full((6,), 0.8)),
    ).requires_grad_()
    active_next = torch.stack(
        (torch.full((6,), 0.3), torch.full((6,), 0.7)),
    )
    prior_active = torch.stack(
        (torch.full((6,), 0.1), torch.full((6,), 0.5)),
    )
    shallow_probability = torch.stack(
        (torch.full((6,), 0.2), torch.full((6,), 0.4)),
    ).requires_grad_()
    scalars = {
        "spectral_route_is_prior_anchored": torch.ones(2),
        "spectral_route_active_phase": torch.tensor([0.5, 0.25]),
        "spectral_route_destination_phase": torch.tensor([0.25, 0.5]),
        "spectral_route_anchor_scale": torch.tensor([0.5, 0.25]),
        "spectral_route_prior_active": prior_active,
        "spectral_route_active": active,
        "spectral_route_active_delta": active_delta,
        "spectral_route_active_next": active_next,
        "spectral_route_delta_prev": torch.zeros_like(active_delta),
        "spectral_route_delta_next": torch.zeros_like(active_delta),
        "spectral_route_shallow_probability": shallow_probability,
        "spectral_route_has_next": torch.ones(2),
        # Curvature must ignore the second sample at the t=0 boundary.
        "spectral_route_has_prev": torch.tensor([1.0, 0.0]),
    }
    term = SpectralRouterRegularizationLoss(
        temporal_weight=0.0,
        dct_weight=0.0,
        gabor_weight=0.0,
        prior_anchor_weight=2.0,
        monotonic_weight=3.0,
        curvature_weight=4.0,
        budget_weight=5.0,
        shallow_weight=6.0,
    )

    loss, logs = term(_context(scalars))

    # Raw diagnostics:
    #   anchor    = mean([1**2, 2**2]) = 2.5
    #   monotonic = mean([0.1, 0.0]) = 0.05
    #   curvature = 2.0 (only sample 0 is valid)
    #   budget    = mean([0.1**2, 0.3**2]) = 0.05
    #   shallow   = mean([0.2, 0.4]) = 0.3
    assert logs[f"{term.name}/prior_anchor"].item() == pytest.approx(2.5)
    assert logs[f"{term.name}/monotonic"].item() == pytest.approx(0.05)
    assert logs[f"{term.name}/curvature"].item() == pytest.approx(2.0)
    assert logs[f"{term.name}/budget"].item() == pytest.approx(0.05)
    assert logs[f"{term.name}/shallow"].item() == pytest.approx(0.3)

    # Effective diagnostics include phase/anchor scaling and valid-neighbour masks.
    assert logs[f"{term.name}/prior_anchor_effective"].item() == pytest.approx(0.75)
    assert logs[f"{term.name}/monotonic_effective"].item() == pytest.approx(0.025)
    assert logs[f"{term.name}/curvature_effective"].item() == pytest.approx(1.0)
    assert logs[f"{term.name}/budget_effective"].item() == pytest.approx(0.01375)
    assert logs[f"{term.name}/shallow_effective"].item() == pytest.approx(0.125)
    expected = (
        2.0 * 0.75
        + 3.0 * 0.025
        + 4.0 * 1.0
        + 5.0 * 0.01375
        + 6.0 * 0.125
    )
    assert loss.item() == pytest.approx(expected)

    loss.backward()
    for tensor in (active_delta, active, shallow_probability):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_missing_raw_tensors_make_each_optional_term_a_noop() -> None:
    term = SpectralRouterRegularizationLoss(
        temporal_weight=0.0,
        dct_weight=0.0,
        gabor_weight=0.0,
        prior_anchor_weight=1.0,
        monotonic_weight=1.0,
        curvature_weight=1.0,
        budget_weight=1.0,
        shallow_weight=1.0,
    )
    scalars = {
        "spectral_route_is_prior_anchored": torch.tensor(1.0),
        "spectral_route_active_phase": torch.tensor(1.0),
        "spectral_route_destination_phase": torch.tensor(1.0),
        "spectral_route_anchor_scale": torch.tensor(1.0),
    }

    loss, logs = term(_context(scalars))

    assert loss.item() == 0.0
    for key in (
        "prior_anchor_effective",
        "monotonic_effective",
        "curvature_effective",
        "budget_effective",
        "shallow_effective",
    ):
        assert logs[f"{term.name}/{key}"].item() == 0.0


def test_active_mass_floor_is_disabled_for_prior_anchored_mode() -> None:
    term = SpectralRouterRegularizationLoss(
        temporal_weight=0.0,
        dct_weight=0.0,
        gabor_weight=0.0,
        active_mass_weight=10.0,
        active_mass_floor=0.4,
    )
    scalars = {
        "spectral_route_active_mass": torch.tensor(0.1),
        "spectral_route_is_learned": torch.tensor(1.0),
        "spectral_route_is_prior_anchored": torch.tensor(1.0),
    }

    loss, logs = term(_context(scalars))

    assert logs[f"{term.name}/active_mass_penalty_raw"].item() == pytest.approx(0.09)
    assert logs[f"{term.name}/active_mass_penalty"].item() == 0.0
    assert loss.item() == 0.0


def test_boundary_masks_can_disable_neighbour_regularization() -> None:
    shape = (2, 2, 3)
    scalars = {
        "spectral_route_is_prior_anchored": torch.ones(2),
        "spectral_route_active": torch.zeros(shape),
        "spectral_route_active_next": torch.ones(shape),
        "spectral_route_active_delta": torch.ones(shape),
        "spectral_route_delta_prev": torch.zeros(shape),
        "spectral_route_delta_next": torch.zeros(shape),
        "spectral_route_has_prev": torch.zeros(2),
        "spectral_route_has_next": torch.zeros(2),
    }
    term = SpectralRouterRegularizationLoss(
        temporal_weight=0.0,
        dct_weight=0.0,
        gabor_weight=0.0,
        monotonic_weight=1.0,
        curvature_weight=1.0,
    )

    loss, logs = term(_context(scalars))

    assert logs[f"{term.name}/monotonic"].item() == 0.0
    assert logs[f"{term.name}/curvature"].item() == 0.0
    assert loss.item() == 0.0


def test_nonfinite_router_diagnostics_produce_finite_float32_loss_and_logs() -> None:
    bad = torch.tensor(
        [
            [float("nan"), float("inf"), float("-inf"), 1.0, -1.0, 0.0],
            [float("inf"), float("nan"), 0.0, 1.0, -1.0, 2.0],
        ],
        dtype=torch.float32,
    )
    scalars = {
        "spectral_route_temporal_smoothness": torch.tensor(float("nan")),
        "spectral_dct_weight_offset": torch.tensor(float("inf")),
        "spectral_gabor_parameter_offset": torch.tensor(float("-inf")),
        "spectral_route_active_mass": torch.tensor(float("nan")),
        "spectral_route_is_learned": torch.tensor(1.0),
        "spectral_route_is_prior_anchored": torch.ones(2),
        "spectral_route_active_phase": torch.ones(2),
        "spectral_route_destination_phase": torch.ones(2),
        "spectral_route_anchor_scale": torch.ones(2),
        "spectral_route_prior_active": bad,
        "spectral_route_active": bad,
        "spectral_route_active_delta": bad,
        "spectral_route_active_next": bad,
        "spectral_route_delta_prev": bad,
        "spectral_route_delta_next": bad,
        "spectral_route_shallow_probability": bad,
    }
    term = SpectralRouterRegularizationLoss(
        prior_anchor_weight=1.0,
        monotonic_weight=1.0,
        curvature_weight=1.0,
        budget_weight=1.0,
        shallow_weight=1.0,
    )

    loss, logs = term(_context(scalars, dtype=torch.bfloat16))

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert all(value.dtype == torch.float32 for value in logs.values())
    assert all(torch.isfinite(value).all() for value in logs.values())


@pytest.mark.parametrize(
    "weight_name",
    (
        "prior_anchor_weight",
        "monotonic_weight",
        "curvature_weight",
        "budget_weight",
        "shallow_weight",
    ),
)
def test_new_weights_must_be_nonnegative(weight_name: str) -> None:
    with pytest.raises(ValueError, match=weight_name):
        SpectralRouterRegularizationLoss(**{weight_name: -1.0})


def test_route_tensor_shape_contract_is_checked() -> None:
    term = SpectralRouterRegularizationLoss(prior_anchor_weight=1.0)
    scalars = {
        "spectral_route_is_prior_anchored": torch.tensor(1.0),
        "spectral_route_active_delta": torch.zeros(2, 5),
    }

    with pytest.raises(ValueError, match="exactly 6 route bands"):
        term(_context(scalars))


def test_masked_mean_preserves_fractional_weight_semantics() -> None:
    values = torch.tensor([2.0])
    zero = torch.zeros(())

    observed = SpectralRouterRegularizationLoss._masked_mean(
        values,
        torch.tensor([0.5]),
        zero,
    )
    empty = SpectralRouterRegularizationLoss._masked_mean(
        values,
        torch.tensor([0.0]),
        zero,
    )

    assert observed.item() == pytest.approx(2.0)
    assert empty.item() == pytest.approx(0.0)
