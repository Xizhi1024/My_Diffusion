from __future__ import annotations

import pytest
import torch


class _BridgeSchedule:
    def __init__(self, steps: int = 100) -> None:
        self.num_train_timesteps = steps
        self.m_t = torch.linspace(0.0, 1.0, steps)
        self.sigma_t = (
            2 * self.m_t * (1 - self.m_t)
        ).sqrt().clamp_min(1e-4)


def _active_table(steps: int = 100) -> torch.Tensor:
    base = torch.linspace(1.0, 0.0, steps)
    rows = torch.stack(
        (
            base,
            base.square(),
            base.sqrt(),
            base.pow(1.25),
            base.pow(1.5),
            base.pow(3.0),
        )
    )
    return rows.reshape(2, 3, steps)


def _router(
    monkeypatch,
    *,
    steps: int = 100,
    active_table: torch.Tensor | None = None,
):
    from src.model.frequency import spectral_router

    active = _active_table(steps) if active_table is None else active_table
    monkeypatch.setattr(
        spectral_router,
        "load_prior_anchor_schedule",
        lambda *args, **kwargs: (
            active.clone(),
            {
                "pipeline_id": "H3_DIRECT_PNG_PRIOR_PREVIEW_V1",
                "schedule_source": "direct_png_preview",
            },
        ),
    )
    return spectral_router.SpectralEvidenceFrequencyRouter(
        output_channels=(16, 16, 8, 4),
        band_scales=(0.5, 0.25),
        ct_reliability_floors=(0.25, 0.50),
        hidden_channels=8,
        dct_enabled=False,
        gabor_enabled=False,
        route_policy="prior_anchored_learned",
        h3_schedule_path="results/preview/h3_prior_preview.json",
        h3_schedule_sha256="a" * 64,
        h3_schedule_source="direct_png_preview",
        h3_repository_root=".",
        h3_allow_unverified_preview_lineage=True,
        h3_num_train_timesteps=steps,
        prior_warmup_epochs=10,
        prior_active_ramp_epochs=10,
        prior_destination_warmup_epochs=30,
        prior_destination_ramp_epochs=10,
        prior_anchor_decay_end_epoch=100,
        prior_anchor_final_scale=0.10,
    )


def _forward(router, *, timestep: int = 40):
    residual = torch.randn(2, 1, 32, 32)
    ct = torch.randn_like(residual)
    return router(
        residual,
        torch.tensor([timestep, timestep + 1]),
        _BridgeSchedule(),
        ct,
    )


def test_prior_anchored_phase_schedule_matches_100_epoch_contract(monkeypatch):
    router = _router(monkeypatch)
    expected = {
        0: (0.0, 0.0),
        9: (0.0, 0.0),
        10: (0.1, 0.0),
        19: (1.0, 0.0),
        20: (1.0, 0.0),
        29: (1.0, 0.0),
        30: (1.0, 0.1),
        39: (1.0, 1.0),
        40: (1.0, 1.0),
        99: (1.0, 1.0),
    }
    for epoch, (active, destination) in expected.items():
        router.set_training_epoch(epoch)
        assert router._prior_active_progress.item() == pytest.approx(active)
        assert router._prior_destination_progress.item() == pytest.approx(
            destination
        )
    assert router._prior_anchor_scale.item() == pytest.approx(0.10)


def test_warmup_is_exact_h3_native_null_and_bypasses_adaptive_heads(monkeypatch):
    router = _router(monkeypatch)
    _, diagnostics = _forward(router, timestep=40)
    timesteps = torch.tensor([40, 41])
    prior_l2 = _active_table()[
        0, :, timesteps
    ].transpose(0, 1)
    prior_l1 = _active_table()[
        1, :, timesteps
    ].transpose(0, 1)
    for routes, prior in (
        (diagnostics["routes_l2"], prior_l2),
        (diagnostics["routes_l1"], prior_l1),
    ):
        expected = torch.stack(
            (prior, torch.zeros_like(prior), 1.0 - prior),
            dim=-1,
        )
        torch.testing.assert_close(routes, expected)
    assert diagnostics["route_prior_active_mae"].item() == 0.0
    assert diagnostics["route_shallow_mass"].item() == 0.0
    assert all(
        parameter.grad is None
        for module in (
            router.prior_active_heads,
            router.prior_destination_heads,
        )
        for parameter in module.parameters()
    )


def test_active_only_then_destination_release_are_separated(monkeypatch):
    router = _router(monkeypatch)
    with torch.no_grad():
        for head in router.prior_active_heads:
            head.final.bias.fill_(1.0)

    router.set_training_epoch(19)
    _, active_only = _forward(router)
    assert torch.count_nonzero(active_only["routes_l2"][..., 1]).item() == 0
    assert active_only["route_prior_active_mae"].item() > 0
    active_only["routes_l2"][..., 0].mean().backward()
    assert any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in router.prior_active_heads.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in router.prior_destination_heads.parameters()
    )

    router.zero_grad(set_to_none=True)
    router.set_training_epoch(39)
    _, released = _forward(router)
    assert released["routes_l2"][..., 1].mean().item() > 0
    torch.testing.assert_close(
        released["routes_l2"].sum(dim=-1),
        torch.ones_like(released["routes_l2"][..., 0]),
    )
    released["routes_l2"][..., 1].mean().backward()
    assert any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in router.prior_active_heads.parameters()
    )
    assert any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in router.prior_destination_heads.parameters()
    )


def test_prior_endpoints_remain_exact_and_diagnostics_are_finite(monkeypatch):
    router = _router(monkeypatch)
    router.set_training_epoch(99)
    residual = torch.randn(2, 1, 32, 32)
    _, diagnostics = router(
        residual,
        torch.tensor([0, 99]),
        _BridgeSchedule(),
        torch.randn_like(residual),
    )
    for level in ("l2", "l1"):
        routes = diagnostics[f"routes_{level}"]
        assert torch.equal(routes[0, :, 2], torch.zeros(3))
        assert torch.equal(routes[1, :, :2], torch.zeros(3, 2))
        assert torch.equal(routes[1, :, 2], torch.ones(3))
    for key in (
        "route_prior_active",
        "route_active",
        "route_active_delta",
        "route_active_next",
        "route_delta_prev",
        "route_delta_next",
        "route_shallow_probability",
    ):
        assert diagnostics[key].shape == (2, 2, 3)
        assert torch.isfinite(diagnostics[key]).all()


def test_route_masses_partition_one_and_match_phase_invariants(monkeypatch):
    router = _router(monkeypatch)

    # Warmup: destination closed, active correction frozen at zero, so the
    # route must equal the frozen H3 prior exactly.
    router.set_training_epoch(0)
    _, diagnostics = _forward(router, timestep=40)
    assert diagnostics["route_shallow_mass"].item() == pytest.approx(0.0)
    assert diagnostics["route_native_mass"].item() == pytest.approx(
        diagnostics["route_prior_active_mean"].item()
    )
    assert diagnostics["route_null_mass"].item() == pytest.approx(
        1.0 - diagnostics["route_prior_active_mean"].item()
    )

    # Full release: native + shallow + null must sum to 1.0 exactly, and the
    # adaptive active mean is the (possibly corrected) mass split across
    # native + shallow.
    router.set_training_epoch(99)
    _, diagnostics = _forward(router, timestep=40)
    total = (
        diagnostics["route_native_mass"].item()
        + diagnostics["route_shallow_mass"].item()
        + diagnostics["route_null_mass"].item()
    )
    assert total == pytest.approx(1.0, abs=1e-6)
    native_plus_shallow = (
        diagnostics["route_native_mass"].item()
        + diagnostics["route_shallow_mass"].item()
    )
    assert native_plus_shallow == pytest.approx(
        diagnostics["route_active_mean"].item(), abs=1e-6
    )
    assert diagnostics["route_native_mass"].item() >= 0.0
    assert diagnostics["route_shallow_mass"].item() >= 0.0
    assert diagnostics["route_null_mass"].item() >= 0.0
    # Sanity: the four scalar diagnostics are real scalars.
    for key in (
        "route_native_mass",
        "route_shallow_mass",
        "route_null_mass",
        "route_prior_active_mean",
        "route_active_mean",
    ):
        assert diagnostics[key].ndim == 0
        assert torch.isfinite(diagnostics[key])


def test_prior_phase_buffers_survive_state_dict_round_trip(monkeypatch):
    router = _router(monkeypatch)
    router.set_training_epoch(35)
    state = router.state_dict()

    restored = _router(monkeypatch)
    restored.load_state_dict(state)
    assert restored._prior_active_progress.item() == 1.0
    assert restored._prior_destination_progress.item() == pytest.approx(0.6)
    assert (
        restored._prior_anchor_scale.item()
        == router._prior_anchor_scale.item()
    )
    assert restored._prior_active_progress_value == pytest.approx(1.0)
    assert restored._prior_destination_progress_value == pytest.approx(0.6)


def test_verified_prior_is_not_checkpoint_owned_and_legacy_key_is_ignored(
    monkeypatch,
):
    old_table = _active_table()
    new_table = old_table.flip(-1).contiguous()

    old_router = _router(monkeypatch, active_table=old_table)
    old_router.set_training_epoch(35)
    legacy_state = old_router.state_dict()
    assert "_h3_native_active_mass" not in legacy_state
    # Simulate a checkpoint written before the schedule became non-persistent.
    legacy_state["_h3_native_active_mass"] = old_table.clone()

    restored = _router(monkeypatch, active_table=new_table)
    restored.load_state_dict(legacy_state, strict=True)

    torch.testing.assert_close(restored._h3_native_active_mass, new_table)
    assert not torch.equal(restored._h3_native_active_mass, old_table)


def test_legacy_prior_key_is_not_suppressed_for_a_non_prior_policy():
    from src.model.frequency.spectral_router import (
        SpectralEvidenceFrequencyRouter,
    )

    router = SpectralEvidenceFrequencyRouter(
        output_channels=(16, 16, 8, 4),
        band_scales=(0.5, 0.25),
        ct_reliability_floors=(0.25, 0.50),
        hidden_channels=8,
        dct_enabled=False,
        gabor_enabled=False,
        route_policy="native_only",
    )
    incompatible_state = router.state_dict()
    incompatible_state["_h3_native_active_mass"] = _active_table()
    with pytest.raises(RuntimeError, match="Unexpected key"):
        router.load_state_dict(incompatible_state, strict=True)


def test_prior_forward_does_not_read_progress_buffers_with_item(monkeypatch):
    router = _router(monkeypatch)
    router.set_training_epoch(35)
    tracked = {
        router._prior_active_progress.data_ptr(),
        router._prior_destination_progress.data_ptr(),
    }
    original_item = torch.Tensor.item

    def guarded_item(tensor):
        if tensor.data_ptr() in tracked:
            raise AssertionError("prior forward synchronized a progress buffer")
        return original_item(tensor)

    monkeypatch.setattr(torch.Tensor, "item", guarded_item)
    _, diagnostics = _forward(router)
    assert torch.isfinite(diagnostics["route_active"]).all()


def test_shallow_projection_dezero_breaks_cold_start_deadlock():
    """Experiment D: the shallow-path projection gets a small non-zero init so
    it carries gradient from step 0, while staying bias-free (P(0)=0 holds)."""
    from src.model.frequency.spectral_router import BiasFreeZeroProjection

    # Native paths (default init_scale=0.0): zero-init, start inert.
    native = BiasFreeZeroProjection(3, 8)
    assert torch.count_nonzero(native.final.weight) == 0

    # Shallow paths (init_scale=0.01): small non-zero init, still bias-free.
    shallow = BiasFreeZeroProjection(1, 8, init_scale=0.01)
    assert torch.count_nonzero(shallow.final.weight) > 0
    assert shallow.final.bias is None
    # P(0)=0 is preserved by the bias-free construction, not by the init.
    assert torch.allclose(shallow(torch.zeros(2, 1, 8, 8)), torch.zeros(2, 8, 8, 8))
    # Init magnitude stays small (bootstrap, not an artifact source).
    assert shallow.final.weight.abs().mean() < 0.05


def test_prior_anchored_router_dezeros_only_shallow_projections(monkeypatch):
    """The full prior_anchored router de-zeros the two shallow-path projections
    ([2] and l2_to_l1) but leaves the two native-path projections ([0], [1])
    zero-init — so the native branch is untouched and only the deadlocked
    shallow branch is unblocked."""
    router = _router(monkeypatch)
    assert torch.count_nonzero(router.projection_heads[0].final.weight) == 0
    assert torch.count_nonzero(router.projection_heads[1].final.weight) == 0
    assert torch.count_nonzero(router.projection_heads[2].final.weight) > 0
    assert torch.count_nonzero(router.l2_to_l1_projection.final.weight) > 0
