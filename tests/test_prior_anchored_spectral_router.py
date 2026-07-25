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


def _router(monkeypatch, *, steps: int = 100):
    from src.model.frequency import spectral_router

    active = _active_table(steps)
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
        prior_anchor_decay_end_epoch=99,
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
