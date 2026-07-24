import torch
import pytest


def test_gabor_parameter_offset_energy_is_zero_positive_and_trainable():
    from src.model.priors.gabor import GaborPrior

    prior = GaborPrior(scales=2, orientations=4, kernel_size=9)
    initial = prior.parameter_offset_energy()
    assert initial.ndim == 0
    assert initial.item() == 0.0

    with torch.no_grad():
        prior.log_frequency[0] = 0.5
    energy = prior.parameter_offset_energy()
    assert energy.item() > 0.0

    energy.backward()
    assert prior.log_frequency.grad is not None
    assert torch.isfinite(prior.log_frequency.grad).all()


def test_selected_dct_descriptor_is_finite_normalized_and_trainable():
    from src.model.frequency.dct_descriptor import SelectedDCTDescriptor

    module = SelectedDCTDescriptor(pooled_size=8, selected_frequencies=12)
    details = torch.randn(2, 3, 16, 16, requires_grad=True)
    descriptor, diagnostics = module(details)

    assert descriptor.shape == (2, 3, 12)
    assert torch.isfinite(descriptor).all()
    assert torch.allclose(
        diagnostics["frequency_weights"].sum(), torch.tensor(1.0), atol=1e-6
    )
    assert diagnostics["weight_offset_energy"].ndim == 0

    descriptor.sum().backward()
    assert details.grad is not None
    assert torch.isfinite(details.grad).all()
    assert module.weight_offsets.grad is not None
    assert torch.isfinite(module.weight_offsets.grad).all()


def test_selected_dct_descriptor_returns_zero_for_constant_input():
    from src.model.frequency.dct_descriptor import SelectedDCTDescriptor

    module = SelectedDCTDescriptor(pooled_size=8, selected_frequencies=12)
    descriptor, _ = module(torch.ones(1, 3, 8, 8))

    torch.testing.assert_close(
        descriptor,
        torch.zeros_like(descriptor),
        atol=torch.finfo(descriptor.dtype).eps,
        rtol=0.0,
    )


def test_selected_dct_descriptor_does_not_amplify_near_constant_input():
    from src.model.frequency.dct_descriptor import SelectedDCTDescriptor

    module = SelectedDCTDescriptor(pooled_size=8, selected_frequencies=12)
    details = torch.ones(1, 3, 8, 8)
    details[..., 0, 0] += 4 * torch.finfo(details.dtype).eps
    descriptor, _ = module(details)

    assert torch.isfinite(descriptor).all()
    assert descriptor.abs().sum().item() < 1e-3


def test_selected_dct_descriptor_starts_at_declared_prior():
    from src.model.frequency.dct_descriptor import SelectedDCTDescriptor

    module = SelectedDCTDescriptor(pooled_size=8, selected_frequencies=12)
    _, diagnostics = module(torch.ones(1, 3, 8, 8))
    assert torch.allclose(
        diagnostics["frequency_weights"], module.prior_weights, atol=1e-6
    )
    assert diagnostics["weight_offset_energy"].item() == 0.0


def test_selected_dct_descriptor_rejects_invalid_configuration():
    from src.model.frequency.dct_descriptor import SelectedDCTDescriptor

    for kwargs in (
        {"pooled_size": 7, "selected_frequencies": 12},
        {"pooled_size": 8, "selected_frequencies": 0},
        {"pooled_size": 8, "selected_frequencies": 13},
    ):
        try:
            SelectedDCTDescriptor(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"configuration should fail: {kwargs}")


def test_bounded_amplitude_head_starts_neutral_and_respects_limits():
    from src.model.frequency.spectral_router import BoundedAmplitudeHead

    head = BoundedAmplitudeHead(16, hidden_channels=8, minimum=-0.05, maximum=0.10)
    evidence = torch.randn(4, 16)
    initial = head(evidence)
    assert torch.allclose(initial, torch.zeros_like(initial), atol=1e-7)
    with torch.no_grad():
        head.final.weight.fill_(100.0)
    bounded = head(evidence)
    assert bounded.min() >= -0.05
    assert bounded.max() <= 0.10


def test_conservative_route_head_is_null_biased_and_sums_to_one():
    from src.model.frequency.spectral_router import ConservativeRouteHead

    head = ConservativeRouteHead(16, hidden_channels=8, initial_null_probability=0.90)
    probabilities = head(torch.randn(6, 16))
    assert probabilities.shape == (6, 3)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(6), atol=1e-6)
    assert torch.allclose(probabilities[:, 2], torch.full((6,), 0.90), atol=1e-6)


def test_uncertainty_aware_selector_abstains_exactly_to_h3_schedule():
    from src.model.frequency.spectral_router import (
        UncertaintyAwareRouteSelector,
    )

    selector = UncertaintyAwareRouteSelector(confidence_threshold=0.6)
    evidence = torch.tensor(
        [
            [[0.1, 0.2, 0.7], [0.7, 0.2, 0.1]],
            [[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]],
        ]
    )
    h3 = torch.tensor([0.8, 0.1, 0.1]).view(1, 1, 3).expand_as(evidence)
    confidence = torch.tensor([[0.9, 0.2], [0.6, 0.59]])

    selected, diagnostics = selector(evidence, confidence, h3)

    torch.testing.assert_close(selected[0, 0], evidence[0, 0])
    torch.testing.assert_close(selected[0, 1], h3[0, 1])
    torch.testing.assert_close(selected[1, 0], evidence[1, 0])
    torch.testing.assert_close(selected[1, 1], h3[1, 1])
    assert diagnostics["router_active_fraction"].item() == 0.5
    assert diagnostics["router_abstained"].sum().item() == 2


def test_uncertainty_aware_selector_rejects_shape_mismatch():
    from src.model.frequency.spectral_router import (
        UncertaintyAwareRouteSelector,
    )

    selector = UncertaintyAwareRouteSelector(confidence_threshold=0.5)
    try:
        selector(
            torch.ones(2, 3, 3),
            torch.ones(2, 2),
            torch.ones(2, 3, 3),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("confidence shape mismatch must fail")


# ---------------------------------------------------------------------
# Stage V2-05: UncertaintyAwareRouteSelector wired into the router forward.
# The selector defaults OFF (no behaviour change); when ON it is fail-closed
# (requires a frozen threshold + an externally-supplied router_confidence)
# and abstains element-wise to the frozen fixed-policy fallback.
# ---------------------------------------------------------------------


def _selector_router(threshold: float = 0.5, **overrides):
    return _router(
        uncertainty_aware_router_enabled=True,
        uncertainty_aware_confidence_threshold=threshold,
        **overrides,
    )


def test_uncertainty_aware_router_is_off_by_default_and_behaves_unchanged():
    module = _router()  # flag off
    assert module._uncertainty_aware_selector is None
    residual = torch.randn(2, 1, 32, 32)
    _, diagnostics = module(
        residual,
        torch.tensor([20, 60]),
        _BridgeSchedule(),
        torch.randn_like(residual),
    )
    # No selector diagnostics when disabled.
    assert "router_active_fraction" not in diagnostics


def test_uncertainty_aware_router_enabled_requires_learned_policy():
    try:
        _selector_router(route_policy="native_only")
    except ValueError:
        pass
    else:
        raise AssertionError("selector must require a learned policy")


def test_uncertainty_aware_router_enabled_requires_threshold():
    try:
        _router(uncertainty_aware_router_enabled=True)
    except ValueError:
        pass
    else:
        raise AssertionError("selector requires a confidence threshold")


def test_uncertainty_aware_router_fail_closed_without_confidence():
    module = _selector_router()
    with pytest.raises(ValueError, match="router_confidence"):
        module(
            torch.randn(1, 1, 32, 32),
            torch.tensor([40]),
            _BridgeSchedule(),
            torch.randn(1, 1, 32, 32),
        )


def test_uncertainty_aware_router_rejects_bad_confidence_shape():
    module = _selector_router()
    residual = torch.randn(2, 1, 32, 32)
    with pytest.raises(ValueError, match=r"router_confidence must have shape"):
        module(
            residual,
            torch.tensor([20, 60]),
            _BridgeSchedule(),
            torch.randn_like(residual),
            router_confidence=torch.ones(2, 3, 3),  # wrong: must be [B,2,3]
        )


def test_uncertainty_aware_router_full_abstain_equals_fallback_exactly():
    threshold = 0.5
    module = _selector_router(threshold=threshold)
    residual = torch.randn(2, 1, 32, 32)
    ct = torch.randn_like(residual)
    confidence = torch.zeros(2, 2, 3)  # all below threshold -> full abstain

    _, diagnostics = module(
        residual,
        torch.tensor([20, 60]),
        _BridgeSchedule(),
        ct,
        router_confidence=confidence,
    )
    fallback = torch.tensor([1.0, 0.0, 0.0]).view(1, 1, 3).expand(2, 3, 3)
    torch.testing.assert_close(diagnostics["routes_l2"], fallback)
    torch.testing.assert_close(diagnostics["routes_l1"], fallback)
    torch.testing.assert_close(
        diagnostics["router_fallback_routes_l2"], diagnostics["routes_l2"]
    )
    assert diagnostics["router_active_fraction"].item() == 0.0
    assert diagnostics["router_abstained"].sum().item() == 2 * 2 * 3


def test_uncertainty_aware_router_no_abstain_keeps_evidence_routes():
    torch.manual_seed(123)
    baseline = _router()  # selector off -> pure evidence routes
    residual = torch.randn(2, 1, 32, 32)
    ct = torch.randn_like(residual)
    args = (residual, torch.tensor([20, 60]), _BridgeSchedule(), ct)
    _, baseline_diag = baseline(*args)

    torch.manual_seed(123)
    active = _selector_router(threshold=0.5)
    confidence = torch.ones(2, 2, 3)  # all active
    _, active_diag = active(*args, router_confidence=confidence)

    torch.testing.assert_close(active_diag["routes_l2"], baseline_diag["routes_l2"])
    torch.testing.assert_close(active_diag["routes_l1"], baseline_diag["routes_l1"])
    assert active_diag["router_active_fraction"].item() == 1.0


def test_uncertainty_aware_router_mixed_confidence_selects_elementwise():
    torch.manual_seed(7)
    baseline = _router()
    residual = torch.randn(2, 1, 32, 32)
    ct = torch.randn_like(residual)
    args = (residual, torch.tensor([20, 60]), _BridgeSchedule(), ct)
    _, baseline_diag = baseline(*args)
    evidence_l2 = baseline_diag["routes_l2"]
    evidence_l1 = baseline_diag["routes_l1"]

    torch.manual_seed(7)
    module = _selector_router(threshold=0.5)
    # Active grid: level 0 fully active, level 1 fully abstain.
    confidence = torch.zeros(2, 2, 3)
    confidence[:, 0, :] = 1.0
    _, diag = module(*args, router_confidence=confidence)

    fallback = torch.tensor([1.0, 0.0, 0.0]).view(1, 1, 3).expand(2, 3, 3)
    torch.testing.assert_close(diag["routes_l2"], evidence_l2)  # level 0 active
    torch.testing.assert_close(diag["routes_l1"], fallback)     # level 1 abstain
    assert diag["router_active_fraction"].item() == pytest.approx(0.5)


def test_uncertainty_aware_router_is_deterministic_under_same_confidence():
    module = _selector_router(threshold=0.5)
    residual = torch.randn(1, 1, 32, 32)
    ct = torch.randn_like(residual)
    confidence = torch.rand(1, 2, 3)
    common = (residual, torch.tensor([40]), _BridgeSchedule(), ct)
    _, first = module(*common, router_confidence=confidence)
    _, second = module(*common, router_confidence=confidence)
    torch.testing.assert_close(first["routes_l2"], second["routes_l2"])
    torch.testing.assert_close(first["routes_l1"], second["routes_l1"])


def test_uncertainty_aware_router_gradient_only_through_active_routes():
    # Project onto the native-route mass (routes[...,:-1] are NOT constant,
    # unlike the full softmax sum). Full abstain: selected routes are the
    # constant fallback, so route heads receive ZERO gradient. Full active:
    # they receive non-zero gradient.
    torch.manual_seed(11)
    abstain_module = _selector_router(threshold=0.5)
    residual = torch.randn(1, 1, 32, 32)
    ct = torch.randn_like(residual)
    args = (residual, torch.tensor([40]), _BridgeSchedule(), ct)
    _, abstain_diag = abstain_module(*args, router_confidence=torch.zeros(1, 2, 3))
    abstain_diag["routes_l2"][..., 0].sum().backward()
    abstain_grads = [
        p.grad
        for n, p in abstain_module.named_parameters()
        if "route_heads" in n and p.grad is not None
    ]
    assert abstain_grads, "route heads should exist"
    assert all(torch.count_nonzero(g).item() == 0 for g in abstain_grads)

    torch.manual_seed(11)
    active_module = _selector_router(threshold=0.5)
    _, active_diag = active_module(*args, router_confidence=torch.ones(1, 2, 3))
    active_module.zero_grad(set_to_none=True)
    active_diag["routes_l2"][..., 0].sum().backward()
    active_grads = [
        p.grad
        for n, p in active_module.named_parameters()
        if "route_heads" in n and p.grad is not None
    ]
    assert active_grads
    assert any(torch.count_nonzero(g).item() > 0 for g in active_grads)


def test_uncertainty_aware_router_independent_of_training_masks():
    # lesion_score / topq_mask are accepted for API compatibility but must not
    # influence routes; confidence is the only selection input.
    module = _selector_router(threshold=0.5)
    residual = torch.randn(1, 1, 32, 32)
    ct = torch.randn_like(residual)
    confidence = torch.tensor([[[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]]])
    schedule = _BridgeSchedule()
    _, diag_a = module(
        residual,
        torch.tensor([40]),
        schedule,
        ct,
        router_confidence=confidence,
        lesion_score=torch.zeros(1, 1, 32, 32),
        topq_mask=torch.zeros(1, 1, 32, 32, dtype=torch.bool),
    )
    _, diag_b = module(
        residual,
        torch.tensor([40]),
        schedule,
        ct,
        router_confidence=confidence,
        lesion_score=torch.ones(1, 1, 32, 32) * 7.0,
        topq_mask=torch.ones(1, 1, 32, 32, dtype=torch.bool),
    )
    torch.testing.assert_close(diag_a["routes_l2"], diag_b["routes_l2"])
    torch.testing.assert_close(diag_a["routes_l1"], diag_b["routes_l1"])


def test_uncertainty_aware_router_forward_invokes_selector_when_enabled():
    # "main model forward really calls selector": enabling + supplying
    # confidence must change routes relative to the abstain-all case and emit
    # the full selector diagnostic set.
    module = _selector_router(threshold=0.5)
    residual = torch.randn(1, 1, 32, 32)
    ct = torch.randn_like(residual)
    schedule = _BridgeSchedule()
    _, active = module(
        residual,
        torch.tensor([40]),
        schedule,
        ct,
        router_confidence=torch.ones(1, 2, 3),
    )
    for key in (
        "router_confidence",
        "router_active",
        "router_abstained",
        "router_active_fraction",
        "router_selected_routes_l2",
        "router_selected_routes_l1",
        "router_fallback_routes_l2",
        "router_fallback_routes_l1",
        "router_confidence_threshold",
    ):
        assert key in active, key
    _, abstain = module(
        residual,
        torch.tensor([40]),
        schedule,
        ct,
        router_confidence=torch.zeros(1, 2, 3),
    )
    # Active vs full-abstain must produce different L2 routes.
    assert not torch.allclose(active["routes_l2"], abstain["routes_l2"])


def test_uncertainty_aware_router_keeps_l3_zero():
    # L3 injection must remain zero unless a separate mechanism gate passes.
    module = _selector_router(threshold=0.5)
    residual = torch.randn(1, 1, 32, 32)
    injections, _ = module(
        residual,
        torch.tensor([40]),
        _BridgeSchedule(),
        torch.randn_like(residual),
        router_confidence=torch.ones(1, 2, 3),
    )
    assert torch.count_nonzero(injections[0]).item() == 0  # L3 is injections[0]


class _BridgeSchedule:
    def __init__(self, steps: int = 100):
        self.num_train_timesteps = steps
        self.m_t = torch.linspace(0.0, 1.0, steps)
        self.sigma_t = (2 * self.m_t * (1 - self.m_t)).sqrt().clamp_min(1e-4)


def _router(**overrides):
    from src.model.frequency.spectral_router import SpectralEvidenceFrequencyRouter

    kwargs = {
        "output_channels": (16, 16, 8, 4),
        "band_scales": (0.5, 0.25),
        "ct_reliability_floors": (0.25, 0.50),
        "gabor_orientations": 8,
        "hidden_channels": 16,
        "initial_null_probability": 0.90,
    }
    kwargs.update(overrides)
    return SpectralEvidenceFrequencyRouter(**kwargs)


def test_router_first_forward_is_exact_noop_and_routes_are_conservative():
    module = _router()
    residual = torch.randn(2, 1, 32, 32)
    ct = torch.randn_like(residual)
    orientation = torch.rand(2, 8, 32, 32)
    anisotropy = torch.rand(2, 1, 32, 32)

    injections, diagnostics = module(
        residual,
        torch.tensor([20, 60]),
        _BridgeSchedule(),
        ct,
        gabor_orientation=orientation,
        gabor_anisotropy=anisotropy,
    )

    assert [tuple(x.shape) for x in injections] == [
        (2, 16, 4, 4),
        (2, 16, 8, 8),
        (2, 8, 16, 16),
        (2, 4, 32, 32),
    ]
    assert all(torch.count_nonzero(x).item() == 0 for x in injections)
    assert torch.allclose(diagnostics["routes_l2"].sum(-1), torch.ones(2, 3))
    assert torch.allclose(diagnostics["routes_l1"].sum(-1), torch.ones(2, 3))
    assert diagnostics["routes_l2"][..., 2].mean() > 0.89
    assert diagnostics["routes_l1"][..., 2].mean() > 0.89


def test_router_allows_balanced_null_prior():
    module = _router(initial_null_probability=0.50)
    _, diagnostics = module(
        torch.randn(2, 1, 32, 32),
        torch.tensor([20, 60]),
        _BridgeSchedule(),
        torch.randn(2, 1, 32, 32),
    )

    torch.testing.assert_close(
        diagnostics["routes_l2"][..., 2],
        torch.full((2, 3), 0.50),
    )


def test_native_warmup_and_ramp_gradually_release_learned_routes():
    module = _router(
        initial_null_probability=0.50,
        native_warmup_epochs=2,
        routing_ramp_epochs=2,
    )
    residual = torch.randn(1, 1, 32, 32)
    ct = torch.randn_like(residual)
    args = (residual, torch.tensor([40]), _BridgeSchedule(), ct)

    _, warmup = module(*args)
    expected_native = torch.tensor([1.0, 0.0, 0.0]).view(1, 1, 3).expand(1, 3, 3)
    torch.testing.assert_close(warmup["routes_l2"], expected_native)
    assert warmup["route_routing_progress"].item() == 0.0

    module.set_training_epoch(2)
    _, halfway = module(*args)
    expected_halfway = torch.tensor([0.625, 0.125, 0.25]).view(1, 1, 3).expand(1, 3, 3)
    torch.testing.assert_close(halfway["routes_l2"], expected_halfway)
    assert halfway["route_routing_progress"].item() == 0.5

    module.set_training_epoch(3)
    _, released = module(*args)
    expected_released = torch.tensor([0.25, 0.25, 0.50]).view(1, 1, 3).expand(1, 3, 3)
    torch.testing.assert_close(released["routes_l2"], expected_released)
    assert released["route_routing_progress"].item() == 1.0


def test_hard_all_null_short_circuits_before_descriptors_and_projection():
    module = _router(hard_all_null=True)

    class _RejectWork(torch.nn.Module):
        def forward(self, value):
            raise AssertionError("hard all-null must not perform spectral work")

    module.dct_descriptor = _RejectWork()
    module.projection_heads = torch.nn.ModuleList(
        [_RejectWork(), _RejectWork(), _RejectWork()]
    )
    module.l2_to_l1_projection = _RejectWork()
    residual = torch.randn(1, 1, 32, 32)
    injections, diagnostics = module(
        residual,
        torch.tensor([50]),
        _BridgeSchedule(),
        torch.randn_like(residual),
        gabor_orientation=torch.randn(1, 1, 8, 8),
        gabor_anisotropy=torch.randn(1, 2, 8, 8),
    )
    assert all(torch.count_nonzero(x).item() == 0 for x in injections)
    expected_routes = torch.tensor([[[0.0, 0.0, 1.0]]]).expand(1, 3, 3)
    assert torch.equal(diagnostics["routes_l2"], expected_routes)
    assert torch.equal(diagnostics["routes_l1"], expected_routes)
    for level in ("l2", "l1"):
        assert diagnostics[f"route_{level}_active_mass"].item() == 0.0
        assert diagnostics[f"route_{level}_entropy"].item() == 0.0
        for statistic in ("mean", "p10", "p50", "p90"):
            assert diagnostics[f"route_{level}_null_{statistic}"].item() == 1.0
    for level in range(4):
        assert diagnostics[f"injection/l{level}_rms"].item() == 0.0


def test_learned_no_null_routes_are_two_way_normalized_and_trainable():
    module = _router(route_policy="learned_no_null", fixed_prior=(0.5, 0.5))
    residual = torch.randn(2, 1, 32, 32)
    injections, diagnostics = module(
        residual,
        torch.tensor([20, 60]),
        _BridgeSchedule(),
        torch.randn_like(residual),
    )

    assert all(torch.isfinite(value).all() for value in injections)
    for level in ("l2", "l1"):
        routes = diagnostics[f"routes_{level}"]
        assert routes.shape[-1] == 2
        assert torch.isfinite(routes).all()
        torch.testing.assert_close(
            routes.sum(dim=-1),
            torch.ones_like(routes[..., 0]),
        )
        assert not any("null" in key for key in diagnostics if key.startswith(f"route_{level}_"))

    diagnostics["routes_l2"][..., 0].mean().backward()
    gradients = [
        parameter.grad
        for head in module.no_null_route_heads
        for parameter in head.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)


def test_router_half_precision_preserves_reference_dtype_and_stays_finite():
    module = _router().half()
    residual = torch.randn(2, 1, 32, 32, dtype=torch.float16)
    ct = torch.randn_like(residual)

    injections, diagnostics = module(
        residual,
        torch.tensor([20, 60]),
        _BridgeSchedule(),
        ct,
    )

    assert all(value.dtype == torch.float16 for value in injections)
    assert all(torch.isfinite(value).all() for value in injections)
    assert all(torch.isfinite(value).all() for value in diagnostics.values())


def test_router_half_precision_is_finite_at_bridge_endpoints():
    module = _router().half()
    residual = torch.zeros(2, 1, 32, 32, dtype=torch.float16)
    ct = torch.ones_like(residual)
    schedule = _BridgeSchedule()

    injections, diagnostics = module(
        residual,
        torch.tensor([0, schedule.num_train_timesteps - 1]),
        schedule,
        ct,
    )

    assert all(torch.isfinite(value).all() for value in injections)
    assert all(torch.count_nonzero(value).item() == 0 for value in injections)
    for key in (
        "routes_l2",
        "routes_l1",
        "gates_l2",
        "gates_l1",
        "noise_reliability",
        "route_temporal_smoothness",
    ):
        assert torch.isfinite(diagnostics[key]).all(), key
    assert torch.allclose(
        diagnostics["routes_l2"].sum(dim=-1),
        torch.ones(2, 3, dtype=torch.float16),
    )
    assert torch.allclose(
        diagnostics["routes_l1"].sum(dim=-1),
        torch.ones(2, 3, dtype=torch.float16),
    )
    assert diagnostics["routes_l2"][..., 2].mean() > 0.89
    assert diagnostics["routes_l1"][..., 2].mean() > 0.89
    assert diagnostics["gates_l2"].min() >= 0
    assert diagnostics["gates_l2"].max() <= module.gate_max
    assert diagnostics["gates_l1"].min() >= 0
    assert diagnostics["gates_l1"].max() <= module.gate_max
    assert diagnostics["noise_reliability"].min() >= 0
    assert diagnostics["noise_reliability"].max() <= 1


def test_router_normalizes_optional_gabor_evidence_to_reference_dtype():
    module = _router().half()
    residual = torch.randn(2, 1, 32, 32, dtype=torch.float16)
    ct = torch.randn_like(residual)

    injections, diagnostics = module(
        residual,
        torch.tensor([20, 60]),
        _BridgeSchedule(),
        ct,
        gabor_orientation=torch.rand(2, 8, 32, 32, dtype=torch.float32),
        gabor_anisotropy=torch.rand(2, 1, 32, 32, dtype=torch.float64),
        gabor_feat=torch.rand(2, 16, 32, 32, dtype=torch.float32),
    )

    assert all(value.dtype == torch.float16 for value in injections)
    assert diagnostics["gabor_haar_agreement"].dtype == torch.float16
    assert diagnostics["gabor_dct_agreement"].dtype == torch.float16
    assert all(torch.isfinite(value).all() for value in diagnostics.values())


def test_router_receives_gradients_after_zero_projection_learns():
    module = _router()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    residual = torch.randn(2, 1, 32, 32)
    ct = torch.randn_like(residual)
    args = (residual, torch.tensor([20, 60]), _BridgeSchedule(), ct)

    first, _ = module(*args)
    sum(x.sum() for x in first).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second, _ = module(*args)
    sum(x.square().mean() for x in second).backward()
    route_grads = [
        parameter.grad
        for name, parameter in module.named_parameters()
        if "route_heads" in name and parameter.grad is not None
    ]
    assert route_grads
    assert all(torch.isfinite(grad).all() for grad in route_grads)
    assert any(torch.count_nonzero(grad).item() > 0 for grad in route_grads)
