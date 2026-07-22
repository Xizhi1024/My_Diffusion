import torch


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
