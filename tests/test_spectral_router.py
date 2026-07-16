import torch


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
