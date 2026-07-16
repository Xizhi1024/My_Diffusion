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
