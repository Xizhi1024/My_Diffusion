"""Contracts for boundary-reliable subband frequency injection."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _inputs(batch_size: int = 2, size: int = 32):
    from src.model.noise.base import BBDMBridgeSchedule

    return (
        torch.randn(batch_size, 1, size, size),
        torch.randn(batch_size, 1, size, size),
        torch.tensor([999, 700][:batch_size], dtype=torch.long),
        BBDMBridgeSchedule(num_train_timesteps=1000),
        torch.rand(batch_size, 8, size, size),
    )


def _module(**kwargs):
    from src.model.frequency.boundary_reliable import (
        BoundaryReliableFrequencyInjector,
    )

    defaults = {
        "output_channels": (256, 256, 128, 64),
        "band_scales": (0.5, 0.25),
        "use_directional_reliability": False,
    }
    defaults.update(kwargs)
    return BoundaryReliableFrequencyInjector(**defaults)


def test_uses_exactly_two_haar_levels_and_decoder_native_shapes(monkeypatch):
    import src.model.frequency.boundary_reliable as reliable

    residual, ct, timesteps, schedule, _ = _inputs()
    original = reliable.haar_dwt2
    calls = []

    def counted(x):
        calls.append(tuple(x.shape[-2:]))
        return original(x)

    monkeypatch.setattr(reliable, "haar_dwt2", counted)
    module = _module()
    injections, diagnostics = module(residual, timesteps, schedule, ct)

    # Two levels for the noisy state and two for CT; no invented third level.
    assert calls == [(32, 32), (16, 16), (32, 32), (16, 16)]
    assert [tuple(tensor.shape) for tensor in injections] == [
        (2, 256, 4, 4),
        (2, 256, 8, 8),
        (2, 128, 16, 16),
        (2, 64, 32, 32),
    ]
    assert torch.count_nonzero(injections[0]).item() == 0
    assert diagnostics["gates_l2"].shape == (2, 3, 8, 8)
    assert diagnostics["gates_l1"].shape == (2, 3, 16, 16)


def test_l0_is_subband_aligned_inverse_reconstruction_with_zero_lowpass():
    from src.model.frequency.haar import haar_dwt2, haar_idwt2

    image = torch.randn(1, 1, 32, 32)
    module = _module()
    ll1, details1, _ = module.decompose(image)
    reconstructed = module.reconstruct_l0(details1)
    expected = haar_idwt2(torch.zeros_like(ll1), details1)

    assert reconstructed.shape == image.shape
    assert torch.allclose(reconstructed, expected)
    assert not torch.allclose(reconstructed, image)


def test_gates_are_finite_bounded_and_noise_reliability_tracks_bridge_snr():
    residual, ct, _, schedule, _ = _inputs()
    module = _module(gate_max=0.25)

    _, low = module(residual, torch.tensor([999, 999]), schedule, ct)
    _, high = module(residual, torch.tensor([1, 1]), schedule, ct)

    for key in ("gates_l2", "gates_l1"):
        assert torch.isfinite(low[key]).all()
        assert low[key].min() >= 0
        assert low[key].max() <= 0.25
    assert high["noise_reliability"].mean() > low["noise_reliability"].mean()
    assert torch.isfinite(low["gate_tv"])


def test_ct_soft_reliability_uses_energy_not_signed_coefficients():
    module = _module(use_ct_reliability=True)
    residual_details = tuple(torch.randn(1, 1, 8, 8) for _ in range(3))
    ct_details = tuple(torch.randn(1, 1, 8, 8) for _ in range(3))

    positive = module.cross_modal_reliability(residual_details, ct_details)
    sign_flipped = module.cross_modal_reliability(
        residual_details, tuple(-detail for detail in ct_details)
    )

    assert torch.allclose(positive, sign_flipped)
    assert positive.min() >= 0
    assert positive.max() <= 1


def test_ct_reliability_floor_is_monotonic_and_preserves_exact_endpoints():
    module = _module(ct_reliability_floors=(0.25, 0.50))
    raw = torch.tensor([0.0, 0.4, 1.0])

    floored = module.apply_reliability_floor(raw, 0.50)

    assert torch.allclose(floored, torch.tensor([0.50, 0.70, 1.00]))
    assert torch.all(floored[1:] >= floored[:-1])


def test_ct_reliability_uses_separate_native_level_floors():
    residual_details = tuple(torch.randn(1, 1, 8, 8) for _ in range(3))
    ct_details = tuple(torch.randn(1, 1, 8, 8) for _ in range(3))
    raw_module = _module(ct_reliability_floors=(0.0, 0.0))
    floor_module = _module(ct_reliability_floors=(0.25, 0.50))

    raw_l2 = raw_module.cross_modal_reliability(
        residual_details, ct_details, level_index=0
    )
    raw_l1 = raw_module.cross_modal_reliability(
        residual_details, ct_details, level_index=1
    )
    floor_l2 = floor_module.cross_modal_reliability(
        residual_details, ct_details, level_index=0
    )
    floor_l1 = floor_module.cross_modal_reliability(
        residual_details, ct_details, level_index=1
    )

    assert torch.allclose(floor_l2, 0.25 + 0.75 * raw_l2)
    assert torch.allclose(floor_l1, 0.50 + 0.50 * raw_l1)


def test_l1_floor_one_removes_ct_attenuation_without_disabling_l2_ct_gate():
    residual = torch.randn(1, 1, 32, 32)
    ct_a = torch.zeros_like(residual)
    ct_b = torch.randn_like(residual) * 8.0
    from src.model.noise.base import BBDMBridgeSchedule

    module = _module(
        ct_reliability_floors=(0.0, 1.0),
        use_content_reliability=False,
    )
    schedule = BBDMBridgeSchedule(num_train_timesteps=1000)
    timestep = torch.tensor([100])

    _, diag_a = module(residual, timestep, schedule, ct_a)
    _, diag_b = module(residual, timestep, schedule, ct_b)

    assert torch.allclose(diag_a["gates_l1"], diag_b["gates_l1"])
    assert not torch.allclose(diag_a["gates_l2"], diag_b["gates_l2"])


def test_ct_reliability_floor_rejects_invalid_configuration():
    import pytest

    with pytest.raises(ValueError):
        _module(ct_reliability_floors=(0.0,))
    with pytest.raises(ValueError):
        _module(ct_reliability_floors=(-0.1, 0.5))
    with pytest.raises(ValueError):
        _module(ct_reliability_floors=(0.5, 1.1))


def test_shared_mode_has_one_gate_but_subband_mode_can_separate_lh_hl_hh():
    residual = torch.randn(1, 1, 32, 32)
    ct = torch.zeros_like(residual)
    from src.model.noise.base import BBDMBridgeSchedule

    schedule = BBDMBridgeSchedule(num_train_timesteps=1000)
    timestep = torch.tensor([100])
    shared = _module(use_ct_reliability=False, use_subband_gates=False)
    _, shared_diag = shared(residual, timestep, schedule, ct)
    assert torch.allclose(shared_diag["gates_l1"][:, 0], shared_diag["gates_l1"][:, 1])
    assert torch.allclose(shared_diag["gates_l1"][:, 1], shared_diag["gates_l1"][:, 2])

    separate = _module(use_ct_reliability=False, use_subband_gates=True)
    with torch.no_grad():
        separate.content_gates[1].net[-1].bias.copy_(torch.tensor([-4.0, 0.0, 4.0]))
    _, separate_diag = separate(residual, timestep, schedule, ct)
    means = separate_diag["gates_l1"].mean(dim=(0, 2, 3))
    assert means[0] < means[1] < means[2]


def test_content_reliability_can_be_disabled_for_fixed_release_ablation():
    residual = torch.randn(1, 1, 32, 32)
    ct = torch.zeros_like(residual)
    from src.model.noise.base import BBDMBridgeSchedule

    module = _module(
        use_ct_reliability=False,
        use_content_reliability=False,
        use_subband_gates=True,
    )
    with torch.no_grad():
        module.content_gates[1].net[-1].bias.copy_(torch.tensor([-4.0, 0.0, 4.0]))
    _, diagnostics = module(
        residual,
        torch.tensor([100]),
        BBDMBridgeSchedule(num_train_timesteps=1000),
        ct,
    )
    gates = diagnostics["gates_l1"]
    assert torch.allclose(gates[:, 0], gates[:, 1])
    assert torch.allclose(gates[:, 1], gates[:, 2])


def test_gabor_quadrature_energy_only_changes_directional_reliability():
    module = _module(
        use_directional_reliability=True,
        gabor_orientations=8,
    )
    orientation = torch.zeros(1, 8, 32, 32)
    orientation[:, 0] = 4.0
    factors = module.directional_reliability(orientation, (16, 16))

    assert factors.shape == (1, 3, 16, 16)
    assert not torch.allclose(factors[:, 0], factors[:, 1])
    assert not any("gabor" in name and "head" in name for name, _ in module.named_modules())


def test_zero_initialized_skip_residuals_are_exact_noops_but_learnable():
    residual, ct, timesteps, schedule, orientation = _inputs()
    module = _module(
        use_directional_reliability=True,
        gabor_orientations=8,
    )
    injections, _ = module(
        residual,
        timesteps,
        schedule,
        ct,
        gabor_orientation=orientation,
    )

    assert all(torch.count_nonzero(tensor).item() == 0 for tensor in injections)
    loss = sum((tensor - 1.0).square().mean() for tensor in injections[1:])
    loss.backward()
    assert all(head.final.weight.grad is not None for head in module.projection_heads)
