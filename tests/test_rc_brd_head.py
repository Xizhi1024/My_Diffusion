"""RC-BRD bounded specialist head tests (DESIGN §11; PRD FR-4; v1.0g AUDIT 5).

Covers: zero-init identity (bitwise torch.equal against base; tanh(0)=0);
gate values bounded in (0, a_max]; the v1.0g absolute correction bound
|output − base| ≤ |c|·a_max·d_max via the tanh magnitude cap (including
saturation with large-Δ inputs); condition_feat=None runs; CT provenance
(CTFeatureToken accepted, bare tensors rejected, escape hatch warns once);
background_delta_energy L1 semantics against hand-computed values; passthrough
for un-covered scales; config and input fail-closed checks (incl. d_max).
CPU only, explicit seeds, fp32 tolerance rtol=1e-5 atol=1e-6.
"""

from __future__ import annotations

import math
import os
import sys
import warnings

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.rc_brd.head import (
    BoundedSpecialistHead,
    CTFeatureToken,
    SpecialistConfig,
    background_delta_energy,
)

BATCH = 2
# two-level Haar layout for a 32x32 residual image: level-2 bands 8x8, level-1 16x16
BAND_SHAPES = {
    "LL2": (BATCH, 1, 8, 8),
    "LH2": (BATCH, 1, 8, 8),
    "HL2": (BATCH, 1, 8, 8),
    "HH2": (BATCH, 1, 8, 8),
    "LH1": (BATCH, 1, 16, 16),
    "HL1": (BATCH, 1, 16, 16),
    "HH1": (BATCH, 1, 16, 16),
}
ALL_BANDS = tuple(BAND_SHAPES)


def make_bands(seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {band: torch.randn(shape, generator=generator) for band, shape in BAND_SHAPES.items()}


def make_condition(seed: int = 1, channels: int = 64, spatial: int = 16) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, channels, spatial, spatial, generator=generator)


def ct_token(head: BoundedSpecialistHead, seed: int = 1, channels: int = 64,
             spatial: int = 16) -> CTFeatureToken:
    """v1.0g: wrap a CT feature tensor in the head-issued provenance token."""
    return head.issue_ct_token(make_condition(seed, channels, spatial))


def make_c(value: float = 0.7) -> dict[str, torch.Tensor]:
    return {band: torch.tensor(value) for band in ALL_BANDS}


def make_head(**config_overrides) -> BoundedSpecialistHead:
    return BoundedSpecialistHead(SpecialistConfig(**config_overrides))


def randomize(head: BoundedSpecialistHead, seed: int = 1234) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in head.parameters():
            param.add_(torch.randn_like(param) * 0.5)


def make_hooked_head(head: BoundedSpecialistHead) -> dict[str, torch.Tensor]:
    """Register hooks capturing each band's Δ output; returns the capture dict."""
    captured: dict[str, torch.Tensor] = {}
    for name, module in head.named_modules():
        if not name.startswith("_deltas."):
            continue
        band = name[len("_deltas."):]
        if "." in band:  # skip nested submodules (in_proj/act/out_conv)
            continue
        module.register_forward_hook(
            lambda mod, inputs, output, band=band: captured.__setitem__(
                band, output.detach().clone()))
    return captured


# ---------------------------------------------------------------------------
# Zero-init identity (FR-4.1)
# ---------------------------------------------------------------------------

def test_forward_identity_at_init_with_condition_and_c():
    head = make_head()
    bands = make_bands()
    out = head(bands, ct_token(head), make_c())
    assert set(out) == set(bands)
    for band in ALL_BANDS:
        assert torch.equal(out[band], bands[band])  # 逐位相等（Δ 零初始化，tanh(0)=0）
        assert out[band].dtype == bands[band].dtype


def test_forward_identity_at_init_without_condition_and_c():
    head = make_head()
    bands = make_bands()
    out = head(bands, None, None)
    for band in ALL_BANDS:
        assert torch.equal(out[band], bands[band])


# ---------------------------------------------------------------------------
# Gates (FR-4.2)
# ---------------------------------------------------------------------------

def test_gate_values_at_init_equal_a_max_times_gate_init():
    head = make_head()
    values = head.gate_values()
    assert set(values) == set(ALL_BANDS)
    for value in values.values():
        assert value == pytest.approx(0.25 * 0.1, abs=1e-6)  # a_max·σ(h)=a_max·gate_init


def test_gate_values_stay_within_bounds_after_random_perturbation():
    head = make_head()
    randomize(head, seed=7)
    for value in head.gate_values().values():
        assert 0.0 < value <= 0.25  # σ(h)∈(0,1) ⇒ a_b∈(0, a_max]
    # near-saturated gate still respects the frozen bound
    params = dict(head.named_parameters())
    with torch.no_grad():
        params["_gates.LL2"].fill_(12.0)  # σ(12)≈0.999994
    assert head.gate_values()["LL2"] <= 0.25
    assert head.gate_values()["LL2"] > 0.249


# ---------------------------------------------------------------------------
# condition_feat=None + output shapes
# ---------------------------------------------------------------------------

def test_condition_none_runs_and_keeps_shapes():
    head = make_head()
    randomize(head, seed=11)  # non-trivial Δ path with zero condition input
    bands = make_bands()
    out = head(bands, None, make_c(0.5))
    assert set(out) == set(bands)
    for band in ALL_BANDS:
        assert out[band].shape == bands[band].shape
        assert torch.isfinite(out[band]).all()


def test_condition_downsampled_for_level2_bands():
    head = make_head()
    randomize(head, seed=13)
    bands = make_bands()
    out = head(bands, ct_token(head, spatial=16), None)
    assert torch.isfinite(out["LL2"]).all()
    assert out["LL2"].shape == bands["LL2"].shape


# ---------------------------------------------------------------------------
# v1.0g bounded increment: ẑ0 = base + c·a_b·d_max·tanh(Δ_b)
# → absolute bound |output − base| ≤ |c|·a_max·d_max (FR-4.2; AUDIT 5 §3.3/§6)
# ---------------------------------------------------------------------------

def test_increment_equals_c_times_gate_times_d_max_times_tanh_delta():
    head = make_head()
    randomize(head, seed=21)
    bands = make_bands()
    captured = make_hooked_head(head)
    c_value = 0.7
    out = head(bands, ct_token(head), make_c(c_value))
    assert set(captured) == set(ALL_BANDS)
    d_max = head.config.d_max
    for band in ALL_BANDS:
        increment = out[band] - bands[band]
        gate = head.gate_values()[band]
        expected = c_value * gate * d_max * torch.tanh(captured[band])
        torch.testing.assert_close(increment, expected, rtol=1e-5, atol=1e-6)


def test_increment_absolutely_bounded_by_c_a_max_d_max():
    """Absolute elementwise bound with saturated tanh (large-Δ inputs)."""
    head = make_head()
    randomize(head, seed=22)
    generator = torch.Generator().manual_seed(220)
    # large-magnitude base bands drive the Δ heads deep into tanh saturation
    bands = {band: torch.randn(shape, generator=generator) * 100.0
             for band, shape in BAND_SHAPES.items()}
    c_value = 0.6
    out = head(bands, ct_token(head), make_c(c_value))
    a_max, d_max = head.config.a_max, head.config.d_max
    bound = a_max * d_max * c_value
    for band in ALL_BANDS:
        increment = (out[band] - bands[band]).abs().detach()
        # 绝对上界 |c|·a_max·d_max holds elementwise even at saturation
        assert (increment <= bound + 1e-6).all()
        # saturation reached: with tanh≈±1 the increment approaches this
        # band's *achievable* max |c|·a_b·d_max (a_b = its current gate)
        achievable = c_value * head.gate_values()[band] * d_max
        assert float(increment.max()) > 0.9 * achievable


def test_increment_bound_holds_without_c_and_random_gates():
    head = make_head()
    randomize(head, seed=24)
    bands = make_bands()
    out = head(bands, ct_token(head), None)  # c_effective=None → c=1 (ceiling)
    bound = head.config.a_max * head.config.d_max
    for band in ALL_BANDS:
        assert ((out[band] - bands[band]).abs() <= bound + 1e-6).all()


def test_default_d_max_and_custom_value():
    assert SpecialistConfig().d_max == 0.10  # v1.0g DESIGN §5 default
    head = make_head(d_max=0.05)
    randomize(head, seed=25)
    bands = make_bands()
    out = head(bands, ct_token(head), make_c(1.0))
    for band in ALL_BANDS:
        assert ((out[band] - bands[band]).abs() <= 0.25 * 0.05 + 1e-6).all()


def test_per_sample_c_vectors_broadcast():
    head = make_head()
    randomize(head, seed=23)
    bands = make_bands()
    c_map = {band: torch.tensor([0.1, 0.9]) for band in ALL_BANDS}
    captured = make_hooked_head(head)
    out = head(bands, ct_token(head), c_map)
    increment = out["LH1"] - bands["LH1"]
    gates = head.gate_values()["LH1"]
    d_max = head.config.d_max
    expected0 = 0.1 * gates * d_max * torch.tanh(captured["LH1"])
    expected1 = 0.9 * gates * d_max * torch.tanh(captured["LH1"])
    torch.testing.assert_close(increment[0:1], expected0[0:1], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(increment[1:2], expected1[1:2], rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# background_delta_energy (FR-4.5, [审计] §5/§8)
# ---------------------------------------------------------------------------

def test_background_delta_energy_l1_semantics_manual():
    delta_low = torch.tensor([[[[1.0, -2.0], [0.0, 3.0]]]])  # [1,1,2,2]
    delta_high = torch.tensor([[[[0.5, 0.0], [-4.0, 1.5]]]])
    soft_mask_inv = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])  # one background-zeroed cell
    # low: |1|+|0(-2 masked)|+|0|+|3| = 4; high: |0.5|+0+|4|+|1.5| = 6 → total 10
    energy = background_delta_energy(
        {"LL2": delta_low, "LH1": delta_high}, soft_mask_inv)
    assert energy.item() == pytest.approx(10.0, abs=1e-6)


def test_background_delta_energy_broadcasts_mask_over_batch():
    delta = torch.tensor([[[[2.0, -2.0]]], [[[4.0, 0.0]]]])  # [2,1,1,2]
    soft_mask_inv = torch.ones(1, 1, 1, 2)
    energy = background_delta_energy({"HH2": delta}, soft_mask_inv)
    assert energy.item() == pytest.approx(2.0 + 2.0 + 4.0 + 0.0, abs=1e-6)


def test_background_delta_energy_resolution_mismatch_fails_closed():
    delta = torch.zeros(1, 1, 4, 4)
    with pytest.raises(ValueError, match="resolution"):
        background_delta_energy({"LL2": delta}, torch.ones(1, 1, 2, 2))
    with pytest.raises(ValueError):
        background_delta_energy({}, torch.ones(1, 1, 4, 4))


# ---------------------------------------------------------------------------
# Passthrough + fail-closed validation
# ---------------------------------------------------------------------------

def test_uncovered_scales_pass_through_untouched():
    head = make_head(apply_scales=(1,))  # specialist only on the level-1 bands
    assert set(head.gate_values()) == {"LH1", "HL1", "HH1"}
    randomize(head, seed=31)
    bands = make_bands()
    c_map = make_c(0.8)
    out = head(bands, ct_token(head), c_map)
    for band in ("LL2", "LH2", "HL2", "HH2"):
        assert torch.equal(out[band], bands[band])  # base 全秩常开，未覆盖带原样返回
    assert not torch.equal(out["LH1"], bands["LH1"])  # specialist band modified


def test_invalid_configs_raise_value_error():
    for overrides in (
        {"a_max": 0.0},
        {"a_max": -0.1},
        {"d_max": 0.0},
        {"d_max": -0.05},
        {"d_max": float("nan")},
        {"d_max": float("inf")},
        {"gate_init": 0.0},
        {"gate_init": 1.0},
        {"channels": 0},
        {"apply_scales": ()},
        {"apply_scales": (3,)},
        {"apply_scales": (1, 1)},
        {"apply_scales": (1.5,)},
    ):
        with pytest.raises(ValueError):
            make_head(**overrides)


def test_band_input_validation():
    head = make_head()
    with pytest.raises(ValueError, match="band"):
        head({}, ct_token(head), None)  # empty dict
    with pytest.raises(ValueError, match="unknown band"):
        head({"XX9": torch.zeros(1, 1, 4, 4)}, ct_token(head), None)
    with pytest.raises(ValueError, match="band"):
        head({"LL2": torch.zeros(2, 3, 8, 8)}, ct_token(head), None)  # band channel ≠ 1
    mismatched = {"LL2": torch.zeros(2, 1, 8, 8), "LH1": torch.zeros(3, 1, 16, 16)}
    with pytest.raises(ValueError, match="batch"):
        head(mismatched, ct_token(head), None)


def test_c_effective_structural_validation():
    head = make_head()
    bands = make_bands()
    with pytest.raises(ValueError, match="unknown band"):
        head(bands, None, {"XX9": torch.tensor(0.5)})
    partial = {band: torch.tensor(0.5) for band in ALL_BANDS if band != "HH1"}
    with pytest.raises(ValueError, match="missing specialist bands"):
        head(bands, None, partial)
    wrong_batch = {band: torch.tensor([0.5, 0.5, 0.5]) for band in ALL_BANDS}
    with pytest.raises(ValueError, match="batch"):
        head(bands, None, wrong_batch)
    matrix_c = {band: torch.zeros(2, 2) for band in ALL_BANDS}
    with pytest.raises(ValueError, match="scalar or"):
        head(bands, None, matrix_c)
    with pytest.raises(ValueError, match="non-finite"):
        head(bands, None, {band: torch.tensor(math.nan) for band in ALL_BANDS})


def test_condition_channel_validation():
    head = make_head()
    bands = make_bands()
    with pytest.raises(ValueError, match="channels"):
        head(bands, ct_token(head, channels=3), None)


# ---------------------------------------------------------------------------
# v1.0g CT provenance (AUDIT 5 GT-leak X; DESIGN §5)
# ---------------------------------------------------------------------------

def test_bare_tensor_condition_rejected_by_default():
    head = make_head()
    bands = make_bands()
    with pytest.raises(ValueError, match="CTFeatureToken"):
        head(bands, make_condition(), None)  # shape checks alone prove nothing


def test_token_from_foreign_head_rejected():
    head_a, head_b = make_head(), make_head()
    bands = make_bands()
    foreign = head_b.issue_ct_token(make_condition())  # right shape, wrong session
    with pytest.raises(ValueError, match="session"):
        head_a(bands, foreign, None)


def test_hand_built_token_without_session_secret_rejected():
    head = make_head()
    bands = make_bands()
    forged = CTFeatureToken(make_condition(), token=object())  # not head-issued
    with pytest.raises(ValueError, match="session"):
        head(bands, forged, None)


def test_ct_feature_token_validates_and_is_immutable():
    with pytest.raises(ValueError, match="CTFeatureToken"):
        CTFeatureToken(torch.zeros(5), token=object())  # not [B,C,h,w]
    with pytest.raises(ValueError, match="token"):
        CTFeatureToken(make_condition(), token=None)
    token = CTFeatureToken(make_condition(), token=object())
    with pytest.raises(AttributeError):
        token.tensor = torch.zeros(1)


def test_allow_unverified_tokens_escape_hatch_warns_once():
    head = BoundedSpecialistHead(SpecialistConfig(), allow_unverified_tokens=True)
    assert head.allow_unverified_tokens is True
    bands = make_bands()
    randomize(head, seed=41)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = head(bands, make_condition(), None)  # accepted, warned once
        out_again = head(bands, make_condition(), None)  # second call: no warn
    messages = [str(w.message) for w in caught
                if "allow_unverified_tokens" in str(w.message)]
    assert len(messages) == 1  # 一次性 warning
    assert torch.isfinite(out["LH1"]).all() and torch.isfinite(out_again["LH1"]).all()


def test_issued_token_runs_and_zero_init_identity_holds():
    head = make_head()
    bands = make_bands()
    out = head(bands, head.issue_ct_token(make_condition()), make_c(0.7))
    for band in ALL_BANDS:
        assert torch.equal(out[band], bands[band])  # token 正常 + init 恒等
