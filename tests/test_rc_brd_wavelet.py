"""RC-BRD wavelet layer tests (DESIGN §11 / PRD FR-1; [计划] §3.2, §9 M1).

CPU-only, no real data, deterministic (explicit torch.Generator seeds).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.rc_brd.wavelet import (
    BAND_NAMES,
    DETAIL_BANDS,
    band_groups,
    haar_forward2,
    haar_inverse2,
    roundtrip_error,
)


def _random_image(batch: int = 2, size: int = 32, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 1, size, size, generator=gen)


def test_band_name_constants():
    # [计划] §3.2 band layout; detail bands exclude only the LL2 approximation.
    assert BAND_NAMES == ("LL2", "LH2", "HL2", "HH2", "LH1", "HL1", "HH1")
    assert DETAIL_BANDS == ("LH2", "HL2", "HH2", "LH1", "HL1", "HH1")


def test_forward_band_shapes():
    bands = haar_forward2(_random_image(batch=2, size=32, seed=1))
    assert set(bands.keys()) == set(BAND_NAMES)
    assert bands["LL2"].shape == (2, 1, 8, 8)      # level 2: H/4, W/4
    assert bands["HH2"].shape == (2, 1, 8, 8)
    assert bands["LH1"].shape == (2, 1, 16, 16)    # level 1: H/2, W/2
    assert bands["HH1"].shape == (2, 1, 16, 16)


def test_roundtrip_below_fp32_threshold():
    # DESIGN §11: roundtrip < 1e-5 in fp32 ([计划] §9 M1, threshold per dtype).
    for size, seed in ((8, 2), (32, 3), (64, 4)):
        x = _random_image(size=size, seed=seed)
        assert x.dtype == torch.float32
        assert roundtrip_error(x) < 1e-5
        recon = haar_inverse2(haar_forward2(x))
        assert torch.allclose(recon, x, rtol=1e-5, atol=1e-6)


def test_parseval_energy_conservation():
    # Orthonormality of the two-level Haar transform: ‖x‖² == Σ‖band‖².
    x = _random_image(size=32, seed=5)
    bands = haar_forward2(x)
    energy_x = x.to(torch.float64).pow(2).sum()
    energy_bands = sum(b.to(torch.float64).pow(2).sum() for b in bands.values())
    assert torch.allclose(energy_x, energy_bands, rtol=1e-5, atol=1e-6)


def test_band_groups_three():
    # [计划] §3.2 confirmatory compression (C2): low / oriented-mid / high.
    groups = band_groups(3)
    assert groups == {
        "low": ["LL2"],
        "mid": ["LH2", "HL2", "HH2"],
        "high": ["LH1", "HL1", "HH1"],
    }


def test_band_groups_seven():
    groups = band_groups(7)
    assert list(groups.keys()) == list(BAND_NAMES)
    for name in BAND_NAMES:
        assert groups[name] == [name]


def test_band_groups_returns_fresh_dicts():
    a = band_groups(3)
    a["low"].append("LH2")
    assert band_groups(3) == {"low": ["LL2"], "mid": ["LH2", "HL2", "HH2"], "high": ["LH1", "HL1", "HH1"]}


@pytest.mark.parametrize("bad", [4, 0, -3, 2, "3", None, 3.0])
def test_band_groups_invalid_values(bad):
    with pytest.raises(ValueError):
        band_groups(bad)


def test_forward_invalid_shapes_raise():
    with pytest.raises(ValueError):
        haar_forward2(torch.randn(4, 8, 8))            # ndim != 4
    with pytest.raises(ValueError):
        haar_forward2(torch.randn(1, 1, 34, 32))       # H not divisible by 4
    with pytest.raises(ValueError):
        haar_forward2(torch.randn(1, 1, 32, 30))       # W not divisible by 4
    with pytest.raises(ValueError):
        haar_forward2(torch.randn(1, 2, 32, 32))       # channels != 1
    with pytest.raises(ValueError):
        haar_forward2([1, 1, 8, 8])                    # not a tensor


def test_inverse_missing_key_raises():
    bands = haar_forward2(_random_image(seed=6))
    del bands["HL2"]
    with pytest.raises(ValueError):
        haar_inverse2(bands)


def test_inverse_inconsistent_shapes_raise():
    gen = torch.Generator().manual_seed(7)
    bands = haar_forward2(_random_image(batch=2, size=32, seed=8))
    broken = dict(bands)
    broken["LH1"] = torch.randn(2, 1, 15, 15, generator=gen)   # level-1 mismatch
    with pytest.raises(ValueError):
        haar_inverse2(broken)
    broken2 = dict(bands)
    broken2["LL2"] = torch.randn(2, 1, 7, 7, generator=gen)    # level-2 mismatch
    with pytest.raises(ValueError):
        haar_inverse2(broken2)


def test_inverse_level_relation_raises():
    gen = torch.Generator().manual_seed(9)
    bands = haar_forward2(_random_image(batch=2, size=32, seed=10))
    broken = dict(bands)
    for name in ("LH1", "HL1", "HH1"):
        broken[name] = torch.randn(2, 1, 8, 8, generator=gen)  # not 2x level-2 grid
    with pytest.raises(ValueError):
        haar_inverse2(broken)


def test_inverse_wrong_channel_raises():
    bands = haar_forward2(_random_image(seed=11))
    bands["LL2"] = torch.randn(1, 2, 2, 2)
    with pytest.raises(ValueError):
        haar_inverse2(bands)
