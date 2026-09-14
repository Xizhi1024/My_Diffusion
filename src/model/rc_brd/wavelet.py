"""Two-level orthonormal Haar band utilities for RC-BRD.

Math source: [计划] §3.2 (orthogonal residual coordinates, band layout and
the 3/7 band-group decomposition); interface contract: DESIGN §2
(docs/prd/DESIGN_RC_BRD_v1.md).  The one-level orthonormal primitives are
reused from ``src/model/frequency/haar.py`` (DESIGN §2: reuse, do not
reimplement the convolutions).
"""

from __future__ import annotations

import torch

from ..frequency.haar import haar_dwt2, haar_idwt2

# [计划] §3.2: two-level Haar band layout (level-2 bands first, then level-1).
BAND_NAMES: tuple[str, ...] = ("LL2", "LH2", "HL2", "HH2", "LH1", "HL1", "HH1")
# Detail bands: BAND_NAMES without the level-2 approximation band "LL2".
DETAIL_BANDS: tuple[str, ...] = tuple(name for name in BAND_NAMES if name != "LL2")

_LEVEL2_BANDS: tuple[str, ...] = ("LL2", "LH2", "HL2", "HH2")
_LEVEL1_DETAIL_BANDS: tuple[str, ...] = ("LH1", "HL1", "HH1")
# Residual images are single-channel [B,1,H,W] (DESIGN §2).
_IMAGE_CHANNEL_COUNT = 1


def _validate_image_tensor(x: torch.Tensor) -> None:
    """Raise ValueError unless ``x`` is a [B,1,H,W] tensor with H, W divisible by 4."""
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"expected a torch.Tensor, got {type(x)!r}")
    if x.ndim != 4:
        raise ValueError(f"haar_forward2 expects [B,1,H,W], got shape {tuple(x.shape)}")
    if x.shape[1] != _IMAGE_CHANNEL_COUNT:
        raise ValueError(
            f"haar_forward2 expects {_IMAGE_CHANNEL_COUNT} image channel(s), "
            f"got channels={x.shape[1]}"
        )
    height, width = int(x.shape[-2]), int(x.shape[-1])
    if height % 4 or width % 4:
        raise ValueError(
            f"haar_forward2 requires H and W divisible by 4 (two-level Haar), "
            f"got (H, W)=({height}, {width})"
        )


def haar_forward2(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Two-level orthonormal Haar analysis of a residual image.

    [B,1,H,W] (H, W divisible by 4) → {band: [B,1,h,w]} with the band layout
    `BAND_NAMES`: level-2 bands have h=H/4, w=W/4; level-1 detail bands have
    h=H/2, w=W/2.  The transform is orthonormal (Parseval holds: ‖x‖² equals
    the sum of band energies).  Illegal shapes raise ValueError.
    Math source: [计划] §3.2; primitives from src/model/frequency/haar.py.
    """
    _validate_image_tensor(x)
    # Level 1 (finest), then level 2 applied to the low-pass channel.
    ll1, (lh1, hl1, hh1) = haar_dwt2(x)
    ll2, (lh2, hl2, hh2) = haar_dwt2(ll1)
    return {
        "LL2": ll2,
        "LH2": lh2,
        "HL2": hl2,
        "HH2": hh2,
        "LH1": lh1,
        "HL1": hl1,
        "HH1": hh1,
    }


def _validated_band_tensors(bands: dict) -> dict[str, torch.Tensor]:
    """Validate band keys/shapes and return the 7 canonical band tensors."""
    if not isinstance(bands, dict):
        raise ValueError(
            f"haar_inverse2 expects a dict[str, torch.Tensor], got {type(bands)!r}"
        )
    missing = [name for name in BAND_NAMES if name not in bands]
    if missing:
        raise ValueError(f"haar_inverse2: missing band keys {missing}")
    checked: dict[str, torch.Tensor] = {}
    for name in BAND_NAMES:
        band = bands[name]
        if not isinstance(band, torch.Tensor) or band.ndim != 4:
            raise ValueError(
                f"haar_inverse2: band {name!r} must be a [B,1,h,w] tensor, "
                f"got {type(band).__name__} with shape {getattr(band, 'shape', None)}"
            )
        if band.shape[1] != _IMAGE_CHANNEL_COUNT:
            raise ValueError(
                f"haar_inverse2: band {name!r} must have 1 channel, got {band.shape[1]}"
            )
        checked[name] = band
    ref_level2 = tuple(checked["LL2"].shape)
    for name in _LEVEL2_BANDS:
        if tuple(checked[name].shape) != ref_level2:
            raise ValueError(
                f"haar_inverse2: level-2 band {name!r} shape {tuple(checked[name].shape)} "
                f"does not match LL2 shape {ref_level2}"
            )
    ref_level1 = tuple(checked["LH1"].shape)
    for name in _LEVEL1_DETAIL_BANDS:
        if tuple(checked[name].shape) != ref_level1:
            raise ValueError(
                f"haar_inverse2: level-1 band {name!r} shape {tuple(checked[name].shape)} "
                f"does not match LH1 shape {ref_level1}"
            )
    if ref_level1[:2] != ref_level2[:2]:
        raise ValueError(
            f"haar_inverse2: batch/channel dims disagree between levels "
            f"({ref_level1[:2]} vs {ref_level2[:2]})"
        )
    if ref_level1[-2] != 2 * ref_level2[-2] or ref_level1[-1] != 2 * ref_level2[-1]:
        raise ValueError(
            "haar_inverse2: level-1 spatial dims must be twice the level-2 dims, "
            f"got {ref_level1[-2:]} vs {ref_level2[-2:]}"
        )
    return checked


def haar_inverse2(bands: dict[str, torch.Tensor]) -> torch.Tensor:
    """Two-level orthonormal Haar synthesis.

    Band dict (`BAND_NAMES` keys) → [B,1,H,W].  Missing keys, non-tensor
    entries, or inconsistent shapes raise ValueError.  Exact inverse of
    :func:`haar_forward2` up to floating-point roundoff.
    Math source: [计划] §3.2.
    """
    checked = _validated_band_tensors(bands)
    ll1 = haar_idwt2(
        checked["LL2"], (checked["LH2"], checked["HL2"], checked["HH2"])
    )
    x = haar_idwt2(ll1, (checked["LH1"], checked["HL1"], checked["HH1"]))
    return x


def band_groups(n_groups: int) -> dict[str, list[str]]:
    """Band-group decomposition used by the confirmatory contract (C2).

    n_groups=3 → {"low": ["LL2"], "mid": ["LH2", "HL2", "HH2"],
    "high": ["LH1", "HL1", "HH1"]} ([计划] §3.2 pre-registered compression,
    [审计] §1.1); n_groups=7 → one group per band (group name = band name).
    Any other value raises ValueError.  A fresh dict is returned per call.
    """
    if isinstance(n_groups, bool) or not isinstance(n_groups, int):
        raise ValueError(f"n_groups must be an int (3 or 7), got {n_groups!r}")
    if n_groups == 3:
        return {
            "low": ["LL2"],
            "mid": ["LH2", "HL2", "HH2"],
            "high": ["LH1", "HL1", "HH1"],
        }
    if n_groups == 7:
        return {name: [name] for name in BAND_NAMES}
    raise ValueError(f"n_groups must be 3 or 7, got {n_groups}")


def roundtrip_error(x: torch.Tensor) -> float:
    """Max absolute error of the analysis→synthesis round trip.

    Returns max |haar_inverse2(haar_forward2(x)) − x| as a Python float
    ([计划] §9 M1 smoke check; fp32 threshold 1e-5 per DESIGN §11).
    """
    return float(haar_inverse2(haar_forward2(x)).sub(x).abs().max().item())
