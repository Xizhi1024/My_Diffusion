"""Small dependency-free orthonormal two-dimensional Haar transform."""

from __future__ import annotations

from typing import Tuple

import torch


HaarDetails = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _validate_image(x: torch.Tensor) -> None:
    if x.ndim != 4:
        raise ValueError(f"Haar transform expects [B,C,H,W], got shape {tuple(x.shape)}")
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError(
            f"Haar transform requires even spatial dimensions, got {tuple(x.shape[-2:])}"
        )


def haar_dwt2(x: torch.Tensor) -> tuple[torch.Tensor, HaarDetails]:
    """Apply one level of an orthonormal 2D Haar transform.

    Returns ``(LL, (LH, HL, HH))``. The factor of two makes the transform
    orthonormal in 2D, so energy and coefficient losses remain comparable.
    """
    _validate_image(x)
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]

    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, (lh, hl, hh)


def haar_idwt2(ll: torch.Tensor, details: HaarDetails) -> torch.Tensor:
    """Invert one level of :func:`haar_dwt2`."""
    if len(details) != 3:
        raise ValueError("Haar inverse requires exactly three detail tensors")
    lh, hl, hh = details
    if not (ll.shape == lh.shape == hl.shape == hh.shape):
        raise ValueError("LL/LH/HL/HH coefficient shapes must match")

    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5

    out = ll.new_empty(*ll.shape[:-2], ll.shape[-2] * 2, ll.shape[-1] * 2)
    out[..., 0::2, 0::2] = a
    out[..., 0::2, 1::2] = b
    out[..., 1::2, 0::2] = c
    out[..., 1::2, 1::2] = d
    return out


def reconstruct_lowpass(ll: torch.Tensor, levels: int = 2) -> torch.Tensor:
    """Reconstruct an image from an LL coefficient with all details fixed to zero."""
    if levels < 1:
        raise ValueError(f"levels must be positive, got {levels}")
    x = ll
    for _ in range(levels):
        zeros = tuple(torch.zeros_like(x) for _ in range(3))
        x = haar_idwt2(x, zeros)  # type: ignore[arg-type]
    return x
