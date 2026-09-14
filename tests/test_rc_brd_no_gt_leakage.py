"""RC-BRD no-GT-leakage locks at the head/contract layer (DESIGN §11; PRD FR-4.4).

Scope note: the *complete* inference-time GT-leakage defence (batch keys,
sampler inputs, enabled=false regression) lives at the integration layer and
is asserted by tests/test_rc_brd_integration.py. This file locks the
fail-closed guards that the head and the contract themselves own:

1. v1.0g provenance (AUDIT 5 GT-leak X): BoundedSpecialistHead accepts a
   condition_feat only as a CTFeatureToken issued by its own session; a bare
   tensor derived from GT — even one with exactly the CT stream's shape —
   raises ValueError, because shape checks cannot prove provenance.
2. The token's wrapped tensor must still match the CT feature stream's
   channel count and spatial resolution; full-resolution GT-mask-shaped
   tensors raise ValueError.
3. BoundedSpecialistHead rejects c_effective outside [0, 1] (+tiny float
   tolerance) — contract values are shrunk probabilities, never raw GT stats.
4. RecoverabilityContract.validate rejects GT-derived size_thresholds keys
   (convention: quantities prefixed "gt_" may never enter the contract).

CPU only; no real data; explicit seeds.
"""

from __future__ import annotations

import os
import sys
import warnings

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.rc_brd.contract import ContractViolationError, RecoverabilityContract
from src.model.rc_brd.head import BoundedSpecialistHead, SpecialistConfig

BATCH = 2
FULL_RES = 32  # full image resolution of the residual (GT tensors live here)
CT_STREAM_RES = 16  # finest band resolution == the CT feature stream scale
BAND_SHAPES = {
    "LL2": (BATCH, 1, 8, 8),
    "LH2": (BATCH, 1, 8, 8),
    "HL2": (BATCH, 1, 8, 8),
    "HH2": (BATCH, 1, 8, 8),
    "LH1": (BATCH, 1, 16, 16),
    "HL1": (BATCH, 1, 16, 16),
    "HH1": (BATCH, 1, 16, 16),
}


def make_bands(seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {band: torch.randn(shape, generator=generator) for band, shape in BAND_SHAPES.items()}


def make_head() -> BoundedSpecialistHead:
    return BoundedSpecialistHead(SpecialistConfig())


def make_contract(**overrides) -> RecoverabilityContract:
    payload = {
        "fold_id": "fold_0",
        "band_groups": {"low": ["LL2"], "mid": ["LH2", "HL2", "HH2"],
                        "high": ["LH1", "HL1", "HH1"]},
        "log_snr_grid": [-10.0, 0.0, 10.0],
        "c_values": {"low": [0.2, 0.3, 0.4], "mid": [0.4, 0.5, 0.6],
                     "high": [0.6, 0.7, 0.8]},
        "mean_checkpoint_sha256": "a" * 64,
        "b_active": ["low", "mid", "high"],
        "psd_floors": {"low": 1e-4, "mid": 2e-4, "high": 5e-4},
        "size_thresholds": {"small_lesion_q25": 12.0},
        "kappa_grid": [0.0, 0.5],
        "s_ref": 1.0,
        "support_mode": "floor_gated",
        "eta_max": 0.8,
        "floor_rho": 0.1,
    }
    payload.update(overrides)
    return RecoverabilityContract.from_payload(payload)


# ---------------------------------------------------------------------------
# Guard 1 (v1.0g): condition_feat must be a session-issued CTFeatureToken
# whose wrapped tensor matches the CT-shaped feature stream
# ---------------------------------------------------------------------------

def _gt_mask(seed: int = 3) -> torch.Tensor:
    return (torch.rand(BATCH, 1, FULL_RES, FULL_RES,
                       generator=torch.Generator().manual_seed(seed)) > 0.5).float()


def test_head_rejects_same_shape_gt_derived_bare_tensor():
    """AUDIT 5 X item: a GT-derived tensor projected to *exactly* the CT
    stream's shape [B,64,16,16] still fails — provenance, not shape, is the
    gate; bare tensors are rejected by default."""
    head = make_head()
    bands = make_bands()
    gt_derived = torch.nn.functional.avg_pool2d(_gt_mask(), kernel_size=2)
    gt_derived = gt_derived.repeat(1, 64, 1, 1) * 3.0 - 1.0  # [B,64,16,16]
    assert tuple(gt_derived.shape) == (BATCH, 64, CT_STREAM_RES, CT_STREAM_RES)
    with pytest.raises(ValueError, match="CTFeatureToken"):
        head(bands, gt_derived, None)


def test_head_rejects_any_bare_tensor_even_well_formed_ct_shape():
    head = make_head()
    bands = make_bands()
    ct_shaped = torch.randn(BATCH, 64, CT_STREAM_RES, CT_STREAM_RES,
                            generator=torch.Generator().manual_seed(7))
    with pytest.raises(ValueError, match="CTFeatureToken"):
        head(bands, ct_shaped, None)


def test_head_rejects_token_from_foreign_head_session():
    head, other = make_head(), make_head()
    bands = make_bands()
    ct_stream = torch.randn(BATCH, 64, CT_STREAM_RES, CT_STREAM_RES,
                            generator=torch.Generator().manual_seed(4))
    with pytest.raises(ValueError, match="session"):
        head(bands, other.issue_ct_token(ct_stream), None)


def test_head_rejects_full_resolution_token():
    head = make_head()
    bands = make_bands()
    # a tensor streamed at the full 32x32 image resolution (GT-shaped) instead
    # of the 16x16 finest-band resolution of the CT feature flow — wrapped in
    # a *properly issued* token so the resolution check is what fails
    full_res = torch.randn(BATCH, 64, FULL_RES, FULL_RES)
    with pytest.raises(ValueError, match="resolution"):
        head(bands, head.issue_ct_token(full_res), None)


def test_head_rejects_single_channel_full_resolution_mask_token():
    head = make_head()
    bands = make_bands()
    with pytest.raises(ValueError, match="channels"):
        head(bands, head.issue_ct_token(_gt_mask()), None)


def test_head_accepts_properly_issued_ct_token():
    head = make_head()
    bands = make_bands()
    ct_stream = torch.randn(BATCH, 64, CT_STREAM_RES, CT_STREAM_RES,
                            generator=torch.Generator().manual_seed(4))
    out = head(bands, head.issue_ct_token(ct_stream), None)
    for band, base in bands.items():
        assert torch.equal(out[band], base)  # zero-init identity still holds


def test_escape_hatch_accepts_gt_derived_bare_tensor_with_one_warning():
    """allow_unverified_tokens=True is the explicit calibration/testing escape
    hatch: the same GT-derived tensor is accepted, with exactly one warning."""
    head = BoundedSpecialistHead(SpecialistConfig(), allow_unverified_tokens=True)
    bands = make_bands()
    gt_derived = torch.nn.functional.avg_pool2d(_gt_mask(seed=9), kernel_size=2)
    gt_derived = gt_derived.repeat(1, 64, 1, 1) * 3.0 - 1.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = head(bands, gt_derived, None)
        head(bands, gt_derived, None)  # second call must not warn again
    matched = [w for w in caught if "allow_unverified_tokens" in str(w.message)]
    assert len(matched) == 1
    assert torch.isfinite(out["LH1"]).all()


# ---------------------------------------------------------------------------
# Guard 2: c_effective must be a shrunk in-range probability
# ---------------------------------------------------------------------------

def test_head_rejects_out_of_range_c():
    head = make_head()
    bands = make_bands()
    for bad in (-0.5, 1.5, 2.0):
        c_map = {band: torch.tensor(bad) for band in bands}
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            head(bands, None, c_map)


def test_head_accepts_boundary_c_values():
    head = make_head()
    bands = make_bands()
    for edge in (0.0, 1.0):
        c_map = {band: torch.tensor(edge) for band in bands}
        out = head(bands, None, c_map)
        assert torch.equal(out["LL2"], bands["LL2"])  # c=0/1 still identity at init


# ---------------------------------------------------------------------------
# Guard 3: GT-derived quantities are forbidden inside the contract
# ---------------------------------------------------------------------------

def test_contract_rejects_gt_derived_size_thresholds():
    contract = make_contract(size_thresholds={"gt_small_lesion_q25": 12.0})
    with pytest.raises(ContractViolationError, match="gt_"):
        contract.validate()


def test_contract_accepts_ct_derived_size_thresholds():
    make_contract(size_thresholds={"small_lesion_q25": 12.0}).validate()
