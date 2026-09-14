"""RC-BRD recoverability contract tests (DESIGN §11; PRD FR-2, C4/C5).

Covers: legal construction + to_payload/from_payload/load round trips;
tamper detection on every field (including c values, fold, SHA); fold / mean
SHA mismatch on load; effective_c interpolation / endpoint clamping / eta
shrinkage; floor_gated vs stratified_mixture field constraints; kappa_grid
must contain 0; b_active unknown groups; psd_floors domains; GT-derived
size_thresholds keys. CPU only, no real data, fp32 tolerance rtol=1e-5
atol=1e-6.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import sys
import uuid

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.rc_brd.contract import (
    CONTRACT_SCHEMA_VERSION,
    SUPPORT_MODES,
    ContractViolationError,
    RecoverabilityContract,
    compute_contract_sha256,
    expand_group_to_bands,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture()
def artifact_dir() -> pathlib.Path:
    """Workspace-anchored scratch dir for contract JSON artifacts.

    pytest's tmp_path roots in the system TEMP area, which sandboxed runs may
    be unable to write into; a repo-relative dir keeps the tests runnable in
    both sandboxed and normal environments.
    """
    run_dir = REPO_ROOT / ".t_dir" / "rc_brd_contract_tests" / uuid.uuid4().hex[:12]
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

BAND_GROUPS = {
    "low": ["LL2"],
    "mid": ["LH2", "HL2", "HH2"],
    "high": ["LH1", "HL1", "HH1"],
}
LOG_SNR_GRID = [-10.0, 0.0, 10.0]
MEAN_SHA = "a" * 64


def make_payload(**overrides) -> dict:
    payload = {
        "fold_id": "fold_0",
        "band_groups": {g: list(bands) for g, bands in BAND_GROUPS.items()},
        "log_snr_grid": list(LOG_SNR_GRID),
        "c_values": {
            "low": [0.2, 0.3, 0.4],
            "mid": [0.4, 0.5, 0.6],
            "high": [0.6, 0.7, 0.8],
        },
        "mean_checkpoint_sha256": MEAN_SHA,
        "b_active": ["low", "mid", "high"],
        "psd_floors": {"low": 1e-4, "mid": 2e-4, "high": 5e-4},
        "size_thresholds": {"small_lesion_q25": 12.0},
        "kappa_grid": [0.0, 0.25, 0.5],
        "s_ref": 1.0,
        "support_mode": "floor_gated",
        "eta_max": 0.8,
        "floor_rho": 0.1,
    }
    payload.update(overrides)
    return payload


def make_contract(**overrides) -> RecoverabilityContract:
    return RecoverabilityContract.from_payload(make_payload(**overrides))


# ---------------------------------------------------------------------------
# Legal construction + round trips
# ---------------------------------------------------------------------------

def test_valid_contract_passes_validation():
    contract = make_contract()
    contract.validate()
    assert contract.fold_id == "fold_0"
    assert contract.contract_sha256 == compute_contract_sha256(contract.to_payload())
    assert CONTRACT_SCHEMA_VERSION == 1
    assert SUPPORT_MODES == ("stratified_mixture", "floor_gated")


def test_payload_roundtrip_preserves_fields_and_hash():
    contract = make_contract()
    payload = contract.to_payload()
    assert "contract_sha256" not in payload  # payload excludes the self hash
    rebuilt = RecoverabilityContract.from_payload(payload)
    rebuilt.validate()
    assert rebuilt == contract  # dataclass equality over identical fields
    for field in dataclasses.fields(RecoverabilityContract):
        assert getattr(rebuilt, field.name) == getattr(contract, field.name)


def test_from_payload_is_int_and_hash_stable():
    # int-encoded numbers must coerce to float so canonical hashing is stable
    payload = make_payload(s_ref=1, eta_max=0, kappa_grid=[0, 1])
    contract = RecoverabilityContract.from_payload(payload)
    contract.validate()
    again = RecoverabilityContract.from_payload(contract.to_payload())
    assert again.contract_sha256 == contract.contract_sha256


def test_from_payload_is_creation_path_and_recomputes_hash():
    # v1.0g (DESIGN §3): from_payload keeps the *creation* semantics — a
    # pre-existing contract_sha256 key is tolerated and always overwritten by
    # the recomputed hash. The anti-tamper flip lives in load(), below.
    payload = make_payload()
    payload["contract_sha256"] = "0" * 64  # stale/tampered hash must be recomputed
    contract = RecoverabilityContract.from_payload(payload)
    contract.validate()
    assert contract.contract_sha256 == compute_contract_sha256(contract.to_payload())


def test_load_rejects_stale_stored_hash_legal_domain_tamper(artifact_dir):
    """v1.0g anti-tamper flip (AUDIT 5 §3.3 X): load() must verify the stored
    contract_sha256 against the de-hashed payload *before* trusting it. A
    payload edited after freezing — even when every tampered field stays in
    its legal domain (c changed within [0,1]) — raises ContractViolationError
    because the stored hash no longer matches the recomputed one."""
    contract = make_contract()
    artifact = dict(contract.to_payload())
    artifact["contract_sha256"] = contract.contract_sha256
    # in-domain tamper: change a c value but keep it inside [0,1]
    artifact["c_values"]["low"] = [0.3, 0.3, 0.4]
    path = artifact_dir / "tampered_legal_domain.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ContractViolationError, match="contract_sha256 mismatch"):
        RecoverabilityContract.load(path)


def test_load_rejects_missing_stored_hash(artifact_dir):
    # hash-less artifacts cannot be authenticated -> fail closed (v1.0g)
    path = artifact_dir / "hashless.json"
    path.write_text(json.dumps(make_payload()), encoding="utf-8")
    with pytest.raises(ContractViolationError, match="missing"):
        RecoverabilityContract.load(path)


def test_load_rejects_garbage_stored_hash(artifact_dir):
    contract = make_contract()
    artifact = dict(contract.to_payload())
    artifact["contract_sha256"] = "f" * 64  # well-formed hex, wrong value
    path = artifact_dir / "wrong_hash.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ContractViolationError, match="contract_sha256 mismatch"):
        RecoverabilityContract.load(path)


def test_load_roundtrip(artifact_dir):
    contract = make_contract()
    artifact = dict(contract.to_payload())
    artifact["contract_sha256"] = contract.contract_sha256
    path = artifact_dir / "recoverability_contract_fold0.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    loaded = RecoverabilityContract.load(path)
    loaded.validate()
    assert loaded == contract
    assert RecoverabilityContract.load(path, expected_fold="fold_0",
                                       expected_mean_sha=MEAN_SHA) == contract


def test_load_missing_and_unknown_fields(artifact_dir):
    # artifacts carry a *matching* stored hash so the anti-tamper gate passes
    # and from_payload's missing/unknown-field checks are what fail
    bad = make_payload()
    del bad["s_ref"]
    bad["contract_sha256"] = compute_contract_sha256(bad)
    path = artifact_dir / "missing.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ContractViolationError):
        RecoverabilityContract.load(path)
    bad = make_payload(extra_metadata={"probe": 1})
    bad["contract_sha256"] = compute_contract_sha256(
        {k: v for k, v in bad.items() if k != "contract_sha256"})
    path = artifact_dir / "unknown.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ContractViolationError):
        RecoverabilityContract.load(path)


# ---------------------------------------------------------------------------
# Tamper detection (DESIGN §11: 篡改任一字段 → ContractViolationError)
# ---------------------------------------------------------------------------

TAMPERED_FIELDS = {
    "fold_id": "fold_999",
    "band_groups": {"low": ["LL2"]},
    "log_snr_grid": (-5.0, 0.0, 5.0),
    "c_values": {"low": (0.9, 0.9, 0.9), "mid": (0.4, 0.5, 0.6), "high": (0.6, 0.7, 0.8)},
    "mean_checkpoint_sha256": "b" * 64,
    "b_active": ("low",),
    "psd_floors": {"low": 3e-4, "mid": 2e-4, "high": 5e-4},
    "size_thresholds": {"small_lesion_q25": 99.0},
    "kappa_grid": (0.0, 0.5, 1.0),
    "s_ref": 2.0,
    "support_mode": "stratified_mixture",
    "eta_max": 0.3,
    "floor_rho": 0.2,
}


@pytest.mark.parametrize("field,value", sorted(TAMPERED_FIELDS.items()))
def test_tampering_any_field_breaks_self_hash(field, value):
    contract = make_contract()
    tampered = dataclasses.replace(contract, **{field: value})
    assert tampered.contract_sha256 == contract.contract_sha256  # hash kept stale
    with pytest.raises(ContractViolationError):
        tampered.validate()


def test_tampering_hash_field_breaks_validation():
    contract = make_contract()
    tampered = dataclasses.replace(contract, contract_sha256="c" * 64)
    with pytest.raises(ContractViolationError):
        tampered.validate()


def test_tampered_json_domain_violation_on_load(artifact_dir):
    # attacker rehashes after tampering: the stored-hash check passes, but the
    # domain violation (c=-0.5 outside [0,1]) still fails validate() fail-closed
    payload = make_payload()
    payload["c_values"]["low"] = [-0.5, 0.3, 0.4]
    payload["contract_sha256"] = compute_contract_sha256(payload)
    path = artifact_dir / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractViolationError):
        RecoverabilityContract.load(path)


def test_load_expected_fold_mismatch(artifact_dir):
    contract = make_contract()
    artifact = dict(contract.to_payload())
    artifact["contract_sha256"] = contract.contract_sha256
    path = artifact_dir / "c.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ContractViolationError, match="fold mismatch"):
        RecoverabilityContract.load(path, expected_fold="fold_1")
    with pytest.raises(ContractViolationError, match="mean checkpoint SHA mismatch"):
        RecoverabilityContract.load(path, expected_fold="fold_0",
                                    expected_mean_sha=MEAN_SHA.replace("a", "b"))


def test_load_expected_mean_sha_mismatch(artifact_dir):
    contract = make_contract()
    artifact = dict(contract.to_payload())
    artifact["contract_sha256"] = contract.contract_sha256
    path = artifact_dir / "c.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    other_sha = "f" * 64
    with pytest.raises(ContractViolationError, match="mean checkpoint SHA mismatch"):
        RecoverabilityContract.load(path, expected_fold="fold_0", expected_mean_sha=other_sha)


# ---------------------------------------------------------------------------
# effective_c: interpolation / clamping / shrinkage ([计划] §3.5, [审计] §3.3)
# ---------------------------------------------------------------------------

def test_effective_c_piecewise_linear_interpolation():
    contract = make_contract(eta_max=0.5)
    # c~ = 0.25 + 0.5*c  ⇒  c = 2*c~ - 0.5; group "low" grid c = [0.2, 0.3, 0.4]
    log_snr = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=torch.float32)
    c_tilde = contract.effective_c("low", log_snr)
    recovered = 2.0 * c_tilde.double() - 0.5
    expected = torch.tensor([0.20, 0.25, 0.30, 0.35, 0.40], dtype=torch.float64)
    torch.testing.assert_close(recovered, expected, rtol=1e-5, atol=1e-6)


def test_effective_c_endpoint_clamping_outside_grid():
    contract = make_contract(eta_max=0.8)
    log_snr = torch.tensor([-100.0, 100.0], dtype=torch.float32)
    c_tilde = contract.effective_c("low", log_snr)
    # below grid → c=0.2; above grid → c=0.4; c~ = 0.1 + 0.8*c ([计划] §3.5 钳制)
    expected = torch.tensor([0.1 + 0.8 * 0.2, 0.1 + 0.8 * 0.4], dtype=torch.float64)
    torch.testing.assert_close(c_tilde.double(), expected, rtol=1e-5, atol=1e-6)


def test_effective_c_eta_max_zero_is_constant_half():
    contract = make_contract(eta_max=0.0)
    log_snr = torch.tensor([-100.0, -3.0, 0.0, 4.0, 50.0], dtype=torch.float32)
    c_tilde = contract.effective_c("mid", log_snr)
    assert torch.all(c_tilde == 0.5)  # η_max=0 → c̃ ≡ 0.5 ([审计] §3.3)


def test_effective_c_eta_max_08_shrinkage_formula():
    contract = make_contract(eta_max=0.8)
    for group, grid_c in contract.c_values.items():
        c_tilde = contract.effective_c(group, torch.tensor(LOG_SNR_GRID))
        raw = torch.tensor(grid_c, dtype=torch.float32)
        expected = 0.2 * 0.5 + 0.8 * raw  # c̃ = (1−η)·0.5 + η·c with η=0.8
        torch.testing.assert_close(c_tilde, expected, rtol=1e-5, atol=1e-6)


def test_effective_c_preserves_shape_and_dtype():
    contract = make_contract()
    log_snr = torch.linspace(-12.0, 12.0, 6).reshape(2, 3)
    c_tilde = contract.effective_c("high", log_snr)
    assert c_tilde.shape == log_snr.shape
    assert c_tilde.dtype == log_snr.dtype
    assert torch.isfinite(c_tilde).all()
    assert c_tilde.min() >= 0.0 and c_tilde.max() <= 1.0


def test_effective_c_unknown_group_raises_keyerror():
    contract = make_contract()
    with pytest.raises(KeyError):
        contract.effective_c("ghost", torch.zeros(3))


# ---------------------------------------------------------------------------
# b_active semantics (v1.0g, DESIGN §3; AUDIT 5 triage: inactive = ordinary
# clock with kappa-effect exactly 0)
# ---------------------------------------------------------------------------

def test_effective_c_inactive_group_is_identity_half():
    contract = make_contract(b_active=["mid", "high"])   # "low" inactive
    log_snr = torch.tensor([-100.0, -10.0, -3.0, 0.0, 4.0, 10.0, 100.0])
    c_tilde = contract.effective_c("low", log_snr)
    assert torch.all(c_tilde == 0.5)  # c̃ ≡ 0.5 恒等，无论 λ 落在网格何处
    assert c_tilde.shape == log_snr.shape
    assert c_tilde.dtype == log_snr.dtype
    # active groups still run the full interpolate/clip/shrink pipeline
    active = contract.effective_c("mid", torch.tensor(LOG_SNR_GRID))
    expected = 0.2 * 0.5 + 0.8 * torch.tensor([0.4, 0.5, 0.6])
    torch.testing.assert_close(active, expected, rtol=1e-5, atol=1e-6)


def test_effective_c_inactive_group_kappa_effect_is_zero():
    contract = make_contract(b_active=["mid", "high"], kappa_grid=[0.0, 0.5])
    kappa = 0.5
    log_snr = torch.linspace(-12.0, 12.0, 9)
    for group in contract.band_groups:
        c_tilde = contract.effective_c(group, log_snr)
        rho = torch.exp(kappa * (2.0 * c_tilde.double() - 1.0))  # [审计] §3.1
        if group in contract.b_active:
            assert float((rho - 1.0).abs().max()) > 1e-3  # genuinely warped
        else:
            torch.testing.assert_close(rho, torch.ones_like(rho), rtol=0, atol=0)
            # ρ ≡ 1 ⇒ d m/du = 1: the ordinary clock, κ-effect exactly 0


def test_effective_c_empty_b_active_all_identity():
    contract = make_contract(b_active=[])
    log_snr = torch.linspace(-10.0, 10.0, 5)
    for group in contract.band_groups:
        assert torch.all(contract.effective_c(group, log_snr) == 0.5)


# ---------------------------------------------------------------------------
# Finite-domain locks (v1.0g, AUDIT 5 κ/λ/σ triage): NaN/inf in any float
# field → ContractViolationError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field,bad_value", [
    ("s_ref", float("nan")),
    ("s_ref", float("inf")),
    ("eta_max", float("nan")),
    ("eta_max", float("inf")),
    ("floor_rho", float("nan")),
    ("floor_rho", float("-inf")),
    ("log_snr_grid", [-10.0, float("nan"), 10.0]),
    ("log_snr_grid", [-10.0, float("inf"), 10.0]),
    ("kappa_grid", [0.0, float("nan")]),
    ("kappa_grid", [0.0, float("inf")]),
])
def test_nonfinite_scalar_fields_fail_closed(field, bad_value):
    contract = make_contract(**{field: bad_value})
    with pytest.raises(ContractViolationError):
        contract.validate()


@pytest.mark.parametrize("field,bad_value", [
    ("c_values", {"low": [0.2, float("nan"), 0.4], "mid": [0.4, 0.5, 0.6],
                  "high": [0.6, 0.7, 0.8]}),
    ("c_values", {"low": [0.2, float("inf"), 0.4], "mid": [0.4, 0.5, 0.6],
                  "high": [0.6, 0.7, 0.8]}),
])
def test_nonfinite_c_values_fail_closed(field, bad_value):
    payload = make_payload()
    payload[field] = bad_value
    contract = RecoverabilityContract.from_payload(payload)
    with pytest.raises(ContractViolationError):
        contract.validate()


@pytest.mark.parametrize("bad_floor", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_psd_floors_fail_closed(bad_floor):
    floors = dict(make_payload()["psd_floors"])
    floors["mid"] = bad_floor
    contract = make_contract(psd_floors=floors)
    with pytest.raises(ContractViolationError):
        contract.validate()


# ---------------------------------------------------------------------------
# Field domain constraints ([审计] §3.3/§5; PRD C4/C5)
# ---------------------------------------------------------------------------

def test_support_mode_must_be_known():
    with pytest.raises(ContractViolationError):
        make_contract(support_mode="banana").validate()
    assert set(SUPPORT_MODES) == {"stratified_mixture", "floor_gated"}


def test_floor_gated_rho_domain():
    for bad_rho in (0.0, 1.0, -0.1, 1.5):
        contract = make_contract(support_mode="floor_gated", floor_rho=bad_rho)
        with pytest.raises(ContractViolationError):
            contract.validate()
    make_contract(support_mode="floor_gated", floor_rho=0.1).validate()


def test_stratified_mixture_ignores_floor_rho():
    # floor_rho is semantically unused in stratified_mixture (DESIGN §3)
    make_contract(support_mode="stratified_mixture", floor_rho=0.0).validate()


def test_eta_max_domain():
    for bad in (-0.1, 1.0, 1.5):
        contract = make_contract(eta_max=bad)
        with pytest.raises(ContractViolationError):
            contract.validate()
    make_contract(eta_max=0.0).validate()
    make_contract(eta_max=0.99).validate()


def test_kappa_grid_must_contain_zero():
    contract = make_contract(kappa_grid=[0.1, 0.5])
    with pytest.raises(ContractViolationError, match="kappa"):
        contract.validate()
    make_contract(kappa_grid=[0.0]).validate()


def test_b_active_unknown_group_raises():
    contract = make_contract(b_active=["low", "ghost"])
    with pytest.raises(ContractViolationError):
        contract.validate()
    make_contract(b_active=[]).validate()  # empty subset is legal ([DESIGN] §3 subset semantics)


def test_psd_floors_must_be_positive():
    for bad in (0.0, -1e-3):
        floors = dict(make_payload()["psd_floors"])
        floors["mid"] = bad
        contract = make_contract(psd_floors=floors)
        with pytest.raises(ContractViolationError):
            contract.validate()


def test_c_values_must_be_clipped_unit_range():
    payload = make_payload()
    payload["c_values"]["high"] = [0.6, 0.7, 1.4]
    contract = RecoverabilityContract.from_payload(payload)
    with pytest.raises(ContractViolationError):
        contract.validate()


def test_log_snr_grid_must_be_increasing():
    contract = make_contract(log_snr_grid=[0.0, -10.0, 10.0])
    with pytest.raises(ContractViolationError):
        contract.validate()


def test_mean_sha_must_be_hex64():
    for bad in ("xyz", "A" * 64, "a" * 63):
        contract = make_contract(mean_checkpoint_sha256=bad)
        with pytest.raises(ContractViolationError):
            contract.validate()


def test_size_thresholds_gt_prefix_forbidden():
    thresholds = {"gt_small_lesion_q25": 12.0}  # GT-derived quantity in the contract
    contract = make_contract(size_thresholds=thresholds)
    with pytest.raises(ContractViolationError, match="gt_"):
        contract.validate()


# ---------------------------------------------------------------------------
# expand_group_to_bands + compute_contract_sha256
# ---------------------------------------------------------------------------

def test_expand_group_to_bands_shares_group_reference():
    c_groups = {
        "low": torch.tensor(0.2),
        "mid": torch.tensor(0.5),
        "high": torch.tensor(0.8),
    }
    expanded = expand_group_to_bands(c_groups, BAND_GROUPS)
    assert set(expanded) == {"LL2", "LH2", "HL2", "HH2", "LH1", "HL1", "HH1"}
    assert expanded["LH2"] is c_groups["mid"]
    assert expanded["HL2"] is c_groups["mid"]  # 组内共享同一 Tensor 引用
    assert expanded["LL2"] is c_groups["low"]
    assert expanded["HH1"] is c_groups["high"]
    assert "XX1" not in expanded  # 未知带（不在任何组）不出现在输出


def test_expand_group_to_bands_missing_group_fails_closed():
    with pytest.raises(KeyError):
        expand_group_to_bands({"low": torch.tensor(0.2)}, BAND_GROUPS)


def test_compute_contract_sha256_is_canonical():
    payload = make_payload()
    assert compute_contract_sha256(payload) == compute_contract_sha256(
        dict(reversed(list(payload.items()))))  # key-order independent
    assert compute_contract_sha256(payload) == compute_contract_sha256(
        {"log_snr_grid": tuple(payload["log_snr_grid"]), **{
            k: v for k, v in payload.items() if k != "log_snr_grid"}})  # tuple == list
    changed = make_payload(s_ref=2.0)
    assert compute_contract_sha256(payload) != compute_contract_sha256(changed)
