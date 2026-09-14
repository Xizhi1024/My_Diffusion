"""RC-BRD recoverability contract (FR-2, DESIGN §3).

A frozen, fold-specific artifact that pins the CT-conditioned recoverability
evidence measured by the R0 probe ([计划] §3.4-3.5). The contract is a
checkpoint deployment dependency: schema, fold identity, mean-checkpoint SHA
and self hash are all validated fail-closed before any RC-BRD path runs
(FR-2.2/FR-2.4; [计划] §3.7 "合同是 checkpoint 的部署依赖").

Math sources
------------
- piecewise-linear c interpolation + endpoint clamping + artifact fields:
  [计划] §3.5
- shrinkage c~ = (1-eta)*0.5 + eta*c with eta = eta_max < 1: [审计] §3.3
- support modes (stratified_mixture vs floor_gated with floor rho): [审计] §5,
  PRD §3 C4/C5
- v1.0g (AUDIT 5 §3.3): load() is anti-tamper (the stored self hash is
  verified against the de-hashed payload *before* the payload is trusted);
  groups outside b_active get the identity c~=0.5 (ordinary clock,
  kappa-effect 0); every float field is checked isfinite.

Import policy (DESIGN §1): stdlib + torch only. This module must not import
wavelet/schedule/head from the rc_brd package - the contract side of the
package is independent of the wavelet<-schedule<-head chain.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Mapping

import torch

SUPPORT_MODES = ("stratified_mixture", "floor_gated")  # C4 ([审计] §5)
CONTRACT_SCHEMA_VERSION = 1

# GT-derived quantities are forbidden inside the contract (inference must only
# read CT evidence; [计划] §3.7/§4). Locked by test_rc_brd_no_gt_leakage.py.
_GT_DERIVED_KEY_PREFIX = "gt_"
_C0 = 0.5  # [审计] §3.3: c0=0.5 corresponds to the ordinary (uncontracted) clock


class ContractViolationError(RuntimeError):
    """Raised when a recoverability contract fails fail-closed validation."""


def compute_contract_sha256(payload: Mapping) -> str:
    """Return the sha256 hex of the canonical JSON serialization of payload.

    Canonical form is json.dumps(payload, sort_keys=True,
    separators=(",", ":")) (DESIGN §3). Tuples and lists serialize
    identically, so the hash is stable across payload round trips.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Small pure validators (shared by RecoverabilityContract.validate)
# ---------------------------------------------------------------------------

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractViolationError(message)


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_hex64(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _checked_group_map(band_groups: object) -> list[str]:
    """Validate the group->bands mapping; return all bands ([计划] §3.2)."""
    _require(isinstance(band_groups, Mapping) and len(band_groups) > 0,
             "band_groups must be a non-empty mapping of group name -> band list")
    all_bands: list[str] = []
    for group, band_list in band_groups.items():
        _require(isinstance(group, str) and group, "band_groups keys must be non-empty strings")
        _require(isinstance(band_list, (list, tuple)) and len(band_list) > 0,
                 f"band_groups[{group!r}] must be a non-empty list of band names")
        for band in band_list:
            _require(isinstance(band, str) and band,
                     f"band_groups[{group!r}] entries must be non-empty strings")
            all_bands.append(band)
    _require(len(set(all_bands)) == len(all_bands),
             "each band may appear in at most one band_groups entry")
    return all_bands


def _checked_float_seq(values: object, name: str) -> list[float]:
    _require(isinstance(values, (list, tuple)), f"{name} must be a list/tuple of floats")
    for value in values:
        _require(_is_finite_number(value), f"{name} entries must be finite numbers, got {value!r}")
    return [float(value) for value in values]


def _checked_float_mapping(mapping: object, name: str, *, expected_keys: set | None = None) -> None:
    _require(isinstance(mapping, Mapping), f"{name} must be a mapping")
    _require(len(mapping) > 0, f"{name} must be non-empty")
    for key, value in mapping.items():
        _require(isinstance(key, str) and key, f"{name} keys must be non-empty strings")
        _require(_is_finite_number(value), f"{name}[{key!r}] must be a finite number")
    if expected_keys is not None:
        _require(set(mapping) == expected_keys,
                 f"{name} keys must equal band_groups keys; "
                 f"missing={sorted(expected_keys - set(mapping))} extra={sorted(set(mapping) - expected_keys)}")


def _interp_clamped(x: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation with endpoint clamping (np.interp semantics).

    [计划] §3.5: 区间内分段线性插值；区间外钳制到最近端点。xs must be
    strictly increasing (guaranteed by validate()); computed in float64.
    """
    n = xs.numel()
    if n == 1:
        return ys[0].expand_as(x).clone()
    right = torch.searchsorted(xs, x).clamp(1, n - 1)
    x0 = xs[right - 1]
    x1 = xs[right]
    y0 = ys[right - 1]
    y1 = ys[right]
    t = ((x - x0) / (x1 - x0)).clamp(0.0, 1.0)
    return y0 + t * (y1 - y0)


@dataclass(frozen=True)
class RecoverabilityContract:
    """Frozen recoverability contract (DESIGN §3; [计划] §3.5, [审计] §3/§5)."""

    fold_id: str
    band_groups: dict[str, list[str]]        # 组名→带名列表（来自 wavelet.band_groups）
    log_snr_grid: tuple[float, ...]          # 升序 Λ
    c_values: dict[str, tuple[float, ...]]   # 组名→网格点 c 值（已 clip 到 [0,1]）
    mean_checkpoint_sha256: str              # 64 位 hex
    b_active: tuple[str, ...]                # 生效组名子集 ⊆ band_groups.keys()
    psd_floors: dict[str, float]             # 组名→PSD/noise floor（>0）
    size_thresholds: dict[str, float]        # 2D 尺寸层阈值（small-lesion q25 等）
    kappa_grid: tuple[float, ...]            # 预注册 κ 候选（必含 0.0）
    s_ref: float                             # >0
    support_mode: str                        # SUPPORT_MODES 之一
    eta_max: float                           # C5，∈[0,1)
    floor_rho: float                         # C4 floor_gated 时 ∈(0,1)；stratified_mixture 时忽略
    contract_sha256: str                     # 载荷自哈希（canonical JSON, sha256, 不含本字段自身）

    def validate(self) -> None:
        """Fail-closed validation (DESIGN §3 / FR-2.2).

        Checks field presence/types, value domains, b_active subset, kappa
        grid containing 0.0, and that contract_sha256 matches the recomputed
        canonical payload hash. Any violation raises ContractViolationError.

        v1.0g finite-domain lock (AUDIT 5 κ/λ/σ triage): every float field is
        required to be isfinite — s_ref / eta_max / floor_rho via
        _is_finite_number, log_snr_grid / c_values / kappa_grid entries via
        _checked_float_seq, psd_floors / size_thresholds values via
        _checked_float_mapping. NaN or +/-inf raises ContractViolationError.
        """
        self._validate_layout()
        self._validate_domains()
        expected = compute_contract_sha256(self.to_payload())
        _require(_is_hex64(self.contract_sha256),
                 "contract_sha256 must be 64-char lowercase hex")
        _require(self.contract_sha256 == expected,
                 f"contract_sha256 mismatch: stored {self.contract_sha256!r} != "
                 f"recomputed {expected!r} (payload tampered or hash stale)")

    def _validate_layout(self) -> None:
        _require(isinstance(self.fold_id, str) and self.fold_id,
                 "fold_id must be a non-empty string")
        group_keys = set(self.band_groups)
        _checked_group_map(self.band_groups)
        grid = _checked_float_seq(self.log_snr_grid, "log_snr_grid")
        _require(len(grid) > 0, "log_snr_grid must be non-empty")
        _require(all(grid[i] < grid[i + 1] for i in range(len(grid) - 1)),
                 "log_snr_grid must be strictly increasing")
        _require(isinstance(self.c_values, Mapping) and set(self.c_values) == group_keys,
                 "c_values keys must equal band_groups keys")
        for group, values in self.c_values.items():
            got = _checked_float_seq(values, f"c_values[{group!r}]")
            _require(len(got) == len(grid),
                     f"c_values[{group!r}] must have exactly len(log_snr_grid)={len(grid)} entries")
        _require(_is_hex64(self.mean_checkpoint_sha256),
                 "mean_checkpoint_sha256 must be 64-char lowercase hex")
        _require(isinstance(self.b_active, (list, tuple)),
                 "b_active must be a tuple of group names")
        _require(all(isinstance(g, str) for g in self.b_active),
                 "b_active entries must be strings")
        _require(len(set(self.b_active)) == len(self.b_active),
                 "b_active must not repeat group names")
        _require(set(self.b_active) <= group_keys,
                 f"b_active must be a subset of band_groups keys; unknown: "
                 f"{sorted(set(self.b_active) - group_keys)}")
        _checked_float_mapping(self.psd_floors, "psd_floors", expected_keys=group_keys)
        _checked_float_mapping(self.size_thresholds, "size_thresholds")
        kappa = _checked_float_seq(self.kappa_grid, "kappa_grid")
        _require(len(kappa) > 0 and 0.0 in kappa,
                 "kappa_grid must contain 0.0 (D1 strict fallback, DESIGN §3)")

    def _validate_domains(self) -> None:
        for group, values in self.c_values.items():
            for value in values:
                _require(0.0 <= float(value) <= 1.0,
                         f"c_values[{group!r}] entries must lie in [0,1] ([计划] §3.5 clip)")
        for group, value in self.psd_floors.items():
            _require(float(value) > 0.0, f"psd_floors[{group!r}] must be > 0")
        for key, value in self.size_thresholds.items():
            _require(float(value) > 0.0, f"size_thresholds[{key!r}] must be > 0")
            _require(not key.startswith(_GT_DERIVED_KEY_PREFIX),
                     f"size_thresholds key {key!r} starts with {_GT_DERIVED_KEY_PREFIX!r}: "
                     "GT-derived quantities are forbidden inside the contract")
        _require(_is_finite_number(self.s_ref) and float(self.s_ref) > 0.0,
                 "s_ref must be > 0")
        _require(isinstance(self.support_mode, str) and self.support_mode in SUPPORT_MODES,
                 f"support_mode must be one of {SUPPORT_MODES}, got {self.support_mode!r}")
        _require(_is_finite_number(self.eta_max) and 0.0 <= float(self.eta_max) < 1.0,
                 "eta_max must lie in [0,1) ([审计] §3.3)")
        _require(_is_finite_number(self.floor_rho), "floor_rho must be a finite float")
        if self.support_mode == "floor_gated":
            _require(0.0 < float(self.floor_rho) < 1.0,
                     "floor_rho must lie in (0,1) when support_mode='floor_gated' "
                     "([审计] §5: non-zero rho keeps the CT-invisible channel)")

    def effective_c(self, group: str, log_snr: torch.Tensor) -> torch.Tensor:
        """Effective contract value c~ for one group at the given log-SNR.

        Pipeline ([计划] §3.5 + [审计] §3.3): piecewise-linear interpolation on
        log_snr_grid -> clamp to [0,1] -> shrinkage c~ = (1-eta)*0.5 + eta*c
        with eta = eta_max. Computed in float64, cast back to the input dtype.
        Returns a tensor with the same shape as log_snr. Unknown groups raise
        KeyError.

        v1.0g (DESIGN §3; AUDIT 5 b_active triage): a group that is *not* in
        b_active runs the ordinary clock — effective_c returns the constant
        identity value c0 = 0.5, so rho = exp{kappa*(2*0.5-1)} = 1 and the
        kappa-effect is exactly zero for that group.
        """
        if group not in self.c_values:
            raise KeyError(
                f"unknown contract group {group!r}; known groups: {sorted(self.c_values)}")
        x = torch.as_tensor(log_snr)
        if group not in set(self.b_active):
            out_dtype = x.dtype if x.is_floating_point() else torch.float32
            return torch.full(x.shape, _C0, dtype=out_dtype, device=x.device)
        out_dtype = x.dtype if x.is_floating_point() else torch.float32
        shape = x.shape
        xs = torch.tensor([float(v) for v in self.log_snr_grid], dtype=torch.float64)
        ys = torch.tensor([float(v) for v in self.c_values[group]], dtype=torch.float64)
        c = _interp_clamped(x.to(dtype=torch.float64).reshape(-1), xs, ys)
        c = c.clamp(0.0, 1.0)  # [计划] §3.5: clip 到 [0,1]
        eta = float(self.eta_max)
        c_tilde = (1.0 - eta) * _C0 + eta * c  # [审计] §3.3: 收缩合同
        return c_tilde.reshape(shape).to(dtype=out_dtype)

    def to_payload(self) -> dict:
        """Canonical JSON-native payload dict, without contract_sha256 (DESIGN §3)."""
        return {
            "fold_id": str(self.fold_id),
            "band_groups": {
                str(g): [str(b) for b in bands] for g, bands in self.band_groups.items()
            },
            "log_snr_grid": [float(v) for v in self.log_snr_grid],
            "c_values": {
                str(g): [float(v) for v in values] for g, values in self.c_values.items()
            },
            "mean_checkpoint_sha256": str(self.mean_checkpoint_sha256),
            "b_active": [str(g) for g in self.b_active],
            "psd_floors": {str(g): float(v) for g, v in self.psd_floors.items()},
            "size_thresholds": {str(k): float(v) for k, v in self.size_thresholds.items()},
            "kappa_grid": [float(v) for v in self.kappa_grid],
            "s_ref": float(self.s_ref),
            "support_mode": str(self.support_mode),
            "eta_max": float(self.eta_max),
            "floor_rho": float(self.floor_rho),
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "RecoverabilityContract":
        """Build a contract from a canonical payload; recompute the self hash.

        Numeric fields are coerced to float so that to_payload()/from_payload()
        round trips are hash-stable regardless of int/float JSON encoding. A
        pre-existing "contract_sha256" key is tolerated and always overwritten
        by the recomputed hash (DESIGN §3: 重算并填 contract_sha256). This
        constructor does not validate; call validate() (or load()) to verify.
        """
        if not isinstance(payload, Mapping):
            raise ContractViolationError("contract payload must be a mapping")
        required = {
            "fold_id", "band_groups", "log_snr_grid", "c_values",
            "mean_checkpoint_sha256", "b_active", "psd_floors", "size_thresholds",
            "kappa_grid", "s_ref", "support_mode", "eta_max", "floor_rho",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ContractViolationError(f"payload is missing fields: {missing}")
        unknown = sorted(set(payload) - required - {"contract_sha256"})
        if unknown:
            raise ContractViolationError(f"payload has unknown fields: {unknown}")
        try:
            contract = cls(
                fold_id=payload["fold_id"],
                band_groups={g: list(bands) for g, bands in payload["band_groups"].items()},
                log_snr_grid=tuple(float(v) for v in payload["log_snr_grid"]),
                c_values={g: tuple(float(v) for v in vals)
                          for g, vals in payload["c_values"].items()},
                mean_checkpoint_sha256=payload["mean_checkpoint_sha256"],
                b_active=tuple(payload["b_active"]),
                psd_floors={g: float(v) for g, v in payload["psd_floors"].items()},
                size_thresholds={k: float(v) for k, v in payload["size_thresholds"].items()},
                kappa_grid=tuple(float(v) for v in payload["kappa_grid"]),
                s_ref=float(payload["s_ref"]),
                support_mode=payload["support_mode"],
                eta_max=float(payload["eta_max"]),
                floor_rho=float(payload["floor_rho"]),
                contract_sha256="",
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ContractViolationError(f"payload has malformed fields: {exc}") from exc
        object.__setattr__(contract, "contract_sha256",
                           compute_contract_sha256(contract.to_payload()))
        return contract

    @classmethod
    def load(cls, path: "str | os.PathLike[str]", *,
             expected_fold: "str | None" = None,
             expected_mean_sha: "str | None" = None) -> "RecoverabilityContract":
        """Read a contract JSON file and verify it fail-closed (FR-2.4).

        v1.0g anti-tamper sequence (DESIGN §3; AUDIT 5 §3.3 X item): read JSON
        -> **first verify the stored contract_sha256 equals the hash recomputed
        over the de-hashed payload** (mismatch, or a hash-less artifact, raises
        ContractViolationError *even when every field is inside its legal
        domain* — a stale stored hash means the artifact was edited after
        freezing) -> from_payload (creation path, recomputes/fills the hash)
        -> validate (domains + self-hash) -> fold / mean-SHA identity checks.
        Missing/unreadable files and malformed JSON propagate their native
        errors. from_payload itself keeps the creation semantics: it always
        recomputes and fills contract_sha256, ignoring any stored value.
        """
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ContractViolationError("contract JSON payload must be a mapping")
        stored = payload.get("contract_sha256")
        if not isinstance(stored, str) or not stored:
            raise ContractViolationError(
                "contract artifact is missing the stored contract_sha256 "
                "(v1.0g anti-tamper: load() refuses hash-less payloads)")
        dehashed = {key: value for key, value in payload.items()
                    if key != "contract_sha256"}
        recomputed = compute_contract_sha256(dehashed)
        if stored != recomputed:
            raise ContractViolationError(
                f"stored contract_sha256 mismatch: stored {stored!r} != "
                f"recomputed {recomputed!r} — the payload was modified after "
                "freezing (tampered artifact) even though all fields may sit in "
                "their legal domains (v1.0g anti-tamper, AUDIT 5 §3.3)")
        contract = cls.from_payload(payload)
        contract.validate()
        if expected_fold is not None and contract.fold_id != expected_fold:
            raise ContractViolationError(
                f"contract fold mismatch: expected {expected_fold!r}, "
                f"got {contract.fold_id!r}")
        if expected_mean_sha is not None and contract.mean_checkpoint_sha256 != expected_mean_sha:
            raise ContractViolationError(
                f"mean checkpoint SHA mismatch: expected {expected_mean_sha!r}, "
                f"got {contract.mean_checkpoint_sha256!r}")
        return contract


def expand_group_to_bands(c_groups: Mapping[str, torch.Tensor],
                          groups: Mapping[str, list[str]]) -> dict[str, torch.Tensor]:
    """Expand group-level c tensors to band-level c (DESIGN §3).

    Every band of every group in groups maps to the group's tensor; bands of
    one group share the same Tensor reference (no copy). Bands that appear in
    no group are absent from the output; groups covering all bands is the
    caller's responsibility. A group listed in groups but missing from
    c_groups raises KeyError (fail-closed).
    """
    if not isinstance(c_groups, Mapping) or not isinstance(groups, Mapping):
        raise ValueError("c_groups and groups must both be mappings")
    expanded: dict[str, torch.Tensor] = {}
    for group, band_list in groups.items():
        if not isinstance(band_list, (list, tuple)):
            raise ValueError(f"groups[{group!r}] must be a list of band names")
        c_value = c_groups[group]  # KeyError on unknown/missing group (fail-closed)
        for band in band_list:
            expanded[str(band)] = c_value
    return expanded


def mean_weights_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Canonical SHA-256 over a mean-predictor state dict (DESIGN v1.0b).

    Used both when freezing a contract (freeze_recoverability_contract.py)
    and when validating one at model-build time (SLMFBBDM integration), so
    the two sides cannot drift. Deterministic recipe: iterate keys in sorted
    order and feed key.encode() plus tensor bytes (detach/cpu/contiguous,
    floating tensors cast to float32) into one sha256 stream.
    """
    digest = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        detached = tensor.detach().cpu()
        if detached.is_floating_point() and detached.dtype != torch.float32:
            detached = detached.to(dtype=torch.float32)
        digest.update(str(key).encode("utf-8"))
        digest.update(detached.contiguous().numpy().tobytes())
    return digest.hexdigest()
