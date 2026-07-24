"""Fail-closed loader for the H3-v2 full-timestep native/null schedule.

The schedule is deliberately narrower than the learned three-way router.  It
contains one frozen active mass for every ``(level, band, timestep)`` and maps
that mass to ``[native, shallow, null] = [a, 0, 1-a]``.  In particular, the
loader never interpolates, never consults an image, and never invents a
shallow-route probability.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


H3_V2_PIPELINE_ID = "H3_V2_FULL_TIMESTEP_NATIVE_NULL_CALIBRATION_V1"
H3_V2_ROUTE_ORDER = ("native", "shallow", "null")
H3_V2_BAND_ORDER = (
    "l2_lh",
    "l2_hl",
    "l2_hh",
    "l1_lh",
    "l1_hl",
    "l1_hh",
)


def _file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _self_hash(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("schedule_sha256", None)
    return _canonical_sha256(canonical)


def load_h3_native_null_schedule(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_num_train_timesteps: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load and validate a frozen H3-v2 schedule.

    Returns an active-mass tensor with shape ``[2, 3, T]`` ordered as
    ``[L2, L1]`` and ``[LH, HL, HH]``.  Any identity, coverage, monotonicity,
    range, or native/null mapping mismatch raises before model construction.
    """

    schedule_path = Path(path)
    if not schedule_path.is_file():
        raise FileNotFoundError(f"H3-v2 schedule not found: {schedule_path}")
    expected_hash = str(expected_file_sha256).lower()
    if len(expected_hash) != 64:
        raise ValueError("h3_schedule_sha256 must be a full SHA-256 hex digest")
    observed_hash = _file_sha256(schedule_path)
    if observed_hash != expected_hash:
        raise ValueError(
            "H3-v2 schedule file SHA-256 mismatch: "
            f"expected {expected_hash}, observed {observed_hash}"
        )

    try:
        payload = json.loads(schedule_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load H3-v2 schedule {schedule_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("H3-v2 schedule root must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("H3-v2 schedule schema_version must be 1")
    if payload.get("pipeline_id") != H3_V2_PIPELINE_ID:
        raise ValueError(
            "H3-v2 schedule pipeline_id mismatch: "
            f"{payload.get('pipeline_id')!r}"
        )
    if payload.get("decision") != "PASS":
        raise ValueError("H3-v2 schedule must carry decision=PASS")
    if payload.get("inference_schedule_allowed") is not True:
        raise ValueError("H3-v2 schedule does not allow inference consumption")
    if payload.get("production_activation_allowed") is not False:
        raise ValueError(
            "H3-v2 exploratory schedule must not claim production activation"
        )
    if payload.get("schedule_sha256") != _self_hash(payload):
        raise ValueError("H3-v2 schedule canonical self-hash mismatch")

    route_order = tuple(payload.get("route_order", ()))
    if route_order != H3_V2_ROUTE_ORDER:
        raise ValueError(
            f"H3-v2 route order must be {H3_V2_ROUTE_ORDER}, got {route_order}"
        )
    band_order = tuple(payload.get("band_order", ()))
    if band_order != H3_V2_BAND_ORDER:
        raise ValueError(
            f"H3-v2 band order must be {H3_V2_BAND_ORDER}, got {band_order}"
        )
    timestep_count = payload.get("num_train_timesteps")
    if timestep_count != int(expected_num_train_timesteps):
        raise ValueError(
            "H3-v2 timestep count mismatch: "
            f"expected {expected_num_train_timesteps}, got {timestep_count}"
        )
    if payload.get("lookup_policy") != "exact_integer_timestep_only":
        raise ValueError("H3-v2 lookup policy must forbid interpolation")
    if payload.get("shallow_route_policy") != "structurally_zero":
        raise ValueError("H3-v2 shallow route must be structurally zero")
    forbidden = {
        "target_pet",
        "lesion_mask",
        "recoverability_label",
        "single_slice_residual_band_evidence",
        "random_batch_adjacency",
    }
    declared_forbidden = set(payload.get("forbidden_runtime_inputs", ()))
    if not forbidden.issubset(declared_forbidden):
        raise ValueError("H3-v2 schedule omits required forbidden runtime inputs")

    active_payload = payload.get("native_active_mass")
    if not isinstance(active_payload, dict):
        raise ValueError("H3-v2 native_active_mass must be an object")
    rows: list[list[float]] = []
    for band in H3_V2_BAND_ORDER:
        values = active_payload.get(band)
        if not isinstance(values, list) or len(values) != timestep_count:
            raise ValueError(
                f"H3-v2 band {band!r} must contain exactly {timestep_count} values"
            )
        try:
            row = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"H3-v2 band {band!r} contains a non-number") from exc
        tensor_row = torch.tensor(row, dtype=torch.float64)
        if not torch.isfinite(tensor_row).all():
            raise ValueError(f"H3-v2 band {band!r} contains a non-finite value")
        if bool(((tensor_row < 0.0) | (tensor_row > 1.0)).any()):
            raise ValueError(f"H3-v2 band {band!r} leaves [0, 1]")
        if bool((tensor_row[1:] - tensor_row[:-1] > 1e-7).any()):
            raise ValueError(
                f"H3-v2 band {band!r} is not non-increasing over timesteps"
            )
        rows.append(row)

    active = torch.tensor(rows, dtype=torch.float32).reshape(2, 3, timestep_count)
    metadata = {
        "path": str(schedule_path.resolve()),
        "file_sha256": observed_hash,
        "schedule_sha256": payload["schedule_sha256"],
        "pipeline_id": payload["pipeline_id"],
        "num_train_timesteps": timestep_count,
        "band_order": list(band_order),
        "route_order": list(route_order),
    }
    return active, metadata


def native_null_routes(
    active_mass: torch.Tensor,
    timestep: torch.Tensor,
    *,
    level_index: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return exact ``[B, 3 bands, 3 routes]`` schedule rows."""

    if active_mass.ndim != 3 or active_mass.shape[:2] != (2, 3):
        raise ValueError("H3-v2 active_mass must have shape [2,3,T]")
    if level_index not in (0, 1):
        raise ValueError("level_index must identify L2 (0) or L1 (1)")
    if timestep.ndim != 1:
        raise ValueError("timestep must have shape [B]")
    if timestep.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("H3-v2 timestep lookup requires an integer tensor")
    steps = int(active_mass.shape[-1])
    if bool(((timestep < 0) | (timestep >= steps)).any()):
        raise ValueError(f"H3-v2 timestep must be in [0, {steps - 1}]")
    table = active_mass[level_index].transpose(0, 1)
    active = table.to(device=timestep.device)[timestep.long()].to(reference)
    shallow = torch.zeros_like(active)
    return torch.stack((active, shallow, 1.0 - active), dim=-1)
