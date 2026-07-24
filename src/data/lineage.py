"""Strict cache-lineage loading and checkpoint embedding.

Formal mechanism runs opt in with ``data.require_cache_lineage: true``.  The
loader then fails before training unless the sealed cache metadata is
self-consistent and agrees with the locked Stage 0A dataset contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


REQUIRED_CHECKPOINT_LINEAGE_FIELDS = (
    "manifest_semantic_sha256",
    "raw_png_combined_sha256",
    "preprocessing_config_sha256",
    "dataset_contract_sha256",
    "cache_payload_sha256",
    "cache_metadata_sha256",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CacheLineageError(ValueError):
    """Raised when a strict run cannot prove its cache lineage."""


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CacheLineageError(
            f"Cannot read {label} JSON at {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CacheLineageError(f"{label} must be a JSON object: {path}")
    return payload


def _resolve(root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _validate_self_hash(
    payload: Mapping[str, Any], hash_field: str, label: str
) -> None:
    body = dict(payload)
    declared = str(body.pop(hash_field, ""))
    computed = _sha256(_canonical_json_bytes(body))
    if not declared or declared != computed:
        raise CacheLineageError(
            f"{label} self-hash mismatch: declared={declared!r}, "
            f"computed={computed!r}"
        )


def _validate_contract(
    contract_path: Path, lineage: Mapping[str, Any]
) -> None:
    contract = _read_object(contract_path, "dataset contract")
    _validate_self_hash(contract, "contract_sha256", "dataset contract")
    expected = {
        "manifest_semantic_sha256": contract.get("manifest", {}).get(
            "semantic_sha256"
        ),
        "raw_png_combined_sha256": contract.get("raw_png", {}).get(
            "combined_sha256"
        ),
        "preprocessing_config_sha256": contract.get(
            "preprocessing_config_sha256"
        ),
        "dataset_contract_sha256": contract.get("contract_sha256"),
    }
    mismatches = {
        field: {"contract": expected_value, "cache_lineage": lineage.get(field)}
        for field, expected_value in expected.items()
        if lineage.get(field) != expected_value
    }
    if mismatches:
        raise CacheLineageError(
            "Cache lineage does not match the locked dataset contract: "
            + json.dumps(mismatches, sort_keys=True)
        )


def load_checkpoint_data_lineage(
    config: Mapping[str, Any],
    *,
    root: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    """Load verified cache metadata for inclusion in a checkpoint.

    Non-formal legacy configurations remain loadable when
    ``require_cache_lineage`` is false.  Formal configurations fail closed.
    """

    data_config = config.get("data", {})
    if not isinstance(data_config, Mapping):
        raise CacheLineageError("config.data must be a mapping")
    strict = bool(data_config.get("require_cache_lineage", False))
    if bool(data_config.get("use_fake_data", False)):
        if strict:
            raise CacheLineageError(
                "A strict lineage run cannot use data.use_fake_data=true"
            )
        return None

    repository_root = Path(root or Path.cwd()).resolve()
    cache_dir_value = str(data_config.get("cache_dir", "") or "").strip()
    lineage_value = str(data_config.get("cache_lineage", "") or "").strip()
    if lineage_value:
        lineage_path = _resolve(repository_root, lineage_value)
    elif cache_dir_value:
        lineage_path = _resolve(repository_root, cache_dir_value) / "cache_lineage.json"
    else:
        if strict:
            raise CacheLineageError(
                "Strict lineage requires data.cache_dir or data.cache_lineage"
            )
        return None

    if not lineage_path.is_file():
        if strict:
            raise CacheLineageError(
                f"Required sealed cache lineage not found: {lineage_path}"
            )
        return None

    lineage = _read_object(lineage_path, "cache lineage")
    _validate_self_hash(
        lineage, "cache_metadata_sha256", "cache lineage"
    )
    missing_or_invalid = [
        field
        for field in REQUIRED_CHECKPOINT_LINEAGE_FIELDS
        if not _SHA256_RE.fullmatch(str(lineage.get(field, "")))
    ]
    if missing_or_invalid:
        raise CacheLineageError(
            "Cache lineage has missing/invalid SHA-256 fields: "
            + ", ".join(missing_or_invalid)
        )
    if lineage.get("lineage_type") != "verified_png_tensor_cache":
        raise CacheLineageError(
            "Cache lineage_type must be 'verified_png_tensor_cache'"
        )

    if cache_dir_value:
        expected_parent = _resolve(repository_root, cache_dir_value)
        if lineage_path.parent.resolve() != expected_parent:
            raise CacheLineageError(
                "Cache lineage must be stored inside the configured cache_dir: "
                f"lineage={lineage_path}, cache_dir={expected_parent}"
            )

    contract_value = str(
        data_config.get("dataset_contract", "") or ""
    ).strip()
    if contract_value:
        contract_path = _resolve(repository_root, contract_value)
        if not contract_path.is_file():
            raise CacheLineageError(
                f"Configured dataset contract not found: {contract_path}"
            )
        _validate_contract(contract_path, lineage)
    elif strict:
        raise CacheLineageError(
            "Strict lineage requires data.dataset_contract"
        )

    return dict(lineage)


def attach_data_lineage(
    checkpoint: Mapping[str, Any],
    data_lineage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a checkpoint payload with verified lineage at a stable top-level key."""

    payload = dict(checkpoint)
    if data_lineage is not None:
        payload["data_lineage"] = dict(data_lineage)
    return payload


def validate_checkpoint_data_lineage(
    checkpoint: Mapping[str, Any],
    expected_lineage: Mapping[str, Any] | None,
    *,
    required: bool,
    context: str = "checkpoint",
) -> dict[str, Any] | None:
    """Require an input checkpoint to match the current sealed cache exactly."""

    embedded = checkpoint.get("data_lineage")
    if not isinstance(embedded, Mapping):
        if required:
            raise CacheLineageError(
                f"{context} lacks required top-level data_lineage"
            )
        return None
    embedded_dict = dict(embedded)
    _validate_self_hash(
        embedded_dict,
        "cache_metadata_sha256",
        f"{context} data lineage",
    )
    invalid = [
        field
        for field in REQUIRED_CHECKPOINT_LINEAGE_FIELDS
        if not _SHA256_RE.fullmatch(str(embedded_dict.get(field, "")))
    ]
    if invalid:
        raise CacheLineageError(
            f"{context} has missing/invalid lineage fields: "
            + ", ".join(invalid)
        )
    if embedded_dict.get("lineage_type") != "verified_png_tensor_cache":
        raise CacheLineageError(
            f"{context} has unsupported lineage_type "
            f"{embedded_dict.get('lineage_type')!r}"
        )
    if expected_lineage is not None:
        mismatches = {
            field: {
                "checkpoint": embedded_dict.get(field),
                "current_cache": expected_lineage.get(field),
            }
            for field in REQUIRED_CHECKPOINT_LINEAGE_FIELDS
            if embedded_dict.get(field) != expected_lineage.get(field)
        }
        if mismatches:
            raise CacheLineageError(
                f"{context} lineage differs from current cache: "
                + json.dumps(mismatches, sort_keys=True)
            )
    elif required:
        raise CacheLineageError(
            f"{context} lineage cannot be checked without current cache lineage"
        )
    return embedded_dict
