"""FR-6.6 (DESIGN RC-BRD_v1 S10): fail-closed run-ledger artifact validation.

Validates that a run ledger JSON carries every required artifact of [计划] §5.1
(lesion_routed_diffusion_plan_v1.md):

* run_id, git commit + dirty flag, config content + sha256;
* env lock, GPU, seed, start/end times;
* outer fold, outer-train/outer-test patient-id hashes, inner-split hash;
* mean/diffusion/contract checkpoints and their sha256;
* per-patient metric table, per-fold summary, prediction manifest;
* frozen checkpoint policy, NFE, sampling seed;
* deviation log (may be empty, but the key must exist).

Report: JSON with one entry per required artifact {ok, missing, detail}.
Missing any required artifact -> exit 1.  --require-artifacts restricts the
checked subset (default: the full preregistered list); unknown names -> exit 2.

Standalone batch-3B deliverable: stdlib only, no rc_brd dependency.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

LEDGER_SCHEMA_VERSION = 1  # batch 3B ledger schema (this script defines it)
_HEX_DIGITS = set("0123456789abcdefABCDEF")
_EXIT_OK = 0
_EXIT_MISSING = 1  # DESIGN S10: missing required artifact
_EXIT_USAGE = 2  # unknown --require-artifacts name / bad usage


class LedgerValidationError(RuntimeError):
    """Fail-closed error for unreadable / malformed ledger files."""


def _spec(key: str, kind: str, source: str) -> tuple[str, str, str]:
    return (key, kind, source)


# Ordered [计划] §5.1 checklist: (ledger key, value kind, provenance note).
REQUIRED_ARTIFACTS: tuple[tuple[str, str, str], ...] = (
    _spec("run_id", "nonempty_str", "§5.1-1 run identifier"),
    _spec("git_commit", "nonempty_str", "§5.1-1 git commit"),
    _spec("git_dirty", "bool", "§5.1-1 dirty flag"),
    _spec("config", "nonempty", "§5.1-1 config content or recorded path"),
    _spec("config_sha256", "sha256_hex", "§5.1-1 config sha256"),
    _spec("env_lock", "nonempty", "§5.1-2 container/interpreter lock"),
    _spec("gpu", "nonempty", "§5.1-2 CUDA/GPU info"),
    _spec("seed", "int", "§5.1-2 random seed"),
    _spec("started_at", "nonempty_str", "§5.1-2 run start time"),
    _spec("ended_at", "nonempty_str", "§5.1-2 run end time"),
    _spec("outer_fold", "nonempty_str", "§5.1-3 outer fold id"),
    _spec("outer_train_patient_hash", "nonempty_str", "§5.1-3 outer-train patient hash"),
    _spec("outer_test_patient_hash", "nonempty_str", "§5.1-3 outer-test patient hash"),
    _spec("inner_split_hash", "nonempty_str", "§5.1-3 inner split hash"),
    _spec("mean_checkpoint", "nonempty", "§5.1-4 mean checkpoint"),
    _spec("mean_checkpoint_sha256", "sha256_hex", "§5.1-4 mean checkpoint sha256"),
    _spec("diffusion_checkpoint", "nonempty", "§5.1-4 diffusion checkpoint"),
    _spec("diffusion_checkpoint_sha256", "sha256_hex", "§5.1-4 diffusion sha256"),
    _spec("contract_artifact", "nonempty", "§5.1-4 contract artifact"),
    _spec("contract_sha256", "sha256_hex", "§5.1-4 contract sha256"),
    _spec("patient_metrics_table", "nonempty", "§5.1-5 per-patient metrics table"),
    _spec("per_fold_summary", "nonempty", "§5.1-5 per-fold summary"),
    _spec("prediction_manifest", "nonempty", "§5.1-5 prediction manifest"),
    _spec("checkpoint_policy", "nonempty_str", "§5.1-6 frozen checkpoint policy"),
    _spec("nfe", "positive_int", "§5.1-6 fixed NFE"),
    _spec("sampling_seed", "int", "§5.1-6 sampling seed"),
    _spec("deviation_log", "list", "§5.1-6 deviation log (may be empty)"),
)

_SPEC_BY_KEY: dict[str, tuple[str, str, str]] = {
    spec[0]: spec for spec in REQUIRED_ARTIFACTS
}


def _short(value: Any, limit: int = 48) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in _HEX_DIGITS for char in value)
    )


def _is_nonempty(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, tuple)):
        return len(value) > 0
    return isinstance(value, (int, float, str))


def _kind_ok(kind: str, value: Any) -> bool:
    if kind == "nonempty_str":
        return isinstance(value, str) and bool(value.strip())
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "positive_int":
        return isinstance(value, int) and not isinstance(value, bool) and value > 0
    if kind == "sha256_hex":
        return _is_sha256_hex(value)
    if kind == "list":
        return isinstance(value, list)
    if kind == "nonempty":
        return _is_nonempty(value)
    raise ValueError(f"unknown artifact kind: {kind}")


def check_artifact(
    ledger: Mapping[str, Any], key: str, kind: str
) -> dict[str, Any]:
    """Check one ledger key: {ok, missing, detail} (task report contract)."""
    if key not in ledger:
        return {"ok": False, "missing": True, "detail": "key absent from ledger"}
    value = ledger[key]
    if _kind_ok(kind, value):
        return {
            "ok": True,
            "missing": False,
            "detail": f"{kind} ok: {_short(value)}",
        }
    return {
        "ok": False,
        "missing": False,
        "detail": f"invalid {kind}: {_short(value)}",
    }


def validate_run_ledger(
    ledger: Mapping[str, Any],
    required: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate the [计划] §5.1 artifact checklist of one run ledger.

    required=None -> the full preregistered list; otherwise a subset of known
    artifact names (unknown names raise ValueError).
    """
    keys = (
        list(required)
        if required is not None
        else [spec[0] for spec in REQUIRED_ARTIFACTS]
    )
    unknown = sorted(key for key in keys if key not in _SPEC_BY_KEY)
    if unknown:
        raise ValueError(
            f"unknown artifact names: {unknown}; "
            f"valid: {sorted(_SPEC_BY_KEY)}"
        )
    checks = {
        key: check_artifact(ledger, key, _SPEC_BY_KEY[key][1]) for key in keys
    }
    failed = [key for key in keys if not checks[key]["ok"]]
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "required_artifacts": list(keys),
        "checks": checks,
        "n_required": len(keys),
        "n_ok": len(keys) - len(failed),
        "n_failed": len(failed),
        "all_ok": not failed,
        "missing_required": failed,
    }


def load_ledger(path: str | Path) -> dict[str, Any]:
    """Load a run ledger JSON; malformed files raise LedgerValidationError."""
    p = Path(path)
    if not p.is_file():
        raise LedgerValidationError(f"ledger file not found: {p}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LedgerValidationError(f"ledger is not valid JSON: {p}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LedgerValidationError(f"ledger must be a JSON object: {p}")
    return payload


def _split_names(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    names = [part for part in raw.replace(",", " ").split() if part]
    return names or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the [计划] §5.1 required-artifact run ledger "
        "(DESIGN RC-BRD_v1 §10)."
    )
    parser.add_argument("--ledger", required=True, help="run ledger json path")
    parser.add_argument(
        "--require-artifacts",
        default=None,
        help="comma-separated subset of artifact keys (default: full list)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    required = _split_names(args.require_artifacts)
    try:
        ledger = load_ledger(args.ledger)
        report = validate_run_ledger(ledger, required)
    except LedgerValidationError as exc:
        print(f"[validate_run_ledger] FAIL: {exc}", file=sys.stderr)
        return _EXIT_MISSING
    except ValueError as exc:
        print(f"[validate_run_ledger] usage error: {exc}", file=sys.stderr)
        return _EXIT_USAGE
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["all_ok"]:
        print(
            f"[validate_run_ledger] FAIL: missing/invalid artifacts: "
            f"{report['missing_required']}",
            file=sys.stderr,
        )
        return _EXIT_MISSING
    return _EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
