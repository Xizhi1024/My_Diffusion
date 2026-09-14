"""Fast checkpoint-lineage inventory against an already sealed Stage 0C cache.

Unlike ``run_cloud_stage0c_gate.py``, this command does not re-hash raw PNGs or
recompare every cache tensor.  It verifies the sealed cache-lineage self-hash
and locked dataset contract, then inspects checkpoint metadata with fake
tensors so large model/optimizer storages are not materialized.

Use the full Stage 0C gate whenever the raw data or cache may have changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cloud_stage0c_gate import _validate_checkpoints
from src.data.lineage import CacheLineageError, load_checkpoint_data_lineage


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row}) if rows else ()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(columns), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    output = args.output.resolve()
    cache_dir = args.cache_dir.resolve()
    contract = args.contract.resolve()
    output.mkdir(parents=True, exist_ok=True)

    lineage_config = {
        "data": {
            "cache_dir": str(cache_dir),
            "cache_lineage": str(cache_dir / "cache_lineage.json"),
            "dataset_contract": str(contract),
            "require_cache_lineage": True,
        }
    }
    try:
        lineage = load_checkpoint_data_lineage(lineage_config, root=root)
    except CacheLineageError as exc:
        decision = {
            "stage": "00C_checkpoint_lineage_inventory",
            "scope": "checkpoint_metadata_against_sealed_stage0c",
            "decision": "FAIL",
            "checkpoint_lineage": "NOT_EVALUATED",
            "sealed_cache_lineage": "FAIL",
            "formal_checkpoint_claims_allowed": False,
            "fatal_error": str(exc),
            "stop_rule": "Stop this inventory and rerun the full Stage 0C gate.",
        }
        _write_json(output / "decision.json", decision)
        return decision

    print(
        "Verified sealed cache metadata: "
        f"{lineage['cache_metadata_sha256']}",
        flush=True,
    )
    status, evidence, rows = _validate_checkpoints(
        args.checkpoint,
        root,
        lineage,
        hash_checkpoints=args.hash_checkpoints,
        metadata_only=True,
        show_progress=True,
    )
    _write_csv(output / "checkpoint_lineage.csv", rows)
    passed = status == "PASS"
    decision = {
        "stage": "00C_checkpoint_lineage_inventory",
        "scope": "checkpoint_metadata_against_sealed_stage0c",
        "decision": "PASS" if passed else "FAIL",
        "checkpoint_lineage": status,
        "sealed_cache_lineage": "PASS",
        "cloud_training_gate": "UNCHANGED_FROM_PRIOR_STAGE0C_PASS",
        "training_allowed": True,
        "formal_checkpoint_claims_allowed": passed,
        "model_mechanism_claims_allowed": False,
        "fingerprints": {
            field: lineage[field]
            for field in (
                "manifest_semantic_sha256",
                "raw_png_combined_sha256",
                "preprocessing_config_sha256",
                "dataset_contract_sha256",
                "cache_payload_sha256",
                "cache_metadata_sha256",
            )
        },
        "checkpoint_evidence": evidence,
        "stop_rule": (
            "All requested checkpoints have exact lineage; later mechanism gates are still required."
            if passed
            else "Quarantine failed checkpoints. This does not revoke the sealed cache or block a new lineaged run."
        ),
    }
    _write_json(output / "decision.json", decision)
    _write_json(
        output / "execution_metadata.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "root": root.as_posix(),
            "cache_dir": cache_dir.as_posix(),
            "contract": contract.as_posix(),
            "checkpoint_arguments": list(args.checkpoint),
            "checkpoint_file_hashing": args.hash_checkpoints,
            "raw_pngs_rehashed": False,
            "cache_tensors_recompared": False,
        },
    )
    return decision


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache/tensors_main")
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/dataset_contract_stage0a_v1.json"),
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint file or directory; repeat for multiple inputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/mechanism_validation/00C_checkpoint_inventory_legacy"
        ),
    )
    parser.add_argument(
        "--hash-checkpoints",
        action="store_true",
        help="Also SHA-256 every checkpoint file; much slower for legacy inventory.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.root = args.root.resolve()
    args.cache_dir = _resolve(args.root, args.cache_dir)
    args.contract = _resolve(args.root, args.contract)
    args.output = _resolve(args.root, args.output)
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        decision = run(args)
    except KeyboardInterrupt:
        print(
            "\nInterrupted safely; no cache or checkpoint was modified.",
            file=sys.stderr,
            flush=True,
        )
        return 130
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0 if decision["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
