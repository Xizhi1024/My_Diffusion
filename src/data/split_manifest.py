"""Patient-level split manifest: ensures no patient leaks across train/val/test.

Generates split_manifest.csv from .npz cache, grouping by patient_id.
CachedDataset reads this manifest to strictly enforce patient-level splits.

Usage:
    python -m src.data.split_manifest --cache-dir cache/tensors --output cache/split_manifest.csv --val-ratio 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

_CSV_COLUMNS = ["sample_id", "patient_id", "slice_id", "split", "cache_path"]


def _parse_sample_id(sid: str) -> Tuple[str, int]:
    """Extract patient_id (first 3 chars) and slice_id (next 3 chars)."""
    pid = sid[:3] if len(sid) >= 3 else sid
    try:
        slc = int(sid[3:6]) if len(sid) >= 6 else 0
    except ValueError:
        slc = 0
    return pid, slc


def _compute_fingerprint(rows: List[Dict[str, str]], seed: int) -> str:
    payload = json.dumps(
        sorted((r["sample_id"], r["patient_id"], r["split"]) for r in rows),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def generate_split_manifest(
    cache_dir: Path,
    output_csv: Path,
    val_ratio: float = 0.15,
    seed: int = 42,
    test_patient_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Scan .npz cache, group by patient, assign splits, write manifest.

    Returns stats dict.
    """
    cache_dir = Path(cache_dir)
    npz_files = sorted(cache_dir.glob("*.npz"))
    if not npz_files:
        raise ValueError(f"No .npz files in {cache_dir}")

    # Group by patient_id
    patient_samples: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
    for npz_path in npz_files:
        sid = npz_path.stem
        pid, slc = _parse_sample_id(sid)
        patient_samples[pid].append((sid, npz_path))

    patient_ids = sorted(patient_samples)

    # Assign splits
    rng = random.Random(seed)

    # Determine test patients
    if test_patient_ids:
        test_pids = set(test_patient_ids)
    else:
        # Default: last 15% of patients by sorted order as test (configurable)
        n_test = max(1, int(len(patient_ids) * 0.10))
        test_pids = set(patient_ids[-n_test:])

    # Remove test patients from train/val pool
    train_val_pids = [p for p in patient_ids if p not in test_pids]
    rng.shuffle(train_val_pids)

    n_val = int(len(train_val_pids) * val_ratio)
    # Ensure at least 1 val patient when we have >= 2 train_val patients
    if n_val == 0 and val_ratio > 0 and len(train_val_pids) >= 2:
        n_val = 1
    val_pids = set(train_val_pids[:n_val])
    train_pids = set(train_val_pids[n_val:])

    # Build rows
    rows: List[Dict[str, str]] = []
    for pid in patient_ids:
        if pid in test_pids:
            split = "test"
        elif pid in val_pids:
            split = "val"
        else:
            split = "train"

        for sid, npz_path in patient_samples[pid]:
            _, slc = _parse_sample_id(sid)
            rows.append({
                "sample_id": sid,
                "patient_id": pid,
                "slice_id": str(slc),
                "split": split,
                "cache_path": str(npz_path.resolve()),
            })

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # Write JSON summary
    json_path = output_csv.with_suffix(".json")
    stats = {
        "total_samples": len(rows),
        "total_patients": len(patient_ids),
        "train_patients": len(train_pids),
        "val_patients": len(val_pids),
        "test_patients": len(test_pids),
        "train_samples": sum(1 for r in rows if r["split"] == "train"),
        "val_samples": sum(1 for r in rows if r["split"] == "val"),
        "test_samples": sum(1 for r in rows if r["split"] == "test"),
        "val_ratio": val_ratio,
        "seed": seed,
        "fingerprint": _compute_fingerprint(rows, seed),
        "train_patient_ids": sorted(train_pids),
        "val_patient_ids": sorted(val_pids),
        "test_patient_ids": sorted(test_pids),
        "generated_at": str(argparse.ArgumentParser().parse_args([])),  # placeholder
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

    # Verify no patient overlap
    assert not (train_pids & val_pids), "train/val patient overlap!"
    assert not (train_pids & test_pids), "train/test patient overlap!"
    assert not (val_pids & test_pids), "val/test patient overlap!"

    return stats


def load_split_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Read split manifest CSV and return as dict keyed by sample_id."""
    manifest: Dict[str, Dict[str, str]] = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            manifest[row["sample_id"]] = row
    return manifest


class SplitManifest:
    """In-memory split manifest with fast lookup and validation."""

    def __init__(self, manifest_path: Path):
        self.path = Path(manifest_path)
        self._by_sample: Dict[str, Dict[str, str]] = load_split_manifest(manifest_path)
        self._patient_split: Dict[str, str] = {}
        for sid, row in self._by_sample.items():
            self._patient_split[row["patient_id"]] = row["split"]

        self.train_samples = [s for s, r in self._by_sample.items() if r["split"] == "train"]
        self.val_samples = [s for s, r in self._by_sample.items() if r["split"] == "val"]
        self.test_samples = [s for s, r in self._by_sample.items() if r["split"] == "test"]

        # Load JSON stats if available
        json_path = manifest_path.with_suffix(".json")
        self.stats = {}
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                self.stats = json.load(f)

    def get_split(self, sample_id: str) -> str:
        entry = self._by_sample.get(sample_id)
        return entry["split"] if entry else ""

    def get_patient_split(self, patient_id: str) -> str:
        return self._patient_split.get(patient_id, "")

    def get_samples_for_split(self, split: str) -> List[str]:
        return [s for s, r in self._by_sample.items() if r["split"] == split]

    def validate_no_overlap(self) -> bool:
        splits_by_patient: Dict[str, Set[str]] = defaultdict(set)
        for sid, row in self._by_sample.items():
            splits_by_patient[row["patient_id"]].add(row["split"])
        violations = {p: ss for p, ss in splits_by_patient.items() if len(ss) > 1}
        if violations:
            print(f"[SplitManifest] WARNING: {len(violations)} patients appear in multiple splits: {list(violations)[:5]}...")
            return False
        return True

    def __len__(self) -> int:
        return len(self._by_sample)

    def __repr__(self) -> str:
        if self.stats:
            return (f"SplitManifest({len(self)} samples, "
                    f"train={self.stats.get('train_patients','?')}p/{self.stats.get('train_samples','?')}s, "
                    f"val={self.stats.get('val_patients','?')}p/{self.stats.get('val_samples','?')}s, "
                    f"test={self.stats.get('test_patients','?')}p/{self.stats.get('test_samples','?')}s)")
        return f"SplitManifest({len(self)} samples)"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate patient-level split manifest")
    ap.add_argument("--cache-dir", required=True, type=Path, help="Path to .npz cache directory")
    ap.add_argument("--output", required=True, type=Path, help="Output CSV path (e.g. cache/split_manifest.csv)")
    ap.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio of non-test patients")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-patients", type=str, nargs="*", default=None,
                    help="Explicit test patient IDs (space-separated)")
    ap.add_argument("--print-stats", action="store_true")
    args = ap.parse_args(argv)

    test_pids = set(args.test_patients) if args.test_patients else None

    stats = generate_split_manifest(
        cache_dir=args.cache_dir,
        output_csv=args.output,
        val_ratio=args.val_ratio,
        seed=args.seed,
        test_patient_ids=test_pids,
    )

    if args.print_stats:
        print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))

    print(f"[split_manifest] {stats['total_samples']} samples, {stats['total_patients']} patients")
    print(f"  train: {stats['train_patients']} patients, {stats['train_samples']} samples")
    print(f"  val:   {stats['val_patients']} patients, {stats['val_samples']} samples")
    print(f"  test:  {stats['test_patients']} patients, {stats['test_samples']} samples")
    print(f"  fingerprint: {stats['fingerprint']}")
    print(f"  written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
