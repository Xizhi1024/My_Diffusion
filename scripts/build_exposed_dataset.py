"""Build exposed dataset manifest from raw Data directory.

Usage::

    python scripts/build_exposed_dataset.py \
        --raw-root Data \
        --out-root exposed_data/ct_pet_seg \
        --dicom-root Data/subset_A \
        --test-png-root Data/subset_B \
        --split-mode keep-subset-b-test \
        --val-ratio 0.15 \
        --seed 42 \
        --sample-id-regex "^(?P<patient_id>\\d{3})(?P<slice_id>\\d{3})$" \
        --dicom-slice-order z_desc \
        --dicom-index-offset 0 \
        --link-mode manifest-only
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from src.data.data_manifest import (
    DEFAULT_SAMPLE_ID_REGEX,
    assign_splits,
    build_sample_records,
    compute_dataset_fingerprint,
    write_csv,
    write_jsonl,
    write_report,
    write_split_csvs,
)


def _resolve(path_str: str) -> Path:
    return Path(path_str).resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build exposed dataset manifest from raw Data directory.",
    )
    parser.add_argument(
        "--raw-root", required=True, type=_resolve,
        help="Root of raw Data (contains subset_A).",
    )
    parser.add_argument(
        "--out-root", required=True, type=_resolve,
        help="Output directory for manifest and report files.",
    )
    parser.add_argument(
        "--dicom-root", required=True, type=_resolve,
        help="Root containing DICOM patient directories (usually Data/subset_A).",
    )
    parser.add_argument(
        "--test-png-root", required=True, type=_resolve,
        help="Root containing subset_B PNGs (used to identify test samples).",
    )
    parser.add_argument(
        "--split-mode", default="keep-subset-b-test",
        help="Split mode (first version only supports 'keep-subset-b-test').",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Fraction of non-test patients reserved for validation. (default: 0.15)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for patient-level split. (default: 42)",
    )
    parser.add_argument(
        "--sample-id-regex", default=DEFAULT_SAMPLE_ID_REGEX,
        help="Regex with named groups 'patient_id' and 'slice_id'.",
    )
    parser.add_argument(
        "--dicom-slice-order", default="z_desc",
        choices=["z_desc", "z_asc", "instance_asc", "filename"],
        help="Sort order for DICOM slices within a patient. (default: z_desc)",
    )
    parser.add_argument(
        "--dicom-index-offset", type=int, default=0,
        help="Offset applied to DICOM slice index. (default: 0)",
    )
    parser.add_argument(
        "--link-mode", default="manifest-only",
        help="Data linking mode. First version only supports 'manifest-only'.",
    )

    args = parser.parse_args(argv)

    # ── guardrail ───────────────────────────────────────────────────
    if args.link_mode != "manifest-only":
        print(
            f"[ERROR] --link-mode={args.link_mode!r} is not implemented. "
            "Only 'manifest-only' is supported in this version.",
            file=sys.stderr,
        )
        raise NotImplementedError(
            f"link_mode={args.link_mode!r} not supported"
        )

    if args.split_mode != "keep-subset-b-test":
        print(
            f"[ERROR] --split-mode={args.split_mode!r} is not implemented. "
            "Only 'keep-subset-b-test' is supported.",
            file=sys.stderr,
        )
        raise NotImplementedError(
            f"split_mode={args.split_mode!r} not supported"
        )

    if not args.raw_root.is_dir():
        raise FileNotFoundError(f"raw-root not found: {args.raw_root}")
    if not args.dicom_root.is_dir():
        raise FileNotFoundError(f"dicom-root not found: {args.dicom_root}")
    if not args.test_png_root.is_dir():
        raise FileNotFoundError(f"test-png-root not found: {args.test_png_root}")

    # ── build records ───────────────────────────────────────────────
    print(f"[build_exposed_dataset] Scanning PNG & DICOM sources ...")
    records = build_sample_records(
        raw_root=args.raw_root,
        dicom_root=args.dicom_root,
        test_png_root=args.test_png_root,
        sample_id_regex=args.sample_id_regex,
        dicom_slice_order=args.dicom_slice_order,
        dicom_index_offset=args.dicom_index_offset,
    )
    print(f"[build_exposed_dataset] {len(records)} sample records built.")

    # ── assign splits ───────────────────────────────────────────────
    test_sample_ids = {
        r.sample_id
        for r in records
        if "subset_B" in (r.ct_png_source_subset, r.pet_png_source_subset)
    }
    print(f"[build_exposed_dataset] {len(test_sample_ids)} test sample IDs from subset_B.")

    assign_splits(
        records=records,
        test_sample_ids=test_sample_ids,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_n = sum(1 for r in records if r.split == "train")
    val_n = sum(1 for r in records if r.split == "val")
    test_n = sum(1 for r in records if r.split == "test")
    print(f"[build_exposed_dataset] Splits: train={train_n}, val={val_n}, test={test_n}")

    # ── compute fingerprint ─────────────────────────────────────────
    fingerprint = compute_dataset_fingerprint(records)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── write outputs ───────────────────────────────────────────────
    print(f"[build_exposed_dataset] Writing outputs to {args.out_root} ...")
    args.out_root.mkdir(parents=True, exist_ok=True)

    write_csv(records, args.out_root / "manifest_all.csv")
    write_jsonl(records, args.out_root / "manifest_all.jsonl")

    write_csv(
        records,
        args.out_root / f"manifest_snapshot_{timestamp}_{fingerprint}.csv",
    )
    write_jsonl(
        records,
        args.out_root / f"manifest_snapshot_{timestamp}_{fingerprint}.jsonl",
    )

    with open(args.out_root / "dataset_fingerprint.txt", "w") as f:
        f.write(f"{fingerprint}\n")

    write_report(records, args.out_root)

    write_split_csvs(records, args.out_root / "splits")

    # ── summary ─────────────────────────────────────────────────────
    print("[build_exposed_dataset] Done. Output files:")
    for p in sorted(args.out_root.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(args.out_root)}")


if __name__ == "__main__":
    main()
