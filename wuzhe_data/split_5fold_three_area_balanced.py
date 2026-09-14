#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Patient-level 5-fold split with three tumor-area strata.

Split rules:
  1. Patient-level split: all slices from one patient belong to one validation fold.
  2. Each validation fold is balanced close to 20% by patient count.
  3. Each validation fold is balanced close to 20% by slice count.
  4. White mask area is divided into 3 equal-frequency classes: small, medium, large.
  5. Each tumor-area class is balanced close to 20% in every validation fold.
  6. Per-patient slice-count distribution is used as an auxiliary balance term.

Default input:
  <project_root>/data/origin_data

Default outputs:
  <project_root>/wuzhe_data/origin_wuzhe_data/fold_1 ... fold_5
      Raw validation fold data, one folder per fold.

  <project_root>/wuzhe_data/fold_1 ... fold_5
      Combined 5-fold datasets. Each fold contains train/ and val/.

Example:
  python wuzhe_data/split_5fold_three_area_balanced.py
  python wuzhe_data/split_5fold_three_area_balanced.py --materialize hardlink
  python wuzhe_data/split_5fold_three_area_balanced.py --materialize none
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import shutil
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


MASK_CLASSES = 3
N_FOLDS = 5
AREA_LEVELS = ["small", "medium", "large"]


@dataclass(frozen=True)
class SliceRecord:
    file_name: str
    patient_id: str
    area: int
    area_class: int


@dataclass
class PatientRecord:
    patient_id: str
    slices: List[SliceRecord] = field(default_factory=list)

    @property
    def n_slices(self) -> int:
        return len(self.slices)

    @property
    def class_counts(self) -> List[int]:
        counts = [0] * MASK_CLASSES
        for item in self.slices:
            counts[item.area_class] += 1
        return counts

    @property
    def total_area(self) -> int:
        return sum(item.area for item in self.slices)

    @property
    def mean_area(self) -> float:
        return self.total_area / self.n_slices if self.n_slices else 0.0


def read_mask_area(mask_path: Path, threshold: int) -> int:
    try:
        from PIL import Image  # type: ignore

        with Image.open(mask_path) as img:
            img = img.convert("L")
            hist = img.histogram()
            return sum(hist[threshold + 1 :])
    except ImportError:
        return read_png_area_stdlib(mask_path, threshold)


def read_png_area_stdlib(mask_path: Path, threshold: int) -> int:
    raw = mask_path.read_bytes()
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {mask_path}")

    pos = 8
    width = height = bit_depth = color_type = interlace = None
    palette: List[Tuple[int, int, int]] = []
    idat_parts: List[bytes] = []

    while pos < len(raw):
        length = struct.unpack(">I", raw[pos : pos + 4])[0]
        chunk_type = raw[pos + 4 : pos + 8]
        data = raw[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", data)
        elif chunk_type == b"PLTE":
            palette = [(data[i], data[i + 1], data[i + 2]) for i in range(0, len(data), 3)]
        elif chunk_type == b"IDAT":
            idat_parts.append(data)
        elif chunk_type == b"IEND":
            break

    if width is None or height is None or bit_depth is None or color_type is None:
        raise ValueError(f"PNG is missing IHDR: {mask_path}")
    if interlace != 0:
        raise ValueError(f"Interlaced PNG is not supported without Pillow: {mask_path}")
    if bit_depth != 8:
        raise ValueError(f"Only 8-bit PNG is supported without Pillow: {mask_path}")

    channels_by_type = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    if color_type not in channels_by_type:
        raise ValueError(f"Unsupported PNG color type {color_type}: {mask_path}")

    channels = channels_by_type[color_type]
    stride = width * channels
    decompressed = zlib.decompress(b"".join(idat_parts))
    rows: List[bytearray] = []
    offset = 0
    for _ in range(height):
        filter_type = decompressed[offset]
        offset += 1
        row = bytearray(decompressed[offset : offset + stride])
        offset += stride
        prev = rows[-1] if rows else bytearray(stride)
        unfilter_row(row, prev, filter_type, channels)
        rows.append(row)

    area = 0
    for row in rows:
        if color_type == 0:
            area += sum(1 for value in row if value > threshold)
        elif color_type == 3:
            for idx in row:
                if idx < len(palette):
                    r, g, b = palette[idx]
                    if max(r, g, b) > threshold:
                        area += 1
        elif color_type == 2:
            for i in range(0, len(row), 3):
                if max(row[i], row[i + 1], row[i + 2]) > threshold:
                    area += 1
        elif color_type == 4:
            for i in range(0, len(row), 2):
                if row[i] > threshold and row[i + 1] > 0:
                    area += 1
        elif color_type == 6:
            for i in range(0, len(row), 4):
                if max(row[i], row[i + 1], row[i + 2]) > threshold and row[i + 3] > 0:
                    area += 1
    return area


def unfilter_row(row: bytearray, prev: bytearray, filter_type: int, bpp: int) -> None:
    if filter_type == 0:
        return
    if filter_type == 1:
        for i in range(len(row)):
            left = row[i - bpp] if i >= bpp else 0
            row[i] = (row[i] + left) & 0xFF
    elif filter_type == 2:
        for i in range(len(row)):
            row[i] = (row[i] + prev[i]) & 0xFF
    elif filter_type == 3:
        for i in range(len(row)):
            left = row[i - bpp] if i >= bpp else 0
            up = prev[i]
            row[i] = (row[i] + ((left + up) // 2)) & 0xFF
    elif filter_type == 4:
        for i in range(len(row)):
            left = row[i - bpp] if i >= bpp else 0
            up = prev[i]
            up_left = prev[i - bpp] if i >= bpp else 0
            row[i] = (row[i] + paeth_predictor(left, up, up_left)) & 0xFF
    else:
        raise ValueError(f"Unsupported PNG filter type: {filter_type}")


def paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def extract_patient_id(file_name: str, regex: str) -> str:
    match = re.match(regex, Path(file_name).stem)
    if not match:
        raise ValueError(f"Cannot extract patient id from {file_name!r} with regex {regex!r}")
    return match.group(1)


def assign_area_classes(items: List[Tuple[str, str, int]]) -> List[SliceRecord]:
    sorted_items = sorted(items, key=lambda item: (item[2], item[0]))
    n = len(sorted_items)
    class_by_name: Dict[str, int] = {}
    for rank, (file_name, _, _) in enumerate(sorted_items):
        class_by_name[file_name] = min(MASK_CLASSES - 1, (rank * MASK_CLASSES) // max(1, n))
    return sorted(
        [
            SliceRecord(file_name=file_name, patient_id=patient_id, area=area, area_class=class_by_name[file_name])
            for file_name, patient_id, area in items
        ],
        key=lambda item: (item.patient_id, item.file_name),
    )


def build_patient_records(slice_records: Sequence[SliceRecord]) -> List[PatientRecord]:
    by_patient: Dict[str, PatientRecord] = {}
    for item in slice_records:
        by_patient.setdefault(item.patient_id, PatientRecord(item.patient_id)).slices.append(item)
    return sorted(by_patient.values(), key=lambda patient: patient.patient_id)


def squared_relative_error(value: float, target: float) -> float:
    denom = max(target, 1.0)
    return ((value - target) / denom) ** 2


def make_slice_count_bins(patients: Sequence[PatientRecord]) -> Dict[str, int]:
    ordered = sorted(patients, key=lambda patient: (patient.n_slices, patient.patient_id))
    bins: Dict[str, int] = {}
    for rank, patient in enumerate(ordered):
        bins[patient.patient_id] = min(MASK_CLASSES - 1, (rank * MASK_CLASSES) // max(1, len(ordered)))
    return bins


def fold_cost(
    folds: Sequence[Sequence[PatientRecord]],
    all_patients: Sequence[PatientRecord],
    slice_count_balance_weight: float,
) -> float:
    total_patients = len(all_patients)
    total_slices = sum(patient.n_slices for patient in all_patients)
    total_classes = [0] * MASK_CLASSES
    for patient in all_patients:
        for idx, value in enumerate(patient.class_counts):
            total_classes[idx] += value

    target_patients = total_patients / N_FOLDS
    target_slices = total_slices / N_FOLDS
    target_classes = [value / N_FOLDS for value in total_classes]
    slice_count_bins = make_slice_count_bins(all_patients)
    total_bin_counts = [0] * MASK_CLASSES
    for patient in all_patients:
        total_bin_counts[slice_count_bins[patient.patient_id]] += 1
    target_bin_counts = [value / N_FOLDS for value in total_bin_counts]
    low_slice_total = sum(1 for patient in all_patients if patient.n_slices <= 2)
    target_low_slice = low_slice_total / N_FOLDS

    cost = 0.0
    for fold in folds:
        patient_count = len(fold)
        slice_count = sum(patient.n_slices for patient in fold)
        class_counts = [0] * MASK_CLASSES
        bin_counts = [0] * MASK_CLASSES
        low_slice_count = 0
        for patient in fold:
            if patient.n_slices <= 2:
                low_slice_count += 1
            bin_counts[slice_count_bins[patient.patient_id]] += 1
            for idx, value in enumerate(patient.class_counts):
                class_counts[idx] += value

        cost += 3.0 * squared_relative_error(patient_count, target_patients)
        cost += 5.0 * squared_relative_error(slice_count, target_slices)
        for idx in range(MASK_CLASSES):
            cost += 8.0 * squared_relative_error(class_counts[idx], target_classes[idx])
            cost += slice_count_balance_weight * squared_relative_error(bin_counts[idx], target_bin_counts[idx])
        cost += (slice_count_balance_weight * 1.5) * squared_relative_error(
            low_slice_count, target_low_slice
        )
    return cost


def make_folds(
    patients: Sequence[PatientRecord],
    seed: int,
    repeats: int,
    swap_rounds: int,
    slice_count_balance_weight: float,
) -> List[List[PatientRecord]]:
    rng = random.Random(seed)
    best_folds: List[List[PatientRecord]] = []
    best_cost = math.inf

    for repeat_idx in range(repeats):
        local_rng = random.Random(rng.randint(0, 10**9) + repeat_idx)
        order = list(patients)
        local_rng.shuffle(order)
        order.sort(
            key=lambda patient: (
                -patient.n_slices,
                -max(patient.class_counts),
                -sum(abs(value - (patient.n_slices / MASK_CLASSES)) for value in patient.class_counts),
                patient.patient_id,
            )
        )

        folds: List[List[PatientRecord]] = [[] for _ in range(N_FOLDS)]
        for patient in order:
            candidates = []
            for fold_idx in range(N_FOLDS):
                trial = [list(fold) for fold in folds]
                trial[fold_idx].append(patient)
                candidates.append(
                    (
                        fold_cost(trial, patients, slice_count_balance_weight),
                        len(folds[fold_idx]),
                        fold_idx,
                    )
                )
            _, _, best_idx = min(candidates)
            folds[best_idx].append(patient)

        folds = improve_by_swapping(folds, patients, local_rng, swap_rounds, slice_count_balance_weight)
        cost = fold_cost(folds, patients, slice_count_balance_weight)
        if cost < best_cost:
            best_cost = cost
            best_folds = [list(fold) for fold in folds]

    for fold in best_folds:
        fold.sort(key=lambda patient: patient.patient_id)
    return best_folds


def improve_by_swapping(
    folds: List[List[PatientRecord]],
    patients: Sequence[PatientRecord],
    rng: random.Random,
    rounds: int,
    slice_count_balance_weight: float,
) -> List[List[PatientRecord]]:
    best_cost = fold_cost(folds, patients, slice_count_balance_weight)
    for _ in range(rounds):
        improved = False
        fold_indices = list(range(N_FOLDS))
        rng.shuffle(fold_indices)
        for a_idx in fold_indices:
            for b_idx in fold_indices:
                if a_idx >= b_idx or not folds[a_idx] or not folds[b_idx]:
                    continue
                a_order = list(range(len(folds[a_idx])))
                b_order = list(range(len(folds[b_idx])))
                rng.shuffle(a_order)
                rng.shuffle(b_order)
                for pa_idx in a_order:
                    for pb_idx in b_order:
                        trial = [list(fold) for fold in folds]
                        trial[a_idx][pa_idx], trial[b_idx][pb_idx] = trial[b_idx][pb_idx], trial[a_idx][pa_idx]
                        cost = fold_cost(trial, patients, slice_count_balance_weight)
                        if cost + 1e-12 < best_cost:
                            folds = trial
                            best_cost = cost
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return folds


def infer_modalities(data_root: Path, label_dir_name: str, label_files: Sequence[str]) -> List[str]:
    label_set = set(label_files)
    modalities = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        files = {path.name for path in child.iterdir() if path.is_file()}
        if label_set.issubset(files):
            modalities.append(child.name)
    if label_dir_name not in modalities:
        modalities.append(label_dir_name)
    return sorted(set(modalities))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def patient_fold_map(folds: Sequence[Sequence[PatientRecord]]) -> Dict[str, int]:
    result = {}
    for fold_idx, fold in enumerate(folds, start=1):
        for patient in fold:
            result[patient.patient_id] = fold_idx
    return result


def write_outputs(
    output_root: Path,
    folds: Sequence[Sequence[PatientRecord]],
    slice_records: Sequence[SliceRecord],
    patients: Sequence[PatientRecord],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fold_by_patient = patient_fold_map(folds)

    write_csv(
        output_root / "slice_manifest.csv",
        ["file_name", "patient_id", "validation_fold", "mask_area", "area_class", "area_level"],
        (
            {
                "file_name": item.file_name,
                "patient_id": item.patient_id,
                "validation_fold": fold_by_patient[item.patient_id],
                "mask_area": item.area,
                "area_class": item.area_class,
                "area_level": AREA_LEVELS[item.area_class],
            }
            for item in slice_records
        ),
    )

    write_csv(
        output_root / "patient_summary.csv",
        ["patient_id", "validation_fold", "n_slices", "total_area", "mean_area"]
        + [f"class_{idx}_slices" for idx in range(MASK_CLASSES)],
        (
            {
                "patient_id": patient.patient_id,
                "validation_fold": fold_by_patient[patient.patient_id],
                "n_slices": patient.n_slices,
                "total_area": patient.total_area,
                "mean_area": f"{patient.mean_area:.4f}",
                **{f"class_{idx}_slices": patient.class_counts[idx] for idx in range(MASK_CLASSES)},
            }
            for patient in patients
        ),
    )

    summary_rows = []
    for fold_idx, val_patients in enumerate(folds, start=1):
        val_ids = {patient.patient_id for patient in val_patients}
        train_patients = [patient for patient in patients if patient.patient_id not in val_ids]
        for split_name, split_patients in [("train", train_patients), ("val", list(val_patients))]:
            class_counts = [0] * MASK_CLASSES
            for patient in split_patients:
                for idx, value in enumerate(patient.class_counts):
                    class_counts[idx] += value
            row = {
                "fold": fold_idx,
                "split": split_name,
                "patients": len(split_patients),
                "slices": sum(patient.n_slices for patient in split_patients),
                "patients_le_2_slices": sum(1 for patient in split_patients if patient.n_slices <= 2),
            }
            row.update({f"class_{idx}_slices": class_counts[idx] for idx in range(MASK_CLASSES)})
            summary_rows.append(row)

    write_csv(
        output_root / "fold_summary.csv",
        ["fold", "split", "patients", "slices", "patients_le_2_slices"]
        + [f"class_{idx}_slices" for idx in range(MASK_CLASSES)],
        summary_rows,
    )

    for fold_idx, val_patients in enumerate(folds, start=1):
        val_ids = {patient.patient_id for patient in val_patients}
        write_csv(
            output_root / f"fold_{fold_idx}.csv",
            ["file_name", "patient_id", "split", "validation_fold", "mask_area", "area_class", "area_level"],
            (
                {
                    "file_name": item.file_name,
                    "patient_id": item.patient_id,
                    "split": "val" if item.patient_id in val_ids else "train",
                    "validation_fold": fold_idx,
                    "mask_area": item.area,
                    "area_class": item.area_class,
                    "area_level": AREA_LEVELS[item.area_class],
                }
                for item in slice_records
            ),
        )


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported materialize mode: {mode}")


def materialize_files(
    data_root: Path,
    output_root: Path,
    modalities: Sequence[str],
    folds: Sequence[Sequence[PatientRecord]],
    mode: str,
) -> None:
    if mode == "none":
        return

    origin_root = output_root / "origin_wuzhe_data"
    all_patients = {patient.patient_id: patient for fold in folds for patient in fold}
    total_origin_files = sum(patient.n_slices for fold in folds for patient in fold) * len(modalities)
    total_combined_files = total_origin_files * N_FOLDS
    total_files = total_origin_files + total_combined_files
    done = 0
    print(f"Materializing files with mode={mode}: about {total_files} files ...")

    for fold_idx, val_patients in enumerate(folds, start=1):
        val_ids = {patient.patient_id for patient in val_patients}

        for patient in val_patients:
            for item in patient.slices:
                for modality in modalities:
                    src = data_root / modality / item.file_name
                    dst = origin_root / f"fold_{fold_idx}" / modality / item.file_name
                    link_or_copy(src, dst, mode)
                    done += 1
                    if done % 500 == 0 or done == total_files:
                        print(f"  {done}/{total_files} files done")

        splits = {
            "val": [patient for patient in all_patients.values() if patient.patient_id in val_ids],
            "train": [patient for patient in all_patients.values() if patient.patient_id not in val_ids],
        }
        for split_name, split_patients in splits.items():
            for patient in split_patients:
                for item in patient.slices:
                    for modality in modalities:
                        src = data_root / modality / item.file_name
                        dst = output_root / f"fold_{fold_idx}" / split_name / modality / item.file_name
                        link_or_copy(src, dst, mode)
                        done += 1
                        if done % 500 == 0 or done == total_files:
                            print(f"  {done}/{total_files} files done")


def print_summary(folds: Sequence[Sequence[PatientRecord]], patients: Sequence[PatientRecord]) -> None:
    total_patients = len(patients)
    total_slices = sum(patient.n_slices for patient in patients)
    print(f"Total patients: {total_patients}")
    print(f"Total slices:   {total_slices}")
    print("Validation folds:")
    for fold_idx, fold in enumerate(folds, start=1):
        class_counts = [0] * MASK_CLASSES
        for patient in fold:
            for idx, value in enumerate(patient.class_counts):
                class_counts[idx] += value
        patient_pct = len(fold) / total_patients * 100
        slice_pct = sum(patient.n_slices for patient in fold) / total_slices * 100
        low_slice = sum(1 for patient in fold if patient.n_slices <= 2)
        print(
            f"  fold_{fold_idx}: patients={len(fold)} ({patient_pct:.1f}%), "
            f"slices={sum(patient.n_slices for patient in fold)} ({slice_pct:.1f}%), "
            f"classes={class_counts}, patients_le_2_slices={low_slice}"
        )


def default_paths() -> Tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "wuzhe_data":
        project_root = script_dir.parent
    elif (script_dir / "origin_data").is_dir():
        project_root = script_dir.parent
    else:
        project_root = script_dir
    data_root = project_root / "data" / "origin_data"
    if not data_root.is_dir():
        data_root = script_dir / "origin_data"
    output_root = project_root / "wuzhe_data"
    return data_root, output_root


def parse_args() -> argparse.Namespace:
    data_root, output_root = default_paths()
    parser = argparse.ArgumentParser(description="Patient-level balanced 5-fold split with three tumor-area strata.")
    parser.add_argument("--data-root", type=Path, default=data_root, help="Dataset root directory.")
    parser.add_argument("--label-dir", default="label", help="Label/mask subdirectory name.")
    parser.add_argument("--output-root", type=Path, default=output_root, help="Output directory.")
    parser.add_argument(
        "--patient-regex",
        default=r"^(\d{3})",
        help="Regex used on file stem to extract patient id. The first capture group is used.",
    )
    parser.add_argument("--threshold", type=int, default=127, help="Foreground threshold for white mask pixels.")
    parser.add_argument("--seed", type=int, default=20260707, help="Random seed for reproducible splitting.")
    parser.add_argument("--repeats", type=int, default=30, help="Number of random restarts.")
    parser.add_argument("--swap-rounds", type=int, default=300, help="Pairwise swap optimization rounds per restart.")
    parser.add_argument(
        "--slice-count-balance-weight",
        type=float,
        default=1.0,
        help="Auxiliary penalty weight for balancing per-patient slice-count distribution. Use 0 to disable it.",
    )
    parser.add_argument(
        "--modalities",
        nargs="*",
        default=None,
        help="Subdirectories to materialize. Default: dirs containing all label file names.",
    )
    parser.add_argument(
        "--materialize",
        choices=["none", "copy", "hardlink"],
        default="copy",
        help="Create physical fold folders. CSV manifests are always written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    label_dir = data_root / args.label_dir
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

    label_files = sorted(path.name for path in label_dir.iterdir() if path.is_file())
    if not label_files:
        raise FileNotFoundError(f"No label files found in: {label_dir}")

    raw_items: List[Tuple[str, str, int]] = []
    for file_name in label_files:
        patient_id = extract_patient_id(file_name, args.patient_regex)
        area = read_mask_area(label_dir / file_name, args.threshold)
        raw_items.append((file_name, patient_id, area))

    slice_records = assign_area_classes(raw_items)
    patients = build_patient_records(slice_records)
    folds = make_folds(
        patients,
        seed=args.seed,
        repeats=args.repeats,
        swap_rounds=args.swap_rounds,
        slice_count_balance_weight=args.slice_count_balance_weight,
    )

    modalities = args.modalities
    if modalities is None:
        modalities = infer_modalities(data_root, args.label_dir, label_files)

    write_outputs(output_root, folds, slice_records, patients)
    materialize_files(data_root, output_root, modalities, folds, args.materialize)
    print_summary(folds, patients)
    print(f"Output written to: {output_root}")


if __name__ == "__main__":
    main()
