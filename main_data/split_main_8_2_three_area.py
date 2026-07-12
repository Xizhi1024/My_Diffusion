#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Patient-level 8:2 split for the main dataset with three tumor-area strata.

Split rules:
  1. Patient-level split: one patient's slices appear only in train or val.
  2. Patient count is balanced close to 8:2.
  3. Slice count is balanced close to 8:2.
  4. White mask area in label masks is divided into 3 equal-frequency classes,
     and each class is balanced close to 8:2.
  5. Per-patient slice-count distribution is used as an auxiliary balance term.

Default input:
  <project_root>/data/origin_data

Default output:
  <project_root>/main_data

Example:
  python main_data/split_main_8_2_three_area.py
  python main_data/split_main_8_2_three_area.py --materialize copy
  python main_data/split_main_8_2_three_area.py --materialize hardlink
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
    """Count foreground pixels. Uses Pillow if installed, otherwise a PNG reader."""
    try:
        from PIL import Image  # type: ignore

        with Image.open(mask_path) as img:
            img = img.convert("L")
            hist = img.histogram()
            return sum(hist[threshold + 1 :])
    except ImportError:
        return read_png_area_stdlib(mask_path, threshold)


def read_png_area_stdlib(mask_path: Path, threshold: int) -> int:
    """Count foreground pixels for common non-interlaced 8-bit PNG masks."""
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
    """Assign equal-frequency classes by mask area: 0=small, 1=medium, 2=large."""
    sorted_items = sorted(items, key=lambda x: (x[2], x[0]))
    n = len(sorted_items)
    class_by_name: Dict[str, int] = {}
    for rank, (file_name, _, _) in enumerate(sorted_items):
        class_by_name[file_name] = min(MASK_CLASSES - 1, (rank * MASK_CLASSES) // n)

    return sorted(
        [
            SliceRecord(file_name=file_name, patient_id=pid, area=area, area_class=class_by_name[file_name])
            for file_name, pid, area in items
        ],
        key=lambda x: (x.patient_id, x.file_name),
    )


def build_patient_records(slice_records: Sequence[SliceRecord]) -> List[PatientRecord]:
    by_patient: Dict[str, PatientRecord] = {}
    for item in slice_records:
        by_patient.setdefault(item.patient_id, PatientRecord(item.patient_id)).slices.append(item)
    return sorted(by_patient.values(), key=lambda x: x.patient_id)


def split_cost(
    val_patients: Sequence[PatientRecord],
    total_patients: Sequence[PatientRecord],
    val_ratio: float,
    slice_count_balance_weight: float,
) -> float:
    val_ids = {p.patient_id for p in val_patients}
    train_patients = [p for p in total_patients if p.patient_id not in val_ids]

    total_patient_count = len(total_patients)
    total_slice_count = sum(p.n_slices for p in total_patients)
    total_classes = [0] * MASK_CLASSES
    for patient in total_patients:
        for idx, value in enumerate(patient.class_counts):
            total_classes[idx] += value

    val_patient_count = len(val_patients)
    val_slice_count = sum(p.n_slices for p in val_patients)
    val_classes = [0] * MASK_CLASSES
    for patient in val_patients:
        for idx, value in enumerate(patient.class_counts):
            val_classes[idx] += value

    train_patient_count = len(train_patients)
    train_slice_count = sum(p.n_slices for p in train_patients)
    train_classes = [total_classes[i] - val_classes[i] for i in range(MASK_CLASSES)]

    target_val_patients = total_patient_count * val_ratio
    target_train_patients = total_patient_count * (1 - val_ratio)
    target_val_slices = total_slice_count * val_ratio
    target_train_slices = total_slice_count * (1 - val_ratio)
    target_val_classes = [value * val_ratio for value in total_classes]
    target_train_classes = [value * (1 - val_ratio) for value in total_classes]
    slice_count_bin_by_patient = make_slice_count_bins(total_patients)
    total_slice_count_bins = [0] * MASK_CLASSES
    val_slice_count_bins = [0] * MASK_CLASSES
    train_slice_count_bins = [0] * MASK_CLASSES
    for patient in total_patients:
        total_slice_count_bins[slice_count_bin_by_patient[patient.patient_id]] += 1
    for patient in val_patients:
        val_slice_count_bins[slice_count_bin_by_patient[patient.patient_id]] += 1
    for patient in train_patients:
        train_slice_count_bins[slice_count_bin_by_patient[patient.patient_id]] += 1

    low_slice_total = sum(1 for patient in total_patients if patient.n_slices <= 2)
    low_slice_val = sum(1 for patient in val_patients if patient.n_slices <= 2)
    low_slice_train = low_slice_total - low_slice_val
    target_low_slice_val = low_slice_total * val_ratio
    target_low_slice_train = low_slice_total * (1 - val_ratio)

    cost = 0.0
    cost += 3.0 * squared_relative_error(val_patient_count, target_val_patients)
    cost += 1.5 * squared_relative_error(train_patient_count, target_train_patients)
    cost += 5.0 * squared_relative_error(val_slice_count, target_val_slices)
    cost += 2.5 * squared_relative_error(train_slice_count, target_train_slices)
    for idx in range(MASK_CLASSES):
        cost += 8.0 * squared_relative_error(val_classes[idx], target_val_classes[idx])
        cost += 4.0 * squared_relative_error(train_classes[idx], target_train_classes[idx])
    for idx, total_count in enumerate(total_slice_count_bins):
        cost += slice_count_balance_weight * squared_relative_error(
            val_slice_count_bins[idx], total_count * val_ratio
        )
        cost += (slice_count_balance_weight / 2) * squared_relative_error(
            train_slice_count_bins[idx], total_count * (1 - val_ratio)
        )
    cost += (slice_count_balance_weight * 1.5) * squared_relative_error(
        low_slice_val, target_low_slice_val
    )
    cost += (slice_count_balance_weight * 0.75) * squared_relative_error(
        low_slice_train, target_low_slice_train
    )
    return cost


def make_slice_count_bins(patients: Sequence[PatientRecord]) -> Dict[str, int]:
    """Put patients into quantile bins by number of slices for auxiliary balancing."""
    ordered = sorted(patients, key=lambda patient: (patient.n_slices, patient.patient_id))
    bins: Dict[str, int] = {}
    for rank, patient in enumerate(ordered):
        bins[patient.patient_id] = min(MASK_CLASSES - 1, (rank * MASK_CLASSES) // max(1, len(ordered)))
    return bins


def squared_relative_error(value: float, target: float) -> float:
    denom = max(target, 1.0)
    return ((value - target) / denom) ** 2


def make_single_split(
    patients: Sequence[PatientRecord],
    val_ratio: float,
    seed: int,
    repeats: int,
    swap_rounds: int,
    slice_count_balance_weight: float,
) -> Tuple[List[PatientRecord], List[PatientRecord]]:
    target_val_count = round(len(patients) * val_ratio)
    rng = random.Random(seed)
    best_val: List[PatientRecord] = []
    best_cost = math.inf

    for repeat_idx in range(repeats):
        local_rng = random.Random(rng.randint(0, 10**9) + repeat_idx)
        order = list(patients)
        local_rng.shuffle(order)
        order.sort(
            key=lambda p: (
                -p.n_slices,
                -max(p.class_counts),
                -sum(abs(c - (p.n_slices / MASK_CLASSES)) for c in p.class_counts),
                p.patient_id,
            )
        )

        val: List[PatientRecord] = []
        for index, patient in enumerate(order):
            if len(val) >= target_val_count:
                continue
            take_trial = val + [patient]
            skip_trial = list(val)
            remaining_slots = target_val_count - len(val)
            remaining_patients = len(order) - index
            if remaining_patients <= remaining_slots:
                val = take_trial
            elif split_cost(take_trial, patients, val_ratio, slice_count_balance_weight) <= split_cost(
                skip_trial, patients, val_ratio, slice_count_balance_weight
            ):
                val = take_trial

        if len(val) != target_val_count:
            selected = {p.patient_id for p in val}
            rest = [p for p in order if p.patient_id not in selected]
            val.extend(rest[: target_val_count - len(val)])

        val = improve_single_split(
            val,
            patients,
            val_ratio,
            local_rng,
            swap_rounds,
            slice_count_balance_weight,
        )
        current_cost = split_cost(val, patients, val_ratio, slice_count_balance_weight)
        if current_cost < best_cost:
            best_cost = current_cost
            best_val = list(val)

    val_ids = {p.patient_id for p in best_val}
    train = [p for p in patients if p.patient_id not in val_ids]
    return sorted(train, key=lambda p: p.patient_id), sorted(best_val, key=lambda p: p.patient_id)


def improve_single_split(
    val: List[PatientRecord],
    patients: Sequence[PatientRecord],
    val_ratio: float,
    rng: random.Random,
    rounds: int,
    slice_count_balance_weight: float,
) -> List[PatientRecord]:
    val_ids = {p.patient_id for p in val}
    train = [p for p in patients if p.patient_id not in val_ids]
    best_cost = split_cost(val, patients, val_ratio, slice_count_balance_weight)

    for _ in range(rounds):
        improved = False
        val_order = list(range(len(val)))
        train_order = list(range(len(train)))
        rng.shuffle(val_order)
        rng.shuffle(train_order)
        for vi in val_order:
            for ti in train_order:
                trial_val = list(val)
                trial_train = list(train)
                trial_val[vi], trial_train[ti] = trial_train[ti], trial_val[vi]
                current_cost = split_cost(trial_val, patients, val_ratio, slice_count_balance_weight)
                if current_cost + 1e-12 < best_cost:
                    val = trial_val
                    train = trial_train
                    best_cost = current_cost
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return val


def infer_modalities(data_root: Path, label_dir_name: str, label_files: Sequence[str]) -> List[str]:
    label_set = set(label_files)
    modalities = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        files = {p.name for p in child.iterdir() if p.is_file()}
        if label_set.issubset(files):
            modalities.append(child.name)
    if label_dir_name not in modalities:
        modalities.append(label_dir_name)
    return sorted(set(modalities))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(
    output_root: Path,
    slice_records: Sequence[SliceRecord],
    patients: Sequence[PatientRecord],
    train_patients: Sequence[PatientRecord],
    val_patients: Sequence[PatientRecord],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    val_ids = {p.patient_id for p in val_patients}

    write_csv(
        output_root / "split.csv",
        ["file_name", "patient_id", "split", "mask_area", "area_class", "area_level"],
        (
            {
                "file_name": item.file_name,
                "patient_id": item.patient_id,
                "split": "val" if item.patient_id in val_ids else "train",
                "mask_area": item.area,
                "area_class": item.area_class,
                "area_level": ["small", "medium", "large"][item.area_class],
            }
            for item in slice_records
        ),
    )

    write_csv(
        output_root / "patient_summary.csv",
        ["patient_id", "split", "n_slices", "total_area", "mean_area"]
        + [f"class_{i}_slices" for i in range(MASK_CLASSES)],
        (
            {
                "patient_id": patient.patient_id,
                "split": "val" if patient.patient_id in val_ids else "train",
                "n_slices": patient.n_slices,
                "total_area": patient.total_area,
                "mean_area": f"{patient.mean_area:.4f}",
                **{f"class_{i}_slices": patient.class_counts[i] for i in range(MASK_CLASSES)},
            }
            for patient in patients
        ),
    )

    summary_rows = []
    for split_name, split_patients in [("train", train_patients), ("val", val_patients)]:
        class_counts = [0] * MASK_CLASSES
        for patient in split_patients:
            for idx, value in enumerate(patient.class_counts):
                class_counts[idx] += value
        row = {
            "split": split_name,
            "patients": len(split_patients),
            "slices": sum(p.n_slices for p in split_patients),
        }
        row.update({f"class_{i}_slices": class_counts[i] for i in range(MASK_CLASSES)})
        summary_rows.append(row)

    write_csv(
        output_root / "split_summary.csv",
        ["split", "patients", "slices"] + [f"class_{i}_slices" for i in range(MASK_CLASSES)],
        summary_rows,
    )


def materialize_files(
    data_root: Path,
    output_root: Path,
    modalities: Sequence[str],
    train_patients: Sequence[PatientRecord],
    val_patients: Sequence[PatientRecord],
    mode: str,
) -> None:
    if mode == "none":
        return

    splits = {"train": train_patients, "val": val_patients}
    total_files = sum(p.n_slices for p in train_patients + val_patients) * len(modalities)
    done = 0
    print(f"Materializing files with mode={mode}: about {total_files} files ...")

    for split_name, split_patients in splits.items():
        for patient in split_patients:
            for item in patient.slices:
                for modality in modalities:
                    src = data_root / modality / item.file_name
                    dst = output_root / split_name / modality / item.file_name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists():
                        done += 1
                        continue
                    if mode == "hardlink":
                        try:
                            os.link(src, dst)
                        except OSError:
                            shutil.copy2(src, dst)
                    elif mode == "copy":
                        shutil.copy2(src, dst)
                    else:
                        raise ValueError(f"Unsupported materialize mode: {mode}")
                    done += 1
                    if done % 500 == 0 or done == total_files:
                        print(f"  {done}/{total_files} files done")


def print_summary(train_patients: Sequence[PatientRecord], val_patients: Sequence[PatientRecord]) -> None:
    all_patients = list(train_patients) + list(val_patients)
    total_patients = len(all_patients)
    total_slices = sum(p.n_slices for p in all_patients)
    print(f"Total patients: {total_patients}")
    print(f"Total slices:   {total_slices}")
    for split_name, split_patients in [("train", train_patients), ("val", val_patients)]:
        class_counts = [0] * MASK_CLASSES
        for patient in split_patients:
            for idx, value in enumerate(patient.class_counts):
                class_counts[idx] += value
        patient_pct = len(split_patients) / total_patients * 100
        slice_pct = sum(p.n_slices for p in split_patients) / total_slices * 100
        class_text = ", ".join(f"class_{i}={value}" for i, value in enumerate(class_counts))
        print(
            f"{split_name}: patients={len(split_patients)} ({patient_pct:.1f}%), "
            f"slices={sum(p.n_slices for p in split_patients)} ({slice_pct:.1f}%), "
            f"{class_text}"
        )


def default_paths() -> Tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "main_data":
        project_root = script_dir.parent
    elif (script_dir / "origin_data").is_dir():
        project_root = script_dir.parent
    else:
        project_root = script_dir
    data_root = project_root / "data" / "origin_data"
    if not data_root.is_dir():
        data_root = script_dir / "origin_data"
    output_root = project_root / "main_data"
    return data_root, output_root


def parse_args() -> argparse.Namespace:
    data_root, output_root = default_paths()
    parser = argparse.ArgumentParser(description="Patient-level 8:2 split with three tumor-area strata.")
    parser.add_argument("--data-root", type=Path, default=data_root, help="Dataset root directory.")
    parser.add_argument("--label-dir", default="label", help="Label/mask subdirectory name.")
    parser.add_argument("--output-root", type=Path, default=output_root, help="Output directory.")
    parser.add_argument(
        "--patient-regex",
        default=r"^(\d{3})",
        help="Regex used on file stem to extract patient id. The first capture group is used.",
    )
    parser.add_argument("--threshold", type=int, default=127, help="Foreground threshold for white mask pixels.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio. Default is 0.2.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splitting.")
    parser.add_argument("--repeats", type=int, default=100, help="Number of random restarts.")
    parser.add_argument("--swap-rounds", type=int, default=3000, help="Pairwise swap optimization rounds per restart.")
    parser.add_argument(
        "--slice-count-balance-weight",
        type=float,
        default=1.0,
        help=(
            "Auxiliary penalty weight for balancing per-patient slice-count distribution. "
            "Use 0 to disable it."
        ),
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
        help="Create physical train/val folders. Use none to write CSV only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    label_dir = data_root / args.label_dir
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

    label_files = sorted(p.name for p in label_dir.iterdir() if p.is_file())
    if not label_files:
        raise FileNotFoundError(f"No label files found in: {label_dir}")

    raw_items: List[Tuple[str, str, int]] = []
    for file_name in label_files:
        patient_id = extract_patient_id(file_name, args.patient_regex)
        area = read_mask_area(label_dir / file_name, args.threshold)
        raw_items.append((file_name, patient_id, area))

    slice_records = assign_area_classes(raw_items)
    patients = build_patient_records(slice_records)
    train_patients, val_patients = make_single_split(
        patients=patients,
        val_ratio=args.val_ratio,
        seed=args.seed,
        repeats=args.repeats,
        swap_rounds=args.swap_rounds,
        slice_count_balance_weight=args.slice_count_balance_weight,
    )

    modalities = args.modalities
    if modalities is None:
        modalities = infer_modalities(data_root, args.label_dir, label_files)

    write_outputs(output_root, slice_records, patients, train_patients, val_patients)
    materialize_files(data_root, output_root, modalities, train_patients, val_patients, args.materialize)
    print_summary(train_patients, val_patients)
    print(f"Output written to: {output_root}")


if __name__ == "__main__":
    main()
