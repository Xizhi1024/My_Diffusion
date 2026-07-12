#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze and visualize the patient-level 8:2 split quality.

Default usage:
  python main_data/visualize_split_quality.py

Inputs, read from the same folder by default:
  split.csv
  patient_summary.csv
  split_summary.csv
  train/<modality> and val/<modality> folders, if materialized

Outputs:
  split_quality_report.html
  split_quality_summary.csv
  split_quality_checks.csv

The script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


SPLITS = ("train", "val")
AREA_LABELS = {0: "small", 1: "medium", 2: "large"}
COLORS = {
    "train": "#2878b5",
    "val": "#c95f43",
    "ok": "#2f8f5b",
    "warn": "#d18b00",
    "bad": "#c43c39",
    "text": "#1f2933",
    "muted": "#6b7280",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def safe_pct(part: float, total: float) -> float:
    return part / total if total else 0.0


def pct_text(value: float) -> str:
    return f"{value * 100:.2f}%"


def esc(value: object) -> str:
    return html.escape(str(value))


def infer_area_classes(split_rows: Sequence[Dict[str, str]]) -> List[int]:
    classes = sorted({to_int(row.get("area_class")) for row in split_rows})
    return classes or [0, 1, 2]


def derive_patient_rows(split_rows: Sequence[Dict[str, str]], area_classes: Sequence[int]) -> List[Dict[str, str]]:
    by_patient: Dict[str, Dict[str, object]] = {}
    for row in split_rows:
        patient_id = row.get("patient_id", "")
        record = by_patient.setdefault(
            patient_id,
            {
                "patient_id": patient_id,
                "split": row.get("split", ""),
                "n_slices": 0,
                "total_area": 0,
                "class_counts": Counter(),
            },
        )
        record["n_slices"] = int(record["n_slices"]) + 1
        record["total_area"] = int(record["total_area"]) + to_int(row.get("mask_area"))
        record["class_counts"][to_int(row.get("area_class"))] += 1  # type: ignore[index]

    patient_rows = []
    for record in by_patient.values():
        n_slices = int(record["n_slices"])
        total_area = int(record["total_area"])
        class_counts: Counter = record["class_counts"]  # type: ignore[assignment]
        row = {
            "patient_id": str(record["patient_id"]),
            "split": str(record["split"]),
            "n_slices": str(n_slices),
            "total_area": str(total_area),
            "mean_area": f"{total_area / n_slices:.4f}" if n_slices else "0",
        }
        for cls in area_classes:
            row[f"class_{cls}_slices"] = str(class_counts.get(cls, 0))
        patient_rows.append(row)
    return sorted(patient_rows, key=lambda item: item["patient_id"])


def summarize(patient_rows: Sequence[Dict[str, str]], area_classes: Sequence[int]) -> Dict[str, Dict[str, int]]:
    summary = {
        split: {"patients": 0, "slices": 0, **{f"class_{cls}": 0 for cls in area_classes}}
        for split in SPLITS
    }
    for row in patient_rows:
        split = row.get("split", "")
        if split not in summary:
            continue
        summary[split]["patients"] += 1
        summary[split]["slices"] += to_int(row.get("n_slices"))
        for cls in area_classes:
            summary[split][f"class_{cls}"] += to_int(row.get(f"class_{cls}_slices"))
    return summary


def patient_leakage(split_rows: Sequence[Dict[str, str]]) -> List[str]:
    splits_by_patient: Dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        splits_by_patient[row.get("patient_id", "")].add(row.get("split", ""))
    return sorted(pid for pid, splits in splits_by_patient.items() if len(splits & set(SPLITS)) > 1)


def split_file_names(split_rows: Sequence[Dict[str, str]], split: str) -> set[str]:
    return {row.get("file_name", "") for row in split_rows if row.get("split") == split}


def modality_file_report(data_root: Path, split_rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    rows = []
    for split in SPLITS:
        expected = split_file_names(split_rows, split)
        split_dir = data_root / split
        if not split_dir.is_dir():
            continue
        for modality_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            actual = {path.name for path in modality_dir.iterdir() if path.is_file()}
            rows.append(
                {
                    "split": split,
                    "modality": modality_dir.name,
                    "expected": len(expected),
                    "actual": len(actual),
                    "missing": len(expected - actual),
                    "extra": len(actual - expected),
                }
            )
    return rows


def slice_count_distribution(patient_rows: Sequence[Dict[str, str]]) -> Dict[int, Dict[str, int]]:
    dist: Dict[int, Dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0})
    for row in patient_rows:
        split = row.get("split", "")
        if split in SPLITS:
            dist[to_int(row.get("n_slices"))][split] += 1
    return dict(sorted(dist.items()))


def patient_slice_strata(patient_rows: Sequence[Dict[str, str]], bins: int = 5) -> List[Dict[str, object]]:
    ordered = sorted(patient_rows, key=lambda row: (to_int(row.get("n_slices")), row.get("patient_id", "")))
    result = [{"label": f"Q{idx + 1}", "min": None, "max": None, "train": 0, "val": 0} for idx in range(bins)]
    if not ordered:
        return result
    for rank, row in enumerate(ordered):
        idx = min(bins - 1, (rank * bins) // len(ordered))
        n_slices = to_int(row.get("n_slices"))
        split = row.get("split", "")
        result[idx]["min"] = n_slices if result[idx]["min"] is None else min(int(result[idx]["min"]), n_slices)
        result[idx]["max"] = n_slices if result[idx]["max"] is None else max(int(result[idx]["max"]), n_slices)
        if split in SPLITS:
            result[idx][split] = int(result[idx][split]) + 1
    for item in result:
        item["range"] = f"{item['min']}-{item['max']}"
    return result


def bar_svg(
    labels: Sequence[str],
    train_values: Sequence[float],
    val_values: Sequence[float],
    title: str,
    width: int = 920,
    height: int = 320,
) -> str:
    if not labels:
        return ""
    margin_left, margin_right, margin_top, margin_bottom = 62, 26, 42, 64
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_value = max([1.0] + list(train_values) + list(val_values))
    group_w = plot_w / len(labels)
    bar_w = min(30, group_w * 0.32)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
        f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" class="chart-title">{esc(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" class="axis"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" class="axis"/>',
    ]
    for tick in range(5):
        value = max_value * tick / 4
        y = margin_top + plot_h - (value / max_value) * plot_h
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{margin_left - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value:.0f}</text>')
    for idx, label in enumerate(labels):
        center = margin_left + group_w * idx + group_w / 2
        for offset, split, values in [(-bar_w * 0.65, "train", train_values), (bar_w * 0.65, "val", val_values)]:
            value = values[idx]
            bar_h = (value / max_value) * plot_h
            x = center + offset - bar_w / 2
            y = margin_top + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{COLORS[split]}" rx="2"/>')
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" class="bar-label">{value:.0f}</text>')
        parts.append(f'<text x="{center:.1f}" y="{height - 32}" text-anchor="middle" class="tick label">{esc(label)}</text>')
    parts.append(f'<rect x="{width - 160}" y="14" width="12" height="12" fill="{COLORS["train"]}"/><text x="{width - 142}" y="24" class="legend">train</text>')
    parts.append(f'<rect x="{width - 88}" y="14" width="12" height="12" fill="{COLORS["val"]}"/><text x="{width - 70}" y="24" class="legend">val</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def ratio_svg(labels: Sequence[str], values: Sequence[float], expected: float, title: str) -> str:
    width, height = 920, max(250, 56 + 34 * len(labels))
    margin_left, margin_right, margin_top, margin_bottom = 168, 48, 46, 36
    plot_w = width - margin_left - margin_right
    row_h = (height - margin_top - margin_bottom) / max(1, len(labels))
    target_x = margin_left + plot_w * expected
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
        f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" class="chart-title">{esc(title)}</text>',
        f'<line x1="{target_x:.1f}" y1="{margin_top - 8}" x2="{target_x:.1f}" y2="{height - margin_bottom + 8}" class="target"/>',
        f'<text x="{target_x:.1f}" y="{margin_top - 15}" text-anchor="middle" class="target-label">target {pct_text(expected)}</text>',
    ]
    for idx, label in enumerate(labels):
        y = margin_top + row_h * idx + row_h * 0.25
        bar_h = row_h * 0.46
        value = values[idx]
        bar_w = plot_w * min(max(value, 0.0), 1.0)
        diff = abs(value - expected)
        color = COLORS["ok"] if diff <= 0.025 else COLORS["warn"] if diff <= 0.06 else COLORS["bad"]
        parts.append(f'<text x="{margin_left - 10}" y="{y + bar_h * 0.72:.1f}" text-anchor="end" class="tick label">{esc(label)}</text>')
        parts.append(f'<rect x="{margin_left}" y="{y:.1f}" width="{plot_w}" height="{bar_h:.1f}" class="bar-bg"/>')
        parts.append(f'<rect x="{margin_left}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="2"/>')
        parts.append(f'<text x="{margin_left + bar_w + 8:.1f}" y="{y + bar_h * 0.72:.1f}" class="bar-label">{pct_text(value)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "".join(f"<th>{esc(item)}</th>" for item in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def badge(ok: bool, text: str) -> str:
    return f'<span class="badge {"ok" if ok else "bad"}">{esc(text)}</span>'


def build_analysis(data_root: Path, expected_val_ratio: float, low_slice_threshold: int) -> Dict[str, object]:
    split_rows = read_csv(data_root / "split.csv")
    if not split_rows:
        raise FileNotFoundError(f"split.csv not found or empty: {data_root / 'split.csv'}")

    area_classes = infer_area_classes(split_rows)
    patient_rows = read_csv(data_root / "patient_summary.csv") or derive_patient_rows(split_rows, area_classes)
    summary = summarize(patient_rows, area_classes)
    leaks = patient_leakage(split_rows)
    modality_rows = modality_file_report(data_root, split_rows)

    total_patients = summary["train"]["patients"] + summary["val"]["patients"]
    total_slices = summary["train"]["slices"] + summary["val"]["slices"]
    low_counts = {
        split: sum(
            1 for row in patient_rows if row.get("split") == split and to_int(row.get("n_slices")) <= low_slice_threshold
        )
        for split in SPLITS
    }
    total_low = low_counts["train"] + low_counts["val"]

    ratio_rows: List[Dict[str, object]] = [
        {
            "criterion": "patients",
            "train": summary["train"]["patients"],
            "val": summary["val"]["patients"],
            "total": total_patients,
            "val_ratio": safe_pct(summary["val"]["patients"], total_patients),
        },
        {
            "criterion": "slices",
            "train": summary["train"]["slices"],
            "val": summary["val"]["slices"],
            "total": total_slices,
            "val_ratio": safe_pct(summary["val"]["slices"], total_slices),
        },
    ]
    for cls in area_classes:
        train_count = summary["train"][f"class_{cls}"]
        val_count = summary["val"][f"class_{cls}"]
        label = AREA_LABELS.get(cls, f"class_{cls}")
        ratio_rows.append(
            {
                "criterion": f"area_{cls}_{label}",
                "train": train_count,
                "val": val_count,
                "total": train_count + val_count,
                "val_ratio": safe_pct(val_count, train_count + val_count),
            }
        )
    ratio_rows.append(
        {
            "criterion": f"patients_le_{low_slice_threshold}_slices",
            "train": low_counts["train"],
            "val": low_counts["val"],
            "total": total_low,
            "val_ratio": safe_pct(low_counts["val"], total_low),
        }
    )

    checks = []
    for row in ratio_rows:
        value = float(row["val_ratio"])
        tolerance = 0.08 if str(row["criterion"]).startswith("patients_le_") else 0.04
        checks.append(
            {
                "check": str(row["criterion"]),
                "value": pct_text(value),
                "target": pct_text(expected_val_ratio),
                "absolute_deviation": pct_text(abs(value - expected_val_ratio)),
                "status": "pass" if abs(value - expected_val_ratio) <= tolerance else "warn",
            }
        )
    checks.append(
        {
            "check": "patient_leakage",
            "value": len(leaks),
            "target": 0,
            "absolute_deviation": len(leaks),
            "status": "pass" if not leaks else "fail",
        }
    )
    missing_total = sum(to_int(row["missing"]) for row in modality_rows)
    extra_total = sum(to_int(row["extra"]) for row in modality_rows)
    checks.append(
        {
            "check": "physical_files",
            "value": f"missing={missing_total}, extra={extra_total}",
            "target": "missing=0, extra=0",
            "absolute_deviation": missing_total + extra_total,
            "status": "pass" if modality_rows and missing_total == 0 and extra_total == 0 else "warn",
        }
    )

    return {
        "split_rows": split_rows,
        "patient_rows": patient_rows,
        "area_classes": area_classes,
        "summary": summary,
        "ratio_rows": ratio_rows,
        "checks": checks,
        "leaks": leaks,
        "modality_rows": modality_rows,
        "low_counts": low_counts,
        "total_patients": total_patients,
        "total_slices": total_slices,
    }


def build_report(data_root: Path, analysis: Dict[str, object], expected_val_ratio: float, low_slice_threshold: int) -> str:
    area_classes: List[int] = analysis["area_classes"]  # type: ignore[assignment]
    patient_rows: List[Dict[str, str]] = analysis["patient_rows"]  # type: ignore[assignment]
    summary: Dict[str, Dict[str, int]] = analysis["summary"]  # type: ignore[assignment]
    ratio_rows: List[Dict[str, object]] = analysis["ratio_rows"]  # type: ignore[assignment]
    checks: List[Dict[str, object]] = analysis["checks"]  # type: ignore[assignment]
    leaks: List[str] = analysis["leaks"]  # type: ignore[assignment]
    modality_rows: List[Dict[str, object]] = analysis["modality_rows"]  # type: ignore[assignment]

    total_patients = int(analysis["total_patients"])
    total_slices = int(analysis["total_slices"])
    val_patient_ratio = safe_pct(summary["val"]["patients"], total_patients)
    val_slice_ratio = safe_pct(summary["val"]["slices"], total_slices)

    check_badges = " ".join(
        badge(row["status"] == "pass", f"{row['check']}: {row['status']}") for row in checks
    )

    ratio_labels = [str(row["criterion"]) for row in ratio_rows]
    ratio_values = [float(row["val_ratio"]) for row in ratio_rows]

    overview_table = table(
        ["criterion", "train", "val", "total", "val ratio"],
        [
            [row["criterion"], row["train"], row["val"], row["total"], pct_text(float(row["val_ratio"]))]
            for row in ratio_rows
        ],
    )

    area_labels = [f"{cls} {AREA_LABELS.get(cls, '')}".strip() for cls in area_classes]
    area_train = [summary["train"][f"class_{cls}"] for cls in area_classes]
    area_val = [summary["val"][f"class_{cls}"] for cls in area_classes]

    dist = slice_count_distribution(patient_rows)
    dist_labels = [str(key) for key in dist.keys()]
    dist_train = [dist[key]["train"] for key in dist.keys()]
    dist_val = [dist[key]["val"] for key in dist.keys()]

    strata = patient_slice_strata(patient_rows)
    strata_labels = [f"{item['label']} ({item['range']})" for item in strata]
    strata_train = [int(item["train"]) for item in strata]
    strata_val = [int(item["val"]) for item in strata]

    modality_table = (
        table(["split", "modality", "expected", "actual", "missing", "extra"],
              [[row["split"], row["modality"], row["expected"], row["actual"], row["missing"], row["extra"]] for row in modality_rows])
        if modality_rows
        else "<p>No materialized train/val folders were found.</p>"
    )
    leakage_text = "None" if not leaks else ", ".join(leaks[:100])
    check_table = table(
        ["check", "value", "target", "absolute deviation", "status"],
        [[row["check"], row["value"], row["target"], row["absolute_deviation"], row["status"]] for row in checks],
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Main Data Split Quality Report</title>
<style>
body {{ margin: 0; background: #f5f7fb; color: {COLORS['text']}; font-family: Arial, "Microsoft YaHei", sans-serif; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 28px; }}
h2 {{ margin: 28px 0 12px; font-size: 20px; }}
p {{ color: {COLORS['muted']}; line-height: 1.6; }}
.cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }}
.card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px 16px; }}
.card-label {{ color: {COLORS['muted']}; font-size: 13px; }}
.card-value {{ font-size: 28px; font-weight: 700; margin-top: 4px; }}
.panel {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px; margin: 14px 0; overflow-x: auto; }}
.badge {{ display: inline-block; padding: 6px 10px; border-radius: 999px; color: #fff; font-size: 13px; margin: 4px 6px 4px 0; }}
.badge.ok {{ background: {COLORS['ok']}; }}
.badge.bad {{ background: {COLORS['bad']}; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; }}
th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; white-space: nowrap; }}
th {{ background: #f3f4f6; font-weight: 700; }}
.chart-title {{ font-size: 16px; font-weight: 700; fill: #111827; }}
.axis {{ stroke: #374151; stroke-width: 1; }}
.grid {{ stroke: #e5e7eb; stroke-width: 1; }}
.tick, .legend, .bar-label, .target-label {{ fill: #4b5563; font-size: 12px; }}
.label {{ font-size: 11px; }}
.target {{ stroke: #111827; stroke-width: 1.5; stroke-dasharray: 5 4; }}
.bar-bg {{ fill: #eef2f7; }}
.footer {{ color: {COLORS['muted']}; font-size: 13px; margin-top: 30px; }}
@media (max-width: 760px) {{ .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} main {{ padding: 16px; }} }}
</style>
</head>
<body>
<main>
<h1>主数据 8:2 患者级划分效果分析</h1>
<p>数据目录：{esc(data_root.resolve())}</p>
<p>检查目标：患者级 8:2、切片级 8:2、三类肿瘤区域大小均接近 8:2。</p>
<div>{check_badges}</div>
<div class="cards">
  <div class="card"><div class="card-label">Total patients</div><div class="card-value">{total_patients}</div></div>
  <div class="card"><div class="card-label">Total slices</div><div class="card-value">{total_slices}</div></div>
  <div class="card"><div class="card-label">Val patients</div><div class="card-value">{pct_text(val_patient_ratio)}</div></div>
  <div class="card"><div class="card-label">Val slices</div><div class="card-value">{pct_text(val_slice_ratio)}</div></div>
</div>

<h2>总体比例</h2>
<div class="panel">{ratio_svg(ratio_labels, ratio_values, expected_val_ratio, "Validation ratio by criterion")}</div>
<div class="panel">{overview_table}</div>

<h2>肿瘤区域大小三等级分布</h2>
<div class="panel">{bar_svg(area_labels, area_train, area_val, "Tumor-area class slice counts")}</div>

<h2>患者切片数量分布</h2>
<div class="panel">{bar_svg(dist_labels, dist_train, dist_val, "Patient count by slice number", width=1020, height=360)}</div>
<div class="panel">{bar_svg(strata_labels, strata_train, strata_val, "Patient slice-count strata")}</div>

<h2>实体文件检查</h2>
<div class="panel">{modality_table}</div>

<h2>检查项</h2>
<div class="panel">{check_table}</div>

<h2>患者泄漏检查</h2>
<div class="panel"><p>{esc(leakage_text)}</p></div>

<p class="footer">Generated by visualize_split_quality.py. Charts are inline SVG and require no external packages.</p>
</main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Analyze and visualize main_data split quality.")
    parser.add_argument("--data-root", type=Path, default=script_dir, help="Folder containing split.csv.")
    parser.add_argument("--output", type=Path, default=None, help="HTML output path.")
    parser.add_argument("--expected-val-ratio", type=float, default=0.2, help="Expected validation ratio.")
    parser.add_argument(
        "--low-slice-threshold",
        type=int,
        default=2,
        help="Patients with <= this many slices are counted separately.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output = (args.output or data_root / "split_quality_report.html").resolve()

    analysis = build_analysis(data_root, args.expected_val_ratio, args.low_slice_threshold)
    ratio_rows: List[Dict[str, object]] = analysis["ratio_rows"]  # type: ignore[assignment]
    checks: List[Dict[str, object]] = analysis["checks"]  # type: ignore[assignment]

    write_csv(
        data_root / "split_quality_summary.csv",
        ["criterion", "train", "val", "total", "val_ratio"],
        [
            {
                **row,
                "val_ratio": pct_text(float(row["val_ratio"])),
            }
            for row in ratio_rows
        ],
    )
    write_csv(
        data_root / "split_quality_checks.csv",
        ["check", "value", "target", "absolute_deviation", "status"],
        checks,
    )

    report = build_report(data_root, analysis, args.expected_val_ratio, args.low_slice_threshold)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Report written to: {output}")
    print(f"Summary written to: {data_root / 'split_quality_summary.csv'}")
    print(f"Checks written to:  {data_root / 'split_quality_checks.csv'}")


if __name__ == "__main__":
    main()
