#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze and visualize the quality of the 5-fold patient-level split.

Default usage:
  python wuzhe_data/visualize_5fold_split_quality.py

Inputs, read from the same folder by default:
  slice_manifest.csv
  patient_summary.csv
  fold_summary.csv
  fold_1.csv ... fold_5.csv
  fold_1/train, fold_1/val ... if materialized
  origin_wuzhe_data/fold_1 ... fold_5 if materialized

Outputs:
  fivefold_quality_report.html
  fivefold_quality_summary.csv
  fivefold_quality_checks.csv

The script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


N_FOLDS = 5
SPLITS = ("train", "val")
AREA_CLASSES = (0, 1, 2)
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


def esc(value: object) -> str:
    return html.escape(str(value))


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def ratio(part: float, total: float) -> float:
    return part / total if total else 0.0


def derive_fold_summary(data_root: Path) -> List[Dict[str, str]]:
    rows = []
    for fold_idx in range(1, N_FOLDS + 1):
        fold_rows = read_csv(data_root / f"fold_{fold_idx}.csv")
        for split in SPLITS:
            split_rows = [row for row in fold_rows if row.get("split") == split]
            patients = sorted({row.get("patient_id", "") for row in split_rows})
            class_counts = {
                cls: sum(1 for row in split_rows if to_int(row.get("area_class")) == cls)
                for cls in AREA_CLASSES
            }
            row = {
                "fold": str(fold_idx),
                "split": split,
                "patients": str(len(patients)),
                "slices": str(len(split_rows)),
                "patients_le_2_slices": "",
            }
            row.update({f"class_{cls}_slices": str(class_counts[cls]) for cls in AREA_CLASSES})
            rows.append(row)
    return rows


def fold_rows_by_split(summary_rows: Sequence[Dict[str, str]]) -> Dict[int, Dict[str, Dict[str, int]]]:
    result: Dict[int, Dict[str, Dict[str, int]]] = defaultdict(dict)
    for row in summary_rows:
        fold = to_int(row.get("fold"))
        split = row.get("split", "")
        if split not in SPLITS:
            continue
        result[fold][split] = {
            "patients": to_int(row.get("patients")),
            "slices": to_int(row.get("slices")),
            "patients_le_2_slices": to_int(row.get("patients_le_2_slices")),
            **{f"class_{cls}": to_int(row.get(f"class_{cls}_slices")) for cls in AREA_CLASSES},
        }
    return result


def patient_fold_assignments(data_root: Path) -> Dict[str, set[int]]:
    assignments: Dict[str, set[int]] = defaultdict(set)
    manifest = read_csv(data_root / "slice_manifest.csv")
    if manifest:
        for row in manifest:
            assignments[row.get("patient_id", "")].add(to_int(row.get("validation_fold")))
        return assignments

    for fold_idx in range(1, N_FOLDS + 1):
        fold_rows = read_csv(data_root / f"fold_{fold_idx}.csv")
        for row in fold_rows:
            if row.get("split") == "val":
                assignments[row.get("patient_id", "")].add(fold_idx)
    return assignments


def modality_file_report(data_root: Path) -> List[Dict[str, object]]:
    report = []
    for fold_idx in range(1, N_FOLDS + 1):
        fold_rows = read_csv(data_root / f"fold_{fold_idx}.csv")
        if not fold_rows:
            continue
        for split in SPLITS:
            expected = {row.get("file_name", "") for row in fold_rows if row.get("split") == split}
            split_dir = data_root / f"fold_{fold_idx}" / split
            if not split_dir.is_dir():
                continue
            for modality_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                actual = {path.name for path in modality_dir.iterdir() if path.is_file()}
                report.append(
                    {
                        "fold": fold_idx,
                        "split": split,
                        "modality": modality_dir.name,
                        "expected": len(expected),
                        "actual": len(actual),
                        "missing": len(expected - actual),
                        "extra": len(actual - expected),
                    }
                )

        expected_origin = {row.get("file_name", "") for row in fold_rows if row.get("split") == "val"}
        origin_dir = data_root / "origin_wuzhe_data" / f"fold_{fold_idx}"
        if origin_dir.is_dir():
            for modality_dir in sorted(path for path in origin_dir.iterdir() if path.is_dir()):
                actual = {path.name for path in modality_dir.iterdir() if path.is_file()}
                report.append(
                    {
                        "fold": fold_idx,
                        "split": "origin_val",
                        "modality": modality_dir.name,
                        "expected": len(expected_origin),
                        "actual": len(actual),
                        "missing": len(expected_origin - actual),
                        "extra": len(actual - expected_origin),
                    }
                )
    return report


def bar_svg(labels: Sequence[str], values: Sequence[float], title: str, target: float | None = None) -> str:
    width, height = 940, 310
    margin_left, margin_right, margin_top, margin_bottom = 70, 28, 42, 58
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_value = max([1.0] + list(values) + ([target] if target else []))
    group_w = plot_w / max(1, len(labels))
    bar_w = min(48, group_w * 0.5)
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
    if target is not None:
        y = margin_top + plot_h - (target / max_value) * plot_h
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" class="target"/>')
        parts.append(f'<text x="{width - margin_right}" y="{y - 6:.1f}" text-anchor="end" class="target-label">target {target:.1f}</text>')
    for idx, label in enumerate(labels):
        value = values[idx]
        h = (value / max_value) * plot_h
        x = margin_left + group_w * idx + (group_w - bar_w) / 2
        y = margin_top + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{COLORS["val"]}" rx="2"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" class="bar-label">{value:.0f}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{height - 30}" text-anchor="middle" class="tick label">{esc(label)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def grouped_bar_svg(labels: Sequence[str], series: Dict[str, Sequence[float]], title: str) -> str:
    width, height = 980, 330
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 42, 62
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_value = max([1.0] + [value for values in series.values() for value in values])
    group_w = plot_w / max(1, len(labels))
    keys = list(series)
    bar_w = min(22, group_w / (len(keys) + 1))
    colors = ["#2878b5", "#c95f43", "#2f8f5b", "#d18b00", "#7851a9"]
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
    for label_idx, label in enumerate(labels):
        center = margin_left + group_w * label_idx + group_w / 2
        start = center - (bar_w * len(keys)) / 2
        for key_idx, key in enumerate(keys):
            value = series[key][label_idx]
            h = (value / max_value) * plot_h
            x = start + key_idx * bar_w
            y = margin_top + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 2:.1f}" height="{h:.1f}" fill="{colors[key_idx % len(colors)]}" rx="2"/>')
        parts.append(f'<text x="{center:.1f}" y="{height - 30}" text-anchor="middle" class="tick label">{esc(label)}</text>')
    legend_x = width - 310
    for idx, key in enumerate(keys):
        x = legend_x + idx * 100
        parts.append(f'<rect x="{x}" y="14" width="12" height="12" fill="{colors[idx % len(colors)]}"/><text x="{x + 18}" y="24" class="legend">{esc(key)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "".join(f"<th>{esc(item)}</th>" for item in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def badge(ok: bool, text: str) -> str:
    return f'<span class="badge {"ok" if ok else "bad"}">{esc(text)}</span>'


def analyze(data_root: Path, expected_val_ratio: float) -> Dict[str, object]:
    summary_rows = read_csv(data_root / "fold_summary.csv") or derive_fold_summary(data_root)
    if not summary_rows:
        raise FileNotFoundError(f"fold_summary.csv or fold_*.csv not found in: {data_root}")
    by_fold = fold_rows_by_split(summary_rows)
    assignments = patient_fold_assignments(data_root)
    file_rows = modality_file_report(data_root)

    total_patients = by_fold[1]["train"]["patients"] + by_fold[1]["val"]["patients"]
    total_slices = by_fold[1]["train"]["slices"] + by_fold[1]["val"]["slices"]
    total_classes = {
        cls: by_fold[1]["train"][f"class_{cls}"] + by_fold[1]["val"][f"class_{cls}"]
        for cls in AREA_CLASSES
    }
    total_low = by_fold[1]["train"].get("patients_le_2_slices", 0) + by_fold[1]["val"].get("patients_le_2_slices", 0)

    ratio_rows = []
    for fold_idx in range(1, N_FOLDS + 1):
        val = by_fold[fold_idx]["val"]
        ratio_rows.append(
            {
                "fold": fold_idx,
                "patients": val["patients"],
                "patient_ratio": ratio(val["patients"], total_patients),
                "slices": val["slices"],
                "slice_ratio": ratio(val["slices"], total_slices),
                "patients_le_2_slices": val.get("patients_le_2_slices", 0),
                "patients_le_2_slices_ratio": ratio(val.get("patients_le_2_slices", 0), total_low),
                **{
                    f"class_{cls}_slices": val[f"class_{cls}"]
                    for cls in AREA_CLASSES
                },
                **{
                    f"class_{cls}_ratio": ratio(val[f"class_{cls}"], total_classes[cls])
                    for cls in AREA_CLASSES
                },
            }
        )

    leakage = sorted(patient for patient, folds in assignments.items() if len(folds) != 1)
    missing_total = sum(to_int(row["missing"]) for row in file_rows)
    extra_total = sum(to_int(row["extra"]) for row in file_rows)

    checks = []
    for row in ratio_rows:
        fold = row["fold"]
        for key in ["patient_ratio", "slice_ratio", "class_0_ratio", "class_1_ratio", "class_2_ratio"]:
            value = float(row[key])
            checks.append(
                {
                    "check": f"fold_{fold}_{key}",
                    "value": pct(value),
                    "target": pct(expected_val_ratio),
                    "absolute_deviation": pct(abs(value - expected_val_ratio)),
                    "status": "pass" if abs(value - expected_val_ratio) <= 0.04 else "warn",
                }
            )
        value = float(row["patients_le_2_slices_ratio"])
        checks.append(
            {
                "check": f"fold_{fold}_patients_le_2_slices_ratio",
                "value": pct(value),
                "target": pct(expected_val_ratio),
                "absolute_deviation": pct(abs(value - expected_val_ratio)),
                "status": "pass" if abs(value - expected_val_ratio) <= 0.08 else "warn",
            }
        )
    checks.append(
        {
            "check": "patient_validation_fold_uniqueness",
            "value": len(leakage),
            "target": 0,
            "absolute_deviation": len(leakage),
            "status": "pass" if not leakage else "fail",
        }
    )
    checks.append(
        {
            "check": "physical_files",
            "value": f"missing={missing_total}, extra={extra_total}",
            "target": "missing=0, extra=0",
            "absolute_deviation": missing_total + extra_total,
            "status": "pass" if file_rows and missing_total == 0 and extra_total == 0 else "warn",
        }
    )

    return {
        "summary_rows": summary_rows,
        "by_fold": by_fold,
        "ratio_rows": ratio_rows,
        "checks": checks,
        "leakage": leakage,
        "file_rows": file_rows,
        "total_patients": total_patients,
        "total_slices": total_slices,
        "total_classes": total_classes,
        "total_low": total_low,
    }


def build_report(data_root: Path, analysis: Dict[str, object], expected_val_ratio: float) -> str:
    ratio_rows: List[Dict[str, object]] = analysis["ratio_rows"]  # type: ignore[assignment]
    checks: List[Dict[str, object]] = analysis["checks"]  # type: ignore[assignment]
    leakage: List[str] = analysis["leakage"]  # type: ignore[assignment]
    file_rows: List[Dict[str, object]] = analysis["file_rows"]  # type: ignore[assignment]

    labels = [f"fold_{row['fold']}" for row in ratio_rows]
    patient_values = [float(row["patients"]) for row in ratio_rows]
    slice_values = [float(row["slices"]) for row in ratio_rows]
    low_values = [float(row["patients_le_2_slices"]) for row in ratio_rows]
    class_series = {
        f"{cls} {AREA_LABELS[cls]}": [float(row[f"class_{cls}_slices"]) for row in ratio_rows]
        for cls in AREA_CLASSES
    }
    ratio_table_rows = []
    for row in ratio_rows:
        ratio_table_rows.append(
            [
                f"fold_{row['fold']}",
                row["patients"],
                pct(float(row["patient_ratio"])),
                row["slices"],
                pct(float(row["slice_ratio"])),
                row["class_0_slices"],
                pct(float(row["class_0_ratio"])),
                row["class_1_slices"],
                pct(float(row["class_1_ratio"])),
                row["class_2_slices"],
                pct(float(row["class_2_ratio"])),
                row["patients_le_2_slices"],
                pct(float(row["patients_le_2_slices_ratio"])),
            ]
        )

    file_table = (
        table(
            ["fold", "split", "modality", "expected", "actual", "missing", "extra"],
            [
                [row["fold"], row["split"], row["modality"], row["expected"], row["actual"], row["missing"], row["extra"]]
                for row in file_rows
            ],
        )
        if file_rows
        else "<p>No materialized fold folders were found.</p>"
    )
    check_table = table(
        ["check", "value", "target", "absolute deviation", "status"],
        [[row["check"], row["value"], row["target"], row["absolute_deviation"], row["status"]] for row in checks],
    )
    check_badges = " ".join(badge(row["status"] == "pass", f"{row['check']}: {row['status']}") for row in checks[-2:])
    leakage_text = "None" if not leakage else ", ".join(leakage[:100])
    total_patients = int(analysis["total_patients"])
    total_slices = int(analysis["total_slices"])

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Five-Fold Split Quality Report</title>
<style>
body {{ margin: 0; background: #f5f7fb; color: {COLORS['text']}; font-family: Arial, "Microsoft YaHei", sans-serif; }}
main {{ max-width: 1200px; margin: 0 auto; padding: 28px; }}
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
.target {{ stroke: #111827; stroke-width: 1.5; stroke-dasharray: 5 4; }}
.tick, .legend, .bar-label, .target-label {{ fill: #4b5563; font-size: 12px; }}
.label {{ font-size: 11px; }}
.footer {{ color: {COLORS['muted']}; font-size: 13px; margin-top: 30px; }}
@media (max-width: 760px) {{ .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} main {{ padding: 16px; }} }}
</style>
</head>
<body>
<main>
<h1>患者级五折交叉验证数据划分效果分析</h1>
<p>数据目录：{esc(data_root.resolve())}</p>
<p>检查目标：每折验证集约占 20%，并平衡患者数、切片数、三类肿瘤面积分布和患者切片数分布。</p>
<div>{check_badges}</div>
<div class="cards">
  <div class="card"><div class="card-label">Total patients</div><div class="card-value">{total_patients}</div></div>
  <div class="card"><div class="card-label">Total slices</div><div class="card-value">{total_slices}</div></div>
  <div class="card"><div class="card-label">Folds</div><div class="card-value">{N_FOLDS}</div></div>
  <div class="card"><div class="card-label">Target val ratio</div><div class="card-value">{pct(expected_val_ratio)}</div></div>
</div>

<h2>每折验证集比例</h2>
<div class="panel">{table(["fold", "patients", "patient ratio", "slices", "slice ratio", "small", "small ratio", "medium", "medium ratio", "large", "large ratio", "<=2 slice patients", "<=2 ratio"], ratio_table_rows)}</div>

<h2>验证集患者数与切片数</h2>
<div class="panel">{bar_svg(labels, patient_values, "Validation patient count by fold", target=total_patients / N_FOLDS)}</div>
<div class="panel">{bar_svg(labels, slice_values, "Validation slice count by fold", target=total_slices / N_FOLDS)}</div>

<h2>肿瘤区域三等级分布</h2>
<div class="panel">{grouped_bar_svg(labels, class_series, "Validation slice counts by tumor-area class")}</div>

<h2>低切片患者分布</h2>
<div class="panel">{bar_svg(labels, low_values, "Patients with <= 2 slices by validation fold", target=int(analysis["total_low"]) / N_FOLDS if int(analysis["total_low"]) else None)}</div>

<h2>实体文件检查</h2>
<div class="panel">{file_table}</div>

<h2>检查项</h2>
<div class="panel">{check_table}</div>

<h2>患者验证折唯一性检查</h2>
<div class="panel"><p>{esc(leakage_text)}</p></div>

<p class="footer">Generated by visualize_5fold_split_quality.py. Charts are inline SVG and require no external packages.</p>
</main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Analyze and visualize 5-fold split quality.")
    parser.add_argument("--data-root", type=Path, default=script_dir, help="Folder containing fold_summary.csv.")
    parser.add_argument("--output", type=Path, default=None, help="HTML output path.")
    parser.add_argument("--expected-val-ratio", type=float, default=0.2, help="Expected validation fold ratio.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output = (args.output or data_root / "fivefold_quality_report.html").resolve()
    analysis = analyze(data_root, args.expected_val_ratio)
    ratio_rows: List[Dict[str, object]] = analysis["ratio_rows"]  # type: ignore[assignment]
    checks: List[Dict[str, object]] = analysis["checks"]  # type: ignore[assignment]

    write_csv(
        data_root / "fivefold_quality_summary.csv",
        [
            "fold",
            "patients",
            "patient_ratio",
            "slices",
            "slice_ratio",
            "class_0_slices",
            "class_0_ratio",
            "class_1_slices",
            "class_1_ratio",
            "class_2_slices",
            "class_2_ratio",
            "patients_le_2_slices",
            "patients_le_2_slices_ratio",
        ],
        [
            {
                **row,
                "patient_ratio": pct(float(row["patient_ratio"])),
                "slice_ratio": pct(float(row["slice_ratio"])),
                "class_0_ratio": pct(float(row["class_0_ratio"])),
                "class_1_ratio": pct(float(row["class_1_ratio"])),
                "class_2_ratio": pct(float(row["class_2_ratio"])),
                "patients_le_2_slices_ratio": pct(float(row["patients_le_2_slices_ratio"])),
            }
            for row in ratio_rows
        ],
    )
    write_csv(
        data_root / "fivefold_quality_checks.csv",
        ["check", "value", "target", "absolute_deviation", "status"],
        checks,
    )
    output.write_text(build_report(data_root, analysis, args.expected_val_ratio), encoding="utf-8")
    print(f"Report written to: {output}")
    print(f"Summary written to: {data_root / 'fivefold_quality_summary.csv'}")
    print(f"Checks written to:  {data_root / 'fivefold_quality_checks.csv'}")


if __name__ == "__main__":
    main()
