#!/usr/bin/env python
"""按新数据集 (Data/data) 的患者级 train/val 分布，重新标注老数据集的 split_manifest.csv。

读取 Data/data/patient_split_summary.csv 作为 患者->split 的权威映射，
遍历现有 cache/split_manifest.csv，**仅更新 split 列**（其余列原样保留）。
不碰源码、不动物理文件、完全可逆（原文件备份为 .bak）。
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "Data" / "data" / "patient_split_summary.csv"
SPLIT_CSV = ROOT / "cache" / "split_manifest.csv"
FALLBACK_SPLIT = "train"  # 老数据集里存在但新数据集映射缺失的患者


def load_patient_split(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pid = row["patient_id"].strip()
            mapping[pid] = row["split"].strip()
    return mapping


def main() -> int:
    if not SUMMARY.exists():
        print(f"[ERROR] 新数据集清单不存在: {SUMMARY}", file=sys.stderr)
        return 1
    if not SPLIT_CSV.exists():
        print(f"[ERROR] 老数据集清单不存在: {SPLIT_CSV}", file=sys.stderr)
        return 1

    patient_split = load_patient_split(SUMMARY)
    n_new_train = sum(1 for v in patient_split.values() if v == "train")
    n_new_val = sum(1 for v in patient_split.values() if v == "val")
    print(f"[info] 新数据集患者映射: train={n_new_train}, val={n_new_val}")

    bak = SPLIT_CSV.with_suffix(".csv.bak")
    shutil.copy2(SPLIT_CSV, bak)
    print(f"[info] 已备份原文件 -> {bak}")

    with open(SPLIT_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    unknown: dict[str, int] = defaultdict(int)
    old_counts: dict[str, int] = defaultdict(int)
    new_counts: dict[str, int] = defaultdict(int)
    per_patient: dict[str, set] = defaultdict(set)

    for r in rows:
        pid = r["patient_id"].strip()
        old_counts[r["split"]] += 1
        sp = patient_split.get(pid)
        if sp is None:
            unknown[pid] += 1
            sp = FALLBACK_SPLIT
        r["split"] = sp
        new_counts[sp] += 1
        per_patient[pid].add(sp)

    if unknown:
        print(
            f"[WARN] {len(unknown)} 个老数据集患者不在新数据集映射中，"
            f"默认归 {FALLBACK_SPLIT}: {sorted(unknown)}",
            file=sys.stderr,
        )

    multi = {p: s for p, s in per_patient.items() if len(s) > 1}
    assert not multi, f"患者跨 split: {multi}"

    with open(SPLIT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    train_p = sorted({r["patient_id"] for r in rows if r["split"] == "train"})
    val_p = sorted({r["patient_id"] for r in rows if r["split"] == "val"})
    test_p = sorted({r["patient_id"] for r in rows if r["split"] == "test"})

    payload = json.dumps(
        sorted((r["sample_id"], r["patient_id"], r["split"]) for r in rows),
        sort_keys=True,
    )
    fp = hashlib.sha256(payload.encode()).hexdigest()[:12]

    stats = {
        "total_samples": len(rows),
        "total_patients": len(train_p) + len(val_p) + len(test_p),
        "train_patients": len(train_p),
        "val_patients": len(val_p),
        "test_patients": len(test_p),
        "train_samples": new_counts["train"],
        "val_samples": new_counts["val"],
        "test_samples": new_counts["test"],
        "source": "resynced to Data/data/patient_split_summary.csv",
        "fingerprint": fp,
        "train_patient_ids": train_p,
        "val_patient_ids": val_p,
        "test_patient_ids": test_p,
        "unknown_patients_defaulted_to_train": sorted(unknown),
    }
    with open(SPLIT_CSV.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n[done] 重新划分完成: {SPLIT_CSV}")
    print("  旧分布: " + ", ".join(f"{k}={v}" for k, v in sorted(old_counts.items())))
    print("  新分布: " + ", ".join(f"{k}={v}" for k, v in sorted(new_counts.items())))
    print(f"  患者: train={len(train_p)}, val={len(val_p)}, test={len(test_p)}")
    print(f"  fingerprint: {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
