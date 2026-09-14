"""Run all comparison experiments on main_data and five-fold wuzhe_data.

Protocol:
  1. Main experiment: train/evaluate every model on main_data.
  2. Five-fold CV: train/evaluate every model on wuzhe_data/fold_1..fold_5.

Each run trains on train split, selects ckpt_best.pt by validation loss, and
computes final metrics on the validation split.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ["pix2pix", "cyclegan", "reggan", "cpdm", "district_gan"]


def _config_for(model: str) -> Path:
    return PROJECT_ROOT / "comparison_experiments" / "configs" / f"{model}.yaml"


def _run(cmd: List[str], *, dry_run: bool = False) -> None:
    print("\n" + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def _load_summary(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _latest_checkpoint(exp_name: str) -> Path:
    ckpt_dir = PROJECT_ROOT / "checkpoints" / exp_name
    best = ckpt_dir / "ckpt_best.pt"
    if best.exists():
        return best
    candidates = sorted(ckpt_dir.glob("ckpt_epoch*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}")
    return candidates[-1]


def _train_one(
    model: str,
    exp_name: str,
    png_root: Path,
    args: argparse.Namespace,
) -> Path:
    config = _config_for(model)
    if not config.exists():
        raise FileNotFoundError(f"Config not found: {config}")
    cmd = [
        sys.executable,
        "comparison_experiments/train_comparison.py",
        "--config", str(config),
        "--override", f"experiment.name={exp_name}",
        "--override", f"data.png_root={png_root}",
        "--override", "data.cache_dir=",
        "--override", "data.split_manifest=",
        "--override", "data.train_split_name=train",
        "--override", "data.val_split_name=val",
        "--override", "data.png_ct_dir=ct",
        "--override", f"data.png_pet_dir={args.pet_dir}",
        "--override", "data.png_label_dir=label",
        "--override", "data.png_require_label=true",
        "--override", f"data.image_size={args.image_size}",
        "--override", f"data.batch_size={args.batch_size}",
        "--override", f"data.val_batch_size={args.val_batch_size}",
        "--override", f"training.num_epochs={args.epochs}",
        "--override", f"runtime.eval_interval={args.eval_interval}",
        "--override", f"runtime.save_interval={args.save_interval}",
        "--override", f"runtime.sample_interval={args.sample_interval}",
        "--override", f"runtime.num_workers={args.num_workers}",
        "--override", "training.early_stopping.enabled=true",
        "--override", f"training.early_stopping.patience={args.early_stop_patience}",
        "--override", f"training.early_stopping.min_delta={args.early_stop_min_delta}",
        "--override", f"training.early_stopping.warmup_epochs={args.early_stop_warmup}",
    ]
    if args.no_amp:
        cmd += ["--override", "runtime.amp=false"]
    _run(cmd, dry_run=args.dry_run)
    return PROJECT_ROOT / "checkpoints" / exp_name / "ckpt_best.pt"


def _evaluate_one(
    exp_name: str,
    png_root: Path,
    args: argparse.Namespace,
) -> Dict:
    ckpt = Path("DRY_RUN_CKPT.pt") if args.dry_run else _latest_checkpoint(exp_name)
    out_dir = PROJECT_ROOT / args.output_root / exp_name
    cmd = [
        sys.executable,
        "comparison_experiments/evaluate_comparison.py",
        "--checkpoint", str(ckpt),
        "--png-root", str(png_root),
        "--split", "val",
        "--pet-dir", args.pet_dir,
        "--image-size", str(args.image_size),
        "--batch-size", str(args.val_batch_size),
        "--output-dir", str(out_dir),
    ]
    if args.device:
        cmd += ["--device", args.device]
    _run(cmd, dry_run=args.dry_run)
    if args.dry_run:
        return {"experiment": exp_name}
    summary = _load_summary(out_dir / "metrics_summary.json")
    summary["experiment"] = exp_name
    return summary


def _run_train_eval(model: str, exp_name: str, root: Path, phase: str, fold: str, args: argparse.Namespace) -> Dict:
    print(f"\n===== {phase} | {fold} | {model} =====")
    _train_one(model, exp_name, root, args)
    summary = _evaluate_one(exp_name, root, args)
    summary.update({"phase": phase, "fold": fold, "model": model})
    return summary


def _write_aggregate(rows: List[Dict], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "all_metrics_summary.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(output_root / "all_metrics_summary.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run main + five-fold comparison experiments")
    parser.add_argument("--main-root", default="main_data")
    parser.add_argument("--cv-root", default="wuzhe_data")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated model list")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--pet-dir", default="pet_peizhuan", help="Use registered PET target by default")
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--sample-interval", type=int, default=999999)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--early-stop-patience", type=int, default=80)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stop-warmup", type=int, default=50)
    parser.add_argument("--output-root", default="output/comparison_metrics")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    main_root = (PROJECT_ROOT / args.main_root).resolve()
    cv_root = (PROJECT_ROOT / args.cv_root).resolve()
    output_root = (PROJECT_ROOT / args.output_root).resolve()

    rows: List[Dict] = []
    failures: List[Dict] = []

    def guarded(model: str, exp_name: str, root: Path, phase: str, fold: str) -> None:
        try:
            rows.append(_run_train_eval(model, exp_name, root, phase, fold, args))
            _write_aggregate(rows, output_root)
        except Exception as exc:
            failure = {"phase": phase, "fold": fold, "model": model, "error": str(exc)}
            failures.append(failure)
            print(f"FAILED: {failure}")
            if not args.continue_on_error:
                raise

    # 1) Main experiment first.
    for model in models:
        guarded(model, f"main_{model}", main_root, "main", "main_8_2")

    # 2) Five-fold CV after all main experiments.
    for fold_idx in range(1, 6):
        fold_name = f"fold_{fold_idx}"
        fold_root = cv_root / fold_name
        for model in models:
            guarded(model, f"cv_{fold_name}_{model}", fold_root, "fivefold", fold_name)

    _write_aggregate(rows, output_root)
    if failures:
        with open(output_root / "failures.json", "w", encoding="utf-8") as fh:
            json.dump(failures, fh, indent=2, ensure_ascii=False)
        raise SystemExit(f"{len(failures)} experiment(s) failed; see failures.json")
    print(f"\nAll experiments complete. Summary: {output_root / 'all_metrics_summary.csv'}")


if __name__ == "__main__":
    main()
