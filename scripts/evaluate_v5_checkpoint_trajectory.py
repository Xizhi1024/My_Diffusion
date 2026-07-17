#!/usr/bin/env python3
"""Evaluate a V5 experiment across its checkpoint trajectory.

For each experiment, evaluate at every checkpoint epoch in
  {50, 100, 150, 200, 250, 300} plus best-lesion and best-combined.

Output one JSON per experiment in
  results/spectral_router_ablations_v5/trajectory/

Usage:
    python scripts/evaluate_v5_checkpoint_trajectory.py \
        --experiment-dir results/spectral_router_ablations_v5/promote/evidence-s3/t_native \
        --config configs/experiments/slmf_png_spectral_router_v5.yaml \
        --output-dir results/spectral_router_ablations_v5/trajectory
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _find_checkpoints(experiment_dir: Path) -> Dict[str, Path]:
    """Discover checkpoint files and return {label: path}."""
    if not experiment_dir.is_dir():
        raise NotADirectoryError(str(experiment_dir))

    checkpoints: Dict[str, Path] = {}
    # Each experiment is output to a directory; checkpoints are typically in
    # <output_dir>/<experiment_name>/checkpoints/ or similar.
    for pattern in ("**/ckpt_epoch*.pt", "**/ckpt_best_*.pt"):
        for ckpt in sorted(experiment_dir.rglob(pattern)):
            stem = ckpt.stem  # e.g. ckpt_epoch0050, ckpt_best_lesion
            checkpoints[stem] = ckpt
    return checkpoints


def _trajectory_epochs(plan_config: Dict[str, Any]) -> List[int]:
    """Return the ordered checkpoint epochs to evaluate."""
    promote = plan_config.get("promote", {})
    raw = promote.get("trajectory_checkpoints", [50, 100, 150, 200, 250, 300])
    return sorted(int(e) for e in raw)


def evaluate_checkpoint(
    config_path: Path,
    checkpoint_path: Path,
    *,
    split: str = "val",
    max_samples: int = 64,
    seed: int = 42,
    mc_steps: int = 20,
    weights: str = "ema",
    device: Optional[str] = None,
    python: str = sys.executable,
) -> Dict[str, Any]:
    """Run evaluate.py on a single checkpoint and return the parsed JSON."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        output_path = Path(tmp.name)

    cmd = [
        python, "-m", "scripts.evaluate",
        "--config", str(config_path),
        "--checkpoint", str(checkpoint_path),
        "--weights", weights,
        "--split", split,
        "--max-samples", str(max_samples),
        "--seed", str(seed),
        "--mc-steps", str(mc_steps),
        "--output", str(output_path),
    ]
    if device:
        cmd.extend(["--device", device])

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        result = json.loads(output_path.read_text(encoding="utf-8"))
        return result
    finally:
        if output_path.exists():
            output_path.unlink()


def collect_trajectory(
    experiment_dir: Path,
    config_path: Path,
    *,
    split: str = "val",
    max_samples: int = 64,
    seed: int = 42,
    mc_steps: int = 20,
    weights: str = "ema",
    device: Optional[str] = None,
    python: str = sys.executable,
    whitelist_epochs: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Evaluate all checkpoints for one experiment and return the trajectory.

    Args:
        whitelist_epochs: Only evaluate checkpoints whose epoch number is in
            this list.  When None, all ``ckpt_epoch*.pt`` files are evaluated.
    """
    checkpoints = _find_checkpoints(experiment_dir)
    trajectory: Dict[str, Any] = {
        "experiment_dir": str(experiment_dir),
        "config": str(config_path),
    }
    results: Dict[str, Any] = {}

    # Build whitelist from epoch labels
    allowed_epochs: Optional[set[int]] = (
        set(whitelist_epochs) if whitelist_epochs is not None else None
    )

    # Evaluate fixed-epoch checkpoints
    for epoch_label, ckpt_path in sorted(checkpoints.items()):
        if not epoch_label.startswith("ckpt_epoch"):
            continue
        # Filter by whitelist if set
        if allowed_epochs is not None:
            try:
                epoch_num = int(epoch_label[len("ckpt_epoch"):])
            except ValueError:
                continue
            if epoch_num not in allowed_epochs:
                continue
        print(f"  Evaluating {epoch_label} ...")
        try:
            result = evaluate_checkpoint(
                config_path, ckpt_path,
                split=split, max_samples=max_samples, seed=seed,
                mc_steps=mc_steps, weights=weights, device=device,
                python=python,
            )
            results[epoch_label] = result
        except Exception as exc:
            print(f"  [WARN] {epoch_label} failed: {exc}", file=sys.stderr)
            results[epoch_label] = {"error": str(exc)}

    # Evaluate best-* checkpoints
    for best_kind in ("lesion", "combined"):
        best_label = f"ckpt_best_{best_kind}"
        ckpt_path = checkpoints.get(best_label)
        if ckpt_path is None:
            continue
        print(f"  Evaluating {best_label} ...")
        try:
            result = evaluate_checkpoint(
                config_path, ckpt_path,
                split=split, max_samples=max_samples, seed=seed,
                mc_steps=mc_steps, weights=weights, device=device,
                python=python,
            )
            results[best_label] = result
        except Exception as exc:
            print(f"  [WARN] {best_label} failed: {exc}", file=sys.stderr)
            results[best_label] = {"error": str(exc)}

    trajectory["results"] = results
    return trajectory


def _extract_summary_metrics(
    trajectory: Dict[str, Any],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Extract key scalar metrics from each checkpoint result.

    Returns {label: {metric_name: value}}.
    """
    summary: Dict[str, Dict[str, Optional[float]]] = {}
    key_metrics = [
        "lesion_topq_peak_error_norm_mean",
        "lesion_peak_error_norm_mean",
        "lesion_signed_bias_mean",
        "lesion_underestimate_rate",
        "ssim_mean",
        "mae_mean",
        "failure_any_mean",
        "false_hotspot_density_mean",
        "stripe_excess_mean",
        "small_lesion_topq_peak_error_norm_mean",
        "small_lesion_signed_bias_mean",
        "small_lesion_underestimate_rate",
    ]
    for label, result in trajectory.get("results", {}).items():
        if isinstance(result, dict) and "error" not in result:
            metrics = result.get("aggregate", result)
            summary[label] = {
                m: metrics.get(m)
                for m in key_metrics
            }
        else:
            summary[label] = {}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir", type=Path, required=True,
        help="Directory containing experiment checkpoints",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Experiment config YAML.  If omitted, "
            "{experiment-dir}/resolved_config.yaml is used."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default="results/spectral_router_ablations_v5/trajectory",
        help="Output directory for trajectory JSON files",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mc-steps", type=int, default=20)
    parser.add_argument("--weights", default="ema")
    parser.add_argument("--device")
    parser.add_argument(
        "--epochs", type=int, nargs="+",
        default=[50, 100, 150, 200, 250, 300],
        help="Whitelist of checkpoint epochs to evaluate (default: 50 100 150 200 250 300)",
    )
    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        resolved = args.experiment_dir / "resolved_config.yaml"
        if resolved.is_file():
            config_path = resolved
            print(f"Auto-detected config: {config_path}")
        else:
            print("ERROR: --config is required when resolved_config.yaml is not present "
                  "in the experiment directory.", file=sys.stderr)
            sys.exit(1)

    print(f"Trajectory evaluation for: {args.experiment_dir}")
    print(f"Config: {config_path}")

    trajectory = collect_trajectory(
        args.experiment_dir,
        config_path,
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
        mc_steps=args.mc_steps,
        weights=args.weights,
        device=args.device,
        whitelist_epochs=args.epochs,
    )

    experiment_name = args.experiment_dir.name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{experiment_name}_trajectory.json"
    output_path.write_text(
        json.dumps(trajectory, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nTrajectory saved to: {output_path}")

    # Print summary table
    summary = _extract_summary_metrics(trajectory)
    if summary:
        print("\nMetric summary:")
        headers = sorted(
            {m for metrics in summary.values() for m in metrics}
        )
        print(f"{'Checkpoint':<22}", end="")
        for h in headers:
            print(f"{h:<14}", end="")
        print()
        for label, metrics in sorted(summary.items()):
            print(f"{label:<22}", end="")
            for h in headers:
                val = metrics.get(h)
                if val is not None:
                    print(f"{val:<14.6f}", end="")
                else:
                    print(f"{'--':<14}", end="")
            print()


if __name__ == "__main__":
    main()
