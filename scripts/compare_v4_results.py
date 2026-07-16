"""Create deterministic patient-paired comparisons for V4 evaluation JSONs."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


def _load_result(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload.get("per_patient"), Mapping):
        raise ValueError(f"Result {path} has no per_patient mapping")
    return payload


def compare_result_pair(
    left_path: Path,
    right_path: Path,
    metric_directions: Mapping[str, str],
    *,
    seed: int = 42,
    resamples: int = 10_000,
) -> Dict[str, Any]:
    """Compare left to right on their finite intersecting patient records."""
    if resamples < 1:
        raise ValueError("resamples must be positive")
    left = _load_result(Path(left_path))["per_patient"]
    right = _load_result(Path(right_path))["per_patient"]
    shared_ids = sorted(set(left) & set(right))
    report: Dict[str, Any] = {
        "left": str(left_path),
        "right": str(right_path),
        "shared_patients": len(shared_ids),
        "metrics": {},
    }

    for metric_index, (metric, direction) in enumerate(metric_directions.items()):
        if direction not in {"lower", "higher"}:
            raise ValueError(f"Unknown direction {direction!r} for {metric!r}")
        pairs = []
        for patient_id in shared_ids:
            left_value = left[patient_id].get(metric)
            right_value = right[patient_id].get(metric)
            try:
                left_float = float(left_value)
                right_float = float(right_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(left_float) and math.isfinite(right_float):
                pairs.append((left_float, right_float))

        if pairs:
            pair_array = np.asarray(pairs, dtype=np.float64)
            differences = pair_array[:, 0] - pair_array[:, 1]
            ties = np.isclose(differences, 0.0, atol=1e-12, rtol=0.0)
            wins = differences < 0.0 if direction == "lower" else differences > 0.0
            losses = differences > 0.0 if direction == "lower" else differences < 0.0
            median_difference = float(np.median(differences))
        else:
            differences = np.empty(0, dtype=np.float64)
            ties = wins = losses = np.empty(0, dtype=bool)
            median_difference = None

        interval = None
        if differences.size >= 2:
            rng = np.random.default_rng(seed + metric_index)
            indices = rng.integers(
                0, differences.size, size=(resamples, differences.size)
            )
            bootstrapped = np.median(differences[indices], axis=1)
            interval = [
                float(np.percentile(bootstrapped, 2.5)),
                float(np.percentile(bootstrapped, 97.5)),
            ]

        report["metrics"][metric] = {
            "direction": direction,
            "paired_patients": int(differences.size),
            "wins": int(wins.sum()),
            "ties": int(ties.sum()),
            "losses": int(losses.sum()),
            "median_difference": median_difference,
            "bootstrap_95_ci": interval,
        }
    return report


def compare_all_results(
    result_paths: Mapping[str, Path],
    metric_directions: Mapping[str, str],
    *,
    seed: int = 42,
    resamples: int = 10_000,
) -> Dict[str, Any]:
    comparisons = []
    for left_id, right_id in itertools.combinations(sorted(result_paths), 2):
        comparison = compare_result_pair(
            Path(result_paths[left_id]),
            Path(result_paths[right_id]),
            metric_directions,
            seed=seed,
            resamples=resamples,
        )
        comparison.update({"left_id": left_id, "right_id": right_id})
        comparisons.append(comparison)
    return {"seed": seed, "resamples": resamples, "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True, help="ID=path")
    parser.add_argument("--metrics", required=True, help="JSON metric-direction mapping")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()

    result_paths = {}
    for value in args.result:
        result_id, separator, path = value.partition("=")
        if not separator or not result_id or not path:
            raise ValueError(f"Invalid --result value: {value!r}")
        result_paths[result_id] = Path(path)
    metric_directions = json.loads(args.metrics)
    report = compare_all_results(
        result_paths,
        metric_directions,
        seed=args.seed,
        resamples=args.resamples,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
