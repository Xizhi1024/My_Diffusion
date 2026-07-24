"""Freeze H3-derived time/task releases without validation feedback.

Calibration-patient median crossing log-SNR values are mapped to the nearest
predeclared H3 timestep.  Those releases set loss ``active_tau_max`` values.
The epoch warmup/ramp is a fixed fraction of the predeclared formal endpoint;
neither validation metrics nor checkpoint performance can change it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mechanism_validation.common import (
    decision_guardrails,
    file_sha256,
    load_json,
    write_json,
)


SCHEMA_VERSION = 1
TASK_TO_LOSSES = {
    "coarse": ["lesion_roi_l1"],
    "shape": ["outside_peak_ranking"],
    "intensity": ["topk_lesion", "normalized_lesion_peak"],
    "local_frequency": ["boundary_frequency", "residual_wavelet"],
}


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _nearest_timestep(
    target_log_snr: float,
    grid: list[tuple[int, float]],
) -> tuple[int, float]:
    return min(
        grid,
        key=lambda item: (abs(item[1] - target_log_snr), item[0]),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    output = _resolve(root, args.output)
    output.mkdir(parents=True, exist_ok=True)
    contract = load_json(_resolve(root, args.contract))
    h3_path = _resolve(root, args.h3_decision)
    h4_path = _resolve(root, args.h4_decision)
    head_path = _resolve(root, args.ct_head_decision)
    h3 = load_json(h3_path)
    h4 = load_json(h4_path)
    head = load_json(head_path)
    for label, decision in (("H3", h3), ("H4", h4), ("CT head", head)):
        if decision.get("decision") != "PASS" or not decision.get(
            "next_stage_allowed"
        ):
            raise RuntimeError(f"{label} is not PASS; curriculum is blocked")
        if decision.get("dataset_contract_sha256") != contract.get(
            "contract_sha256"
        ):
            raise RuntimeError(f"{label} dataset contract mismatch")

    h3_dir = h3_path.parent
    crossings = _read_csv(h3_dir / "patient_crossings.csv")
    curves = _read_csv(h3_dir / "patient_recoverability_curves.csv")
    analysis = load_json(h3_dir / "analysis_spec.json")
    num_timesteps = int(analysis["num_train_timesteps"])
    calibration_crossings = [
        row for row in crossings if row.get("partition") == "calibration"
    ]
    if len(calibration_crossings) < 5:
        raise RuntimeError("Insufficient calibration patients for curriculum")
    grid_pairs = sorted(
        {
            (int(row["timestep"]), float(row["log_snr"]))
            for row in curves
            if row.get("partition") == "calibration"
        }
    )
    if not grid_pairs:
        raise RuntimeError("H3 calibration time grid is empty")

    task_releases: dict[str, Any] = {}
    loss_active_tau_max: dict[str, float] = {}
    for task, losses in TASK_TO_LOSSES.items():
        key = f"{task}_crossing_log_snr"
        values = [
            float(row[key])
            for row in calibration_crossings
            if row.get(key) not in (None, "")
            and math.isfinite(float(row[key]))
        ]
        if len(values) < max(5, len(calibration_crossings) // 2):
            raise RuntimeError(
                f"Calibration crossing not identifiable for {task}"
            )
        median_log_snr = float(statistics.median(values))
        timestep, observed_log_snr = _nearest_timestep(
            median_log_snr,
            grid_pairs,
        )
        tau = float(timestep / max(num_timesteps - 1, 1))
        task_releases[task] = {
            "calibration_patients_identifiable": len(values),
            "median_crossing_log_snr": median_log_snr,
            "nearest_predeclared_timestep": timestep,
            "nearest_log_snr": observed_log_snr,
            "active_tau_max": tau,
            "mapped_losses": losses,
        }
        for loss in losses:
            loss_active_tau_max[loss] = tau

    warmup = max(1, round(args.formal_epochs * args.warmup_fraction))
    ramp = max(1, round(args.formal_epochs * args.ramp_fraction))
    if warmup + ramp >= args.formal_epochs:
        raise ValueError("Curriculum warmup+ramp must end before final epoch")
    curriculum = {
        "schema_version": 1,
        "formal_epochs": args.formal_epochs,
        "source": "H3 calibration patients only",
        "task_releases": task_releases,
        "loss_active_tau_max": loss_active_tau_max,
        "router_epoch_schedule": {
            "native_warmup_epochs": warmup,
            "routing_ramp_epochs": ramp,
            "fractions_predeclared": {
                "warmup": args.warmup_fraction,
                "ramp": args.ramp_fraction,
            },
        },
        "ct_support_head": {
            "enabled": True,
            "selected_haar_band": head["CT_support_head"]["selected_band"],
            "response_direction": head["CT_support_head"]["direction"],
            "support_only": True,
            "mask_input": False,
        },
        "validation_used": False,
        "adaptive_checkpoint_selection": False,
    }
    write_json(output / "frozen_curriculum.json", curriculum)
    decision = {
        "schema_version": SCHEMA_VERSION,
        "stage": "05_mechanism_curriculum",
        "decision": "PASS",
        "dataset_contract_sha256": contract["contract_sha256"],
        "cache_metadata_sha256": h4.get("cache_metadata_sha256"),
        "mechanism_partition_sha256": h3.get(
            "mechanism_partition_sha256"
        ),
        "guardrails": decision_guardrails(),
        "curriculum": {
            "status": "PASS",
            "path": (output / "frozen_curriculum.json").as_posix(),
            "sha256": file_sha256(output / "frozen_curriculum.json"),
            "task_releases": task_releases,
            "router_epoch_schedule": curriculum["router_epoch_schedule"],
        },
        "next_stage_allowed": True,
        "stop_rule": (
            "Curriculum frozen; subsequent model runs may not modify it from "
            "validation/test performance."
        ),
    }
    write_json(output / "decision.json", decision)
    write_json(
        output / "execution_metadata.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "training_performed": False,
            "validation_used_to_choose_course": False,
        },
    )
    return decision


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/dataset_contract_stage0a_v1.json"),
    )
    parser.add_argument(
        "--h3-decision",
        type=Path,
        default=Path(
            "results/mechanism_validation/02_h3_recoverability_curves/decision.json"
        ),
    )
    parser.add_argument(
        "--h4-decision",
        type=Path,
        default=Path(
            "results/mechanism_validation/03_h4_noise_calibration/decision.json"
        ),
    )
    parser.add_argument(
        "--ct-head-decision",
        type=Path,
        default=Path(
            "results/mechanism_validation/04_ct_support_head/decision.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mechanism_validation/05_curriculum"),
    )
    parser.add_argument("--formal-epochs", type=int, default=50)
    parser.add_argument("--warmup-fraction", type=float, default=0.10)
    parser.add_argument("--ramp-fraction", type=float, default=0.10)
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.root = args.root.resolve()
    if args.formal_epochs <= 2:
        parser.error("--formal-epochs must exceed 2")
    if not 0.0 < args.warmup_fraction < 1.0:
        parser.error("--warmup-fraction must be in (0,1)")
    if not 0.0 < args.ramp_fraction < 1.0:
        parser.error("--ramp-fraction must be in (0,1)")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        decision = run(args)
    except Exception as exc:
        output = _resolve(args.root, args.output)
        failure = {
            "schema_version": SCHEMA_VERSION,
            "stage": "05_mechanism_curriculum",
            "decision": "FAIL",
            "dataset_contract_sha256": None,
            "guardrails": decision_guardrails(
                checkpoint_lineage="NOT_EVALUATED"
            ),
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "next_stage_allowed": False,
            "stop_rule": "Stop curriculum and dependent formal model runs.",
        }
        write_json(output / "decision.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
        return 2
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    return 0 if decision["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
