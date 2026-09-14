"""Freeze the CT support head after H1/H4 pass.

This stage does not train a model and does not reinterpret H1 as amplitude
prediction.  It records the only CT role allowed downstream: a bounded spatial
support/reliability field in the H1-selected band and direction.  H1's held-out
gate and H4's checkpoint lineage are consumed unchanged.
"""

from __future__ import annotations

import argparse
import json
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


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    output = _resolve(root, args.output)
    output.mkdir(parents=True, exist_ok=True)
    contract = load_json(_resolve(root, args.contract))
    h1_path = _resolve(root, args.h1_decision)
    h4_path = _resolve(root, args.h4_decision)
    h1 = load_json(h1_path)
    h4 = load_json(h4_path)
    contract_sha = contract.get("contract_sha256")
    if h1.get("decision") != "PASS":
        raise RuntimeError("H1 is not PASS; CT support head is blocked")
    if h4.get("decision") != "PASS" or not h4.get("next_stage_allowed"):
        raise RuntimeError("H4 is not PASS; CT support head is blocked")
    if {
        h1.get("dataset_contract_sha256"),
        h4.get("dataset_contract_sha256"),
    } != {contract_sha}:
        raise RuntimeError("H1/H4 dataset contract mismatch")
    support = h1.get("H1_support_localization", {})
    amplitude = h1.get("H1_pet_band_amplitude", {})
    selected_band = str(support.get("selected_band", ""))
    direction = str(support.get("direction", ""))
    ci = support.get("validation_ci95", [None, None])
    raw = support.get("secondary_selected_band_minus_raw_ct", {})
    eligible = bool(
        support.get("status") == "PASS"
        and selected_band
        and direction in {"+", "-"}
        and isinstance(ci, list)
        and len(ci) == 2
        and float(ci[0]) > 0.5
        and float(support.get("validation_permutation_p", 1.0)) < 0.05
        and float(raw.get("ci95_low", float("-inf"))) > 0.0
        and float(raw.get("permutation_p", 1.0)) < 0.05
        and amplitude.get("status") == "PASS_NO_RECOVERABLE_INCREMENT"
    )
    head = {
        "schema_version": 1,
        "head_type": "fixed_analytic_ct_spatial_support",
        "selected_haar_band": selected_band,
        "response_direction": direction,
        "input": "normalized CT only",
        "operation": [
            "two-level Haar response in selected band",
            "apply frozen H1 response direction",
            "per-slice robust median/MAD normalization without fitted threshold",
            "sigmoid to [0,1] spatial reliability",
        ],
        "allowed_router_role": "multiplicative bounded spatial support/reliability",
        "forbidden_router_role": "PET lesion-band amplitude target or substitute",
        "mask_input": False,
        "trainable": False,
        "source_decisions": {
            "H1": {
                "path": h1_path.as_posix(),
                "sha256": file_sha256(h1_path),
            },
            "H4": {
                "path": h4_path.as_posix(),
                "sha256": file_sha256(h4_path),
            },
        },
    }
    write_json(output / "ct_support_head.json", head)
    decision = {
        "schema_version": SCHEMA_VERSION,
        "stage": "04_CT_support_head",
        "decision": "PASS" if eligible else "FAIL",
        "dataset_contract_sha256": contract_sha,
        "cache_metadata_sha256": h4.get("cache_metadata_sha256"),
        "mechanism_partition_sha256": h4.get(
            "mechanism_partition_sha256"
        ),
        "guardrails": decision_guardrails(),
        "CT_support_head": {
            "status": "PASS" if eligible else "FAIL",
            "selected_band": selected_band,
            "direction": direction,
            "H1_validation_directional_auc": support.get(
                "validation_directional_auc"
            ),
            "H1_validation_ci95": ci,
            "H1_selected_minus_raw": raw,
            "PET_amplitude_role_excluded": (
                amplitude.get("status")
                == "PASS_NO_RECOVERABLE_INCREMENT"
            ),
            "configuration": (output / "ct_support_head.json").as_posix(),
        },
        "next_stage_allowed": eligible,
        "stop_rule": (
            "CT may enter the router only as bounded spatial support."
            if eligible
            else "Stop the CT support head; do not retune H1 thresholds."
        ),
    }
    write_json(output / "decision.json", decision)
    write_json(
        output / "execution_metadata.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "new_model_training_performed": False,
            "validation_used_to_select_new_thresholds": False,
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
        "--h1-decision",
        type=Path,
        default=Path(
            "results/mechanism_validation/00B_h1_spectral_asymmetry/decision.json"
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
        "--output",
        type=Path,
        default=Path("results/mechanism_validation/04_ct_support_head"),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.root = args.root.resolve()
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        decision = run(args)
    except Exception as exc:
        output = _resolve(args.root, args.output)
        failure = {
            "schema_version": SCHEMA_VERSION,
            "stage": "04_CT_support_head",
            "decision": "FAIL",
            "dataset_contract_sha256": None,
            "guardrails": decision_guardrails(
                checkpoint_lineage="NOT_EVALUATED"
            ),
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "next_stage_allowed": False,
            "stop_rule": "Stop the CT head and dependent router stages.",
        }
        write_json(output / "decision.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
        return 2
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    return 0 if decision["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
