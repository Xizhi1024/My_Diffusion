"""Fail-closed audit of the frozen H4-v2 bundle as a production-router input.

This audit deliberately separates two questions:

1. Did the internal exploratory H4-v2 analysis use the sealed source artifact?
2. Can the same evidence be computed during CT-only PET synthesis without a
   target PET image or a lesion mask?

The first may remain true while the second fails.  A failure here never edits
or relabels H4-v1/H4-v2 results; it blocks only their production-router use.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = (
    ROOT
    / "results"
    / "mechanism_validation_v2"
    / "04_main_integration_freeze"
    / "production_calibration_bundle.json"
)
DEFAULT_CONTRACT = (
    ROOT
    / "results"
    / "mechanism_validation_v2"
    / "04_main_integration_freeze"
    / "resolved_integration_contract.json"
)
DEFAULT_GENERATOR = ROOT / "scripts" / "validate_h4_noise_band_calibration.py"
DEFAULT_SOURCE_CSV = (
    ROOT
    / "results"
    / "mechanism_validation"
    / "03_h4_noise_calibration"
    / "sample_band_evidence.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "mechanism_validation_v2"
    / "05A_inference_admissibility_audit"
)

sys.path.insert(0, str(ROOT))

from src.mechanism_validation.common import canonical_json_sha256, file_sha256


PIPELINE_ID = "V2_INFERENCE_ADMISSIBILITY_AUDIT_V1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _self_hash_matches(
    payload: Mapping[str, Any],
    field: str,
) -> bool:
    expected = payload.get(field)
    if not isinstance(expected, str) or not expected:
        return False
    unhashed = dict(payload)
    unhashed.pop(field, None)
    return canonical_json_sha256(unhashed) == expected


def _slice_key(node: ast.Subscript) -> str | None:
    value = node.slice
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def _name_set(node: ast.AST) -> set[str]:
    return {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name)
    }


def inspect_h4_generator(source: str) -> dict[str, Any]:
    """Extract the inference-relevant inputs of ``_sample_band_rows``."""

    tree = ast.parse(source)
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_sample_band_rows"
        ),
        None,
    )
    if function is None:
        raise ValueError("H4 generator lacks _sample_band_rows")

    batch_reads: set[str] = set()
    calls: set[str] = set()
    residual_uses_pet = False
    for node in ast.walk(function):
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == "batch":
                key = _slice_key(node)
                if key is not None:
                    batch_reads.add(key)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
        elif isinstance(node, ast.Assign):
            targets = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if "residual" in targets and "pet" in _name_set(node.value):
                residual_uses_pet = True

    return {
        "batch_reads": sorted(batch_reads),
        "reads_target_pet": "pet" in batch_reads,
        "reads_lesion_mask": "mask" in batch_reads,
        "residual_uses_target_pet": residual_uses_pet,
        "uses_masked_band_energy": "_masked_band_energy" in calls,
        "uses_masked_band_alignment": "_masked_band_alignment" in calls,
    }


def build_decision(
    *,
    bundle_path: Path,
    contract_path: Path,
    generator_path: Path,
) -> dict[str, Any]:
    bundle = _load_object(bundle_path)
    contract = _load_object(contract_path)
    generator = inspect_h4_generator(
        generator_path.read_text(encoding="utf-8")
    )

    samples_value = bundle.get("source", {}).get(
        "sample_band_evidence_sha256"
    )
    source_csv = DEFAULT_SOURCE_CSV
    source_csv_exists = source_csv.is_file()
    source_csv_hash = file_sha256(source_csv) if source_csv_exists else None
    source_hash_matches = (
        isinstance(samples_value, str)
        and source_csv_hash == samples_value
    )

    forbidden = set(
        contract.get("context_acquisition_rule", {}).get(
            "forbidden_as_inference_input", []
        )
    )
    contract_forbids_pet = any("target PET" in item for item in forbidden)
    contract_forbids_mask = any("mask" in item for item in forbidden)

    lineage_checks = {
        "bundle_self_hash": _self_hash_matches(bundle, "bundle_sha256"),
        "contract_self_hash": _self_hash_matches(
            contract, "contract_sha256"
        ),
        "source_csv_exists": source_csv_exists,
        "source_csv_sha256_matches_bundle": source_hash_matches,
        "h4_v1_failure_preserved": (
            bundle.get("source", {}).get("h4_v1_outcome") == "FAIL"
        ),
    }
    provenance_conflict = {
        "generator_reads_target_pet": generator["reads_target_pet"],
        "generator_builds_residual_from_target_pet": generator[
            "residual_uses_target_pet"
        ],
        "generator_reads_lesion_mask": generator["reads_lesion_mask"],
        "generator_uses_masked_band_energy": generator[
            "uses_masked_band_energy"
        ],
        "contract_forbids_target_pet": contract_forbids_pet,
        "contract_forbids_lesion_mask": contract_forbids_mask,
    }
    conflict_proven = all(provenance_conflict.values())
    lineage_ok = all(lineage_checks.values())
    inference_admissible = lineage_ok and not conflict_proven
    decision = "PASS" if inference_admissible else "FAIL"

    payload: dict[str, Any] = {
        "schema_version": 2,
        "stage": "05A_v2_inference_admissibility_audit",
        "pipeline_id": PIPELINE_ID,
        "decision": decision,
        "decision_scope": (
            "production_inference_availability_of_the_frozen_H4_v2_evidence"
        ),
        "lineage_checks": lineage_checks,
        "generator_semantics": generator,
        "contract_conflict": provenance_conflict,
        "conflict_proven": conflict_proven,
        "production_inference_admissible": inference_admissible,
        "dataset_internal_h4_v2_result_preserved": True,
        "h4_v1_outcome_preserved": "FAIL",
        "model_mechanism_claims_allowed": False,
        "production_router_activation_allowed": inference_admissible,
        "next_stage_allowed": inference_admissible,
        "blocked_stages": (
            []
            if inference_admissible
            else [
                "05_v2_main_integration_activation",
                "08_h5_v2_router_ablations",
                "09_h6_v2_final_integration_with_H4_v2_router",
            ]
        ),
        "unblocked_independent_work": [
            "H3 evidence-free fixed schedule implementation and audit",
            "CT-support gate under its separately frozen H1 contract",
            "artifact-safety tooling that does not claim H4-v2 activation",
        ],
        "conclusion": (
            "PASS: the frozen evidence is available without forbidden inputs."
            if inference_admissible
            else (
                "FAIL: the sealed H4-v2 bundle was fitted from H4-v1 evidence "
                "computed on target-PET residuals inside lesion masks. Those "
                "inputs are unavailable/forbidden during CT-only synthesis. "
                "Internal exploratory H4-v2 support is preserved, but this "
                "bundle cannot activate a production router."
            )
        ),
        "stop_rule": (
            "Do not implement a single-slice CT proxy and call it H4-v2. "
            "Do not activate the uncertainty-aware router, H5-v2, or the "
            "H4-v2-dependent portion of H6. A newly defined observable "
            "CT/current-state evidence would be a new mechanism and requires "
            "a new pipeline_id, inner-only development, and a frozen gate."
        ),
        "artifacts": {
            "bundle": _display_path(bundle_path),
            "bundle_sha256": file_sha256(bundle_path),
            "contract": _display_path(contract_path),
            "contract_sha256": file_sha256(contract_path),
            "generator": _display_path(generator_path),
            "generator_sha256": file_sha256(generator_path),
            "source_csv": _display_path(source_csv),
            "source_csv_sha256": source_csv_hash,
        },
        "created_at_utc": _utc_now(),
    }
    payload["decision_sha256"] = canonical_json_sha256(payload)
    return payload


def build_missing_artifact_decision(
    missing: list[Path],
    *,
    bundle_path: Path,
    contract_path: Path,
    generator_path: Path,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 2,
        "stage": "05A_v2_inference_admissibility_audit",
        "pipeline_id": PIPELINE_ID,
        "decision": "FAIL",
        "failure_phase": "MISSING_FROZEN_ARTIFACT_FAIL_CLOSED",
        "scientific_gate_evaluated": False,
        "missing_artifacts": [_display_path(path) for path in missing],
        "required_artifacts": {
            "bundle": _display_path(bundle_path),
            "contract": _display_path(contract_path),
            "generator": _display_path(generator_path),
            "source_csv": _display_path(DEFAULT_SOURCE_CSV),
        },
        "h4_v1_outcome_preserved": "FAIL",
        "dataset_internal_h4_v2_result_preserved": True,
        "model_mechanism_claims_allowed": False,
        "production_router_activation_allowed": False,
        "next_stage_allowed": False,
        "conclusion": (
            "The V2-05A scientific gate was not evaluated because one or more "
            "sealed Stage-04/source artifacts are absent on this machine."
        ),
        "remediation": (
            "Copy the exact frozen artifacts from the development repository. "
            "Do not regenerate or refreeze them on the cloud machine; then rerun "
            "this audit."
        ),
        "stop_rule": (
            "Do not run V2-05 activation, H5-v2, or H6-v2 while any required "
            "frozen artifact is missing."
        ),
        "created_at_utc": _utc_now(),
    }
    payload["decision_sha256"] = canonical_json_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--generator", type=Path, default=DEFAULT_GENERATOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-scientific-fail-exit-zero",
        action="store_true",
        help="Write FAIL but return zero; intended only for report aggregation.",
    )
    args = parser.parse_args()

    bundle_path = args.bundle.resolve()
    contract_path = args.contract.resolve()
    generator_path = args.generator.resolve()
    required = (
        bundle_path,
        contract_path,
        generator_path,
        DEFAULT_SOURCE_CSV,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        decision = build_missing_artifact_decision(
            missing,
            bundle_path=bundle_path,
            contract_path=contract_path,
            generator_path=generator_path,
        )
    else:
        decision = build_decision(
            bundle_path=bundle_path,
            contract_path=contract_path,
            generator_path=generator_path,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "decision.json"
    output.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(f"Wrote: {output}")
    if (
        decision["decision"] != "PASS"
        and not args.allow_scientific_fail_exit_zero
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
