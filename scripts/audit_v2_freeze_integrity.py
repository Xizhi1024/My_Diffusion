"""V2 main-pipeline freeze integrity audit.

Runs BEFORE any V2 main-model integration change.  It re-derives, without
re-freezing, that the frozen H4-v2 artefacts are still self-consistent and that
H4-v1 is still FAIL.  If any check fails the audit writes ``FAIL`` and exits
non-zero so the cloud one-click pipeline halts before touching the main model.

This audit is read-only: it never rewrites a frozen artefact.  If a frozen file
has changed, the correct action is to STOP and report, not to re-freeze.

Run from the repo root:
    python scripts/audit_v2_freeze_integrity.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mechanism_validation.common import (
    canonical_json_sha256,
    file_sha256,
    load_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = REPO_ROOT / "results" / "mechanism_validation_v2"

# Spec-locked reference values (一、不可更改的事实).  The audit compares the
# on-disk artefacts against these AND against the values recorded inside the
# frozen plan itself; both must agree.
SPEC_PIPELINE_ID = "H4_V2_INTERNAL_EXPLORATORY_NESTED_CV_V1"
SPEC_CONFIG_SHA256 = (
    "891a30ece1793a3d13ae30df55597a0f4683861d8ed3c9428742fe851b973e83"
)
SPEC_PLAN_SHA256 = (
    "40afd531dd93be2ba78be12ec32ba52b309c68e1d22802553b54db23958732a5"
)
SPEC_PARTITION_FINGERPRINT = (
    "0146d21dc3b242e65977fda52f32cf3c6efe79b8fd11011b392068b69a0ba9ad"
)
SPEC_REQUIRED_DIFFICULT_PATIENTS = ("044", "153")
SPEC_TOTAL_PATIENTS = 155
SPEC_PARTITIONS = {"mechanism_train": 99, "calibration": 25, "validation": 31}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_h4_v1_fail() -> dict:
    path = REPO_ROOT / "results" / "mechanism_validation" / "03_h4_noise_calibration" / "decision.json"
    decision = load_json(path)
    status = str(decision.get("decision", decision.get("status", "")))
    return {
        "check": "h4_v1_decision_is_FAIL",
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "observed_status": status,
        "passed": status == "FAIL",
    }


def _check_plan_self_hash(plan: dict, plan_path: Path) -> dict:
    body = dict(plan)
    declared = str(body.pop("plan_sha256", ""))
    computed = canonical_json_sha256(body)
    return {
        "check": "frozen_plan_self_hash",
        "path": plan_path.relative_to(REPO_ROOT).as_posix(),
        "declared": declared,
        "computed": computed,
        "spec_value": SPEC_PLAN_SHA256,
        "passed": declared == computed == SPEC_PLAN_SHA256,
    }


def _check_config_sha(plan: dict) -> dict:
    declared = str(plan.get("config_sha256", ""))
    return {
        "check": "frozen_config_sha256_matches_spec",
        "declared": declared,
        "spec_value": SPEC_CONFIG_SHA256,
        "passed": declared == SPEC_CONFIG_SHA256,
    }


def _check_partition_fingerprint(plan: dict) -> dict:
    partition = plan.get("partition", {})
    declared = str(partition.get("fingerprint_sha256", ""))
    return {
        "check": "partition_fingerprint_matches_spec",
        "declared": declared,
        "spec_value": SPEC_PARTITION_FINGERPRINT,
        "passed": declared == SPEC_PARTITION_FINGERPRINT,
    }


def _check_frozen_files(plan: dict) -> dict:
    frozen = plan.get("frozen_files", {})
    records = []
    all_ok = True
    for name, meta in frozen.items():
        path = REPO_ROOT / meta["path"]
        exists = path.is_file()
        computed = file_sha256(path) if exists else ""
        ok = exists and computed == meta["sha256"]
        all_ok = all_ok and ok
        records.append(
            {
                "name": name,
                "path": meta["path"],
                "exists": exists,
                "declared_sha256": meta["sha256"],
                "computed_sha256": computed,
                "passed": ok,
            }
        )
    return {
        "check": "frozen_partition_files_unchanged",
        "files": records,
        "passed": all_ok,
    }


def _check_patient_integrity(plan: dict) -> dict:
    dataset = plan.get("dataset", {})
    patients = int(dataset.get("patients", -1))
    samples = int(dataset.get("samples", -1))
    required = sorted(str(p) for p in dataset.get("required_difficult_patients_present", []))
    spec_required = sorted(SPEC_REQUIRED_DIFFICULT_PATIENTS)
    excluded = int(dataset.get("patients_excluded", -1))
    partitions = dataset.get("original_partitions", {})

    patients_ok = patients == SPEC_TOTAL_PATIENTS
    required_ok = required == spec_required
    excluded_ok = excluded == 0
    partition_ok = all(
        partitions.get(role) == count for role, count in SPEC_PARTITIONS.items()
    )

    # Independently confirm patient count + difficult patients from the frozen
    # outer assignment CSV (do not trust the plan summary alone).
    assignments_path = (
        V2_ROOT
        / "02_h4_v2_internal_exploratory_nested_cv"
        / "00_frozen_plan"
        / "outer_patient_assignments.csv"
    )
    csv_patient_set = set()
    if assignments_path.is_file():
        with assignments_path.open("r", encoding="utf-8-sig", newline="") as handle:
            import csv

            reader = csv.DictReader(handle)
            for row in reader:
                patient = str(row.get("patient_id", "")).strip().zfill(3)
                if patient:
                    csv_patient_set.add(patient)
    csv_ok = len(csv_patient_set) == SPEC_TOTAL_PATIENTS and all(
        p in csv_patient_set for p in SPEC_REQUIRED_DIFFICULT_PATIENTS
    )

    return {
        "check": "patient_integrity_155_no_exclusions",
        "plan_patients": patients,
        "plan_samples": samples,
        "plan_partitions": partitions,
        "required_difficult_patients": required,
        "patients_excluded": excluded,
        "csv_unique_patients": len(csv_patient_set),
        "csv_difficult_present": {
            p: (p in csv_patient_set) for p in SPEC_REQUIRED_DIFFICULT_PATIENTS
        },
        "passed": patients_ok and required_ok and excluded_ok and partition_ok and csv_ok,
    }


def _check_h4_v2_pipeline_pass() -> dict:
    path = (
        V2_ROOT
        / "02_h4_v2_internal_exploratory_nested_cv"
        / "99_pipeline"
        / "decision.json"
    )
    decision = load_json(path)
    status = str(decision.get("decision", ""))
    v1 = str(decision.get("H4_v1_preserved", ""))
    pid = str(decision.get("pipeline_id", ""))
    return {
        "check": "h4_v2_pipeline_PASS_and_preserves_v1_FAIL",
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "observed_status": status,
        "h4_v1_preserved": v1,
        "pipeline_id": pid,
        "passed": (
            status == "PASS"
            and v1 == "FAIL"
            and pid == SPEC_PIPELINE_ID
        ),
    }


def _check_working_tree_clean_on_frozen() -> dict:
    """Frozen H4-v2 artefacts must not be modified in the working tree."""
    import subprocess

    frozen_rel = [
        "results/mechanism_validation_v2/02_h4_v2_internal_exploratory_nested_cv",
        "results/mechanism_validation/03_h4_noise_calibration/decision.json",
    ]
    modified = []
    for rel in frozen_rel:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", rel],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.stdout.strip():
            modified.extend(result.stdout.strip().splitlines())
    return {
        "check": "frozen_arteffacts_unmodified_in_working_tree",
        "modified_paths": modified,
        "passed": not modified,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=(
            "results/mechanism_validation_v2/99_v2_model_pipeline/"
            "freeze_integrity/decision.json"
        ),
        help="Where to write the audit decision.json",
    )
    args = parser.parse_args()

    plan_path = (
        V2_ROOT
        / "02_h4_v2_internal_exploratory_nested_cv"
        / "00_frozen_plan"
        / "frozen_plan.json"
    )
    # The frozen H4-v2 verification artifacts are immutable and MUST be present
    # byte-identically on every machine that runs V2 work.  results/ is gitignored,
    # so they do not sync via git; surface exactly what is missing before the
    # cryptic FileNotFoundError.
    required_paths = [
        plan_path,
        plan_path.parent / "outer_patient_assignments.csv",
        plan_path.parent / "nested_patient_roles.csv",
        plan_path.parent / "outer_fold_balance.csv",
        V2_ROOT
        / "02_h4_v2_internal_exploratory_nested_cv"
        / "99_pipeline"
        / "decision.json",
        REPO_ROOT / "results" / "mechanism_validation"
        / "03_h4_noise_calibration" / "decision.json",
    ]
    missing = [p for p in required_paths if not p.is_file()]
    if missing:
        print("Frozen H4-v2 verification artifacts are missing on this machine:")
        for p in missing:
            print(f"  MISSING: {p.relative_to(REPO_ROOT)}")
        print(
            "These are immutable freeze artifacts (SHA-256 checked). results/ is "
            "gitignored, so copy them byte-identically from the machine where "
            "H4-v2 was frozen (e.g. via remote desktop) to the same relative paths "
            "here, then re-run."
        )
        return 1

    plan = load_json(plan_path)

    checks = [
        _check_h4_v1_fail(),
        _check_plan_self_hash(plan, plan_path),
        _check_config_sha(plan),
        _check_partition_fingerprint(plan),
        _check_frozen_files(plan),
        _check_patient_integrity(plan),
        _check_h4_v2_pipeline_pass(),
        _check_working_tree_clean_on_frozen(),
    ]

    overall = all(bool(c.get("passed")) for c in checks)
    payload = {
        "schema_version": 2,
        "stage": "00_v2_freeze_integrity_audit",
        "pipeline_id": SPEC_PIPELINE_ID,
        "decision": "PASS" if overall else "FAIL",
        "scope": "read_only_re_derivation_of_frozen_h4_v2_artefacts",
        "refreeze_performed": False,
        "checks": checks,
        "spec_reference": {
            "pipeline_id": SPEC_PIPELINE_ID,
            "config_sha256": SPEC_CONFIG_SHA256,
            "plan_sha256": SPEC_PLAN_SHA256,
            "partition_fingerprint_sha256": SPEC_PARTITION_FINGERPRINT,
            "total_patients": SPEC_TOTAL_PATIENTS,
            "required_difficult_patients": list(SPEC_REQUIRED_DIFFICULT_PATIENTS),
            "original_partitions": SPEC_PARTITIONS,
        },
        "created_at_utc": _utc_now(),
    }
    body = dict(payload)
    payload["audit_sha256"] = canonical_json_sha256(body)

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Freeze integrity decision: {payload['decision']}")
    for check in checks:
        marker = "OK " if check["passed"] else "XX "
        print(f"  {marker}{check['check']}")
    print(f"Wrote: {output_path.relative_to(REPO_ROOT)}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
