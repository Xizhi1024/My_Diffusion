"""V2 main-pipeline orchestrator (cloud Win entry point, pwsh-free).

Run via pixi (consistent with every other task in pixi.toml):

    pixi run python scripts/run_v2_main_pipeline.py --stage freeze
    pixi run python scripts/run_v2_main_pipeline.py --stage excluded_mean
    pixi run python scripts/run_v2_main_pipeline.py --stage all

Honest contract:
  * Stages with a real implementation (freeze, excluded_mean, inference_audit)
    are executed.
  * Stages whose patient-level evaluation still needs (a) a leakage-free
    context-aware inference path and (b) a per-comparator runner on the cloud
    GPU are reported as DEFERRED_NOT_RUN with their exact missing prerequisite.
    They are NEVER relabelled PASS by this script.

Sub-scripts are launched with the SAME interpreter (``sys.executable``), so the
command works whether or not the shell has already activated the pixi env.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFERRED_STAGES = {
    "ct_support": "06A_ct_support_gate",
    "curriculum": "06B_curriculum_gate",
    "artifact_safety": "07_artifact_safety",
    "h5_v2": "08_h5_v2_router_ablations",
    "h6_v2": "09_h6_v2_final_integration",
}


def _run_python(script: str, *extra_args: str) -> int:
    cmd = [sys.executable, str(REPO_ROOT / script), *extra_args]
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def run_freeze() -> int:
    print("\n==== V2 stage: freeze (integrity audit) ====")
    code = _run_python("scripts/audit_v2_freeze_integrity.py")
    if code != 0:
        print("Freeze integrity audit FAILED; halting before any main-model change.")
    return code


def run_excluded_mean(epochs: int, seed: int, guard_px: int, output_dir: str) -> int:
    """Stage V2-03: retrain the pathology-excluded mean with forced fail-closed lineage."""
    print("\n==== V2 stage: excluded_mean (V2-03 retrain) ====")
    if run_freeze() != 0:
        return 1
    overrides = [
        "experiment.name=freq_mean_excluded_v1",
        "modules.conditional_mean.pathology_exclusion.enabled=true",
        f"modules.conditional_mean.pathology_exclusion.guard_radius_px={guard_px}",
        "data.require_cache_lineage=true",
        "data.dataset_contract=configs/dataset_contract_stage0a_v1.json",
        "data.cache_lineage=cache/tensors_main/cache_lineage.json",
        "data.use_fake_data=false",
    ]
    args = [
        "--config", "configs/experiments/slmf_png_residual_frequency.yaml",
        "--output-dir", output_dir,
        "--epochs", str(epochs),
        "--seed", str(seed),
    ]
    for ov in overrides:
        args += ["--override", ov]
    code = _run_python("scripts/pretrain_conditional_mean.py", *args)
    if code != 0:
        print(f"Excluded-mean retraining failed with exit code {code}.")
        return code
    sidecar = Path(output_dir) / "mean_best.pt.fingerprint.json"
    if not sidecar.is_file():
        print(f"WARNING: expected fingerprint sidecar missing: {sidecar}")
    else:
        print(f"Audit the sidecar before any cutover: {sidecar}")
    print("NOTE: the production v5 checkpoint path is intentionally NOT changed.")
    return 0


def run_inference_audit() -> int:
    """Stage V2-05A: prove the frozen evidence is available at inference."""
    print("\n==== V2 stage: inference_audit (V2-05A) ====")
    if run_freeze() != 0:
        return 1
    code = _run_python("scripts/audit_v2_inference_admissibility.py")
    if code != 0:
        print(
            "Inference admissibility FAIL; uncertainty-aware router activation, "
            "H5-v2, and H4-v2-dependent H6 are blocked."
        )
    return code


def report_deferred(short_name: str) -> int:
    stage_dir = DEFERRED_STAGES[short_name]
    decision = f"results/mechanism_validation_v2/{stage_dir}/decision.json"
    print(f"\n==== V2 stage: {short_name} ====")
    print(f"DEFERRED_NOT_RUN: {short_name}")
    print(f"  Protocol decision: {decision}")
    print("  Prerequisite missing: leakage-free context-aware inference path + per-comparator cloud runner.")
    print("  This stage is NOT marked PASS. Do not relabel it.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "freeze",
            "excluded_mean",
            "inference_audit",
            *DEFERRED_STAGES,
            "all",
        ],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guard-px", type=int, default=8)
    parser.add_argument("--output-dir", default="checkpoints/freq_mean_excluded_v1")
    args = parser.parse_args()

    if args.stage == "freeze":
        return run_freeze()
    if args.stage == "excluded_mean":
        return run_excluded_mean(args.epochs, args.seed, args.guard_px, args.output_dir)
    if args.stage == "inference_audit":
        return run_inference_audit()
    if args.stage == "all":
        if run_freeze() != 0:
            return 1
        code = run_excluded_mean(args.epochs, args.seed, args.guard_px, args.output_dir)
        if code != 0:
            return code
        audit_code = run_inference_audit()
        if audit_code != 0:
            print(
                "\nChain state: V2-03 PASS; V2-05A FAIL. "
                "Downstream H4-v2 router stages stopped fail-closed."
            )
            return audit_code
        for short in DEFERRED_STAGES:
            report_deferred(short)
        print("\nChain state: freeze PASS; V2-03 retrained on cloud; V2-05 selector wired+unit-tested but OFF;")
        print("             V2-06A/B, 07, 08, 09 remain DEFERRED_NOT_RUN pending the context path + runners.")
        return code
    # Deferred stage requested directly: still gate on freeze first.
    if run_freeze() != 0:
        return 1
    return report_deferred(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
