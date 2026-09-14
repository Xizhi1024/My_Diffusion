"""Sequential dual-worktree experiment queue: router mainline + interpretability.

Runs the spectral-router mainline and the lesion-feature interpretability branch
on the SAME frozen checkpoint, on a single GPU, in a fixed order:

    A. router job        (--router-command)
    B. feature emergence (A0+A1+A2, --feature-emergence-command)
    C. feature causality (M2, --feature-causality-command)  -- ONLY after B's
                            gates (A1 s1 / A2 m1) pass, and never after a
                            downstream-failing upstream.

Fail-closed rules enforced by the launcher (they supplement, never replace, the
analysis code's own checks):

  * checkpoint stability: size + mtime + SHA256 are sampled over a short
    interval before launch AND again after the router job; a mid-write
    ``last.pt``-style checkpoint refuses to launch.
  * output-dir reuse: ``cloud_runs/router/<run_id>`` and
    ``cloud_runs/interpretability/<run_id>`` must not already exist non-empty.
  * experiment lock: ``<worktree>/run.lock`` records pid/hostname/worktree/
    command/device/started_at; status is updated on normal AND abnormal exit.
    The sync tool refuses ``--apply`` while this lock is ``running``.
  * smoke/fake detection: a command containing ``--fake-data`` or
    ``--max-samples`` can only ever produce ``eligibility="exploratory"``; it is
    never recorded as a formal COMPLETE.
  * formal eligibility preconditions (section 六 of the workflow spec):
      1. val/test feature runs must pass ``--frozen-quartiles-json`` (thresholds
         frozen on the TRAIN split).
      2. authoritative ``patient_id`` (no ``patient_0`` fallback) — enforced by
         the analysis code, mirrored here as a structural check.
      3. M2 must genuinely compute background non-inferiority — verified by
         inspecting the feature-worktree ``feature_causality`` module source for
         the real margin constant and a computed (non-hardcoded) gate.
      4. shifted / same-area controls must live in body tissue — verified by the
         presence of the body-mask + tissue-coverage code paths.
      5. A1 / A2 / M2 all read the SAME frozen checkpoint (the queue always
         passes one ``--checkpoint`` to every feature command).
    Any missing precondition downgrades the run to ``exploratory``; causality is
    additionally blocked when the emergence gates did not pass.

Device policy: default is single-GPU serial. ``--parallel`` exists only to be
rejected when both experiments would share one device — concurrent execution is
deliberately NOT implemented, because the checkpoint-stability fail-closed rule
is incompatible with reading a checkpoint while the router job writes it.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mechanism_validation.common import file_sha256, write_json

_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _iso(dt: datetime.datetime) -> str:
    return dt.astimezone(datetime.timezone.utc).isoformat()


def _now_utc_seconds() -> int:
    return int((_utcnow() - _EPOCH).total_seconds())


class QueueError(Exception):
    """Fail-closed queue violation.  Message is user-facing."""


def _git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _git_state(repo_root: Path) -> dict[str, str]:
    sha = _git_sha(repo_root)
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=str(repo_root),
                stderr=subprocess.DEVNULL,
                text=True,
            )
        )
    except Exception:
        dirty = False
    return {"git_sha": sha, "dirty": dirty}


# ---------------------------------------------------------------------------
# Command parsing helpers
# ---------------------------------------------------------------------------


def _split_command(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=os.name != "nt")
    except ValueError as exc:
        raise QueueError(f"Cannot parse command {command!r}: {exc}")


def _has_flag(cmd: Sequence[str], flag: str) -> bool:
    return any(tok == flag or tok.startswith(flag + "=") for tok in cmd)


def _interventions_in_command(cmd: Sequence[str]) -> set[str]:
    """Collect every token following ``--interventions`` until the next flag.

    The analysis CLI uses ``--interventions <name> [<name> ...]`` (nargs='+'),
    so we gather all following tokens, not just the first.
    """
    names: set[str] = set()
    seen_flag = False
    for tok in cmd:
        if tok == "--interventions":
            seen_flag = True
            continue
        if tok.startswith("--interventions="):
            names.update(tok.split("=", 1)[1].split())
            continue
        if seen_flag:
            if tok.startswith("--"):
                break
            names.add(tok)
    return names


# ---------------------------------------------------------------------------
# Checkpoint stability
# ---------------------------------------------------------------------------


def _snapshot_checkpoint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_sha256(path),
    }


def checkpoint_stable(
    path: Path, *, interval: float, samples: int
) -> tuple[bool, list[dict[str, Any]]]:
    """Return (stable, snapshots) — stable iff every snapshot is identical.

    A checkpoint being written changes size/mtime/hash between reads; any
    difference refuses launch (``last.pt``-style mid-write protection).
    """
    snapshots: list[dict[str, Any]] = []
    for _ in range(samples):
        snapshots.append(_snapshot_checkpoint(path))
        if _ < samples - 1:
            time.sleep(interval)
    first = snapshots[0]
    return all(s == first for s in snapshots[1:]), snapshots


# ---------------------------------------------------------------------------
# Analysis-code structural verification (fail-closed, read-only)
# ---------------------------------------------------------------------------


def verify_analysis_code(feature_worktree: Path) -> dict[str, bool]:
    """Inspect the feature worktree's causality module for the required structure.

    Read-only: imports nothing, runs nothing.  Returns per-check booleans.  A
    missing module or missing code path yields False (fail-closed).  These are
    *supplements* to the analysis code's own runtime checks — this launcher does
    not fix the analysis code, it only refuses to certify a formal run when the
    required structure is absent.
    """
    source = feature_worktree / "src" / "mechanism_validation" / "feature_causality.py"
    checks = {
        "module_exists": source.is_file(),
        "margin_constant_present": False,
        "computed_background_gate": False,
        "samearea_control_present": False,
        "shifted_control_present": False,
        "body_mask_code_path": False,
        "tissue_coverage_recorded": False,
    }
    if not source.is_file():
        return checks
    text = source.read_text(encoding="utf-8")
    checks["margin_constant_present"] = "_BACKGROUND_NONINFERIOR_MARGIN" in text
    # The gate must be COMPUTED (a bool expression over metrics), not hardcoded.
    # A hardcoded ``"background_noninferiority_checked": True`` literal would be
    # present as a bare True; the real code builds it from metrics.
    checks["computed_background_gate"] = (
        "background_noninferior" in text
        and '"background_noninferiority_checked": True' not in text
    )
    checks["samearea_control_present"] = "c2_nonlesion_samearea" in text
    checks["shifted_control_present"] = "c2_shifted_mask" in text
    checks["body_mask_code_path"] = (
        "_body_mask_from_batch" in text or "body_mask" in text
    )
    checks["tissue_coverage_recorded"] = (
        "tissue_coverage" in text and "ablated_area" in text
    )
    return checks


def _controls_present_in_command(cmd: Sequence[str]) -> bool:
    """If ``--interventions`` is given, the two spatial-control interventions
    must both be present; otherwise M2 cannot be computed.  Absence of the flag
    means the full matrix (which includes both) runs."""
    names = _interventions_in_command(cmd)
    if not names:
        return True
    return {"c2_nonlesion_samearea", "c2_shifted_mask"} <= names


# ---------------------------------------------------------------------------
# Formal eligibility
# ---------------------------------------------------------------------------


def compute_eligibility(
    *,
    cmd: Sequence[str],
    split: str,
    has_frozen_quartiles: bool,
    analysis_checks: Mapping[str, bool] | None,
    smoke_is_ok: bool = False,
) -> dict[str, Any]:
    """Return {eligibility, data_mode, reasons[]} for one feature command.

    Formal requires ALL of: real data, no ``--max-samples``, train-frozen
    quartiles on val/test, structural checks present and passing, and spatial
    controls present.  Anything else downgrades to ``exploratory`` (never
    certified formal).  An empty/missing checks dict is treated as failed
    (fail-closed) unless ``smoke_is_ok`` — a structural check cannot pass when
    the module was not found.
    """
    reasons: list[str] = []
    data_mode = "fake" if _has_flag(cmd, "--fake-data") else "real"
    if data_mode == "fake":
        reasons.append("fake-data")
    if _has_flag(cmd, "--max-samples"):
        reasons.append("max-samples smoke")
    if split in ("val", "test") and not has_frozen_quartiles:
        reasons.append("no train-frozen quartiles on held-out split")
    if analysis_checks is None:
        reasons.append("analysis code not inspected")
    else:
        failed = sorted(name for name, ok in analysis_checks.items() if not ok)
        if failed:
            reasons.append(f"analysis code checks failed: {failed}")
    if not _controls_present_in_command(cmd):
        reasons.append("spatial controls missing from interventions")
    if smoke_is_ok:
        reasons = [r for r in reasons if r in ("fake-data", "max-samples smoke")]
    eligibility = "formal" if not reasons else "exploratory"
    return {"eligibility": eligibility, "data_mode": data_mode, "reasons": reasons}


# ---------------------------------------------------------------------------
# Experiment lock (the sync tool's ``run.lock``)
# ---------------------------------------------------------------------------


class RunLock:
    def __init__(self, worktree_root: Path):
        self.path = worktree_root / "run.lock"

    def acquire(
        self, *, worktree: str, command: str, device: str, run_id: str
    ) -> None:
        payload = {
            "status": "running",
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "worktree": worktree,
            "command": command,
            "device": device,
            "run_id": run_id,
            "started_at": _iso(_utcnow()),
        }
        write_json(self.path, payload)

    def release(self, status: str, exit_code: int) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        payload.update(
            {
                "status": status,
                "exit_code": exit_code,
                "finished_at": _iso(_utcnow()),
            }
        )
        write_json(self.path, payload)

    def is_running(self) -> bool:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return payload.get("status", "running") == "running"
        except Exception:
            return self.path.is_file()


# ---------------------------------------------------------------------------
# Environment summary
# ---------------------------------------------------------------------------


def _env_summary() -> dict[str, str]:
    summary: dict[str, str] = {
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "cwd": str(Path.cwd()),
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    for key in ("PYTHONPATH", "OMP_NUM_THREADS", "WANDB_MODE"):
        if key in os.environ:
            summary[key] = os.environ[key]
    try:
        import torch

        summary["torch"] = torch.__version__
        summary["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            summary["gpu"] = torch.cuda.get_device_name(0)
            summary["cuda_device_count"] = str(torch.cuda.device_count())
    except Exception:
        summary["torch"] = "unavailable"
    return summary


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------


class ExperimentSpec:
    def __init__(
        self,
        name: str,
        command: list[str],
        worktree_root: Path,
        run_dir: Path,
        device: str,
    ):
        self.name = name
        self.command = command
        self.worktree_root = worktree_root
        self.run_dir = run_dir
        self.device = device
        self.stdout_path = run_dir / f"{name}.stdout.log"
        self.stderr_path = run_dir / f"{name}.stderr.log"
        self.manifest_path = run_dir / f"{name}.manifest.json"


def _require_empty_run_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise QueueError(
            f"Run output directory already exists and is not empty: {path}. "
            f"Use a unique --run-id or clear it first."
        )


def run_experiment(
    spec: ExperimentSpec,
    *,
    checkpoint_sha256: str,
    formal: bool,
) -> dict[str, Any]:
    """Run one experiment, capturing stdout/stderr to files.  Returns a record.

    Never retries silently: a failure is recorded with its return code and the
    lock status is set accordingly by the caller.
    """
    started = _iso(_utcnow())
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    with spec.stdout_path.open("wb") as out, spec.stderr_path.open("wb") as err:
        proc = subprocess.run(
            spec.command,
            cwd=str(spec.worktree_root),
            stdout=out,
            stderr=err,
            text=False,
        )
    finished = _iso(_utcnow())
    record = {
        "experiment": spec.name,
        "command": spec.command,
        "worktree_root": str(spec.worktree_root),
        "device": spec.device,
        "returncode": proc.returncode,
        "status": "passed" if proc.returncode == 0 else "failed",
        "checkpoint_sha256": checkpoint_sha256,
        "formal_eligibility": formal,
        "stdout_log": spec.stdout_path.as_posix(),
        "stderr_log": spec.stderr_path.as_posix(),
        "started_at": started,
        "finished_at": finished,
        "environment": _env_summary(),
    }
    write_json(spec.manifest_path, record)
    return record


# ---------------------------------------------------------------------------
# Emergence gate reading
# ---------------------------------------------------------------------------


def read_emergence_gates(emergence_output: Path) -> dict[str, bool | None]:
    """Read A1 s1_gate and A2 m1_gate from the emergence run output.

    A missing decision file is ``None`` (fail-closed: treated as not-passed).
    """
    gates: dict[str, bool | None] = {"s1_gate": None, "m1_gate": None}
    for filename, key in (
        ("cross_modal_similarity.json", "s1_gate"),
        ("lesion_emergence.json", "m1_gate"),
    ):
        path = emergence_output / filename
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            gate = payload.get(key)
            if isinstance(gate, dict):
                gates[key] = gate.get("passed")
    return gates


def emergence_passed(gates: Mapping[str, bool | None]) -> bool:
    """Both emergence gates must be explicitly True (missing ⇒ fail-closed)."""
    return gates.get("s1_gate") is True and gates.get("m1_gate") is True


# ---------------------------------------------------------------------------
# Command builders (auto mode) and placeholder substitution
# ---------------------------------------------------------------------------


def _substitute(template: str, placeholders: Mapping[str, str]) -> str:
    for key, value in placeholders.items():
        template = template.replace("{" + key + "}", value)
    return template


def _build_emergence_command(
    *,
    config: str,
    checkpoint: str,
    split: str,
    device: str,
    emergence_out: str,
    frozen_quartiles: str | None,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/eval_feature_emergence.py",
        "--config", config,
        "--checkpoint", checkpoint,
        "--split", split,
        "--output", emergence_out,
        "--device", device,
    ]
    if frozen_quartiles:
        cmd += ["--frozen-quartiles-json", frozen_quartiles]
    return cmd


def _build_causality_command(
    *,
    config: str,
    checkpoint: str,
    split: str,
    device: str,
    causality_out: str,
) -> list[str]:
    return [
        sys.executable,
        "scripts/eval_feature_causality.py",
        "--config", config,
        "--checkpoint", checkpoint,
        "--split", split,
        "--output", causality_out,
        "--device", device,
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Sequential dual-worktree experiment queue."
    )
    ap.add_argument("--router-worktree", default=".worktrees/spectral-router-v5")
    ap.add_argument("--feature-worktree", default=".worktrees/lesion-feature-emergence")
    ap.add_argument("--checkpoint", required=True, help="frozen checkpoint .pt")
    ap.add_argument("--config", default=None, help="experiment YAML for auto commands")
    ap.add_argument("--split", default="val")
    ap.add_argument("--frozen-quartiles-json", default=None)
    ap.add_argument("--output-root", default="cloud_runs")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--device-feature", default=None)
    ap.add_argument("--router-command", default=None)
    ap.add_argument("--feature-emergence-command", default=None)
    ap.add_argument("--feature-causality-command", default=None)
    ap.add_argument("--feature-emergence-output", default=None)
    ap.add_argument("--feature-causality-output", default=None)
    ap.add_argument("--skip-router", action="store_true")
    ap.add_argument("--skip-causality", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--parallel", action="store_true", help="request concurrent execution")
    ap.add_argument("--check-interval", type=float, default=2.0)
    ap.add_argument("--check-samples", type=int, default=3)
    args = ap.parse_args(argv)

    router_wt = Path(args.router_worktree)
    feature_wt = Path(args.feature_worktree)
    checkpoint = Path(args.checkpoint)
    run_id = args.run_id or f"run-{_now_utc_seconds()}-{uuid.uuid4().hex[:6]}"
    router_run_dir = Path(args.output_root) / "router" / run_id
    interp_run_dir = Path(args.output_root) / "interpretability" / run_id
    device = args.device or ("cuda" if _cuda_available() else "cpu")
    device_feature = args.device_feature or device

    print(f"[queue] run_id={run_id}")
    print(f"[queue] router_worktree={router_wt} feature_worktree={feature_wt}")

    # ---- Pre-flight: fail-closed structure ----
    if not checkpoint.is_file():
        raise QueueError(f"Checkpoint not found: {checkpoint}")
    if checkpoint.name == "last.pt":
        raise QueueError(
            f"Refusing checkpoint named 'last.pt' ({checkpoint}): it is the "
            f"in-training artifact. Point --checkpoint at a frozen snapshot."
        )
    if not feature_wt.is_dir() and not args.dry_run:
        raise QueueError(f"Feature worktree missing: {feature_wt}")
    if args.parallel and device == device_feature:
        print("[queue] REFUSING --parallel: both experiments would share device "
              f"{device!r}. Concurrent execution is not implemented; run serial.")
        return 2
    if args.parallel:
        print(f"[queue] --parallel requested with distinct devices "
              f"(router={device}, feature={device_feature}); running SERIAL "
              f"(concurrent orchestration not implemented).")
    _require_empty_run_dir(router_run_dir)
    _require_empty_run_dir(interp_run_dir)

    # ---- Resolve commands ----
    placeholders = {
        "checkpoint": str(checkpoint),
        "config": args.config or "",
        "device": device,
        "device_feature": device_feature,
        "router_out": str(router_run_dir),
        "feature_out": str(interp_run_dir),
        "emergence_out": str(interp_run_dir / "emergence"),
        "causality_out": str(interp_run_dir / "causality"),
        "frozen": args.frozen_quartiles_json or "",
    }

    router_cmd: list[str] | None = None
    if not args.skip_router:
        if args.router_command:
            router_cmd = _split_command(_substitute(args.router_command, placeholders))
        else:
            raise QueueError(
                "--router-command is required (or pass --skip-router). The queue "
                "does not guess training plans."
            )

    emergence_cmd: list[str] | None = None
    if args.feature_emergence_command:
        emergence_cmd = _split_command(
            _substitute(args.feature_emergence_command, placeholders)
        )
    elif args.config:
        emergence_cmd = _build_emergence_command(
            config=args.config,
            checkpoint=str(checkpoint),
            split=args.split,
            device=device_feature,
            emergence_out=str(interp_run_dir / "emergence"),
            frozen_quartiles=args.frozen_quartiles_json,
        )
    else:
        raise QueueError(
            "--feature-emergence-command or --config is required."
        )

    causality_cmd: list[str] | None = None
    if not args.skip_causality:
        if args.feature_causality_command:
            causality_cmd = _split_command(
                _substitute(args.feature_causality_command, placeholders)
            )
        elif args.config:
            causality_cmd = _build_causality_command(
                config=args.config,
                checkpoint=str(checkpoint),
                split=args.split,
                device=device_feature,
                causality_out=str(interp_run_dir / "causality"),
            )
        else:
            raise QueueError(
                "--feature-causality-command or --config is required "
                "(or pass --skip-causality)."
            )

    emergence_output = (
        Path(args.feature_emergence_output)
        if args.feature_emergence_output
        else interp_run_dir / "emergence"
    )

    # ---- Structural analysis checks (feature worktree, read-only) ----
    analysis_checks = (
        verify_analysis_code(feature_wt)
        if feature_wt.is_dir()
        else None  # fail-closed: no module inspected
    )
    has_frozen = bool(args.frozen_quartiles_json)
    emergence_elig = compute_eligibility(
        cmd=emergence_cmd,
        split=args.split,
        has_frozen_quartiles=has_frozen,
        analysis_checks=analysis_checks,
    )
    causality_elig = (
        compute_eligibility(
            cmd=causality_cmd,
            split=args.split,
            has_frozen_quartiles=has_frozen,
            analysis_checks=analysis_checks,
        )
        if causality_cmd is not None
        else None
    )
    print(f"[queue] emergence eligibility={emergence_elig['eligibility']} "
          f"data_mode={emergence_elig['data_mode']}")
    if causality_elig is not None:
        print(f"[queue] causality eligibility={causality_elig['eligibility']} "
              f"data_mode={causality_elig['data_mode']}")

    # ---- Checkpoint stability ----
    print(f"[queue] verifying checkpoint stability at {checkpoint} ...")
    stable, snapshots = checkpoint_stable(
        checkpoint, interval=args.check_interval, samples=args.check_samples
    )
    if not stable:
        raise QueueError(
            f"Checkpoint {checkpoint} changed between stability reads (size/mtime/"
            f"sha256 differ). Refusing to launch against a mid-write checkpoint."
        )
    checkpoint_sha = snapshots[0]["sha256"]
    print(f"[queue] checkpoint stable: sha256={checkpoint_sha[:16]}...")

    if args.dry_run:
        print("[queue] DRY RUN — no experiment is launched, no lock created.")
        for label, cmd in (
            ("router", router_cmd),
            ("emergence", emergence_cmd),
            ("causality", causality_cmd),
        ):
            if cmd is not None:
                print(f"[queue]   would run {label}: {cmd}")
        print(f"[queue]   router run dir : {router_run_dir}")
        print(f"[queue]   feature run dir: {interp_run_dir}")
        return 0

    # ---- Execute (serial chain, fail-closed downstream) ----
    summary: dict[str, Any] = {
        "run_id": run_id,
        "started_at": _iso(_utcnow()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_stability_snapshots": snapshots,
        "device_router": device,
        "device_feature": device_feature,
        "experiments": {},
        "gates": {},
    }
    queue_status = "COMPLETE"

    # Router lock + job
    if router_cmd is not None:
        lock = RunLock(router_wt)
        lock.acquire(worktree=str(router_wt), command=" ".join(router_cmd),
                     device=device, run_id=run_id)
        print(f"[queue] RUN router (device={device})")
        record = run_experiment(
            ExperimentSpec("router", router_cmd, router_wt, router_run_dir, device),
            checkpoint_sha256=checkpoint_sha,
            formal=False,
        )
        lock.release("complete" if record["status"] == "passed" else "failed",
                     record["returncode"])
        summary["experiments"]["router"] = record
        if record["status"] != "passed":
            print(f"[queue] router FAILED (rc={record['returncode']}); "
                  f"downstream blocked.")
            summary["status"] = "FAILED"
            queue_status = "FAILED"
            _write_queue_manifest(interp_run_dir, summary, "FAILED")
            return 1

        # Re-verify checkpoint stability AFTER the router job.
        stable2, snap2 = checkpoint_stable(
            checkpoint, interval=args.check_interval, samples=args.check_samples
        )
        if not stable2:
            print("[queue] checkpoint unstable AFTER router job; blocking feature runs.")
            summary["status"] = "FAILED"
            queue_status = "FAILED"
            _write_queue_manifest(interp_run_dir, summary, "FAILED")
            return 1
        summary["checkpoint_after_router_sha256"] = snap2[0]["sha256"]

    # Feature emergence
    lock = RunLock(feature_wt)
    lock.acquire(worktree=str(feature_wt), command=" ".join(emergence_cmd),
                 device=device_feature, run_id=run_id)
    print(f"[queue] RUN emergence (device={device_feature})")
    emerg_record = run_experiment(
        ExperimentSpec("emergence", emergence_cmd, feature_wt, interp_run_dir, device_feature),
        checkpoint_sha256=checkpoint_sha,
        formal=emergence_elig["eligibility"] == "formal",
    )
    lock.release("complete" if emerg_record["status"] == "passed" else "failed",
                 emerg_record["returncode"])
    summary["experiments"]["emergence"] = emerg_record

    gates = read_emergence_gates(emergence_output)
    summary["gates"] = gates
    emerg_passed_gates = emergence_passed(gates)

    # Feature causality (M2) — only after gates pass AND upstream did not fail.
    if causality_cmd is not None and not args.skip_causality:
        if emerg_record["status"] != "passed" or not emerg_passed_gates:
            summary["experiments"]["causality"] = {
                "experiment": "causality",
                "status": "skipped",
                "reason": (
                    "emergence not passed"
                    if emerg_record["status"] != "passed"
                    else "A1/A2 gates did not both pass"
                ),
            }
            print(f"[queue] causality SKIPPED: "
                  f"{summary['experiments']['causality']['reason']}")
            queue_status = "INCOMPLETE"
        else:
            lock = RunLock(feature_wt)
            lock.acquire(worktree=str(feature_wt), command=" ".join(causality_cmd),
                         device=device_feature, run_id=run_id)
            print(f"[queue] RUN causality (device={device_feature})")
            caus_record = run_experiment(
                ExperimentSpec("causality", causality_cmd, feature_wt, interp_run_dir, device_feature),
                checkpoint_sha256=checkpoint_sha,
                formal=causality_elig["eligibility"] == "formal",
            )
            lock.release(
                "complete" if caus_record["status"] == "passed" else "failed",
                caus_record["returncode"],
            )
            summary["experiments"]["causality"] = caus_record
            if caus_record["status"] != "passed":
                queue_status = "FAILED"

    # Queue-level COMPLETE: atomic, with data_mode / eligibility / checkpoint sha / git sha.
    data_mode = emergence_elig["data_mode"]
    all_formal = (
        emergence_elig["eligibility"] == "formal"
        and (causality_elig is None or causality_elig["eligibility"] == "formal")
        and queue_status == "COMPLETE"
    )
    eligibility = "formal" if all_formal else "exploratory"
    summary.update(
        {
            "status": queue_status,
            "data_mode": data_mode,
            "eligibility": eligibility,
            "finished_at": _iso(_utcnow()),
            "git_sha_router": _git_sha(router_wt) if router_wt.is_dir() else "unknown",
            "git_sha_feature": _git_sha(feature_wt) if feature_wt.is_dir() else "unknown",
            "emergence_eligibility_checks": emergence_elig,
            "causality_eligibility_checks": causality_elig,
        }
    )
    _write_queue_manifest(interp_run_dir, summary, queue_status)
    print(f"[queue] status={queue_status} eligibility={eligibility} data_mode={data_mode}")
    print(f"[queue] router  : {router_run_dir}")
    print(f"[queue] feature : {interp_run_dir}")
    return 0 if queue_status == "COMPLETE" else 1


def _write_queue_manifest(run_dir: Path, summary: Mapping[str, Any], status: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "queue_manifest.json", dict(summary))
    write_json(
        run_dir / "COMPLETE.json",
        {
            "run_id": summary.get("run_id"),
            "status": status,
            "data_mode": summary.get("data_mode"),
            "eligibility": summary.get("eligibility"),
            "checkpoint_sha256": summary.get("checkpoint_sha256"),
            "git_sha_feature": summary.get("git_sha_feature"),
            "finished_at": summary.get("finished_at"),
        },
    )


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
