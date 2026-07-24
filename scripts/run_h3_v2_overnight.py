"""Run the isolated H3-v2 calibration and matched model experiment overnight.

The runner is deliberately fail closed:

* it requires a real CUDA device and never accepts the CPU debug profile;
* it seals and rechecks all registered runtime sources and historical evidence;
* calibration reads only mechanism-train/calibration patients;
* a calibration FAIL is a completed scientific stop, not an infrastructure
  failure;
* model training starts only after calibration PASS and writes exclusively
  below this run directory;
* no outcome authorizes production training/activation, H5, or H6.

Windows entry point::

    powershell -NoProfile -ExecutionPolicy Bypass -File \
        scripts/run_h3_v2_overnight.ps1

An interrupted calibration resumes verified timestep shards.  If interruption
occurs after model training starts, ``--resume`` creates a fresh, separately
owned experiment attempt; it never claims exact optimizer/RNG continuation and
never overwrites the partial attempt.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PIPELINE_ID = "H3_V2_NATIVE_NULL_OVERNIGHT_EXPERIMENT_V1"
CALIBRATION_PIPELINE_ID = "H3_V2_FULL_TIMESTEP_NATIVE_NULL_CALIBRATION_V1"
STAGE_ROOT = (
    ROOT
    / "results"
    / "mechanism_validation_v2"
    / "05C_h3_v2_full_timestep_native_null"
)
RUNS_ROOT = STAGE_ROOT / "overnight_runs"
PIPELINE_LOCK = STAGE_ROOT / ".overnight.lock"
GPU_LOCK = (
    ROOT
    / "results"
    / "mechanism_validation_v2"
    / ".development_gpu.lock"
)
PROTOCOL_CONFIG = ROOT / "configs" / "h3_v2_full_timestep_native_null_v1.json"
CALIBRATION_SCRIPT = ROOT / "scripts" / "calibrate_h3_v2_full_timestep_native_null.py"
BUILDER_SCRIPT = ROOT / "scripts" / "build_h3_v2_experiment_configs.py"
ANALYZER_SCRIPT = ROOT / "scripts" / "analyze_h3_v2_experiment.py"
TRAIN_SCRIPT = ROOT / "scripts" / "train_v2.py"
EVALUATE_SCRIPT = ROOT / "scripts" / "evaluate.py"
FREEZE_AUDIT = ROOT / "scripts" / "audit_v2_freeze_integrity.py"

TARGETED_TESTS = (
    ROOT / "tests" / "test_h3_v2_full_timestep_native_null.py",
    ROOT / "tests" / "test_h3_v2_overnight_runner.py",
    ROOT / "tests" / "test_v2_inference_admissibility.py",
)

# Every path here is read-only for this runner.  New output lives below the
# current run directory, which is not contained by any protected root.
PROTECTED_ROOTS = (
    ROOT / "results" / "mechanism_validation" / "03_h4_noise_calibration",
    ROOT
    / "results"
    / "mechanism_validation_v2"
    / "02_h4_v2_internal_exploratory_nested_cv",
    ROOT
    / "results"
    / "mechanism_validation_v2"
    / "04_main_integration_freeze",
    ROOT
    / "results"
    / "mechanism_validation_v2"
    / "05A_inference_admissibility_audit",
    ROOT
    / "results"
    / "mechanism_validation_v2"
    / "05B_h3_fixed_schedule_inference",
    ROOT / "checkpoints",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def config_self_hash(config: Mapping[str, Any]) -> str:
    body = dict(config)
    body.pop("config_sha256", None)
    return canonical_sha256(body)


def validate_protocol() -> tuple[dict[str, Any], dict[str, str]]:
    if not PROTOCOL_CONFIG.is_file():
        raise FileNotFoundError(PROTOCOL_CONFIG)
    config = read_object(PROTOCOL_CONFIG)
    declared = str(config.get("config_sha256", "")).lower()
    observed = config_self_hash(config)
    if declared != observed:
        raise ValueError(
            "H3-v2 protocol self-hash mismatch: "
            f"declared={declared}, observed={observed}"
        )
    if config.get("pipeline_id") != CALIBRATION_PIPELINE_ID:
        raise ValueError("Unexpected H3-v2 calibration pipeline identity")
    if config.get("historical_results_preserved", {}).get(
        "must_not_overwrite_or_relabel"
    ) is not True:
        raise ValueError("Protocol does not preserve historical H3/H4 decisions")
    stop = config.get("stop_and_claim_rules", {})
    forbidden_true = (
        "production_training",
        "production_activation",
        "h5_started",
        "h6_started",
        "next_production_stage_allowed",
    )
    if any(stop.get(key) is not False for key in forbidden_true):
        raise ValueError("Protocol contains an unauthorized activation claim")
    mapping = config.get("calibration", {}).get("mapping", {})
    if mapping.get("shallow_route_policy") != "structurally_zero":
        raise ValueError("H3-v2 shallow route must be structurally zero")

    sources: dict[str, str] = {}
    for label, spec in config.get("runtime_sources", {}).items():
        if not isinstance(spec, Mapping):
            raise ValueError(f"runtime_sources.{label} must be an object")
        path = resolve_path(str(spec.get("path", "")))
        expected = str(spec.get("file_sha256", "")).lower()
        if len(expected) != 64:
            raise ValueError(f"runtime_sources.{label} is not frozen")
        if not path.is_file():
            raise FileNotFoundError(path)
        current = sha256_file(path)
        if current != expected:
            raise ValueError(
                f"runtime_sources.{label} hash mismatch: "
                f"expected={expected}, observed={current}"
            )
        sources[str(path.relative_to(ROOT))] = current
    return config, sources


def snapshot_paths(
    *,
    runtime_sources: Mapping[str, str],
) -> dict[str, Any]:
    protocol = read_object(PROTOCOL_CONFIG)
    files: dict[str, str] = {}
    missing_roots: list[str] = []
    for relative, digest in runtime_sources.items():
        files[f"source::{relative}"] = digest
    extra_files = (
        PROTOCOL_CONFIG,
        ROOT / "configs" / "experiments" / "slmf_png_spectral_router_v5.yaml",
        ROOT / "main_data" / "split_manifest.csv",
        ROOT / "configs" / "dataset_contract_stage0a_v1.json",
        ROOT / "pixi.lock",
        resolve_path(protocol["dataset_contract"]["cache_lineage_path"]),
        resolve_path(protocol["mean_checkpoint"]["decision_path"]),
        resolve_path(protocol["mean_checkpoint"]["sidecar_path"]),
    )
    for path in extra_files:
        if not path.is_file():
            raise FileNotFoundError(path)
        files[f"input::{path.relative_to(ROOT)}"] = sha256_file(path)
    for root in PROTECTED_ROOTS:
        if not root.exists():
            missing_roots.append(str(root.relative_to(ROOT)))
            continue
        for path in sorted(
            (candidate for candidate in root.rglob("*") if candidate.is_file()),
            key=lambda item: item.as_posix(),
        ):
            if is_relative_to(path, STAGE_ROOT):
                continue
            files[f"protected::{path.relative_to(ROOT)}"] = sha256_file(path)
    return {
        "schema_version": 1,
        "files": files,
        "missing_optional_protected_roots": missing_roots,
        "snapshot_sha256": canonical_sha256(files),
    }


def _process_alive(pid: int) -> bool | None:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return None
        return True
    process_query_limited_information = 0x1000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint32,
    ]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.OpenProcess(
        process_query_limited_information,
        False,
        int(pid),
    )
    if not handle:
        return False if ctypes.get_last_error() == 87 else None
    kernel32.CloseHandle(handle)
    return True


class StageFailure(RuntimeError):
    def __init__(self, message: str, exit_code: int = 2):
        super().__init__(message)
        self.exit_code = int(exit_code) if int(exit_code) else 2


class ExclusiveFileLock:
    """Cooperative create-new lock; only the token owner may release it."""

    def __init__(self, path: Path, run_dir: Path, purpose: str):
        self.path = path
        self.run_dir = run_dir.resolve()
        self.purpose = purpose
        self.token = uuid.uuid4().hex
        self.fd: int | None = None

    def _recover_same_run_dead_owner(self) -> bool:
        try:
            before = self.path.stat()
            owner = read_object(self.path)
            owner_run = Path(str(owner["run_dir"])).resolve()
            owner_pid = int(owner["pid"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if owner_run != self.run_dir or _process_alive(owner_pid) is not False:
            return False
        try:
            after = self.path.stat()
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                return False
            self.path.unlink()
            return True
        except OSError:
            return False

    def acquire(self, *, resume: bool) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            try:
                self.fd = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
                break
            except FileExistsError as exc:
                if resume and attempt == 0 and self._recover_same_run_dead_owner():
                    continue
                try:
                    owner = self.path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                except OSError:
                    owner = "<unreadable>"
                raise StageFailure(
                    f"{self.purpose} lock is active or unverifiable:\n"
                    f"  {self.path}\n{owner}",
                    11,
                ) from exc
        if self.fd is None:
            raise StageFailure(f"Could not acquire {self.purpose} lock", 11)
        record = {
            "schema_version": 1,
            "purpose": self.purpose,
            "token": self.token,
            "pid": os.getpid(),
            "run_dir": str(self.run_dir),
            "created_at_utc": utc_now(),
        }
        os.write(
            self.fd,
            (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode(
                "utf-8"
            ),
        )
        os.fsync(self.fd)

    def release(self) -> None:
        if self.fd is None:
            return
        os.close(self.fd)
        self.fd = None
        try:
            owner = read_object(self.path)
            if owner.get("token") == self.token:
                self.path.unlink()
        except (OSError, ValueError, json.JSONDecodeError):
            return


class KeepWindowsAwake:
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self) -> None:
        self.enabled = False
        self.detail = "not entered"

    def __enter__(self) -> "KeepWindowsAwake":
        if os.name != "nt":
            self.detail = "non-Windows; no execution-state request"
            return self
        result = ctypes.windll.kernel32.SetThreadExecutionState(  # type: ignore[attr-defined]
            self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
        )
        if result == 0:
            raise StageFailure(
                "Windows sleep inhibition failed; refusing unattended work",
                12,
            )
        self.enabled = True
        self.detail = "ES_CONTINUOUS|ES_SYSTEM_REQUIRED"
        return self

    def __exit__(self, *_: object) -> None:
        if self.enabled and os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(  # type: ignore[attr-defined]
                self.ES_CONTINUOUS
            )


class RunLogger:
    def __init__(self, path: Path, *, append: bool):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open(
            "a" if append else "w",
            encoding="utf-8",
            newline="\n",
            buffering=1,
        )
        self.lock = threading.Lock()

    def emit(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        with self.lock:
            print(line, flush=True)
            self.handle.write(line + "\n")
            self.handle.flush()

    def stage_line(self, stage_id: str, line: str) -> None:
        clean = line.rstrip("\r\n")
        with self.lock:
            print(f"[{stage_id}] {clean}", flush=True)
            self.handle.write(f"[{utc_now()}] [{stage_id}] {clean}\n")
            self.handle.flush()

    def close(self) -> None:
        with self.lock:
            self.handle.flush()
            self.handle.close()


def resolve_run_dir(value: str | None, *, resume: bool) -> Path:
    if value is None:
        if resume:
            raise ValueError("--resume requires --run-dir")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{run_id}-{os.getpid()}"
        return (RUNS_ROOT / run_id).resolve()
    path = resolve_path(value)
    if path == RUNS_ROOT.resolve() or not is_relative_to(path, RUNS_ROOT):
        raise ValueError(
            f"run directory must be a child of {RUNS_ROOT}, got {path}"
        )
    return path


class OvernightRunner:
    def __init__(
        self,
        args: argparse.Namespace,
        run_dir: Path,
        keep_awake: KeepWindowsAwake,
    ):
        self.args = args
        self.run_dir = run_dir
        self.keep_awake = keep_awake
        self.logs_dir = run_dir / "logs"
        self.stages_dir = run_dir / "stages"
        self.artifacts_dir = run_dir / "artifacts"
        self.context_path = run_dir / "run_context.json"
        self.state_path = run_dir / "state.json"
        self.summary_path = run_dir / "summary.json"
        self.heartbeat_path = run_dir / "heartbeat.json"
        self.logger = RunLogger(
            run_dir / "overnight.log",
            append=bool(args.resume),
        )
        self.context: dict[str, Any] = {}
        self.state: dict[str, Any] = {}
        self.current_process: subprocess.Popen[str] | None = None
        self.stop_heartbeat = threading.Event()
        self.heartbeat_thread: threading.Thread | None = None
        self.heartbeat_error: str | None = None
        self.scientific_decision: str | None = None
        self.experiment_decision: str | None = None
        self.development_training_started = False

    def emit(self, message: str) -> None:
        self.logger.emit(message)

    def update_state(self, **changes: Any) -> None:
        self.state.update(changes)
        self.state["updated_at_utc"] = utc_now()
        write_json_atomic(self.state_path, self.state)

    def write_summary(
        self,
        execution_status: str,
        *,
        error: str | None = None,
        exit_code: int | None = None,
    ) -> None:
        payload = {
            "schema_version": 1,
            "pipeline_id": PIPELINE_ID,
            "run_id": self.run_dir.name,
            "run_dir": str(self.run_dir),
            "execution_status": execution_status,
            "calibration_decision": self.scientific_decision,
            "experiment_decision": self.experiment_decision,
            "development_training_started": self.development_training_started,
            "production_training_started": False,
            "production_activation_allowed": False,
            "h5_h6_started": False,
            "does_not_override_05B": True,
            "current_stage": self.state.get("current_stage"),
            "completed_stages": list(self.state.get("completed_stages", [])),
            "experiment_attempt": self.state.get("experiment_attempt", 1),
            "started_at_utc": self.context.get("started_at_utc"),
            "updated_at_utc": utc_now(),
            "error": error,
            "exit_code": exit_code,
        }
        write_json_atomic(self.summary_path, payload)

    def start_heartbeat(self) -> None:
        def worker() -> None:
            while not self.stop_heartbeat.wait(self.args.heartbeat_seconds):
                try:
                    write_json_atomic(
                        self.heartbeat_path,
                        {
                            "schema_version": 1,
                            "pipeline_id": PIPELINE_ID,
                            "run_id": self.run_dir.name,
                            "pid": os.getpid(),
                            "heartbeat_at_utc": utc_now(),
                            "status": self.state.get("status"),
                            "current_stage": self.state.get("current_stage"),
                            "child_pid": self.state.get("child_pid"),
                            "experiment_attempt": self.state.get(
                                "experiment_attempt",
                                1,
                            ),
                        },
                    )
                except Exception as exc:  # pragma: no cover - disk failure
                    self.heartbeat_error = f"{type(exc).__name__}: {exc}"
                    self.stop_heartbeat.set()

        self.heartbeat_thread = threading.Thread(
            target=worker,
            daemon=True,
            name="h3-v2-heartbeat",
        )
        self.heartbeat_thread.start()

    def stop_heartbeat_thread(self) -> None:
        self.stop_heartbeat.set()
        if self.heartbeat_thread is not None:
            self.heartbeat_thread.join(timeout=5)
        write_json_atomic(
            self.heartbeat_path,
            {
                "schema_version": 1,
                "pipeline_id": PIPELINE_ID,
                "run_id": self.run_dir.name,
                "pid": os.getpid(),
                "heartbeat_at_utc": utc_now(),
                "status": self.state.get("status"),
                "current_stage": self.state.get("current_stage"),
                "child_pid": None,
            },
        )

    def _record_path(self, stage_id: str) -> Path:
        return self.stages_dir / f"{stage_id}.json"

    def _stage_log_path(self, stage_id: str) -> Path:
        if stage_id.startswith(
            ("05_", "06_", "07_", "08_", "09_", "10_", "11_")
        ):
            return (
                self._experiment_plan_path().parent
                / "logs"
                / f"{stage_id}.log"
            )
        return self.logs_dir / f"{stage_id}.log"

    def _runner_contract(self) -> dict[str, Any]:
        return {
            "epochs": int(self.args.epochs),
            "device_index": int(self.args.device_index),
            "shard_size": int(self.args.shard_size),
            "skip_full_tests": bool(self.args.skip_full_tests),
            "min_free_disk_gb": float(self.args.min_free_disk_gb),
            "min_free_gpu_gb": float(self.args.min_free_gpu_gb),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        }

    def _record_completed(
        self,
        *,
        stage_id: str,
        label: str,
        command: Sequence[str] | None,
        started_at: str,
        artifacts: Sequence[Path],
        validation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        artifact_rows = []
        for path in artifacts:
            if not path.is_file():
                raise StageFailure(f"Stage artifact is missing: {path}", 20)
            artifact_rows.append(
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
            )
        record = {
            "schema_version": 1,
            "pipeline_id": PIPELINE_ID,
            "stage_id": stage_id,
            "label": label,
            "status": "PASS",
            "command": list(command) if command is not None else None,
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "exit_code": 0,
            "validation": dict(validation or {}),
            "artifacts": artifact_rows,
        }
        write_json_atomic(self._record_path(stage_id), record)
        completed = list(self.state.get("completed_stages", []))
        if stage_id not in completed:
            completed.append(stage_id)
        self.update_state(
            completed_stages=completed,
            current_stage=None,
            child_pid=None,
        )
        self.write_summary("RUNNING")
        return record

    def _resume_record(
        self,
        stage_id: str,
        command: Sequence[str] | None,
    ) -> dict[str, Any] | None:
        if not self.args.resume:
            return None
        path = self._record_path(stage_id)
        if not path.is_file():
            return None
        record = read_object(path)
        expected_command = list(command) if command is not None else None
        if (
            record.get("pipeline_id") != PIPELINE_ID
            or record.get("stage_id") != stage_id
            or record.get("status") != "PASS"
            or record.get("command") != expected_command
        ):
            raise StageFailure(f"Stage resume record is not reusable: {path}", 16)
        for artifact in record.get("artifacts", []):
            artifact_path = Path(str(artifact["path"]))
            if (
                not artifact_path.is_file()
                or sha256_file(artifact_path) != artifact.get("sha256")
            ):
                raise StageFailure(
                    f"Completed stage artifact changed: {artifact_path}",
                    16,
                )
        self.emit(f"resume: verified and skipped {stage_id}")
        return record

    def prepare_context(self) -> None:
        protocol, runtime_sources = validate_protocol()
        if self.args.resume:
            self.context = read_object(self.context_path)
            self.state = read_object(self.state_path)
            if (
                self.context.get("pipeline_id") != PIPELINE_ID
                or Path(str(self.context.get("run_dir"))).resolve()
                != self.run_dir.resolve()
                or self.context.get("protocol_config_sha256")
                != protocol["config_sha256"]
                or self.context.get("runtime_source_hashes")
                != runtime_sources
                or self.context.get("runner_contract")
                != self._runner_contract()
                or Path(str(self.context.get("python_executable"))).resolve()
                != Path(sys.executable).resolve()
            ):
                raise StageFailure("Resume context identity mismatch", 16)
            current_snapshot = snapshot_paths(runtime_sources=runtime_sources)
            if (
                current_snapshot.get("snapshot_sha256")
                != self.context.get("input_snapshot", {}).get("snapshot_sha256")
            ):
                raise StageFailure(
                    "Registered sources or protected historical evidence changed "
                    "since this run began",
                    16,
                )
            self.scientific_decision = self.state.get("calibration_decision")
            self.experiment_decision = self.state.get("experiment_decision")
            self.development_training_started = bool(
                self.state.get("development_training_started", False)
            )
            self._prepare_resume_experiment_attempt()
            self.update_state(status="RUNNING", current_stage=None, child_pid=None)
            self.emit("resume context and immutable inputs verified")
            return

        snapshot = snapshot_paths(runtime_sources=runtime_sources)
        self.context = {
            "schema_version": 1,
            "pipeline_id": PIPELINE_ID,
            "run_id": self.run_dir.name,
            "run_dir": str(self.run_dir),
            "protocol_config_path": str(PROTOCOL_CONFIG),
            "protocol_config_sha256": protocol["config_sha256"],
            "runtime_source_hashes": runtime_sources,
            "runner_contract": self._runner_contract(),
            "input_snapshot": snapshot,
            "python_executable": sys.executable,
            "python_version": sys.version,
            "command_line": list(sys.argv),
            "keep_awake": self.keep_awake.detail,
            "started_at_utc": utc_now(),
        }
        self.state = {
            "schema_version": 1,
            "pipeline_id": PIPELINE_ID,
            "run_id": self.run_dir.name,
            "status": "RUNNING",
            "current_stage": None,
            "child_pid": None,
            "completed_stages": [],
            "experiment_attempt": 1,
            "calibration_decision": None,
            "experiment_decision": None,
            "development_training_started": False,
            "resume_resource_preflights": [],
            "created_at_utc": utc_now(),
        }
        write_json_atomic(self.context_path, self.context)
        write_json_atomic(
            self.artifacts_dir / "input_snapshot.json",
            snapshot,
        )
        self.update_state()
        self.write_summary("RUNNING")

    def _prepare_resume_experiment_attempt(self) -> None:
        """Restart an interrupted training attempt without overwriting it."""
        current = self.state.get("current_stage")
        status = str(self.state.get("status", ""))
        if not isinstance(current, str):
            return
        if not current.startswith(("07_", "08_")):
            return
        if status not in {"FAILED", "INTERRUPTED", "RUNNING"}:
            return
        old_attempt = int(self.state.get("experiment_attempt", 1))
        new_attempt = old_attempt + 1
        for stage_id in (
            "05_build_experiment_configs",
            "06_gpu_model_smoke",
            "07_train_no_route",
            "08_train_h3_v2",
            "09_evaluate_no_route",
            "10_evaluate_h3_v2",
            "11_analyze_experiment",
        ):
            record = self._record_path(stage_id)
            if not record.exists():
                continue
            archived = self.stages_dir / (
                f"{stage_id}.attempt_{old_attempt:02d}.json"
            )
            if archived.exists():
                raise StageFailure(
                    f"Cannot archive duplicate attempt record: {archived}",
                    16,
                )
            os.replace(record, archived)
        completed = [
            stage
            for stage in self.state.get("completed_stages", [])
            if not str(stage).startswith(("05_", "06_", "07_", "08_", "09_", "10_", "11_"))
        ]
        self.state["experiment_attempt"] = new_attempt
        self.state["completed_stages"] = completed
        self.state["current_stage"] = None
        self.state["child_pid"] = None
        self.emit(
            "interrupted model training detected; preserving attempt "
            f"{old_attempt:02d} and starting clean attempt {new_attempt:02d}"
        )

    def _terminate_process_tree(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        else:
            process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)

    def _stream_process(
        self,
        *,
        stage_id: str,
        command: Sequence[str],
        log_path: Path,
        timeout_seconds: float,
    ) -> int:
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONUNBUFFERED": "1",
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "PIXI_LOCKED": "1",
            }
        )
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        )
        process = subprocess.Popen(
            list(command),
            cwd=str(ROOT),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        self.current_process = process
        self.update_state(child_pid=process.pid)
        assert process.stdout is not None
        output_queue: queue.Queue[str | None] = queue.Queue()

        def reader() -> None:
            try:
                for line in process.stdout:
                    output_queue.put(line)
            finally:
                output_queue.put(None)

        reader_thread = threading.Thread(target=reader, daemon=True)
        reader_thread.start()
        deadline = (
            time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
        )
        stream_closed = False
        timed_out = False
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", newline="\n", buffering=1) as log:
            while True:
                if self.heartbeat_error is not None:
                    self._terminate_process_tree(process)
                    raise StageFailure(
                        f"Heartbeat write failed: {self.heartbeat_error}",
                        18,
                    )
                if deadline is not None and time.monotonic() >= deadline:
                    timed_out = True
                    self.logger.stage_line(
                        stage_id,
                        f"TIMEOUT after {timeout_seconds:.0f} seconds",
                    )
                    self._terminate_process_tree(process)
                    break
                try:
                    item = output_queue.get(timeout=0.5)
                except queue.Empty:
                    if process.poll() is not None and stream_closed:
                        break
                    continue
                if item is None:
                    stream_closed = True
                    if process.poll() is not None:
                        break
                    continue
                log.write(item)
                log.flush()
                self.logger.stage_line(stage_id, item)
        reader_thread.join(timeout=5)
        code = process.wait()
        self.current_process = None
        self.update_state(child_pid=None)
        return 124 if timed_out else code

    def run_command_stage(
        self,
        *,
        stage_id: str,
        label: str,
        command: Sequence[str],
        timeout_seconds: float,
        artifact_paths: Sequence[Path] = (),
        validation: Mapping[str, Any] | None = None,
        validator: (
            Callable[[], tuple[Mapping[str, Any], Sequence[Path]]] | None
        ) = None,
    ) -> dict[str, Any]:
        resumed = self._resume_record(stage_id, command)
        if resumed is not None:
            return resumed
        self.emit(f"==== {stage_id}: {label} ====")
        self.emit(subprocess.list2cmdline(list(command)))
        self.update_state(
            status="RUNNING",
            current_stage=stage_id,
            child_pid=None,
        )
        started = utc_now()
        log_path = self._stage_log_path(stage_id)
        try:
            code = self._stream_process(
                stage_id=stage_id,
                command=command,
                log_path=log_path,
                timeout_seconds=timeout_seconds,
            )
        except BaseException:
            if self.current_process is not None:
                self._terminate_process_tree(self.current_process)
                self.current_process = None
            raise
        if code != 0:
            raise StageFailure(f"{label} exited with code {code}", code)
        combined_validation = dict(validation or {})
        extra_artifacts: Sequence[Path] = ()
        if validator is not None:
            observed_validation, extra_artifacts = validator()
            combined_validation.update(dict(observed_validation))
        artifacts = [log_path, *artifact_paths, *extra_artifacts]
        return self._record_completed(
            stage_id=stage_id,
            label=label,
            command=command,
            started_at=started,
            artifacts=artifacts,
            validation=combined_validation,
        )

    def run_internal_stage(
        self,
        *,
        stage_id: str,
        label: str,
        action,
    ) -> dict[str, Any]:
        resumed = self._resume_record(stage_id, None)
        if resumed is not None:
            return resumed
        self.emit(f"==== {stage_id}: {label} ====")
        self.update_state(status="RUNNING", current_stage=stage_id)
        started = utc_now()
        validation, artifacts = action()
        return self._record_completed(
            stage_id=stage_id,
            label=label,
            command=None,
            started_at=started,
            artifacts=artifacts,
            validation=validation,
        )

    def _historical_failure_checks(self) -> dict[str, Any]:
        historical_checks: dict[str, Any] = {}
        for label, path, expected_pipeline, required_false in (
            (
                "05A",
                ROOT
                / "results"
                / "mechanism_validation_v2"
                / "05A_inference_admissibility_audit"
                / "decision.json",
                "V2_INFERENCE_ADMISSIBILITY_AUDIT_V1",
                (
                    "production_inference_admissible",
                    "production_router_activation_allowed",
                    "next_stage_allowed",
                ),
            ),
            (
                "05B",
                ROOT
                / "results"
                / "mechanism_validation_v2"
                / "05B_h3_fixed_schedule_inference"
                / "decision.json",
                "H3_FIXED_SCHEDULE_INFERENCE_V1",
                (
                    "h3_fixed_schedule_inference_admissible",
                    "model_evaluation_allowed",
                    "production_activation_allowed",
                    "next_stage_allowed",
                ),
            ),
        ):
            decision = read_object(path)
            body = dict(decision)
            declared = str(body.pop("decision_sha256", ""))
            if (
                decision.get("pipeline_id") != expected_pipeline
                or decision.get("decision") != "FAIL"
                or declared != canonical_sha256(body)
                or any(decision.get(field) is not False for field in required_false)
            ):
                raise StageFailure(
                    f"Historical {label} FAIL decision is missing or invalid",
                    29,
                )
            historical_checks[label] = {
                "path": str(path),
                "file_sha256": sha256_file(path),
                "decision_sha256": declared,
                "decision": "FAIL",
            }
        return historical_checks

    def _resource_preflight_payload(self) -> dict[str, Any]:
        import torch

        if not torch.cuda.is_available():
            raise StageFailure(
                "CUDA is unavailable; formal H3-v2 work refuses CPU fallback",
                30,
            )
        device_count = torch.cuda.device_count()
        if self.args.device_index < 0 or self.args.device_index >= device_count:
            raise StageFailure(
                f"CUDA device index {self.args.device_index} is unavailable",
                30,
            )
        torch.cuda.set_device(self.args.device_index)
        device = torch.device(f"cuda:{self.args.device_index}")
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        minimum = int(self.args.min_free_gpu_gb * 1024**3)
        if free_bytes < minimum:
            raise StageFailure(
                f"Only {free_bytes / 1024**3:.2f} GiB GPU memory is free; "
                f"{self.args.min_free_gpu_gb:.2f} GiB required",
                30,
            )
        left = torch.randn((8, 192, 192), device=device)
        right = torch.randn((8, 192, 192), device=device)
        result = torch.bmm(left, right)
        torch.cuda.synchronize(device)
        if not torch.isfinite(result).all().item():
            raise StageFailure("CUDA smoke kernel produced non-finite output", 30)
        del left, right, result
        torch.cuda.empty_cache()
        disk = shutil.disk_usage(self.run_dir)
        if disk.free < int(self.args.min_free_disk_gb * 1024**3):
            raise StageFailure(
                f"Only {disk.free / 1024**3:.2f} GiB disk is free; "
                f"{self.args.min_free_disk_gb:.2f} GiB required",
                31,
            )
        properties = torch.cuda.get_device_properties(device)
        payload = {
            "schema_version": 1,
            "device_index": self.args.device_index,
            "device_name": properties.name,
            "device_total_memory_bytes": int(properties.total_memory),
            "cuda_free_memory_bytes": int(free_bytes),
            "cuda_total_memory_bytes": int(total_bytes),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "actual_cuda_kernel_passed": True,
            "disk_free_bytes": int(disk.free),
            "cpu_fallback_allowed": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
            "created_at_utc": utc_now(),
        }
        return payload

    def _gpu_preflight(self) -> tuple[dict[str, Any], list[Path]]:
        historical_checks = self._historical_failure_checks()
        payload = self._resource_preflight_payload()
        payload["launch_kind"] = "initial"
        payload["historical_failures_preserved"] = historical_checks
        path = self.artifacts_dir / "preflight" / "gpu_and_disk.json"
        if self.args.resume and path.exists():
            raise StageFailure(
                "Unsealed initial resource preflight artifact exists; "
                "refusing to overwrite it during resume",
                16,
            )
        write_json_atomic(path, payload)
        return payload, [path]

    def _resume_resource_preflight(self) -> Path:
        """Recheck current launch resources without replacing stage-00 evidence."""
        historical_checks = self._historical_failure_checks()
        payload = self._resource_preflight_payload()
        payload["launch_kind"] = "resume"
        payload["historical_failures_preserved"] = historical_checks
        launch_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + f"-{os.getpid()}-{uuid.uuid4().hex}"
        )
        path = (
            self.artifacts_dir
            / "preflight"
            / "resume_launches"
            / f"{launch_id}.json"
        )
        write_json_atomic(path, payload)
        records = list(self.state.get("resume_resource_preflights", []))
        records.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "created_at_utc": payload["created_at_utc"],
                "pid": os.getpid(),
            }
        )
        self.update_state(resume_resource_preflights=records)
        self.emit(f"resume launch resource preflight passed: {path}")
        return path

    def _final_integrity(self) -> tuple[dict[str, Any], list[Path]]:
        _, runtime_sources = validate_protocol()
        current = snapshot_paths(runtime_sources=runtime_sources)
        initial = self.context["input_snapshot"]
        if current["snapshot_sha256"] != initial["snapshot_sha256"]:
            initial_files = initial["files"]
            current_files = current["files"]
            changed = sorted(
                key
                for key in set(initial_files) | set(current_files)
                if initial_files.get(key) != current_files.get(key)
            )
            raise StageFailure(
                "Protected source/input/evidence changed during H3-v2 run: "
                + ", ".join(changed[:20]),
                40,
            )
        payload = {
            "schema_version": 1,
            "decision": "PASS",
            "initial_snapshot_sha256": initial["snapshot_sha256"],
            "final_snapshot_sha256": current["snapshot_sha256"],
            "protected_files_unchanged": True,
            "production_training_started": False,
            "h5_h6_started": False,
            "created_at_utc": utc_now(),
        }
        path = self.artifacts_dir / "final_integrity" / "decision.json"
        write_json_atomic(path, payload)
        return payload, [path]

    def _calibration_decision_path(self) -> Path:
        return self.artifacts_dir / "calibration" / "decision.json"

    def _validate_calibration_output(
        self,
    ) -> tuple[dict[str, Any], list[Path]]:
        decision_path = self._calibration_decision_path()
        calibration = read_object(decision_path)
        body = dict(calibration)
        declared = str(body.pop("decision_sha256", ""))
        if (
            calibration.get("pipeline_id") != CALIBRATION_PIPELINE_ID
            or calibration.get("calibration_complete") is not True
            or calibration.get("decision") not in {"PASS", "FAIL"}
            or declared != canonical_sha256(body)
            or calibration.get("production_training_allowed") is not False
            or calibration.get("production_activation_allowed") is not False
            or calibration.get("h5_h6_allowed") is not False
        ):
            raise StageFailure(
                "Calibration decision is incomplete, unsealed, or unsafe",
                33,
            )
        artifacts: list[Path] = []
        if calibration["decision"] == "PASS":
            schedule = decision_path.parent / "frozen_schedule.json"
            if (
                not schedule.is_file()
                or sha256_file(schedule)
                != calibration.get("schedule_file_sha256")
                or calibration.get("model_experiment_allowed") is not True
            ):
                raise StageFailure("Calibration PASS schedule is not sealed", 33)
            artifacts.append(schedule)
        elif calibration.get("model_experiment_allowed") is not False:
            raise StageFailure("Calibration FAIL unexpectedly allows training", 33)
        return {
            "decision": calibration["decision"],
            "decision_sha256": declared,
            "model_experiment_allowed": calibration[
                "model_experiment_allowed"
            ],
        }, artifacts

    def _experiment_attempt_name(self) -> str:
        return f"attempt_{int(self.state.get('experiment_attempt', 1)):02d}"

    def _experiment_output_root(self) -> Path:
        return self.artifacts_dir / "model_experiment"

    def _experiment_plan_path(self) -> Path:
        return (
            self._experiment_output_root()
            / "attempts"
            / self._experiment_attempt_name()
            / "experiment_plan.json"
        )

    @staticmethod
    def _plan_variants(plan: Mapping[str, Any]) -> Mapping[str, Any]:
        variants = plan.get("variants", plan.get("experiments"))
        if not isinstance(variants, Mapping):
            raise StageFailure("Experiment plan lacks variants", 34)
        return variants

    @staticmethod
    def _variant(
        variants: Mapping[str, Any],
        *names: str,
    ) -> Mapping[str, Any]:
        for name in names:
            value = variants.get(name)
            if isinstance(value, Mapping):
                return value
        raise StageFailure(f"Experiment plan lacks variant {names}", 34)

    def _load_plan(self) -> dict[str, Any]:
        path = self._experiment_plan_path()
        if not path.is_file():
            raise StageFailure(f"Experiment plan is missing: {path}", 34)
        plan = read_object(path)
        declared = str(plan.get("plan_sha256", ""))
        body = dict(plan)
        body.pop("plan_sha256", None)
        if declared != canonical_sha256(body):
            raise StageFailure("Experiment plan self-hash mismatch", 34)
        return plan

    def _model_smoke(self) -> tuple[dict[str, Any], list[Path]]:
        import torch

        from src.model.config_utils import load_full_config
        from src.model.slmf_bbdm import SLMFBBDM

        plan = self._load_plan()
        variants = self._plan_variants(plan)
        rows: dict[str, Any] = {}
        for canonical_name, aliases in (
            ("no_route", ("no_route",)),
            (
                "h3_v2_native_null",
                ("h3_v2_native_null", "h3_native_null"),
            ),
        ):
            spec = self._variant(variants, *aliases)
            config_path = resolve_path(str(spec["config_path"]))
            config = load_full_config(str(config_path))
            if config.get("runtime", {}).get("require_cuda") is not True:
                raise StageFailure(f"{canonical_name} does not require CUDA", 35)
            model = SLMFBBDM.from_config(config).cuda()
            router = getattr(model, "residual_preconditioner", None)
            if router is None:
                raise StageFailure(f"{canonical_name} lacks frequency module", 35)
            effective = getattr(router, "_effective_policy", None)
            hard_null = bool(getattr(router, "hard_all_null", False))
            if canonical_name == "no_route" and not hard_null:
                raise StageFailure("no_route is not hard-all-null", 35)
            if (
                canonical_name == "h3_v2_native_null"
                and effective != "h3_native_null"
            ):
                raise StageFailure("H3-v2 schedule policy is not active", 35)
            rows[canonical_name] = {
                "config_path": str(config_path),
                "config_sha256": sha256_file(config_path),
                "effective_policy": effective,
                "hard_all_null": hard_null,
                "parameters": int(
                    sum(parameter.numel() for parameter in model.parameters())
                ),
                "cuda_device": str(next(model.parameters()).device),
            }
            del model
            torch.cuda.empty_cache()
        path = (
            self._experiment_plan_path().parent
            / "audits"
            / "gpu_model_smoke.json"
        )
        write_json_atomic(path, rows)
        return rows, [path]

    def _validate_training_checkpoint(
        self,
        variant_spec: Mapping[str, Any],
    ) -> Path:
        import torch

        path = resolve_path(
            str(
                variant_spec.get(
                    "checkpoint_path",
                    variant_spec.get("final_checkpoint_path", ""),
                )
            )
        )
        if not path.is_file():
            raise StageFailure(f"Final checkpoint is missing: {path}", 36)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        expected_epochs = int(
            read_object(self._experiment_plan_path()).get(
                "epochs",
                self.args.epochs,
            )
        )
        if int(checkpoint.get("epoch", -1)) != expected_epochs:
            raise StageFailure(
                f"Final checkpoint epoch mismatch: {path}",
                36,
            )
        return path

    def _validate_training_stage(
        self,
        variant_spec: Mapping[str, Any],
    ) -> tuple[dict[str, Any], list[Path]]:
        checkpoint = self._validate_training_checkpoint(variant_spec)
        return {
            "variant": str(variant_spec["variant"]),
            "final_checkpoint": str(checkpoint),
            "final_checkpoint_sha256": sha256_file(checkpoint),
            "final_epoch": int(
                variant_spec.get("expected_final_epoch", self.args.epochs)
            ),
        }, [checkpoint]

    def _validate_evaluation(self, path: Path) -> dict[str, Any]:
        payload = read_object(path)
        if int(payload.get("num_patients", -1)) != 31:
            raise StageFailure(f"Evaluation does not contain 31 patients: {path}", 37)
        if int(payload.get("num_samples", -1)) != 237:
            raise StageFailure(f"Evaluation does not contain 237 samples: {path}", 37)
        per_patient = payload.get("per_patient")
        if not isinstance(per_patient, Mapping) or len(per_patient) != 31:
            raise StageFailure(f"Evaluation patient table is incomplete: {path}", 37)
        return {
            "num_patients": 31,
            "num_samples": 237,
            "patient_ids": sorted(str(value) for value in per_patient),
        }

    def _seal_evaluation_provenance(
        self,
        *,
        variant_spec: Mapping[str, Any],
        evaluation_path: Path,
        validation: Mapping[str, Any],
    ) -> Path:
        plan = self._load_plan()
        config_path = resolve_path(str(variant_spec["config_path"]))
        checkpoint_path = self._validate_training_checkpoint(variant_spec)
        provenance_path = resolve_path(
            str(variant_spec["evaluation_provenance_path"])
        )
        invariant = {
            "schema_version": 1,
            "pipeline_id": "H3_V2_MATCHED_MODEL_EVALUATION_PROVENANCE_V1",
            "variant": str(variant_spec["variant"]),
            "plan_path": str(self._experiment_plan_path()),
            "plan_sha256": str(plan["plan_sha256"]),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_epoch": int(
                variant_spec.get("expected_final_epoch", self.args.epochs)
            ),
            "eval_path": str(evaluation_path),
            "eval_sha256": sha256_file(evaluation_path),
            "weights": "ema",
            "split": "test",
            "full_split": True,
            "seed": int(
                read_object(PROTOCOL_CONFIG)["exploratory_model_experiment"][
                    "evaluation_seed"
                ]
            ),
            "mc_steps": 20,
            "num_patients": int(validation["num_patients"]),
            "num_samples": int(validation["num_samples"]),
            "patient_ids": list(validation["patient_ids"]),
            "production_activation_allowed": False,
            "h5_h6_allowed": False,
        }
        if provenance_path.exists():
            existing = read_object(provenance_path)
            declared = str(existing.get("provenance_sha256", ""))
            body = dict(existing)
            body.pop("provenance_sha256", None)
            if declared != canonical_sha256(body):
                raise StageFailure(
                    f"Evaluation provenance self-hash mismatch: {provenance_path}",
                    37,
                )
            comparable = dict(body)
            comparable.pop("created_at_utc", None)
            if comparable != invariant:
                raise StageFailure(
                    f"Existing evaluation provenance changed: {provenance_path}",
                    37,
                )
            return provenance_path
        payload = {**invariant, "created_at_utc": utc_now()}
        payload["provenance_sha256"] = canonical_sha256(payload)
        write_json_atomic(provenance_path, payload)
        return provenance_path

    def _validate_evaluation_stage(
        self,
        variant_spec: Mapping[str, Any],
        output_path: Path,
    ) -> tuple[dict[str, Any], list[Path]]:
        validation = self._validate_evaluation(output_path)
        provenance_path = self._seal_evaluation_provenance(
            variant_spec=variant_spec,
            evaluation_path=output_path,
            validation=validation,
        )
        return {
            **validation,
            "variant": str(variant_spec["variant"]),
            "evaluation_provenance": str(provenance_path),
            "evaluation_provenance_sha256": sha256_file(provenance_path),
        }, [provenance_path]

    @staticmethod
    def _validate_analysis_output(
        decision_path: Path,
    ) -> tuple[dict[str, Any], list[Path]]:
        analysis = read_object(decision_path)
        body = dict(analysis)
        declared = str(body.pop("decision_sha256", ""))
        if (
            declared != canonical_sha256(body)
            or analysis.get("decision") not in {"SUPPORT", "NO_SUPPORT"}
            or analysis.get("production_training_allowed") is not False
            or analysis.get("production_activation_allowed") is not False
            or analysis.get("h5_h6_allowed") is not False
            or analysis.get("does_not_override_05B") is not True
        ):
            raise StageFailure(
                "Experiment analysis decision is unsealed or unsafe",
                38,
            )
        return {
            "decision": analysis["decision"],
            "decision_sha256": declared,
            "production_activation_allowed": False,
            "h5_h6_allowed": False,
        }, []

    def run(self) -> int:
        self.prepare_context()
        self.start_heartbeat()
        python = sys.executable
        if self.args.resume:
            self._resume_resource_preflight()
        self.run_internal_stage(
            stage_id="00_cuda_disk_preflight",
            label="Formal CUDA kernel, GPU memory, and disk gate",
            action=self._gpu_preflight,
        )

        freeze_pre = self.artifacts_dir / "freeze_pre" / "decision.json"
        self.run_command_stage(
            stage_id="01_freeze_integrity_pre",
            label="Read-only V2 freeze integrity before H3-v2",
            command=[
                python,
                str(FREEZE_AUDIT),
                "--output",
                str(freeze_pre.relative_to(ROOT)),
            ],
            timeout_seconds=self.args.test_timeout_seconds,
            artifact_paths=[freeze_pre],
        )

        targeted_temp = (
            Path(tempfile.gettempdir())
            / "h3_v2_overnight_pytest"
            / self.run_dir.name
            / "targeted"
        )
        targeted_command = [
            python,
            "-m",
            "pytest",
            "-q",
            *[str(path) for path in TARGETED_TESTS],
            "--basetemp",
            str(targeted_temp),
        ]
        self.run_command_stage(
            stage_id="02_targeted_tests",
            label="H3-v2 runner, native-null, and admissibility tests",
            command=targeted_command,
            timeout_seconds=self.args.test_timeout_seconds,
        )

        if not self.args.skip_full_tests:
            full_temp = (
                Path(tempfile.gettempdir())
                / "h3_v2_overnight_pytest"
                / self.run_dir.name
                / "full"
            )
            self.run_command_stage(
                stage_id="03_full_tests",
                label="Complete repository test suite",
                command=[
                    python,
                    "-m",
                    "pytest",
                    "-q",
                    "tests",
                    "--basetemp",
                    str(full_temp),
                ],
                timeout_seconds=self.args.test_timeout_seconds,
            )

        calibration_dir = self.artifacts_dir / "calibration"
        calibration_command = [
            python,
            str(CALIBRATION_SCRIPT),
            "--root",
            str(ROOT),
            "--config",
            str(PROTOCOL_CONFIG),
            "--output-dir",
            str(calibration_dir),
            "--device",
            f"cuda:{self.args.device_index}",
            "--require-cuda",
            "--shard-size",
            str(self.args.shard_size),
            "--allow-scientific-fail-exit-zero",
        ]
        self.run_command_stage(
            stage_id="04_full_timestep_calibration",
            label="Direct 0..999 native/null calibration and one-shot gate",
            command=calibration_command,
            timeout_seconds=self.args.calibration_timeout_seconds,
            artifact_paths=[self._calibration_decision_path()],
            validator=self._validate_calibration_output,
        )
        calibration = read_object(self._calibration_decision_path())
        self.scientific_decision = str(calibration["decision"])
        self.update_state(calibration_decision=self.scientific_decision)

        if self.scientific_decision == "FAIL":
            self.run_internal_stage(
                stage_id="12_final_integrity",
                label="Final protected-evidence and source integrity",
                action=self._final_integrity,
            )
            self.update_state(status="COMPLETED_CALIBRATION_FAIL")
            self.write_summary(
                "COMPLETED_EXPECTED_CALIBRATION_FAIL",
                exit_code=0,
            )
            self.emit(
                "H3-v2 calibration completed with scientific FAIL; no model "
                "training was started."
            )
            return 0

        attempt_name = self._experiment_attempt_name()
        builder_command = [
            python,
            str(BUILDER_SCRIPT),
            "--root",
            str(ROOT),
            "--protocol-config",
            str(PROTOCOL_CONFIG),
            "--calibration-decision",
            str(self._calibration_decision_path()),
            "--output-dir",
            str(self._experiment_output_root()),
            "--run-id",
            self.run_dir.name,
            "--epochs",
            str(self.args.epochs),
            "--attempt",
            attempt_name,
        ]
        self.run_command_stage(
            stage_id="05_build_experiment_configs",
            label=f"Build sealed matched experiment configs ({attempt_name})",
            command=builder_command,
            timeout_seconds=self.args.test_timeout_seconds,
            artifact_paths=[self._experiment_plan_path()],
            validator=lambda: (
                {
                    "plan_sha256": self._load_plan()["plan_sha256"],
                    "attempt": self._experiment_attempt_name(),
                },
                [],
            ),
        )
        plan = self._load_plan()
        variants = self._plan_variants(plan)
        no_route = self._variant(variants, "no_route")
        h3_route = self._variant(
            variants,
            "h3_v2_native_null",
            "h3_native_null",
        )

        self.run_internal_stage(
            stage_id="06_gpu_model_smoke",
            label="Construct both exact experiment models on CUDA",
            action=self._model_smoke,
        )

        self.development_training_started = True
        self.update_state(development_training_started=True)
        for stage_id, label, variant in (
            (
                "07_train_no_route",
                "Train matched hard-all-null reference",
                no_route,
            ),
            (
                "08_train_h3_v2",
                "Train matched H3-v2 native/null candidate",
                h3_route,
            ),
        ):
            config_path = resolve_path(str(variant["config_path"]))
            self.run_command_stage(
                stage_id=stage_id,
                label=label,
                command=[python, str(TRAIN_SCRIPT), "--config", str(config_path)],
                timeout_seconds=self.args.training_timeout_seconds,
                validator=lambda variant=variant: self._validate_training_stage(
                    variant
                ),
            )

        evaluation_rows = []
        for stage_id, label, variant in (
            (
                "09_evaluate_no_route",
                "Evaluate no-route reference on exposed validation/test role",
                no_route,
            ),
            (
                "10_evaluate_h3_v2",
                "Evaluate H3-v2 candidate on exposed validation/test role",
                h3_route,
            ),
        ):
            config_path = resolve_path(str(variant["config_path"]))
            checkpoint_path = self._validate_training_checkpoint(variant)
            output_path = resolve_path(
                str(
                    variant.get(
                        "eval_output_path",
                        variant.get("evaluation_output_path", ""),
                    )
                )
            )
            command = [
                python,
                str(EVALUATE_SCRIPT),
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--weights",
                "ema",
                "--split",
                "test",
                "--device",
                f"cuda:{self.args.device_index}",
                "--seed",
                str(
                    read_object(PROTOCOL_CONFIG)[
                        "exploratory_model_experiment"
                    ]["evaluation_seed"]
                ),
                "--mc-steps",
                "20",
                "--output",
                str(output_path),
            ]
            self.run_command_stage(
                stage_id=stage_id,
                label=label,
                command=command,
                timeout_seconds=self.args.evaluation_timeout_seconds,
                artifact_paths=[output_path],
                validator=lambda variant=variant, output_path=output_path: (
                    self._validate_evaluation_stage(variant, output_path)
                ),
            )
            record = read_object(self._record_path(stage_id))
            evaluation_rows.append(dict(record["validation"]))
        if evaluation_rows[0]["patient_ids"] != evaluation_rows[1]["patient_ids"]:
            raise StageFailure("Evaluation patient sets are not paired", 37)

        analysis_dir = (
            self._experiment_plan_path().parent / "analysis"
        )
        analysis_decision = analysis_dir / "decision.json"
        analysis_patients = analysis_dir / "patient_differences.csv"
        analyzer_command = [
            python,
            str(ANALYZER_SCRIPT),
            "--root",
            str(ROOT),
            "--protocol-config",
            str(PROTOCOL_CONFIG),
            "--calibration-decision",
            str(self._calibration_decision_path()),
            "--experiment-plan",
            str(self._experiment_plan_path()),
            "--output-dir",
            str(analysis_dir),
        ]
        self.run_command_stage(
            stage_id="11_analyze_experiment",
            label="Patient-level paired H3-v2 internal support decision",
            command=analyzer_command,
            timeout_seconds=self.args.test_timeout_seconds,
            artifact_paths=[analysis_decision, analysis_patients],
            validator=lambda: self._validate_analysis_output(
                analysis_decision
            ),
        )
        analysis = read_object(analysis_decision)
        self.experiment_decision = str(
            analysis.get("decision", analysis.get("scientific_decision", ""))
        )
        self.update_state(experiment_decision=self.experiment_decision)

        self.run_internal_stage(
            stage_id="12_final_integrity",
            label="Final protected-evidence and source integrity",
            action=self._final_integrity,
        )
        self.update_state(status="COMPLETED")
        self.write_summary("COMPLETED", exit_code=0)
        self.emit(
            "H3-v2 isolated experiment completed. Production/H5/H6 remain blocked."
        )
        return 0

    def fail(self, exc: BaseException, exit_code: int) -> None:
        if self.current_process is not None:
            try:
                self._terminate_process_tree(self.current_process)
            finally:
                self.current_process = None
        error = f"{type(exc).__name__}: {exc}"
        self.emit(f"FAILED CLOSED: {error}")
        self.emit(traceback.format_exc())
        try:
            self.update_state(status="FAILED", child_pid=None)
            self.write_summary("FAILED", error=error, exit_code=exit_code)
        except Exception:
            pass

    def close(self) -> None:
        self.stop_heartbeat_thread()
        self.logger.close()


def build_plan(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    attempt = 1
    calibration_dir = run_dir / "artifacts" / "calibration"
    experiment_root = run_dir / "artifacts" / "model_experiment"
    attempt_dir = experiment_root / "attempts" / f"attempt_{attempt:02d}"
    stages = [
        "00_cuda_disk_preflight",
        "01_freeze_integrity_pre",
        "02_targeted_tests",
    ]
    if not args.skip_full_tests:
        stages.append("03_full_tests")
    stages.extend(
        [
            "04_full_timestep_calibration",
            "IF calibration FAIL: 12_final_integrity and stop successfully",
            "IF calibration PASS: 05_build_experiment_configs",
            "IF calibration PASS: 06_gpu_model_smoke",
            "IF calibration PASS: 07_train_no_route",
            "IF calibration PASS: 08_train_h3_v2",
            "IF calibration PASS: 09_evaluate_no_route",
            "IF calibration PASS: 10_evaluate_h3_v2",
            "IF calibration PASS: 11_analyze_experiment",
            "12_final_integrity",
        ]
    )
    return {
        "schema_version": 1,
        "pipeline_id": PIPELINE_ID,
        "run_dir": str(run_dir),
        "protocol_config": str(PROTOCOL_CONFIG),
        "calibration_output": str(calibration_dir),
        "experiment_attempt_dir": str(attempt_dir),
        "epochs_per_variant": args.epochs,
        "stages": stages,
        "hard_guards": {
            "cuda_required": True,
            "cpu_fallback_allowed": False,
            "calibration_fail_stops_before_training": True,
            "all_mutable_training_outputs_below_run_dir": True,
            "production_training_started": False,
            "production_activation_allowed": False,
            "h5_h6_started": False,
            "does_not_override_05B": True,
        },
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-full-tests", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=25)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--min-free-disk-gb", type=float, default=15.0)
    parser.add_argument("--min-free-gpu-gb", type=float, default=8.0)
    parser.add_argument(
        "--test-timeout-seconds",
        type=float,
        default=2 * 60 * 60,
    )
    parser.add_argument(
        "--calibration-timeout-seconds",
        type=float,
        default=24 * 60 * 60,
    )
    parser.add_argument(
        "--training-timeout-seconds",
        type=float,
        default=24 * 60 * 60,
    )
    parser.add_argument(
        "--evaluation-timeout-seconds",
        type=float,
        default=8 * 60 * 60,
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.resume and not args.run_dir:
        parser.error("--resume requires --run-dir")
    if args.resume and args.dry_run:
        parser.error("--resume cannot be combined with --dry-run")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.device_index < 0:
        parser.error("--device-index must be non-negative")
    if args.device_index != 0:
        parser.error(
            "this frozen V1 runner supports CUDA device index 0 only; select "
            "the desired physical GPU before launch with CUDA_VISIBLE_DEVICES"
        )
    if args.shard_size <= 0 or 1000 % args.shard_size:
        parser.error("--shard-size must divide 1000 exactly")
    if args.heartbeat_seconds <= 0:
        parser.error("--heartbeat-seconds must be positive")
    if args.min_free_disk_gb < 0 or args.min_free_gpu_gb < 0:
        parser.error("free-space thresholds must be non-negative")
    for name in (
        "test_timeout_seconds",
        "calibration_timeout_seconds",
        "training_timeout_seconds",
        "evaluation_timeout_seconds",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_dir = resolve_run_dir(args.run_dir, resume=bool(args.resume))
    except ValueError as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps(build_plan(args, run_dir), ensure_ascii=False, indent=2))
        return 0

    if args.resume:
        if not run_dir.is_dir():
            print(f"Resume directory does not exist: {run_dir}", file=sys.stderr)
            return 2
    else:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(
                f"Run directory already exists; use --resume: {run_dir}",
                file=sys.stderr,
            )
            return 2

    pipeline_lock = ExclusiveFileLock(PIPELINE_LOCK, run_dir, "H3-v2 pipeline")
    gpu_lock = ExclusiveFileLock(GPU_LOCK, run_dir, "development GPU")
    runner: OvernightRunner | None = None
    exit_code = 2
    try:
        pipeline_lock.acquire(resume=bool(args.resume))
        gpu_lock.acquire(resume=bool(args.resume))
        with KeepWindowsAwake() as keep_awake:
            runner = OvernightRunner(args, run_dir, keep_awake)
            try:
                exit_code = runner.run()
            except KeyboardInterrupt as exc:
                exit_code = 130
                runner.fail(exc, exit_code)
            except BaseException as exc:
                exit_code = (
                    exc.exit_code if isinstance(exc, StageFailure) else 2
                )
                runner.fail(exc, exit_code)
            finally:
                runner.close()
    except StageFailure as exc:
        print(f"FAILED CLOSED: {exc}", file=sys.stderr, flush=True)
        exit_code = exc.exit_code
    finally:
        gpu_lock.release()
        pipeline_lock.release()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
