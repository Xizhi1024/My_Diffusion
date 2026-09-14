"""Reliable unattended runner for the H3 fixed-schedule inference audit.

This runner deliberately has no production/model-training stage and never
starts H5/H6.  The full test suite can exercise bounded training test fixtures.
A scientific ``FAIL`` from the H3 gate is the expected, successfully audited
outcome; infrastructure, integrity, or test failures still stop the run with a
non-zero exit code.

Recommended Windows entry point::

    powershell -NoProfile -ExecutionPolicy Bypass -File \
        scripts/run_h3_fixed_schedule_overnight.ps1

Resume an interrupted run by naming its exact run directory::

    powershell -NoProfile -ExecutionPolicy Bypass -File \
        scripts/run_h3_fixed_schedule_overnight.ps1 \
        --resume --run-dir results/.../overnight_runs/<run-id>
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
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


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ID = "H3_FIXED_SCHEDULE_OVERNIGHT_AUDIT_V1"
STAGE_ROOT = (
    REPO_ROOT
    / "results"
    / "mechanism_validation_v2"
    / "05B_h3_fixed_schedule_inference"
)
DEFAULT_RUNS_ROOT = STAGE_ROOT / "overnight_runs"
LOCK_PATH = STAGE_ROOT / ".overnight.lock"
CANONICAL_H3_DECISION = STAGE_ROOT / "decision.json"
V2_05A_DECISION = (
    REPO_ROOT
    / "results"
    / "mechanism_validation_v2"
    / "05A_inference_admissibility_audit"
    / "decision.json"
)
FREEZE_AUDIT = REPO_ROOT / "scripts" / "audit_v2_freeze_integrity.py"
H3_AUDIT = REPO_ROOT / "scripts" / "audit_h3_fixed_schedule_inference.py"
H3_CONFIG = REPO_ROOT / "configs" / "h3_fixed_schedule_inference_v1.json"
CALIBRATION_BUNDLE = (
    REPO_ROOT
    / "results"
    / "mechanism_validation_v2"
    / "04_main_integration_freeze"
    / "production_calibration_bundle.json"
)
INTEGRATION_CONTRACT = (
    REPO_ROOT
    / "results"
    / "mechanism_validation_v2"
    / "04_main_integration_freeze"
    / "resolved_integration_contract.json"
)
H4_SOURCE_CSV = (
    REPO_ROOT
    / "results"
    / "mechanism_validation"
    / "03_h4_noise_calibration"
    / "sample_band_evidence.csv"
)
H4_GENERATOR = REPO_ROOT / "scripts" / "validate_h4_noise_band_calibration.py"
WINDOWS_LAUNCHER = (
    REPO_ROOT / "scripts" / "run_h3_fixed_schedule_overnight.ps1"
)
PROTECTED_BASELINE_FILES = (
    V2_05A_DECISION,
    H3_CONFIG,
    CALIBRATION_BUNDLE,
    INTEGRATION_CONTRACT,
    H4_SOURCE_CSV,
    H4_GENERATOR,
)
OPTIONAL_PROTECTED_BASELINE_FILES = (CANONICAL_H3_DECISION,)
TARGETED_TESTS = (
    REPO_ROOT / "tests" / "test_h3_fixed_schedule_inference.py",
    REPO_ROOT / "tests" / "test_v2_inference_admissibility.py",
)
SOURCE_HYGIENE_FILES = (
    H3_CONFIG,
    FREEZE_AUDIT,
    H3_AUDIT,
    Path(__file__).resolve(),
    WINDOWS_LAUNCHER,
    *TARGETED_TESTS,
)
PROTECTED_OUTPUT_ROOTS = (
    REPO_ROOT / "results" / "mechanism_validation" / "03_h4_noise_calibration",
    REPO_ROOT
    / "results"
    / "mechanism_validation_v2"
    / "02_h4_v2_internal_exploratory_nested_cv",
    REPO_ROOT
    / "results"
    / "mechanism_validation_v2"
    / "04_main_integration_freeze",
    REPO_ROOT
    / "results"
    / "mechanism_validation_v2"
    / "05A_inference_admissibility_audit",
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


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def inspect_git_worktree() -> dict[str, Any]:
    """Inspect only repository-local Git metadata.

    A copied execution directory is allowed to have no ``.git`` entry. If the
    entry exists, however, Git must be available and must recognize this exact
    root; otherwise the check fails closed instead of silently falling back.
    """

    metadata = REPO_ROOT / ".git"
    git_executable = shutil.which("git")
    record: dict[str, Any] = {
        "git_metadata_present": metadata.exists(),
        "git_metadata_path": str(metadata),
        "git_executable": git_executable,
        "is_worktree": False,
    }
    if not metadata.exists():
        record["mode"] = "NON_GIT_EXPORT"
        record["reason"] = "repository-local .git metadata is absent"
        return record
    if git_executable is None:
        record["mode"] = "GIT_METADATA_UNVERIFIABLE"
        record["reason"] = "repository-local .git exists but git is unavailable"
        return record
    try:
        probe = subprocess.run(
            [
                git_executable,
                "-C",
                str(REPO_ROOT),
                "rev-parse",
                "--is-inside-work-tree",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except Exception as exc:
        record["mode"] = "GIT_METADATA_UNVERIFIABLE"
        record["reason"] = f"{type(exc).__name__}: {exc}"
        return record
    try:
        top_level_probe = subprocess.run(
            [
                git_executable,
                "-C",
                str(REPO_ROOT),
                "rev-parse",
                "--show-toplevel",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except Exception as exc:
        record["mode"] = "GIT_METADATA_UNVERIFIABLE"
        record["reason"] = (
            "Git worktree root probe failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return record
    observed_top_level: str | None = None
    exact_root = False
    if top_level_probe.returncode == 0 and top_level_probe.stdout.strip():
        try:
            observed = Path(top_level_probe.stdout.strip()).resolve()
            observed_top_level = str(observed)
            exact_root = os.path.normcase(str(observed)) == os.path.normcase(
                str(REPO_ROOT.resolve())
            )
        except OSError:
            exact_root = False
    record.update(
        {
            "probe_exit_code": probe.returncode,
            "probe_stdout": probe.stdout.strip(),
            "probe_stderr": probe.stderr.strip(),
            "top_level_exit_code": top_level_probe.returncode,
            "top_level_stdout": top_level_probe.stdout.strip(),
            "top_level_stderr": top_level_probe.stderr.strip(),
            "observed_top_level": observed_top_level,
            "expected_top_level": str(REPO_ROOT.resolve()),
            "exact_root": exact_root,
            "is_worktree": (
                probe.returncode == 0
                and probe.stdout.strip().lower() == "true"
                and exact_root
            ),
        }
    )
    record["mode"] = (
        "GIT_WORKTREE"
        if record["is_worktree"]
        else "GIT_METADATA_UNVERIFIABLE"
    )
    return record


def source_hygiene_identity() -> dict[str, Any]:
    git_probe = inspect_git_worktree()
    if git_probe["git_metadata_present"] and git_probe["is_worktree"]:
        return {
            "mode": "GIT_WORKTREE",
            "command": [
                str(git_probe["git_executable"]),
                "-C",
                str(REPO_ROOT),
                "diff",
                "--check",
            ],
            "git_probe": git_probe,
        }
    if not git_probe["git_metadata_present"]:
        return {
            "mode": "NON_GIT_EXPORT_SOURCE_HYGIENE_V1",
            "command": ["internal", "non_git_source_hygiene_v1"],
            "git_probe": git_probe,
        }
    return {
        "mode": "GIT_METADATA_UNVERIFIABLE_FAIL_CLOSED",
        "command": None,
        "git_probe": git_probe,
    }


def inspect_source_hygiene() -> dict[str, Any]:
    """Fail-closed static hygiene check for a non-Git execution export."""

    file_records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for path in SOURCE_HYGIENE_FILES:
        relative = (
            str(path.relative_to(REPO_ROOT))
            if is_relative_to(path, REPO_ROOT)
            else str(path)
        )
        record: dict[str, Any] = {
            "path": relative,
            "exists": path.is_file(),
        }
        if not path.is_file():
            issues.append({"path": relative, "kind": "missing_file"})
            file_records.append(record)
            continue
        try:
            raw = path.read_bytes()
        except OSError as exc:
            record["readable"] = False
            issues.append(
                {
                    "path": relative,
                    "kind": "read_error",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            file_records.append(record)
            continue
        record["readable"] = True
        record["sha256"] = hashlib.sha256(raw).hexdigest()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            record["utf8_decodable"] = False
            issues.append(
                {
                    "path": relative,
                    "kind": "utf8_decode_error",
                    "detail": str(exc),
                }
            )
            file_records.append(record)
            continue
        record["utf8_decodable"] = True
        trailing_whitespace_lines: list[int] = []
        conflict_marker_lines: list[int] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if line.rstrip(" \t") != line:
                trailing_whitespace_lines.append(line_number)
            stripped = line.strip()
            if (
                stripped.startswith("<<<<<<< ")
                or stripped == "======="
                or stripped.startswith(">>>>>>> ")
            ):
                conflict_marker_lines.append(line_number)
        final_newline = not raw or raw.endswith((b"\n", b"\r"))
        record.update(
            {
                "trailing_whitespace_lines": trailing_whitespace_lines,
                "conflict_marker_lines": conflict_marker_lines,
                "final_newline": final_newline,
            }
        )
        if trailing_whitespace_lines:
            issues.append(
                {
                    "path": relative,
                    "kind": "trailing_whitespace",
                    "lines": trailing_whitespace_lines,
                }
            )
        if conflict_marker_lines:
            issues.append(
                {
                    "path": relative,
                    "kind": "merge_conflict_marker",
                    "lines": conflict_marker_lines,
                }
            )
        if not final_newline:
            issues.append({"path": relative, "kind": "missing_final_newline"})
        file_records.append(record)
    return {
        "schema_version": 1,
        "mode": "NON_GIT_EXPORT_SOURCE_HYGIENE_V1",
        "decision": "PASS" if not issues else "FAIL",
        "checked_file_count": len(file_records),
        "files": file_records,
        "issues": issues,
        "created_at_utc": utc_now(),
    }


def resolve_run_dir(value: str | None, *, resume: bool) -> Path:
    if value:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        run_dir = candidate.resolve()
    else:
        if resume:
            raise ValueError("--resume requires the exact --run-dir")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = (DEFAULT_RUNS_ROOT / f"{stamp}-{os.getpid()}").resolve()

    if not is_relative_to(run_dir, REPO_ROOT.resolve()):
        raise ValueError(
            "--run-dir must be inside the repository because the frozen "
            "integrity auditor only supports repository-relative outputs"
        )
    for protected in PROTECTED_OUTPUT_ROOTS:
        protected = protected.resolve()
        if is_relative_to(run_dir, protected) or is_relative_to(protected, run_dir):
            raise ValueError(
                f"Run directory overlaps protected frozen output: {protected}"
            )
    if run_dir == STAGE_ROOT.resolve() or is_relative_to(
        CANONICAL_H3_DECISION.resolve(), run_dir
    ):
        raise ValueError(
            "Run directory must be a child run folder, not the canonical "
            "05B stage directory"
        )
    return run_dir


def resolve_pytest_temp_root(
    value: str | None,
    *,
    run_dir: Path,
) -> Path:
    if value:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        root = candidate.resolve()
    elif not is_relative_to(run_dir, REPO_ROOT.resolve()):
        root = (run_dir / "pytest_tmp").resolve()
    else:
        # One legacy V2-05A test deliberately distinguishes repository-relative
        # artifact paths from external absolute paths. Putting pytest tmp_path
        # below REPO_ROOT changes that test's semantics, so the unique run-id
        # directory must live outside the repository for the default layout.
        root = (
            Path(tempfile.gettempdir())
            / "h3_fixed_schedule_overnight"
            / run_dir.name
        ).resolve()
    if is_relative_to(root, REPO_ROOT.resolve()):
        raise ValueError(
            "pytest temp root must be outside the repository; a legacy "
            "fail-closed test intentionally checks external absolute paths"
        )
    if root.name != run_dir.name:
        raise ValueError(
            "pytest temp root must end with the exact run_id so pytest cannot "
            "clear a shared or ambiguously named directory"
        )
    return root


def process_identity(pid: int) -> tuple[str, int | None]:
    """Return (state, creation identity) without treating access denial as dead."""
    if pid <= 0:
        return "dead", None
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return "dead", None
        except PermissionError:
            return "unknown", None
        return "alive", None

    class FILETIME(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime", ctypes.c_uint32),
            ("dwHighDateTime", ctypes.c_uint32),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint32,
    ]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetProcessTimes.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(FILETIME),
        ctypes.POINTER(FILETIME),
        ctypes.POINTER(FILETIME),
        ctypes.POINTER(FILETIME),
    ]
    kernel32.GetProcessTimes.restype = ctypes.c_int
    kernel32.GetExitCodeProcess.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    query_limited_information = 0x1000
    synchronize = 0x00100000
    handle = kernel32.OpenProcess(
        query_limited_information | synchronize,
        False,
        int(pid),
    )
    if not handle:
        error = ctypes.get_last_error()
        # ERROR_INVALID_PARAMETER means there is no such PID. Access denied
        # and every other error are deliberately treated as unverifiable/alive.
        return ("dead", None) if error == 87 else ("unknown", None)
    try:
        creation = FILETIME()
        exit_time = FILETIME()
        kernel = FILETIME()
        user = FILETIME()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            return "unknown", None
        identity = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        exit_code = ctypes.c_uint32()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return "unknown", identity
        return ("alive" if exit_code.value == 259 else "dead"), identity
    finally:
        kernel32.CloseHandle(handle)


class StageFailure(RuntimeError):
    def __init__(self, message: str, exit_code: int = 2):
        super().__init__(message)
        self.exit_code = int(exit_code) if int(exit_code) != 0 else 2


class ExclusiveRunLock:
    """Existence lock with an ownership token.

    Another process's lock is never removed automatically.  The file is deleted
    only by the process that successfully created it and can still prove the
    token matches.
    """

    def __init__(self, path: Path, run_dir: Path):
        self.path = path
        self.run_dir = run_dir
        self.token = uuid.uuid4().hex
        self.fd: int | None = None
        self.file_identity: tuple[int, int] | None = None

    def _recover_same_run_stale_lock(self) -> bool:
        """Remove only a proven-dead lock for this exact resumed run."""
        try:
            before = self.path.stat()
            payload = load_json(self.path)
            owner_run_dir = Path(str(payload["run_dir"])).resolve()
            owner_pid = int(payload["pid"])
            owner_creation = payload.get("process_creation_time")
            owner_creation = (
                int(owner_creation) if owner_creation is not None else None
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if owner_run_dir != self.run_dir.resolve():
            return False
        state, observed_creation = process_identity(owner_pid)
        same_process = (
            state == "alive"
            and owner_creation is not None
            and observed_creation == owner_creation
        )
        if same_process or state == "unknown":
            return False
        if state == "alive" and owner_creation is None:
            return False
        # A live PID with a different creation time is PID reuse, not the lock
        # owner. Re-stat immediately before removal to avoid replacing a lock
        # that changed while it was inspected.
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
                recovered = (
                    resume
                    and attempt == 0
                    and self._recover_same_run_stale_lock()
                )
                if recovered:
                    continue
                try:
                    owner = self.path.read_text(
                        encoding="utf-8", errors="replace"
                    )
                except OSError:
                    owner = "<unreadable>"
                raise StageFailure(
                    "Another active, different-run, or unverifiable overnight "
                    "lock exists. It was not removed:\n"
                    f"  {self.path}\nOwner record:\n{owner}",
                    exit_code=11,
                ) from exc
        if self.fd is None:
            raise StageFailure("Failed to acquire overnight lock", 11)
        stat = os.fstat(self.fd)
        self.file_identity = (stat.st_dev, stat.st_ino)
        _, creation_identity = process_identity(os.getpid())
        payload = {
            "token": self.token,
            "pid": os.getpid(),
            "process_creation_time": creation_identity,
            "run_dir": str(self.run_dir),
            "created_at_utc": utc_now(),
        }
        assert self.fd is not None
        os.write(
            self.fd,
            (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
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
            before = self.path.stat()
            payload = load_json(self.path)
            after = self.path.stat()
            identity_matches = self.file_identity == (
                before.st_dev,
                before.st_ino,
            )
            unchanged_while_checked = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            if (
                identity_matches
                and unchanged_while_checked
                and payload.get("token") == self.token
            ):
                self.path.unlink()
        except (OSError, ValueError, json.JSONDecodeError):
            return


class KeepWindowsAwake:
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self) -> None:
        self.enabled = False
        self.detail = "not requested"

    def __enter__(self) -> "KeepWindowsAwake":
        if os.name != "nt":
            self.detail = "non-Windows platform; no Windows power request made"
            return self
        result = ctypes.windll.kernel32.SetThreadExecutionState(  # type: ignore[attr-defined]
            self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
        )
        if result == 0:
            raise StageFailure(
                "SetThreadExecutionState failed; refusing an unattended run "
                "that may sleep",
                exit_code=12,
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
        self.path = path
        self.handle = path.open(
            "a" if append else "w",
            encoding="utf-8",
            newline="\n",
            buffering=1,
        )
        self.lock = threading.Lock()
        self.last_output_at = utc_now()

    def emit(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        with self.lock:
            print(line, flush=True)
            self.handle.write(line + "\n")
            self.handle.flush()
            self.last_output_at = utc_now()

    def stage_line(self, stage_id: str, line: str) -> None:
        clean = line.rstrip("\r\n")
        with self.lock:
            print(f"[{stage_id}] {clean}", flush=True)
            self.handle.write(f"[{utc_now()}] [{stage_id}] {clean}\n")
            self.handle.flush()
            self.last_output_at = utc_now()

    def close(self) -> None:
        with self.lock:
            self.handle.flush()
            self.handle.close()


Validator = Callable[
    [str, Path, int, Mapping[str, Any]], tuple[dict[str, Any], list[dict[str, str]]]
]


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
        self.pytest_temp_root = resolve_pytest_temp_root(
            args.pytest_temp_root,
            run_dir=run_dir,
        )
        self.state_path = run_dir / "state.json"
        self.heartbeat_path = run_dir / "heartbeat.json"
        self.summary_path = run_dir / "summary.json"
        self.context_path = run_dir / "run_context.json"
        self.logs_dir = run_dir / "logs"
        self.stages_dir = run_dir / "stages"
        self.artifacts_dir = run_dir / "artifacts"
        self.h3_decision_path = (
            self.artifacts_dir / "h3_gate" / "decision.json"
        )
        self.logger = RunLogger(
            run_dir / "overnight.log",
            append=bool(args.resume),
        )
        self.state_lock = threading.Lock()
        self.state: dict[str, Any] = {
            "schema_version": 1,
            "pipeline_id": PIPELINE_ID,
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "pid": os.getpid(),
            "status": "INITIALIZING",
            "current_stage": None,
            "child_pid": None,
            "started_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "last_output_at_utc": self.logger.last_output_at,
            "completed_stages": [],
            "scientific_decision": None,
            "next_stage_allowed": False,
            "pytest_temp_root": str(self.pytest_temp_root),
        }
        self.context: dict[str, Any] = {}
        self.stop_heartbeat = threading.Event()
        self.heartbeat_thread: threading.Thread | None = None
        self.heartbeat_error: str | None = None
        self.current_process: subprocess.Popen[str] | None = None

    def emit(self, message: str) -> None:
        self.logger.emit(message)

    def update_state(self, **changes: Any) -> None:
        with self.state_lock:
            self.state.update(changes)
            self.state["updated_at_utc"] = utc_now()
            self.state["last_output_at_utc"] = self.logger.last_output_at
            write_json_atomic(self.state_path, self.state)

    def write_summary(
        self,
        execution_status: str,
        *,
        error: str | None = None,
        exit_code: int | None = None,
    ) -> None:
        stage_records: dict[str, Any] = {}
        if self.stages_dir.is_dir():
            for path in sorted(self.stages_dir.glob("*.json")):
                try:
                    stage_records[path.stem] = load_json(path)
                except Exception as exc:  # preserve reporting even if corrupt
                    stage_records[path.stem] = {
                        "status": "UNREADABLE",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
        payload = {
            "schema_version": 1,
            "pipeline_id": PIPELINE_ID,
            "run_id": self.run_dir.name,
            "run_dir": str(self.run_dir),
            "execution_status": execution_status,
            "scientific_decision": self.state.get("scientific_decision"),
            "next_stage_allowed": False,
            "production_training_started": False,
            "bounded_test_fixture_training_may_run": not bool(
                self.args.skip_full_tests
            ),
            "h5_h6_started": False,
            "full_tests_skipped": bool(self.args.skip_full_tests),
            "started_at_utc": self.state.get("started_at_utc"),
            "updated_at_utc": utc_now(),
            "error": error,
            "exit_code": exit_code,
            "stages": stage_records,
        }
        write_json_atomic(self.summary_path, payload)

    def start_heartbeat(self) -> None:
        def worker() -> None:
            while not self.stop_heartbeat.wait(self.args.heartbeat_seconds):
                with self.state_lock:
                    payload = {
                        "schema_version": 1,
                        "pipeline_id": PIPELINE_ID,
                        "run_id": self.run_dir.name,
                        "pid": os.getpid(),
                        "status": self.state.get("status"),
                        "current_stage": self.state.get("current_stage"),
                        "child_pid": self.state.get("child_pid"),
                        "heartbeat_at_utc": utc_now(),
                        "last_output_at_utc": self.logger.last_output_at,
                    }
                try:
                    write_json_atomic(self.heartbeat_path, payload)
                except OSError as exc:
                    self.heartbeat_error = f"{type(exc).__name__}: {exc}"
                    self.stop_heartbeat.set()
                    return

        self.heartbeat_thread = threading.Thread(
            target=worker,
            name="overnight-heartbeat",
            daemon=True,
        )
        self.heartbeat_thread.start()

    def stop_heartbeat_thread(self) -> None:
        self.stop_heartbeat.set()
        if self.heartbeat_thread is not None:
            self.heartbeat_thread.join(timeout=5)
        with self.state_lock:
            payload = {
                "schema_version": 1,
                "pipeline_id": PIPELINE_ID,
                "run_id": self.run_dir.name,
                "pid": os.getpid(),
                "status": self.state.get("status"),
                "current_stage": self.state.get("current_stage"),
                "child_pid": None,
                "heartbeat_at_utc": utc_now(),
                "last_output_at_utc": self.logger.last_output_at,
                "terminal": True,
            }
        write_json_atomic(self.heartbeat_path, payload)

    def _environment_record(self) -> dict[str, Any]:
        usage = shutil.disk_usage(self.run_dir)
        pixi = shutil.which("pixi")
        git = shutil.which("git")
        record: dict[str, Any] = {
            "created_at_utc": utc_now(),
            "repo_root": str(REPO_ROOT),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "pixi_executable": pixi,
            "git_executable": git,
            "disk": {
                "path": str(self.run_dir),
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "free_gib": usage.free / (1024**3),
            },
            "keep_awake": {
                "enabled": self.keep_awake.enabled,
                "detail": self.keep_awake.detail,
            },
        }
        if usage.free < int(self.args.min_free_disk_gb * (1024**3)):
            raise StageFailure(
                f"Only {usage.free / (1024**3):.2f} GiB free; "
                f"minimum is {self.args.min_free_disk_gb:.2f} GiB",
                exit_code=13,
            )

        cuda_probe = (
            "import json,torch; "
            "d={'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
            "'cuda_available':torch.cuda.is_available(),"
            "'device_count':torch.cuda.device_count()}; "
            "d.update({'device_name':torch.cuda.get_device_name(0),"
            "'capability':list(torch.cuda.get_device_capability(0)),"
            "'mem_get_info':list(torch.cuda.mem_get_info())} "
            "if torch.cuda.is_available() else {}); print(json.dumps(d))"
        )
        try:
            probe = subprocess.run(
                [sys.executable, "-c", cuda_probe],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
            )
            record["cuda_probe"] = {
                "exit_code": probe.returncode,
                "stdout": probe.stdout.strip(),
                "stderr": probe.stderr.strip(),
            }
            if probe.returncode == 0 and probe.stdout.strip():
                record["cuda"] = json.loads(probe.stdout.strip().splitlines()[-1])
        except Exception as exc:
            record["cuda_probe"] = {
                "error": f"{type(exc).__name__}: {exc}",
                "enforced": False,
            }
        # CUDA/GPU is intentionally recorded, never used as a gate for this audit.
        record["cuda_required"] = False
        for key, command in (
            ("pixi_version", [pixi, "--version"] if pixi else None),
            (
                "git_head",
                [git, "rev-parse", "HEAD"] if git else None,
            ),
            (
                "nvidia_smi",
                [
                    shutil.which("nvidia-smi") or "nvidia-smi",
                    "--query-gpu=index,name,driver_version,memory.total,"
                    "memory.used,memory.free,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader",
                ]
                if shutil.which("nvidia-smi")
                else None,
            ),
        ):
            if command is None:
                record[key] = {"available": False}
                continue
            probe = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
            record[key] = {
                "exit_code": probe.returncode,
                "stdout": probe.stdout.strip(),
                "stderr": probe.stderr.strip(),
                "enforced": False,
            }
        return record

    def _input_hashes(self) -> dict[str, str]:
        required = {
            REPO_ROOT / "pixi.toml",
            REPO_ROOT / "pixi.lock",
            REPO_ROOT / "pyproject.toml",
            FREEZE_AUDIT,
            H3_AUDIT,
            Path(__file__).resolve(),
            *TARGETED_TESTS,
        }
        source_specs = (
            (REPO_ROOT / "tests", {".py", ".json", ".yaml", ".yml"}),
            (REPO_ROOT / "src", {".py", ".json", ".yaml", ".yml"}),
            (REPO_ROOT / "scripts", {".py", ".ps1"}),
            (REPO_ROOT / "configs", {".json", ".yaml", ".yml"}),
        )
        for root, suffixes in source_specs:
            if not root.is_dir():
                raise StageFailure(
                    f"Required source tree is missing: {root}",
                    exit_code=14,
                )
            required.update(
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix.lower() in suffixes
                and "__pycache__" not in path.parts
            )
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise StageFailure(
                "Required overnight input is missing:\n"
                + "\n".join(f"  {path}" for path in missing),
                exit_code=14,
            )
        return {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in sorted(
                required,
                key=lambda item: item.relative_to(REPO_ROOT).as_posix(),
            )
        }

    @staticmethod
    def _protected_baseline_hashes() -> dict[str, str]:
        missing = [
            path for path in PROTECTED_BASELINE_FILES if not path.is_file()
        ]
        if missing:
            raise StageFailure(
                "Protected frozen input is missing:\n"
                + "\n".join(f"  {path}" for path in missing),
                exit_code=15,
            )
        missing_roots = [
            path for path in PROTECTED_OUTPUT_ROOTS if not path.is_dir()
        ]
        if missing_roots:
            raise StageFailure(
                "Protected frozen output root is missing:\n"
                + "\n".join(f"  {path}" for path in missing_roots),
                exit_code=15,
            )
        protected_paths = set(PROTECTED_BASELINE_FILES)
        for root in PROTECTED_OUTPUT_ROOTS:
            protected_paths.update(
                path for path in root.rglob("*") if path.is_file()
            )
        hashes = {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in sorted(
                protected_paths,
                key=lambda item: item.relative_to(REPO_ROOT).as_posix(),
            )
        }
        for path in OPTIONAL_PROTECTED_BASELINE_FILES:
            hashes[str(path.relative_to(REPO_ROOT))] = (
                sha256_file(path) if path.is_file() else "MISSING"
            )
        return hashes

    @staticmethod
    def _validate_v2_05a(payload: Mapping[str, Any]) -> None:
        expected = {
            "decision": "FAIL",
            "conflict_proven": True,
            "next_stage_allowed": False,
        }
        errors = [
            f"{key}={payload.get(key)!r}, expected {value!r}"
            for key, value in expected.items()
            if payload.get(key) != value
        ]
        if errors:
            raise StageFailure(
                "Frozen V2-05A decision is not the expected fail-closed "
                "baseline: " + "; ".join(errors),
                exit_code=15,
            )

    def prepare_context(self) -> None:
        if self.args.resume:
            self.context = load_json(self.context_path)
            if self.context.get("pipeline_id") != PIPELINE_ID:
                raise StageFailure("Resume pipeline_id mismatch", exit_code=16)
            context_identity = {
                "run_id": self.run_dir.name,
                "run_dir": str(self.run_dir),
                "repo_root": str(REPO_ROOT),
            }
            identity_errors = [
                f"{key}={self.context.get(key)!r}, expected {value!r}"
                for key, value in context_identity.items()
                if self.context.get(key) != value
            ]
            if identity_errors:
                raise StageFailure(
                    "Resume context identity mismatch: "
                    + "; ".join(identity_errors),
                    exit_code=16,
                )
            current_hygiene = source_hygiene_identity()
            current_hygiene_identity = {
                "mode": current_hygiene["mode"],
                "command": current_hygiene["command"],
            }
            if current_hygiene_identity != self.context.get(
                "source_hygiene_identity"
            ):
                raise StageFailure(
                    "Source hygiene mode changed since this run started; "
                    "refusing a mixed Git/non-Git resume",
                    exit_code=16,
                )
            original_options = self.context.get("options", {})
            if bool(original_options.get("skip_full_tests")) != bool(
                self.args.skip_full_tests
            ):
                raise StageFailure(
                    "--skip-full-tests must match the original run when resuming",
                    exit_code=16,
                )
            if str(self.pytest_temp_root) != str(
                original_options.get("pytest_temp_root")
            ):
                raise StageFailure(
                    "pytest temp root must match the original run when resuming",
                    exit_code=16,
                )
            exact_numeric_options = {
                "stage_timeout_seconds": self.args.stage_timeout_seconds,
                "heartbeat_seconds": self.args.heartbeat_seconds,
                "min_free_disk_gb": self.args.min_free_disk_gb,
            }
            option_errors = [
                f"{key}={original_options.get(key)!r}, expected {value!r}"
                for key, value in exact_numeric_options.items()
                if original_options.get(key) != value
            ]
            if option_errors:
                raise StageFailure(
                    "Resume operational options must match the original run: "
                    + "; ".join(option_errors),
                    exit_code=16,
                )
            current_hashes = self._input_hashes()
            if current_hashes != self.context.get("input_hashes"):
                raise StageFailure(
                    "Code/config/test inputs changed since this run started; "
                    "refusing mixed-version resume",
                    exit_code=16,
                )
            current_v2 = load_json(V2_05A_DECISION)
            self._validate_v2_05a(current_v2)
            current_protected = self._protected_baseline_hashes()
            if current_protected != self.context.get(
                "protected_baseline_sha256"
            ):
                raise StageFailure(
                    "One or more protected frozen inputs changed since the "
                    "run began",
                    exit_code=16,
                )
            prior_state = load_json(self.state_path)
            state_identity = {
                "pipeline_id": PIPELINE_ID,
                "run_id": self.run_dir.name,
                "run_dir": str(self.run_dir),
            }
            state_errors = [
                f"{key}={prior_state.get(key)!r}, expected {value!r}"
                for key, value in state_identity.items()
                if prior_state.get(key) != value
            ]
            if state_errors:
                raise StageFailure(
                    "Resume state identity mismatch: "
                    + "; ".join(state_errors),
                    exit_code=16,
                )
            self.state["started_at_utc"] = prior_state.get(
                "started_at_utc", self.state["started_at_utc"]
            )
            self.state["completed_stages"] = list(
                prior_state.get("completed_stages", [])
            )
            self.emit(f"Resuming exact run directory: {self.run_dir}")
            return

        v2_payload = load_json(V2_05A_DECISION)
        self._validate_v2_05a(v2_payload)
        self.context = {
            "schema_version": 1,
            "pipeline_id": PIPELINE_ID,
            "run_id": self.run_dir.name,
            "run_dir": str(self.run_dir),
            "repo_root": str(REPO_ROOT),
            "created_at_utc": utc_now(),
            "options": {
                "skip_full_tests": bool(self.args.skip_full_tests),
                "stage_timeout_seconds": self.args.stage_timeout_seconds,
                "heartbeat_seconds": self.args.heartbeat_seconds,
                "min_free_disk_gb": self.args.min_free_disk_gb,
                "pytest_temp_root": str(self.pytest_temp_root),
            },
            "input_hashes": self._input_hashes(),
            "protected_baseline_sha256": self._protected_baseline_hashes(),
            "source_hygiene_identity": {
                key: value
                for key, value in source_hygiene_identity().items()
                if key in {"mode", "command"}
            },
            "v2_05a_baseline_path": str(V2_05A_DECISION),
            "v2_05a_baseline_sha256": sha256_file(V2_05A_DECISION),
            "v2_05a_baseline_semantics": {
                "decision": v2_payload["decision"],
                "conflict_proven": v2_payload["conflict_proven"],
                "next_stage_allowed": v2_payload["next_stage_allowed"],
            },
        }
        write_json_atomic(self.context_path, self.context)

    def _record_path(self, stage_id: str) -> Path:
        return self.stages_dir / f"{stage_id}.json"

    def _artifact_records_valid(self, record: Mapping[str, Any]) -> bool:
        log_hash = record.get("log_sha256")
        if log_hash:
            log_path = Path(str(record.get("log", "")))
            if (
                not log_path.is_file()
                or not is_relative_to(
                    log_path.resolve(), self.logs_dir.resolve()
                )
                or sha256_file(log_path) != log_hash
            ):
                return False
        for artifact in record.get("artifacts", []):
            if not isinstance(artifact, Mapping):
                return False
            path = Path(str(artifact.get("path", "")))
            expected = str(artifact.get("sha256", ""))
            if (
                not path.is_file()
                or not is_relative_to(
                    path.resolve(), self.artifacts_dir.resolve()
                )
                or sha256_file(path) != expected
            ):
                return False
        return True

    def _resume_record(
        self, stage_id: str, command: Sequence[str] | None
    ) -> dict[str, Any] | None:
        if not self.args.resume:
            return None
        path = self._record_path(stage_id)
        if not path.is_file():
            return None
        record = load_json(path)
        if record.get("stage_id") != stage_id:
            raise StageFailure(
                f"Resume stage record identity mismatch for {stage_id}",
                exit_code=17,
            )
        status = record.get("status")
        allowed_skip = (
            stage_id == "04_full_tests"
            and bool(self.args.skip_full_tests)
            and status == "SKIPPED"
        )
        if status != "PASS" and not allowed_skip:
            return None
        if command is not None and list(command) != record.get("command"):
            raise StageFailure(
                f"Resume command changed for completed stage {stage_id}",
                exit_code=17,
            )
        if not self._artifact_records_valid(record):
            raise StageFailure(
                f"Resume artifacts are missing or changed for stage {stage_id}",
                exit_code=17,
            )
        self.emit(f"Resume: verified and skipped completed stage {stage_id}")
        if stage_id == "02_h3_gate":
            if not self.h3_decision_path.is_file():
                raise StageFailure(
                    "Run-specific H3 decision is missing",
                    exit_code=17,
                )
            h3_payload = load_json(self.h3_decision_path)
            if (
                h3_payload.get("decision") != "FAIL"
                or h3_payload.get("next_stage_allowed") is not False
            ):
                raise StageFailure(
                    "Run-specific H3 decision no longer has fail-closed "
                    "semantics",
                    exit_code=17,
                )
            self.state["scientific_decision"] = "FAIL"
            self.state["next_stage_allowed"] = False
        completed = list(self.state.get("completed_stages", []))
        if stage_id not in completed:
            completed.append(stage_id)
        self.update_state(completed_stages=completed)
        return record

    def run_environment_stage(self) -> None:
        stage_id = "00_environment"
        if self._resume_record(stage_id, None):
            return
        started = utc_now()
        self.update_state(status="RUNNING", current_stage=stage_id)
        record = self._environment_record()
        artifact_path = self.artifacts_dir / "environment.json"
        write_json_atomic(artifact_path, record)
        log_path = self.logs_dir / f"{stage_id}.log"
        write_bytes_atomic(
            log_path,
            (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode(
                "utf-8"
            ),
        )
        result = {
            "stage_id": stage_id,
            "label": "Environment, disk, and non-gating CUDA inventory",
            "status": "PASS",
            "command": None,
            "started_at_utc": started,
            "completed_at_utc": utc_now(),
            "exit_code": 0,
            "log": str(log_path),
            "artifacts": [
                {"path": str(artifact_path), "sha256": sha256_file(artifact_path)}
            ],
        }
        write_json_atomic(self._record_path(stage_id), result)
        completed = list(self.state.get("completed_stages", []))
        completed.append(stage_id)
        self.update_state(completed_stages=completed, current_stage=None)
        self.emit(
            "Environment recorded; CUDA availability is informational only "
            "(this runner starts no production/model-training stage)."
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
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    def _stream_process(
        self,
        stage_id: str,
        command: Sequence[str],
        log_path: Path,
        timeout_seconds: float,
    ) -> tuple[int, bool]:
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
            cwd=str(REPO_ROOT),
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
                        "Heartbeat persistence failed while a subprocess was "
                        f"running: {self.heartbeat_error}",
                        exit_code=18,
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
        return (124 if timed_out else code), timed_out

    def run_command_stage(
        self,
        stage_id: str,
        label: str,
        command: Sequence[str],
        validator: Validator | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        resumed = self._resume_record(stage_id, command)
        if resumed is not None:
            return resumed
        self.emit(f"==== {stage_id}: {label} ====")
        self.emit(subprocess.list2cmdline(list(command)))
        self.update_state(status="RUNNING", current_stage=stage_id)
        started_at = utc_now()
        started_ns = time.time_ns()
        log_path = self.logs_dir / f"{stage_id}.log"
        code = 2
        timed_out = False
        try:
            code, timed_out = self._stream_process(
                stage_id,
                command,
                log_path,
                float(self.args.stage_timeout_seconds),
            )
        except BaseException:
            if self.current_process is not None:
                try:
                    self._terminate_process_tree(self.current_process)
                finally:
                    self.current_process = None
                    try:
                        self.update_state(child_pid=None)
                    except OSError:
                        pass
            raise
        artifacts: list[dict[str, str]] = []
        validation: dict[str, Any] = {}
        error: str | None = None
        if code != 0:
            error = (
                f"{label} timed out"
                if timed_out
                else f"{label} exited with code {code}"
            )
        else:
            try:
                if validator is not None:
                    validation, artifacts = validator(
                        stage_id,
                        log_path,
                        started_ns,
                        metadata or {},
                    )
            except Exception as exc:
                code = exc.exit_code if isinstance(exc, StageFailure) else 2
                error = f"{type(exc).__name__}: {exc}"

        result = {
            "stage_id": stage_id,
            "label": label,
            "status": "PASS" if code == 0 else ("TIMEOUT" if timed_out else "FAIL"),
            "command": list(command),
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "exit_code": code,
            "timed_out": timed_out,
            "error": error,
            "log": str(log_path),
            "log_sha256": sha256_file(log_path) if log_path.is_file() else None,
            "validation": validation,
            "artifacts": artifacts,
        }
        write_json_atomic(self._record_path(stage_id), result)
        if code != 0:
            raise StageFailure(error or f"{label} failed", exit_code=code)
        completed = list(self.state.get("completed_stages", []))
        completed.append(stage_id)
        self.update_state(completed_stages=completed, current_stage=None)
        self.write_summary("RUNNING")
        return result

    @staticmethod
    def _freeze_semantic_sha(payload: Mapping[str, Any]) -> str:
        body = dict(payload)
        body.pop("created_at_utc", None)
        body.pop("audit_sha256", None)
        return canonical_sha256(body)

    def validate_freeze(
        self,
        _: str,
        __: Path,
        ___: int,
        metadata: Mapping[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        decision_path = Path(str(metadata["decision_path"]))
        payload = load_json(decision_path)
        if payload.get("decision") != "PASS":
            raise StageFailure(
                f"Freeze audit decision is {payload.get('decision')!r}", 21
            )
        if payload.get("refreeze_performed") is not False:
            raise StageFailure("Freeze audit unexpectedly reports a refreeze", 21)
        checks = payload.get("checks")
        if not isinstance(checks, list) or not checks or any(
            not isinstance(item, Mapping) or item.get("passed") is not True
            for item in checks
        ):
            raise StageFailure("One or more freeze-integrity checks did not pass", 21)
        semantic = self._freeze_semantic_sha(payload)
        expected = metadata.get("expected_semantic_sha256")
        if expected is not None and semantic != expected:
            raise StageFailure(
                "Final freeze audit differs semantically from the initial audit",
                21,
            )
        return (
            {
                "decision": "PASS",
                "semantic_sha256": semantic,
                "audit_file_sha256": sha256_file(decision_path),
            },
            [{"path": str(decision_path), "sha256": sha256_file(decision_path)}],
        )

    def validate_h3(
        self,
        _: str,
        __: Path,
        started_ns: int,
        metadata: Mapping[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        decision_path = Path(str(metadata["decision_path"]))
        if decision_path != self.h3_decision_path.resolve():
            raise StageFailure("Unexpected H3 decision output path", 22)
        if not decision_path.is_file():
            raise StageFailure(f"H3 decision was not written: {decision_path}", 22)
        stat = decision_path.stat()
        before_mtime = metadata.get("before_mtime_ns")
        if stat.st_mtime_ns + 2_000_000_000 < started_ns or (
            before_mtime is not None and stat.st_mtime_ns == before_mtime
        ):
            raise StageFailure(
                "H3 canonical decision was not refreshed by this invocation",
                22,
            )
        payload = load_json(decision_path)
        expected = {
            "pipeline_id": "H3_FIXED_SCHEDULE_INFERENCE_V1",
            "decision": "FAIL",
            "next_stage_allowed": False,
            "model_evaluation_allowed": False,
            "production_activation_allowed": False,
        }
        errors = [
            f"{key}={payload.get(key)!r}, expected {value!r}"
            for key, value in expected.items()
            if payload.get(key) != value
        ]
        if errors:
            raise StageFailure(
                "H3 gate did not produce the expected fail-closed decision: "
                + "; ".join(errors),
                22,
            )
        self._assert_v2_05a_unchanged()
        self.update_state(scientific_decision="FAIL", next_stage_allowed=False)
        return (
            {
                **expected,
                "run_specific_path": str(decision_path),
                "run_specific_sha256": sha256_file(decision_path),
            },
            [{"path": str(decision_path), "sha256": sha256_file(decision_path)}],
        )

    def _assert_v2_05a_unchanged(self) -> None:
        payload = load_json(V2_05A_DECISION)
        self._validate_v2_05a(payload)
        actual = sha256_file(V2_05A_DECISION)
        expected = str(self.context["v2_05a_baseline_sha256"])
        if actual != expected:
            raise StageFailure(
                "Frozen V2-05A decision bytes changed during the overnight run",
                23,
            )

    def validate_final_integrity(
        self,
        stage_id: str,
        log_path: Path,
        started_ns: int,
        metadata: Mapping[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        freeze_validation, artifacts = self.validate_freeze(
            stage_id, log_path, started_ns, metadata
        )
        self._assert_v2_05a_unchanged()
        h3_record = load_json(self._record_path("02_h3_gate"))
        if not self._artifact_records_valid(h3_record):
            raise StageFailure(
                "Run-specific H3 decision changed after its audited stage", 23
            )
        if self._input_hashes() != self.context.get("input_hashes"):
            raise StageFailure(
                "Code/config/test inputs changed while the overnight run was active",
                23,
            )
        protected_hashes = self._protected_baseline_hashes()
        if protected_hashes != self.context.get("protected_baseline_sha256"):
            raise StageFailure(
                "A protected config/bundle/contract/H4 source changed during "
                "the overnight run",
                23,
            )
        payload = {
            "schema_version": 1,
            "pipeline_id": PIPELINE_ID,
            "created_at_utc": utc_now(),
            "freeze_pre_semantic_sha256": metadata[
                "expected_semantic_sha256"
            ],
            "freeze_post_semantic_sha256": freeze_validation[
                "semantic_sha256"
            ],
            "v2_05a_sha256_unchanged": True,
            "v2_05a_sha256": self.context["v2_05a_baseline_sha256"],
            "v2_05a_decision": "FAIL",
            "protected_baseline_sha256": protected_hashes,
            "h3_run_specific_decision_sha256_unchanged": True,
            "h3_decision": "FAIL",
            "next_stage_allowed": False,
            "production_training_started": False,
            "bounded_test_fixture_training_may_run": not bool(
                self.args.skip_full_tests
            ),
            "h5_h6_started": False,
        }
        final_path = self.artifacts_dir / "final_integrity.json"
        write_json_atomic(final_path, payload)
        artifacts.append(
            {"path": str(final_path), "sha256": sha256_file(final_path)}
        )
        return payload, artifacts

    def mark_skipped(self, stage_id: str, label: str, reason: str) -> None:
        if self._resume_record(stage_id, None):
            return
        result = {
            "stage_id": stage_id,
            "label": label,
            "status": "SKIPPED",
            "command": None,
            "started_at_utc": utc_now(),
            "completed_at_utc": utc_now(),
            "exit_code": 0,
            "reason": reason,
            "artifacts": [],
        }
        write_json_atomic(self._record_path(stage_id), result)
        completed = list(self.state.get("completed_stages", []))
        completed.append(stage_id)
        self.update_state(completed_stages=completed)
        self.emit(f"Skipped {stage_id}: {reason}")

    def run_source_hygiene_stage(self) -> None:
        stage_id = "05_git_diff_check"
        hygiene = source_hygiene_identity()
        current_identity = {
            "mode": hygiene["mode"],
            "command": hygiene["command"],
        }
        if current_identity != self.context.get("source_hygiene_identity"):
            raise StageFailure(
                "Source hygiene mode changed during this run; refusing to "
                "switch between Git and non-Git verification",
                exit_code=24,
            )
        git_probe = hygiene["git_probe"]
        if hygiene["mode"] == "GIT_WORKTREE":
            self.run_command_stage(
                stage_id,
                "Whitespace/error check for tracked changes",
                list(hygiene["command"]),
            )
            return
        if hygiene["mode"] == "GIT_METADATA_UNVERIFIABLE_FAIL_CLOSED":
            raise StageFailure(
                "Repository-local .git metadata exists but the exact Git "
                "worktree root cannot be verified: " + str(git_probe),
                exit_code=24,
            )

        internal_command = list(hygiene["command"])
        if self._resume_record(stage_id, internal_command):
            return
        label = "Source hygiene check for a non-Git execution export"
        self.emit(f"==== {stage_id}: {label} ====")
        self.emit(
            "Repository-local .git metadata is absent; checking the frozen "
            "pipeline source files directly."
        )
        self.update_state(status="RUNNING", current_stage=stage_id)
        started_at = utc_now()
        report = inspect_source_hygiene()
        input_hashes = self.context.get("input_hashes", {})
        report["static_hygiene_scope"] = "critical_frozen_pipeline_files"
        report["input_snapshot_file_count"] = len(input_hashes)
        report["input_snapshot_sha256"] = canonical_sha256(input_hashes)
        artifact_path = self.artifacts_dir / "source_hygiene.json"
        log_path = self.logs_dir / f"{stage_id}.log"
        write_json_atomic(artifact_path, report)
        write_bytes_atomic(
            log_path,
            (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode(
                "utf-8"
            ),
        )
        passed = report["decision"] == "PASS"
        exit_code = 0 if passed else 24
        result = {
            "stage_id": stage_id,
            "label": label,
            "status": "PASS" if passed else "FAIL",
            "command": internal_command,
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "exit_code": exit_code,
            "error": (
                None
                if passed
                else "One or more source hygiene checks failed"
            ),
            "log": str(log_path),
            "log_sha256": sha256_file(log_path),
            "validation": {
                "mode": report["mode"],
                "decision": report["decision"],
                "checked_file_count": report["checked_file_count"],
                "issue_count": len(report["issues"]),
                "git_probe": git_probe,
            },
            "artifacts": [
                {
                    "path": str(artifact_path),
                    "sha256": sha256_file(artifact_path),
                }
            ],
        }
        write_json_atomic(self._record_path(stage_id), result)
        if not passed:
            raise StageFailure(
                "Non-Git source hygiene check failed: "
                + json.dumps(report["issues"], ensure_ascii=False),
                exit_code=exit_code,
            )
        completed = list(self.state.get("completed_stages", []))
        completed.append(stage_id)
        self.update_state(completed_stages=completed, current_stage=None)
        self.write_summary("RUNNING")
        self.emit(
            "Non-Git source hygiene check passed for "
            f"{report['checked_file_count']} frozen pipeline files."
        )

    def run(self) -> int:
        self.prepare_context()
        self.pytest_temp_root.mkdir(parents=True, exist_ok=True)
        self.start_heartbeat()
        self.update_state(status="RUNNING")
        self.write_summary("RUNNING")
        self.emit(f"Run directory: {self.run_dir}")
        self.emit(
            "Scientific H3 FAIL is expected. No production/model-training, "
            "H5, or H6 stage exists in this plan; the full test suite may "
            "exercise bounded training fixtures."
        )

        self.run_environment_stage()

        freeze_pre_path = (
            self.artifacts_dir / "freeze_pre" / "decision.json"
        ).resolve()
        freeze_pre = self.run_command_stage(
            "01_freeze_pre",
            "Initial frozen H4-v1/H4-v2 integrity audit",
            [
                sys.executable,
                "-u",
                str(FREEZE_AUDIT),
                "--output",
                str(freeze_pre_path),
            ],
            self.validate_freeze,
            {"decision_path": str(freeze_pre_path)},
        )
        freeze_semantic = str(
            freeze_pre["validation"]["semantic_sha256"]
        )

        before_mtime = (
            self.h3_decision_path.stat().st_mtime_ns
            if self.h3_decision_path.is_file()
            else None
        )
        self.run_command_stage(
            "02_h3_gate",
            "H3 evidence-free fixed-schedule inference audit",
            [
                sys.executable,
                "-u",
                str(H3_AUDIT),
                "--output-dir",
                str(self.h3_decision_path.parent),
                "--allow-scientific-fail-exit-zero",
            ],
            self.validate_h3,
            {
                "before_mtime_ns": before_mtime,
                "decision_path": str(self.h3_decision_path.resolve()),
            },
        )

        self.run_command_stage(
            "03_targeted_tests",
            "H3 and V2-05A fail-closed targeted tests",
            [
                sys.executable,
                "-u",
                "-m",
                "pytest",
                "-q",
                str(TARGETED_TESTS[0].relative_to(REPO_ROOT)),
                str(TARGETED_TESTS[1].relative_to(REPO_ROOT)),
                "--basetemp",
                str(self.pytest_temp_root / "targeted"),
            ],
        )

        if self.args.skip_full_tests:
            self.mark_skipped(
                "04_full_tests",
                "Full repository test suite",
                "--skip-full-tests requested",
            )
        else:
            self.run_command_stage(
                "04_full_tests",
                "Full repository test suite",
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "pytest",
                    "-q",
                    "tests",
                    "--basetemp",
                    str(self.pytest_temp_root / "full"),
                ],
            )

        self.run_source_hygiene_stage()

        freeze_post_path = (
            self.artifacts_dir / "freeze_post" / "decision.json"
        ).resolve()
        self.run_command_stage(
            "06_freeze_post",
            "Final frozen integrity and protected-hash confirmation",
            [
                sys.executable,
                "-u",
                str(FREEZE_AUDIT),
                "--output",
                str(freeze_post_path),
            ],
            self.validate_final_integrity,
            {
                "decision_path": str(freeze_post_path),
                "expected_semantic_sha256": freeze_semantic,
            },
        )

        status = (
            "COMPLETED_EXPECTED_SCIENTIFIC_FAIL_FULL_TESTS_SKIPPED"
            if self.args.skip_full_tests
            else "COMPLETED_EXPECTED_SCIENTIFIC_FAIL"
        )
        self.update_state(
            status=status,
            current_stage=None,
            child_pid=None,
            scientific_decision="FAIL",
            next_stage_allowed=False,
            completed_at_utc=utc_now(),
        )
        self.write_summary(status, exit_code=0)
        self.emit(
            "Overnight audit completed correctly: H3 decision=FAIL, "
            "next_stage_allowed=false, no production training/H5/H6 started."
        )
        return 0

    def fail(self, exc: BaseException, exit_code: int) -> None:
        error = f"{type(exc).__name__}: {exc}"
        self.update_state(
            status="FAILED",
            current_stage=None,
            child_pid=None,
            next_stage_allowed=False,
            error=error,
            completed_at_utc=utc_now(),
        )
        self.write_summary("FAILED", error=error, exit_code=exit_code)
        self.emit(f"FAILED CLOSED: {error}")
        self.emit(traceback.format_exc())

    def close(self) -> None:
        try:
            self.stop_heartbeat_thread()
        finally:
            self.logger.close()


def build_plan(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    freeze_pre = run_dir / "artifacts" / "freeze_pre" / "decision.json"
    freeze_post = run_dir / "artifacts" / "freeze_post" / "decision.json"
    h3_decision = run_dir / "artifacts" / "h3_gate" / "decision.json"
    pytest_temp_root = resolve_pytest_temp_root(
        args.pytest_temp_root,
        run_dir=run_dir,
    )
    hygiene = source_hygiene_identity()
    if hygiene["mode"] == "GIT_WORKTREE":
        source_hygiene_stage = {
            "id": "05_git_diff_check",
            "mode": "GIT_WORKTREE",
            "command": hygiene["command"],
        }
    elif hygiene["mode"] == "NON_GIT_EXPORT_SOURCE_HYGIENE_V1":
        source_hygiene_stage = {
            "id": "05_git_diff_check",
            "mode": "NON_GIT_EXPORT_SOURCE_HYGIENE_V1",
            "command": hygiene["command"],
            "checked_files": [
                str(path.relative_to(REPO_ROOT))
                for path in SOURCE_HYGIENE_FILES
            ],
        }
    else:
        source_hygiene_stage = {
            "id": "05_git_diff_check",
            "mode": "GIT_METADATA_UNVERIFIABLE_FAIL_CLOSED",
            "command": None,
            "git_probe": hygiene["git_probe"],
        }
    stages: list[dict[str, Any]] = [
        {"id": "00_environment", "kind": "internal_inventory"},
        {
            "id": "01_freeze_pre",
            "command": [
                sys.executable,
                "-u",
                str(FREEZE_AUDIT),
                "--output",
                str(freeze_pre),
            ],
        },
        {
            "id": "02_h3_gate",
            "command": [
                sys.executable,
                "-u",
                str(H3_AUDIT),
                "--output-dir",
                str(h3_decision.parent),
                "--allow-scientific-fail-exit-zero",
            ],
            "expected_decision": "FAIL",
            "next_stage_allowed": False,
        },
        {
            "id": "03_targeted_tests",
            "command": [
                sys.executable,
                "-u",
                "-m",
                "pytest",
                "-q",
                *(str(path.relative_to(REPO_ROOT)) for path in TARGETED_TESTS),
                "--basetemp",
                str(pytest_temp_root / "targeted"),
            ],
        },
        {
            "id": "04_full_tests",
            "skipped": bool(args.skip_full_tests),
            "command": None
            if args.skip_full_tests
            else [
                sys.executable,
                "-u",
                "-m",
                "pytest",
                "-q",
                "tests",
                "--basetemp",
                str(pytest_temp_root / "full"),
            ],
        },
        source_hygiene_stage,
        {
            "id": "06_freeze_post",
            "command": [
                sys.executable,
                "-u",
                str(FREEZE_AUDIT),
                "--output",
                str(freeze_post),
            ],
        },
    ]
    return {
        "pipeline_id": PIPELINE_ID,
        "dry_run": True,
        "repo_root": str(REPO_ROOT),
        "run_dir": str(run_dir),
        "resume": bool(args.resume),
        "stage_timeout_seconds": args.stage_timeout_seconds,
        "heartbeat_seconds": args.heartbeat_seconds,
        "min_free_disk_gb": args.min_free_disk_gb,
        "pytest_temp_root": str(pytest_temp_root),
        "stages": stages,
        "explicitly_forbidden": ["production_training_stage", "H5", "H6"],
        "full_tests_may_exercise_bounded_training_fixtures": not bool(
            args.skip_full_tests
        ),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Exact run directory. Relative paths resolve from the repository "
            "root. Required with --resume."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only verified completed stages in the exact --run-dir.",
    )
    parser.add_argument(
        "--skip-full-tests",
        action="store_true",
        help="Run the gates and targeted tests but explicitly skip tests/ in full.",
    )
    parser.add_argument(
        "--stage-timeout-seconds",
        type=float,
        default=6 * 60 * 60,
        help="Hard timeout for each subprocess stage; 0 disables it.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
        help="Atomic heartbeat.json update interval.",
    )
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=5.0,
        help="Fail before audits/tests when the run drive has less free space.",
    )
    parser.add_argument(
        "--pytest-temp-root",
        default=None,
        help=(
            "Unique pytest temporary root outside the repository. The default "
            "is the system temp directory plus this run_id."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact no-training plan without creating files.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.stage_timeout_seconds < 0:
        parser.error("--stage-timeout-seconds must be non-negative")
    if args.heartbeat_seconds <= 0:
        parser.error("--heartbeat-seconds must be positive")
    if args.min_free_disk_gb < 0:
        parser.error("--min-free-disk-gb must be non-negative")
    if args.resume and not args.run_dir:
        parser.error("--resume requires --run-dir")
    if args.resume and args.dry_run:
        parser.error("--resume and --dry-run cannot be combined")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_dir = resolve_run_dir(args.run_dir, resume=args.resume)
        pytest_temp_root = resolve_pytest_temp_root(
            args.pytest_temp_root,
            run_dir=run_dir,
        )
    except ValueError as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(json.dumps(build_plan(args, run_dir), ensure_ascii=False, indent=2))
        return 0

    if args.resume:
        if not run_dir.is_dir():
            print(f"Resume run directory does not exist: {run_dir}", file=sys.stderr)
            return 2
    else:
        if pytest_temp_root.exists():
            print(
                "Fresh-run pytest temp root already exists; refusing to let "
                f"pytest clear it: {pytest_temp_root}",
                file=sys.stderr,
            )
            return 2
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(
                f"Run directory already exists; use --resume explicitly: {run_dir}",
                file=sys.stderr,
            )
            return 2

    lock = ExclusiveRunLock(LOCK_PATH, run_dir)
    runner: OvernightRunner | None = None
    exit_code = 2
    try:
        lock.acquire(resume=bool(args.resume))
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
        lock.release()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
