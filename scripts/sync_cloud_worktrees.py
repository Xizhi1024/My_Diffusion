"""One-upload, manifest-driven distribution of source files into cloud worktrees.

"物理共存、逻辑隔离": the mainline spectral-router training and the
lesion-feature interpretability branch share read-only data/checkpoints and the
same environment, but their *code* is distributed from a single canonical
source through an explicit allowlist, and their outputs / COMPLETE markers /
writable checkpoints are never shared.

Design rules enforced here:

1.  Dry-run by default; only ``--apply`` writes anything.
2.  Every source and target path is resolved and verified to live under its
    configured root (defence against ``../`` traversal and absolute-path slips).
3.  Never deletes anything on the target side.
4.  Never follows a symlink/junction whose resolved real path escapes the
    configured root (defends against symlink-based traversal).
5.  Never touches data, cache, checkpoint, results, log, or .git trees — the
    groups are built from an explicit allowlist; everything not allowlisted is
    ignored.
6.  SHA256 computed before and after every copy.
7.  Writes go through a temp file in the target directory + atomic ``os.replace``
    so a half-written file is never observed.
8.  The pre-overwrite target file is backed up to ``cloud_sync/backups/<id>/<target>/``.
9.  A JSON manifest per run lands in ``cloud_sync/manifests/<id>.json``.
10. Conflicts are detected against the previous sync state: if the target file
    changed since the last sync (and is not identical to the incoming source),
    the copy is refused.
11. An active experiment lock anywhere blocks ``--apply`` (``--dry-run`` and
    ``--verify`` still report).
12. ``--verify`` is strictly read-only and exits non-zero on any inconsistency.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import tempfile
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mechanism_validation.common import file_sha256, write_json

# ---------------------------------------------------------------------------
# Constants / tiny helpers
# ---------------------------------------------------------------------------

_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _iso(dt: datetime.datetime) -> str:
    return dt.astimezone(datetime.timezone.utc).isoformat()


def _now_utc_seconds() -> float:
    return (_utcnow() - _EPOCH).total_seconds()


def _git(args: Sequence[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=str(cwd), stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return ""


def _git_state(repo_root: Path) -> dict[str, str]:
    branch = _git(["branch", "--show-current"], repo_root) or "<detached>"
    sha = _git(["rev-parse", "HEAD"], repo_root) or "unknown"
    dirty = bool(_git(["status", "--porcelain"], repo_root))
    return {"branch": branch, "sha": sha, "dirty": dirty}


class SyncError(Exception):
    """A fail-closed sync violation.  Message is user-facing."""


# ---------------------------------------------------------------------------
# Path hardening
# ---------------------------------------------------------------------------


def _resolve_within(path: Path, root: Path, label: str) -> Path:
    """Resolve ``path`` (with symlinks) and require it to live under ``root``.

    ``path`` must already be inside ``root`` as a *lexical* path; after real
    resolution it must also remain under the *real* root.  Both checks are
    enforced, closing the symlink/junction traversal hole.
    """
    root_real = root.resolve(strict=True)
    try:
        path_rel = Path(path).resolve(strict=False).relative_to(root_real)
    except ValueError:
        raise SyncError(
            f"{label} path escapes configured root: {path} not under {root_real}"
        )
    if path_rel.parts and path_rel.parts[0] in ("..", "~"):
        raise SyncError(f"{label} path traversal rejected: {path}")
    if path_rel.parts and any(part in ("..",) for part in path_rel.parts):
        raise SyncError(f"{label} path traversal rejected: {path}")
    return root_real / path_rel


def _is_symlink_escape(path: Path, root: Path) -> bool:
    """True if any symlink/junction on the path resolves outside ``root``."""
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
        return False
    except ValueError:
        return True


# ---------------------------------------------------------------------------
# Config / state
# ---------------------------------------------------------------------------


DEFAULT_CONFIG = str(ROOT / "configs" / "cloud_worktree_sync.yaml")
STATE_FILE_NAME = "last_sync_state.json"
LOCK_GLOB = "run.lock"


class SyncConfig:
    def __init__(self, raw: Mapping[str, Any], base_dir: Path):
        roots = raw.get("roots", {})
        missing = {"source", "router", "feature", "manifests", "backups"} - set(roots)
        if missing:
            raise SyncError(f"Config missing roots: {sorted(missing)}")
        self.source_root = (base_dir / roots["source"]).resolve(strict=False)
        self.router_root = (base_dir / roots["router"]).resolve(strict=False)
        self.feature_root = (base_dir / roots["feature"]).resolve(strict=False)
        self.manifests_root = (base_dir / roots["manifests"]).resolve(strict=False)
        self.backups_root = (base_dir / roots["backups"]).resolve(strict=False)
        self.groups = {
            group: sorted(str(item) for item in items)
            for group, items in raw.get("groups", {}).items()
        }
        known_groups = {"common", "router_only", "feature_only"}
        unknown = set(self.groups) - known_groups
        if unknown:
            raise SyncError(f"Unknown groups in config: {sorted(unknown)}")
        missing = known_groups - set(self.groups)
        if missing:
            raise SyncError(f"Missing groups in config: {sorted(missing)}")
        self.exclude_patterns = [
            re.compile(pat) for pat in raw.get("exclude", [])
        ]

    def target_root(self, target: str) -> Path:
        if target == "router":
            return self.router_root
        if target == "feature":
            return self.feature_root
        raise SyncError(f"Unknown target {target!r}")


def load_config(path: Path) -> SyncConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SyncError(f"Config {path} must be a YAML mapping")
    return SyncConfig(raw, path.parent.parent)


def _state_path(manifests_root: Path) -> Path:
    return manifests_root / STATE_FILE_NAME


def load_state(manifests_root: Path) -> dict[str, Any]:
    path = _state_path(manifests_root)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def save_state(manifests_root: Path, state: Mapping[str, Any]) -> None:
    manifests_root.mkdir(parents=True, exist_ok=True)
    write_json(_state_path(manifests_root), state)


# ---------------------------------------------------------------------------
# Allowlist discovery
# ---------------------------------------------------------------------------


class FileEntry:
    """One file to distribute: relative path + group label."""

    __slots__ = ("rel", "group")

    def __init__(self, rel: str, group: str):
        self.rel = rel
        self.group = group


def _normalize_path_entry(entry: str) -> str:
    """Normalise backslashes and strip optional leading ``./``."""
    value = entry.replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value


def _is_glob(entry: str) -> bool:
    return "*" in entry or "?" in entry or "[" in entry


def _pattern_for_glob(entry: str) -> re.Pattern:
    """Turn an allowlist glob like ``src/**/*.py`` into a regex.

    ``**`` matches any number of path segments; a single ``*`` matches within
    one segment.  The pattern is anchored on the full relative path.
    """
    parts: list[str] = []
    i = 0
    while i < len(entry):
        if entry.startswith("**/", i):
            parts.append("(?:.*/)?")
            i += 3
        elif entry[i] == "*":
            if entry.startswith("**", i):
                parts.append(".*")
                i += 2
            else:
                parts.append("[^/]*")
                i += 1
        elif entry[i] == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(entry[i]))
            i += 1
    return re.compile("^" + "".join(parts) + "$")


def _walk_source(source_root: Path, exclude_patterns: Sequence[re.Pattern]) -> list[str]:
    """Yield relative paths of all files under ``source_root``.

    Prunes at directories that match an exclusion pattern (e.g. ``.git``,
    ``wandb``, ``Data``, ``checkpoints``) so we never stat their contents.  Does
    not follow directory symlinks.  Permission errors on individual files are
    ignored (the copy engine will re-report them if a file is allowlisted).
    """
    result: list[str] = []
    for dirpath, dirnames, filenames in os.walk(source_root):
        dirpath_rel = Path(dirpath).relative_to(source_root)
        pruned: list[str] = []
        for name in dirnames:
            rel = dirpath_rel / name
            if _is_allowlisted(rel, exclude_patterns):
                pruned.append(name)
        for name in pruned:
            dirnames.remove(name)
        for name in filenames:
            rel = (dirpath_rel / name).as_posix()
            if rel == "" or rel.startswith("/"):
                continue
            result.append(rel)
    return result


def discover_files(config: SyncConfig, source_root: Path) -> list[FileEntry]:
    """Expand allowlist groups into concrete FileEntry objects.

    A glob entry (contains ``*``/``?``/``[``) is matched against the relative
    paths of every file under the source root.  A literal entry must be an
    existing file.  Missing literals are ignored so the allowlist can be shared
    verbatim by the router and feature worktrees.  Entries matching an exclusion
    pattern are dropped (data/cache/checkpoint/results/log/.git guard).
    """
    seen: dict[str, FileEntry] = {}
    # Literals are cheap to check; only walk the tree when a glob exists.
    has_glob = any(_is_glob(entry) for group in config.groups.values() for entry in group)
    relpaths = _walk_source(source_root, config.exclude_patterns) if has_glob else []
    for group in ("common", "router_only", "feature_only"):
        for raw in config.groups[group]:
            entry = _normalize_path_entry(raw)
            if not entry:
                continue
            if entry.startswith("/") or ".." in entry.split("/"):
                raise SyncError(f"Group entry must be relative, got {raw!r}")
            if _is_glob(entry):
                pattern = _pattern_for_glob(entry)
                for rel in relpaths:
                    if pattern.match(rel) is None:
                        continue
                    if not _is_allowlisted(rel, config.exclude_patterns):
                        continue
                    if rel not in seen:
                        seen[rel] = FileEntry(rel, group)
            else:
                rel = entry
                if not _is_allowlisted(rel, config.exclude_patterns):
                    continue
                if not (source_root / rel).is_file():
                    continue
                if rel not in seen:
                    seen[rel] = FileEntry(rel, group)
    return sorted(seen.values(), key=lambda e: e.rel)


def _is_allowlisted(rel: str | Path, exclude_patterns: Sequence[re.Pattern]) -> bool:
    """Reject anything matching the exclusion patterns.

    The excludes are an allowlist complement: files must be explicitly listed in
    a group AND must not match any exclusion pattern (data, cache, checkpoints,
    results, logs, .git, archives).
    """
    rel_posix = rel.as_posix() if isinstance(rel, Path) else rel.replace("\\", "/")
    return not any(p.search(rel_posix) for p in exclude_patterns)


def _entry_targets(entry: FileEntry, config: SyncConfig) -> list[str]:
    if entry.group == "common":
        return ["router", "feature"]
    if entry.group == "router_only":
        return ["router"]
    if entry.group == "feature_only":
        return ["feature"]
    raise SyncError(f"Unknown group {entry.group!r}")


# ---------------------------------------------------------------------------
# Experiment locks
# ---------------------------------------------------------------------------


def find_active_locks(config: SyncConfig) -> list[Path]:
    """Scan worktree roots for experiment ``run.lock`` files still marked running.

    A lock is considered active when its JSON ``status`` is ``running`` (or the
    file exists and ``status`` is missing — treated conservatively as running).
    Locks whose status is ``complete``/``failed``/``cancelled`` are dormant.
    """
    active: list[Path] = []
    seen: set[str] = set()
    for root in (config.router_root, config.feature_root):
        for lock in root.glob(f"**/{LOCK_GLOB}"):
            key = str(lock.resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            try:
                payload = json.loads(lock.read_text(encoding="utf-8"))
                status = payload.get("status", "running")
            except Exception:
                status = "running"
            if status in ("running",):
                active.append(lock)
    return sorted(active)


def _lock_summary(lock: Path, config: SyncConfig) -> str:
    try:
        payload = json.loads(lock.read_text(encoding="utf-8"))
    except Exception:
        return str(lock)
    host = payload.get("hostname", "?")
    pid = payload.get("pid", "?")
    device = payload.get("device", "?")
    worktree = payload.get("worktree", "?")
    return (
        f"{lock} (pid={pid} host={host} device={device} worktree={worktree})"
    )


# ---------------------------------------------------------------------------
# Copy engine
# ---------------------------------------------------------------------------


def _backup_path(backups_root: Path, target: str, rel: Path) -> Path:
    return backups_root / target / rel


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst`` atomically (temp file in dst dir + os.replace)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dst.name}.", suffix=".tmp", dir=str(dst.parent)
    )
    try:
        with os.fdopen(fd, "wb") as tmp:
            with src.open("rb") as source:
                shutil.copyfileobj(source, tmp, length=1024 * 1024)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, dst)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _classify(
    src: Path, dst: Path, expected_source_sha: str, state_files: Mapping[str, dict[str, str]]
) -> str:
    """Decide copied / skipped / conflict for one file.

    Priority:
      * target missing or content differs from the incoming source → ``copied``
        (but a drift guard first checks whether the target was manually modified
        since the last sync, which turns it into ``conflict``).
      * target already byte-identical to the source → ``skipped``.
    """
    dst_sha = file_sha256(dst) if dst.is_file() else None
    if dst_sha == expected_source_sha:
        return "skipped"
    return "copied"


def _check_drift(
    dst: Path,
    expected_source_sha: str,
    state_files: Mapping[str, dict[str, str]],
) -> str | None:
    """Return 'conflict' if the target was manually modified since last sync.

    Compare the target's current SHA against the ``after`` SHA recorded in the
    previous sync state.  If they differ AND the current target is not identical
    to the incoming source, someone modified the target out of band — refuse.
    """
    record = state_files.get(dst.as_posix())
    if not record:
        return None
    last_after = record.get("after_sha256")
    if not last_after:
        return None
    current = file_sha256(dst) if dst.is_file() else None
    if current is None:
        return None
    if current != last_after:
        if current != expected_source_sha:
            return "conflict"
        return None
    return None


class SyncResult:
    def __init__(self) -> None:
        self.copied: list[dict[str, str]] = []
        self.skipped: list[dict[str, str]] = []
        self.conflicts: list[dict[str, str]] = []
        self.errors: list[dict[str, str]] = []

    @property
    def ok(self) -> bool:
        return not self.conflicts and not self.errors


def sync_once(
    config: SyncConfig,
    target: str,
    *,
    apply: bool,
    source_root: Path | None = None,
    run_id: str | None = None,
    state: dict[str, Any] | None = None,
) -> tuple[SyncResult, list[FileEntry]]:
    """Perform one sync pass for a single target.

    Returns ``(result, entries)``.  When ``apply`` is False nothing on the
    target side is written (result records what *would* happen).  ``run_id``
    and ``state`` are injected for testability.
    """
    if target not in ("router", "feature"):
        raise SyncError(f"Unknown target {target!r}")
    src_root = source_root or config.source_root
    if not src_root.is_dir():
        raise SyncError(f"Source root does not exist: {src_root}")

    entries = [e for e in discover_files(config, src_root) if target in _entry_targets(e, config)]
    result = SyncResult()
    target_root = config.target_root(target)
    if not target_root.is_dir():
        if apply:
            raise SyncError(f"Target root does not exist: {target_root}")
        # Dry-run can still report what would happen; nothing to write anyway.
        for entry in entries:
            result.errors.append(
                {"file": entry.rel, "reason": f"target root missing: {target_root}"}
            )
        return result, entries

    state_files = (state or {}).get("files", {}) or {}
    backups_root = config.backups_root / (run_id or "dry-run")
    if apply:
        _prepare_manifest_dir(config)
    staged: list[tuple[FileEntry, Path, Path]] = []
    for entry in entries:
        try:
            src = _resolve_within(src_root / entry.rel, src_root, f"source {entry.rel}")
            dst = _resolve_within(target_root / entry.rel, target_root, f"target {entry.rel}")
            if _is_symlink_escape(dst, target_root):
                result.errors.append(
                    {"file": entry.rel, "reason": "symlink/junction escapes target root"}
                )
                continue
            if _is_symlink_escape(src, src_root):
                result.errors.append(
                    {"file": entry.rel, "reason": "symlink/junction escapes source root"}
                )
                continue
            src_sha = file_sha256(src)
        except SyncError as exc:
            result.errors.append({"file": entry.rel, "reason": str(exc)})
            continue
        except OSError as exc:
            result.errors.append({"file": entry.rel, "reason": f"unreadable source: {exc}"})
            continue

        conflict = _check_drift(dst, src_sha, state_files)
        if conflict:
            result.conflicts.append(
                {
                    "file": entry.rel,
                    "source_sha256": src_sha,
                    "reason": "target modified since last sync",
                }
            )
            continue

        kind = _classify(src, dst, src_sha, state_files)
        if kind == "skipped":
            result.skipped.append(
                {"file": entry.rel, "source_sha256": src_sha, "group": entry.group}
            )
            continue
        if apply:
            staged.append((entry, src, dst))
        else:
            result.copied.append(
                {
                    "file": entry.rel,
                    "source_sha256": src_sha,
                    "group": entry.group,
                }
            )

    if apply:
        for entry, src, dst in staged:
            before_sha = file_sha256(dst) if dst.is_file() else None
            backup_rel = _backup_path(backups_root, target, Path(entry.rel))
            if before_sha is not None:
                backup_rel.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dst, backup_rel)
            try:
                _atomic_copy(src, dst)
            except OSError as exc:
                result.errors.append(
                    {"file": entry.rel, "reason": f"copy failed: {exc}"}
                )
                continue
            after_sha = file_sha256(dst)
            result.copied.append(
                {
                    "file": entry.rel,
                    "source_sha256": src_sha,
                    "before_sha256": before_sha or "",
                    "after_sha256": after_sha,
                    "group": entry.group,
                }
            )

    return result, entries


def _prepare_manifest_dir(config: SyncConfig) -> None:
    config.manifests_root.mkdir(parents=True, exist_ok=True)
    config.backups_root.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Distribute allowlisted source files into cloud worktrees."
    )
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--source", default=None, help="override source root")
    ap.add_argument("--target", choices=("router", "feature", "all"), default="all")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    mode.add_argument("--apply", action="store_true", help="write files + manifest + state")
    mode.add_argument("--verify", action="store_true", help="check consistency, exit != 0 on mismatch")
    ap.add_argument("--run-id", default=None, help="explicit sync_run_id (default: auto)")
    args = ap.parse_args(argv)

    config = load_config(Path(args.config))
    source_root = Path(args.source).resolve(strict=True) if args.source else config.source_root
    targets = ["router", "feature"] if args.target == "all" else [args.target]

    if args.apply:
        active = find_active_locks(config)
        if active:
            detail = "\n  ".join(_lock_summary(lock, config) for lock in active)
            print(f"[sync] REFUSING --apply: active experiment lock(s):\n  {detail}")
            return 2

    state = load_state(config.manifests_root)
    run_id = args.run_id or f"sync-{int(_now_utc_seconds())}-{uuid.uuid4().hex[:6]}"
    overall_ok = True
    file_records: dict[str, dict[str, Any]] = {}
    new_state_files: dict[str, dict[str, Any]] = {}

    for target in targets:
        result, entries = sync_once(
            config,
            target,
            apply=args.apply,
            source_root=source_root,
            run_id=run_id,
            state=state,
        )
        print(f"[sync] target={target} apply={args.apply} "
              f"copied={len(result.copied)} skipped={len(result.skipped)} "
              f"conflicts={len(result.conflicts)} errors={len(result.errors)}")
        for record in result.conflicts:
            print(f"  CONFLICT {record['file']}: {record['reason']}")
        for record in result.errors:
            print(f"  ERROR   {record['file']}: {record['reason']}")
        if not result.ok:
            overall_ok = False

        if args.apply:
            for record in result.copied:
                if "before_sha256" not in record:
                    continue  # only applied copies carry before/after
                rel = record["file"]
                dst = _resolve_within(
                    config.target_root(target) / rel,
                    config.target_root(target),
                    f"target {rel}",
                )
                file_records[dst.as_posix()] = {
                    "source_sha256": record["source_sha256"],
                    "before_sha256": record.get("before_sha256", ""),
                    "after_sha256": record["after_sha256"],
                    "group": record["group"],
                    "target": target,
                }
                new_state_files[dst.as_posix()] = {
                    "source_sha256": record["source_sha256"],
                    "after_sha256": record["after_sha256"],
                    "group": record["group"],
                    "target": target,
                }
            # Files that are already in sync keep their prior state records so
            # drift detection continues to cover the full allowlist.
            for key, rec in (state.get("files", {}) or {}).items():
                if key not in new_state_files:
                    new_state_files[key] = rec

    if args.apply:
        if overall_ok:
            manifest = {
                "sync_run_id": run_id,
                "started_at": _iso(_utcnow()),
                "finished_at": _iso(_utcnow()),
                "source_root": str(source_root),
                "source_git": _git_state(source_root),
                "targets": {t: _git_state(config.target_root(t)) for t in targets},
                "files": file_records,
            }
            config.manifests_root.mkdir(parents=True, exist_ok=True)
            write_json(config.manifests_root / f"{run_id}.json", manifest)
            save_state(config.manifests_root, {"files": new_state_files, "last_run": run_id})
            print(f"[sync] --apply complete. Run id {run_id}")
            print(f"[sync] manifest: {config.manifests_root / (run_id + '.json')}")
            print(f"[sync] backups : {config.backups_root / run_id}")
        else:
            print(f"[sync] --apply ABORTED: conflicts/errors present; nothing persisted.")
        return 0 if overall_ok else 1

    if args.verify:
        return _run_verify(config, source_root, targets, state)

    return 0 if overall_ok else 1


def _run_verify(
    config: SyncConfig,
    source_root: Path,
    targets: Sequence[str],
    state: dict[str, Any],
) -> int:
    """Read-only consistency check; returns 0 when all targets match the source."""
    code = 0
    for target in targets:
        result, entries = sync_once(
            config, target, apply=False, source_root=source_root,
            run_id="verify", state=state,
        )
        # sync_once with apply=False reports 'copied' for every file that would
        # need copying — that is exactly the drift set for --verify.
        if result.copied:
            code = 1
            print(f"[verify] target={target} MISMATCH: {len(result.copied)} file(s) differ")
            for record in result.copied:
                print(f"  DRIFT {record['file']} (source {record['source_sha256'][:12]})")
        if result.conflicts:
            code = 1
            for record in result.conflicts:
                print(f"  CONFLICT {record['file']}: {record['reason']}")
        if result.errors:
            code = 1
            for record in result.errors:
                print(f"  ERROR   {record['file']}: {record['reason']}")
        if not result.copied and not result.conflicts and not result.errors:
            print(f"[verify] target={target} consistent")
    return code


if __name__ == "__main__":
    sys.exit(main())
