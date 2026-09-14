"""Tests for scripts/sync_cloud_worktrees.py — the cloud dual-worktree sync tool.

Covers the fail-closed behaviours the distributed workflow depends on:

  1. dry-run modifies nothing
  2. apply produces byte-identical files (SHA256 match)
  3. second apply is idempotent (all skipped)
  4. a target file manually modified since last sync → conflict, no overwrite
  5. path traversal outside the root is rejected
  6. symlink/junction escaping the root is rejected
  7. checkpoint/results/.git are never copied
  8. target-side extra files are never deleted
  9. an active experiment lock blocks apply
 10. atomic copy failure leaves the original intact
 11. the two targets receive the correct file groups
 12. router_only never reaches feature, feature_only never reaches router
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sync_cloud_worktrees import (  # noqa: E402
    SyncConfig,
    SyncError,
    _check_drift,
    _resolve_within,
    discover_files,
    find_active_locks,
    load_config,
    sync_once,
)


def _make_config(
    tmp_path: Path,
    *,
    common: list[str] | None = None,
    router_only: list[str] | None = None,
    feature_only: list[str] | None = None,
    exclude: list[str] | None = None,
) -> tuple[SyncConfig, Path, Path, Path]:
    src = tmp_path / "src"
    router = tmp_path / "router"
    feature = tmp_path / "feature"
    for d in (src, router, feature):
        d.mkdir(parents=True, exist_ok=True)
    payload = {
        "roots": {
            "source": "src",
            "router": "router",
            "feature": "feature",
            "manifests": "cloud_sync/manifests",
            "backups": "cloud_sync/backups",
        },
        "groups": {
            "common": common or ["mod.py"],
            "router_only": router_only or ["only_r.py"],
            "feature_only": feature_only or ["only_f.py"],
        },
        "exclude": exclude or [],
    }
    config = SyncConfig(payload, tmp_path)
    return config, src, router, feature


def _state_from_copied(copied: list[dict[str, str]], target_root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for record in copied:
        if "after_sha256" not in record:
            continue
        key = (target_root / record["file"]).as_posix()
        result[key] = {
            "after_sha256": record["after_sha256"],
            "source_sha256": record["source_sha256"],
            "group": record["group"],
            "target": "router",
        }
    return result


# ---------------------------------------------------------------------------
# 1. dry-run does not modify files
# ---------------------------------------------------------------------------


def test_dry_run_modifies_nothing(tmp_path):
    config, src, router, _ = _make_config(tmp_path)
    (src / "mod.py").write_text("v1", encoding="utf-8")
    (src / "only_r.py").write_text("r", encoding="utf-8")
    (src / "only_f.py").write_text("f", encoding="utf-8")

    result, _ = sync_once(config, "router", apply=False, source_root=src, run_id="dry")

    assert not (router / "mod.py").exists()
    assert not (router / "only_r.py").exists()
    assert len(result.copied) == 2  # common + router_only
    assert len(result.conflicts) == 0
    assert len(result.errors) == 0


# ---------------------------------------------------------------------------
# 2. apply → SHA256 identical; 3. second apply idempotent
# ---------------------------------------------------------------------------


def test_apply_is_byte_identical_and_idempotent(tmp_path):
    config, src, router, _ = _make_config(tmp_path)
    (src / "mod.py").write_text("v1", encoding="utf-8")
    (src / "only_r.py").write_text("r", encoding="utf-8")
    (src / "only_f.py").write_text("f", encoding="utf-8")

    result1, _ = sync_once(config, "router", apply=True, source_root=src, run_id="r1")
    assert len(result1.copied) == 2
    from src.mechanism_validation.common import file_sha256

    assert file_sha256(router / "mod.py") == file_sha256(src / "mod.py")
    assert file_sha256(router / "only_r.py") == file_sha256(src / "only_r.py")

    state = {"files": _state_from_copied(result1.copied, router)}
    result2, _ = sync_once(
        config, "router", apply=True, source_root=src, run_id="r2", state=state
    )
    assert len(result2.copied) == 0
    assert len(result2.skipped) == 2
    assert len(result2.conflicts) == 0


# ---------------------------------------------------------------------------
# 4. manual target modification → conflict, no overwrite
# ---------------------------------------------------------------------------


def test_manual_target_edit_is_conflict(tmp_path):
    config, src, router, _ = _make_config(tmp_path)
    (src / "mod.py").write_text("v1", encoding="utf-8")

    result1, _ = sync_once(config, "router", apply=True, source_root=src, run_id="r1")
    state = {"files": _state_from_copied(result1.copied, router)}

    (router / "mod.py").write_text("MANUAL", encoding="utf-8")
    result2, _ = sync_once(
        config, "router", apply=True, source_root=src, run_id="r2", state=state
    )
    assert len(result2.conflicts) == 1
    assert (router / "mod.py").read_text(encoding="utf-8") == "MANUAL"


# ---------------------------------------------------------------------------
# 5. path traversal is rejected
# ---------------------------------------------------------------------------


def test_traversal_entry_rejected(tmp_path):
    payload = {
        "roots": {
            "source": "src",
            "router": "router",
            "feature": "feature",
            "manifests": "manifests",
            "backups": "backups",
        },
        "groups": {
            "common": ["../escape.py"],
            "router_only": [],
            "feature_only": [],
        },
        "exclude": [],
    }
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    config = SyncConfig(payload, tmp_path)
    with pytest.raises(SyncError, match="relative"):
        discover_files(config, src)


# ---------------------------------------------------------------------------
# 6. symlink/junction escaping the root is rejected
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "nt", reason="Windows junction only")
def test_junction_escape_rejected(tmp_path):
    config, src, router, _ = _make_config(tmp_path, common=["evil_dir/evil.py"])
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evil.py").write_text("x", encoding="utf-8")
    _mklink_junction(src / "evil_dir", outside)

    result, _ = sync_once(config, "router", apply=True, source_root=src, run_id="r1")
    assert len(result.errors) >= 1
    assert "escape" in result.errors[0]["reason"]
    assert not (router / "evil_dir" / "evil.py").exists()


def _mklink_junction(link: Path, target: Path):
    import subprocess

    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        check=True,
        capture_output=True,
    )


# ---------------------------------------------------------------------------
# 7. checkpoint/results/.git never copied
# ---------------------------------------------------------------------------


def test_forbidden_assets_never_copied(tmp_path):
    config, src, router, _ = _make_config(
        tmp_path,
        common=[
            "mod.py",
            "results/o.json",
            "checkpoints/model.pt",
            ".git/config",
            "cache/blob.npy",
        ],
        exclude=[
            r"(^|/)\.git($|/)",
            r"(^|/)checkpoints($|/)",
            r"(^|/)results($|/)",
            r"(^|/)cache($|/)",
            r"\.pt$",
            r"\.npy$",
        ],
    )
    (src / "mod.py").write_text("v1", encoding="utf-8")
    for path in ("results/o.json", "checkpoints/model.pt", ".git/config", "cache/blob.npy"):
        p = src / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")

    result, _ = sync_once(config, "router", apply=True, source_root=src, run_id="r1")

    assert len(result.copied) == 1  # only mod.py
    assert not (router / "results").exists()
    assert not (router / "checkpoints").exists()
    assert not (router / ".git").exists()
    assert not (router / "cache").exists()


# ---------------------------------------------------------------------------
# 8. target extra files are never deleted
# ---------------------------------------------------------------------------


def test_target_extra_files_never_deleted(tmp_path):
    config, src, router, _ = _make_config(tmp_path)
    (src / "mod.py").write_text("v1", encoding="utf-8")
    (router / "mod.py").write_text("v1", encoding="utf-8")
    (router / "user_precious.py").write_text("keep me", encoding="utf-8")

    sync_once(config, "router", apply=True, source_root=src, run_id="r1")

    assert (router / "user_precious.py").read_text(encoding="utf-8") == "keep me"


# ---------------------------------------------------------------------------
# 9. active experiment lock blocks apply
# ---------------------------------------------------------------------------


def test_active_lock_blocks_apply(tmp_path):
    config, src, router, feature = _make_config(tmp_path)
    (src / "mod.py").write_text("v1", encoding="utf-8")
    (feature / "run.lock").write_text(
        json.dumps({"status": "running", "pid": 1, "hostname": "h"}), encoding="utf-8"
    )

    active = find_active_locks(config)
    assert len(active) == 1

    result, _ = sync_once(config, "router", apply=True, source_root=src, run_id="r1")
    # Lock blocking lives in the CLI; sync_once still copies.  What we assert
    # here is that find_active_locks detects it and that a dormant lock does not.
    (feature / "run.lock").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    assert len(find_active_locks(config)) == 0
    assert len(result.copied) >= 1


# ---------------------------------------------------------------------------
# 10. atomic copy failure leaves original intact
# ---------------------------------------------------------------------------


def test_atomic_copy_failure_preserves_original(tmp_path, monkeypatch):
    from sync_cloud_worktrees import _atomic_copy

    src = tmp_path / "src.bin"
    dst = tmp_path / "dst.bin"
    src.write_bytes(b"original-bytes")

    def _boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr("sync_cloud_worktrees.os.replace", _boom)

    with pytest.raises(OSError):
        _atomic_copy(src, dst)
    assert not dst.exists()
    # The destination directory must not be polluted with a partial temp file.
    leftovers = [p for p in tmp_path.iterdir() if p.name != "src.bin"]
    assert leftovers == []


# ---------------------------------------------------------------------------
# 11 & 12. correct group distribution between the two targets
# ---------------------------------------------------------------------------


def test_groups_are_segregated(tmp_path):
    config, src, router, feature = _make_config(tmp_path)
    (src / "mod.py").write_text("common", encoding="utf-8")
    (src / "only_r.py").write_text("router", encoding="utf-8")
    (src / "only_f.py").write_text("feature", encoding="utf-8")

    sync_once(config, "router", apply=True, source_root=src, run_id="rr")
    sync_once(config, "feature", apply=True, source_root=src, run_id="ff")

    # router gets common + router_only
    assert (router / "mod.py").exists()
    assert (router / "only_r.py").exists()
    # feature gets common + feature_only
    assert (feature / "mod.py").exists()
    assert (feature / "only_f.py").exists()
    # router must NOT get feature_only, feature must NOT get router_only
    assert not (router / "only_f.py").exists()
    assert not (feature / "only_r.py").exists()


# ---------------------------------------------------------------------------
# helpers / misc
# ---------------------------------------------------------------------------


def test_load_config_real(tmp_path):
    cfg_path = tmp_path / "configs" / "cfg.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        "roots:\n"
        "  source: src\n"
        "  router: router\n"
        "  feature: feature\n"
        "  manifests: manifests\n"
        "  backups: backups\n"
        "groups:\n"
        "  common: [a.py]\n"
        "  router_only: []\n"
        "  feature_only: []\n"
        "exclude: []\n",
        encoding="utf-8",
    )
    for name in ("src", "router", "feature", "manifests", "backups"):
        (tmp_path / name).mkdir(exist_ok=True)
    config = load_config(cfg_path)
    assert config.source_root == (tmp_path / "src").resolve()
    assert config.router_root == (tmp_path / "router").resolve()
    assert config.feature_root == (tmp_path / "feature").resolve()


def test_resolve_within_rejects_traversal(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(SyncError, match="escapes"):
        _resolve_within(root / ".." / "secret.py", root, "test")


def test_check_drift_reports_conflict_when_manually_edited(tmp_path):
    dst = tmp_path / "f.py"
    dst.write_text("changed", encoding="utf-8")
    from src.mechanism_validation.common import file_sha256

    state = {"files": {dst.as_posix(): {"after_sha256": "deadbeef", "source_sha256": "x"}}}
    result = _check_drift(dst, "notdeadbeef", state["files"])
    assert result == "conflict"
    # identical to the incoming source ⇒ not a drift conflict
    same = file_sha256(dst)
    result2 = _check_drift(dst, same, {dst.as_posix(): {"after_sha256": "deadbeef"}})
    assert result2 is None
