#!/usr/bin/env python3
"""Build the deployable cloud runner zip for the perceptual-x0 ablation.

Always packages scripts/cloud_run_perceptual_x0_ablation.ps1 (re-encoded as
UTF-8 with BOM + CRLF so PowerShell 5.1 parses the Chinese comments and the
backtick line continuations correctly).  Extra files given with --add keep
their repo-relative paths, so configs/plan or fixed scripts ride along.

Prints the zip path, entries, SHA256, and the exact one-line command to paste
on the cloud worktree.

Usage (from the repo root):
  python scripts/build_cloud_runner_zip.py
  python scripts/build_cloud_runner_zip.py \
      --add scripts/pretrain_tiny_segmenter.py \
      --add configs/experiments/perceptual_x0_ablation_plan_v1.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "cloud_run_perceptual_x0_ablation.ps1"
ARTIFACTS = ROOT / "artifacts"
# The user's ToDesk file transfer can only see this parent directory; the
# actual worktree is one level deeper and not visible from the transfer pane.
CLOUD_PARENT = r"D:\ECPC-IDS-SEVEN-Work3"
CLOUD_PROJECT = r"D:\ECPC-IDS-SEVEN-Work3\pfm_simple_20260803_retry"


def _ps1_bytes(text: str) -> bytes:
    """UTF-8 BOM + CRLF so Windows PowerShell 5.1 reads UTF-8 (not the ANSI
    codepage) and backtick continuations are unambiguous."""
    text = text.replace("\r\n", "\n").replace("\n", "\r\n")
    return text.encode("utf-8-sig")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=None,
        help="zip path (default artifacts/pfm_cloud_runner_<date>.zip)",
    )
    ap.add_argument(
        "--add",
        action="append",
        default=[],
        help="extra repo-relative path to bundle (repeatable)",
    )
    ap.add_argument(
        "--console-ascii",
        action="store_true",
        help="print the cloud command in pure ASCII (no Chinese) so it survives "
        "any PowerShell console codepage on the cloud",
    )
    args = ap.parse_args(argv)

    if not RUNNER.is_file():
        raise SystemExit(f"runner not found: {RUNNER}")
    extras = []
    for extra in args.add:
        path = (ROOT / extra).resolve()
        if not path.is_file():
            raise SystemExit(f"--add path not found: {extra}")
        extras.append(path.relative_to(ROOT))

    out = (
        Path(args.out)
        if args.out
        else ARTIFACTS / f"pfm_cloud_runner_{datetime.now():%Y%m%d}.zip"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)

    entries: list[tuple[str, bytes]] = [
        (
            "scripts/cloud_run_perceptual_x0_ablation.ps1",
            _ps1_bytes(RUNNER.read_text(encoding="utf-8")),
        )
    ]
    for rel in extras:
        source = (ROOT / rel).read_bytes()
        payload = _ps1_bytes(source.decode("utf-8")) if rel.suffix.lower() == ".ps1" else source
        entries.append((rel.as_posix(), payload))

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, data in entries:
            z.writestr(rel, data)

    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"zip    : {out}")
    for rel, _ in entries:
        print(f"  entry: {rel}")
    print(f"sha256 : {sha}")

    # The ToDesk transfer pane only shows CLOUD_PARENT, so the zip is dropped
    # there.  The worktree path is hardcoded ($W) — the user is already inside
    # it in PowerShell, and the command cds there itself.  The zip is found by
    # prefix in the parent, extracted into the worktree, and the runner runs.
    cmd = (
        f"$W='{CLOUD_PROJECT}'; Set-Location $W; "
        f"$Z=@(gci -Path '{CLOUD_PARENT}' -Filter 'pfm_cloud_runner*.zip' -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1); "
        f"if(-not $Z){{throw 'no pfm_cloud_runner*.zip under {CLOUD_PARENT}'}}; "
        f"Expand-Archive -Force $Z.FullName $W; "
        f"if($?){{ powershell -ExecutionPolicy Bypass -File \"$W\\scripts\\cloud_run_perceptual_x0_ablation.ps1\" }}"
    )
    if args.console_ascii:
        ascii_project = CLOUD_PROJECT.encode("ascii", "replace").decode("ascii")
        if ascii_project != CLOUD_PROJECT:
            print()
            print("NOTE: cloud project path contains non-ASCII characters; the")
            print("ASCII command below only works if the path is truly ASCII.")
        cmd = cmd.replace(CLOUD_PROJECT, ascii_project)

    print()
    print("Cloud upload: copy this zip to the ToDesk-visible parent directory")
    print("(name unchanged):")
    print(f"  {CLOUD_PARENT}\\{out.name}")
    print()
    print("Cloud command (ONE line, paste after upload; auto-cds to the")
    print("worktree, auto-finds the pfm_cloud_runner*.zip, extracts, runs):")
    print(f"  {cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
