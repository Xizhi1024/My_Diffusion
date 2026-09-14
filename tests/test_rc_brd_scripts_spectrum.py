"""Script tests: audit_residual_spectrum.py (S0, FR-6.1; DESIGN S10 row 2).

Covers: the exact synthetic oracle (orthogonal LL2/HH1 patterns -> coverage =
hh1^2/(ll2^2+hh1^2) in EVERY stratum, psd/noise-floor/peak-range/var-ratio
closed forms); q25 frozen from TRAIN masks only ([计划] S5.3 leakage line,
outer-test lesion areas must not move it; no train masks -> ValueError);
gate verdicts >0.70 / boundary 0.60 / <0.50 linked to ll2_must_enter_r0 and
the 0.50/0.60/0.70 sensitivity ladder ([裁决] S4); seeded reproducibility of
the area-matched background sampling; identity mean and tiny
--mean-checkpoint (mean + comparison prefix formats, .t_dir scratch);
data-cache and png-root source channels; CLI smoke writing the DESIGN-named
residual_spectrum_fold{k}.json with every required field. Function-level
main(argv) only (no subprocess).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.mean_predictor import FullImagePETPredictor  # noqa: E402
from src.model.rc_brd import mean_weights_sha256  # noqa: E402


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module  # dataclasses resolves cls.__module__ eagerly
    spec.loader.exec_module(module)
    return module


AUDIT = _load_script("audit_residual_spectrum")


@pytest.fixture()
def t_dir() -> Path:
    """Repo-anchored scratch dir (pytest tmp_path is unusable in the sandbox)."""
    run_dir = REPO_ROOT / ".t_dir" / "rc_brd_spectrum" / uuid.uuid4().hex[:12]
    run_dir.mkdir(parents=True)
    yield run_dir
    shutil.rmtree(run_dir, ignore_errors=True)


def synth_spec(**over: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {"seed": 0, "image_size": 64, "n_train": 1, "n_test": 1,
                            "train_lesion_areas": [4, 9, 16, 25],
                            "test_lesion_areas": [36, 49]}
    spec.update(over)
    return spec


def run_audit(t_dir: Path, *, spec: dict[str, Any] | None = None, source: str = "synthetic",
              npz_dir: Path | None = None, data_cache: Path | None = None,
              png_root: Path | None = None, mean: Any = "synthetic-identity",
              fold: str = "0", seed: int = 0, tag: str = "run") -> tuple[int, dict[str, Any]]:
    """Drive main(argv) at directory-mode --out; returns (exit code, report)."""
    out_dir = t_dir / f"{tag}_out"
    args = ["--source", source, "--fold", fold, "--out", str(out_dir), "--seed", str(seed)]
    if source == "synthetic":
        args += ["--synthetic-spec", json.dumps(spec if spec is not None else synth_spec())]
    if npz_dir is not None:
        args += ["--npz-dir", str(npz_dir)]
    if data_cache is not None:
        args += ["--data-cache", str(data_cache)]
    if png_root is not None:
        args += ["--png-root", str(png_root)]
    if mean == "synthetic-identity":
        args += ["--mean", "synthetic-identity"]
    else:
        args += ["--mean-checkpoint", str(mean)]
    code = AUDIT.main(args)
    out_file = out_dir / f"residual_spectrum_fold{fold}.json"
    report = json.loads(out_file.read_text(encoding="utf-8")) if out_file.exists() else {}
    return code, report


def disk(size: int, center: tuple[int, int], area: int) -> np.ndarray:
    """Independent test-side disk builder (audit-free oracle)."""
    rows, cols = np.indices((size, size))
    dist = (rows - center[0]) ** 2 + (cols - center[1]) ** 2
    keep = np.argsort(dist.ravel(), kind="stable")[:area]
    out = np.zeros((size, size), dtype=np.float32)
    out[np.unravel_index(keep, dist.shape)] = 1.0
    return out


def write_npz_item(path: Path, *, split: str, areas: list[int], size: int = 64,
                   pet_value: float = 0.5,
                   centers: list[tuple[int, int]] | None = None) -> None:
    """One npz-dir item: disk lesions + constant pet (pure-LL2 coverage 0)."""
    if centers is None:  # default 13px grid; use spaced centers for big areas
        centers = [(6 + 13 * (j % 4), 6 + 13 * (j // 4)) for j in range(len(areas))]
    mask = np.zeros((size, size), dtype=np.float32)
    for center, area in zip(centers, areas):
        mask += disk(size, center, area)
    np.savez(path, ct=np.zeros((1, size, size), dtype=np.float32),
             pet=np.full((1, size, size), pet_value, dtype=np.float32),
             mask=mask[None].astype(np.float32),
             split=np.asarray(split), patient_id=np.asarray(f"p_{split}"))


# ---------------------------------------------------------------------------
# Exact synthetic oracle ([计划] S3.3 E_{b,s} / coverage)
# ---------------------------------------------------------------------------

def test_synthetic_coverage_matches_hand_computation(t_dir: Path):
    # ll2=1, hh1=2 -> coverage = 4/(1+4) = 0.8 in EVERY stratum (module docstring)
    code, report = run_audit(t_dir, spec=synth_spec(ll2_amplitude=1.0, hh1_amplitude=2.0),
                             tag="oracle")
    assert code == 0
    detail = report["coverage"]["B_detail"]
    for stratum in ("small_lesion", "non_small_lesion", "area_matched_background", "whole"):
        assert detail[stratum] == pytest.approx(0.8, abs=1e-6), stratum
    for stratum, value in report["coverage"]["with_ll2"].items():  # 含 LL2 对照 == 1
        assert value == pytest.approx(1.0, abs=1e-9), stratum
    # [审计] S3 var(pet-mean)/var(pet): identity mean + zero CT -> r0 == pet
    assert report["var_ratio"] == pytest.approx(1.0, abs=1e-6)
    # PSD口径: per-coefficient mean-square energy; LL2 grid is 16x sparser than pixels
    assert report["psd"]["LL2"] == pytest.approx(16.0 * 1.0 ** 2, rel=1e-6)
    assert report["psd"]["HH1"] == pytest.approx(4.0 * 2.0 ** 2, rel=1e-6)
    for band in ("LH2", "HL2", "HH2", "LH1", "HL1"):
        assert report["psd"][band] == pytest.approx(0.0, abs=1e-5)
    # noise floor: median per-coefficient HH1 power == 4*hh1^2
    assert report["noise_floor"]["band"] == "HH1"
    assert report["noise_floor"]["value"] == pytest.approx(16.0, rel=1e-6)
    # lesion peak residual range: r0 = ll2 +/- hh1 inside every lesion stratum
    for key in ("all_lesion", "small_lesion", "non_small_lesion"):
        peak = report["lesion_peak_residual_range"][key]
        assert peak["min"] == pytest.approx(-1.0, abs=1e-6)
        assert peak["max"] == pytest.approx(3.0, abs=1e-6)
    # per-band energy table carries every band x stratum cell
    assert set(report["coverage"]["per_band_energy"]) == {
        "LL2", "LH2", "HL2", "HH2", "LH1", "HL1", "HH1"}
    assert report["coverage"]["per_band_energy"]["LL2"]["small_lesion"] > 0.0


def test_q25_frozen_from_train_masks_only(t_dir: Path):
    # [计划] S5.3 leakage line: outer-test lesion areas must not move q25.
    train_areas = [4, 9, 16, 25]  # q25 = 7.75 (linear percentile)
    expected = float(np.percentile(train_areas, 25))
    for tag, test_areas in (("A", [400, 900]), ("B", [36, 49])):
        npz_dir = t_dir / f"npz_{tag}"
        npz_dir.mkdir()
        write_npz_item(npz_dir / "train_000.npz", split="train", areas=train_areas)
        # well-separated centers keep the two big test disks distinct components
        write_npz_item(npz_dir / "test_000.npz", split="test", areas=test_areas,
                       centers=[(10, 10), (10, 50)])
        code, report = run_audit(t_dir, source="npz-dir", npz_dir=npz_dir, tag=f"q25_{tag}")
        assert code == 0
        assert report["coverage"]["q25_train_pixels"] == pytest.approx(expected)
        # sanity: pooling the test areas into the percentile WOULD change it
        assert expected != pytest.approx(
            float(np.percentile(train_areas + test_areas, 25)))
        # strata split follows the frozen threshold in both items
        counts = report["coverage"]["stratum_pixel_counts"]
        assert counts["small_lesion"] == 4  # only the area-4 component is <= 7.75
        assert counts["non_small_lesion"] == sum(train_areas[1:]) + sum(test_areas)


def test_q25_without_train_masks_fails_closed(t_dir: Path):
    items = [AUDIT.AuditItem("t0", "p", "test", np.zeros((8, 8), np.float32),
                             np.zeros((8, 8), np.float32), np.ones((8, 8), np.float32))]
    with pytest.raises(ValueError):
        AUDIT.freeze_q25(items)


# ---------------------------------------------------------------------------
# Gate verdicts and the LL2 recommendation ([裁决] S4)
# ---------------------------------------------------------------------------

def test_gate_three_levels_and_ll2_recommendation(t_dir: Path):
    # high: coverage 0.8 -> pass at every preregistered floor
    code, high = run_audit(t_dir, spec=synth_spec(ll2_amplitude=1.0, hh1_amplitude=2.0),
                           tag="high")
    assert code == 0 and high["gate"] is True and high["ll2_must_enter_r0"] is False
    assert all(high["sensitivity"][k]["gate"] is True for k in ("0.50", "0.60", "0.70"))
    # low: coverage 0.2 -> fail-closed, LL2 must enter the R0 scope
    code, low = run_audit(t_dir, spec=synth_spec(ll2_amplitude=2.0, hh1_amplitude=1.0),
                          tag="low")
    assert code == 0 and low["gate"] is False and low["ll2_must_enter_r0"] is True
    assert all(low["sensitivity"][k]["gate"] is False for k in ("0.50", "0.60", "0.70"))
    assert all(low["sensitivity"][k]["ll2_must_enter_r0"] is True
               for k in ("0.50", "0.60", "0.70"))
    assert "LL2" in low["gate_rule"]
    # boundary: exact 0.60 mix -> the >= rule decides on the stored values
    code, edge = run_audit(t_dir, spec=synth_spec(ll2_amplitude=(2 / 3) ** 0.5,
                                                  hh1_amplitude=1.0), tag="edge")
    assert code == 0
    cov = edge["coverage"]["B_detail"]
    assert min(cov["small_lesion"], cov["whole"]) == pytest.approx(0.60, abs=1e-6)
    expect = min(cov["small_lesion"], cov["whole"]) >= 0.60
    assert edge["gate"] is expect
    assert edge["sensitivity"]["0.60"]["gate"] is edge["gate"]
    assert edge["sensitivity"]["0.60"]["coverage_min"] == pytest.approx(
        min(cov["small_lesion"], cov["whole"]))


# ---------------------------------------------------------------------------
# Seeded area-matched background sampling
# ---------------------------------------------------------------------------

def test_area_matched_background_sampling_reproducible(t_dir: Path):
    spec = synth_spec(gradient_amplitude=1.0, n_train=2, n_test=2)
    code_a, first = run_audit(t_dir, spec=spec, seed=0, tag="rep0a")
    code_b, second = run_audit(t_dir, spec=spec, seed=0, tag="rep0b")
    assert code_a == 0 and code_b == 0
    assert first == second  # same seed -> bit-identical report
    code_c, other = run_audit(t_dir, spec=spec, seed=1, tag="rep1")
    assert code_c == 0
    moved = abs(first["coverage"]["B_detail"]["area_matched_background"]
                - other["coverage"]["B_detail"]["area_matched_background"])
    assert moved > 1e-6  # a different seed moves the background ROI placement
    # area matching: every lesion component contributes exactly its own area
    total_lesion = 2 * sum([4, 9, 16, 25]) + 2 * sum([36, 49])
    counts = first["coverage"]["stratum_pixel_counts"]
    assert counts["area_matched_background"] == total_lesion


# ---------------------------------------------------------------------------
# Mean channels (identity + checkpoint; [计划] S3.1 frozen strong mean)
# ---------------------------------------------------------------------------

def test_mean_checkpoint_mean_and_comparison_formats_run(t_dir: Path):
    torch.manual_seed(0)
    model = FullImagePETPredictor(in_channels=1, base_channels=4)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    mean_ckpt = t_dir / "mean.pt"  # "mean" format: {"model": state, format_version 1}
    torch.save({"model": state, "format_version": 1}, mean_ckpt)
    code, report = run_audit(t_dir, spec=synth_spec(ct_amplitude=0.3), mean=mean_ckpt,
                             tag="ck_mean")
    assert code == 0
    assert report["mean"]["kind"] == "checkpoint"
    assert report["mean"]["format"] == "mean"
    assert report["mean"]["state_sha256"] == mean_weights_sha256(state)
    comparison = t_dir / "comparison.pt"  # comparison prefix stripping (slmf_bbdm)
    torch.save({"model": {f"generator.{key}": value for key, value in state.items()}},
               comparison)
    code, report = run_audit(t_dir, spec=synth_spec(ct_amplitude=0.3), mean=comparison,
                             tag="ck_comp")
    assert code == 0
    assert report["mean"]["format"] == "comparison"
    assert report["mean"]["state_sha256"] == mean_weights_sha256(state)
    code, report = run_audit(t_dir, spec=synth_spec(), mean="synthetic-identity",
                             tag="ck_ident")
    assert code == 0 and report["mean"]["kind"] == "synthetic-identity"
    assert report["n_items"] == {"total": 2, "train": 1, "test": 1}


# ---------------------------------------------------------------------------
# Other source channels + CLI smoke (DESIGN S10 row 2)
# ---------------------------------------------------------------------------

def test_data_cache_source_runs(t_dir: Path):
    cache = t_dir / "cache"
    cache.mkdir()
    for stem, split in (("001001", "train"), ("001002", "test")):
        # constant pet is pure LL2 -> coverage(B_detail) = 0 -> gate false, exit 0
        np.savez(cache / f"{stem}.npz", ct=np.zeros((1, 64, 64), np.float32),
                 pet=np.full((1, 64, 64), 0.5, np.float32),
                 mask=disk(64, (6, 6), 4)[None].astype(np.float32))
        (cache / f"{stem}_meta.json").write_text(
            json.dumps({"split": split, "patient_id": "001"}), encoding="utf-8")
    code, report = run_audit(t_dir, source="data-cache", data_cache=cache, tag="cache")
    assert code == 0
    assert report["n_items"] == {"total": 2, "train": 1, "test": 1}
    assert report["coverage"]["q25_train_pixels"] == pytest.approx(4.0)
    assert report["coverage"]["B_detail"]["whole"] == pytest.approx(0.0, abs=1e-9)
    assert report["gate"] is False and report["ll2_must_enter_r0"] is True


def test_png_root_source_runs(t_dir: Path):
    pytest.importorskip("PIL", reason="Pillow unavailable")
    from PIL import Image
    root = t_dir / "png"
    size = 8
    for split, centers in (("train", ((1, 1), (5, 6))), ("test", ((4, 2),))):
        sdir = root / split
        (sdir / "ct").mkdir(parents=True)
        (sdir / "pet").mkdir()
        (sdir / "label").mkdir()
        for j, (r0, c0) in enumerate(centers):
            name = f"{split}_{j}.png"
            Image.fromarray(np.full((size, size), 128, np.uint8)).save(sdir / "ct" / name)
            Image.fromarray(np.full((size, size), 200, np.uint8)).save(sdir / "pet" / name)
            label = np.zeros((size, size), np.uint8)
            label[r0, c0] = 255
            Image.fromarray(label).save(sdir / "label" / name)
    code, report = run_audit(t_dir, source="png-root", png_root=root, tag="png")
    assert code == 0
    assert report["n_items"] == {"total": 3, "train": 2, "test": 1}


def test_cli_smoke_writes_design_named_json_with_all_fields(t_dir: Path):
    code, report = run_audit(t_dir, spec=synth_spec(ll2_amplitude=2.0, hh1_amplitude=1.0),
                             fold="0", tag="smoke")
    out_file = t_dir / "smoke_out" / "residual_spectrum_fold0.json"
    assert code == 0 and out_file.is_file()  # DESIGN S10 file name
    for key in ("fold", "gate", "coverage", "sensitivity", "var_ratio", "psd",
                "noise_floor", "lesion_peak_residual_range", "gate_rule",
                "ll2_must_enter_r0", "schema_version", "seed"):
        assert key in report, key
    assert report["fold"] == "0"
    assert set(report["sensitivity"]) == {"0.50", "0.60", "0.70"}
    assert set(report["sensitivity"]["0.70"]) == {
        "floor", "gate", "ll2_must_enter_r0", "coverage_min", "empty_strata"}
    # exit semantics ([裁决] S4 / DESIGN S10): a failed gate stays exit 0
    assert report["gate"] is False and code == 0
    assert report["gate_empty_strata"] == []
