"""Script tests: scripts/compute_band_powers.py (band_snr clock P_g sealing).

Covers (DESIGN_RC_BRD_clock_v2 §2.3):
- BandPowerAccumulator: pooled per-coefficient mean equals the hand-rolled
  P_g = ΣΣc²/Σn_b (NOT the group total — the Parseval 1:3:12 coefficient
  share must not leak into P_g); fail-closed coverage/empty errors.
- load_mean_state: {"model": ...} / {"state_dict": ...} / raw state dict.
- End-to-end compute_band_powers on a minimal .t_dir cache: payload schema
  and P_g agree with an independent manual recompute (frozen mean,
  z0 = PET − mean_pet(ct), haar_forward2, float64 pooling).
- Fail-closed: missing checkpoint key / nonexistent checkpoint path /
  split with no cached samples.
- CLI main(argv): yaml config -> sealed json matching the direct call;
  missing config file -> exit 1 without writing anything.

Function-level main(argv) only (no subprocess); .t_dir scratch fixture.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.rc_brd import BAND_NAMES, band_groups, haar_forward2, mean_weights_sha256  # noqa: E402
from src.model.mean_predictor import (  # noqa: E402
    LowFrequencyPETPredictor,
    build_mean_predictor,
)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module  # dataclasses resolves cls.__module__ eagerly
    spec.loader.exec_module(module)
    return module


BP = _load_script("compute_band_powers")

GROUPS3 = {"low": ["LL2"], "mid": ["LH2", "HL2", "HH2"], "high": ["LH1", "HL1", "HH1"]}
_LEVEL2_BANDS = ("LL2", "LH2", "HL2", "HH2")


@pytest.fixture()
def t_dir() -> Path:
    """Repo-anchored scratch dir (pytest tmp_path is unusable in the sandbox)."""
    run_dir = REPO_ROOT / ".t_dir" / "rc_brd_band_powers" / uuid.uuid4().hex[:12]
    run_dir.mkdir(parents=True)
    yield run_dir
    shutil.rmtree(run_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Unit: BandPowerAccumulator
# ---------------------------------------------------------------------------

def _band_dict(batch: int, values: dict[str, float]) -> dict[str, torch.Tensor]:
    """Band tensors with the true haar_forward2 grid geometry of an 8x8 image:
    level-2 bands on a [2,2] grid, level-1 detail bands on a [4,4] grid."""
    out: dict[str, torch.Tensor] = {}
    for name in BAND_NAMES:
        grid = 2 if name in _LEVEL2_BANDS else 4
        out[name] = torch.full((batch, 1, grid, grid), float(values[name]),
                               dtype=torch.float32)
    return out


def _band_numel(band: str, batch: int) -> int:
    """Coefficients of one band for one update (level-2: 4, level-1: 16)."""
    return batch * (4 if band in _LEVEL2_BANDS else 16)


def test_accumulator_pooled_mean_matches_hand_computation():
    acc = BP.BandPowerAccumulator(GROUPS3)
    # Two updates with different batch sizes and different per-band amplitudes.
    v1 = {"LL2": 4.0, "LH2": 0.5, "HL2": 0.5, "HH2": 0.5,
          "LH1": 1.0, "HL1": 2.0, "HH1": -1.5}
    v2 = {"LL2": 1.0, "LH2": 2.0, "HL2": 2.0, "HH2": 2.0,
          "LH1": 3.0, "HL1": 0.5, "HH1": 2.0}
    acc.update(_band_dict(2, v1))
    acc.update(_band_dict(1, v2))
    tables = [(2, v1), (1, v2)]

    def expected_group_power(bands: list[str]) -> float:
        num = den = 0.0
        for band in bands:
            for batch, values in tables:
                num += _band_numel(band, batch) * values[band] ** 2
                den += _band_numel(band, batch)
        return num / den

    powers = acc.group_powers()
    assert set(powers) == {"low", "mid", "high"}
    for group, bands in GROUPS3.items():
        assert powers[group] == pytest.approx(expected_group_power(bands), rel=1e-10)
    # Per-band diagnostic = plain per-band mean square.
    per_band = acc.band_powers()
    for band, value in per_band.items():
        hand = sum(_band_numel(band, b) * v[band] ** 2 for b, v in tables) / \
            sum(_band_numel(band, b) for b, _ in tables)
        assert value == pytest.approx(hand, rel=1e-10)


def test_accumulator_parseval_per_coefficient_not_group_total():
    # Grids straight from the real transform on an 8x8 residual.
    ref = haar_forward2(torch.zeros(1, 1, 8, 8))
    assert tuple(ref["LL2"].shape) == (1, 1, 2, 2)
    assert tuple(ref["LH1"].shape) == (1, 1, 4, 4)

    # Every band: alternating +/-1 coefficients -> per-coefficient variance 1
    # in every band regardless of grid size (same per-coefficient power).
    bands: dict[str, torch.Tensor] = {}
    for name, shape in ((n, ref[n].shape) for n in BAND_NAMES):
        sign = torch.ones(shape)
        sign.view(-1)[1::2] = -1.0
        bands[name] = sign.contiguous()

    acc = BP.BandPowerAccumulator(GROUPS3)
    acc.update(bands)
    powers = acc.group_powers()
    # Per-coefficient semantics: equal across groups despite unequal sizes.
    assert powers["low"] == pytest.approx(powers["mid"], rel=1e-12)
    assert powers["low"] == pytest.approx(powers["high"], rel=1e-12)
    assert powers["low"] == pytest.approx(1.0, rel=1e-12)
    # ... while group TOTALS stay in the Parseval 1:3:12 ratio (n=4/12/48).
    totals = {g: sum(float(bands[b].double().square().sum().item()) for b in bs)
              for g, bs in GROUPS3.items()}
    assert totals == {"low": 4.0, "mid": 12.0, "high": 48.0}
    assert totals["mid"] == pytest.approx(3.0 * totals["low"])
    assert totals["high"] == pytest.approx(12.0 * totals["low"])
    assert len(set(totals.values())) == 3  # totals are pairwise distinct
    # Guard the actual regression: P_g must never equal the group total.
    for group in GROUPS3:
        assert powers[group] != totals[group]


def test_accumulator_rejects_bad_group_coverage():
    with pytest.raises(ValueError):  # band missing entirely
        BP.BandPowerAccumulator({"low": ["LL2"], "mid": ["LH2", "HL2", "HH2"]})
    with pytest.raises(ValueError):  # no groups at all
        BP.BandPowerAccumulator({})
    with pytest.raises(ValueError):  # band covered twice
        BP.BandPowerAccumulator({"low": ["LL2", "LL2"], "mid": ["LH2", "HL2", "HH2"],
                                 "high": ["LH1", "HL1", "HH1"]})
    with pytest.raises(ValueError):  # unknown band name
        BP.BandPowerAccumulator({"low": ["LL2"], "mid": ["LH2", "HL2", "HH2"],
                                 "high": ["LH1", "HL1", "XX9"]})
    # Exact canonical layouts are accepted.
    BP.BandPowerAccumulator(band_groups(3))
    BP.BandPowerAccumulator(band_groups(7))


def test_accumulator_update_requires_every_band():
    acc = BP.BandPowerAccumulator(GROUPS3)
    full = _band_dict(1, {b: 1.0 for b in BAND_NAMES})
    victim = dict(full)
    del victim["HH2"]
    with pytest.raises(ValueError):
        acc.update(victim)
    with pytest.raises(ValueError):  # rejected update left nothing accumulated
        acc.group_powers()
    acc.update(full)  # the accumulator still works afterwards
    assert set(acc.group_powers()) == {"low", "mid", "high"}


def test_accumulator_group_powers_empty_fails():
    acc = BP.BandPowerAccumulator(GROUPS3)
    with pytest.raises(ValueError):
        acc.group_powers()


def test_accumulator_seven_groups_match_band_powers():
    values = {"LL2": 2.0, "LH2": 1.0, "HL2": 0.5, "HH2": 3.0,
              "LH1": 1.5, "HL1": 0.25, "HH1": 4.0}
    acc = BP.BandPowerAccumulator(band_groups(7))
    acc.update(_band_dict(3, values))
    powers = acc.group_powers()
    assert set(powers) == set(BAND_NAMES)
    for band, value in values.items():
        assert powers[band] == pytest.approx(value ** 2, rel=1e-10)
        assert acc.band_powers()[band] == pytest.approx(value ** 2, rel=1e-10)


def test_load_mean_state_accepts_model_state_dict_and_raw(t_dir):
    state = {"w": torch.arange(6, dtype=torch.float32).reshape(2, 3),
             "b": torch.ones(1, dtype=torch.float32)}
    wrapped_model = t_dir / "wrapped_model.pt"
    torch.save({"format_version": 1, "model": state}, wrapped_model)
    wrapped_sd = t_dir / "wrapped_sd.pt"
    torch.save({"state_dict": state}, wrapped_sd)
    raw = t_dir / "raw.pt"
    torch.save(state, raw)
    for path in (wrapped_model, wrapped_sd, raw):
        loaded = BP.load_mean_state(path)
        assert set(loaded) == {"w", "b"}
        assert torch.allclose(loaded["w"], state["w"])
        assert torch.allclose(loaded["b"], state["b"])


# ---------------------------------------------------------------------------
# End-to-end: minimal cache in .t_dir
# ---------------------------------------------------------------------------

@dataclass
class MiniCache:
    config: dict[str, Any]
    mean_cfg: dict[str, Any]
    state: dict[str, torch.Tensor]
    cache_dir: Path
    manifest_csv: Path
    ckpt: Path


def _build_mini_cache(base: Path) -> MiniCache:
    cache = base / "cache"
    cache.mkdir()
    rng = np.random.default_rng(11)
    sample_rows = [("001000", "001", "0"), ("001001", "001", "1"),
                   ("002000", "002", "0"), ("002001", "002", "1")]
    for sid, _pid, _slc in sample_rows:
        ct = rng.uniform(-1.0, 1.0, size=(1, 64, 64)).astype(np.float32)
        pet = (rng.normal(0.3, 1.0, size=(1, 64, 64))).astype(np.float32)
        np.savez_compressed(cache / f"{sid}.npz", ct=ct, pet=pet)
    manifest = base / "split_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["sample_id", "patient_id", "slice_id", "split", "cache_path"])
        writer.writeheader()
        for sid, pid, slc in sample_rows:
            writer.writerow({"sample_id": sid, "patient_id": pid, "slice_id": slc,
                             "split": "train",
                             "cache_path": str((cache / f"{sid}.npz").resolve())})
    torch.manual_seed(5)
    model = LowFrequencyPETPredictor(base_channels=8, levels=2)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ckpt = base / "mean_best.pt"
    torch.save({"format_version": 1, "model": state}, ckpt)
    mean_cfg = {"enabled": True, "architecture": "low_frequency", "base_channels": 8,
                "checkpoint": str(ckpt), "freeze": True}
    config = {"data": {"cache_dir": str(cache), "split_manifest": str(manifest),
                       "image_size": 64},
              "modules": {"conditional_mean": mean_cfg}}
    return MiniCache(config, mean_cfg, state, cache, manifest, ckpt)


@pytest.fixture()
def mini(t_dir: Path) -> MiniCache:
    return _build_mini_cache(t_dir)


def _reference_powers(mini: MiniCache, n_groups: int = 3):
    """Independent manual recompute: frozen mean, residual, haar, float64 pooling."""
    predictor = build_mean_predictor(mini.mean_cfg)
    predictor.load_state_dict(mini.state)
    predictor.eval()
    cts, pets = [], []
    for npz_path in sorted(mini.cache_dir.glob("*.npz")):  # CachedDataset order
        with np.load(npz_path) as data:
            cts.append(torch.from_numpy(data["ct"].copy()))
            pets.append(torch.from_numpy(data["pet"].copy()))
    ct = torch.stack(cts)
    pet = torch.stack(pets)
    with torch.no_grad():
        mean_pet = predictor(ct)["mean_pet"]
    bands = haar_forward2(pet - mean_pet)
    sum_sq = {b: float(bands[b].to(dtype=torch.float64).square().sum().item())
              for b in BAND_NAMES}
    count = {b: int(bands[b].numel()) for b in BAND_NAMES}
    groups = band_groups(n_groups)
    per_group = {g: sum(sum_sq[b] for b in bs) / sum(count[b] for b in bs)
                 for g, bs in groups.items()}
    per_band = {b: sum_sq[b] / count[b] for b in BAND_NAMES}
    return per_group, per_band


def test_compute_band_powers_end_to_end_matches_manual_recompute(mini):
    expected, by_band = _reference_powers(mini)
    payload = BP.compute_band_powers(mini.config, "fold_0")
    assert payload["schema_version"] == 2
    assert payload["stage"] == "rc_brd_band_powers"
    assert payload["fold"] == "fold_0"
    assert payload["split"] == "train"
    assert payload["split_label"] == "outer_train_fold_0"
    assert payload["n_samples"] == 4
    assert payload["n_patients"] == 2
    assert payload["image_size"] == 64
    assert payload["band_groups"] == GROUPS3
    assert set(payload["band_powers"]) == {"low", "mid", "high"}
    for group, value in expected.items():
        assert payload["band_powers"][group] == pytest.approx(value, rel=1e-5)
    assert set(payload["band_powers_by_band"]) == set(BAND_NAMES)
    for band, value in by_band.items():
        assert payload["band_powers_by_band"][band] == pytest.approx(value, rel=1e-5)
    assert payload["mean_checkpoint"] == str(mini.ckpt)
    assert payload["mean_checkpoint_sha256"] == mean_weights_sha256(mini.state)
    assert "per-coefficient" in payload["definition"]
    # v2 provenance seal: canonical-JSON self hash over the de-hashed
    # payload (freeze verifies it before trusting fold/split/MeanNet SHA).
    from src.model.rc_brd import compute_contract_sha256
    dehashed = {k: v for k, v in payload.items() if k != "artifact_sha256"}
    assert payload["artifact_sha256"] == compute_contract_sha256(dehashed)


def test_compute_band_powers_seven_groups_matches_by_band(mini):
    _expected, by_band = _reference_powers(mini)
    payload7 = BP.compute_band_powers(mini.config, "fold_0", n_groups=7)
    assert set(payload7["band_powers"]) == set(BAND_NAMES)
    assert payload7["band_groups"] == {b: [b] for b in BAND_NAMES}
    for band, value in by_band.items():
        assert payload7["band_powers"][band] == pytest.approx(value, rel=1e-5)
        assert payload7["band_powers_by_band"][band] == pytest.approx(value, rel=1e-5)


# ---------------------------------------------------------------------------
# Fail-closed config / checkpoint validation
# ---------------------------------------------------------------------------

def test_compute_band_powers_requires_checkpoint_key(mini):
    config = json.loads(json.dumps(mini.config))
    del config["modules"]["conditional_mean"]["checkpoint"]
    with pytest.raises(ValueError):
        BP.compute_band_powers(config, "fold_0")


def test_compute_band_powers_empty_checkpoint_string(mini):
    config = json.loads(json.dumps(mini.config))
    config["modules"]["conditional_mean"]["checkpoint"] = ""
    with pytest.raises(ValueError):
        BP.compute_band_powers(config, "fold_0")


def test_compute_band_powers_missing_checkpoint_file(mini, t_dir):
    config = json.loads(json.dumps(mini.config))
    config["modules"]["conditional_mean"]["checkpoint"] = str(t_dir / "absent.pt")
    with pytest.raises(FileNotFoundError):
        BP.compute_band_powers(config, "fold_0")


def test_compute_band_powers_split_without_samples_fails_closed(mini):
    # Every manifest row is split=train, so the val side must be empty ->
    # hard error, never a silent cross-split read.
    with pytest.raises(ValueError):
        BP.compute_band_powers(mini.config, "fold_0", split="val")


# ---------------------------------------------------------------------------
# CLI main(argv)
# ---------------------------------------------------------------------------

def test_cli_writes_json_matching_direct_call(t_dir, mini):
    cfg_path = t_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(mini.config), encoding="utf-8")
    out = t_dir / "band_powers.json"
    # --device cpu: the CLI default is auto (cuda when available), whose
    # float reduction order differs from the CPU direct call below by
    # ~1e-9 relative — beyond this test's rel=1e-9 determinism pin
    # (flaky on GPU machines; CI never reached this file before).
    rc = BP.main(["--config", str(cfg_path), "--fold", "fold_0",
                  "--device", "cpu", "--out", str(out)])
    assert rc == 0
    sealed = json.loads(out.read_text(encoding="utf-8"))
    assert sealed["split_label"] == "outer_train_fold_0"
    assert sealed["n_samples"] == 4
    assert sealed["n_patients"] == 2
    direct = BP.compute_band_powers(mini.config, "fold_0")
    assert set(sealed["band_powers"]) == set(direct["band_powers"])
    for group, value in direct["band_powers"].items():
        assert sealed["band_powers"][group] == pytest.approx(value, rel=1e-9)
    assert sealed["mean_checkpoint_sha256"] == direct["mean_checkpoint_sha256"]


def test_cli_missing_config_exits_1(t_dir):
    out = t_dir / "never_written.json"
    rc = BP.main(["--config", str(t_dir / "absent.yaml"),
                  "--fold", "fold_0", "--out", str(out)])
    assert rc == 1
    assert not out.exists()
