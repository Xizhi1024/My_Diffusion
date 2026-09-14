from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.validate_perceptual_x0_few_step import (  # noqa: E402
    _endpoint_direction,
    derive_mechanism_partition,
)


def test_endpoint_direction_ssim_is_higher_is_better():
    """The audit found SSIM direction was inverted; this locks the fix."""
    assert _endpoint_direction("ssim") is False
    assert _endpoint_direction("mae") is True
    assert _endpoint_direction("small_lesion_topq_peak_error_norm") is True
    assert _endpoint_direction("lesion_boundary_gradient_mae_norm") is True


def _write_manifest(path: Path) -> Path:
    rows = [
        "sample_id,patient_id,slice_id,split,cache_path",
        "t001,pt01,1,train,cache/tensors/t001.npz",
        "t002,pt01,2,train,cache/tensors/t002.npz",
        "t003,pt02,1,train,cache/tensors/t003.npz",
        "t004,pt03,1,train,cache/tensors/t004.npz",
        "t005,pt04,1,train,cache/tensors/t005.npz",
        "t006,pt05,1,train,cache/tensors/t006.npz",
        "v001,v01,1,val,cache/tensors/v001.npz",
        "v002,v01,2,val,cache/tensors/v002.npz",
        "v003,v02,1,val,cache/tensors/v003.npz",
    ]
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def test_partition_derives_calibration_from_train_and_validation_from_val(tmp_path):
    manifest = _write_manifest(tmp_path / "split_manifest.csv")
    partition, samples = derive_mechanism_partition(manifest, root=tmp_path)
    # 5 train patients, 20% calibration = 1 patient; 2 val patients = validation.
    assert len(samples["calibration"]["patients"]) == 1
    assert len(samples["validation"]["patients"]) == 2
    assert set(partition.values()) <= {"mechanism_train", "calibration", "validation"}
    # Calibration patients must be drawn from train, not val.
    cal = set(samples["calibration"]["patients"])
    val = set(samples["validation"]["patients"])
    assert cal.isdisjoint(val)


def test_partition_rejects_missing_manifest(tmp_path):
    with pytest.raises(Exception):
        derive_mechanism_partition(tmp_path / "nope.csv", root=tmp_path)


def test_ssim_non_inferiority_uses_fail_closed_lower_ci_bound():
    """The audit's P1 finding: SSIM used the optimistic CI upper bound.

    With effect CI [-0.14, 0.04] and margin 0.10, the old code passed because
    upper (0.04) >= -margin; the fail-closed check on the lower bound (-0.14)
    must FAIL.  A genuinely non-inferior arm (e.g. CI [-0.05, 0.02]) must pass.
    """
    from scripts.validate_perceptual_x0_few_step import _gate_non_inferiority

    def _effects(ci_low, ci_high):
        # effect_statistics recomputes the bootstrap CI, so we bypass it by
        # monkeypatching effect_statistics to return the frozen CI.
        return ci_low, ci_high

    # Patch effect_statistics inside the validator module.
    import scripts.validate_perceptual_x0_few_step as validator_module

    original = validator_module.effect_statistics

    def _frozen_statistics(effects, *, seed, replicates=10_000):
        return {
            "patients": len(effects),
            "estimate": (effects[0]["_ci_low"] + effects[0]["_ci_high"]) / 2,
            "ci95_low": effects[0]["_ci_low"],
            "ci95_high": effects[0]["_ci_high"],
            "sign_flip_p": 0.5,
            "bootstrap_replicates": replicates,
        }

    validator_module.effect_statistics = _frozen_statistics
    try:
        # SSIM is higher-is-better.  CI lower bound below -margin → FAIL.
        bad_effects = [
            {
                "patient_id": "p1",
                "metric": "ssim_mean",
                "left": 0.90,
                "right": 0.80,
                "effect_left_better": -0.05,
                "_ci_low": -0.14,
                "_ci_high": 0.04,
            }
        ]
        gate = _gate_non_inferiority(
            ni_effects={"ssim": bad_effects},
            margins={"ssim": 0.10},
            plan={},
            seed=1,
        )
        assert gate["endpoints"]["ssim"]["pass"] is False

        # A non-inferior arm (CI lower bound above -margin) → PASS.
        good_effects = [
            {
                "patient_id": "p1",
                "metric": "ssim_mean",
                "left": 0.90,
                "right": 0.85,
                "effect_left_better": 0.05,
                "_ci_low": -0.05,
                "_ci_high": 0.02,
            }
        ]
        gate_ok = _gate_non_inferiority(
            ni_effects={"ssim": good_effects},
            margins={"ssim": 0.10},
            plan={},
            seed=1,
        )
        assert gate_ok["endpoints"]["ssim"]["pass"] is True
    finally:
        validator_module.effect_statistics = original


def test_cached_dataset_fails_closed_on_missing_manifest(tmp_path):
    """A provided but missing split_manifest must raise, not silently fall
    back to _meta.json splits (which could mix train/validation patients)."""
    from src.data.dataset import CachedDataset

    # Create a cache dir with a fake npz so the dataset construction would
    # otherwise succeed.
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "t001.npz").write_bytes(b"not-a-real-npz")

    with pytest.raises(FileNotFoundError):
        CachedDataset(
            cache,
            split="train",
            split_manifest=tmp_path / "does_not_exist.csv",
            required_keys=["ct", "pet", "mask"],
        )
