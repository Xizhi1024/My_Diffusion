from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class _BridgeSchedule:
    def __init__(self, steps: int = 100):
        self.num_train_timesteps = steps
        self.m_t = torch.linspace(0.0, 1.0, steps)
        self.sigma_t = (
            2 * self.m_t * (1 - self.m_t)
        ).sqrt().clamp_min(1e-4)


def _router(**overrides):
    from src.model.frequency.spectral_router import (
        SpectralEvidenceFrequencyRouter,
    )

    config = {
        "output_channels": (16, 16, 8, 4),
        "band_scales": (0.5, 0.25),
        "ct_reliability_floors": (0.0, 0.0),
        "hidden_channels": 8,
        "gabor_enabled": False,
        "dct_enabled": False,
        "use_content_reliability": False,
        "ct_support_enabled": True,
        "ct_support_band": "l2_hh",
        "ct_support_direction": "-",
        "ct_support_only": True,
    }
    config.update(overrides)
    return SpectralEvidenceFrequencyRouter(**config)


def test_h1_ct_support_is_bounded_and_shared_across_pet_bands():
    module = _router()
    ct = torch.zeros(1, 1, 32, 32)
    ct[:, :, ::2, ::2] = 1.0
    _, _, details_l2 = module.decompose(ct)
    support = module._ct_support_field(details_l2)

    assert support.shape == (1, 1, 8, 8)
    assert support.min() >= 0.0
    assert support.max() <= 1.0
    assert torch.isfinite(support).all()


def test_support_only_mode_removes_ct_from_learned_evidence():
    module = _router()
    residual = torch.randn(2, 1, 32, 32)
    _, residual_l1, residual_l2 = module.decompose(residual)
    schedule = _BridgeSchedule()
    timesteps = torch.tensor([20, 60])
    noise = module.noise_reliability(timesteps, schedule)
    ct_a = torch.randn_like(residual)
    ct_b = torch.randn_like(residual) * 10.0
    _, _, ct_a_l2 = module.decompose(ct_a)
    _, _, ct_b_l2 = module.decompose(ct_b)
    gate = torch.ones(2, 3, 8, 8) * module.gate_max

    evidence_a, _ = module._level_evidence(
        0,
        residual_l2,
        ct_a_l2,
        gate,
        noise[:, 0],
        timesteps,
        schedule,
        None,
        None,
        None,
    )
    evidence_b, _ = module._level_evidence(
        0,
        residual_l2,
        ct_b_l2,
        gate,
        noise[:, 0],
        timesteps,
        schedule,
        None,
        None,
        None,
    )
    torch.testing.assert_close(evidence_a, evidence_b)


def test_no_recoverability_zeroes_h4_evidence_and_logsnr():
    module = _router(use_noise_release=False)
    residual = torch.randn(2, 3, 8, 8)
    timesteps = torch.tensor([20, 60])
    schedule = _BridgeSchedule()
    calibrated = module._noise_calibrated_band_evidence(
        residual, timesteps, schedule
    )
    _, log_snr = module._time_features(
        timesteps, schedule, 0, residual
    )
    assert torch.count_nonzero(calibrated) == 0
    assert torch.count_nonzero(log_snr) == 0


def test_target_relative_hotspots_use_target_not_prediction_percentile():
    from scripts.evaluate import compute_target_relative_false_hotspots

    target = np.zeros((1, 16, 16), dtype=np.float32)
    pred = target.copy()
    mask = np.zeros_like(target)
    mask[:, 6:10, 6:10] = 1.0
    pred[:, 1, 1] = 0.5
    metrics = compute_target_relative_false_hotspots(pred, target, mask)

    assert metrics["target_relative_false_hotspot_count"] == 1.0
    assert metrics["target_relative_false_hotspot_density"] > 0.0


def test_formal_pipeline_has_no_unimplemented_stage():
    root = Path(__file__).resolve().parents[1]
    pipeline = json.loads(
        (
            root / "configs/mechanism_validation_pipeline_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert all(stage["implemented"] for stage in pipeline["stages"])
    assert [stage["hypothesis"] for stage in pipeline["stages"] if stage[
        "hypothesis"
    ] in {"H1", "H2", "H3", "H4", "H5", "H6"}] == [
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
    ]
