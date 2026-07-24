from __future__ import annotations

import numpy as np

from scripts.validate_h3_logsnr_recoverability import (
    TASKS,
    crossing_log_snr,
    select_calibration_pair,
)
from scripts.validate_h4_noise_band_calibration import (
    BANDS,
    calibration_permutation_null,
    fit_calibrated_probes,
    validation_probe_statistics,
)


def test_crossing_uses_fixed_monotone_envelope() -> None:
    curve = [
        {"log_snr": -5.0, "recoverability": 0.10},
        {"log_snr": -2.0, "recoverability": 0.60},
        {"log_snr": 0.0, "recoverability": 0.55},
        {"log_snr": 2.0, "recoverability": 0.90},
    ]
    assert crossing_log_snr(curve, 0.58) == -2.0
    assert np.isnan(crossing_log_snr(curve, 0.95))


def test_calibration_selects_pair_and_direction_without_validation() -> None:
    rows = []
    for patient_index in range(25):
        row = {"partition": "calibration", "patient_id": str(patient_index)}
        for task_index, task in enumerate(TASKS):
            row[f"{task}_crossing_log_snr"] = float(task_index)
        rows.append(row)
    assert select_calibration_pair(rows) == (
        "local_frequency",
        "coarse",
    )


def _synthetic_h4_rows() -> list[dict]:
    rng = np.random.default_rng(12)
    rows = []
    roles = (
        ("mechanism_train", 40),
        ("calibration", 15),
        ("validation", 20),
    )
    for role, patients in roles:
        for patient_index in range(patients):
            patient_signal = rng.normal(scale=0.2)
            for band_index, band in enumerate(BANDS):
                for timestep in (100, 500, 900):
                    log_snr = (500 - timestep) / 100.0
                    evidence = rng.normal() + patient_signal
                    recoverability = (
                        0.5
                        + 0.03 * log_snr
                        + 0.18 * evidence
                        + 0.01 * band_index
                        + rng.normal(scale=0.02)
                    )
                    rows.append(
                        {
                            "partition": role,
                            "patient_id": f"{role}-{patient_index}",
                            "band": band,
                            "timestep": timestep,
                            "log_snr": log_snr,
                            "noise_calibrated_evidence": evidence,
                            "recoverability": recoverability,
                        }
                    )
    return rows


def test_h4_evidence_probe_improves_over_logsnr_baseline() -> None:
    rows = _synthetic_h4_rows()
    fitted = fit_calibrated_probes(rows)
    validation = validation_probe_statistics(
        rows,
        fitted,
        bootstrap_replicates=500,
        seed=33,
    )
    assert fitted["calibration_skill"] > 0.5
    assert validation["incremental_mse_skill"] > 0.5
    assert validation["ci95_low"] > 0.0
    null = calibration_permutation_null(
        fitted,
        replicates=20,
        seed=44,
    )
    assert np.isfinite(null).all()
