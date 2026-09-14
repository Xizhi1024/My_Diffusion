from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from scripts.calibrate_h3_v2_full_timestep_native_null import (
    _config_self_hash,
    _fit_and_gate,
    _validate_protocol_config,
)
from src.model.frequency.h3_native_null_schedule import (
    H3_V2_BAND_ORDER,
    H3_V2_PIPELINE_ID,
    H3_V2_ROUTE_ORDER,
    load_h3_native_null_schedule,
    native_null_routes,
)
from src.model.frequency.spectral_router import SpectralEvidenceFrequencyRouter
from src.model.trainer import resolve_checkpoint_dir, resolve_sample_dir


ROOT = Path(__file__).resolve().parents[1]


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _schedule_rows() -> dict[str, list[float]]:
    rows = (
        [1.0, 0.75, 0.50, 0.00],
        [1.0, 0.50, 0.25, 0.00],
        [0.75, 0.50, 0.25, 0.00],
        [1.0, 0.75, 0.25, 0.00],
        [0.50, 0.50, 0.25, 0.00],
        [1.0, 1.00, 0.50, 0.25],
    )
    return {
        band: list(values)
        for band, values in zip(H3_V2_BAND_ORDER, rows, strict=True)
    }


def _write_schedule(
    directory: Path,
    *,
    active_mass: dict[str, list[float]] | None = None,
    shallow_route_policy: str = "structurally_zero",
) -> tuple[Path, str, dict[str, Any]]:
    active_mass = active_mass or _schedule_rows()
    timestep_counts = {len(values) for values in active_mass.values()}
    assert len(timestep_counts) == 1
    num_train_timesteps = timestep_counts.pop()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "pipeline_id": H3_V2_PIPELINE_ID,
        "decision": "PASS",
        "inference_schedule_allowed": True,
        "production_activation_allowed": False,
        "num_train_timesteps": num_train_timesteps,
        "route_order": list(H3_V2_ROUTE_ORDER),
        "band_order": list(H3_V2_BAND_ORDER),
        "lookup_policy": "exact_integer_timestep_only",
        "shallow_route_policy": shallow_route_policy,
        "forbidden_runtime_inputs": [
            "target_pet",
            "lesion_mask",
            "recoverability_label",
            "single_slice_residual_band_evidence",
            "random_batch_adjacency",
        ],
        "native_active_mass": active_mass,
    }
    payload["schedule_sha256"] = _canonical_sha256(payload)
    path = directory / "frozen_schedule.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path, _file_sha256(path), payload


def _router_kwargs(path: Path, digest: str) -> dict[str, Any]:
    return {
        "output_channels": (8, 8, 4, 2),
        "band_scales": (0.5, 0.25),
        "ct_reliability_floors": (0.0, 0.0),
        "hidden_channels": 4,
        "gabor_enabled": False,
        "dct_enabled": False,
        "use_content_reliability": False,
        "route_policy": "h3_native_null",
        "h3_schedule_path": str(path),
        "h3_schedule_sha256": digest,
        "h3_num_train_timesteps": 4,
    }


def _gate_config(
    *,
    train_patients: int,
    calibration_patients: int,
    timesteps: int,
) -> dict[str, Any]:
    return {
        "calibration_gates": {
            "required_mechanism_train_patients": train_patients,
            "required_calibration_patients": calibration_patients,
            "required_band_count": len(H3_V2_BAND_ORDER),
            "required_timestep_count": timesteps,
            "all_values_finite_and_in_range": True,
            "all_frozen_curves_non_increasing": True,
            "minimum_active_mass_at_t0": 0.98,
            "maximum_active_mass_at_t999": 0.10,
            "calibration_schedule_vs_band_constant": {
                "required_bootstrap_ci95_high_below": 0.0,
                "minimum_improved_patient_fraction": 0.80,
                "bootstrap_replicates": 200,
                "bootstrap_seed": 17,
            },
        }
    }


def _synthetic_recoverability(
    *,
    train_patients: int = 4,
    calibration_patients: int = 4,
    timesteps: int = 9,
    calibration_matches_schedule: bool,
) -> tuple[np.ndarray, list[str], list[str], dict[str, Any]]:
    active = np.linspace(1.0, 0.0, timesteps, dtype=np.float64)
    expected = 0.5 + 0.5 * active
    per_patient = np.broadcast_to(
        expected[None, :],
        (len(H3_V2_BAND_ORDER), timesteps),
    ).copy()
    training = np.repeat(per_patient[None], train_patients, axis=0)
    if calibration_matches_schedule:
        calibration = np.repeat(
            per_patient[None],
            calibration_patients,
            axis=0,
        )
    else:
        band_constant = per_patient.mean(axis=1, keepdims=True)
        calibration = np.repeat(
            band_constant[None],
            calibration_patients,
            axis=0,
        )
        calibration = np.repeat(calibration, timesteps, axis=2)
    values = np.concatenate((training, calibration), axis=0)
    patient_ids = [
        *(f"train-{index}" for index in range(train_patients)),
        *(f"cal-{index}" for index in range(calibration_patients)),
    ]
    roles = [
        *(["mechanism_train"] * train_patients),
        *(["calibration"] * calibration_patients),
    ]
    config = _gate_config(
        train_patients=train_patients,
        calibration_patients=calibration_patients,
        timesteps=timesteps,
    )
    return values, patient_ids, roles, config


def test_schedule_loader_and_exact_native_null_lookup(tmp_path: Path) -> None:
    path, digest, _ = _write_schedule(tmp_path)

    active, metadata = load_h3_native_null_schedule(
        path,
        expected_file_sha256=digest,
        expected_num_train_timesteps=4,
    )

    assert active.shape == (2, 3, 4)
    assert metadata["file_sha256"] == digest
    assert metadata["band_order"] == list(H3_V2_BAND_ORDER)
    reference = torch.zeros(2, 1, dtype=torch.float64)
    timesteps = torch.tensor([0, 2], dtype=torch.int64)
    routes = native_null_routes(
        active,
        timesteps,
        level_index=0,
        reference=reference,
    )
    expected = torch.tensor(
        [
            [[1.00, 0.0, 0.00], [1.00, 0.0, 0.00], [0.75, 0.0, 0.25]],
            [[0.50, 0.0, 0.50], [0.25, 0.0, 0.75], [0.25, 0.0, 0.75]],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(routes, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        routes.sum(dim=-1),
        torch.ones(2, 3, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(routes[..., 1]).item() == 0

    with pytest.raises(ValueError, match="integer tensor"):
        native_null_routes(
            active,
            timesteps.float(),
            level_index=0,
            reference=reference,
        )
    with pytest.raises(ValueError, match=r"must be in \[0, 3\]"):
        native_null_routes(
            active,
            torch.tensor([4]),
            level_index=0,
            reference=reference,
        )


def test_schedule_loader_rejects_file_hash_mismatch(tmp_path: Path) -> None:
    path, _, _ = _write_schedule(tmp_path)

    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        load_h3_native_null_schedule(
            path,
            expected_file_sha256="0" * 64,
            expected_num_train_timesteps=4,
        )


def test_schedule_loader_rejects_canonical_self_hash_mismatch(
    tmp_path: Path,
) -> None:
    path, _, payload = _write_schedule(tmp_path)
    payload["schedule_sha256"] = "0" * 64
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical self-hash mismatch"):
        load_h3_native_null_schedule(
            path,
            expected_file_sha256=_file_sha256(path),
            expected_num_train_timesteps=4,
        )


def test_schedule_loader_rejects_timestep_count_mismatch(tmp_path: Path) -> None:
    path, digest, _ = _write_schedule(tmp_path)

    with pytest.raises(ValueError, match="timestep count mismatch"):
        load_h3_native_null_schedule(
            path,
            expected_file_sha256=digest,
            expected_num_train_timesteps=5,
        )


def test_schedule_loader_rejects_nonmonotonic_band(tmp_path: Path) -> None:
    rows = _schedule_rows()
    rows["l2_hl"] = [1.0, 0.25, 0.50, 0.0]
    path, digest, _ = _write_schedule(tmp_path, active_mass=rows)

    with pytest.raises(ValueError, match="not non-increasing"):
        load_h3_native_null_schedule(
            path,
            expected_file_sha256=digest,
            expected_num_train_timesteps=4,
        )


def test_schedule_loader_rejects_nonzero_shallow_policy(tmp_path: Path) -> None:
    path, digest, _ = _write_schedule(
        tmp_path,
        shallow_route_policy="calibrated",
    )

    with pytest.raises(ValueError, match="shallow route must be structurally zero"):
        load_h3_native_null_schedule(
            path,
            expected_file_sha256=digest,
            expected_num_train_timesteps=4,
        )


@pytest.mark.parametrize(
    ("path_present", "hash_present"),
    [(True, False), (False, True), (False, False)],
)
def test_h3_router_requires_both_schedule_path_and_hash(
    tmp_path: Path,
    path_present: bool,
    hash_present: bool,
) -> None:
    path, digest, _ = _write_schedule(tmp_path)
    kwargs = _router_kwargs(path, digest)
    kwargs["h3_schedule_path"] = str(path) if path_present else None
    kwargs["h3_schedule_sha256"] = digest if hash_present else None

    with pytest.raises(ValueError, match="requires both"):
        SpectralEvidenceFrequencyRouter(**kwargs)


def test_h3_router_uses_exact_schedule_and_freezes_evidence_heads(
    tmp_path: Path,
) -> None:
    path, digest, _ = _write_schedule(tmp_path)
    router = SpectralEvidenceFrequencyRouter(**_router_kwargs(path, digest))
    evidence = torch.zeros(2, 3, router.evidence_features)
    reference = torch.zeros(2, 1, 4, 4)
    timesteps = torch.tensor([1, 3], dtype=torch.int64)

    routes, temporal_smoothness = router._acquire_routes(
        1,
        evidence,
        reference,
        timesteps,
        SimpleNamespace(num_train_timesteps=4),
    )

    expected = torch.tensor(
        [
            [[0.75, 0.0, 0.25], [0.50, 0.0, 0.50], [1.00, 0.0, 0.00]],
            [[0.00, 0.0, 1.00], [0.00, 0.0, 1.00], [0.25, 0.0, 0.75]],
        ]
    )
    torch.testing.assert_close(routes, expected, rtol=0.0, atol=0.0)
    assert temporal_smoothness >= 0
    assert all(
        not parameter.requires_grad
        for module in (
            router.amplitude_heads,
            router.route_heads,
            router.no_null_route_heads,
        )
        for parameter in module.parameters()
    )


def test_h3_router_rejects_uncertainty_aware_h4_selector(tmp_path: Path) -> None:
    path, digest, _ = _write_schedule(tmp_path)
    kwargs = _router_kwargs(path, digest)
    kwargs.update(
        uncertainty_aware_router_enabled=True,
        uncertainty_aware_confidence_threshold=0.5,
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        SpectralEvidenceFrequencyRouter(**kwargs)


def test_synthetic_calibration_passes_and_never_allocates_shallow() -> None:
    values, patient_ids, roles, config = _synthetic_recoverability(
        calibration_matches_schedule=True
    )

    schedule, gate, rows = _fit_and_gate(
        patient_recoverability=values,
        patient_ids=patient_ids,
        patient_roles=roles,
        config=config,
    )

    assert gate["decision"] == "PASS"
    assert all(gate["checks"].values())
    assert gate["calibration_schedule_vs_band_constant"][
        "improved_patient_fraction"
    ] == 1.0
    assert tuple(schedule) == H3_V2_BAND_ORDER
    assert all(len(values) == 9 for values in schedule.values())
    assert all(
        earlier >= later
        for values in schedule.values()
        for earlier, later in zip(values, values[1:])
    )
    assert all(row["shallow_probability"] == 0.0 for row in rows)
    assert all(
        row["native_probability"] + row["null_probability"]
        == pytest.approx(1.0)
        for row in rows
    )


def test_synthetic_calibration_fails_when_band_constant_is_better() -> None:
    values, patient_ids, roles, config = _synthetic_recoverability(
        calibration_matches_schedule=False
    )

    _, gate, _ = _fit_and_gate(
        patient_recoverability=values,
        patient_ids=patient_ids,
        patient_roles=roles,
        config=config,
    )

    assert gate["decision"] == "FAIL"
    assert (
        gate["checks"]["calibration_bootstrap_ci95_high_below_zero"] is False
    )
    assert (
        gate["checks"]["calibration_improved_patient_fraction"] is False
    )
    assert gate["calibration_schedule_vs_band_constant"][
        "improved_patient_fraction"
    ] == 0.0


def test_protocol_config_self_hash_and_runtime_source_hashes() -> None:
    path = ROOT / "configs" / "h3_v2_full_timestep_native_null_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    assert payload["pipeline_id"] == H3_V2_PIPELINE_ID
    assert payload["runtime_contract"]["route_policy"] == "h3_native_null"
    assert payload["calibration"]["mapping"]["shallow_route_policy"] == (
        "structurally_zero"
    )
    role = payload["mechanism_role"]
    assert role["stage"] == "A1_availability_screen"
    assert role["tested_component"] == (
        "full_timestep_band_conditioned_native_null_availability"
    )
    assert role["destination_component_status"] == "NOT_EVALUATED"
    assert role["full_ternary_router_claim_allowed"] is False
    assert payload["exploratory_model_experiment"][
        "specificity_controls_deferred_until_candidate_support"
    ] == ["band_constant", "timestep_shuffle", "wrong_band_schedule"]
    assert payload["stop_and_claim_rules"]["full_ternary_router_claim"] is False
    assert (
        payload["stop_and_claim_rules"]["destination_mechanism_claim"] is False
    )
    assert payload["stop_and_claim_rules"]["production_activation"] is False
    assert payload["stop_and_claim_rules"]["h5_started"] is False
    assert payload["stop_and_claim_rules"]["h6_started"] is False

    runtime_hashes = [
        spec["file_sha256"] for spec in payload["runtime_sources"].values()
    ]
    pending = payload["config_sha256"] == "TO_BE_FROZEN" or any(
        digest == "TO_BE_FROZEN" for digest in runtime_hashes
    )
    if pending:
        assert payload["config_sha256"] == "TO_BE_FROZEN"
        pytest.skip("H3-v2 protocol hashes are intentionally pending final freeze")

    assert payload["config_sha256"] == _config_self_hash(payload)
    for spec in payload["runtime_sources"].values():
        source = ROOT / spec["path"]
        assert source.is_file(), source
        assert spec["file_sha256"] == _file_sha256(source)

    validated, observed_hashes = _validate_protocol_config(ROOT, path)
    assert validated == payload
    assert len(observed_hashes) == len(payload["runtime_sources"])


def test_run_owned_training_directories_preserve_legacy_defaults(
    tmp_path: Path,
) -> None:
    legacy = {"experiment": {"name": "legacy-name"}}
    assert resolve_checkpoint_dir(legacy) == str(
        Path("checkpoints") / "legacy-name"
    )
    assert resolve_sample_dir(legacy) == str(
        Path("outputs") / "samples" / "legacy-name"
    )

    strict = {
        "experiment": {"name": "ignored"},
        "training": {"checkpoint_dir": str(tmp_path / "checkpoints")},
        "runtime": {"sample_dir": str(tmp_path / "samples")},
    }
    assert resolve_checkpoint_dir(strict) == str(tmp_path / "checkpoints")
    assert resolve_sample_dir(strict) == str(tmp_path / "samples")

    with pytest.raises(ValueError, match="must not be empty"):
        resolve_checkpoint_dir({"training": {"checkpoint_dir": ""}})
    with pytest.raises(ValueError, match="must not be empty"):
        resolve_sample_dir({"runtime": {"sample_dir": ""}})
