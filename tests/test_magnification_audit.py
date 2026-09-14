from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.mechanism_validation.magnification import PatientBootstrapResult
from src.mechanism_validation.magnification_audit import (
    AuditTables,
    CohortArrays,
    build_primary_gate,
    run_frozen_magnification_audit,
    validate_cohort,
    write_audit_artifacts,
)


class _IdentityModel:
    def sample(
        self,
        batch: dict[str, torch.Tensor],
        *,
        num_steps: int,
        initial_noise: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        assert num_steps == 2
        assert initial_noise.shape == batch["ct"].shape
        return {"synthetic_pet": batch["ct"]}


def _cohort() -> CohortArrays:
    ct = np.zeros((3, 1, 16, 16), dtype=np.float32)
    target = ct.copy()
    mask = np.zeros_like(ct)
    mask[0, 0, 7:9, 7:9] = 1.0
    mask[1, 0, 6:9, 7:10] = 1.0
    mask[2, 0, 5:9, 6:10] = 1.0
    cohort = pd.DataFrame(
        {
            "patient_id": ["001", "001", "002"],
            "sample_id": ["001001", "001002", "002001"],
            "slice_id": [1, 2, 1],
            "lesion_area": [4.0, 9.0, 16.0],
        }
    )
    return CohortArrays(
        cohort=cohort,
        ct=ct,
        target=target,
        mask=mask,
        sample_ids=np.asarray(cohort["sample_id"], dtype=str),
    )


def _bootstrap(
    *,
    slope: float,
    slope_high: float,
    reference: float,
    reference_low: float,
) -> PatientBootstrapResult:
    return PatientBootstrapResult(
        patients=31,
        requested_replicates=10000,
        valid_replicates=10000,
        intercept=reference,
        slope=slope,
        slope_ci95_low=slope - 0.02,
        slope_ci95_high=slope_high,
        x_reference=5.0,
        reference_prediction=reference,
        reference_ci95_low=reference_low,
        reference_ci95_high=reference + 0.02,
    )


def test_validate_cohort_rejects_identity_or_shape_mismatch() -> None:
    cohort = _cohort()
    validate_cohort(cohort)

    with pytest.raises(ValueError, match="sample_ids"):
        validate_cohort(
            CohortArrays(
                cohort=cohort.cohort,
                ct=cohort.ct,
                target=cohort.target,
                mask=cohort.mask,
                sample_ids=np.asarray(["bad", "ids", "here"]),
            )
        )

    with pytest.raises(ValueError, match="shape"):
        validate_cohort(
            CohortArrays(
                cohort=cohort.cohort,
                ct=cohort.ct[:2],
                target=cohort.target,
                mask=cohort.mask,
                sample_ids=cohort.sample_ids,
            )
        )


def test_frozen_audit_emits_full_zoom_and_roundtrip_rows() -> None:
    tables = run_frozen_magnification_audit(
        model=_IdentityModel(),
        cohort=_cohort(),
        device=torch.device("cpu"),
        seeds=(3, 5),
        crop_sizes=(8, 4),
        input_size=16,
        num_steps=2,
        batch_size=2,
    )

    assert len(tables.sample_metrics) == 3 * 2 * 2
    assert len(tables.roundtrip_metrics) == 3 * 2
    assert set(tables.sample_metrics["crop_size"]) == {4, 8}
    assert set(tables.sample_metrics["seed"]) == {3, 5}
    assert {
        "lesion_topq_peak_error_norm_full",
        "lesion_topq_peak_error_norm_zoom",
        "lesion_topq_relative_improvement",
        "context_hotspot_density_full",
        "context_hotspot_density_zoom",
    }.issubset(tables.sample_metrics.columns)
    assert not tables.patient_metrics.empty


def test_primary_gate_cannot_be_rescued_by_sensitivity_crop() -> None:
    primary = {
        "relative_topq": _bootstrap(
            slope=-0.01,
            slope_high=0.01,
            reference=0.04,
            reference_low=0.01,
        ),
        "absolute_topq": _bootstrap(
            slope=-0.01,
            slope_high=0.01,
            reference=0.02,
            reference_low=0.0,
        ),
        "context_hotspot_relative_worsening": 0.0,
        "roundtrip_fraction_of_absolute_gain": 0.0,
    }
    sensitivity = {
        "relative_topq": _bootstrap(
            slope=-0.05,
            slope_high=-0.01,
            reference=0.10,
            reference_low=0.05,
        ),
        "absolute_topq": _bootstrap(
            slope=-0.02,
            slope_high=-0.01,
            reference=0.04,
            reference_low=0.02,
        ),
        "context_hotspot_relative_worsening": 0.0,
        "roundtrip_fraction_of_absolute_gain": 0.0,
    }

    gate = build_primary_gate(
        {48: primary, 96: sensitivity},
        primary_crop_size=48,
    )

    assert gate["decision"] == "FAIL"
    assert gate["primary_crop_size"] == 48
    assert gate["sensitivity_results"]["96"]["would_pass"] is True
    assert gate["sensitivity_can_rescue_primary"] is False


def test_artifact_writer_refuses_overwrite_and_emits_schema(tmp_path) -> None:
    tables = AuditTables(
        sample_metrics=pd.DataFrame([{"patient_id": "001", "value": 1.0}]),
        patient_metrics=pd.DataFrame([{"patient_id": "001", "value": 1.0}]),
        roundtrip_metrics=pd.DataFrame([{"patient_id": "001", "value": 0.0}]),
    )
    output = tmp_path / "audit"
    manifest = {"schema_version": 1, "training_or_optimizer_used": False}
    gate = {"decision": "FAIL", "checks": {}}

    write_audit_artifacts(output, tables=tables, manifest=manifest, gate=gate)

    assert json.loads((output / "audit_manifest.json").read_text()) == manifest
    assert json.loads((output / "gate.json").read_text()) == gate
    assert (output / "sample_metrics.csv").is_file()
    assert (output / "patient_metrics.csv").is_file()
    assert (output / "roundtrip_metrics.csv").is_file()

    with pytest.raises(FileExistsError):
        write_audit_artifacts(
            output,
            tables=tables,
            manifest=manifest,
            gate=gate,
        )

