from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from src.model.frequency.h3_native_null_schedule import (
    H3_V2_BAND_ORDER,
    H3_V2_PIPELINE_ID,
    H3_V2_ROUTE_ORDER,
)
from src.model.frequency.prior_anchor_schedule import (
    DIRECT_PNG_PREVIEW,
    DIRECT_PNG_PREVIEW_PATH_POLICY,
    DIRECT_PNG_PREVIEW_PIPELINE_ID,
    FORMAL_H3_V2,
    load_prior_anchor_schedule,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _preview_payload(steps: int = 4) -> dict:
    rows = {
        band: [1.0, 0.75, 0.25, 0.0][:steps]
        for band in H3_V2_BAND_ORDER
    }
    partition_hash = "b" * 64
    return {
        "schema_version": 1,
        "pipeline_id": DIRECT_PNG_PREVIEW_PIPELINE_ID,
        "generated_at_utc": "2026-07-25T00:00:00+00:00",
        "decision": "PREVIEW_ONLY",
        "preview_only": True,
        "inference_schedule_allowed": False,
        "production_activation_allowed": False,
        "formal_h3_v2_claim_allowed": False,
        "reason": "unit-test preview",
        "paths": {
            "png_root": "Data/data",
            "manifest": "main_data/split_manifest.csv",
            "dataset_contract": "configs/dataset_contract_stage0a_v1.json",
            "mean_checkpoint": "checkpoints/mean/model.pt",
            "output_dir": "artifacts/preview",
        },
        "path_policy": DIRECT_PNG_PREVIEW_PATH_POLICY,
        "input_contract": {
            "source": "raw_png_direct",
            "cache_read": False,
            "cache_written": False,
            "manifest_file_sha256": "a" * 64,
            "mechanism_partition_sha256": partition_hash,
            "image_size": 192,
            "ct_normalization": "grayscale_uint8/127.5-1",
            "pet_normalization": "(255-grayscale_uint8)/127.5-1",
            "mask_normalization": "nearest_resize_then_uint8>127",
            "physical_split_policy": "index_all_split_folders_by_sample_id",
            "raw_png_fingerprint": {
                "algorithm": "sha256",
                "canonicalization": "unit-test",
                "sha256": "c" * 64,
                "file_count": 18,
            },
        },
        "dataset_contract": {
            "contract_name": "ct_pet_png_dataset_contract",
            "contract_sha256": "d" * 64,
            "checks": {
                "raw_png_file_count": True,
                "raw_png_combined_sha256": True,
                "manifest_file_sha256": True,
            },
        },
        "checkpoint": {
            "file_sha256": "e" * 64,
            "epoch": 1,
            "mean_config": {"pathology_policy": "excluded"},
            "mechanism_partition_sha256": partition_hash,
            "has_data_lineage": True,
            "checkpoint_lineage_verified_for_current_png_run": False,
        },
        "quality_control": {
            "empty_lesion_masks": 0,
            "zero_clean_band_norm_samples": {
                band: 0 for band in H3_V2_BAND_ORDER
            },
        },
        "analysis": {
            "analysis_seed": 17,
            "num_train_timesteps": steps,
            "m_schedule": "linear",
            "sigma_scale": 1.0,
            "band_order": list(H3_V2_BAND_ORDER),
            "patient_weighting": "equal_patient_weight",
            "noise_identity": "unit-test",
            "recoverability": "unit-test",
            "active_transform": "clip((recoverability-0.5)/0.5,0,1)",
            "shape_constraint": (
                "per-band isotonic non-increasing over timestep"
            ),
            "dense_curve_method": (
                "exact A=<x,x>, B=<e,e>, C=<x,e> sufficient-statistic "
                "evaluation; no timestep interpolation"
            ),
        },
        "route_mapping_preview": {
            "route_order": list(H3_V2_ROUTE_ORDER),
            "formula": "[a,0,1-a]",
            "shallow_probability": "structurally_zero",
        },
        "crossings": {},
        "preview_native_active_mass": rows,
        "outputs": {
            "curve_csv": "artifacts/preview/h3_prior_curves.csv",
            "patient_npz": (
                "artifacts/preview/h3_prior_patient_curves.npz"
            ),
            "plot_png": "artifacts/preview/h3_prior_curves.png",
        },
    }


def _write_preview(root: Path, payload: dict) -> Path:
    path = root / "artifacts" / "preview" / "h3_prior_preview.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _formal_payload(steps: int = 4) -> dict:
    rows = {
        band: [1.0, 0.75, 0.25, 0.0][:steps]
        for band in H3_V2_BAND_ORDER
    }
    payload = {
        "schema_version": 1,
        "pipeline_id": H3_V2_PIPELINE_ID,
        "decision": "PASS",
        "inference_schedule_allowed": True,
        "production_activation_allowed": False,
        "route_order": list(H3_V2_ROUTE_ORDER),
        "band_order": list(H3_V2_BAND_ORDER),
        "num_train_timesteps": steps,
        "lookup_policy": "exact_integer_timestep_only",
        "shallow_route_policy": "structurally_zero",
        "forbidden_runtime_inputs": [
            "target_pet",
            "lesion_mask",
            "recoverability_label",
            "single_slice_residual_band_evidence",
            "random_batch_adjacency",
        ],
        "native_active_mass": rows,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["schedule_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def test_direct_png_preview_loads_only_with_explicit_lineage_opt_in(
    tmp_path: Path,
) -> None:
    path = _write_preview(tmp_path, _preview_payload())
    digest = _digest(path)

    with pytest.raises(
        ValueError,
        match="allow_unverified_preview_lineage=True",
    ):
        load_prior_anchor_schedule(
            path,
            schedule_source=DIRECT_PNG_PREVIEW,
            expected_file_sha256=digest,
            expected_num_train_timesteps=4,
            repository_root=tmp_path,
        )

    active, metadata = load_prior_anchor_schedule(
        path,
        schedule_source=DIRECT_PNG_PREVIEW,
        expected_file_sha256=digest,
        expected_num_train_timesteps=4,
        repository_root=tmp_path,
        allow_unverified_preview_lineage=True,
    )

    assert active.shape == (2, 3, 4)
    assert active.dtype == torch.float32
    torch.testing.assert_close(
        active[0, 0],
        torch.tensor([1.0, 0.75, 0.25, 0.0]),
        rtol=0.0,
        atol=0.0,
    )
    assert metadata["path"] == "artifacts/preview/h3_prior_preview.json"
    assert not Path(metadata["path"]).is_absolute()
    assert metadata["schedule_source"] == DIRECT_PNG_PREVIEW
    assert metadata["lineage_verified"] is False
    assert metadata["production_activation_allowed"] is False


def test_formal_source_delegates_to_existing_strict_h3_loader(
    tmp_path: Path,
) -> None:
    path = tmp_path / "formal.json"
    path.write_text(
        json.dumps(_formal_payload(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    active, metadata = load_prior_anchor_schedule(
        path,
        schedule_source=FORMAL_H3_V2,
        expected_file_sha256=_digest(path),
        expected_num_train_timesteps=4,
    )

    assert active.shape == (2, 3, 4)
    assert metadata["pipeline_id"] == H3_V2_PIPELINE_ID


def test_formal_source_rejects_preview_lineage_override(
    tmp_path: Path,
) -> None:
    path = tmp_path / "formal.json"
    path.write_text(
        json.dumps(_formal_payload(), ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only valid for direct_png_preview"):
        load_prior_anchor_schedule(
            path,
            schedule_source=FORMAL_H3_V2,
            expected_file_sha256=_digest(path),
            expected_num_train_timesteps=4,
            allow_unverified_preview_lineage=True,
        )


def test_direct_png_preview_rejects_file_hash_mismatch(
    tmp_path: Path,
) -> None:
    path = _write_preview(tmp_path, _preview_payload())

    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        load_prior_anchor_schedule(
            path,
            schedule_source=DIRECT_PNG_PREVIEW,
            expected_file_sha256="0" * 64,
            expected_num_train_timesteps=4,
            repository_root=tmp_path,
            allow_unverified_preview_lineage=True,
        )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("pipeline_id", "wrong", "pipeline_id"),
        ("decision", "PASS", "decision"),
        ("preview_only", False, "preview_only"),
        ("inference_schedule_allowed", True, "inference_schedule_allowed"),
        ("production_activation_allowed", True, "production_activation_allowed"),
        ("formal_h3_v2_claim_allowed", True, "formal_h3_v2_claim_allowed"),
    ],
)
def test_direct_png_preview_rejects_schema_or_flag_mismatch(
    tmp_path: Path,
    key: str,
    value: object,
    message: str,
) -> None:
    payload = _preview_payload()
    payload[key] = value
    path = _write_preview(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        load_prior_anchor_schedule(
            path,
            schedule_source=DIRECT_PNG_PREVIEW,
            expected_file_sha256=_digest(path),
            expected_num_train_timesteps=4,
            repository_root=tmp_path,
            allow_unverified_preview_lineage=True,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_band", "bands mismatch"),
        ("extra_band", "bands mismatch"),
        ("wrong_length", "exactly 4 values"),
        ("increasing", "not non-increasing"),
        ("out_of_range", r"leaves \[0, 1\]"),
        ("non_finite", "non-finite"),
        ("boolean", "non-number"),
    ],
)
def test_direct_png_preview_rejects_invalid_six_by_t_curve(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    payload = _preview_payload()
    active = payload["preview_native_active_mass"]
    if mutation == "missing_band":
        active.pop("l1_hh")
    elif mutation == "extra_band":
        active["l3_hh"] = [1.0, 0.75, 0.25, 0.0]
    elif mutation == "wrong_length":
        active["l2_lh"] = [1.0, 0.5]
    elif mutation == "increasing":
        active["l2_lh"] = [1.0, 0.25, 0.5, 0.0]
    elif mutation == "out_of_range":
        active["l2_lh"][1] = -0.1
    elif mutation == "non_finite":
        active["l2_lh"][1] = float("nan")
    elif mutation == "boolean":
        active["l2_lh"][1] = True
    path = _write_preview(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        load_prior_anchor_schedule(
            path,
            schedule_source=DIRECT_PNG_PREVIEW,
            expected_file_sha256=_digest(path),
            expected_num_train_timesteps=4,
            repository_root=tmp_path,
            allow_unverified_preview_lineage=True,
        )


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("paths", "png_root", "../outside", "normalized repository-relative"),
        ("paths", "manifest", "C:/data/manifest.csv", "repository-relative"),
        ("paths", "mean_checkpoint", r"checkpoints\mean.pt", "POSIX"),
        (
            "outputs",
            "curve_csv",
            "artifacts/elsewhere/curve.csv",
            "paths.output_dir",
        ),
    ],
)
def test_direct_png_preview_rejects_nonportable_payload_paths(
    tmp_path: Path,
    section: str,
    key: str,
    value: str,
    message: str,
) -> None:
    payload = _preview_payload()
    payload[section][key] = value
    path = _write_preview(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        load_prior_anchor_schedule(
            path,
            schedule_source=DIRECT_PNG_PREVIEW,
            expected_file_sha256=_digest(path),
            expected_num_train_timesteps=4,
            repository_root=tmp_path,
            allow_unverified_preview_lineage=True,
        )


def test_direct_png_preview_file_must_remain_within_repository_root(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text(
        json.dumps(_preview_payload(), ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="within repository_root"):
        load_prior_anchor_schedule(
            outside,
            schedule_source=DIRECT_PNG_PREVIEW,
            expected_file_sha256=_digest(outside),
            expected_num_train_timesteps=4,
            repository_root=repository_root,
            allow_unverified_preview_lineage=True,
        )


def test_direct_png_preview_rejects_timestep_and_lineage_shape_mismatch(
    tmp_path: Path,
) -> None:
    payload = _preview_payload()
    payload["checkpoint"][
        "checkpoint_lineage_verified_for_current_png_run"
    ] = True
    path = _write_preview(tmp_path, payload)
    with pytest.raises(ValueError, match="explicitly unverified"):
        load_prior_anchor_schedule(
            path,
            schedule_source=DIRECT_PNG_PREVIEW,
            expected_file_sha256=_digest(path),
            expected_num_train_timesteps=4,
            repository_root=tmp_path,
            allow_unverified_preview_lineage=True,
        )

    payload = _preview_payload()
    path = _write_preview(tmp_path, payload)
    with pytest.raises(ValueError, match="timestep count mismatch"):
        load_prior_anchor_schedule(
            path,
            schedule_source=DIRECT_PNG_PREVIEW,
            expected_file_sha256=_digest(path),
            expected_num_train_timesteps=5,
            repository_root=tmp_path,
            allow_unverified_preview_lineage=True,
        )


def test_prior_anchor_schedule_rejects_unknown_source(
    tmp_path: Path,
) -> None:
    path = _write_preview(tmp_path, _preview_payload())

    with pytest.raises(ValueError, match="schedule_source must be one of"):
        load_prior_anchor_schedule(
            path,
            schedule_source="auto",
            expected_file_sha256=_digest(path),
            expected_num_train_timesteps=4,
            repository_root=tmp_path,
        )
