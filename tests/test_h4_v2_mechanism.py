import numpy as np
import pandas as pd
import pytest


def _rows():
    from src.mechanism_validation.h4_v2 import BANDS

    records = []
    partitions = (
        ("mechanism_train", "dev_a"),
        ("mechanism_train", "dev_b"),
        ("calibration", "cal_a"),
        ("validation", "seen_a"),
    )
    for partition, patient in partitions:
        for slice_id in (1, 2, 3):
            for timestep in (0, 50):
                for band_index, band in enumerate(BANDS):
                    evidence = (
                        0.2 * slice_id
                        + 0.01 * timestep
                        + 0.05 * band_index
                    )
                    records.append(
                        {
                            "partition": partition,
                            "patient_id": patient,
                            "sample_id": f"{patient}_{slice_id}",
                            "slice_id": slice_id,
                            "band": band,
                            "timestep": timestep,
                            "log_snr": 10.0 - timestep / 10.0,
                            "noise_calibrated_evidence": evidence,
                            "recoverability": 0.5 + 0.1 * evidence,
                        }
                    )
    return pd.DataFrame(records)


def test_h4_v2_context_and_abstention_fallback_are_deterministic():
    from src.mechanism_validation.h4_v2 import (
        add_context_features,
        apply_comparators,
        apply_hierarchical_calibration,
        fit_comparators,
        fit_original_standardization,
        fit_population_stats,
    )

    context = add_context_features(_rows())
    assert context["neighbor_count"].max() == 2
    assert context["scale_peer_present"].min() == 1.0
    population = fit_population_stats(context)
    calibrated = apply_hierarchical_calibration(context, population)
    standardization = fit_original_standardization(calibrated)
    models = fit_comparators(calibrated, standardization)

    forced_low = calibrated.copy()
    forced_low["evidence_confidence"] = 0.0
    predictions = apply_comparators(
        forced_low,
        models=models,
        original_standardization=standardization,
        confidence_threshold=0.5,
    )
    np.testing.assert_allclose(
        predictions["prediction_uncertainty_aware"],
        predictions["prediction_h3_fixed_schedule"],
    )
    assert predictions["router_abstained"].min() == 1.0
    assert models["h3_fixed_schedule"]["time_conditioned"] is True
    assert models["no_route"]["time_conditioned"] is False


def test_h4_v2_frozen_probe_detects_mutation():
    from src.mechanism_validation.h4_v2 import (
        BANDS,
        MODEL_ORDER,
        seal_probe,
        validate_probe,
    )

    probe = seal_probe(
        {
            "schema_version": 2,
            "bands": list(BANDS),
            "model_order": list(MODEL_ORDER),
        }
    )
    validate_probe(probe)
    probe["bands"][0] = "tampered"
    with pytest.raises(ValueError, match="self-hash"):
        validate_probe(probe)


def test_cached_dataset_uses_authoritative_manifest_patient_and_slice(tmp_path):
    import csv

    from src.data.dataset import CachedDataset

    cache = tmp_path / "cache"
    cache.mkdir()
    sample_id = "siteA_sample_001"
    tensor = np.zeros((1, 8, 8), dtype=np.float32)
    np.savez(
        cache / f"{sample_id}.npz",
        ct=tensor,
        pet=tensor,
        mask=tensor,
    )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "patient_id", "slice_id", "split"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": sample_id,
                "patient_id": "siteA_patient_17",
                "slice_id": "42",
                "split": "test",
            }
        )

    dataset = CachedDataset(
        cache,
        split="test",
        split_manifest=manifest,
        required_keys=["ct", "pet", "mask"],
    )

    assert dataset.entries[0].patient_id == "siteA_patient_17"
    assert dataset.entries[0].slice_id == 42


def _patient_attributes(count=35):
    partitions = ("mechanism_train", "calibration", "validation")
    records = []
    for index in range(count):
        records.append(
            {
                "patient_id": f"{index:03d}",
                "manifest_split": "val" if index % 5 == 0 else "train",
                "original_partition": partitions[index % len(partitions)],
                "slice_count": 1 + index % 11,
                "mean_mask_area": 10.0 + (index * 17) % 300,
                "mean_pet_ll2_log_contrast": -0.5 + index / count,
            }
        )
    return pd.DataFrame(records)


def test_internal_cv_partition_is_patient_only_complete_and_deterministic():
    from src.mechanism_validation.internal_cv import (
        assert_patient_partition_integrity,
        balanced_partition_search,
        build_nested_roles,
        partition_fingerprint,
    )

    attributes = _patient_attributes()
    first, _ = balanced_partition_search(
        attributes,
        folds=5,
        seed=17,
        candidates=64,
        quantile_bins=4,
    )
    second, _ = balanced_partition_search(
        attributes,
        folds=5,
        seed=17,
        candidates=64,
        quantile_bins=4,
    )
    pd.testing.assert_frame_equal(first, second)
    first = first.rename(columns={"fold": "outer_fold"})
    nested, _ = build_nested_roles(
        attributes,
        first,
        outer_folds=5,
        inner_folds=5,
        inner_seed=41,
        inner_candidates=32,
        quantile_bins=4,
    )
    assert_patient_partition_integrity(
        first,
        nested,
        patient_ids=attributes["patient_id"],
        outer_folds=5,
    )
    assert len(nested) == len(attributes) * 5
    assert first["patient_id"].nunique() == len(attributes)
    assert partition_fingerprint(first, nested) == partition_fingerprint(
        first.copy(),
        nested.copy(),
    )


def test_internal_cv_sealed_mapping_detects_mutation():
    from src.mechanism_validation.internal_cv import (
        seal_mapping,
        validate_sealed_mapping,
    )

    sealed = seal_mapping(
        {"schema_version": 2, "pipeline_id": "test"},
        hash_field="plan_sha256",
    )
    validate_sealed_mapping(sealed, hash_field="plan_sha256")
    sealed["pipeline_id"] = "mutated"
    with pytest.raises(ValueError, match="self-hash mismatch"):
        validate_sealed_mapping(sealed, hash_field="plan_sha256")
