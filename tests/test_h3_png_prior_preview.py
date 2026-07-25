from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from scripts.estimate_h3_prior_from_png import (
    BANDS,
    PIPELINE_ID,
    PngPriorEntry,
    _index_pngs,
    band_alignment_statistics,
    band_tensors,
    build_png_entries,
    fit_isotonic_prior,
    fixed_noise_batch,
    portable_path,
    read_png_triplet,
    recoverability_from_statistics,
)
from src.model.frequency.h3_native_null_schedule import (
    load_h3_native_null_schedule,
)


def _write_png(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8)).save(path)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_id",
                "patient_id",
                "slice_id",
                "split",
                "cache_path",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def _scaled_masked_cosine(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    mask: torch.Tensor,
) -> np.ndarray:
    noisy_flat = (noisy * mask).double().flatten(1)
    clean_flat = (clean * mask).double().flatten(1)
    numerator = (noisy_flat * clean_flat).sum(dim=1)
    denominator = (
        noisy_flat.square().sum(dim=1).sqrt()
        * clean_flat.square().sum(dim=1).sqrt()
    ).clamp_min(1e-8)
    return (
        ((numerator / denominator).clamp(-1.0, 1.0) + 1.0) * 0.5
    ).numpy()


def test_sufficient_statistics_match_explicit_noisy_haar_alignment() -> None:
    generator = torch.Generator().manual_seed(17)
    residual = torch.randn((3, 1, 16, 16), generator=generator)
    noise = torch.randn((3, 1, 16, 16), generator=generator)
    mask = (
        torch.rand((3, 1, 16, 16), generator=generator) > 0.35
    ).float()
    signal_energy, cross_term, noise_energy = band_alignment_statistics(
        residual,
        noise,
        mask,
    )
    signal_scale = np.asarray([1.0, 0.8, 0.35], dtype=np.float64)
    noise_scale = np.asarray([1e-8, 0.2, 0.9], dtype=np.float64)
    actual = recoverability_from_statistics(
        signal_energy,
        cross_term,
        noise_energy,
        signal_scale=signal_scale,
        noise_scale=noise_scale,
    )

    clean_bands = band_tensors(residual)
    mask_l1 = F.max_pool2d(mask, kernel_size=2, stride=2)
    mask_l2 = F.max_pool2d(mask, kernel_size=4, stride=4)
    expected = np.empty_like(actual)
    for timestep, (alpha, sigma) in enumerate(
        zip(signal_scale, noise_scale, strict=True)
    ):
        noisy_bands = band_tensors(alpha * residual + sigma * noise)
        for band_index, band in enumerate(BANDS):
            current_mask = mask_l2 if band.startswith("l2_") else mask_l1
            expected[:, band_index, timestep] = _scaled_masked_cosine(
                noisy_bands[band],
                clean_bands[band],
                current_mask,
            )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_fixed_noise_identity_is_independent_of_batch_order() -> None:
    reference = torch.zeros((3, 1, 8, 8), dtype=torch.float32)
    sample_ids = ("001001", "002001", "003001")
    roles = ("mechanism_train", "calibration", "mechanism_train")
    original = fixed_noise_batch(
        reference,
        sample_ids=sample_ids,
        roles=roles,
        analysis_seed=123,
    )
    order = (2, 0, 1)
    reordered = fixed_noise_batch(
        reference,
        sample_ids=tuple(sample_ids[index] for index in order),
        roles=tuple(roles[index] for index in order),
        analysis_seed=123,
    )

    torch.testing.assert_close(reordered, original[list(order)])


def test_direct_png_inventory_uses_manifest_roles_not_physical_folders(
    tmp_path: Path,
) -> None:
    png_root = tmp_path / "png"
    rows = [
        {
            "sample_id": "001001",
            "patient_id": "001",
            "slice_id": "1",
            "split": "train",
            "cache_path": "unused/001001.npz",
        },
        {
            "sample_id": "002001",
            "patient_id": "002",
            "slice_id": "1",
            "split": "train",
            "cache_path": "unused/002001.npz",
        },
        {
            "sample_id": "003001",
            "patient_id": "003",
            "slice_id": "1",
            "split": "val",
            "cache_path": "unused/003001.npz",
        },
    ]
    # Deliberately store each sample under the opposite historical folder.
    physical = {"001001": "val", "002001": "train", "003001": "train"}
    for row in rows:
        sample_id = row["sample_id"]
        for modality in ("ct", "pet", "label"):
            _write_png(
                png_root / physical[sample_id] / modality / f"{sample_id}.png",
                np.full((4, 4), 255, dtype=np.uint8),
            )
    manifest = tmp_path / "split_manifest.csv"
    _write_manifest(manifest, rows)

    entries, partition, inventory = build_png_entries(
        png_root=png_root,
        manifest_path=manifest,
    )

    assert {entry.sample_id for entry in entries} == {"001001", "002001"}
    assert {entry.role for entry in entries} == {
        "mechanism_train",
        "calibration",
    }
    assert partition["003"] == "validation"
    assert inventory["selected_samples"] == 2
    assert inventory["cache_read"] is False


def test_png_preprocessing_matches_locked_contract(tmp_path: Path) -> None:
    ct_path = tmp_path / "ct.png"
    pet_path = tmp_path / "pet.png"
    mask_path = tmp_path / "mask.png"
    _write_png(ct_path, np.asarray([[0, 255], [127, 128]], dtype=np.uint8))
    _write_png(pet_path, np.asarray([[0, 255], [127, 128]], dtype=np.uint8))
    _write_png(mask_path, np.asarray([[0, 255], [127, 128]], dtype=np.uint8))
    entry = PngPriorEntry(
        sample_id="001001",
        patient_id="001",
        role="mechanism_train",
        ct_path=ct_path,
        pet_path=pet_path,
        mask_path=mask_path,
    )

    ct, pet, mask = read_png_triplet(entry, image_size=2)

    np.testing.assert_allclose(
        ct.numpy()[0],
        np.asarray([[-1.0, 1.0], [127 / 127.5 - 1, 128 / 127.5 - 1]]),
        atol=1e-7,
    )
    np.testing.assert_allclose(pet.numpy(), -ct.numpy(), atol=1e-7)
    np.testing.assert_array_equal(
        mask.numpy()[0],
        np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
    )


def test_pet_subdirectory_order_is_a_priority_not_a_duplicate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "png"
    preferred = root / "train" / "pet" / "001001.png"
    fallback = root / "train" / "pet_peizhuan" / "001001.png"
    _write_png(preferred, np.zeros((2, 2), dtype=np.uint8))
    _write_png(fallback, np.ones((2, 2), dtype=np.uint8))

    index = _index_pngs(root, ("pet", "pet_peizhuan"))

    assert index["001001"] == preferred


def test_isotonic_prior_uses_train_patients_and_is_non_increasing() -> None:
    train_a = np.asarray(
        [[0.95, 0.80, 0.82, 0.60, 0.55]] * len(BANDS),
        dtype=np.float64,
    )
    train_b = np.asarray(
        [[0.90, 0.76, 0.79, 0.58, 0.52]] * len(BANDS),
        dtype=np.float64,
    )
    calibration_a = np.zeros_like(train_a) + 0.51
    calibration_b = np.zeros_like(train_a) + 0.99
    first = fit_isotonic_prior(
        np.stack((train_a, train_b, calibration_a)),
        ("mechanism_train", "mechanism_train", "calibration"),
    )
    second = fit_isotonic_prior(
        np.stack((train_a, train_b, calibration_b)),
        ("mechanism_train", "mechanism_train", "calibration"),
    )

    np.testing.assert_allclose(
        first["isotonic_prior"],
        second["isotonic_prior"],
    )
    assert np.all(np.diff(first["isotonic_prior"], axis=1) <= 1e-12)
    assert np.all((first["isotonic_prior"] >= 0.0))
    assert np.all((first["isotonic_prior"] <= 1.0))


def test_portable_path_never_serializes_machine_absolute_path(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    inside = root / "results" / "curve.json"
    inside.parent.mkdir(parents=True)
    outside = tmp_path / "checkpoint.pt"
    outside.touch()

    assert portable_path(inside, root=root) == "results/curve.json"
    with pytest.raises(ValueError, match="paths under --root"):
        portable_path(outside, root=root)


def test_preview_payload_cannot_pass_formal_schedule_loader(
    tmp_path: Path,
) -> None:
    path = tmp_path / "preview.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline_id": PIPELINE_ID,
                "decision": "PREVIEW_ONLY",
                "preview_only": True,
                "inference_schedule_allowed": False,
                "production_activation_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="pipeline_id mismatch"):
        load_h3_native_null_schedule(
            path,
            expected_file_sha256=digest,
            expected_num_train_timesteps=1000,
        )
