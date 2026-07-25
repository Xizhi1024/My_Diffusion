"""Estimate an exploratory six-band H3 prior directly from paired PNG files.

This worker deliberately bypasses every tensor cache.  It is intended for local
code validation and for portable cloud recomputation when cache layouts differ.
The output is always marked ``preview_only`` and cannot be consumed by the
fail-closed formal H3-v2 schedule loader.

For each sample it forms the frozen conditional-mean residual

    residual = PET - mean_model(CT)

and reuses one deterministic Gaussian noise image across all Brownian-bridge
timesteps.  Haar linearity lets the full 0..T-1 masked-cosine curve be evaluated
exactly from three sufficient statistics per sample and band, without expanding
``samples x timesteps x images`` in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mechanism_validation.common import (  # noqa: E402
    file_sha256,
    partition_sha256,
    patient_partition,
    read_manifest,
    write_csv,
    write_json,
)
from src.model.frequency.haar import haar_dwt2  # noqa: E402
from src.model.mean_predictor import LowFrequencyPETPredictor  # noqa: E402
from src.model.noise.base import BBDMBridgeSchedule  # noqa: E402


SCHEMA_VERSION = 1
PIPELINE_ID = "H3_DIRECT_PNG_PRIOR_PREVIEW_V1"
BANDS = ("l2_lh", "l2_hl", "l2_hh", "l1_lh", "l1_hl", "l1_hh")
DEVELOPMENT_ROLES = ("mechanism_train", "calibration")
DEFAULT_ANALYSIS_SEED = 20260730


@dataclass(frozen=True)
class PngPriorEntry:
    sample_id: str
    patient_id: str
    role: str
    ct_path: Path
    pet_path: Path
    mask_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def portable_path(path: Path, *, root: Path) -> str:
    """Serialize a repo-relative POSIX path without leaking a machine path."""

    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Portable artifacts require paths under --root: {resolved_path.name}"
        ) from exc


def _index_pngs(
    png_root: Path,
    subdirectories: Sequence[str],
) -> dict[str, Path]:
    """Index a modality across physical split folders by sample id.

    The authoritative manifest split and the historical physical folder split
    are not identical.  Physical folders are therefore file stores only.
    """

    result: dict[str, Path] = {}
    selected_subdirectory: dict[str, str] = {}
    for subdirectory in subdirectories:
        for physical_split in ("train", "val", "test"):
            folder = png_root / physical_split / subdirectory
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*.png")):
                existing = result.get(path.stem)
                if existing is not None and existing != path:
                    if selected_subdirectory[path.stem] != subdirectory:
                        # Alternative PET folders are ordered by preference.
                        continue
                    raise RuntimeError(
                        f"Duplicate PNG sample {path.stem!r}: "
                        f"{existing} and {path}"
                    )
                result[path.stem] = path
                selected_subdirectory[path.stem] = subdirectory
    return result


def _patient_subset(
    partition: Mapping[str, str],
    *,
    max_patients_per_role: int | None,
    seed: int,
) -> set[str]:
    selected: set[str] = set()
    for role in DEVELOPMENT_ROLES:
        patients = [
            patient
            for patient, current_role in partition.items()
            if current_role == role
        ]
        patients.sort(
            key=lambda patient: hashlib.sha256(
                f"{seed}|{role}|{patient}".encode("utf-8")
            ).hexdigest()
        )
        if max_patients_per_role is not None:
            patients = patients[:max_patients_per_role]
        selected.update(patients)
    return selected


def _raw_png_fingerprint(
    indexes: Mapping[str, Mapping[str, Path]],
) -> dict[str, Any]:
    records = [
        f"{sample_id}|{modality}|{file_sha256(path)}"
        for modality, index in indexes.items()
        for sample_id, path in index.items()
    ]
    digest = hashlib.sha256(
        "\n".join(sorted(records)).encode("utf-8")
    ).hexdigest()
    return {
        "algorithm": "sha256",
        "canonicalization": (
            "sorted sample_id|modality|file_sha256 records joined with LF"
        ),
        "sha256": digest,
        "file_count": len(records),
    }


def build_png_entries(
    *,
    png_root: Path,
    manifest_path: Path,
    pet_subdirectories: Sequence[str] = ("pet", "pet_peizhuan"),
    max_patients_per_role: int | None = None,
    selection_seed: int = DEFAULT_ANALYSIS_SEED,
) -> tuple[list[PngPriorEntry], dict[str, str], dict[str, Any]]:
    """Build a patient-partitioned direct-PNG inventory."""

    if max_patients_per_role is not None and max_patients_per_role <= 0:
        raise ValueError("max_patients_per_role must be positive when provided")
    manifest = read_manifest(manifest_path)
    partition = patient_partition(manifest)
    selected_patients = _patient_subset(
        partition,
        max_patients_per_role=max_patients_per_role,
        seed=selection_seed,
    )
    indexes = {
        "ct": _index_pngs(png_root, ("ct",)),
        "pet": _index_pngs(png_root, tuple(pet_subdirectories)),
        "mask": _index_pngs(png_root, ("label",)),
    }
    entries: list[PngPriorEntry] = []
    missing: list[str] = []
    for row in manifest:
        sample_id = str(row["sample_id"])
        patient_id = str(row["patient_id"])
        role = partition.get(patient_id, "")
        if role not in DEVELOPMENT_ROLES or patient_id not in selected_patients:
            continue
        absent = [name for name, index in indexes.items() if sample_id not in index]
        if absent:
            missing.append(f"{sample_id}:{','.join(absent)}")
            continue
        entries.append(
            PngPriorEntry(
                sample_id=sample_id,
                patient_id=patient_id,
                role=role,
                ct_path=indexes["ct"][sample_id],
                pet_path=indexes["pet"][sample_id],
                mask_path=indexes["mask"][sample_id],
            )
        )
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"Direct-PNG inventory has {len(missing)} incomplete samples: "
            f"{preview}"
        )
    if not entries:
        raise RuntimeError("Direct-PNG development inventory is empty")

    role_patients = {
        role: sorted(
            {
                entry.patient_id
                for entry in entries
                if entry.role == role
            }
        )
        for role in DEVELOPMENT_ROLES
    }
    if any(not patients for patients in role_patients.values()):
        raise RuntimeError("Both mechanism_train and calibration are required")
    pet_source_counts = Counter(entry.pet_path.parent.name for entry in entries)
    inventory = {
        "indexed_pngs": {
            modality: len(values) for modality, values in indexes.items()
        },
        "raw_png_fingerprint": _raw_png_fingerprint(indexes),
        "selected_samples": len(entries),
        "selected_patients": {
            role: len(patients) for role, patients in role_patients.items()
        },
        "selected_samples_by_role": {
            role: sum(entry.role == role for entry in entries)
            for role in DEVELOPMENT_ROLES
        },
        "selected_pet_source_counts": dict(sorted(pet_source_counts.items())),
        "physical_split_is_file_store_only": True,
        "cache_read": False,
        "cache_written": False,
    }
    return entries, partition, inventory


def _read_grayscale(
    path: Path,
    *,
    image_size: int,
    resample: Image.Resampling,
) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), resample)
        return np.asarray(image, dtype=np.float32)


def read_png_triplet(
    entry: PngPriorEntry,
    *,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read one triplet with the locked Stage-0A PNG preprocessing."""

    ct_raw = _read_grayscale(
        entry.ct_path,
        image_size=image_size,
        resample=Image.Resampling.BILINEAR,
    )
    pet_raw = _read_grayscale(
        entry.pet_path,
        image_size=image_size,
        resample=Image.Resampling.BILINEAR,
    )
    mask_raw = _read_grayscale(
        entry.mask_path,
        image_size=image_size,
        resample=Image.Resampling.NEAREST,
    )
    ct = ct_raw / 127.5 - 1.0
    pet = (255.0 - pet_raw) / 127.5 - 1.0
    mask = (mask_raw > 127.0).astype(np.float32)
    return (
        torch.from_numpy(ct[None].astype(np.float32)),
        torch.from_numpy(pet[None].astype(np.float32)),
        torch.from_numpy(mask[None]),
    )


class DirectPngPriorDataset(Dataset):
    def __init__(self, entries: Sequence[PngPriorEntry], *, image_size: int):
        self.entries = list(entries)
        self.image_size = int(image_size)
        if not self.entries:
            raise ValueError("DirectPngPriorDataset cannot be empty")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        ct, pet, mask = read_png_triplet(entry, image_size=self.image_size)
        return {
            "ct": ct,
            "pet": pet,
            "mask": mask,
            "sample_id": entry.sample_id,
            "patient_id": entry.patient_id,
            "role": entry.role,
        }


def load_mean_predictor(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[LowFrequencyPETPredictor, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Mean checkpoint must contain an object")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("Mean checkpoint lacks a model state")
    mean_config = checkpoint.get("mean_config", {})
    if not isinstance(mean_config, Mapping):
        raise ValueError("Mean checkpoint mean_config must be an object")
    model = LowFrequencyPETPredictor(
        in_channels=int(mean_config.get("in_channels", 1)),
        base_channels=int(mean_config.get("base_channels", 32)),
        levels=int(mean_config.get("levels", 2)),
    )
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    metadata = {
        "file_sha256": file_sha256(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "mean_config": dict(mean_config),
        "mechanism_partition_sha256": checkpoint.get(
            "mechanism_partition_sha256"
        ),
        "has_data_lineage": isinstance(checkpoint.get("data_lineage"), Mapping),
        "checkpoint_lineage_verified_for_current_png_run": False,
    }
    return model, metadata


def _noise_seed(analysis_seed: int, role: str, sample_id: str) -> int:
    digest = hashlib.sha256(
        f"{analysis_seed}|{role}|{sample_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def fixed_noise_batch(
    reference: torch.Tensor,
    *,
    sample_ids: Sequence[str],
    roles: Sequence[str],
    analysis_seed: int,
) -> torch.Tensor:
    if len(sample_ids) != int(reference.shape[0]) or len(roles) != int(
        reference.shape[0]
    ):
        raise ValueError("Noise identity batch size mismatch")
    rows = []
    for sample_id, role in zip(sample_ids, roles, strict=True):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_noise_seed(analysis_seed, role, sample_id))
        rows.append(
            torch.randn(
                tuple(reference.shape[1:]),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            )
        )
    return torch.stack(rows).to(device=reference.device, dtype=reference.dtype)


def band_tensors(image: torch.Tensor) -> dict[str, torch.Tensor]:
    ll1, details1 = haar_dwt2(image)
    _, details2 = haar_dwt2(ll1)
    return {
        "l2_lh": details2[0],
        "l2_hl": details2[1],
        "l2_hh": details2[2],
        "l1_lh": details1[0],
        "l1_hl": details1[1],
        "l1_hh": details1[2],
    }


def band_alignment_statistics(
    residual: torch.Tensor,
    noise: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return masked ``A=<x,x>``, ``B=<x,e>``, and ``C=<e,e>``."""

    if residual.shape != noise.shape or residual.shape != mask.shape:
        raise ValueError("residual, noise, and mask must have identical shapes")
    clean_bands = band_tensors(residual)
    noise_bands = band_tensors(noise)
    mask_l1 = F.max_pool2d(mask.float(), kernel_size=2, stride=2)
    mask_l2 = F.max_pool2d(mask.float(), kernel_size=4, stride=4)
    a_rows: list[torch.Tensor] = []
    cross_rows: list[torch.Tensor] = []
    noise_rows: list[torch.Tensor] = []
    for band in BANDS:
        current_mask = mask_l2 if band.startswith("l2_") else mask_l1
        clean = (clean_bands[band] * current_mask).double().flatten(1)
        epsilon = (noise_bands[band] * current_mask).double().flatten(1)
        a_rows.append(clean.square().sum(dim=1))
        cross_rows.append((clean * epsilon).sum(dim=1))
        noise_rows.append(epsilon.square().sum(dim=1))
    return tuple(
        torch.stack(rows, dim=1).cpu().numpy()
        for rows in (a_rows, cross_rows, noise_rows)
    )


def recoverability_from_statistics(
    signal_energy: np.ndarray,
    cross_term: np.ndarray,
    noise_energy: np.ndarray,
    *,
    signal_scale: np.ndarray,
    noise_scale: np.ndarray,
) -> np.ndarray:
    """Evaluate exact scaled cosine for every sample, band, and timestep."""

    a = np.asarray(signal_energy, dtype=np.float64)
    b = np.asarray(cross_term, dtype=np.float64)
    c = np.asarray(noise_energy, dtype=np.float64)
    alpha = np.asarray(signal_scale, dtype=np.float64)
    sigma = np.asarray(noise_scale, dtype=np.float64)
    if a.shape != b.shape or a.shape != c.shape or a.ndim != 2:
        raise ValueError("Band statistics must share shape [samples, bands]")
    if alpha.shape != sigma.shape or alpha.ndim != 1:
        raise ValueError("Schedule scales must share shape [timesteps]")
    numerator = alpha[None, None, :] * a[:, :, None]
    numerator += sigma[None, None, :] * b[:, :, None]
    noisy_energy = (
        np.square(alpha[None, None, :]) * a[:, :, None]
        + 2.0
        * alpha[None, None, :]
        * sigma[None, None, :]
        * b[:, :, None]
        + np.square(sigma[None, None, :]) * c[:, :, None]
    )
    noisy_energy = np.maximum(noisy_energy, 0.0)
    denominator = np.sqrt(a[:, :, None]) * np.sqrt(noisy_energy)
    denominator = np.maximum(denominator, 1e-8)
    cosine = np.clip(numerator / denominator, -1.0, 1.0)
    return ((cosine + 1.0) * 0.5).astype(np.float64)


def fit_isotonic_prior(
    patient_recoverability: np.ndarray,
    patient_roles: Sequence[str],
) -> dict[str, np.ndarray]:
    values = np.asarray(patient_recoverability, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != len(BANDS):
        raise ValueError("patient_recoverability must have shape [P,6,T]")
    roles = np.asarray(list(patient_roles), dtype=object)
    if roles.shape != (values.shape[0],):
        raise ValueError("patient role count mismatch")
    train = values[roles == "mechanism_train"]
    calibration = values[roles == "calibration"]
    if train.size == 0 or calibration.size == 0:
        raise ValueError("Both development roles are required")
    train_mean = train.mean(axis=0)
    calibration_mean = calibration.mean(axis=0)
    raw_active = np.clip((train_mean - 0.5) / 0.5, 0.0, 1.0)
    timesteps = np.arange(values.shape[2], dtype=np.float64)
    prior = np.empty_like(raw_active)
    for band_index in range(len(BANDS)):
        prior[band_index] = IsotonicRegression(
            increasing=False,
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        ).fit_transform(timesteps, raw_active[band_index])
    return {
        "mechanism_train_mean": train_mean,
        "calibration_mean": calibration_mean,
        "raw_active": raw_active,
        "isotonic_prior": np.clip(prior, 0.0, 1.0),
    }


@torch.no_grad()
def estimate_patient_curves(
    *,
    dataset: DirectPngPriorDataset,
    model: torch.nn.Module,
    schedule: BBDMBridgeSchedule,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    analysis_seed: int,
) -> tuple[
    np.ndarray,
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Estimate patient-equal recoverability curves from raw PNG batches."""

    patients = sorted({(entry.role, entry.patient_id) for entry in dataset.entries})
    patient_ids = [patient for _, patient in patients]
    patient_roles = [role for role, _ in patients]
    patient_index = {key: index for index, key in enumerate(patients)}
    timesteps = int(schedule.num_train_timesteps)
    sums = np.zeros((len(patients), len(BANDS), timesteps), dtype=np.float64)
    counts = np.zeros(len(patients), dtype=np.int64)
    zero_signal_counts = np.zeros(len(BANDS), dtype=np.int64)
    signal_scale = (1.0 - schedule.m_t).double().cpu().numpy()
    noise_scale = schedule.sigma_t.double().cpu().numpy()
    log_snr = np.log(
        np.maximum(np.square(signal_scale), 1e-8)
        / np.maximum(np.square(noise_scale), 1e-8)
    )
    log_snr = np.clip(log_snr, -20.0, 20.0)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(num_workers) > 0,
        drop_last=False,
    )
    processed = 0
    for batch_index, batch in enumerate(loader):
        ct = batch["ct"].to(device, non_blocking=True)
        pet = batch["pet"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        sample_ids = [str(value) for value in batch["sample_id"]]
        roles = [str(value) for value in batch["role"]]
        patient_batch = [str(value) for value in batch["patient_id"]]
        empty_mask = mask.flatten(1).sum(dim=1) <= 0
        if bool(empty_mask.any()):
            invalid = [
                sample_ids[index]
                for index in torch.nonzero(empty_mask).flatten().cpu().tolist()
            ]
            raise ValueError(
                "H3 prior requires a non-empty lesion mask for every sample: "
                f"{invalid}"
            )
        prediction = model(ct)
        if not isinstance(prediction, Mapping) or "mean_pet" not in prediction:
            raise ValueError("Mean predictor must return a mean_pet tensor")
        residual = (pet - prediction["mean_pet"]).float()
        fixed_noise = fixed_noise_batch(
            residual,
            sample_ids=sample_ids,
            roles=roles,
            analysis_seed=analysis_seed,
        )
        signal_energy, cross_term, noise_energy = band_alignment_statistics(
            residual,
            fixed_noise,
            mask,
        )
        zero_signal_counts += np.sum(signal_energy <= 0.0, axis=0)
        curves = recoverability_from_statistics(
            signal_energy,
            cross_term,
            noise_energy,
            signal_scale=signal_scale,
            noise_scale=noise_scale,
        )
        indices = np.asarray(
            [
                patient_index[(role, patient)]
                for role, patient in zip(roles, patient_batch, strict=True)
            ],
            dtype=np.int64,
        )
        np.add.at(sums, indices, curves)
        np.add.at(counts, indices, 1)
        processed += len(sample_ids)
        print(
            f"direct-PNG batch {batch_index + 1}: "
            f"{processed}/{len(dataset)} samples",
            flush=True,
        )
    if np.any(counts <= 0):
        raise RuntimeError("A selected patient has no PNG samples")
    curves = sums / counts[:, None, None]
    if not np.isfinite(curves).all():
        raise RuntimeError("Patient recoverability contains non-finite values")
    quality_control = {
        "empty_lesion_masks": 0,
        "zero_clean_band_norm_samples": {
            band: int(zero_signal_counts[index])
            for index, band in enumerate(BANDS)
        },
        "zero_clean_band_norm_policy": (
            "same as formal masked cosine: denominator clamp yields "
            "recoverability=0.5 when numerator is zero"
        ),
    }
    return (
        curves,
        patient_ids,
        patient_roles,
        counts,
        signal_scale,
        log_snr,
        quality_control,
    )


def _crossing_summary(prior: np.ndarray) -> dict[str, dict[str, int | None]]:
    summary: dict[str, dict[str, int | None]] = {}
    for band_index, band in enumerate(BANDS):
        summary[band] = {}
        for threshold in (0.9, 0.5, 0.1):
            indices = np.flatnonzero(prior[band_index] >= threshold)
            summary[band][f"last_timestep_at_or_above_{threshold:.1f}"] = (
                int(indices[-1]) if indices.size else None
            )
    return summary


def _curve_rows(
    fitted: Mapping[str, np.ndarray],
    *,
    log_snr: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for band_index, band in enumerate(BANDS):
        for timestep in range(len(log_snr)):
            active = float(fitted["isotonic_prior"][band_index, timestep])
            rows.append(
                {
                    "band": band,
                    "timestep": timestep,
                    "log_snr": float(log_snr[timestep]),
                    "mechanism_train_mean_recoverability": float(
                        fitted["mechanism_train_mean"][band_index, timestep]
                    ),
                    "calibration_mean_recoverability": float(
                        fitted["calibration_mean"][band_index, timestep]
                    ),
                    "raw_above_chance_active_mass": float(
                        fitted["raw_active"][band_index, timestep]
                    ),
                    "isotonic_prior_active_mass": active,
                    "native_probability": active,
                    "shallow_probability": 0.0,
                    "null_probability": 1.0 - active,
                }
            )
    return rows


def _write_plot(
    path: Path,
    fitted: Mapping[str, np.ndarray],
    *,
    log_snr: np.ndarray,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "l2_lh": "#1f77b4",
        "l2_hl": "#ff7f0e",
        "l2_hh": "#2ca02c",
        "l1_lh": "#d62728",
        "l1_hl": "#9467bd",
        "l1_hh": "#8c564b",
    }
    prior = fitted["isotonic_prior"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    timesteps = np.arange(prior.shape[1])
    for band_index, band in enumerate(BANDS):
        axes[0].plot(
            timesteps,
            prior[band_index],
            label=band,
            color=colors[band],
            linewidth=1.8,
        )
        axes[1].plot(
            log_snr,
            prior[band_index],
            label=band,
            color=colors[band],
            linewidth=1.8,
        )
    axes[0].set_xlabel("BBDM timestep (higher = noisier)")
    axes[0].set_ylabel("H3 active prior")
    axes[0].set_title("Direct-PNG H3 prior preview")
    axes[1].set_xlabel("log-SNR")
    axes[1].set_ylabel("H3 active prior")
    axes[1].set_title("Same prior in log-SNR coordinates")
    for axis in axes:
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _validate_h3_preview_parameters(args: argparse.Namespace) -> None:
    expected = {
        "image_size": (int(args.image_size), 192),
        "num_train_timesteps": (int(args.num_train_timesteps), 1000),
        "m_schedule": (str(args.m_schedule), "linear"),
        "sigma_scale": (float(args.sigma_scale), 1.0),
    }
    mismatches = {
        name: {"observed": observed, "required": required}
        for name, (observed, required) in expected.items()
        if observed != required
    }
    if mismatches:
        raise ValueError(
            "H3 direct-PNG preview parameters are protocol-fixed: "
            f"{mismatches}"
        )


def _checkpoint_is_pathology_excluded(metadata: Mapping[str, Any]) -> bool:
    config = metadata.get("mean_config", {})
    if not isinstance(config, Mapping):
        return False
    if config.get("pathology_policy") == "excluded":
        return True
    exclusion = config.get("pathology_exclusion", {})
    return isinstance(exclusion, Mapping) and exclusion.get("enabled") is True


def _validate_dataset_contract(
    path: Path,
    *,
    manifest_path: Path,
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError("Dataset contract must contain an object")
    raw_spec = payload.get("raw_png", {})
    manifest_spec = payload.get("manifest", {})
    fingerprint = inventory.get("raw_png_fingerprint", {})
    checks = {
        "raw_png_file_count": fingerprint.get("file_count")
        == raw_spec.get("file_count"),
        "raw_png_combined_sha256": fingerprint.get("sha256")
        == raw_spec.get("combined_sha256"),
        "manifest_file_sha256": file_sha256(manifest_path)
        == manifest_spec.get("source_file_sha256"),
    }
    if not all(checks.values()):
        raise ValueError(f"Direct-PNG dataset contract mismatch: {checks}")
    return {
        "contract_name": payload.get("contract_name"),
        "contract_sha256": payload.get("contract_sha256"),
        "checks": checks,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_h3_preview_parameters(args)
    root = args.root.resolve()
    png_root = _resolve(root, args.png_root)
    manifest_path = _resolve(root, args.manifest)
    dataset_contract_path = _resolve(root, args.dataset_contract)
    checkpoint_path = _resolve(root, args.mean_checkpoint)
    output_dir = _resolve(root, args.output_dir)
    entries, partition, inventory = build_png_entries(
        png_root=png_root,
        manifest_path=manifest_path,
        pet_subdirectories=args.pet_subdirectories,
        max_patients_per_role=args.max_patients_per_role,
        selection_seed=args.analysis_seed,
    )
    dataset_contract = _validate_dataset_contract(
        dataset_contract_path,
        manifest_path=manifest_path,
        inventory=inventory,
    )
    device = _device(args.device)
    model, checkpoint_metadata = load_mean_predictor(
        checkpoint_path,
        device=device,
    )
    if not _checkpoint_is_pathology_excluded(checkpoint_metadata):
        raise ValueError(
            "H3 prior requires a pathology-excluded conditional-mean checkpoint"
        )
    expected_partition_hash = partition_sha256(partition)
    declared_partition_hash = checkpoint_metadata.get(
        "mechanism_partition_sha256"
    )
    if (
        declared_partition_hash
        and declared_partition_hash != expected_partition_hash
    ):
        raise ValueError(
            "Mean checkpoint mechanism partition does not match the manifest"
        )
    dataset = DirectPngPriorDataset(entries, image_size=args.image_size)
    schedule = BBDMBridgeSchedule(
        num_train_timesteps=args.num_train_timesteps,
        m_schedule=args.m_schedule,
        sigma_scale=args.sigma_scale,
    )
    (
        patient_curves,
        patient_ids,
        patient_roles,
        patient_sample_counts,
        signal_scale,
        log_snr,
        quality_control,
    ) = estimate_patient_curves(
        dataset=dataset,
        model=model,
        schedule=schedule,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        analysis_seed=args.analysis_seed,
    )
    fitted = fit_isotonic_prior(patient_curves, patient_roles)
    prior = fitted["isotonic_prior"]
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "h3_prior_curves.csv"
    json_path = output_dir / "h3_prior_preview.json"
    npz_path = output_dir / "h3_prior_patient_curves.npz"
    plot_path = output_dir / "h3_prior_curves.png"
    write_csv(csv_path, _curve_rows(fitted, log_snr=log_snr))
    np.savez_compressed(
        npz_path,
        patient_recoverability=patient_curves.astype(np.float32),
        patient_ids=np.asarray(patient_ids, dtype="U64"),
        patient_roles=np.asarray(patient_roles, dtype="U32"),
        patient_sample_counts=patient_sample_counts,
        band_order=np.asarray(BANDS, dtype="U16"),
        signal_scale=signal_scale,
        log_snr=log_snr,
    )
    if not args.no_plot:
        _write_plot(plot_path, fitted, log_snr=log_snr)

    serialized_paths = {
        "png_root": portable_path(png_root, root=root),
        "manifest": portable_path(manifest_path, root=root),
        "dataset_contract": portable_path(dataset_contract_path, root=root),
        "mean_checkpoint": portable_path(checkpoint_path, root=root),
        "output_dir": portable_path(output_dir, root=root),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_id": PIPELINE_ID,
        "generated_at_utc": utc_now(),
        "decision": "PREVIEW_ONLY",
        "preview_only": True,
        "inference_schedule_allowed": False,
        "production_activation_allowed": False,
        "formal_h3_v2_claim_allowed": False,
        "reason": (
            "Direct raw-PNG prior estimation bypasses cache and validates the "
            "curve implementation, but formal cloud checkpoint/cache lineage "
            "must be re-established by the locked H3-v2 calibration."
        ),
        "paths": serialized_paths,
        "path_policy": (
            "paths are serialized repo-relative with POSIX separators; "
            "resolved machine paths are never persisted"
        ),
        "input_contract": {
            "source": "raw_png_direct",
            "cache_read": False,
            "cache_written": False,
            "manifest_file_sha256": file_sha256(manifest_path),
            "mechanism_partition_sha256": partition_sha256(partition),
            "image_size": args.image_size,
            "ct_normalization": "grayscale_uint8/127.5-1",
            "pet_normalization": "(255-grayscale_uint8)/127.5-1",
            "mask_normalization": "nearest_resize_then_uint8>127",
            "physical_split_policy": "index_all_split_folders_by_sample_id",
            **inventory,
        },
        "dataset_contract": dataset_contract,
        "checkpoint": checkpoint_metadata,
        "quality_control": quality_control,
        "analysis": {
            "analysis_seed": args.analysis_seed,
            "num_train_timesteps": args.num_train_timesteps,
            "m_schedule": args.m_schedule,
            "sigma_scale": args.sigma_scale,
            "band_order": list(BANDS),
            "patient_weighting": "equal_patient_weight",
            "noise_identity": (
                "sha256(analysis_seed|partition_role|sample_id), fixed across "
                "all timesteps"
            ),
            "recoverability": (
                "lesion-masked Haar-band cosine alignment scaled to [0,1]"
            ),
            "active_transform": "clip((recoverability-0.5)/0.5,0,1)",
            "shape_constraint": "per-band isotonic non-increasing over timestep",
            "dense_curve_method": (
                "exact A=<x,x>, B=<e,e>, C=<x,e> sufficient-statistic "
                "evaluation; no timestep interpolation"
            ),
        },
        "route_mapping_preview": {
            "route_order": ["native", "shallow", "null"],
            "formula": "[a,0,1-a]",
            "shallow_probability": "structurally_zero",
        },
        "crossings": _crossing_summary(prior),
        "preview_native_active_mass": {
            band: [float(value) for value in prior[index].tolist()]
            for index, band in enumerate(BANDS)
        },
        "outputs": {
            "curve_csv": portable_path(csv_path, root=root),
            "patient_npz": portable_path(npz_path, root=root),
            "plot_png": (
                portable_path(plot_path, root=root) if not args.no_plot else None
            ),
        },
    }
    write_json(json_path, payload)
    return {
        "pipeline_id": PIPELINE_ID,
        "decision": "PREVIEW_ONLY",
        "samples": len(entries),
        "patients": len(patient_ids),
        "device": str(device),
        "json": portable_path(json_path, root=root),
        "csv": portable_path(csv_path, root=root),
        "plot": (
            portable_path(plot_path, root=root) if not args.no_plot else None
        ),
    }


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--png-root", type=Path, default=Path("Data/data"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("main_data/split_manifest.csv"),
    )
    parser.add_argument(
        "--dataset-contract",
        type=Path,
        default=Path("configs/dataset_contract_stage0a_v1.json"),
    )
    parser.add_argument("--mean-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/h3_png_prior_preview"),
    )
    parser.add_argument(
        "--pet-subdirectories",
        nargs="+",
        default=["pet", "pet_peizhuan"],
    )
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--m-schedule", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--sigma-scale", type=float, default=1.0)
    parser.add_argument("--analysis-seed", type=int, default=DEFAULT_ANALYSIS_SEED)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-patients-per-role",
        type=int,
        default=None,
        help="Deterministic patient-balanced preview subset; default uses all.",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    result = run(_parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
