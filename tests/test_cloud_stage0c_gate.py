from __future__ import annotations

import json
import hashlib
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.lineage import (
    CacheLineageError,
    attach_data_lineage,
    load_checkpoint_data_lineage,
)
from scripts.run_cloud_stage0c_gate import (
    CACHE_LINEAGE_NAME,
    _load_contract_expected_arrays,
    _seal_or_validate_cache_lineage,
    _select_raw_root,
    _validate_cache,
    _validate_checkpoints,
)


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)


def _semantic_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_cache_seal_and_checkpoint_lineage(tmp_path: Path) -> None:
    sample_id = "001001"
    raw_root = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    base = np.arange(64, dtype=np.uint8).reshape(8, 8) * 4
    raw_paths = {
        "ct": raw_root / "train" / "ct" / f"{sample_id}.png",
        "pet": raw_root / "train" / "pet" / f"{sample_id}.png",
        "mask": raw_root / "train" / "label" / f"{sample_id}.png",
    }
    _write_png(raw_paths["ct"], base)
    _write_png(raw_paths["pet"], 255 - base)
    _write_png(raw_paths["mask"], (base > 127).astype(np.uint8) * 255)
    expected = _load_contract_expected_arrays(raw_paths, (192, 192))

    scale_meta = {
        "pet_physical_kind": "png_intensity",
        "pet_invert": True,
        "suv_ok": False,
        "pet_suv_available": False,
        "patient_id": "001",
        "slice_id": 1,
    }
    np.savez_compressed(
        cache_dir / f"{sample_id}.npz",
        ct=expected["ct"][None].astype(np.float32),
        pet=expected["pet"][None].astype(np.float32),
        mask=expected["mask"][None].astype(np.float32),
        scale_meta_json=np.frombuffer(
            json.dumps(scale_meta).encode("utf-8"), dtype=np.uint8
        ),
    )
    sidecar = {
        "sample_id": sample_id,
        "patient_id": "001",
        "slice_id": 1,
        "split": "train",
        "source": "png",
        "has_label": True,
        "scale_meta": scale_meta,
    }
    (cache_dir / f"{sample_id}_meta.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )

    contract = {
        "contract_sha256": "contract",
        "manifest": {"semantic_sha256": "manifest"},
        "raw_png": {"combined_sha256": "raw"},
        "preprocessing_config_sha256": "preprocessing",
        "preprocessing": {"output_image_size": [192, 192]},
    }
    manifest_by_id = {
        sample_id: {
            "sample_id": sample_id,
            "patient_id": "001",
            "slice_id": "1",
            "split": "train",
        }
    }
    raw_indexes = {
        modality: {sample_id: path} for modality, path in raw_paths.items()
    }

    bijection, content, _, rows, lineage = _validate_cache(
        cache_dir, manifest_by_id, raw_indexes, contract
    )
    assert bijection
    assert content
    assert rows[0]["sample_pass"]
    assert not rows[0]["pet_matches_noninverted"]
    assert lineage is not None

    sealed, evidence = _seal_or_validate_cache_lineage(
        cache_dir, lineage, seal=True
    )
    assert sealed
    assert evidence["written"]
    assert (cache_dir / CACHE_LINEAGE_NAME).is_file()

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {"lineage": lineage, "config": {"data": {"use_fake": False}}},
        checkpoint_path,
    )
    status, _, checkpoint_rows = _validate_checkpoints(
        [str(checkpoint_path)], tmp_path, lineage
    )
    assert status == "PASS"
    assert checkpoint_rows[0]["lineage_pass"]

    with np.load(cache_dir / f"{sample_id}.npz", allow_pickle=False) as archive:
        corrupted_payload = {key: np.asarray(archive[key]) for key in archive.files}
    corrupted_payload["pet"] = -corrupted_payload["pet"]
    np.savez_compressed(cache_dir / f"{sample_id}.npz", **corrupted_payload)
    _, corrupted_content, _, corrupted_rows, _ = _validate_cache(
        cache_dir, manifest_by_id, raw_indexes, contract
    )
    assert not corrupted_content
    assert not corrupted_rows[0]["numerical_pass"]

    corrupted_payload["pet"] = expected["pet_noninverted"][None].astype(
        np.float32
    )
    np.savez_compressed(cache_dir / f"{sample_id}.npz", **corrupted_payload)
    _, noninverted_content, evidence, noninverted_rows, _ = _validate_cache(
        cache_dir, manifest_by_id, raw_indexes, contract
    )
    assert not noninverted_content
    assert noninverted_rows[0]["pet_matches_noninverted"]
    assert evidence["pet_noninverted_match_count"] == 1


def test_raw_root_safe_autodiscovery(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    discovered_root = tmp_path / "main_data"
    for modality, folder in {"ct": "ct", "pet": "pet", "mask": "label"}.items():
        _write_png(
            discovered_root / "train" / folder / "001001.png",
            np.zeros((8, 8), dtype=np.uint8),
        )

    contract = {
        "raw_png": {
            "root": "Data/data",
            "file_count": 3,
            "modalities": {"ct": 1, "pet": 1, "mask": 1},
        }
    }
    selected, evidence = _select_raw_root(repo_root, contract, None)
    assert selected == discovered_root.resolve()
    assert evidence["mode"] == "auto_discovered_by_locked_counts"
    assert evidence["matching_candidate_count"] == 1

    explicit = repo_root / "explicit"
    selected_explicit, explicit_evidence = _select_raw_root(
        repo_root, contract, explicit
    )
    assert selected_explicit == explicit.resolve()
    assert explicit_evidence["mode"] == "explicit"


def test_main_png_cache_task_is_locked_to_contract_preprocessing() -> None:
    root = Path(__file__).resolve().parents[1]
    pixi = tomllib.loads((root / "pixi.toml").read_text(encoding="utf-8"))
    command = pixi["tasks"]["data-cache-png-main"]
    assert "--split-csv main_data/split_manifest.csv" in command
    assert "--split-manifest cache/tensors_main_manifest.csv" in command
    assert "--pet-subdirs pet" in command
    assert "--mask-threshold 0.5" in command
    assert "--pet-invert" in command
    assert "--split-manifest main_data/split_manifest.csv" not in command


def test_strict_training_lineage_is_embedded_and_auditable(
    tmp_path: Path,
) -> None:
    manifest_sha = "1" * 64
    raw_sha = "2" * 64
    preprocessing_sha = "3" * 64
    cache_payload_sha = "4" * 64

    contract_body = {
        "schema_version": 1,
        "manifest": {"semantic_sha256": manifest_sha},
        "raw_png": {"combined_sha256": raw_sha},
        "preprocessing_config_sha256": preprocessing_sha,
    }
    contract = {
        **contract_body,
        "contract_sha256": _semantic_sha256(contract_body),
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    lineage_body = {
        "schema_version": 1,
        "lineage_type": "verified_png_tensor_cache",
        "manifest_semantic_sha256": manifest_sha,
        "raw_png_combined_sha256": raw_sha,
        "preprocessing_config_sha256": preprocessing_sha,
        "dataset_contract_sha256": contract["contract_sha256"],
        "cache_payload_sha256": cache_payload_sha,
        "sample_count": 1191,
    }
    lineage = {
        **lineage_body,
        "cache_metadata_sha256": _semantic_sha256(lineage_body),
    }
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    lineage_path = cache_dir / CACHE_LINEAGE_NAME
    lineage_path.write_text(json.dumps(lineage), encoding="utf-8")

    config = {
        "data": {
            "cache_dir": "cache",
            "cache_lineage": f"cache/{CACHE_LINEAGE_NAME}",
            "dataset_contract": "contract.json",
            "require_cache_lineage": True,
        }
    }
    loaded = load_checkpoint_data_lineage(config, root=tmp_path)
    assert loaded == lineage

    checkpoint = attach_data_lineage({"model": {}}, loaded)
    checkpoint_path = tmp_path / "strict.pt"
    torch.save(checkpoint, checkpoint_path)
    status, _, rows = _validate_checkpoints(
        [str(checkpoint_path)], tmp_path, loaded
    )
    assert status == "PASS"
    assert rows[0]["lineage_pass"]
    fast_status, fast_evidence, fast_rows = _validate_checkpoints(
        [str(checkpoint_path)],
        tmp_path,
        loaded,
        hash_checkpoints=False,
        metadata_only=True,
    )
    assert fast_status == "PASS"
    assert fast_rows[0]["lineage_pass"]
    assert fast_rows[0]["sha256"] == ""
    assert fast_evidence["metadata_only_loading"] is True

    corrupted = dict(lineage)
    corrupted["cache_payload_sha256"] = "5" * 64
    lineage_path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(CacheLineageError, match="self-hash mismatch"):
        load_checkpoint_data_lineage(config, root=tmp_path)
