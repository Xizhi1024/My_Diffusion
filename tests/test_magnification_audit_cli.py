from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.audit_magnification_consistency import (
    load_source_cohort,
    main,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    config = tmp_path / "resolved_config.yaml"
    checkpoint = tmp_path / "checkpoint.pt"
    config.write_text("experiment:\n  name: dry-run\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint-placeholder")

    cohort = pd.DataFrame(
        {
            "index": [0, 1, 2],
            "lesion_area": [4.0, 9.0, 16.0],
            "patient_id": ["001", "001", "002"],
            "sample_id": ["001001", "001002", "002001"],
            "slice_id": [1, 2, 1],
        }
    )
    cohort.to_csv(source / "cohort.csv", index=False)
    ct = np.zeros((3, 1, 16, 16), dtype=np.float32)
    mask = np.zeros_like(ct)
    mask[0, 0, 7:9, 7:9] = 1.0
    mask[1, 0, 6:9, 7:10] = 1.0
    mask[2, 0, 5:9, 6:10] = 1.0
    np.savez_compressed(
        source / "cohort_tensors.npz",
        ct=ct,
        target=ct.copy(),
        mask=mask,
        sample_ids=np.asarray(cohort["sample_id"], dtype=str),
    )
    (source / "COMPLETE.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )
    (source / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config_sha256": _sha256(config),
                "checkpoint_sha256": _sha256(checkpoint),
                "checkpoint_epoch": 100,
                "checkpoint_weights": "model",
                "samples": 3,
                "patients": 2,
            }
        ),
        encoding="utf-8",
    )
    return source, config, checkpoint


def test_source_loader_preserves_zero_padded_identities(tmp_path: Path) -> None:
    source, _, _ = _source_run(tmp_path)

    cohort, source_manifest = load_source_cohort(source)

    assert cohort.cohort["patient_id"].tolist() == ["001", "001", "002"]
    assert cohort.sample_ids.tolist() == ["001001", "001002", "002001"]
    assert source_manifest["samples"] == 3


def test_dry_run_validates_contract_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    source, config, checkpoint = _source_run(tmp_path)
    output = tmp_path / "dry-run"
    arguments = [
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--source-run",
        str(source),
        "--output",
        str(output),
        "--dry-run",
        "--input-size",
        "16",
        "--crop-sizes",
        "8,4",
        "--primary-crop-size",
        "4",
        "--seeds",
        "3,5",
    ]

    assert main(arguments) == 0

    manifest = json.loads(
        (output / "audit_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["execution"]["dry_run"] is True
    assert manifest["execution"]["training_or_optimizer_used"] is False
    assert manifest["cohort"]["samples"] == 3
    assert manifest["audit"]["crop_sizes"] == [8, 4]
    assert (output / "DRY_RUN.json").is_file()
    assert not (output / "sample_metrics.csv").exists()

    with pytest.raises(FileExistsError):
        main(arguments)


def test_dry_run_fails_closed_on_source_hash_mismatch(tmp_path: Path) -> None:
    source, config, checkpoint = _source_run(tmp_path)
    checkpoint.write_bytes(b"different-checkpoint")

    with pytest.raises(ValueError, match="checkpoint hash"):
        main(
            [
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--source-run",
                str(source),
                "--output",
                str(tmp_path / "dry-run"),
                "--dry-run",
                "--input-size",
                "16",
                "--crop-sizes",
                "8,4",
                "--primary-crop-size",
                "4",
            ]
        )
