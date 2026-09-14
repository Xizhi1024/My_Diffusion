from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_all_mechanism_validations import (
    _decision_passes,
    _select_raw_root,
    _validate_pipeline_spec,
    run,
)
from scripts.run_mechanism_stage0_data_audit import (
    _locked_contract_mismatches,
)


def _args(
    root: Path,
    pipeline: Path,
    contract: Path,
    *,
    scope: str = "all",
    allow_partial: bool = False,
    run_id: str = "test-run",
) -> argparse.Namespace:
    return argparse.Namespace(
        root=root,
        pipeline=pipeline,
        contract=contract,
        raw_root=Path("raw"),
        cache_dir=Path("cache"),
        summary_dir=Path("results/mechanism_validation/99_full_pipeline"),
        scope=scope,
        from_stage=None,
        through_stage=None,
        run_id=run_id,
        allow_partial=allow_partial,
        dry_run=False,
    )


def _stage(
    stage_id: str,
    *,
    implemented: bool,
    requires: list[str],
    command: list[str] | None = None,
) -> dict:
    payload = {
        "id": stage_id,
        "title": stage_id,
        "hypothesis": "H1" if stage_id == "s1" else "H2",
        "scopes": ["all"],
        "implemented": implemented,
        "output": f"results/{stage_id}",
        "decision_file": "decision.json",
        "pass_path": "decision",
        "pass_value": "PASS",
        "contract_path": "dataset_contract_sha256",
        "requires": requires,
    }
    if command is not None:
        payload["command"] = command
    if not implemented:
        payload["expected_script"] = f"scripts/{stage_id}.py"
    return payload


def _write_inputs(root: Path, stages: list[dict]) -> tuple[Path, Path]:
    pipeline = root / "pipeline.json"
    contract = root / "contract.json"
    (root / "raw" / "train" / "ct").mkdir(parents=True, exist_ok=True)
    pipeline.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline_id": "test",
                "policy": {},
                "stages": stages,
            }
        ),
        encoding="utf-8",
    )
    contract.write_text(
        json.dumps({"contract_sha256": "locked-contract"}),
        encoding="utf-8",
    )
    return pipeline, contract


def test_repository_pipeline_spec_is_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = json.loads(
        (root / "configs/mechanism_validation_pipeline_v1.json").read_text(
            encoding="utf-8"
        )
    )
    stages = _validate_pipeline_spec(spec)
    assert [stage["hypothesis"] for stage in stages if stage["hypothesis"] in {
        "H1", "H2", "H3", "H4", "H5", "H6"
    }] == ["H1", "H2", "H3", "H4", "H5", "H6"]


def test_decision_requires_exact_locked_contract() -> None:
    stage = _stage("s1", implemented=True, requires=[], command=["noop"])
    passed, errors = _decision_passes(
        stage,
        {"decision": "PASS", "dataset_contract_sha256": "wrong"},
        "locked-contract",
    )
    assert not passed
    assert any("dataset-contract mismatch" in error for error in errors)


def test_decision_profile_assertions_are_enforced() -> None:
    spec = {
        "schema_version": 1,
        "decision_profiles": {
            "formal": {
                "required_assertions": {
                    "guardrails.patient_unit": "patient",
                }
            }
        },
        "stages": [
            {
                **_stage("s1", implemented=True, requires=[], command=["noop"]),
                "decision_profile": "formal",
            }
        ],
    }
    stage = _validate_pipeline_spec(spec)[0]
    passed, errors = _decision_passes(
        stage,
        {
            "decision": "PASS",
            "dataset_contract_sha256": "locked-contract",
            "guardrails": {"patient_unit": "slice"},
        },
        "locked-contract",
    )
    assert not passed
    assert any("guardrails.patient_unit" in error for error in errors)


def test_raw_root_selection_is_portable_between_local_and_cloud(
    tmp_path: Path,
) -> None:
    contract = {"raw_png": {"root": "Data/data"}}
    cloud_root = tmp_path / "main_data"
    (cloud_root / "train" / "ct").mkdir(parents=True)
    assert _select_raw_root(tmp_path, None, contract) == cloud_root.resolve()

    local_root = tmp_path / "Data" / "data"
    (local_root / "train" / "pet").mkdir(parents=True)
    assert _select_raw_root(tmp_path, None, contract) == local_root.resolve()


def test_locked_contract_content_check_does_not_depend_on_physical_root() -> None:
    contract = {
        "manifest": {"semantic_sha256": "manifest"},
        "raw_png": {
            "root": "Data/data",
            "combined_sha256": "raw",
            "file_count": 3,
            "modalities": {"ct": 1, "pet": 1, "mask": 1},
        },
        "splits": {
            "train": {"samples": 1, "patients": 1},
            "val": {"samples": 0, "patients": 0},
            "test": {"samples": 0, "patients": 0},
        },
        "preprocessing_config_sha256": "preprocessing",
    }
    assert _locked_contract_mismatches(
        contract,
        manifest_semantic_sha256="manifest",
        raw_png_combined_sha256="raw",
        raw_png_file_count=3,
        modality_counts={"ct": 1, "pet": 1, "mask": 1},
        split_sample_counts={"train": 1},
        split_patient_counts={"train": 1},
        preprocessing_config_sha256="preprocessing",
    ) == []


def test_incomplete_all_pipeline_fails_before_running_commands(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "ran.txt"
    command = [
        "{python}",
        "-c",
        f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
    ]
    stages = [
        _stage("s1", implemented=True, requires=[], command=command),
        _stage("s2", implemented=False, requires=["s1"]),
    ]
    pipeline, contract = _write_inputs(tmp_path, stages)
    decision = run(_args(tmp_path, pipeline, contract))

    assert decision["decision"] == "NOT_IMPLEMENTED"
    assert decision["commands_executed"] is False
    assert not marker.exists()
    assert decision["stage_status"] == {
        "s1": "BLOCKED_BY_INCOMPLETE_PIPELINE_PREFLIGHT",
        "s2": "NOT_IMPLEMENTED",
    }


def test_failed_gate_blocks_downstream_stage(tmp_path: Path) -> None:
    first_script = tmp_path / "first.py"
    second_script = tmp_path / "second.py"
    marker = tmp_path / "second-ran.txt"
    first_script.write_text(
        "\n".join(
            [
                "import json, pathlib, sys",
                "out = pathlib.Path(sys.argv[1])",
                "out.mkdir(parents=True, exist_ok=True)",
                "(out / 'decision.json').write_text(json.dumps({",
                "  'decision': 'FAIL',",
                "  'dataset_contract_sha256': 'locked-contract'",
                "}), encoding='utf-8')",
                "raise SystemExit(2)",
            ]
        ),
        encoding="utf-8",
    )
    second_script.write_text(
        "\n".join(
            [
                "import pathlib",
                f"pathlib.Path({str(marker)!r}).write_text('ran')",
            ]
        ),
        encoding="utf-8",
    )
    stages = [
        _stage(
            "s1",
            implemented=True,
            requires=[],
            command=["{python}", str(first_script), "{output}"],
        ),
        _stage(
            "s2",
            implemented=True,
            requires=["s1"],
            command=["{python}", str(second_script)],
        ),
    ]
    pipeline, contract = _write_inputs(tmp_path, stages)
    decision = run(_args(tmp_path, pipeline, contract))

    assert decision["decision"] == "FAIL"
    assert decision["stage_status"] == {
        "s1": "FAIL",
        "s2": "BLOCKED_UPSTREAM",
    }
    assert not marker.exists()


def test_from_stage_requires_and_accepts_verified_canonical_prerequisite(
    tmp_path: Path,
) -> None:
    second_script = tmp_path / "second.py"
    second_script.write_text(
        "\n".join(
            [
                "import json, pathlib, sys",
                "out = pathlib.Path(sys.argv[1])",
                "out.mkdir(parents=True, exist_ok=True)",
                "(out / 'decision.json').write_text(json.dumps({",
                "  'decision': 'PASS',",
                "  'dataset_contract_sha256': 'locked-contract'",
                "}), encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )
    stages = [
        _stage(
            "s1",
            implemented=True,
            requires=[],
            command=["{python}", "-c", "raise SystemExit(99)"],
        ),
        _stage(
            "s2",
            implemented=True,
            requires=["s1"],
            command=["{python}", str(second_script), "{output}"],
        ),
    ]
    pipeline, contract = _write_inputs(tmp_path, stages)
    canonical = tmp_path / "results" / "s1"
    canonical.mkdir(parents=True)
    (canonical / "decision.json").write_text(
        json.dumps(
            {
                "decision": "PASS",
                "dataset_contract_sha256": "locked-contract",
            }
        ),
        encoding="utf-8",
    )
    args = _args(tmp_path, pipeline, contract)
    args.from_stage = "s2"
    decision = run(args)

    assert decision["decision"] == "PASS"
    assert decision["stage_status"] == {"s2": "PASS"}
