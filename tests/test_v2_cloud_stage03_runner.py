from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cloud_stage03_runner_is_fail_closed_and_stage_limited():
    text = (
        ROOT / "scripts" / "run_cloud_v2_stage03.ps1"
    ).read_text(encoding="utf-8")

    required = (
        "scripts/audit_v2_freeze_integrity.py",
        "cache/tensors_main/cache_lineage.json",
        "configs/dataset_contract_stage0a_v1.json",
        "scripts/retrain_excluded_mean.ps1",
        "scripts/audit_checkpoint_lineage.py",
        "pathology_exclusion.enabled",
        "checkpoint_sha256",
        "next_stage_allowed = $false",
        "production_cutover_allowed = $false",
    )
    for token in required:
        assert token in text

    assert '-Stage", "ct_support"' not in text
    assert '-Stage", "curriculum"' not in text
    assert '-Stage", "artifact_safety"' not in text
    assert '-Stage", "h5_v2"' not in text
    assert '-Stage", "h6_v2"' not in text


def test_cloud_stage03_runner_refuses_checkpoint_overwrite_by_default():
    text = (
        ROOT / "scripts" / "run_cloud_v2_stage03.ps1"
    ).read_text(encoding="utf-8")

    assert "Refusing to overwrite existing checkpoint" in text
    assert "[switch]$AuditExisting" in text
