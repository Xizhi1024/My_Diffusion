from __future__ import annotations

import hashlib
import json

import torch
import yaml


def test_encoder_roles_exclude_validation():
    from scripts.pretrain_tiny_segmenter import encoder_roles_from_partition

    roles = encoder_roles_from_partition(
        {
            "p1": "mechanism_train",
            "p2": "calibration",
            "p3": "validation",
        }
    )

    assert roles == {"p1": "train", "p2": "calibration"}


def test_recall_metrics_reports_overall_and_small_lesions():
    from scripts.pretrain_tiny_segmenter import recall_metrics

    class IdentitySegmenter(torch.nn.Module):
        stage = 2

        def forward(self, value):
            return value

    small_mask = torch.zeros(1, 1, 8, 8)
    small_mask[:, :, 2:4, 2:4] = 1
    large_mask = torch.zeros(1, 1, 8, 8)
    large_mask[:, :, 1:6, 1:6] = 1
    records = [
        {"pet": small_mask * 2 - 1, "mask": small_mask},
        {"pet": large_mask * 2 - 1, "mask": large_mask},
    ]

    metrics = recall_metrics(
        IdentitySegmenter(),
        records,
        device=torch.device("cpu"),
        small_quantile=0.25,
    )

    assert metrics["lesion_recall"] == 1.0
    assert metrics["small_lesion_recall"] == 1.0
    assert metrics["lesion_precision"] == 1.0
    assert metrics["lesion_dice"] == 1.0
    assert metrics["small_samples"] == 1


def _p1_audit_fixture(tmp_path):
    manifest = tmp_path / "main_data" / "split_manifest.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "sample_id,patient_id,slice_id,split,cache_path\n"
        "s1,p1,1,train,s1.npz\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache" / "tensors"
    cache_dir.mkdir(parents=True)
    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "base", "seed": 42},
                "model": {"objective": "pred_x0", "sample_scheduler": "ddim"},
                "data": {
                    "split_manifest": str(manifest),
                    "cache_dir": str(cache_dir),
                },
                "training": {"num_epochs": 300},
                "runtime": {
                    "early_stopping": {"enabled": False},
                    "best_checkpoint": {"enabled": False},
                },
                "losses": {},
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoints" / "tiny_segmenter_v1" / "segmenter.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"test-segmenter-checkpoint")
    plan = {
        "fairness": {
            "split_manifest": str(manifest),
            "cache_dir": str(cache_dir),
            "shared_initialization_seed": 42,
            "shared_epochs": 300,
            "validation_touched_for_selection": False,
        },
        "arms": {
            "P1_SEG_OUT": {
                "supervision": "segmenter_output",
                "segmenter": {
                    "enabled": True,
                    "checkpoint": str(checkpoint),
                    "require_checkpoint_lineage": True,
                },
                "losses": {
                    "segmenter_consistency": {"enabled": True},
                    "perceptual_x0": {"enabled": False},
                },
            }
        },
    }
    return plan, base_config, checkpoint


def test_p1_audit_requires_passing_segmenter_lineage(tmp_path):
    from scripts.run_perceptual_x0_ablation import audit_arm

    plan, base_config, checkpoint = _p1_audit_fixture(tmp_path)
    blocked = audit_arm(
        plan,
        arm_name="P1_SEG_OUT",
        base_config=base_config,
        root=tmp_path,
    )
    assert blocked["status"] == "BLOCKED"
    assert not blocked["checks"]["segmenter_lineage_exists"]["ok"]

    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint.with_name(checkpoint.name + ".lineage.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": digest,
                "train_patient_split": {"p1": "train", "p2": "calibration"},
                "eval_metrics": {
                    "lesion_recall": 0.82,
                    "small_lesion_recall": 0.76,
                },
                "gates": {
                    "lesion_recall_min": 0.70,
                    "small_lesion_recall_min": 0.70,
                    "pass": True,
                },
            }
        ),
        encoding="utf-8",
    )

    ready = audit_arm(
        plan,
        arm_name="P1_SEG_OUT",
        base_config=base_config,
        root=tmp_path,
    )
    assert ready["status"] == "READY"
    assert ready["checks"]["segmenter_lineage_hash"]["ok"]
    assert ready["checks"]["segmenter_recall_gate"]["ok"]
    assert ready["checks"]["segmenter_partition_roles"]["ok"]
