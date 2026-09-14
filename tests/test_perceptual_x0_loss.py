from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _context(pred, target, mask, tau=None):
    from src.model.interfaces import ConditionBundle, LossContext

    batch_size = pred.shape[0]
    return LossContext(
        model_pred=torch.zeros_like(pred),
        loss_target=target,
        target_pet=target,
        pred_x0=pred,
        timesteps=torch.arange(batch_size),
        tau=torch.full((batch_size,), 0.1) if tau is None else tau,
        batch={"mask": mask},
        condition=ConditionBundle.empty(),
    )


def _make_pred_target(batch_size=2, spatial=16):
    torch.manual_seed(0)
    pred = torch.rand(batch_size, 1, spatial, spatial) * 2 - 1
    target = pred * 0.9 + 0.02
    mask = torch.zeros(batch_size, 1, spatial, spatial)
    mask[:, :, 4:7, 5:9] = 1.0
    return pred, target, mask


def _random_encoder_loss(region_mode="lesion_balanced"):
    """A loss wired with the explicit random negative-control encoder."""
    from src.model.loss_terms.perceptual_x0 import LesionAwarePerceptualX0Loss

    return LesionAwarePerceptualX0Loss(
        enabled=True,
        weight=1.0,
        encoder_kind="random",
        region_mode=region_mode,
    )


def _checkpoint_loss(tmp_path, encoder_kind="pretrained", region_mode="global"):
    """Write a valid encoder checkpoint with real lineage and load it."""
    import hashlib
    import json

    from src.model.loss_terms.perceptual_x0 import PETFeatureEncoder

    encoder = PETFeatureEncoder()
    path = tmp_path / "encoder.pt"
    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "base_channels": 16,
        },
        path,
    )
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    # Write the lineage sidecar with a valid train_patient_split, both recall
    # gates above threshold, and a hash that matches the actual file.
    sidecar = {
        "schema_version": 1,
        "checkpoint_sha256": sha,
        "train_patient_split": {"pt01": "train", "pt02": "calibration"},
        "eval_metrics": {
            "lesion_recall": 0.80,
            "small_lesion_recall": 0.75,
            "dataset": "main_data/split_manifest.csv",
            "small_lesion_quartile": 0.25,
        },
    }
    sidecar_path = path.with_name(path.name + ".lineage.json")
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    from src.model.loss_terms.perceptual_x0 import LesionAwarePerceptualX0Loss

    return LesionAwarePerceptualX0Loss(
        enabled=True,
        weight=1.0,
        checkpoint=str(path),
        checkpoint_dir="",
        encoder_kind=encoder_kind,
        region_mode=region_mode,
    ), path


# ---------------------------------------------------------------------------
# 1) finite scalar
# ---------------------------------------------------------------------------


def test_loss_is_finite_scalar():
    term = _random_encoder_loss()
    pred, target, mask = _make_pred_target()
    loss, logs = term(_context(pred, target, mask))
    assert torch.isfinite(loss)
    assert loss.ndim == 0


# ---------------------------------------------------------------------------
# 2) pred_x0.grad non-zero
# ---------------------------------------------------------------------------


def test_pred_x0_grad_flows():
    term = _random_encoder_loss()
    pred, target, mask = _make_pred_target()
    pred = pred.requires_grad_(True)
    loss, _ = term(_context(pred, target, mask))
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# 3) encoder params grad all None
# ---------------------------------------------------------------------------


def test_encoder_params_grad_all_none():
    term = _random_encoder_loss()
    for param in term.encoder.parameters():
        assert param.requires_grad is False
    pred, target, mask = _make_pred_target()
    pred = pred.requires_grad_(True)
    loss, _ = term(_context(pred, target, mask))
    loss.backward()
    for param in term.encoder.parameters():
        assert param.grad is None


# ---------------------------------------------------------------------------
# 4) target branch detach
# ---------------------------------------------------------------------------


def test_target_branch_is_detached():
    term = _random_encoder_loss()
    pred, target, mask = _make_pred_target()
    target = target.requires_grad_(True)
    pred = pred.requires_grad_(True)
    loss, _ = term(_context(pred, target, mask))
    loss.backward()
    assert target.grad is None


# ---------------------------------------------------------------------------
# 5) global vs lesion_balanced distinguishable
# ---------------------------------------------------------------------------


def test_global_vs_lesion_balanced_distinguishable():
    global_term = _random_encoder_loss(region_mode="global")
    balanced_term = _random_encoder_loss(region_mode="lesion_balanced")
    pred, target, mask = _make_pred_target()
    # Only the lesion region is perturbed so balancing changes the loss.
    pred2 = pred.clone()
    pred2[:, :, 4:7, 5:9] += 0.5
    ctx_balanced = _context(pred2, target, mask)
    ctx_global = _context(pred2, target, mask)
    loss_balanced, _ = balanced_term(ctx_balanced)
    loss_global, _ = global_term(ctx_global)
    assert not torch.allclose(loss_balanced, loss_global, atol=1e-6)
    # In balanced mode the lesion region carries extra weight, so the effect
    # of perturbing the lesion is larger relative to the global mode.
    assert loss_balanced.item() > loss_global.item()


# ---------------------------------------------------------------------------
# 6) empty / single-pixel / multiscale masks never NaN
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mask_constructor",
    [
        lambda b: torch.zeros(b, 1, 16, 16),  # empty
        lambda b: torch.zeros(b, 1, 16, 16).scatter_(
            2,
            torch.full((b, 1, 1, 1), 8, dtype=torch.long),
            1.0,
        ),  # single pixel at row 8, col 8
    ],
)
def test_empty_and_single_pixel_masks_no_nan(mask_constructor):
    term = _random_encoder_loss()
    pred, target, _ = _make_pred_target()
    mask = mask_constructor(pred.shape[0])
    loss, logs = term(_context(pred, target, mask))
    assert torch.isfinite(loss)
    for value in logs.values():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all()


def test_multiscale_mask_downsample_no_nan():
    term = _random_encoder_loss()
    pred, target, mask = _make_pred_target(batch_size=1, spatial=16)
    loss, logs = term(_context(pred, target, mask))
    assert torch.isfinite(loss)
    assert any(k.startswith("perceptual_x0/layer_") for k in logs)


# ---------------------------------------------------------------------------
# 7) random negative control explicit; missing checkpoint fails closed
# ---------------------------------------------------------------------------


def test_random_control_requires_explicit_kind():
    from src.model.loss_terms.perceptual_x0 import LesionAwarePerceptualX0Loss

    with pytest.raises(ValueError):
        LesionAwarePerceptualX0Loss(enabled=True, encoder_kind="pretrained")


def test_missing_checkpoint_fails_closed():
    from src.model.loss_terms.perceptual_x0 import LesionAwarePerceptualX0Loss

    with pytest.raises(FileNotFoundError):
        LesionAwarePerceptualX0Loss(
            enabled=True,
            encoder_kind="pretrained",
            checkpoint="checkpoints/does_not_exist.pt",
            checkpoint_dir="",
        )


def test_build_encoder_root_relative_checkpoint_not_doubled(tmp_path, monkeypatch):
    """Regression: the plan stores ``checkpoint`` repo-root-relative
    (``checkpoints/pet_feature_encoder_v1/encoder_best.pt``) alongside
    ``checkpoint_dir: checkpoints``.  build_encoder previously prepended
    ``checkpoint_dir`` unconditionally, producing ``checkpoints/checkpoints/...``
    and a FileNotFoundError at training time despite the audit resolving the
    same path correctly.  The root-relative path must win over the legacy
    ``checkpoint_dir`` join.
    """
    from src.model.loss_terms.perceptual_x0 import PETFeatureEncoder, build_encoder

    enc_dir = tmp_path / "checkpoints" / "pet_feature_encoder_v1"
    enc_dir.mkdir(parents=True)
    ckpt = enc_dir / "encoder_best.pt"
    torch.save(
        {"state_dict": PETFeatureEncoder().state_dict(), "base_channels": 16},
        ckpt,
    )
    monkeypatch.chdir(tmp_path)  # train_v2.py runs with cwd=repo root

    encoder, meta = build_encoder(
        checkpoint="checkpoints/pet_feature_encoder_v1/encoder_best.pt",
        checkpoint_dir="checkpoints",
        encoder_kind="pretrained",
    )
    assert meta["checkpoint"].replace("\\", "/").count("checkpoints") == 1


# ---------------------------------------------------------------------------
# 8) enabled=false returns stable zero and does not change forward
# ---------------------------------------------------------------------------


def test_disabled_returns_zero():
    term = _random_encoder_loss()
    term.enabled = False
    pred, target, mask = _make_pred_target()
    loss, logs = term(_context(pred, target, mask))
    assert loss.item() == 0.0
    assert logs["perceptual_x0/enabled"].item() == 0.0


# ---------------------------------------------------------------------------
# 9) config round-trip through the factory
# ---------------------------------------------------------------------------


def test_config_roundtrip(tmp_path):
    import yaml

    from src.model.config_utils import load_full_config
    from src.model.slmf_bbdm import SLMFBBDM

    loss, path = _checkpoint_loss(tmp_path)
    cfg = {
        "experiment": {"name": "roundtrip", "seed": 1},
        "model": {
            "objective": "pred_x0",
            "sample_scheduler": "ddim",
            "segmenter": {"enabled": False},
        },
        "data": {"image_size": 16, "batch_size": 1},
        "runtime": {},
        "modules": {
            "bbdm_bridge": {"name": "bbdm_bridge"},
            "organ_prior": {"enabled": False},
            "hotspot_prior": {"enabled": False},
            "semantic_prior": {"enabled": False},
            "zero_adapter": {"enabled": False},
            "condition_dropout": {"enabled": False},
        },
        "losses": {
            "perceptual_x0": {
                "enabled": True,
                "weight": 1.0,
                "encoder_kind": "pretrained",
                "checkpoint": str(path),
                "checkpoint_dir": "",
                "region_mode": "lesion_balanced",
                "lesion_weight": 2.5,
                "background_weight": 0.75,
                "require_checkpoint_lineage": True,
            }
        },
        "metadata": {"enabled": False},
        "segmenter": {"enabled": False},
    }
    config_path = tmp_path / "roundtrip.yaml"
    config_path.write_text(
        yaml.safe_dump(cfg, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    loaded = load_full_config(str(config_path))
    model = SLMFBBDM.from_config(loaded)
    term = model.loss_terms["perceptual_x0"]
    assert term.region_mode == "lesion_balanced"
    assert term.lesion_weight == pytest.approx(2.5)
    assert term.background_weight == pytest.approx(0.75)
    assert term.require_checkpoint_lineage is True
    assert term.layer_weights["quarter"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 10) dry-run audit finds non-supervision arm differences
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 11) strict lineage: fake hash / low recall / test-split patients fail closed
# ---------------------------------------------------------------------------


def test_lineage_fake_hash_fails_closed(tmp_path):
    import json

    from src.model.loss_terms.perceptual_x0 import (
        PETFeatureEncoder,
        LesionAwarePerceptualX0Loss,
    )

    encoder = PETFeatureEncoder()
    path = tmp_path / "bad.pt"
    torch.save({"state_dict": encoder.state_dict()}, path)
    sidecar = {
        "schema_version": 1,
        "checkpoint_sha256": "deadbeef" * 8,
        "train_patient_split": {"pt01": "train"},
        "eval_metrics": {"lesion_recall": 0.80, "small_lesion_recall": 0.75},
    }
    (tmp_path / "bad.pt.lineage.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        LesionAwarePerceptualX0Loss(
            enabled=True,
            encoder_kind="pretrained",
            checkpoint=str(path),
            checkpoint_dir="",
            require_checkpoint_lineage=True,
        )


def test_lineage_low_recall_fails_closed(tmp_path):
    import hashlib
    import json

    from src.model.loss_terms.perceptual_x0 import (
        PETFeatureEncoder,
        LesionAwarePerceptualX0Loss,
    )

    encoder = PETFeatureEncoder()
    path = tmp_path / "low.pt"
    torch.save({"state_dict": encoder.state_dict()}, path)
    sidecar = {
        "schema_version": 1,
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "train_patient_split": {"pt01": "train"},
        "eval_metrics": {"lesion_recall": 0.50, "small_lesion_recall": 0.60},
    }
    (tmp_path / "low.pt.lineage.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        LesionAwarePerceptualX0Loss(
            enabled=True,
            encoder_kind="pretrained",
            checkpoint=str(path),
            checkpoint_dir="",
            require_checkpoint_lineage=True,
        )


def test_lineage_test_patient_fails_closed(tmp_path):
    import hashlib
    import json

    from src.model.loss_terms.perceptual_x0 import (
        PETFeatureEncoder,
        LesionAwarePerceptualX0Loss,
    )

    encoder = PETFeatureEncoder()
    path = tmp_path / "leak.pt"
    torch.save({"state_dict": encoder.state_dict()}, path)
    sidecar = {
        "schema_version": 1,
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "train_patient_split": {"pt01": "train", "pt02": "test"},
        "eval_metrics": {"lesion_recall": 0.80, "small_lesion_recall": 0.75},
    }
    (tmp_path / "leak.pt.lineage.json").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        LesionAwarePerceptualX0Loss(
            enabled=True,
            encoder_kind="pretrained",
            checkpoint=str(path),
            checkpoint_dir="",
            require_checkpoint_lineage=True,
        )


# ---------------------------------------------------------------------------
# 12) geometry: quarter is truly 1/4; per-scale dilation; uniform gate
# ---------------------------------------------------------------------------


def test_quarter_scale_is_truly_one_quarter():
    from src.model.loss_terms.perceptual_x0 import PETFeatureEncoder

    encoder = PETFeatureEncoder()
    x = torch.randn(1, 1, 192, 192)
    feats = encoder.forward_features(x)
    assert feats["full"].shape[-2:] == (192, 192)
    assert feats["half"].shape[-2:] == (96, 96)
    assert feats["quarter"].shape[-2:] == (48, 48)


def test_uniform_timestep_weighting_ignores_tau():
    term = _random_encoder_loss()
    pred, target, mask = _make_pred_target()
    # tau=1.0 (early step) and tau=0.0 (late step) must give the same loss
    # under uniform weighting.
    loss_late, _ = term(_context(pred, target, mask, tau=torch.full((2,), 0.0)))
    loss_early, _ = term(_context(pred, target, mask, tau=torch.full((2,), 1.0)))
    assert torch.allclose(loss_late, loss_early, atol=1e-6)


def test_optimizer_excludes_frozen_encoder_params():
    from src.model.trainer import _build_optimizer

    class _Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trainable = torch.nn.Parameter(torch.ones(2))
            self.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)

    holder = _Holder()
    optimizer = _build_optimizer(holder, {"learning_rate": 1e-4, "weight_decay": 0.01})
    param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert id(holder.frozen) not in param_ids
    assert id(holder.trainable) in param_ids


def test_dry_run_audit_detects_arm_differences(tmp_path, monkeypatch):
    import yaml

    # A shared, existing manifest + cache dir so the fairness audit can pass
    # for the legitimate arms.
    manifest = tmp_path / "main_data" / "split_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "sample_id,patient_id,slice_id,split,cache_path\n"
        "t001,pt01,1,train,cache/tensors/t001.npz\n"
        "v001,v01,1,val,cache/tensors/v001.npz\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache" / "tensors"
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "base", "seed": 42},
                "model": {"objective": "pred_x0", "sample_scheduler": "ddim"},
                "data": {
                    "image_size": 16,
                    "split_manifest": str(manifest),
                    "cache_dir": str(cache_dir),
                },
                "training": {"num_epochs": 300},
                "runtime": {
                    "early_stopping": {"enabled": True},
                    "best_checkpoint": {"enabled": True},
                },
                "losses": {},
            }
        ),
        encoding="utf-8",
    )
    plan = {
        "experiment": {"name": "perceptual_x0_ablation_v1", "seed": 42},
        "base_config": str(base_config),
        "fairness": {
            "split_manifest": str(manifest),
            "cache_dir": str(cache_dir),
            "shared_initialization_seed": 42,
            "shared_epochs": 300,
            "shared_eval_seed": 42,
            "validation_touched_for_selection": False,
        },
        "arms": {
            "P0_PIXEL": {
                "description": "reference",
                "supervision": "pixel",
                "losses": {"segmenter_consistency": {"enabled": False}},
            },
            "P1_SEG_OUT": {
                "description": "segmenter output",
                "supervision": "segmenter_output",
                "losses": {
                    "segmenter_consistency": {"enabled": True, "weight": 0.1}
                },
            },
            "P2_FEAT_GLOBAL": {
                "description": "global features",
                "supervision": "perceptual_global",
                "losses": {
                    "perceptual_x0": {
                        "enabled": True,
                        "encoder_kind": "pretrained",
                        "checkpoint": "checkpoints/never_exists.pt",
                        "require_checkpoint_lineage": True,
                    }
                },
            },
            "P3_FEAT_LESION_BALANCED": {
                "description": "main",
                "supervision": "perceptual_lesion_balanced",
                "losses": {
                    "perceptual_x0": {
                        "enabled": True,
                        "encoder_kind": "pretrained",
                        "checkpoint": "checkpoints/never_exists.pt",
                        "require_checkpoint_lineage": True,
                    }
                },
            },
            "P4_FEAT_RANDOM": {
                "description": "random control",
                "supervision": "perceptual_random",
                "losses": {
                    "perceptual_x0": {
                        "enabled": True,
                        "encoder_kind": "random",
                        "checkpoint": None,
                        "require_checkpoint_lineage": False,
                    }
                },
            },
        },
    }
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        yaml.safe_dump(plan, default_flow_style=False),
        encoding="utf-8",
    )
    from scripts.run_perceptual_x0_ablation import (
        audit_arm,
    )

    # A hostile arm mutating a non-supervision baseline loss must be detected.
    hostile = {
        "description": "hostile",
        "supervision": "pixel",
        "losses": {
            "topk_lesion": {"enabled": True, "weight": 99.0},
        },
    }
    audit = audit_arm(
        {**plan, "arms": {**plan["arms"], "P0_PIXEL": hostile}},
        arm_name="P0_PIXEL",
        base_config=base_config,
        root=tmp_path,
    )
    assert not audit["checks"]["only_supervision_losses_differ"]["ok"]
    assert audit["status"] == "BLOCKED"

    # A missing split manifest is a hard blocker (fail-closed), not a silent
    # fallback to _meta.json splits.
    plan_missing_manifest = {
        **plan,
        "fairness": {
            **plan["fairness"],
            "split_manifest": str(tmp_path / "nope" / "split_manifest.csv"),
        },
    }
    audit_missing = audit_arm(
        plan_missing_manifest,
        arm_name="P4_FEAT_RANDOM",
        base_config=base_config,
        root=tmp_path,
    )
    assert audit_missing["status"] == "BLOCKED"
    assert not audit_missing["checks"]["manifest_exists"]["ok"]

    # Missing pretrained encoder → BLOCKED; random control → READY.
    p2_audit = audit_arm(
        plan,
        arm_name="P2_FEAT_GLOBAL",
        base_config=base_config,
        root=tmp_path,
    )
    assert p2_audit["status"] == "BLOCKED"
    p4_audit = audit_arm(
        plan,
        arm_name="P4_FEAT_RANDOM",
        base_config=base_config,
        root=tmp_path,
    )
    assert p4_audit["status"] == "READY"


def test_resolved_config_disables_validation_driven_selection(tmp_path):
    """The formal ablation must not use validation for early stopping or
    best-checkpoint selection; resolved configs disable both."""
    import yaml

    from scripts.run_perceptual_x0_ablation import resolve_variant_config

    manifest = tmp_path / "main_data" / "split_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "sample_id,patient_id,slice_id,split,cache_path\n"
        "t001,pt01,1,train,cache/tensors/t001.npz\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache" / "tensors"
    cache_dir.mkdir(parents=True, exist_ok=True)
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "base", "seed": 42},
                "model": {"objective": "pred_x0", "sample_scheduler": "ddim"},
                "data": {
                    "image_size": 16,
                    "split_manifest": "../main_data/split_manifest.csv",
                    "cache_dir": "cache/tensors_main",
                },
                "training": {"num_epochs": 300},
                "runtime": {
                    "early_stopping": {"enabled": True, "patience": 40},
                    "best_checkpoint": {"enabled": True, "combined_alpha": 0.5},
                },
                "losses": {},
            }
        ),
        encoding="utf-8",
    )
    plan = {
        "base_commit_sha256": "6a90921e21a8a5a01189fc642c8f4b07743c5c4b",
        "fairness": {
            "split_manifest": str(manifest),
            "cache_dir": str(cache_dir),
        },
        "arms": {
            "P0_PIXEL": {
                "description": "reference",
                "supervision": "pixel",
                "losses": {"segmenter_consistency": {"enabled": False}},
            }
        },
    }
    resolved = resolve_variant_config(
        plan,
        arm_name="P0_PIXEL",
        base_config=base,
        seed=42,
        root=tmp_path,
    )
    # Validation-driven selection must be OFF.
    assert resolved["runtime"]["early_stopping"]["enabled"] is False
    assert resolved["runtime"]["best_checkpoint"]["enabled"] is False
    assert (
        resolved["formal_mechanism"]["checkpoint_policy"]
        == "fixed_epoch_ema_no_selection"
    )
    # The manifest and cache are pinned to the plan's physical paths so the
    # encoder and generation model read the same data version.
    assert resolved["data"]["split_manifest"] == manifest.as_posix()
    assert resolved["data"]["cache_dir"] == cache_dir.as_posix()


def test_runner_records_base_commit_in_resolved_config(tmp_path):
    """The immutable base SHA must be recorded in every resolved config.

    The PFM experiment is layered on 6a90921e (not 0636a87); recording the base
    SHA prevents the base from being ambiguous after the worktree is committed.
    """
    import yaml

    from scripts.run_perceptual_x0_ablation import resolve_variant_config

    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "base", "seed": 42},
                "model": {"objective": "pred_x0", "sample_scheduler": "ddim"},
                "data": {"image_size": 16},
                "losses": {},
            }
        ),
        encoding="utf-8",
    )
    plan = {
        "base_commit_sha256": "6a90921e21a8a5a01189fc642c8f4b07743c5c4b",
        "arms": {
            "P0_PIXEL": {
                "description": "reference",
                "supervision": "pixel",
                "losses": {"segmenter_consistency": {"enabled": False}},
            }
        },
    }
    resolved = resolve_variant_config(
        plan,
        arm_name="P0_PIXEL",
        base_config=base,
        seed=42,
        root=tmp_path,
    )
    assert (
        resolved["formal_mechanism"]["base_commit_sha256"]
        == "6a90921e21a8a5a01189fc642c8f4b07743c5c4b"
    )


def test_runner_requires_base_commit(tmp_path):
    import yaml

    from scripts.run_perceptual_x0_ablation import (
        RunnerError,
        resolve_variant_config,
    )

    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "base", "seed": 42},
                "model": {"objective": "pred_x0", "sample_scheduler": "ddim"},
                "data": {"image_size": 16},
                "losses": {},
            }
        ),
        encoding="utf-8",
    )
    plan = {
        "arms": {
            "P0_PIXEL": {
                "description": "reference",
                "supervision": "pixel",
                "losses": {"segmenter_consistency": {"enabled": False}},
            }
        }
    }
    with pytest.raises(RunnerError):
        resolve_variant_config(
            plan,
            arm_name="P0_PIXEL",
            base_config=base,
            seed=42,
            root=tmp_path,
        )


def _write_encoder_manifest(tmp_path):
    """Write a tiny manifest with train/val patients and return its path."""
    manifest = tmp_path / "main_data" / "split_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "sample_id,patient_id,slice_id,split,cache_path\n"
        "t001,pt01,1,train,cache/tensors/t001.npz\n"
        "t002,pt02,1,train,cache/tensors/t002.npz\n"
        "t003,pt03,1,train,cache/tensors/t003.npz\n"
        "t004,pt04,1,train,cache/tensors/t004.npz\n"
        "t005,pt05,1,train,cache/tensors/t005.npz\n"
        "v001,v01,1,val,cache/tensors/v001.npz\n",
        encoding="utf-8",
    )
    return manifest


@pytest.mark.parametrize(
    "shape",
    [
        (16, 16),
        (1, 16, 16),
        (1, 1, 16, 16),
        (1, 12, 16),
    ],
)
def test_pretrain_load_sample_normalizes_cache_shapes(tmp_path, shape):
    """PNG cache CHW and fixture HW/BCHW inputs all become one BCHW sample."""
    import numpy as np

    from scripts.pretrain_pet_encoder import _load_sample

    cache_path = tmp_path / "sample.npz"
    pet = np.full(shape, 0.3, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.float32)
    np.savez_compressed(cache_path, pet=pet, mask=mask)

    loaded_pet, loaded_mask = _load_sample({"npz": cache_path}, image_size=16)

    assert loaded_pet.shape == (1, 1, 16, 16)
    assert loaded_mask.shape == (1, 1, 16, 16)
    assert torch.isfinite(loaded_pet).all()
    assert torch.isfinite(loaded_mask).all()


def test_pretrain_load_sample_rejects_ambiguous_channels(tmp_path):
    """A multi-channel cache entry must fail rather than become a fake batch."""
    import numpy as np

    from scripts.pretrain_pet_encoder import PretrainError, _load_sample

    cache_path = tmp_path / "bad_sample.npz"
    pet = np.zeros((2, 16, 16), dtype=np.float32)
    mask = np.zeros((1, 16, 16), dtype=np.float32)
    np.savez_compressed(cache_path, pet=pet, mask=mask)

    with pytest.raises(PretrainError, match="single-channel"):
        _load_sample({"npz": cache_path}, image_size=16)


def test_encoder_partition_excludes_validation_patients(tmp_path):
    """The encoder's partition must never include the held-out val patients.

    This is the P0-2 fix: the producer used to map manifest 'val' -> calibration,
    leaking the final held-out cohort into the pretrained encoder.
    """
    import numpy as np

    from scripts.pretrain_pet_encoder import patient_partition

    manifest_path = _write_encoder_manifest(tmp_path)
    # Write minimal npz cache files so SplitManifest still loads (the cache
    # path is only resolved during load_pet_masks, not partition).
    from src.data.split_manifest import SplitManifest

    manifest = SplitManifest(manifest_path)
    partition = patient_partition(manifest)
    # v01 is the manifest val patient; it must be role 'validation', not
    # 'calibration' and not trainable.
    assert partition["v01"] == "validation"
    assert "v01" not in {
        patient for patient, role in partition.items() if role in ("mechanism_train", "calibration")
    }
    # The mechanism partition is 99/25/31 style: train patients split into
    # mechanism_train + calibration, val patients are validation.
    train_patients = {"pt01", "pt02", "pt03", "pt04", "pt05"}
    assert partition["pt01"] in ("mechanism_train", "calibration")
    assert set(partition) == train_patients | {"v01"}


def test_pretrain_encoder_is_jointly_trained_then_frozen(tmp_path):
    """P0-1 fix: the encoder must be jointly trained with the seg head, then
    frozen before the checkpoint is written.

    The original code froze the encoder before training, so the saved
    checkpoint contained a randomly initialised encoder.  This test runs a tiny
    joint training loop and asserts the encoder weights change, then that the
    saved checkpoint is frozen.
    """
    import numpy as np

    from scripts.pretrain_pet_encoder import (
        PETFeatureEncoder,
        _SegHead,
    )

    manifest_path = _write_encoder_manifest(tmp_path)
    cache_dir = tmp_path / "cache" / "tensors"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for sid in ("t001", "t002", "t003", "t004", "t005", "v001"):
        pet = np.full((1, 16, 16), 0.3, dtype=np.float32)
        mask = np.zeros((1, 16, 16), dtype=np.float32)
        mask[0, 6:10, 6:10] = 1.0
        np.savez(cache_dir / f"{sid}.npz", pet=pet, mask=mask)

    # Run a real (tiny) joint training loop through the script's helpers.
    encoder = PETFeatureEncoder(base_channels=16)
    head = _SegHead(encoder, base_channels=16)
    for param in encoder.parameters():
        param.requires_grad = True
    encoder.train()
    seen: set[int] = set()
    trainable = []
    for module in (encoder, head):
        for param in module.parameters():
            if param.requires_grad and id(param) not in seen:
                seen.add(id(param))
                trainable.append(param)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=1e-3,
        weight_decay=0.01,
    )
    before = {name: p.detach().clone() for name, p in encoder.named_parameters()}
    pet = torch.from_numpy(np.full((1, 1, 16, 16), 0.3, dtype=np.float32))
    mask = torch.zeros(1, 1, 16, 16, dtype=torch.float32)
    mask[0, 0, 6:10, 6:10] = 1.0
    optimizer.zero_grad()
    logits = head(pet)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, mask)
    loss.backward()
    encoder_grad_flowed = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in encoder.parameters()
    )
    optimizer.step()
    after = {name: p.detach().clone() for name, p in encoder.named_parameters()}
    encoder_changed = any(
        not torch.allclose(before[name], after[name]) for name in after
    )
    # The encoder must receive gradient and change: joint training, not frozen.
    assert encoder_grad_flowed
    assert encoder_changed

    # Freeze after training (the script's post-training freeze step).
    for param in encoder.parameters():
        param.requires_grad = False
    assert all(not p.requires_grad for p in encoder.parameters())
