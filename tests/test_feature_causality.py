"""Tests for the A2 causal intervention audit (c2 lesion feature necessity)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.model.config_utils import resolve_runtime_profile


def _toy_config():
    cfg = {
        "experiment": {"name": "causal_test", "seed": 42},
        "data": {
            "image_size": 32,
            "batch_size": 2,
            "val_batch_size": 2,
            "augment": False,
            "use_fake_data": True,
            "required_keys": ["ct", "pet", "mask"],
        },
        "runtime": {
            "amp": False,
            "channels_last": False,
            "torch_compile": False,
            "num_workers": 0,
            "eval_sampling_steps": 2,
        },
        "training": {"num_epochs": 1, "learning_rate": 1e-4, "ema": {"decay": 0.999}},
        "model": {
            "objective": "pred_x0",
            "sample_scheduler": "ddim",
            "enable_heteroscedastic": False,
            "self_conditioning": {"enabled": False, "probability": 0.5},
            "base_loss": {
                "mse_weight": 1.0,
                "l1_weight": 1.0,
                "gradient_weight": 0.1,
                "min_snr_enabled": True,
                "min_snr_gamma": 5.0,
            },
        },
        "modules": {
            "gabor": {"enabled": False},
            "organ_prior": {"enabled": False},
            "hotspot_prior": {"enabled": False},
            "semantic_prior": {"enabled": False},
            "zero_adapter": {"enabled": False},
            "condition_dropout": {"enabled": False},
            "scale_adaptive_noise": {"enabled": False, "name": "bbdm_bridge"},
        },
        "losses": {
            "topk_lesion": {"enabled": False},
            "focal_frequency": {"enabled": False},
            "patch_nce": {"enabled": False},
            "roi_suv": {"enabled": False},
            "false_hotspot": {"enabled": False},
            "lesion_roi_l1": {"enabled": False},
            "outside_peak_ranking": {"enabled": False},
            "heteroscedastic_nll": {"enabled": False},
            "hotspot_prior": {"enabled": False},
            "organ_consistency": {"enabled": False},
            "segmenter_consistency": {"enabled": False},
        },
    }
    return resolve_runtime_profile(cfg)


def _make_model():
    from src.model.slmf_bbdm import SLMFBBDM

    model = SLMFBBDM.from_config(_toy_config())
    model.eval()
    return model


def _single_batch(device="cpu"):
    batch = {
        "ct": torch.randn(1, 1, 32, 32),
        "pet": torch.randn(1, 1, 32, 32),
        "mask": torch.zeros(1, 1, 32, 32),
        "meta": [{"patient_id": "fake_p0"}],
    }
    batch["mask"][0, 0, 14:18, 14:18] = 1.0
    if device != "cpu":
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
    return batch


def test_patch_bundle_context_restores_original():
    from src.mechanism_validation.feature_causality import (
        patch_build_condition_bundle,
        c2_lesion_zero,
    )

    model = _make_model()
    original_name = model.build_condition_bundle.__name__
    assert original_name == "build_condition_bundle"

    with patch_build_condition_bundle(model, c2_lesion_zero):
        assert model.build_condition_bundle.__name__ == "patched"

    # After exit, the instance attribute must be gone so the attribute resolves
    # back to the class-level method.
    assert "build_condition_bundle" not in model.__dict__
    assert model.build_condition_bundle.__name__ == "build_condition_bundle"


def test_c2_lesion_zero_modifies_c2_region():
    from src.mechanism_validation.feature_causality import c2_lesion_zero

    model = _make_model()
    batch = _single_batch()
    bundle = model.build_condition_bundle(batch, torch.tensor([0], dtype=torch.long))
    c2_orig = bundle.maps["ct_feat_1"].clone()

    new_bundle = c2_lesion_zero(bundle, batch)
    c2_new = new_bundle.maps["ct_feat_1"]

    # The c2 lesion region must be zeroed.
    mask_ds = F.interpolate(
        batch["mask"].float(), size=c2_orig.shape[2:], mode="nearest"
    )
    assert (c2_new[mask_ds.bool().expand_as(c2_new)] == 0.0).all()
    # The non-lesion region must be unchanged.
    assert torch.equal(c2_new[~mask_ds.bool().expand_as(c2_new)], c2_orig[~mask_ds.bool().expand_as(c2_orig)])


def test_interventions_are_non_destructive():
    from src.mechanism_validation.feature_causality import (
        c2_background_replace,
        c2_lesion_zero,
        c2_nonlesion_samearea,
        c2_shifted_mask,
        ct_inpaint,
    )

    model = _make_model()
    batch = _single_batch()
    bundle = model.build_condition_bundle(batch, torch.tensor([0], dtype=torch.long))
    c2_orig = bundle.maps["ct_feat_1"].clone()

    for fn in (
        c2_background_replace,
        c2_nonlesion_samearea,
        c2_shifted_mask,
        ct_inpaint,
    ):
        out = fn(bundle, batch)
        # The original bundle's c2 must be untouched (shallow copy semantics).
        assert torch.equal(bundle.maps["ct_feat_1"], c2_orig)


def test_nonlesion_samearea_is_exact_same_area_and_nonempty():
    from src.mechanism_validation.feature_causality import (
        _place_nonlesion_mask,
        c2_nonlesion_samearea,
    )

    model = _make_model()
    batch = _single_batch()
    bundle = model.build_condition_bundle(batch, torch.tensor([0], dtype=torch.long))
    c2 = bundle.maps["ct_feat_1"]
    mask_ds = F.interpolate(
        batch["mask"].float(), size=c2.shape[2:], mode="nearest"
    )
    lesion_area = float((mask_ds > 0.5).sum().item())

    # Same-area placement must exist and match lesion area.
    import numpy as np
    from src.mechanism_validation.feature_causality import _place_nonlesion_mask

    rng = np.random.default_rng(0)
    placed, coverage = _place_nonlesion_mask(mask_ds, rng)
    assert placed is not None, "no valid same-area non-lesion site"
    assert float((placed > 0.5).sum().item()) == lesion_area
    assert coverage["ablated_area"] == lesion_area
    assert coverage["tissue_coverage"] == 1.0  # no body mask → full coverage

    # c2_nonlesion_samearea must actually ablate a non-empty, non-overlapping region.
    out = c2_nonlesion_samearea(bundle, batch, seed=0)
    c2_new = out.maps["ct_feat_1"]
    assert not torch.equal(c2_new, c2), "nonlesion same-area should modify c2"
    ablated = (~torch.isclose(c2_new, c2)) & (c2_new == 0.0)
    assert ablated.sum().item() > 0, "no pixels ablated by nonlesion control"
    # Ablated region must NOT overlap the true lesion.
    ablated_ds = ablated.float()
    assert ((ablated_ds > 0.5) & (mask_ds > 0.5)).sum().item() == 0


def test_nonlesion_samearea_respects_body_mask():
    import numpy as np
    from src.mechanism_validation.feature_causality import (
        _place_nonlesion_mask,
        c2_nonlesion_samearea,
    )

    model = _make_model()
    batch = _single_batch()
    bundle = model.build_condition_bundle(batch, torch.tensor([0], dtype=torch.long))
    c2 = bundle.maps["ct_feat_1"]
    mask_ds = F.interpolate(
        batch["mask"].float(), size=c2.shape[2:], mode="nearest"
    )
    # Body mask that covers the whole grid EXCEPT a margin — forces the control
    # region to lie inside tissue.
    body = torch.ones_like(mask_ds)
    body[:, :, :2, :] = 0.0
    body[:, :, -2:, :] = 0.0

    rng = np.random.default_rng(0)
    placed, coverage = _place_nonlesion_mask(mask_ds, rng, body_mask=body)
    assert placed is not None, "no body-constrained same-area site"
    assert coverage["tissue_coverage"] == 1.0
    # Every ablated pixel must be inside the body mask.
    assert ((placed > 0.5) & (body <= 0.5)).sum().item() == 0


def test_shifted_mask_is_nonempty_on_small_grid():
    from src.mechanism_validation.feature_causality import c2_shifted_mask

    model = _make_model()
    batch = _single_batch()
    bundle = model.build_condition_bundle(batch, torch.tensor([0], dtype=torch.long))
    c2_orig = bundle.maps["ct_feat_1"].clone()

    out = c2_shifted_mask(bundle, batch, shift_fraction=0.25)
    c2_new = out.maps["ct_feat_1"]
    # On a 16x16 feature grid a 25% shift (4px) must produce a non-empty region.
    assert not torch.equal(c2_new, c2_orig), "shifted sham should modify c2"
    ablated = (c2_new == 0.0) & (c2_orig != 0.0)
    assert ablated.sum().item() > 0


def test_run_causality_audit_smoke(tmp_path):
    from src.mechanism_validation.feature_causality import run_causality_audit

    model = _make_model()
    batch = _single_batch()

    class _OneLoader:
        def __iter__(self):
            yield batch

    decision = run_causality_audit(
        model,
        _OneLoader(),
        "cpu",
        tmp_path,
        seeds=(0,),
        interventions=("baseline", "c2_lesion_zero", "c2_shifted_mask"),
        num_sampling_steps=2,
    )

    assert (tmp_path / "causal_metrics.csv").is_file()
    assert (tmp_path / "causal_patient_summary.csv").is_file()
    assert (tmp_path / "causal_decision.json").is_file()
    assert decision["m2_gate"]["c2_lesion_zero_worsens_topq"] in (True, False)
