"""Tests for the A0 feature-source provenance audit."""

from __future__ import annotations

import json

import torch


class _ToyWithEncoder(torch.nn.Module):
    def __init__(self, image_size=64):
        super().__init__()
        from src.model.slmf_bbdm import _CTEncoder

        self.ct_encoder = _CTEncoder(
            in_channels=1, base_ch=16, out_channels=(16, 32, 64, 64)
        )
        self.image_size = image_size


def _toy_config():
    return {
        "experiment": {"name": "prov_test", "seed": 42},
        "data": {
            "mode": "png",
            "image_size": 64,
            "cache_dir": "cache/tensors_main",
            "split_manifest": "main_data/split_manifest.csv",
            "cache_lineage": "cache/tensors_main/cache_lineage.json",
            "dataset_contract": "configs/dataset_contract_stage0a_v1.json",
            "augment": True,
            "required_keys": ["ct", "pet", "mask"],
        },
    }


def test_provenance_audit_writes_feature_provenance_json(tmp_path):
    from src.mechanism_validation.feature_provenance import audit_feature_provenance

    model = _ToyWithEncoder(image_size=64)
    model.eval()
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict()}, ckpt)

    payload = audit_feature_provenance(
        model, _toy_config(), ckpt, tmp_path
    )

    assert (tmp_path / "feature_provenance.json").is_file()
    assert payload["decision"]["dual_modal_encoder_claim"] is False
    assert payload["decision"]["representation_is_shared_weight_probe"] is True
    assert payload["checkpoint"]["sha256"]
    assert payload["config"]["sha256"]
    assert len(payload["feature_source"]["layers"]) == 4


def test_provenance_layer_shapes_follow_halving(tmp_path):
    from src.mechanism_validation.feature_provenance import audit_feature_provenance

    model = _ToyWithEncoder(image_size=64)
    model.eval()
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict()}, ckpt)

    payload = audit_feature_provenance(model, _toy_config(), ckpt, tmp_path)
    layers = payload["feature_source"]["layers"]
    # c1 full-res, c2 H/2, c3 H/4, c4 H/8
    assert layers[0]["spatial"] == [64, 64]
    assert layers[1]["spatial"] == [32, 32]
    assert layers[2]["spatial"] == [16, 16]
    assert layers[3]["spatial"] == [8, 8]
    assert layers[1]["tensor_key"] == "ct_feat_1"


def test_provenance_checkpoint_sha256_changes_with_content(tmp_path):
    from src.mechanism_validation.common import file_sha256
    from src.mechanism_validation.feature_provenance import audit_feature_provenance

    model = _ToyWithEncoder()
    model.eval()
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict()}, ckpt)
    sha_before = file_sha256(ckpt)

    # Mutate a weight, re-save.
    with torch.no_grad():
        model.ct_encoder.stem_fuse.weight.add_(0.5)
    torch.save({"model": model.state_dict()}, ckpt)
    sha_after = file_sha256(ckpt)

    assert sha_before != sha_after
    payload = audit_feature_provenance(model, _toy_config(), ckpt, tmp_path)
    assert payload["checkpoint"]["sha256"] == sha_after


def test_provenance_json_is_canonical_parseable(tmp_path):
    from src.mechanism_validation.feature_provenance import audit_feature_provenance

    model = _ToyWithEncoder()
    model.eval()
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict()}, ckpt)

    audit_feature_provenance(model, _toy_config(), ckpt, tmp_path)
    raw = (tmp_path / "feature_provenance.json").read_text(encoding="utf-8")
    json.loads(raw)  # must be valid JSON
    assert "shared-weight representation probe" in raw
