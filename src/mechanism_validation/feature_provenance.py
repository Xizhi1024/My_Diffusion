"""A0: Feature-source provenance audit for the lesion-feature emergence study.

The study (阶段 A) answers three questions on a trained checkpoint:
  RQ-E1  Are the shallow CT/PET features a shared cross-modal anatomy?
  RQ-E2  Does lesion information emerge in the CT mid-layer features?
  RQ-E3  Does the model actually use the c2 lesion information?

Before any of those numbers can be interpreted, we must pin down *where* the
features actually come from: which encoder, whether weights are shared, which
checkpoint (with SHA256), what input normalisation, the per-layer tensor
names/shapes, and — if the features come from the diffusion U-Net rather than a
plain CT encoder — which timestep, noise and seed were fixed.

A hard wording rule: feeding the *same* trained CT encoder the PET image and
calling the result a "native bimodal encoder" is forbidden.  The audit must
describe that as a ``shared-weight representation probe``.

The audit is purely static (no forward pass), so it can run on CPU and is cheap.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

from .common import canonical_json_sha256, file_sha256, write_json


def _layer_shapes(model: torch.nn.Module) -> list[dict[str, Any]]:
    """Describe the multi-scale feature maps produced by the CT encoder.

    Does not run a forward pass: the _CTEncoder.forward return shapes are a
    structural contract (conv layer downsampling), so we report the declared
    output channels plus the resolution relationship to the input image size.
    """

    encoder = getattr(model, "ct_encoder", None)
    if encoder is None:
        return []
    out_channels = []
    for block in ("stem_fuse", "down1", "down2", "down3"):
        module = getattr(encoder, block, None)
        if module is None:
            continue
        if hasattr(module, "out_channels"):
            out_channels.append(int(module.out_channels))
            continue
        # nn.Sequential blocks: the final conv sets the block output channels.
        for layer in reversed(module):
            if isinstance(layer, torch.nn.Conv2d):
                out_channels.append(int(layer.out_channels))
                break
    image_size = int(getattr(model, "image_size", 192))
    levels = []
    for i, channels in enumerate(out_channels):
        scale = 2 ** i  # down1..down3 stride 2; stem_fuse is full-res
        levels.append(
            {
                "name": f"c{i + 1}",
                "tensor_key": f"ct_feat_{i}",
                "channels": channels,
                "spatial": [image_size // scale, image_size // scale],
                "downsample_scale": scale,
            }
        )
    return levels


def _encoder_parameters(model: torch.nn.Module) -> dict[str, Any]:
    encoder = getattr(model, "ct_encoder", None)
    if encoder is None:
        return {}
    total = sum(p.numel() for p in encoder.parameters())
    return {
        "class_name": type(encoder).__name__,
        "parameters": int(total),
        "trainable": int(sum(p.numel() for p in encoder.parameters() if p.requires_grad)),
    }


def _input_normalisation(config: Mapping[str, Any]) -> dict[str, Any]:
    data_cfg = config.get("data", {}) if isinstance(config, Mapping) else {}
    mode = str(data_cfg.get("mode", "png"))
    return {
        "mode": mode,
        "image_size": int(data_cfg.get("image_size", 192)),
        "cache_dir": str(data_cfg.get("cache_dir", "")),
        "split_manifest": str(data_cfg.get("split_manifest", "")),
        "augment": bool(data_cfg.get("augment", True)),
        "required_keys": list(data_cfg.get("required_keys", [])),
        # NOTE: the pixel-domain normalisation is applied at cache build time
        # (see src/data/png_cache.py --pet-invert) and is recorded in the
        # cache lineage, not in this config.  We surface the config pointers so
        # the analyst can resolve the actual values from the lineage.
        "cache_lineage": str(data_cfg.get("cache_lineage", "")),
        "dataset_contract": str(data_cfg.get("dataset_contract", "")),
    }


def audit_feature_provenance(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Audit the *source* of CT/PET feature maps and persist ``feature_provenance.json``.

    Returns the provenance payload (also written to ``output_dir/feature_provenance.json``).
    """

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint_sha256 = file_sha256(ckpt_path)
    config_sha256 = canonical_json_sha256(config)

    shared_encoder = True  # the CT encoder is the only feature producer here
    provenance: dict[str, Any] = {
        "schema_version": "lesion_feature_emergence_v1",
        "audit_stage": "A0",
        "feature_source": {
            "encoder": _encoder_parameters(model),
            "shared_weight": shared_encoder,
            "probe_kind": "shared-weight representation probe",
            "note": (
                "Features are extracted from the single trained CT encoder. "
                "When the PET image is run through the *same* encoder for the "
                "cross-modal similarity probe (RQ-E1), the result is a "
                "shared-weight representation probe — NOT a native bimodal "
                "encoder, and NOT part of the trained model's forward path."
            ),
            "layers": _layer_shapes(model),
            "image_size": int(getattr(model, "image_size", 192)),
        },
        "checkpoint": {
            "path": ckpt_path.as_posix(),
            "sha256": checkpoint_sha256,
            "weights_are_ema": None,  # resolved by the loader, recorded by the CLI
        },
        "config": {
            "sha256": config_sha256,
            "input_normalisation": _input_normalisation(config),
        },
        "timestep_policy": {
            "sampling": (
                "build_condition_bundle(batch, t_batch) is re-run at every DDIM "
                "denoising step (src/model/slmf_bbdm.py sample()). The CT "
                "encoder features (ct_feat_0..3) do NOT depend on timestep; "
                "they are recomputed identically each step. Priors that DO "
                "depend on timestep (e.g. hotspot) are outside the scope of "
                "the A1/A2 encoder audit."
            ),
            "fixed_t_for_extraction": None,  # set by the CLI when running A1/A2
        },
        "decision": {
            "dual_modal_encoder_claim": False,
            "representation_is_shared_weight_probe": True,
        },
    }

    write_json(Path(output_dir) / "feature_provenance.json", provenance)
    return provenance


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
