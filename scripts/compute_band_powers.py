"""Per-fold outer-train band powers P_g for the band_snr clock.

DESIGN_RC_BRD_clock_v2 §2.3 (clock math review correction #3): P_g must
be the MEAN PER-COEFFICIENT power of the group's Haar residual
coefficients (z0 side), estimated on the fold's outer-train split and
sealed before any RC-band training run — never the group TOTAL (the
Parseval 1:3:12/16 coefficient share would inflate the high group ~12x
and destroy the band-SNR semantics).

Evidence-chain position (all fail-closed):

    MeanNet pretrain (mean_best.pt)
        -> [this script] P_g on outer-train  (artifacts/<fold>/band_powers.json)
        -> R0 probe (run_recoverability_probe.py, PASS gate)
        -> freeze --band-powers-file ...     (v2 contract, band_powers sealed)
        -> rc_brd_prod_rc_band_snr.yaml      (clock_mode: band_snr)

The residual is computed exactly as training does it: z0 = PET −
mean_pet(ct) with the FROZEN conditional-mean checkpoint, then
haar_forward2 → per-group pooled mean of squared coefficients.
Leakage: only samples whose split-manifest row maps to the requested
split (default train = the fold's outer-train) are read; the manifest
is passed straight to CachedDataset (patient-level, strict).

CLI: --config <experiment yaml> --fold <fold_id> [--split train]
[--groups 3|7] [--out path] [--batch-size N] [--device auto|cpu|cuda].
Exit codes: 0 ok; 1 any validation/loading failure (never writes a
partial artifact).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset import CachedDataset  # noqa: E402
from src.model.config_utils import load_full_config  # noqa: E402
from src.model.mean_predictor import build_mean_predictor  # noqa: E402
from src.model.rc_brd import (  # noqa: E402
    BAND_NAMES,
    band_groups,
    compute_contract_sha256,
    haar_forward2,
    mean_weights_sha256,
)

# v1: original sealed schema.  v2 (provenance fix): adds the artifact
# self-hash 'artifact_sha256' so freeze_recoverability_contract.py can
# verify the payload was not edited after sealing (same canonical-JSON
# sha256 convention as the recoverability contract itself).
SCHEMA_VERSION = 2


class BandPowerAccumulator:
    """Pooled per-band / per-group mean squared Haar coefficients.

    P_g = (Σ_{b∈g} Σ c²) / (Σ_{b∈g} n_b) — the pooled MEAN PER-COEFFICIENT
    power of the group (not the mean of band means, not the group total).
    Level-2 bands share one grid so pooling equals averaging inside the
    mid group, but the estimator stays exact for any group layout.
    """

    def __init__(self, groups: Mapping[str, list[str]]):
        known = set(BAND_NAMES)
        covered = [b for bands in groups.values() for b in bands]
        if sorted(covered) != sorted(known):
            raise ValueError(
                f"band groups must cover every band exactly once; got {dict(groups)}")
        self._groups = {str(g): list(b) for g, b in groups.items()}
        self._sum_sq: dict[str, float] = {b: 0.0 for b in BAND_NAMES}
        self._count: dict[str, int] = {b: 0 for b in BAND_NAMES}

    def update(self, bands: Mapping[str, torch.Tensor]) -> None:
        """Accumulate one batch of Haar band tensors ([B,1,h,w])."""
        for band in BAND_NAMES:
            if band not in bands:
                raise ValueError(f"band dict is missing band {band!r}")
            coeff = bands[band].to(dtype=torch.float64)
            self._sum_sq[band] += float(coeff.square().sum().item())
            self._count[band] += int(coeff.numel())

    def band_powers(self) -> dict[str, float]:
        """Per-band mean per-coefficient power (diagnostic)."""
        return {b: (self._sum_sq[b] / self._count[b]) if self._count[b] else float("nan")
                for b in BAND_NAMES}

    def group_powers(self) -> dict[str, float]:
        """P_g: pooled mean per-coefficient power per group (§2.3)."""
        out: dict[str, float] = {}
        for group, bands in self._groups.items():
            total_sq = sum(self._sum_sq[b] for b in bands)
            total_n = sum(self._count[b] for b in bands)
            if total_n == 0:
                raise ValueError(f"group {group!r} received no coefficients")
            out[group] = total_sq / total_n
        return out


def load_mean_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """State dict from a mean checkpoint ({"model": ...} / {"state_dict": ...} / raw)."""
    obj = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(obj, Mapping) and isinstance(obj.get("model"), Mapping):
        obj = obj["model"]
    elif isinstance(obj, Mapping) and isinstance(obj.get("state_dict"), Mapping):
        obj = obj["state_dict"]
    if not isinstance(obj, Mapping) or not obj:
        raise ValueError(f"mean checkpoint {checkpoint_path} is not a non-empty state dict")
    return {str(k): v for k, v in obj.items()}


def compute_band_powers(config: dict[str, Any], fold: str, *, split: str = "train",
                        n_groups: int = 3, batch_size: int = 8,
                        device: str = "cpu") -> dict[str, Any]:
    """Core computation; returns the sealed-artifact payload (unwritten)."""
    mean_cfg = dict(config.get("modules", {}).get("conditional_mean", {}) or {})
    checkpoint = str(mean_cfg.get("checkpoint", "") or "")
    if not checkpoint:
        raise ValueError("modules.conditional_mean.checkpoint is required "
                         "(P_g is defined against the FROZEN mean; pretrain "
                         "the MeanNet first)")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"mean checkpoint not found: {checkpoint}")
    data_cfg = dict(config.get("data", {}) or {})
    cache_dir = str(data_cfg.get("cache_dir", "") or "")
    if not cache_dir:
        raise ValueError("data.cache_dir is required")
    manifest = data_cfg.get("split_manifest") or None
    dataset = CachedDataset(cache_dir, split=split, augment=False,
                            split_manifest=Path(manifest) if manifest else None,
                            required_keys=["ct", "pet"])
    if len(dataset) == 0:
        raise ValueError(f"split {split!r} has no cached samples under {cache_dir}")
    groups = band_groups(int(n_groups))
    predictor = build_mean_predictor(mean_cfg)
    state = load_mean_state(checkpoint_path)
    missing, unexpected = predictor.load_state_dict(state, strict=False)
    # strip common prefixes before judging (comparison-format generators)
    def _strip(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if missing:
        for prefix in ("generator.", "module.", "_orig_mod."):
            stripped = _strip(state, prefix)
            if stripped:
                missing, unexpected = predictor.load_state_dict(stripped, strict=False)
                break
    if missing or unexpected:
        raise ValueError(
            f"mean checkpoint does not match the configured architecture "
            f"(missing={sorted(missing)[:5]}, unexpected={sorted(unexpected)[:5]})")
    predictor.eval().to(device)
    accumulator = BandPowerAccumulator(groups)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    n_samples = 0
    with torch.no_grad():
        for batch in loader:
            ct = batch["ct"].to(device)
            pet = batch["pet"].to(device)
            mean_out = predictor(ct)
            mean_pet = mean_out["mean_pet"] if isinstance(mean_out, Mapping) else mean_out
            if mean_pet.shape != pet.shape:
                raise ValueError(
                    f"mean predictor output shape {tuple(mean_pet.shape)} != "
                    f"PET shape {tuple(pet.shape)}")
            residual = pet - mean_pet
            accumulator.update(haar_forward2(residual))
            n_samples += int(pet.shape[0])
    powers = accumulator.group_powers()
    bad = [g for g, p in powers.items()
           if not (math.isfinite(p) and p > 0.0)]
    if bad:
        raise ValueError(f"band powers must be finite and > 0 for {bad}; "
                         "the residual looks degenerate (all-zero mean gap?)")
    patients = sorted({e.patient_id for e in dataset.entries})
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "rc_brd_band_powers",
        "fold": str(fold),
        "split": str(split),
        "split_label": f"outer_train_{fold}" if split == "train" else f"{split}_{fold}",
        "n_samples": n_samples,
        "n_patients": len(patients),
        "image_size": int(data_cfg.get("image_size", 0) or 0),
        "band_groups": {g: list(b) for g, b in groups.items()},
        "band_powers": {g: float(powers[g]) for g in groups},
        "band_powers_by_band": accumulator.band_powers(),
        "mean_checkpoint": checkpoint,
        "mean_checkpoint_sha256": mean_weights_sha256(state),
        "definition": ("mean per-coefficient power of the Haar residual "
                        "coefficients z0 = PET - frozen_mean(ct) on the named "
                        "split (DESIGN_RC_BRD_clock_v2 §2.3; pooled per group, "
                        "never the group total)"),
    }
    # v2 provenance seal: canonical-JSON sha256 over the de-hashed payload
    # (same convention as contract.compute_contract_sha256).  freeze
    # verifies this hash before trusting fold/split/mean-SHA provenance.
    payload["artifact_sha256"] = compute_contract_sha256(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute and seal per-fold outer-train band powers P_g "
                    "(band_snr clock; DESIGN_RC_BRD_clock_v2 §2.3)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", required=True, help="fold id, e.g. fold_0")
    parser.add_argument("--split", default="train",
                        help="manifest split of the outer-train side (default train)")
    parser.add_argument("--groups", type=int, choices=(3, 7), default=3)
    parser.add_argument("--out", default=None,
                        help="output json (default artifacts/rc_brd/<fold>/band_powers.json)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args(argv)
    try:
        config = load_full_config(args.config)
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        payload = compute_band_powers(config, args.fold, split=args.split,
                                      n_groups=args.groups,
                                      batch_size=args.batch_size,
                                      device=device)
    except (OSError, ValueError, KeyError, TypeError, FileNotFoundError,
            json.JSONDecodeError, RuntimeError) as exc:
        print(f"band powers failed: {exc}", file=sys.stderr)
        return 1
    out_path = Path(args.out) if args.out else Path(
        f"artifacts/rc_brd/{args.fold}/band_powers.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                        encoding="utf-8")
    print(f"band powers fold={args.fold} split={args.split} "
          f"n_samples={payload['n_samples']} "
          f"n_patients={payload['n_patients']} -> {out_path}")
    for group, power in payload["band_powers"].items():
        print(f"  P_{group} = {power:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))