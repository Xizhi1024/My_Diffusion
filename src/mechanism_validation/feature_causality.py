"""A2 causal intervention audit for the c2 lesion feature.

RQ-E3  Does the model actually use the c2 lesion information?

Because ``build_condition_bundle`` is re-run at every DDIM denoising step
(``src/model/slmf_bbdm.py sample()``), interventions are applied by
monkey-patching the model's bundle builder with a wrapper that mutates the
c2 feature map (``maps["ct_feat_1"]``) every step.  The model code itself is
untouched.

Intervention matrix (all share the same sampling noise per seed):

  baseline              original bundle (no patch)
  c2_lesion_zero        c2 lesion region zeroed        — tests c2 lesion necessity
  c2_background_replace c2 lesion region filled with ring mean
  c2_nonlesion_samearea same-area *non-lesion* region replaced — spatial specificity
  c2_shifted_mask       c2 region under a shifted mask zeroed  — sham control
  ct_inpaint            raw-CT lesion inpainting (checks full CT lesion dependency)

All statistics are patient-level; slices are aggregated before bootstrap.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .common import bootstrap_mean, sign_flip_p, write_csv, write_json
from .feature_emergence import _patient_keys

# The c2 feature key in the condition bundle.
C2_KEY = "ct_feat_1"

# Pre-registered background non-inferiority margin: the c2 intervention's
# non-lesion TopQ error (unit-interval scale) may not exceed 0.05 above baseline.
_BACKGROUND_NONINFERIOR_MARGIN = 0.05

InterventionFn = Callable[[Any, Mapping[str, Any]], Any]


def _hash_seed(patient: str, sample_idx: int, seed: int) -> int:
    """Stable per-(patient, sample, seed) RNG seed so interventions share noise."""
    digest = hashlib.sha256(
        f"{patient}:{sample_idx}:{seed}".encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


class patch_build_condition_bundle:
    """Context manager: replace ``model.build_condition_bundle`` with a wrapper
    that applies ``intervention_fn(bundle, batch)`` after the original builds.

    Restores the original method on exit.  Safe to nest.
    """

    def __init__(self, model: torch.nn.Module, intervention_fn: InterventionFn):
        self.model = model
        self.intervention_fn = intervention_fn
        self.original = model.build_condition_bundle

    def __enter__(self):
        def patched(batch, timesteps):
            bundle = self.original(batch, timesteps)
            return self.intervention_fn(bundle, batch)

        self.model.build_condition_bundle = patched
        return self

    def __exit__(self, *exc):
        # Remove the instance attribute entirely so the attribute lookup
        # resolves back to the class-level method (a fresh bound method).
        try:
            delattr(self.model, "build_condition_bundle")
        except AttributeError:
            pass
        return False


def _c2(bundle, batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (c2 clone, mask downsampled to c2 spatial dims)."""
    c2 = bundle.maps[C2_KEY]
    mask = batch["mask"]
    if mask.ndim != c2.ndim:
        mask = mask.unsqueeze(1) if mask.ndim == 3 else mask
    mask_ds = F.interpolate(mask.float(), size=c2.shape[2:], mode="nearest")
    mask_ds = (mask_ds > 0.5).float()
    return c2, mask_ds


def c2_lesion_zero(bundle, batch) -> Any:
    c2, mask_ds = _c2(bundle, batch)
    c2 = c2 * (1.0 - mask_ds)
    new = bundle.copy()
    new.maps[C2_KEY] = c2
    return new


def _ring_mask(mask_ds: torch.Tensor, ring_width: int = 3) -> torch.Tensor:
    """Binary ring around the mask, on the feature-grid spatial dims."""
    from .feature_emergence import _dilate

    m = mask_ds.squeeze(0).squeeze(0) if mask_ds.dim() == 4 else mask_ds
    if m.dim() != 2:
        raise ValueError(f"ring mask expects 2D, got {m.shape}")
    dilated = _dilate(m.bool(), ring_width)
    ring = dilated & ~m.bool()
    return ring.float().unsqueeze(0).unsqueeze(0)


def c2_background_replace(bundle, batch, ring_width: int = 3) -> Any:
    """Fill the c2 lesion region with the mean of its surrounding ring."""
    c2, mask_ds = _c2(bundle, batch)
    ring = _ring_mask(mask_ds, ring_width=ring_width).to(c2.device)
    ring_sum = ring.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    ring_mean = (c2 * ring).sum(dim=(2, 3), keepdim=True) / ring_sum
    c2 = c2 * (1.0 - mask_ds) + ring_mean * mask_ds
    new = bundle.copy()
    new.maps[C2_KEY] = c2
    return new


def _body_mask_from_batch(batch, c2) -> torch.Tensor | None:
    """Derive a body/tissue mask at c2 resolution from the batch.

    Priority: ``organ_mask`` (any of the 6 organ channels) > ``mu_map`` > None.
    Returns a [1, 1, H, W] float mask on c2's device, or None when no usable
    tissue signal exists in the batch.
    """
    org = batch.get("organ_mask")
    if torch.is_tensor(org) and org.numel() > 0:
        m = org.float().amax(dim=1, keepdim=True)  # [B, 1, H, W]
        m = F.interpolate(m, size=c2.shape[2:], mode="nearest")
        return (m > 0.5).float().to(c2.device)
    mu = batch.get("mu_map")
    if torch.is_tensor(mu) and mu.numel() > 0:
        m = F.interpolate(mu.float(), size=c2.shape[2:], mode="nearest")
        return (m > 0.5).float().to(c2.device)
    return None


def _place_nonlesion_mask(
    mask_ds: torch.Tensor,
    rng,
    body_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Place a non-lesion control region with the SAME pixel area as the lesion.

    Finds all valid translation offsets of the lesion mask that land entirely
    inside the feature grid, do not overlap the true lesion, and — when
    ``body_mask`` is given — are fully covered by body/tissue.  Picks one
    deterministically from ``rng``.  The region area equals the lesion area
    exactly (it is a clipped translation).  Returns (region, coverage_stats),
    where region is None when no valid placement exists and coverage_stats
    records the requested vs actual ablated area and tissue coverage.
    """
    m = mask_ds.squeeze(0).squeeze(0)
    if m.dim() != 2:
        raise ValueError(f"mask must be 2D, got {m.shape}")
    H, W = m.shape
    coords = torch.nonzero(m.bool(), as_tuple=False)  # [N, 2]
    if coords.shape[0] == 0:
        return None, {"requested_area": 0.0, "ablated_area": 0.0, "tissue_coverage": 0.0}
    ys = coords[:, 0]
    xs = coords[:, 1]
    area = int(coords.shape[0])
    if area >= H * W:
        return None, {
            "requested_area": float(area),
            "ablated_area": 0.0,
            "tissue_coverage": 0.0,
        }
    min_y, max_y = int(ys.min()), int(ys.max())
    min_x, max_x = int(xs.min()), int(xs.max())
    h = max_y - min_y + 1
    w = max_x - min_x + 1
    body = body_mask.squeeze(0).squeeze(0) if body_mask is not None else None
    candidates: list[tuple[int, int]] = []
    for dy in range(H - h + 1):
        for dx in range(W - w + 1):
            if dy == min_y and dx == min_x:
                continue  # same location as the true lesion
            window = torch.zeros_like(m)
            window[dy : dy + h, dx : dx + w] = m[min_y : max_y + 1, min_x : max_x + 1]
            if window.sum().item() != area:
                continue  # would clip the region
            if ((window > 0.5) & (m > 0.5)).any():
                continue  # overlaps the true lesion
            if body is not None and (window > 0.5).any():
                # Require FULL tissue coverage: every ablated pixel lies in body.
                if ((window > 0.5) & (body > 0.5)).sum().item() != area:
                    continue
            candidates.append((dy, dx))
    if not candidates:
        return None, {
            "requested_area": float(area),
            "ablated_area": 0.0,
            "tissue_coverage": 0.0,
        }
    dy, dx = candidates[int(rng.integers(0, len(candidates)))]
    window = torch.zeros_like(m)
    window[dy : dy + h, dx : dx + w] = m[min_y : max_y + 1, min_x : max_x + 1]
    coverage = (
        float(((window > 0.5) & (body > 0.5)).sum().item()) / area
        if body is not None
        else 1.0
    )
    region = window.float().unsqueeze(0).unsqueeze(0)
    return region, {
        "requested_area": float(area),
        "ablated_area": float(area),
        "tissue_coverage": coverage,
    }


def c2_nonlesion_samearea(bundle, batch, seed: int = 0) -> Any:
    """Zero a *non-lesion* region of the SAME pixel area as the lesion.

    Spatial-specificity control: the ablated region is a translation of the
    lesion mask that does not overlap it, is fully contained in the feature
    grid, and (when a body/tissue mask is available) lies entirely within body
    tissue.  Area equals the lesion area exactly.  If no valid placement exists
    the intervention is a no-op.  Coverage stats are recorded in the bundle logs.
    """
    c2, mask_ds = _c2(bundle, batch)
    body = _body_mask_from_batch(batch, c2)
    rng = np.random.default_rng(seed)
    control, coverage = _place_nonlesion_mask(mask_ds, rng, body_mask=body)
    if control is None:
        new = bundle.copy()
        new.logs["nonlesion_samearea"] = coverage
        return new  # no valid same-area non-lesion site; no-op
    control = control.to(c2.device)
    c2 = c2 * (1.0 - control)
    new = bundle.copy()
    new.maps[C2_KEY] = c2
    new.logs["nonlesion_samearea"] = coverage
    return new


def c2_shifted_mask(bundle, batch, shift_fraction: float = 0.25) -> Any:
    """Zero the c2 region under a *shifted* (misaligned) mask — sham control.

    Same ablation as c2_lesion_zero but the zeroed region is offset by a
    fraction of the feature-grid size, so a large effect here would indicate the
    model is not spatially specific about where the lesion is.  The shift is
    grid-relative (not a fixed 32/64px) so it stays a true offset on any feature
    resolution.  Overlap with the true lesion is removed; if the shifted region
    becomes empty the intervention is a no-op.
    """
    c2, mask_ds = _c2(bundle, batch)
    H, W = mask_ds.shape[2], mask_ds.shape[3]
    shift = max(1, int(round(shift_fraction * max(H, W))))
    shifted = torch.roll(mask_ds, shifts=(shift, shift), dims=(2, 3))
    shifted = shifted * (1.0 - mask_ds)  # keep it truly non-overlapping with lesion
    if shifted.sum().item() <= 0.0:
        return bundle.copy()
    c2 = c2 * (1.0 - shifted)
    new = bundle.copy()
    new.maps[C2_KEY] = c2
    return new


def ct_inpaint(bundle, batch) -> Any:
    """Zero the raw-CT lesion region so the full CT lesion dependency is removed.

    NOTE: this is the *bundle*-level variant used by ``patch_build_condition_bundle``.
    It only patches ``bundle.maps["ct"]``, which the adapter concat consumes, but
    the UNet's raw-CT channel (``x_source``) and the re-extracted encoder
    features come from ``batch["ct"]`` — those are NOT reachable here.  For the
    full CT-lesion ablation you must use :func:`ct_inpaint_batch` instead, which
    modifies ``batch["ct"]`` before sampling.
    """
    mask = batch["mask"]
    if mask.ndim != bundle.maps["ct"].ndim:
        mask = mask.unsqueeze(1) if mask.ndim == 3 else mask
    ct = bundle.maps["ct"].clone()
    ct = ct * (1.0 - (mask > 0.5).float())
    new = bundle.copy()
    new.maps["ct"] = ct
    return new


def ct_inpaint_batch(sample: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a copy of ``sample`` with the lesion region of ``ct`` zeroed.

    Modifies ``sample["ct"]`` (not ``bundle.maps["ct"]``) so that BOTH the CT
    encoder (which re-extracts ``ct_feat_0..3``) and the UNet raw-CT channel
    (``x_source``) see an inpainted CT with no lesion signal.  This is the
    correct implementation of the "full CT lesion dependency" ablation.
    """
    ct = sample["ct"]
    mask = sample["mask"]
    if mask.ndim != ct.ndim:
        mask = mask.unsqueeze(1) if mask.ndim == 3 else mask
    ct_inpainted = ct * (1.0 - (mask > 0.5).float())
    return {**sample, "ct": ct_inpainted}


INTERVENTIONS: dict[str, Callable[..., Any]] = {
    "baseline": None,
    "c2_lesion_zero": c2_lesion_zero,
    "c2_background_replace": c2_background_replace,
    "c2_nonlesion_samearea": c2_nonlesion_samearea,
    "c2_shifted_mask": c2_shifted_mask,
    "ct_inpaint": ct_inpaint,
}

# Batch-level interventions modify the sample dict BEFORE sampling (and before
# the encoder re-extracts features), covering the full CT lesion dependency.
BATCH_INTERVENTIONS: dict[str, Callable[..., Any]] = {
    "ct_inpaint": ct_inpaint_batch,
}


def compute_sample_metrics(
    sample: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Wrapper around trainer._compute_pet_sample_metrics with same defaults.

    Returns the trainer metric dict (lesion TopQ/peak/centroid) PLUS the
    background/ non-lesion metrics needed for the M2 background
    non-inferiority gate:
      - ``nonlesion_mae``      : mean abs error over non-lesion pixels
      - ``nonlesion_topq_peak_error_norm``: TopQ error over the strongest
        non-lesion pixels (a proxy for false-hotspot strength)
    """
    from ..model.trainer import _compute_pet_sample_metrics
    from ..model.trainer import _to_unit_interval

    pred = sample["synthetic_pet"]
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    pred = np.squeeze(pred)
    target = np.squeeze(target)
    mask = np.squeeze(mask)

    metrics = _compute_pet_sample_metrics(pred, target, mask) or {}

    pred_u = _to_unit_interval(pred)
    target_u = _to_unit_interval(target)
    nonlesion = ~(mask > 0.5)
    if nonlesion.any():
        metrics["nonlesion_mae"] = float(
            np.abs(pred_u[nonlesion] - target_u[nonlesion]).mean()
        )
        # False-hotspot proxy: error on the top-quantile non-lesion pixels.
        n_topq = max(1, int(np.ceil(nonlesion.sum() * 0.10)))
        topq_vals = pred_u[nonlesion]
        topq_err = np.abs(
            np.partition(topq_vals, -n_topq)[-n_topq:]
            - target_u[nonlesion][np.argsort(topq_vals)[-n_topq:]]
        )
        metrics["nonlesion_topq_peak_error_norm"] = float(topq_err.mean())
    else:
        metrics["nonlesion_mae"] = float("nan")
        metrics["nonlesion_topq_peak_error_norm"] = float("nan")
    return metrics


def run_causality_audit(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device | str,
    output_dir: str | Path,
    *,
    seeds: Sequence[int] = (0, 1, 2, 3),
    interventions: Sequence[str] | None = None,
    max_samples: int | None = None,
    num_sampling_steps: int | None = None,
) -> dict[str, Any]:
    """Run the full intervention matrix on every sample × seed.

    For each sample, for each seed, for each intervention: patch the bundle
    builder, sample with a fresh noise draw seeded by ``(seed, sample_idx)``,
    compute lesion metrics, aggregate to patient level, and compare each
    intervention against baseline with a patient-level bootstrap.

    Writes ``causal_metrics.csv`` and ``causal_decision.json``.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_names = set(INTERVENTIONS) | set(BATCH_INTERVENTIONS)
    inter_list = list(interventions) if interventions else sorted(all_names)
    for name in inter_list:
        if name not in all_names:
            raise ValueError(f"Unknown intervention {name!r}")

    rows: list[dict[str, Any]] = []
    sample_idx = 0
    model.eval()

    for batch in loader:
        if max_samples is not None and sample_idx >= max_samples:
            break
        batch_gpu = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        patients = _patient_keys(batch_gpu)
        for b_idx, patient in enumerate(patients):
            if max_samples is not None and sample_idx >= max_samples:
                break
            sample = {
                k: (v[b_idx : b_idx + 1] if torch.is_tensor(v) else v) for k, v in batch_gpu.items()
            }
            target = sample["pet"]
            mask = sample["mask"]
            for seed in seeds:
                # Fixed per-(sample, seed) RNG state so every intervention sees
                # the SAME sampling noise.  Resetting inside the intervention
                # loop would let baseline's consumption bleed into c2_lesion_zero.
                base_seed = _hash_seed(patient, sample_idx, seed)
                for name in inter_list:
                    torch.manual_seed(base_seed)
                    np.random.seed(base_seed)
                    batch_fn = BATCH_INTERVENTIONS.get(name)
                    bundle_fn = INTERVENTIONS.get(name)
                    sample_in = batch_fn(sample) if batch_fn is not None else sample
                    if bundle_fn is None:
                        result = model.sample(sample_in, num_steps=num_sampling_steps)
                    else:
                        with patch_build_condition_bundle(model, bundle_fn):
                            result = model.sample(sample_in, num_steps=num_sampling_steps)
                    metrics = compute_sample_metrics(result, target, mask)
                    metrics.update(
                        {
                            "patient_id": patient,
                            "sample_idx": sample_idx,
                            "seed": seed,
                            "intervention": name,
                        }
                    )
                    rows.append(metrics)
            sample_idx += 1

    write_csv(out / "causal_metrics.csv", rows)

    # Patient-level aggregation: mean over slices/seeds for each intervention.
    by_patient: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_patient.setdefault((row["intervention"], row["patient_id"]), []).append(row)

    metric_names = [
        "lesion_topq_peak_error_norm",
        "lesion_peak_error_norm",
        "lesion_centroid_distance",
        "outside_inside_peak_ratio",
        "lesion_roi_l1",
        "nonlesion_mae",
        "nonlesion_topq_peak_error_norm",
    ]
    patient_level: list[dict[str, Any]] = []
    for (intervention, patient), sample_rows in sorted(by_patient.items()):
        record: dict[str, Any] = {
            "intervention": intervention,
            "patient_id": patient,
            "samples": len(sample_rows),
        }
        for metric in metric_names:
            values = np.asarray(
                [float(r[metric]) for r in sample_rows if metric in r], dtype=np.float64
            )
            values = values[np.isfinite(values)]
            record[metric] = float(values.mean()) if values.size else float("nan")
        patient_level.append(record)
    write_csv(out / "causal_patient_summary.csv", patient_level)

    # Compare each intervention vs baseline on the primary endpoint.
    baseline_patient = {
        r["patient_id"]: r for r in patient_level if r["intervention"] == "baseline"
    }
    comparisons: list[dict[str, Any]] = []
    for name in inter_list:
        if name == "baseline":
            continue
        # Lesion primary endpoint delta (higher = worse).
        diffs: list[float] = []
        # Background metrics delta (higher = worse; used for non-inferiority).
        bg_mae_diffs: list[float] = []
        bg_topq_diffs: list[float] = []
        for r in patient_level:
            if r["intervention"] != name:
                continue
            base = baseline_patient.get(r["patient_id"])
            if base is None:
                continue
            delta = float(r["lesion_topq_peak_error_norm"]) - float(
                base["lesion_topq_peak_error_norm"]
            )
            if np.isfinite(delta):
                diffs.append(delta)
            for key, sink in (
                ("nonlesion_mae", bg_mae_diffs),
                ("nonlesion_topq_peak_error_norm", bg_topq_diffs),
            ):
                d = float(r[key]) - float(base[key])
                if np.isfinite(d):
                    sink.append(d)
        ci = bootstrap_mean(diffs, seed=42)
        p = sign_flip_p(diffs, seed=42 + 1)
        bg_mae_ci = bootstrap_mean(bg_mae_diffs, seed=42 + 2)
        bg_topq_ci = bootstrap_mean(bg_topq_diffs, seed=42 + 3)
        comparisons.append(
            {
                "intervention": name,
                "n_patients": len(diffs),
                "topq_error_delta_mean": ci.get("estimate"),
                "ci95_low": ci.get("ci95_low"),
                "ci95_high": ci.get("ci95_high"),
                "sign_flip_p": p,
                # M2: does lesion TopQ error worsen (delta > 0) vs baseline?
                "worsens_lesion_topq": bool(
                    ci.get("ci95_low") is not None and ci["ci95_low"] > 0.0
                ),
                "nonlesion_mae_delta_mean": bg_mae_ci.get("estimate"),
                "nonlesion_mae_ci95_high": bg_mae_ci.get("ci95_high"),
                "nonlesion_topq_delta_mean": bg_topq_ci.get("estimate"),
                "nonlesion_topq_ci95_high": bg_topq_ci.get("ci95_high"),
            }
        )

    # M2 gate: c2 lesion zero must clearly worsen the lesion TopQ, and the
    # sham / nonlesion controls must be weaker.
    c2_zero = next((c for c in comparisons if c["intervention"] == "c2_lesion_zero"), None)
    sham = next((c for c in comparisons if c["intervention"] == "c2_shifted_mask"), None)
    nonlesion = next(
        (c for c in comparisons if c["intervention"] == "c2_nonlesion_samearea"), None
    )
    c2_effect = c2_zero.get("ci95_low") if c2_zero else None
    sham_effect = sham.get("ci95_high") if sham else None
    nonlesion_effect = nonlesion.get("ci95_high") if nonlesion else None
    c2_bg_mae = c2_zero.get("nonlesion_mae_ci95_high") if c2_zero else None
    c2_bg_topq = c2_zero.get("nonlesion_topq_ci95_high") if c2_zero else None

    c2_worsens = bool(c2_effect is not None and c2_effect > 0.0)
    sham_weaker = bool(
        c2_effect is not None
        and sham_effect is not None
        and c2_effect > sham_effect
    )
    nonlesion_weaker = bool(
        c2_effect is not None
        and nonlesion_effect is not None
        and c2_effect > nonlesion_effect
    )
    # Background non-inferiority: the c2 intervention must NOT make the
    # non-lesion region materially worse.  A pre-registered margin of 5% of the
    # baseline non-lesion TopQ error is used; when background metrics are
    # unavailable the gate cannot pass.
    background_noninferior = bool(
        c2_bg_mae is not None
        and c2_bg_topq is not None
        and c2_bg_topq <= _BACKGROUND_NONINFERIOR_MARGIN
    )
    # M2 is a CONJUNCTION: c2-zero must worsen lesion TopQ AND both controls
    # must be weaker AND background must not degrade beyond the margin.  When a
    # required control or background metric is absent the run cannot pass.
    m2_passed = bool(
        c2_worsens and sham_weaker and nonlesion_weaker and background_noninferior
    )

    decision: dict[str, Any] = {
        "stage": "A2_causal",
        "primary_endpoint": "lesion_topq_peak_error_norm",
        "background_noninferiority_margin": _BACKGROUND_NONINFERIOR_MARGIN,
        "intervention_comparisons": comparisons,
        "m2_gate": {
            "c2_lesion_zero_worsens_topq": c2_worsens,
            "c2_effect_ci95_low": c2_effect,
            "sham_effect_ci95_high": sham_effect,
            "nonlesion_effect_ci95_high": nonlesion_effect,
            "sham_weaker_than_c2": sham_weaker,
            "nonlesion_weaker_than_c2": nonlesion_weaker,
            "background_noninferiority_checked": background_noninferior,
            "background_nonlesion_topq_ci95_high": c2_bg_topq,
            "background_nonlesion_mae_ci95_high": c2_bg_mae,
            "passed": m2_passed,
        },
    }
    write_json(out / "causal_decision.json", decision)
    return decision
