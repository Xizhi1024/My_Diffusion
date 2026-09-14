"""A1 + A2: Cross-modal similarity and lesion-emergence analysis.

RQ-E1  Are the shallow CT/PET features shared cross-modal anatomy?
       Measured as per-layer linear CKA between the paired CT and PET
       representations of the *same trained CT encoder* (a shared-weight
       representation probe), contrasted against a patient-shuffled pairing.

RQ-E2  Does lesion information emerge in the CT mid-layer features?
       Measured via a channel-agnostic activation map A_l(x, y) per layer,
       then lesion-to-ring log contrast, lesion/background AUROC and AUPRC,
       and lesion-centroid hit rate, stratified by lesion area quartiles
       (quartiles frozen from the training set).

The statistical unit is always the *patient*: per-sample metrics are first
aggregated to a per-patient mean before any bootstrap/permutation test.  Slices
are never treated as independent samples.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import (
    bootstrap_mean,
    sign_flip_p,
    write_csv,
    write_json,
)

FEATURE_KEYS = ("ct_feat_0", "ct_feat_1", "ct_feat_2", "ct_feat_3")


# ---------------------------------------------------------------------------
# RQ-E1: cross-modal similarity (linear CKA)
# ---------------------------------------------------------------------------


def compute_linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear Centered Kernel Alignment between two activation matrices.

    ``X``/``Y`` are ``[N, D]`` (N pooled samples, D features).  Returns a scalar
    in [0, 1].  Centring is done on both Gram matrices.
    """

    def center(K: torch.Tensor) -> torch.Tensor:
        n = K.shape[0]
        eye = torch.eye(n, dtype=K.dtype, device=K.device) - (1.0 / n)
        return eye @ K @ eye

    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    n = X.shape[0]
    if n < 2:
        # Centring with a single sample produces a zero Gram matrix — undefined.
        return float("nan")
    Kx = center(Xc @ Xc.t())
    Ky = center(Yc @ Yc.t())
    hsic = (Kx * Ky).sum()
    denom = (Kx * Kx).sum().sqrt() * (Ky * Ky).sum().sqrt()
    if denom.item() <= 0.0 or not torch.isfinite(denom):
        return float("nan")
    return float((hsic / denom).clamp(0.0, 1.0).item())


def _pool_features(feat: torch.Tensor) -> torch.Tensor:
    """Global average pooling -> [N, C].  ``feat`` is [N, C, H, W]."""
    return feat.mean(dim=(2, 3))


def _patient_pooled_repr(patient_store: Mapping[str, list[torch.Tensor]], key: str) -> torch.Tensor:
    """Fixed-dimension per-patient representation: mean of pooled slice vectors."""
    vecs = patient_store[key]
    return torch.mean(torch.cat(vecs, dim=0), dim=0)


def extract_ct_encoder_features(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Run the *shared-weight* CT encoder on both CT and PET.

    This is a representation probe: the trained CT encoder is applied to the
    PET image purely for analysis.  It is NOT part of the model's forward path
    and must be described as a shared-weight probe, never as a native bimodal
    encoder.
    """
    device = next(model.parameters()).device
    ct = batch["ct"].to(device)
    pet = batch["pet"].to(device)
    ct_feats = model.ct_encoder(ct)
    pet_feats = model.ct_encoder(pet)
    result: dict[str, torch.Tensor] = {}
    for i, (cf, pf) in enumerate(zip(ct_feats, pet_feats)):
        result[f"ct_feat_{i}"] = cf
        result[f"pet_feat_{i}"] = pf
    return result


def _patient_keys(batch: Mapping[str, Any]) -> list[str]:
    """Extract per-sample patient IDs from a batch.

    Handles three collation shapes:
      1. ``meta`` as ``list[dict]`` — each dict is one sample (list collate).
      2. ``meta`` as ``dict[list]`` — PyTorch default collate: one list per
         field, all fields equal length (batch size).
      3. no ``meta`` / no ``patient_id`` — RAISES.  Fallback positional IDs
         (``patient_0``, ``patient_1``) would silently collapse a whole batch
         into one pseudo-patient and poison the patient-level statistics, so
         they are forbidden: a formal run must fail rather than emit garbage.
    """
    meta = batch.get("meta")
    if isinstance(meta, list) and meta:
        ids = []
        for i, m in enumerate(meta):
            if isinstance(m, Mapping):
                pid = m.get("patient_id")
            else:
                pid = m
            if pid is None or str(pid).strip() == "":
                raise ValueError(
                    f"Sample {i} in batch has no meta.patient_id; refusing to "
                    f"fall back to a positional patient id (would corrupt "
                    f"patient-level statistics)."
                )
            ids.append(str(pid))
        return ids
    if isinstance(meta, Mapping):
        ids = meta.get("patient_id")
        if isinstance(ids, (list, tuple)):
            if any(str(pid).strip() == "" for pid in ids):
                raise ValueError("meta.patient_id contains empty entries")
            return [str(pid) for pid in ids]
        if isinstance(ids, str) and ids.strip():
            return [ids] * _batch_size(batch)
        raise ValueError(
            "Batch has meta but no usable meta.patient_id; refusing to fall back "
            "to positional patient ids (would corrupt patient-level statistics)."
        )
    raise ValueError(
        "Batch has no meta with patient_id; refusing to fall back to positional "
        "patient ids (would corrupt patient-level statistics)."
    )


def _batch_size(batch: Mapping[str, Any]) -> int:
    for value in batch.values():
        if torch.is_tensor(value) and value.dim() >= 2:
            return int(value.shape[0])
    return 1


def cross_modal_similarity_analysis(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device | str,
    output_dir: str | Path,
    *,
    seed: int = 42,
    num_layers: int | None = None,
    n_permutations: int = 200,
) -> dict[str, Any]:
    """Paired vs patient-shuffled vs random-init linear CKA.

    Statistic design (audit-mandated):
      1. Each patient is reduced to a *fixed-dimension* representation per
         layer per modality (mean of that patient's pooled slice vectors).
      2. A single overall paired CKA is computed across ALL patients at once.
         This avoids the degenerate N<=2 per-patient CKA that saturates to 1.
      3. The paired CKA is bootstrapped at the *patient* level (resample
         patients with replacement, recompute CKA).
      4. The shuffled null is built by full patient-label permutation (CT of
         patient i paired with PET of patient j != i), producing a null
         distribution under which "shared anatomy" would be a false positive.

    Outputs ``layer_metrics.csv`` and ``patient_summary.csv``.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    device = torch.device(device)

    # Derive the encoder's actual layer count from a probe forward pass on the
    # model's own device (a CPU probe would raise a device mismatch on GPU).
    if num_layers is None:
        probe = torch.zeros(1, 1, 16, 16, device=device)
        n_feats = len(model.ct_encoder(probe))
        num_layers = n_feats

    cohort: dict[str, dict[str, list[torch.Tensor]]] = {}
    all_patients: list[str] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_gpu = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            feats = extract_ct_encoder_features(model, batch_gpu)
            patients = _patient_keys(batch_gpu)
            for idx, patient in enumerate(patients):
                if patient not in cohort:
                    cohort[patient] = {f"ct_feat_{i}": [] for i in range(num_layers)}
                    cohort[patient].update(
                        {f"pet_feat_{i}": [] for i in range(num_layers)}
                    )
                    all_patients.append(patient)
                for i in range(num_layers):
                    cohort[patient][f"ct_feat_{i}"].append(
                        _pool_features(feats[f"ct_feat_{i}"][idx : idx + 1])
                    )
                    cohort[patient][f"pet_feat_{i}"].append(
                        _pool_features(feats[f"pet_feat_{i}"][idx : idx + 1])
                    )

    # Random-init encoder baseline: run a fresh (untrained) CT encoder on both
    # CT and PET, and measure the same paired CKA.  This sets the *floor* that a
    # random encoder produces for shared-weight cross-modal similarity.
    random_init_model = _random_init_encoder(model)
    random_init_feats: dict[int, dict[str, list[tuple[str, torch.Tensor]]]] = {
        i: {} for i in range(num_layers)
    }
    with torch.no_grad():
        for batch in loader:
            ct = batch["ct"].to(device)
            pet = batch["pet"].to(device)
            patients = _patient_keys(batch)
            ct_feats = random_init_model(ct)
            pet_feats = random_init_model(pet)
            for i in range(num_layers):
                for idx, patient in enumerate(patients):
                    bucket = random_init_feats[i].setdefault(patient, [])
                    bucket.append(
                        (
                            _pool_features(ct_feats[i][idx : idx + 1]),
                            _pool_features(pet_feats[i][idx : idx + 1]),
                        )
                    )

    layer_rows: list[dict[str, Any]] = []
    patient_rows: list[dict[str, Any]] = []

    for i in range(num_layers):
        # ---- Per-patient fixed-dimension representation (mean of slices). ----
        ct_repr = {p: _patient_pooled_repr(cohort[p], f"ct_feat_{i}") for p in all_patients}
        pet_repr = {p: _patient_pooled_repr(cohort[p], f"pet_feat_{i}") for p in all_patients}

        # Random-init patient representations.
        ri_ct_repr: dict[str, torch.Tensor] = {}
        ri_pet_repr: dict[str, torch.Tensor] = {}
        for patient, pairs in random_init_feats[i].items():
            ct_vecs = torch.cat([p[0] for p in pairs], dim=0)
            pet_vecs = torch.cat([p[1] for p in pairs], dim=0)
            ri_ct_repr[patient] = torch.mean(ct_vecs, dim=0)
            ri_pet_repr[patient] = torch.mean(pet_vecs, dim=0)

        patients = [p for p in all_patients if p in pet_repr]
        if len(patients) < 3:
            # Not enough patients for a stable CKA/bootstrap/permutation.
            layer_rows.append(
                {
                    "layer": f"c{i + 1}",
                    "feature_key": FEATURE_KEYS[i],
                    "paired_cka_mean": float("nan"),
                    "shuffled_cka_mean": float("nan"),
                    "random_init_cka_mean": float("nan"),
                    "delta_cka_mean": float("nan"),
                    "paired_cka_ci95_low": float("nan"),
                    "paired_cka_ci95_high": float("nan"),
                    "delta_cka_ci95_low": float("nan"),
                    "delta_cka_ci95_high": float("nan"),
                    "permutation_p": float("nan"),
                    "n_patients": len(patients),
                    "n_permutations": 0,
                }
            )
            continue

        X = torch.stack([ct_repr[p] for p in patients])   # [n_patients, C]
        Y = torch.stack([pet_repr[p] for p in patients])  # [n_patients, C]

        paired = compute_linear_cka(X, Y)

        # ---- Patient-level bootstrap of the paired CKA. ----
        rng_boot = np.random.default_rng(seed + 1)
        boot_vals: list[float] = []
        for _ in range(1000):
            idx = rng_boot.integers(0, len(patients), size=len(patients))
            Xb = X[idx]
            Yb = Y[idx]
            boot_vals.append(compute_linear_cka(Xb, Yb))
        boot_vals = [v for v in boot_vals if np.isfinite(v)]
        ci_low, ci_high = (
            (float(np.quantile(boot_vals, 0.025)), float(np.quantile(boot_vals, 0.975)))
            if boot_vals
            else (float("nan"), float("nan"))
        )

        # ---- Shuffled null via full patient-label permutation. ----
        rng_perm = np.random.default_rng(seed + 2)
        perm_vals: list[float] = []
        for _ in range(n_permutations):
            perm = rng_perm.permutation(len(patients))
            # Guarantee no fixed point: permute until no index stays in place.
            while any(perm[j] == j for j in range(len(perm))):
                perm = rng_perm.permutation(len(patients))
            Yp = Y[perm]
            perm_vals.append(compute_linear_cka(X, Yp))
        perm_vals = [v for v in perm_vals if np.isfinite(v)]
        shuffled_mean = float(np.nanmean(perm_vals)) if perm_vals else float("nan")
        # One-sided permutation p: fraction of null CKA >= paired CKA.
        perm_p = (
            (sum(1 for v in perm_vals if v >= paired) + 1) / (len(perm_vals) + 1)
            if perm_vals
            else float("nan")
        )

        # ---- Patient-level bootstrap of the PAIRED − SHUFFLED delta. ----
        # The delta is the quantity that supports "shared anatomy": on each
        # bootstrap draw we resample patients, recompute paired CKA, then
        # subtract the shuffled CKA on the SAME draw (same matched pairs), so
        # the delta distribution is a proper paired contrast.
        rng_delta = np.random.default_rng(seed + 3)
        delta_boot: list[float] = []
        for _ in range(1000):
            idx = rng_delta.integers(0, len(patients), size=len(patients))
            perm = rng_delta.permutation(len(patients))
            while any(perm[j] == j for j in range(len(perm))):
                perm = rng_delta.permutation(len(patients))
            paired_b = compute_linear_cka(X[idx], Y[idx])
            shuffled_b = compute_linear_cka(X[idx], Y[idx][perm])
            if np.isfinite(paired_b) and np.isfinite(shuffled_b):
                delta_boot.append(paired_b - shuffled_b)
        delta_ci_low, delta_ci_high = (
            (float(np.quantile(delta_boot, 0.025)), float(np.quantile(delta_boot, 0.975)))
            if delta_boot
            else (float("nan"), float("nan"))
        )

        # ---- Random-init paired CKA + bootstrap. ----
        ri_patients = [p for p in patients if p in ri_pet_repr]
        ri_paired = float("nan")
        if len(ri_patients) >= 3:
            Xr = torch.stack([ri_ct_repr[p] for p in ri_patients])
            Yr = torch.stack([ri_pet_repr[p] for p in ri_patients])
            ri_paired = compute_linear_cka(Xr, Yr)

        delta = paired - shuffled_mean if np.isfinite(shuffled_mean) else float("nan")

        layer_rows.append(
            {
                "layer": f"c{i + 1}",
                "feature_key": FEATURE_KEYS[i],
                "paired_cka_mean": paired,
                "shuffled_cka_mean": shuffled_mean,
                "random_init_cka_mean": ri_paired,
                "delta_cka_mean": delta,
                "paired_cka_ci95_low": ci_low,
                "paired_cka_ci95_high": ci_high,
                "delta_cka_ci95_low": delta_ci_low,
                "delta_cka_ci95_high": delta_ci_high,
                "permutation_p": perm_p,
                "n_patients": len(patients),
                "n_permutations": n_permutations,
            }
        )
        for patient in patients:
            patient_rows.append(
                {
                    "layer": f"c{i + 1}",
                    "patient_id": patient,
                    "paired_cka_probe": None,  # patient rows are fixed-dim reprs; per-layer stat is cohort-level
                }
            )

    write_csv(out / "layer_metrics.csv", layer_rows)
    write_csv(out / "patient_summary.csv", patient_rows)

    # S1 gate on the shallow layer (c1): the PAIRED − SHUFFLED delta must have
    # a patient-level bootstrap CI lower bound above zero, a permutation p < 0.05,
    # and the paired CKA must exceed the random-init encoder baseline.  Using the
    # delta (not raw paired CKA) avoids the weak "paired CI > 0" criterion.
    c1_row = next((r for r in layer_rows if r["layer"] == "c1"), None)
    s1_gate = {
        "shallow_layer": "c1",
        "paired_cka": c1_row["paired_cka_mean"] if c1_row else float("nan"),
        "shuffled_null_mean": c1_row["shuffled_cka_mean"] if c1_row else float("nan"),
        "delta_cka_mean": c1_row["delta_cka_mean"] if c1_row else float("nan"),
        "delta_cka_ci95_low": c1_row["delta_cka_ci95_low"] if c1_row else float("nan"),
        "delta_cka_ci95_high": c1_row["delta_cka_ci95_high"] if c1_row else float("nan"),
        "permutation_p": c1_row["permutation_p"] if c1_row else float("nan"),
        "random_init_cka": c1_row["random_init_cka_mean"] if c1_row else float("nan"),
        "n_patients": c1_row["n_patients"] if c1_row else 0,
    }
    s1_delta_low = s1_gate["delta_cka_ci95_low"]
    s1_p = s1_gate["permutation_p"]
    s1_paired = s1_gate["paired_cka"]
    s1_ri = s1_gate["random_init_cka"]
    s1_gate["passed"] = bool(
        np.isfinite(s1_delta_low)
        and np.isfinite(s1_p)
        and s1_delta_low > 0.0
        and s1_p < 0.05
        and np.isfinite(s1_paired)
        and np.isfinite(s1_ri)
        and s1_paired > s1_ri
    )

    result = {
        "stage": "A1",
        "num_layers": num_layers,
        "metric": "linear_cka_shared_weight_probe",
        "layer_rows": layer_rows,
        "patient_rows": patient_rows,
        "s1_gate": s1_gate,
    }
    write_json(out / "cross_modal_similarity.json", result)
    return result


def _random_init_encoder(model: torch.nn.Module) -> torch.nn.Module:
    """Build a *fresh untrained* CT encoder for the random-init baseline.

    Mirrors the trained encoder's architecture (same class) but re-initialises
    every parameter in place, so the baseline has identical structure and has
    never seen the data.
    """
    import copy

    trained = getattr(model, "ct_encoder", None)
    if trained is None:
        raise ValueError("model has no ct_encoder to mirror")
    clone = copy.deepcopy(trained)
    for module in clone.modules():
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_uniform_(module.weight, a=5 ** 0.5)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    return clone


# ---------------------------------------------------------------------------
# RQ-E2: lesion emergence (channel-agnostic activation)
# ---------------------------------------------------------------------------


def channel_agnostic_activation(feat: torch.Tensor) -> torch.Tensor:
    """Channel-agnostic activation map.

    z-scores each channel first, then RMS-aggregates across channels:
        A_l(x, y) = sqrt(mean_c z(F_{l,c}(x, y))^2)
    where z is the per-channel standardisation (so channels with large raw
    magnitude do not dominate the map and layer magnitudes are comparable).
    ``feat`` is [N, C, H, W]; returns [N, 1, H, W] float32.
    """
    mean = feat.mean(dim=(2, 3), keepdim=True)
    std = feat.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    z = (feat - mean) / std
    return z.square().mean(dim=1, keepdim=True).sqrt()


def lesion_ring_contrast(
    activation: torch.Tensor,
    mask: torch.Tensor,
    ring_width: int = 3,
) -> float:
    """Log contrast between mean lesion activation and mean surrounding ring.

    The ring is a dilation of the mask by ``ring_width`` pixels minus the mask
    itself.  Returns ``log(mean_lesion / mean_ring)``; NaN when the ring is
    empty or the lesion is empty.
    """
    act = activation.squeeze()
    m = mask.squeeze() > 0.5
    if act.ndim != 2:
        raise ValueError(f"activation must be 2D after squeeze, got {act.shape}")
    if not m.any():
        return float("nan")
    lesion_vals = act[m].float()
    dilated = _dilate(m, ring_width)
    ring = dilated & ~m
    if not ring.any():
        return float("nan")
    ring_vals = act[ring].float()
    lesion_mean = lesion_vals.mean().item()
    ring_mean = ring_vals.mean().item()
    if lesion_mean <= 0.0 or ring_mean <= 0.0:
        return float("nan")
    return float(np.log(lesion_mean / ring_mean))


def _dilate(mask: torch.Tensor, iterations: int) -> torch.Tensor:
    """Binary dilation via max pooling.  ``mask`` is a bool 2D tensor."""
    from torch.nn.functional import max_pool2d

    x = mask.float().unsqueeze(0).unsqueeze(0)
    for _ in range(iterations):
        x = max_pool2d(x, kernel_size=3, stride=1, padding=1)
    return x.squeeze(0).squeeze(0).bool()


def lesion_discriminability(
    activation: torch.Tensor,
    mask: torch.Tensor,
    body_mask: torch.Tensor | None = None,
    ring_width: int = 6,
) -> dict[str, float]:
    """AUROC / AUPRC for lesion pixels vs a *body* background.

    Restricting the negative class to the surrounding body tissue (the dilated
    ring around the lesion, within the body) avoids a trivial separation driven
    by the zero-padding outside the patient body.  Pass ``body_mask`` (full-res
    body/tissue mask) to restrict the ring to tissue; otherwise the ring is used
    directly.
    """
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    act = activation.squeeze()
    m = mask.squeeze() > 0.5
    if body_mask is None:
        dilated = _dilate(m, ring_width)
        ring = dilated & ~m
    else:
        body = body_mask.squeeze() > 0.5
        dilated = _dilate(m, ring_width)
        ring = dilated & ~m & body
    if not m.any() or not ring.any():
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "lesion_pixels": int(m.sum().item()),
            "background_pixels": int(ring.sum().item()),
        }

    act_flat = act.detach().cpu().numpy().reshape(-1)
    labels = np.zeros(act_flat.shape[0], dtype=np.int64)
    labels[m.reshape(-1)] = 1
    # Only the ring/body pixels are the negative class; ignore padding elsewhere.
    neg = ring.reshape(-1)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(neg)[0]
    keep_idx = np.concatenate([pos_idx, neg_idx])
    if keep_idx.size == 0 or len(pos_idx) == 0 or len(neg_idx) == 0:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "lesion_pixels": len(pos_idx),
            "background_pixels": len(neg_idx),
        }
    y = labels[keep_idx]
    scores = act_flat[keep_idx]
    fpr, tpr, _ = roc_curve(y, scores)
    precision, recall, _ = precision_recall_curve(y, scores)
    return {
        "auroc": float(auc(fpr, tpr)),
        "auprc": float(auc(recall, precision)),
        "lesion_pixels": int(len(pos_idx)),
        "background_pixels": int(len(neg_idx)),
    }


def lesion_centroid_hit(
    activation: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Fraction of activation-mass centroid inside the lesion mask."""
    act = activation.squeeze().detach().cpu().numpy()
    m = mask.squeeze().detach().cpu().numpy() > 0.5
    if act.shape != m.shape:
        raise ValueError(f"activation {act.shape} and mask {m.shape} shape mismatch")
    total = act.sum()
    if total <= 0.0:
        return {"centroid_hit": float("nan")}
    ys, xs = np.indices(act.shape)
    cy = float((ys * act).sum() / total)
    cx = float((xs * act).sum() / total)
    cyi = int(round(cy))
    cxi = int(round(cx))
    cyi = min(max(cyi, 0), m.shape[0] - 1)
    cxi = min(max(cxi, 0), m.shape[1] - 1)
    return {"centroid_hit": float(m[cyi, cxi])}


def lesion_emergence_analysis(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device | str,
    output_dir: str | Path,
    *,
    fixed_t: int = 0,
    lesion_quantiles: Sequence[float] = (0.25, 0.5, 0.75),
    ring_width: int = 3,
    frozen_quartiles: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Per-layer channel-agnostic lesion-emergence metrics, patient-aggregated.

    Stratification uses lesion-area quartiles frozen from the *training* masks.
    When the loader split is train itself this is computed from the same data
    (reported as ``frozen_on=loader_split``); callers evaluating a held-out
    split should pass precomputed quantiles.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Phase 1: collect all lesion areas.  The audit requires quartiles frozen
    # from the *training* split.  ``frozen_quartiles`` may be passed precomputed
    # (from a train-only loader); otherwise we freeze from the current loader and
    # mark ``frozen_on`` accordingly so the report is explicit about the source.
    if frozen_quartiles is not None:
        quantiles = [float(q) for q in frozen_quartiles]
        frozen_source = "provided"
    else:
        all_areas: list[float] = []
        for batch in loader:
            for idx in range(_batch_size(batch)):
                m = batch["mask"][idx] if torch.is_tensor(batch.get("mask")) else None
                if m is not None:
                    area = float((m > 0.5).sum().item())
                    all_areas.append(area)
        if not all_areas:
            raise ValueError("No masks found in loader; lesion emergence analysis requires masks")
        if not any(area > 0.0 for area in all_areas):
            raise ValueError(
                "All masks are empty; lesion emergence analysis requires at least one "
                "non-empty lesion mask"
            )
        areas_arr = np.asarray(all_areas, dtype=np.float64)
        quantiles = np.quantile(areas_arr, lesion_quantiles).tolist()
        frozen_source = "loader_split"

    cohort_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_gpu = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            patients = _patient_keys(batch_gpu)
            t_batch = torch.full(
                (_batch_size(batch_gpu),),
                fixed_t,
                device=device,
                dtype=torch.long,
            )
            bundle = model.build_condition_bundle(batch_gpu, t_batch)
            ct_feats = [bundle.maps[key] for key in FEATURE_KEYS if key in bundle.maps]
            for idx, patient in enumerate(patients):
                mask = batch_gpu["mask"][idx]
                lesion_area = float((mask > 0.5).sum().item())
                for i, feat in enumerate(ct_feats):
                    # Upsample the activation to the FULL mask resolution so small
                    # lesions are not erased by downsampling the mask to deep layers.
                    act_low = channel_agnostic_activation(feat[idx : idx + 1])
                    act = F.interpolate(
                        act_low.float(),
                        size=mask.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    # Mask survival: fraction of the ORIGINAL lesion pixels that
                    # are still covered by the feature-grid mask after it is
                    # nearest-neighbour upsampled back to full resolution.  This
                    # keeps both sides on the same (full-res) scale, so a deep
                    # layer with a 2x2 grid cell can report the true fraction of
                    # the lesion it still represents rather than a pixel-count
                    # ratio that shrinks with downsampling.
                    mask_ds = F.interpolate(
                        mask.float().unsqueeze(0),
                        size=act_low.shape[2:],
                        mode="nearest",
                    )
                    mask_up = F.interpolate(
                        mask_ds,
                        size=mask.shape[-2:],
                        mode="nearest",
                    )
                    lesion_full = (mask > 0.5).float().unsqueeze(0)
                    survival = float(
                        ((mask_up > 0.5) & (lesion_full > 0.5)).sum().item()
                    ) / max(float((lesion_full > 0.5).sum().item()), 1.0)
                    contrast = lesion_ring_contrast(
                        act, mask, ring_width=ring_width
                    )
                    disc = lesion_discriminability(act, mask)
                    hit = lesion_centroid_hit(act, mask)
                    quartile_label = _quartile_label(lesion_area, quantiles)
                    cohort_rows.append(
                        {
                            "patient_id": patient,
                            "layer": f"c{i + 1}",
                            "lesion_area": lesion_area,
                            "quartile": quartile_label,
                            "mask_survival": survival,
                            "lesion_ring_contrast": contrast,
                            "auroc": disc["auroc"],
                            "auprc": disc["auprc"],
                            "lesion_pixels": disc["lesion_pixels"],
                            "background_pixels": disc["background_pixels"],
                            "centroid_hit": hit["centroid_hit"],
                        }
                    )
                    metric_rows.append(
                        {
                            "layer": f"c{i + 1}",
                            "quartile": quartile_label,
                            "mask_survival": survival,
                            "lesion_ring_contrast": contrast,
                            "auroc": disc["auroc"],
                            "auprc": disc["auprc"],
                            "centroid_hit": hit["centroid_hit"],
                        }
                    )

    write_csv(out / "cohort.csv", cohort_rows)
    write_csv(out / "lesion_emergence_metrics.csv", metric_rows)

    # Patient-level aggregate per layer × quartile.
    patient_grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in cohort_rows:
        key = (row["layer"], row["quartile"])
        patient_grouped.setdefault(key, []).append(row)
    layer_summary: list[dict[str, Any]] = []
    for (layer, quartile), rows in sorted(patient_grouped.items()):
        patients = sorted({r["patient_id"] for r in rows})
        by_patient: dict[str, list[float]] = {}
        for r in rows:
            by_patient.setdefault(r["patient_id"], []).append(r["lesion_ring_contrast"])
        patient_means = []
        for p in patients:
            values = by_patient[p]
            values = [v for v in values if np.isfinite(v)]
            patient_means.append(float(np.nanmean(values)) if values else float("nan"))
        valid = [v for v in patient_means if np.isfinite(v)]
        ci = bootstrap_mean(valid, seed=42) if valid else {}
        layer_summary.append(
            {
                "layer": layer,
                "quartile": quartile,
                "n_patients": len(patients),
                "lesion_ring_contrast_patient_mean": float(np.nanmean(valid)) if valid else float("nan"),
                "ci95_low": ci.get("ci95_low"),
                "ci95_high": ci.get("ci95_high"),
            }
        )

    # M1 gate: patient-level c2 vs c1 and absolute c2 lesion-ring contrast.
    def _layer_patient_values(layer: str) -> dict[str, float]:
        layer_rows_all = [r for r in cohort_rows if r["layer"] == layer]
        by_patient: dict[str, list[float]] = {}
        for r in layer_rows_all:
            by_patient.setdefault(r["patient_id"], []).append(r["lesion_ring_contrast"])
        return {
            p: float(np.nanmean(values)) for p, values in by_patient.items()
        }

    c2_values_map = _layer_patient_values("c2")
    c1_values_map = _layer_patient_values("c1")
    c2_ci = bootstrap_mean(
        [v for v in c2_values_map.values() if np.isfinite(v)], seed=42
    )
    c1_ci = bootstrap_mean(
        [v for v in c1_values_map.values() if np.isfinite(v)], seed=42
    )

    # Paired c2 − c1 difference per patient (only patients present in both).
    shared_patients = sorted(set(c2_values_map) & set(c1_values_map))
    paired_delta = [
        c2_values_map[p] - c1_values_map[p]
        for p in shared_patients
        if np.isfinite(c2_values_map[p]) and np.isfinite(c1_values_map[p])
    ]
    delta_ci = bootstrap_mean(paired_delta, seed=42 + 1)
    delta_p = sign_flip_p(paired_delta, seed=42 + 2)

    # Smallest-lesion quartile: c2 contrast must be directionally consistent.
    c2_min_q = [
        r for r in layer_summary if r["layer"] == "c2" and r["quartile"] == "Q1"
    ]
    min_q_low = (c2_min_q[0].get("ci95_low") if c2_min_q else None)

    result = {
        "stage": "A2",
        "fixed_t": fixed_t,
        "lesion_quantiles": lesion_quantiles,
        "lesion_area_quartiles_px": quantiles,
        "frozen_on": frozen_source,
        "layer_summary": layer_summary,
        "m1_gate": {
            "c2_lesion_ring_contrast_ci95": c2_ci,
            "c1_lesion_ring_contrast_ci95": c1_ci,
            "c2_minus_c1_paired_ci95": delta_ci,
            "c2_minus_c1_sign_flip_p": delta_p,
            "c2_above_c1": bool(
                delta_ci.get("ci95_low") is not None
                and delta_ci.get("ci95_low", 0.0) > 0.0
            ),
            "smallest_quartile_consistent": bool(
                min_q_low is not None and min_q_low > 0.0
            ),
            # M1 is a CONJUNCTION: c2 contrast CI > 0 AND c2 > c1 AND Q1 consistent.
            "passed": bool(
                c2_ci.get("ci95_low") is not None
                and c2_ci["ci95_low"] > 0.0
                and delta_ci.get("ci95_low") is not None
                and delta_ci["ci95_low"] > 0.0
                and min_q_low is not None
                and min_q_low > 0.0
            ),
        },
    }
    write_json(out / "lesion_emergence.json", result)
    return result


def _quartile_label(area: float, quantiles: Sequence[float]) -> str:
    q1, q2, q3 = quantiles
    if area <= q1:
        return "Q1"
    if area <= q2:
        return "Q2"
    if area <= q3:
        return "Q3"
    return "Q4"
