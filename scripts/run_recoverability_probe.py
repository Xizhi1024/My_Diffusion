"""FR-6.2 (DESIGN S10 row 2): R0 cross-fitted recoverability probe.

Implements [计划] §3.4 with the compression of [审计] §1.1/§2: matched-
capacity closed-form ridge probes are cross-fitted at PATIENT level. The
baseline probe sees only the noisy residual coefficient; the conditional
probe adds CT features through a zero-block adapter (adapter=0 equals the
baseline exactly, [审计] §2 matched-capacity rules). Out-of-fold excess risk
per cell (band group g, log-SNR region k, stratum s):

    Delta_{g,s}(k) = (R_base - R_cond) / (Var(z_tilde) + eps)   # [计划] §3.4

studentized over patient blocks and calibrated against a patient-permuted-CT
max-T null (95% synchronized LCB). Wrong-band CT and label-free spatial-shift
nulls are reported descriptively. PASS ([计划] v1.5 §3.4; DESIGN §10 v1.0f,
PRD v1.0.1 addendum 4, AUDIT_3 ruling 1): >=2 of the small-lesion
confirmatory cells (group x log-SNR region) with synchronized LCB>0 beating
the max-T null, covering >=2 distinct lambda regions; the >=4/5 outer-train
same-direction aggregation is owned by the orchestration layer, and per-cell
judgments are retained in pass_rule.per_cell for it. Grid: compressed = 3
band groups x 3 log-SNR regions x small-lesion as the only confirmatory
stratum; full = 7 bands x K lambda x 3 strata. Patients below the grid
threshold exit 2.

CLI: --config --fold --grid {compressed,full} --n-perm 1000 --out. The data
source is injectable: config probe.source = "synthetic" (seeded generator)
or "npz" (cached probe matrix; array contract in load_npz_probe_data).
numpy only - the ridge fits are closed form, no sklearn.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.model.rc_brd import BAND_NAMES  # noqa: E402

SCHEMA_VERSION = 1
COMPRESSED_GROUPS = ("low", "mid", "high")   # [计划] §3.2 pre-registered 3 groups
FULL_STRATA = ("small_lesion", "non_small_lesion", "background")  # [计划] §3.4
CONFIRMATORY_STRATUM = "small_lesion"       # [审计] §1.1: only confirmatory layer
COMPRESSED_MIN_PATIENTS = 12                # [审计] §1.1 thresholds
FULL_MIN_PATIENTS = 24
MIN_PASS_CELLS = 2                          # [计划] v1.5 §3.4 PASS rule
MIN_PASS_LAMBDA_REGIONS = 2                 # passing cells cover >=2 regions
VAR_EPS = 1e-12                             # [计划] §3.4: /(Var + eps)
SE_GUARD = 1e-12                            # studentization guard


@dataclass(frozen=True)
class ProbeData:
    """Probe matrix: z/u [G,K,P,C]; x [G,K,P,C,D]; stratum ids [G,K,P,C]."""

    patient_ids: tuple[str, ...]
    groups: tuple[str, ...]
    lambdas: np.ndarray
    strata: tuple[str, ...]
    z: np.ndarray
    u: np.ndarray
    x: np.ndarray
    stratum_ids: np.ndarray
    psd_floors: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Data sources (injectable; [计划] §3.4)
# ---------------------------------------------------------------------------

def make_synthetic_data(spec: dict[str, Any]) -> ProbeData:
    """Seeded synthetic probe matrix ([审计] §2 probe model).

    Recoverable groups embed a CT-linear component z = rho*(x@beta) +
    sqrt(1-rho^2)*eps; u = s(lambda)*z + sqrt(1-s^2)*eta with s = sigmoid.
    Cells are standardized per (group, lambda) over the whole cohort (the
    cohort IS outer-train; PSD/noise-floor semantics of [计划] §3.4).
    """
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    groups = tuple(str(g) for g in spec.get("groups", COMPRESSED_GROUPS))
    strata = tuple(str(s) for s in spec.get("strata", (CONFIRMATORY_STRATUM,)))
    lambdas = np.asarray(spec.get("log_snr_centers", (-2.0, 0.0, 2.0)), dtype=float)
    n_patients = int(spec.get("n_patients", 60))
    n_coef = int(spec.get("n_coef", 64))
    n_features = int(spec.get("n_features", 3))
    rho = float(spec.get("signal_rho", 0.85))
    recoverable = {str(g) for g in spec.get("recoverable_groups", ())}
    G, K = len(groups), len(lambdas)
    beta = np.ones(n_features) / np.sqrt(n_features)
    z = np.empty((G, K, n_patients, n_coef))
    u = np.empty_like(z)
    x = np.empty((G, K, n_patients, n_coef, n_features))
    s_grid = 1.0 / (1.0 + np.exp(-lambdas))
    for gi, group in enumerate(groups):
        for k in range(K):
            xk = rng.standard_normal((n_patients, n_coef, n_features))
            noise = rng.standard_normal((n_patients, n_coef))
            if group in recoverable:
                zk = rho * (xk @ beta) + np.sqrt(1.0 - rho ** 2) * noise
            else:
                zk = noise
            uk = s_grid[k] * zk + np.sqrt(1.0 - s_grid[k] ** 2) * \
                rng.standard_normal((n_patients, n_coef))
            z[gi, k], u[gi, k], x[gi, k] = zk, uk, xk
    z, u, x = _standardize(z, u, x)
    per_stratum = -(-n_coef // len(strata))
    sid = np.broadcast_to(np.arange(len(strata), dtype=np.int64)
                          .repeat(per_stratum)[:n_coef], (G, K, n_patients, n_coef)).copy()
    floors = {g: float(np.median(1.0 - s_grid ** 2)) for g in groups}  # noise proxy
    return ProbeData(tuple(f"p{i:03d}" for i in range(n_patients)), groups,
                     lambdas, strata, z, u, x, sid, floors)


def _standardize(z: np.ndarray, u: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, ...]:
    """Per-(group, lambda) cohort standardization ([计划] §3.4)."""
    for gi in range(z.shape[0]):
        for k in range(z.shape[1]):
            z[gi, k] /= z[gi, k].std() + VAR_EPS
            u[gi, k] /= u[gi, k].std() + VAR_EPS
            x[gi, k] /= x[gi, k].std(axis=(0, 1), keepdims=True) + VAR_EPS
    return z, u, x


def load_npz_probe_data(path: Path) -> ProbeData:
    """Load a cached probe matrix (.npz). Array contract: patient_ids/groups/
    strata (string arrays), lambdas [K] float, z/u [G,K,P,C],
    x [G,K,P,C,D], stratum_ids int [G,K,P,C]; optional psd_floors float [G]."""
    with np.load(path, allow_pickle=False) as data:
        required = ("patient_ids", "groups", "lambdas", "strata", "z", "u", "x",
                    "stratum_ids")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"probe npz {path} is missing arrays: {missing}")
        groups = tuple(str(g) for g in data["groups"].tolist())
        strata = tuple(str(s) for s in data["strata"].tolist())
        floors = None
        if "psd_floors" in data.files:
            floors = {g: float(v) for g, v in zip(groups, data["psd_floors"].tolist())}
        return ProbeData(tuple(str(p) for p in data["patient_ids"].tolist()),
                         groups, np.asarray(data["lambdas"], dtype=float), strata,
                         np.asarray(data["z"], dtype=float),
                         np.asarray(data["u"], dtype=float),
                         np.asarray(data["x"], dtype=float),
                         np.asarray(data["stratum_ids"], dtype=np.int64), floors)


def load_probe_data(config: dict[str, Any], grid_groups: tuple[str, ...],
                    grid_strata: tuple[str, ...]) -> ProbeData:
    """Injectable data entry: synthetic gets the grid axes pinned by the run."""
    spec = dict(config.get("probe", {}))
    source = str(spec.get("source", "synthetic"))
    if source == "synthetic":
        spec["groups"], spec["strata"] = list(grid_groups), list(grid_strata)
        return make_synthetic_data(spec)
    if source == "npz":
        data = load_npz_probe_data(Path(str(spec["path"])))
        if set(data.groups) != set(grid_groups):
            raise ValueError(f"probe data groups {data.groups} != grid {grid_groups}")
        return data
    raise ValueError(f"unknown probe data source {source!r} (synthetic|npz)")


# ---------------------------------------------------------------------------
# Closed-form ridge core (patient-pairing permutation algebra)
# ---------------------------------------------------------------------------

def cell_statistics(z: np.ndarray, u: np.ndarray, x: np.ndarray) -> dict[str, Any]:
    """Sufficient statistics for one cell; z,u [P,C], x [P,C,D].

    UX[p,q,d] = sum_c u[p,c]*x[q,c,d] pairs patient p's noisy coefficient
    with patient q's CT features; every patient-permuted fit/evaluation is a
    gather over these pair matrices ([审计] §2 permutation null).
    """
    return {"zz": np.einsum("pc,pc->p", z, z), "uu": np.einsum("pc,pc->p", u, u),
            "uz": np.einsum("pc,pc->p", u, z), "ux": np.einsum("pc,qcd->pqd", u, x),
            "xz": np.einsum("pc,qcd->pqd", z, x), "xx": np.einsum("pcd,pce->pde", x, x),
            "n_rows": int(z.shape[1])}


def patient_folds(n_patients: int, n_folds: int, seed: int) -> list[np.ndarray]:
    """Seeded patient-level cross-fit folds (held-out index arrays)."""
    order = np.random.default_rng(seed).permutation(n_patients)
    folds = max(1, min(int(n_folds), n_patients))
    return [np.sort(part) for part in np.array_split(order, folds)]


def probe_cell_statistics(stats: dict[str, Any], folds: list[np.ndarray],
                          ridge_lambda: float, perms: np.ndarray,
                          denom: float) -> dict[str, np.ndarray]:
    """Cross-fitted ridge under patient pairings perms [R,P] for one cell.

    Conditional fit solves (A + alpha*I) w = b with w = (w_u, w_x); the
    baseline is the SAME system with the CT block zeroed (zero-init adapter,
    [审计] §2): w_x = 0 and w_base = b_u/(a_uu+alpha). alpha scales with the
    training row count so both probes shrink identically (matched capacity).
    """
    P = stats["zz"].shape[0]
    R, D = perms.shape[0], stats["xx"].shape[-1]
    pidx = np.arange(P)
    ux_pi = stats["ux"][pidx[None, :], perms]        # [R,P,D]
    xz_pi = stats["xz"][pidx[None, :], perms]
    xx_pi = stats["xx"][perms]                       # [R,P,D,D]
    eye = np.eye(D)
    d_mat = np.empty((R, P)); r_base = np.empty((R, P)); r_cond = np.empty((R, P))
    for hold in folds:
        train = np.setdiff1d(pidx, hold)
        alpha = ridge_lambda * max(train.size * stats["n_rows"], 1)
        m = np.zeros(P); m[train] = 1.0
        a_uu = float(stats["uu"][train].sum()); b_u = float(stats["uz"][train].sum())
        a_ux = np.einsum("p,rpd->rd", m, ux_pi)
        a_xx = np.einsum("p,rpde->rde", m, xx_pi) + alpha * eye
        b_x = np.einsum("p,rpd->rd", m, xz_pi)
        A = np.zeros((R, D + 1, D + 1))
        A[:, 0, 0] = a_uu + alpha
        A[:, 0, 1:] = a_ux; A[:, 1:, 0] = a_ux; A[:, 1:, 1:] = a_xx
        b = np.concatenate([np.full((R, 1), b_u), b_x], axis=1)
        W = np.linalg.solve(A, b[..., None])[..., 0]  # [R,1+D] batched solve
        w_base = b_u / (a_uu + alpha)
        C = stats["n_rows"]
        zz, uu, uz = stats["zz"][hold], stats["uu"][hold], stats["uz"][hold]
        w_u, w_x = W[:, :1], W[:, 1:]
        lin = w_u * uz[None, :] + np.einsum("rhd,rd->rh", xz_pi[:, hold, :], w_x)
        quad = (w_u ** 2) * uu[None, :] \
            + 2.0 * w_u * np.einsum("rhd,rd->rh", ux_pi[:, hold, :], w_x) \
            + np.einsum("rd,rhde,re->rh", w_x, xx_pi[:, hold], w_x)
        err_cond = (zz[None, :] - 2.0 * lin + quad) / C
        err_base = (zz - 2.0 * w_base * uz + w_base ** 2 * uu) / C
        d_mat[:, hold] = (err_base[None, :] - err_cond) / denom   # [计划] §3.4
        r_base[:, hold] = np.broadcast_to(err_base, err_cond.shape)
        r_cond[:, hold] = err_cond
    return {"d": d_mat, "r_base": r_base, "r_cond": r_cond}


def cell_probe(z: np.ndarray, u: np.ndarray, x: np.ndarray, folds: list[np.ndarray],
               ridge_lambda: float, perms: np.ndarray) -> dict[str, Any]:
    """Observed (identity pairing) + patient-permuted evaluation of one cell."""
    stats = cell_statistics(z, u, x)
    denom = float(z.var()) + VAR_EPS
    identity = np.arange(z.shape[0])[None, :]
    obs = probe_cell_statistics(stats, folds, ridge_lambda, identity, denom)
    null = probe_cell_statistics(stats, folds, ridge_lambda, perms, denom)
    P = z.shape[0]
    d_obs = obs["d"][0]
    se = float(d_obs.std(ddof=1) / np.sqrt(P) + SE_GUARD)
    t_null = null["d"].mean(axis=1) / (null["d"].std(axis=1, ddof=1) / np.sqrt(P)
                                       + SE_GUARD)
    return {"delta": float(d_obs.mean()), "r_base": float(obs["r_base"][0].mean()),
            "r_cond": float(obs["r_cond"][0].mean()), "se": se,
            "t_obs": float(d_obs.mean() / se), "t_null": t_null}


def _cell_arrays(data: ProbeData, gi: int, k: int, si: int) -> tuple[np.ndarray, ...]:
    """Rows of one (group, lambda, stratum) cell as [P,C] / [P,C,D] arrays."""
    mask = data.stratum_ids[gi, k] == si
    if int(mask.sum(axis=1).min()) < 1:
        raise ValueError("every patient needs >=1 coefficient per cell")
    P = data.z.shape[2]
    z = data.z[gi, k][mask].reshape(P, -1)
    u = data.u[gi, k][mask].reshape(P, -1)
    x = data.x[gi, k][mask].reshape(P, z.shape[1], data.x.shape[-1])
    return z, u, x


# ---------------------------------------------------------------------------
# Grid evaluation, nulls, PASS rule ([计划] §3.4)
# ---------------------------------------------------------------------------

def _evaluate_grid(data: ProbeData, folds: list[np.ndarray], ridge: float,
                   perms: np.ndarray) -> tuple[dict[tuple, dict], np.ndarray]:
    """Cross-fit every cell; returns cell stats + [n_perm, n_cells] null T's."""
    n_cells = len(data.groups) * len(data.lambdas) * len(data.strata)
    cells: dict[tuple, dict] = {}
    t_null = np.empty((perms.shape[0], n_cells))
    idx = 0
    for gi, g in enumerate(data.groups):
        for k in range(len(data.lambdas)):
            for si, s in enumerate(data.strata):
                z, u, x = _cell_arrays(data, gi, k, si)
                cell = cell_probe(z, u, x, folds, ridge, perms)
                t_null[:, idx] = cell.pop("t_null")
                idx += 1
                cells[(g, k, s)] = cell
    return cells, t_null


def _nested_delta(cells: dict[tuple, dict], groups: tuple[str, ...],
                  n_lambda: int, strata: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s in strata:
        out[s] = {g: {str(k): cells[(g, k, s)] for k in range(n_lambda)}
                  for g in groups}
    return out


def _apply_lcb(nested: dict[str, Any], groups: tuple[str, ...], n_lambda: int,
               strata: tuple[str, ...], q95: float) -> None:
    """95% synchronized LCB from the max-T null ([计划] §3.4 / [审计] §2)."""
    for s in strata:
        for g in groups:
            for k in range(n_lambda):
                cell = nested[s][g][str(k)]
                cell["lcb95"] = float((cell["t_obs"] - q95) * cell["se"])
                cell["exceeds_maxt_null"] = bool(cell["t_obs"] > q95)


def _null_probes(data: ProbeData, folds: list[np.ndarray], ridge: float,
                 q95: float) -> tuple[dict, dict]:
    """Wrong-band CT and label-free spatial-shift nulls on the confirmatory
    stratum ([计划] §3.4 three nulls; descriptive, reported with LCB)."""
    wrong: dict[str, Any] = {}
    shift: dict[str, Any] = {}
    si = data.strata.index(CONFIRMATORY_STRATUM)
    for gi, g in enumerate(data.groups):
        for k in range(len(data.lambdas)):
            z, u, x = _cell_arrays(data, gi, k, si)
            _, _, x_w = _cell_arrays(data, (gi + 1) % len(data.groups), k, si)
            for name, cell, feats in (("wrong_band_ct", wrong, x_w),
                                      ("spatial_shift", shift, np.roll(x, 1, axis=1))):
                result = cell_probe(z, u, feats, folds, ridge,
                                    np.arange(z.shape[0])[None, :])
                entry = {key: result[key] for key in ("delta", "t_obs", "se")}
                entry["lcb95"] = float((result["t_obs"] - q95) * result["se"])
                cell[f"{g}|lam{k}"] = entry
    return wrong, shift


def evaluate_pass_rule(nested: dict[str, Any], groups: tuple[str, ...],
                       n_lambda: int, q95: float) -> dict[str, Any]:
    """[计划] v1.5 §3.4 PASS rule: among the small-lesion confirmatory cells
    (group x log-SNR region), at least MIN_PASS_CELLS cells hold a
    synchronized LCB>0 that beats the patient-permuted max-T null, and the
    passing cells cover >= MIN_PASS_LAMBDA_REGIONS distinct lambda regions.
    The >=4/5 outer-train same-direction aggregation belongs to the
    orchestration layer; per-cell judgments are retained in per_cell for it
    (DESIGN §10 v1.0f; PRD v1.0.1 addendum 4; AUDIT_3 ruling 1)."""
    per_cell: dict[str, Any] = {}
    passing_cells: list[str] = []
    covered: set[int] = set()
    for g in groups:
        for k in range(n_lambda):
            cell = nested[CONFIRMATORY_STRATUM][g][str(k)]
            ok = bool(cell["lcb95"] > 0.0 and cell["exceeds_maxt_null"])
            per_cell[f"{g}|lam{k}"] = {"delta": cell["delta"],
                                       "t_obs": cell["t_obs"],
                                       "lcb95": cell["lcb95"],
                                       "exceeds_maxt_null":
                                           cell["exceeds_maxt_null"],
                                       "pass": ok}
            if ok:
                passing_cells.append(f"{g}|lam{k}")
                covered.add(int(k))
    passed = (len(passing_cells) >= MIN_PASS_CELLS
              and len(covered) >= MIN_PASS_LAMBDA_REGIONS)
    return {"passed": passed,
            "n_passing_cells": len(passing_cells),
            "passing_cells": passing_cells,
            "covered_lambda_regions": sorted(covered),
            "per_cell": per_cell,
            "rule": {"min_passing_cells": MIN_PASS_CELLS,
                     "min_lambda_region_coverage": MIN_PASS_LAMBDA_REGIONS,
                     "confirmatory_stratum": CONFIRMATORY_STRATUM,
                     "max_t_q95": q95,
                     "cross_fold_note": ">=4/5 outer-train same-direction "
                                        "aggregation is owned by the "
                                        "orchestration layer (per_cell "
                                        "judgments retained for it)"}}


# ---------------------------------------------------------------------------
# Orchestration + CLI (DESIGN S10 row 2)
# ---------------------------------------------------------------------------

def run_probe(config: dict[str, Any], fold: str, grid_mode: str, n_perm: int,
              out_path: Path) -> int:
    """R0 probe orchestration; returns the process exit code (DESIGN S10)."""
    if grid_mode == "compressed":
        groups, strata = tuple(COMPRESSED_GROUPS), (CONFIRMATORY_STRATUM,)
        min_patients = COMPRESSED_MIN_PATIENTS
    elif grid_mode == "full":
        groups, strata = tuple(BAND_NAMES), FULL_STRATA
        min_patients = FULL_MIN_PATIENTS
    else:
        raise ValueError(f"grid must be compressed|full, got {grid_mode!r}")
    data = load_probe_data(config, groups, strata)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_patients = len(data.patient_ids)
    if n_patients < min_patients:      # [DESIGN S10]: insufficient data -> exit 2
        _write_json(out_path, {"status": "insufficient_patients", "fold": fold,
                               "grid": grid_mode, "n_patients": n_patients,
                               "threshold": min_patients})
        return 2
    seed = int(config.get("seed", 0))
    cross = dict(config.get("cross_fit", {}))
    n_folds = int(cross.get("n_folds", 5))
    ridge = float(cross.get("ridge_lambda", 0.01))
    lambdas = np.asarray(data.lambdas, dtype=float)
    if not all(lambdas[i] < lambdas[i + 1] for i in range(len(lambdas) - 1)):
        raise ValueError("log_snr_centers must be strictly increasing")
    folds = patient_folds(n_patients, n_folds, seed)
    rng = np.random.default_rng(seed + 1)
    perms = np.stack([rng.permutation(n_patients) for _ in range(max(1, int(n_perm)))])
    cells, t_null = _evaluate_grid(data, folds, ridge, perms)
    max_t = t_null.max(axis=1)
    q50, q95, q99 = (float(np.quantile(max_t, q)) for q in (0.5, 0.95, 0.99))
    n_lambda = int(len(lambdas))
    nested = _nested_delta(cells, groups, n_lambda, strata)
    _apply_lcb(nested, groups, n_lambda, strata, q95)
    wrong, shift = _null_probes(data, folds, ridge, q95)
    pass_rule = evaluate_pass_rule(nested, groups, n_lambda, q95)
    _write_json(out_path, {
        "schema_version": SCHEMA_VERSION, "fold": fold, "seed": seed,
        "grid": {"mode": grid_mode, "groups": list(groups),
                 "log_snr_centers": lambdas.tolist(), "strata": list(strata),
                 "n_lambda": n_lambda},
        "n_patients": n_patients, "n_perm": int(perms.shape[0]),
        "cross_fit": {"n_folds": len(folds), "ridge_lambda": ridge,
                      "held_out_folds": [f.tolist() for f in folds]},
        "delta": nested,
        "nulls": {"patient_permuted_ct": {"n_perm": int(perms.shape[0]),
                                          "max_t_q50": q50, "max_t_q95": q95,
                                          "max_t_q99": q99},
                  "wrong_band_ct": wrong, "spatial_shift": shift},
        "psd_floors": data.psd_floors,
        "pass_rule": pass_rule})
    print(f"probe fold={fold} grid={grid_mode} passed={pass_rule['passed']} "
          f"n_passing_cells={pass_rule['n_passing_cells']} "
          f"passing_cells={pass_rule['passing_cells']} max_t_q95={q95:.3f}")
    return 0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="R0 cross-fitted recoverability probe ([计划] §3.4)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--grid", choices=("compressed", "full"), default="compressed")
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        return run_probe(config, args.fold, args.grid, args.n_perm, Path(args.out))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
