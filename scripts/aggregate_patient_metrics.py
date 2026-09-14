"""FR-6.5 (DESIGN RC-BRD_v1 S10): patient-level small-lesion metric aggregation
plus the C3 acceptance-gate evaluator.

[计划] §5.3 frozen metric contract: q25 comes from OUTER-TRAIN masks only (2D
connected-component pixel areas) and is then frozen; small_lesion_2d_mae_i =
mean over lesions l of patient i with A_l <= q25_train of MAE(pred_l, target_l);
the statistical unit is the PATIENT (in-patient mean first, then across
patients); patients without a qualifying lesion follow the preregistered
missing rule (excluded from the primary endpoint, retained in the report).
Standalone batch-3B deliverable; CLI per DESIGN S10 (missing prediction file
-> exit 1; invalid --gate -> exit 2; deltas json omitted -> no gate block).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage

Q25_QUANTILE = 0.25  # [计划] §5.3: q25 of pooled outer-train component areas
CONNECTIVITY = 8  # documented choice: diagonal-touching pixels form one lesion
_CONNECTIVITY_STRUCTURE = np.ones((3, 3), dtype=bool)  # 8-neighbourhood
NPZ_PRED_KEY = "pred"
NPZ_TARGET_KEY = "target"
NPZ_MASK_KEY = "mask"
NPZ_STACKED_MASK_KEYS = ("mask", "masks")
METRIC_NAME = "small_lesion_2d_mae"
ALL_LESION_METRIC_NAME = "lesion_2d_mae"
MISSING_RULE = "no qualifying lesion (A_l <= q25_train): excluded from the primary endpoint, retained in the report"
ACCEPTANCE_GATES = ("gate_e_isomorphic", "strict_all_metrics")  # C3 (PRD §3)
FAVOR_DIRECTIONS = ("lower", "upper")  # favorable direction of delta = M - D0
DEFAULT_NI_MARGIN = 0.02  # C3 noninferiority |mean delta| margin (DESIGN §10)


class MetricAggregationError(RuntimeError):
    """Fail-closed error for aggregation inputs (missing files / bad shapes)."""


# --- core metric logic ([计划] §5.3) ---------------------------------------

def lesion_component_areas(mask_2d: np.ndarray) -> np.ndarray:
    """Return 2D connected-component areas (pixel counts) of one mask slice."""
    binary = np.asarray(mask_2d) > 0
    if binary.ndim != 2:
        raise MetricAggregationError(f"mask slice must be 2D, got {binary.shape}")
    if not binary.any():
        return np.empty(0, dtype=np.int64)
    labels, _ = ndimage.label(binary, structure=_CONNECTIVITY_STRUCTURE)
    return np.bincount(labels.ravel())[1:].astype(np.int64)


def _iter_2d_slices(masks: Iterable[np.ndarray] | np.ndarray) -> list[np.ndarray]:
    """Flatten (one stacked array OR an iterable of 2D/3D arrays) to 2D slices."""
    arrays = [masks] if isinstance(masks, np.ndarray) else [np.asarray(m) for m in masks]
    slices: list[np.ndarray] = []
    for arr in arrays:
        if arr.ndim == 2:
            slices.append(arr)
        elif arr.ndim == 3:
            slices.extend(arr[i] for i in range(arr.shape[0]))
        else:
            raise MetricAggregationError(f"mask array must be 2D or 3D, got {arr.shape}")
    return slices


def _pooled_train_areas(train_masks: Iterable[np.ndarray] | np.ndarray) -> np.ndarray:
    per_slice = [lesion_component_areas(s) for s in _iter_2d_slices(train_masks)]
    if not per_slice:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(per_slice).astype(np.int64)


def compute_small_lesion_q25(
    train_masks: Iterable[np.ndarray] | np.ndarray,
) -> float:
    """q25 of pooled component areas, OUTER-TRAIN masks only ([计划] §5.3).

    Frozen before any outer-test lesion is seen; raises when no component."""
  
    areas = _pooled_train_areas(train_masks)
    if areas.size == 0:
        raise MetricAggregationError("outer-train masks hold no 2D lesion components; q25 undefined")
    return float(np.quantile(areas, Q25_QUANTILE))


@dataclass(frozen=True)
class LesionRecord:
    """One 2D connected component: pixel area + within-lesion MAE (§5.3)."""

    area: int
    mae: float


def _as_3d(arr: Any, name: str) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[None, ...]
    if a.ndim != 3:
        raise MetricAggregationError(f"{name} must be 2D or 3D, got {a.shape}")
    return a


def lesion_records(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> list[LesionRecord]:
    """Per-lesion (area, MAE) over all slices of one patient ([计划] §5.3)."""
    p = _as_3d(pred, "pred")
    t = _as_3d(target, "target")
    m = _as_3d(mask, "mask")
    if not (p.shape == t.shape == m.shape):
        raise MetricAggregationError(f"pred/target/mask shapes differ: {p.shape}/{t.shape}/{m.shape}")
    records: list[LesionRecord] = []
    for s in range(p.shape[0]):
        binary = m[s] > 0
        if not binary.any():
            continue
        labels, n_lesions = ndimage.label(binary, structure=_CONNECTIVITY_STRUCTURE)
        absdiff = np.abs(p[s].astype(np.float64) - t[s].astype(np.float64))
        for lesion_id in range(1, n_lesions + 1):
            inside = labels == lesion_id
            records.append(LesionRecord(int(inside.sum()), float(absdiff[inside].mean())))
    return records


@dataclass(frozen=True)
class PatientMetric:
    """Patient-level aggregation of the lesion table ([计划] §5.3)."""

    patient_id: str
    n_slices: int
    n_lesions: int
    n_small_lesions: int
    small_lesion_2d_mae: float  # NaN when no lesion satisfies A_l <= q25
    lesion_2d_mae: float  # mean over ALL lesions (NaN when lesion-free)


def compute_patient_metric(
    patient_id: str,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    q25: float,
) -> PatientMetric:
    """Aggregate one patient: mean over her/his small lesions first (§5.3)."""
    records = lesion_records(pred, target, mask)
    small_maes = [r.mae for r in records if r.area <= q25]
    all_maes = [r.mae for r in records]
    return PatientMetric(
        patient_id=patient_id,
        n_slices=int(_as_3d(pred, "pred").shape[0]),
        n_lesions=len(records),
        n_small_lesions=len(small_maes),
        small_lesion_2d_mae=float(np.mean(small_maes)) if small_maes else float("nan"),
        lesion_2d_mae=float(np.mean(all_maes)) if all_maes else float("nan"),
    )


def _nan_to_none(value: float | None) -> float | None:
    if value is None:
        return None
    return None if value != value else value


def _patient_row(metric: PatientMetric) -> dict[str, Any]:
    row = asdict(metric)
    row[METRIC_NAME] = _nan_to_none(metric.small_lesion_2d_mae)
    row[ALL_LESION_METRIC_NAME] = _nan_to_none(metric.lesion_2d_mae)
    return row


def build_report(
    metrics: Sequence[PatientMetric],
    q25: float,
    n_train_components: int,
) -> dict[str, Any]:
    """JSON-safe report incl. the preregistered missing-rule fields."""
    missing = [m.patient_id for m in metrics if m.n_small_lesions == 0]
    effective = [m for m in metrics if m.n_small_lesions > 0]
    values = [m.small_lesion_2d_mae for m in effective]
    cohort_mean = float(np.mean(values)) if values else None
    return {
        "metric": METRIC_NAME,
        "definition": ("[计划] §5.3: per patient, mean MAE over lesions with "
                       "A_l <= q25_train; in-patient aggregation first; q25 "
                       "frozen from outer-train components only"),
        "q25_quantile": Q25_QUANTILE,
        "q25_threshold": q25,
        "connectivity": CONNECTIVITY,
        "n_train_components": n_train_components,
        "patient_count": len(metrics),
        "effective_patient_count": len(effective),
        "missing_patients": missing,
        "missing_rule": MISSING_RULE,
        "cohort_mean_small_lesion_2d_mae": _nan_to_none(cohort_mean),
        "per_patient": [_patient_row(m) for m in metrics],
        "unmatched_predictions": [],
    }


def aggregate_patient_metrics(
    patient_arrays: Mapping[str, Mapping[str, np.ndarray]],
    train_masks: Iterable[np.ndarray] | np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute the patient-level small-lesion table and report.

    patient_arrays maps patient_id -> {"pred", "target", "mask"} arrays (2D or
    stacked 3D); train_masks supplies outer-train masks for the frozen q25.
    """
    train_areas = _pooled_train_areas(train_masks)
    if train_areas.size == 0:
        raise MetricAggregationError("outer-train masks hold no 2D lesion components; q25 undefined")
    q25 = float(np.quantile(train_areas, Q25_QUANTILE))
    metrics: list[PatientMetric] = []
    for patient_id in sorted(patient_arrays):
        arrays = patient_arrays[patient_id]
        try:
            pred = arrays[NPZ_PRED_KEY]
            target = arrays[NPZ_TARGET_KEY]
            mask = arrays[NPZ_MASK_KEY]
        except KeyError as exc:
            raise MetricAggregationError(f"patient {patient_id}: missing npz key {exc}") from exc
        metrics.append(compute_patient_metric(patient_id, pred, target, mask, q25))
    frame = pd.DataFrame([asdict(m) for m in metrics])
    report = build_report(metrics, q25, int(train_areas.size))
    return frame, report


# --- C3 acceptance gate (PRD §3; DESIGN §10 "C3 评估器语义") ----------------

def _delta_favorable(delta: float, favor: str) -> bool:
    return delta < 0.0 if favor == "lower" else delta > 0.0


def evaluate_acceptance_gate(
    deltas: Mapping[str, tuple[float, float]], *,
    gate: str, primary: str = METRIC_NAME, ni_margin: float, favor: str = "lower",
) -> dict[str, Any]:
    """Evaluate the C3 acceptance gate (PRD §3; DESIGN §10 semantics).

    deltas maps metric -> (mean_delta, ci_bound), delta = M - D0; the CI bound
    (unfavorable side) is carried for reporting, the gate uses point rules:
    gate_e_isomorphic -> PASS iff the primary's mean delta is favorable AND
    every other metric is ONE-SIDED noninferior (DESIGN v1.0d: favor="lower"
    -> mean_delta <= ni_margin; favor="upper" -> mean_delta >= -ni_margin;
    large favorable deltas automatically pass); strict_all_metrics -> PASS
    iff ALL metrics are favorable ([计划] §2.4).  Unknown gate/favor, negative
    margin, or malformed entries raise ValueError; missing primary forces
    FAIL and is listed in "missing".
    """
    if gate not in ACCEPTANCE_GATES:
        raise ValueError(f"unknown gate {gate!r}; expected one of {ACCEPTANCE_GATES}")
    if favor not in FAVOR_DIRECTIONS:
        raise ValueError(f"unknown favor {favor!r}; expected one of {FAVOR_DIRECTIONS}")
    if ni_margin < 0:
        raise ValueError(f"ni_margin must be >= 0, got {ni_margin}")
    for name, value in deltas.items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"deltas[{name!r}] must be (mean_delta, ci_bound)")
    missing = [primary] if primary not in deltas else []
    primary_ok = primary in deltas and _delta_favorable(deltas[primary][0], favor)
    noninferior_ok_per_metric = {
        # DESIGN v1.0d one-sided rule: only UNfavorable excess fails.
        name: (value[0] <= ni_margin if favor == "lower" else value[0] >= -ni_margin)
        for name, value in sorted(deltas.items())
        if name != primary
    }
    all_metrics_better = bool(deltas) and all(
        _delta_favorable(value[0], favor) for value in deltas.values()
    )
    if gate == "gate_e_isomorphic":
        passed = primary_ok and all(noninferior_ok_per_metric.values()) and not missing
    else:  # strict_all_metrics
        passed = all_metrics_better and not missing
    return {
        "gate": gate,
        "primary": primary,
        "favor": favor,
        "ni_margin": ni_margin,
        "primary_direction_ok": primary_ok,
        "noninferior_ok_per_metric": noninferior_ok_per_metric,
        "all_metrics_better": all_metrics_better,
        "verdict": "PASS" if passed else "FAIL",
        "missing": missing,
    }


def load_deltas_json(path: str | Path) -> dict[str, tuple[float, float]]:
    """Load {metric: [mean_delta, ci_bound]} from JSON (fail-closed)."""
    p = Path(path)
    if not p.is_file():
        raise MetricAggregationError(f"deltas json not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise MetricAggregationError(f"{p}: deltas json must be a non-empty object")
    deltas: dict[str, tuple[float, float]] = {}
    for name, value in payload.items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise MetricAggregationError(f"{p}: deltas[{name!r}] must be [mean_delta, ci_bound]")
        deltas[str(name)] = (float(value[0]), float(value[1]))
    return deltas


def load_patient_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load one per-patient prediction npz; require pred/target/mask keys."""
    p = Path(path)
    if not p.is_file():
        raise MetricAggregationError(f"prediction file missing: {p}")
    with np.load(p) as data:
        found = set(data.files)
        required = (NPZ_PRED_KEY, NPZ_TARGET_KEY, NPZ_MASK_KEY)
        missing = [k for k in required if k not in found]
        if missing:
            raise MetricAggregationError(f"{p}: npz missing keys {missing}, found {sorted(found)}")
        return {k: np.asarray(data[k]) for k in required}


def load_split_manifest(path: str | Path) -> list[str]:
    """Load the evaluated-cohort patient ids from a CSV or JSON manifest."""
    p = Path(path)
    if not p.is_file():
        raise MetricAggregationError(f"split manifest not found: {p}")
    if p.suffix.lower() == ".json":
        ids = _patient_ids_from_json(json.loads(p.read_text(encoding="utf-8")), p)
    elif p.suffix.lower() == ".csv":
        frame = pd.read_csv(p)
        if "patient_id" not in frame.columns:
            raise MetricAggregationError(f"{p}: csv needs 'patient_id', found {list(frame.columns)}")
        ids = [str(v) for v in frame["patient_id"].tolist()]
    else:
        raise MetricAggregationError(f"split manifest must be .json or .csv: {p}")
    _reject_duplicates(ids, p)
    if not ids:
        raise MetricAggregationError(f"split manifest lists no patients: {p}")
    return ids


def _patient_ids_from_json(payload: Any, src: Path) -> list[str]:
    if isinstance(payload, list):
        return [str(v) for v in payload]
    if isinstance(payload, dict):
        for key in ("patients", "patient_ids"):
            if isinstance(payload.get(key), list):
                return [str(v) for v in payload[key]]
    raise MetricAggregationError(f"{src}: split json must be a list or " + '{"patients": [...]}')


def _reject_duplicates(ids: Sequence[str], src: Path) -> None:
    seen: set[str] = set()
    duplicates = {v for v in ids if v in seen or seen.add(v)}
    if duplicates:
        raise MetricAggregationError(f"{src}: duplicate patient ids: {sorted(duplicates)}")


def _manifest_entries(payload: Any, base_dir: Path, src: Path) -> dict[str, Path]:
    if not isinstance(payload, dict):
        raise MetricAggregationError(f"{src}: manifest must be a JSON object")
    raw: dict[str, Any] = payload
    if isinstance(payload.get("files"), list):
        raw = {}
        for item in payload["files"]:
            if not isinstance(item, dict) or "patient_id" not in item:
                raise MetricAggregationError(f"{src}: manifest 'files' entries need 'patient_id'")
            raw[str(item["patient_id"])] = item.get("path")
    entries: dict[str, Path] = {}
    for patient_id, value in raw.items():
        if not isinstance(value, str) or not value.strip():
            raise MetricAggregationError(f"{src}: manifest entry '{patient_id}' must be a path string")
        path = Path(value)
        entries[patient_id] = path if path.is_absolute() else base_dir / path
    return entries


def load_predictions(
    spec: str | Path, patient_ids: Sequence[str]
) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    """Load per-patient prediction arrays; missing files -> fail-closed.

    Returns (patient_id -> arrays mapping, unmatched-extras list).  Any split
    patient without a prediction file raises (CLI exits 1).
    """
    p = Path(spec)
    if p.is_dir():
        mapping = {pid: load_patient_npz(p / (pid + ".npz")) for pid in patient_ids}
        id_set = set(patient_ids)
        extras = sorted(f.stem for f in p.glob("*.npz") if f.stem not in id_set)
    elif p.is_file() and p.suffix.lower() == ".json":
        entries = _manifest_entries(json.loads(p.read_text(encoding="utf-8")), p.parent, p)
        mapping = {
            pid: load_patient_npz(entries[pid]) for pid in patient_ids if pid in entries
        }
        extras = sorted(set(entries) - set(patient_ids))
    else:
        raise MetricAggregationError(
            f"--predictions must be a dir of <patient_id>.npz or a .json manifest, got: {p}"
        )
    missing = sorted(pid for pid in patient_ids if pid not in mapping)
    if missing:
        raise MetricAggregationError(f"missing prediction files (fail-closed): {missing}")
    return mapping, extras


def _mask_from_npz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        for key in NPZ_STACKED_MASK_KEYS:
            if key in data.files:
                return np.asarray(data[key])
        raise MetricAggregationError(f"{path}: no {NPZ_STACKED_MASK_KEYS} key, found {sorted(data.files)}")


def load_outer_train_masks(spec: str | Path) -> list[np.ndarray]:
    """Load outer-train masks (q25 threshold source) as 2D/3D array list."""
    p = Path(spec)
    if p.is_dir():
        files = sorted(p.glob("*.npz"))
        if not files:
            raise MetricAggregationError(f"no .npz mask files under: {p}")
        return [_mask_from_npz(f) for f in files]
    if p.is_file():
        return [_mask_from_npz(p)]
    raise MetricAggregationError(f"--outer-train-masks must be a dir or .npz file, got: {p}")


def write_outputs(
    frame: pd.DataFrame, report: Mapping[str, Any], out_stem: str | Path
) -> tuple[Path, Path]:
    """Write <stem>.csv and <stem>.json; NaN serialised as null in JSON."""
    stem = Path(out_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.parent / (stem.name + ".csv")
    json_path = stem.parent / (stem.name + ".json")
    frame.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return csv_path, json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Patient-level small_lesion_2d_mae aggregation + C3 "
        "acceptance gate ([计划] §5.3, PRD §3; DESIGN RC-BRD_v1 §10)."
    )
    parser.add_argument("--predictions", required=True,
                        help="dir of <patient_id>.npz (pred/target/mask) or manifest")
    parser.add_argument("--split-manifest", required=True,
                        help="csv or json listing patient ids of the evaluated split")
    parser.add_argument("--outer-train-masks", required=True,
                        help="outer-train lesion masks (dir of npz or one npz)")
    parser.add_argument("--out", required=True,
                        help="output stem: writes <out>.csv and <out>.json")
    parser.add_argument("--gate", default=ACCEPTANCE_GATES[0], choices=ACCEPTANCE_GATES,
                        help="C3 acceptance gate; invalid value -> argparse exit 2")
    parser.add_argument("--ni-margin", type=float, default=DEFAULT_NI_MARGIN,
                        help="noninferiority |mean delta| margin (default 0.02)")
    parser.add_argument("--deltas-json", default=None,
                        help="optional {metric: [mean_delta, ci_bound]} json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        patient_ids = load_split_manifest(args.split_manifest)
        predictions, extras = load_predictions(args.predictions, patient_ids)
        train_masks = load_outer_train_masks(args.outer_train_masks)
        frame, report = aggregate_patient_metrics(predictions, train_masks)
        report["unmatched_predictions"] = extras
        if args.deltas_json is not None:  # C3 gate block only when deltas given
            report["acceptance_gate"] = evaluate_acceptance_gate(
                load_deltas_json(args.deltas_json),
                gate=args.gate,
                ni_margin=args.ni_margin,
            )
        csv_path, json_path = write_outputs(frame, report, args.out)
    except (MetricAggregationError, OSError, ValueError) as exc:
        print(f"[aggregate_patient_metrics] FAIL: {exc}", file=sys.stderr)
        return 1
    gate_txt = ""
    if "acceptance_gate" in report:
        gate = report["acceptance_gate"]
        gate_txt = f" gate={gate['gate']}:{gate['verdict']}"
    print(f"[aggregate_patient_metrics] patients={report['patient_count']} "
          f"effective={report['effective_patient_count']} "
          f"q25={report['q25_threshold']}{gate_txt}")
    print(f"  csv : {csv_path}")
    print(f"  json: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
