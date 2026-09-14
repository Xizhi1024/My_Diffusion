"""FR-6.5/FR-6.6 patient-metric, C3-gate, and run-ledger contract tests.

PRD §6 matrix row test_patient_metric_contract.py (DESIGN §11 minimal set,
additions allowed): q25 frozen from outer-train only; in-patient aggregation
first; preregistered missing rule; hand-computed small-sample agreement;
fail-closed ledger checks.  All data is synthetic CPU arrays ([计划] §5.4 M0).
"""

import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.aggregate_patient_metrics import (
    MetricAggregationError,
    aggregate_patient_metrics,
    compute_small_lesion_q25,
    evaluate_acceptance_gate,
    lesion_component_areas,
    load_deltas_json,
    load_outer_train_masks,
    load_patient_npz,
    load_predictions,
    load_split_manifest,
    main as aggregate_main,
)
from scripts.validate_run_ledger import (
    REQUIRED_ARTIFACTS,
    load_ledger,
    main as ledger_main,
    validate_run_ledger,
)

CANVAS = 32
TRAIN_SLICE_AREAS = [1, 2, 3, 4, 5]  # pooled outer-train component areas
# np.quantile linear method: h=(n-1)*0.25=1.0 -> sorted[1] = 2.0 exactly.
MANUAL_Q25 = 2.0
P1_SMALL = 0.15  # mean(0.1, 0.2) over the two lesions with area <= 2
P2_SMALL = 0.4
COHORT_MEAN = 0.275  # mean(0.15, 0.4) across effective patients
LESION_POOLED = (0.1 + 0.2 + 0.4) / 3  # WRONG aggregation, must differ


# ---------------------------------------------------------------------------
# Synthetic samples (blob = 1 x area pixel row; blobs separated by 2 rows)
# ---------------------------------------------------------------------------

def _synth_slice(lesions):
    """lesions: list of (area, abs offset). Returns (pred, target, mask)."""
    mask = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    target = np.zeros((CANVAS, CANVAS), dtype=np.float64)
    pred = np.zeros((CANVAS, CANVAS), dtype=np.float64)
    for idx, (area, offset) in enumerate(lesions):
        row, col = 2 + 3 * idx, 2
        mask[row, col:col + area] = 1
        target[row, col:col + area] = 1.0
        pred[row, col:col + area] = 1.0 - offset
    return pred, target, mask


def _patient(slices):
    """slices: list of per-slice lesion lists -> pred/target/mask stacks."""
    pred = [_synth_slice(s)[0] for s in slices]
    target = [_synth_slice(s)[1] for s in slices]
    mask = [_synth_slice(s)[2] for s in slices]
    return {
        "pred": np.stack(pred),
        "target": np.stack(target),
        "mask": np.stack(mask),
    }


def _train_masks():
    """Outer-train stacks: T1 areas {1,3,5}, T2 areas {2,4} -> pooled [1..5]."""
    t1 = _synth_slice([(1, 0.0), (3, 0.0), (5, 0.0)])[2][None, ...]
    t2 = _synth_slice([(2, 0.0), (4, 0.0)])[2][None, ...]
    return [t1, t2]


def _hand_sample_patients():
    return {
        "P1": _patient([[(1, 0.1), (2, 0.2), (9, 0.5)]]),
        "P2": _patient([[(2, 0.4)], [(16, 0.0)]]),  # two slices
        "P3": _patient([[(9, 0.3), (16, 0.1)]]),
    }


def _write_run_inputs(root, patients, train, split_ids=None):
    """Write prediction npz + outer-train masks + split csv.

    split_ids defaults to the patients keys; pass an explicit list to cover
    patients whose predictions are absent or leave extras unmatched.
    """
    preds = root / "predictions"
    preds.mkdir()
    for pid, arrays in patients.items():
        np.savez(preds / (pid + ".npz"), **arrays)
    train_dir = root / "outer_train"
    train_dir.mkdir()
    for idx, mask in enumerate(train):
        np.savez(train_dir / ("T%d.npz" % idx), mask=mask)
    split = root / "split.csv"
    ids = sorted(patients) if split_ids is None else list(split_ids)
    pd.DataFrame({"patient_id": ids}).to_csv(split, index=False)
    return preds, train_dir, split


# ---------------------------------------------------------------------------
# q25 threshold: outer-train only ([计划] §5.3)
# ---------------------------------------------------------------------------

class TestSmallLesionQ25:
    def test_component_areas_count_pixels(self):
        areas = lesion_component_areas(_synth_slice([(1, 0.0), (3, 0.0), (5, 0.0)])[2])
        assert sorted(areas.tolist()) == [1, 3, 5]

    def test_q25_matches_manual_value(self):
        # Manual: sorted areas [1,2,3,4,5], h=(5-1)*0.25=1 -> value 2 exactly.
        assert compute_small_lesion_q25(_train_masks()) == pytest.approx(MANUAL_Q25)

    def test_q25_empty_train_masks_fail_closed(self):
        with pytest.raises(MetricAggregationError, match="no 2D lesion components"):
            compute_small_lesion_q25([np.zeros((8, 8), dtype=np.uint8)])

    def test_stacked_single_npz_loader(self, tmp_path):
        stack = np.zeros((2, 8, 8), dtype=np.uint8)
        stack[0, 0, :3] = 1
        stack[1, 4, :2] = 1
        path = tmp_path / "train.npz"
        np.savez(path, masks=stack)
        masks = load_outer_train_masks(path)
        # sorted areas [2,3]: h=(2-1)*0.25=0.25 -> 2 + 0.25*(3-2) = 2.25.
        assert compute_small_lesion_q25(masks) == pytest.approx(2.25)


class TestLeakageGuard:
    def test_outer_test_areas_do_not_move_frozen_threshold(self):
        patients = {"P_leak": _patient([[(1, 0.05)]])}  # area 1 < all-train-ish
        frame, report = aggregate_patient_metrics(patients, _train_masks())
        assert report["q25_threshold"] == pytest.approx(MANUAL_Q25)
        row = frame.iloc[0]
        assert row["n_small_lesions"] == 1
        assert row["small_lesion_2d_mae"] == pytest.approx(0.05)

    def test_pooling_test_areas_would_have_changed_q25(self):
        # Counterfactual: if outer-test areas leaked into the threshold, q25
        # of [1(test),1,2,3,4,5] would be 1.25, not the frozen 2.0.
        pooled = float(np.quantile([1] + TRAIN_SLICE_AREAS, 0.25))
        assert pooled != pytest.approx(MANUAL_Q25)
        assert pooled == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# Aggregation semantics ([计划] §5.3)
# ---------------------------------------------------------------------------

class TestPatientAggregation:
    def test_matches_hand_computed_small_sample(self):
        frame, report = aggregate_patient_metrics(
            _hand_sample_patients(), _train_masks()
        )
        by_id = frame.set_index("patient_id")
        assert by_id.loc["P1", "small_lesion_2d_mae"] == pytest.approx(P1_SMALL)
        assert by_id.loc["P2", "small_lesion_2d_mae"] == pytest.approx(P2_SMALL)
        assert math.isnan(by_id.loc["P3", "small_lesion_2d_mae"])
        assert report["q25_threshold"] == pytest.approx(MANUAL_Q25)
        assert report["cohort_mean_small_lesion_2d_mae"] == pytest.approx(COHORT_MEAN)
        # Whole-lesion means are reported alongside (P1: (0.1+0.2+0.5)/3).
        assert by_id.loc["P1", "lesion_2d_mae"] == pytest.approx(0.8 / 3.0)
        assert by_id.loc["P2", "n_slices"] == 2
        assert by_id.loc["P1", "n_small_lesions"] == 2
        assert by_id.loc["P2", "n_small_lesions"] == 1

    def test_in_patient_mean_first_then_across_patients(self):
        frame, report = aggregate_patient_metrics(
            _hand_sample_patients(), _train_masks()
        )
        cohort = report["cohort_mean_small_lesion_2d_mae"]
        # Patient-first (correct): mean(0.15, 0.4) = 0.275.
        assert cohort == pytest.approx(COHORT_MEAN)
        # Lesion-pooled (wrong): (0.1+0.2+0.4)/3 differs -> aggregation order
        # is observable and must follow the preregistered patient-first rule.
        assert cohort != pytest.approx(LESION_POOLED)
        assert COHORT_MEAN != pytest.approx(LESION_POOLED)

    def test_boundary_area_equal_to_q25_counts_as_small(self):
        # P1/P2 area-2 lesions rely on A_l <= q25 being inclusive.
        frame, _ = aggregate_patient_metrics(_hand_sample_patients(), _train_masks())
        by_id = frame.set_index("patient_id")
        assert by_id.loc["P1", "n_small_lesions"] == 2
        assert by_id.loc["P2", "n_small_lesions"] == 1

    def test_missing_rule_and_effective_patient_count(self):
        frame, report = aggregate_patient_metrics(
            _hand_sample_patients(), _train_masks()
        )
        assert report["patient_count"] == 3
        assert report["effective_patient_count"] == 2
        assert report["missing_patients"] == ["P3"]
        assert "retained in the report" in report["missing_rule"]
        # Retained, not silently dropped: P3 keeps a row with NaN metric and
        # is serialised as null (not dropped) in the JSON report.
        assert "P3" in set(frame["patient_id"])
        p3 = [r for r in report["per_patient"] if r["patient_id"] == "P3"][0]
        assert p3["small_lesion_2d_mae"] is None

    def test_shape_mismatch_fails_closed(self):
        bad = {"P1": {
            "pred": np.zeros((1, 4, 4)),
            "target": np.zeros((1, 4, 4)),
            "mask": np.zeros((1, 8, 8)),
        }}
        with pytest.raises(MetricAggregationError, match="shapes differ"):
            aggregate_patient_metrics(bad, _train_masks())


# ---------------------------------------------------------------------------
# Aggregate CLI (DESIGN §10: missing prediction file -> exit 1)
# ---------------------------------------------------------------------------

class TestAggregateCli:
    def _run(self, tmp_path, patients):
        preds, train_dir, split = _write_run_inputs(
            tmp_path, patients, _train_masks()
        )
        out = tmp_path / "metrics"
        rc = aggregate_main(["--predictions", str(preds),
                             "--split-manifest", str(split),
                             "--outer-train-masks", str(train_dir),
                             "--out", str(out)])
        return rc, out

    def test_cli_end_to_end_matches_hand_values(self, tmp_path):
        rc, out = self._run(tmp_path, _hand_sample_patients())
        assert rc == 0
        assert (tmp_path / "metrics.csv").is_file()
        report = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(tmp_path / "metrics.csv").set_index("patient_id")
        assert frame.loc["P1", "small_lesion_2d_mae"] == pytest.approx(P1_SMALL)
        assert frame.loc["P2", "small_lesion_2d_mae"] == pytest.approx(P2_SMALL)
        assert math.isnan(frame.loc["P3", "small_lesion_2d_mae"])
        assert report["q25_threshold"] == pytest.approx(MANUAL_Q25)
        assert report["effective_patient_count"] == 2
        assert report["missing_patients"] == ["P3"]
        assert report["cohort_mean_small_lesion_2d_mae"] == pytest.approx(COHORT_MEAN)
        assert report["unmatched_predictions"] == []
        assert "acceptance_gate" not in report  # no --deltas-json -> no block

    def test_missing_prediction_file_exits_1(self, tmp_path, capsys):
        patients = _hand_sample_patients()
        patients.pop("P2")  # split manifest below still lists P1,P2,P3
        preds, train_dir, split = _write_run_inputs(
            tmp_path, patients, _train_masks(), split_ids=["P1", "P2", "P3"]
        )
        rc = aggregate_main(["--predictions", str(preds),
                             "--split-manifest", str(split),
                             "--outer-train-masks", str(train_dir),
                             "--out", str(tmp_path / "metrics")])
        assert rc == 1
        assert "P2" in capsys.readouterr().err
        with pytest.raises(MetricAggregationError, match="P2"):
            load_predictions(preds, ["P1", "P2", "P3"])

    def test_npz_missing_required_key_raises(self, tmp_path):
        path = tmp_path / "P9.npz"
        np.savez(path, pred=np.zeros((1, 4, 4)), target=np.zeros((1, 4, 4)))
        with pytest.raises(MetricAggregationError, match="mask"):
            load_patient_npz(path)

    def test_extra_prediction_files_are_reported_not_silent(self, tmp_path):
        patients = _hand_sample_patients()
        patients["PX"] = _patient([[(1, 0.0)]])
        preds, train_dir, split = _write_run_inputs(
            tmp_path,
            patients,
            _train_masks(),
            split_ids=["P1", "P2", "P3"],  # PX stays unmatched
        )
        rc = aggregate_main(["--predictions", str(preds),
                             "--split-manifest", str(split),
                             "--outer-train-masks", str(train_dir),
                             "--out", str(tmp_path / "metrics")])
        assert rc == 0
        report = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
        assert report["unmatched_predictions"] == ["PX"]
        assert "PX" not in {r["patient_id"] for r in report["per_patient"]}

    def test_split_manifest_json_and_duplicate_rejection(self, tmp_path):
        j = tmp_path / "split.json"
        j.write_text(json.dumps({"patients": ["P1", "P2"]}), encoding="utf-8")
        assert load_split_manifest(j) == ["P1", "P2"]
        d = tmp_path / "dup.json"
        d.write_text(json.dumps(["P1", "P1"]), encoding="utf-8")
        with pytest.raises(MetricAggregationError, match="duplicate"):
            load_split_manifest(d)


class TestAcceptanceGate:
    # primary favorable (lower better); others within margin 0.02
    DELTAS = {"small_lesion_2d_mae": (-0.05, -0.01),
              "whole_image_mae": (0.01, 0.02), "ssim": (0.005, 0.01)}

    def test_gate_e_isomorphic_primary_unfavorable_fails(self):
        deltas = dict(self.DELTAS, small_lesion_2d_mae=(0.05, 0.08))
        gate = evaluate_acceptance_gate(deltas, gate="gate_e_isomorphic", ni_margin=0.02)
        assert gate["verdict"] == "FAIL"
        assert gate["primary_direction_ok"] is False

    def test_gate_e_isomorphic_one_metric_beyond_margin_fails(self):
        # Large UNfavorable secondary delta (+0.5 >> margin) breaks the gate.
        deltas = dict(self.DELTAS, whole_image_mae=(0.5, 0.6))
        gate = evaluate_acceptance_gate(deltas, gate="gate_e_isomorphic", ni_margin=0.02)
        assert gate["verdict"] == "FAIL"
        assert gate["noninferior_ok_per_metric"]["whole_image_mae"] is False
        assert gate["primary_direction_ok"] is True  # primary alone insufficient

    def test_one_sided_margin_large_favorable_passes(self):
        # DESIGN v1.0d: |delta| rule replaced by one-sided margin — a large
        # FAVORABLE secondary delta (lower better) must stay noninferior.
        deltas = dict(self.DELTAS, whole_image_mae=(-0.5, -0.6))
        gate = evaluate_acceptance_gate(deltas, gate="gate_e_isomorphic", ni_margin=0.02)
        assert gate["noninferior_ok_per_metric"]["whole_image_mae"] is True
        assert gate["verdict"] == "PASS"

    def test_gate_e_isomorphic_all_noninferior_passes(self):
        gate = evaluate_acceptance_gate(self.DELTAS, gate="gate_e_isomorphic", ni_margin=0.02)
        assert gate["verdict"] == "PASS"
        assert all(gate["noninferior_ok_per_metric"].values())
        assert gate["missing"] == []

    def test_strict_all_metrics_any_unfavorable_fails(self):
        gate = evaluate_acceptance_gate(self.DELTAS, gate="strict_all_metrics", ni_margin=0.02)
        assert gate["verdict"] == "FAIL"
        assert gate["all_metrics_better"] is False
        good = {"small_lesion_2d_mae": (-0.05, -0.01), "whole_image_mae": (-0.01, -0.02)}
        passing = evaluate_acceptance_gate(good, gate="strict_all_metrics", ni_margin=0.02)
        assert passing["verdict"] == "PASS" and passing["all_metrics_better"] is True

    def test_invalid_gate_favor_margin_raise(self):
        with pytest.raises(ValueError, match="unknown gate"):
            evaluate_acceptance_gate(self.DELTAS, gate="bogus", ni_margin=0.02)
        with pytest.raises(ValueError, match="unknown favor"):
            evaluate_acceptance_gate(self.DELTAS, gate="gate_e_isomorphic",
                                     ni_margin=0.02, favor="sideways")
        with pytest.raises(ValueError, match="ni_margin"):
            evaluate_acceptance_gate(self.DELTAS, gate="gate_e_isomorphic", ni_margin=-0.1)

    def test_missing_primary_forces_fail(self):
        gate = evaluate_acceptance_gate(
            {"whole_image_mae": (0.0, 0.0)}, gate="gate_e_isomorphic", ni_margin=0.02)
        assert gate["verdict"] == "FAIL"
        assert gate["missing"] == ["small_lesion_2d_mae"]

    def test_favor_upper_flips_favorable_direction(self):
        good = {"small_lesion_2d_mae": (0.05, 0.01), "ssim": (0.5, 0.6)}
        gate = evaluate_acceptance_gate(
            good, gate="gate_e_isomorphic", ni_margin=0.02, favor="upper")
        assert gate["primary_direction_ok"] is True and gate["verdict"] == "PASS"
        assert gate["noninferior_ok_per_metric"]["ssim"] is True  # big favorable
        bad = {"small_lesion_2d_mae": (0.05, 0.01), "ssim": (-0.5, -0.6)}
        gate = evaluate_acceptance_gate(
            bad, gate="gate_e_isomorphic", ni_margin=0.02, favor="upper")
        assert gate["noninferior_ok_per_metric"]["ssim"] is False
        assert gate["verdict"] == "FAIL"

    def test_cli_gate_flag_invalid_exits_2(self, capsys):
        with pytest.raises(SystemExit) as exc:
            aggregate_main(["--predictions", "x", "--split-manifest", "y",
                            "--outer-train-masks", "z", "--out", "o",
                            "--gate", "bogus"])
        assert exc.value.code == 2
        assert "invalid choice" in capsys.readouterr().err

    def test_cli_deltas_json_round_trip(self, tmp_path):
        preds, train_dir, split = _write_run_inputs(
            tmp_path, _hand_sample_patients(), _train_masks()
        )
        deltas_path = tmp_path / "deltas.json"
        deltas_path.write_text(json.dumps(self.DELTAS), encoding="utf-8")
        argv = ["--predictions", str(preds), "--split-manifest", str(split),
                "--outer-train-masks", str(train_dir), "--out", str(tmp_path / "m"),
                "--deltas-json", str(deltas_path), "--ni-margin", "0.02"]
        assert aggregate_main(argv) == 0
        report = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
        gate = report["acceptance_gate"]
        assert gate["gate"] == "gate_e_isomorphic" and gate["verdict"] == "PASS"
        assert gate["ni_margin"] == 0.02
        # FAIL verdict keeps exit 0: the gate result lives in the JSON (§10).
        deltas_path.write_text(
            json.dumps(dict(self.DELTAS, whole_image_mae=(0.5, 0.6))), encoding="utf-8")
        assert aggregate_main(argv) == 0
        gate2 = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
        assert gate2["acceptance_gate"]["verdict"] == "FAIL"
        ni = gate2["acceptance_gate"]["noninferior_ok_per_metric"]
        assert ni["whole_image_mae"] is False

    def test_deltas_json_malformed_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"metric": [1.0]}), encoding="utf-8")
        with pytest.raises(MetricAggregationError, match="mean_delta"):
            load_deltas_json(bad)


# ---------------------------------------------------------------------------
# Run-ledger validation (FR-6.6, [计划] §5.1)
# ---------------------------------------------------------------------------

def _full_ledger():
    return {
        "run_id": "rc_brd_fold0_m", "git_commit": "a" * 40, "git_dirty": False,
        "config": {"modules": {"rc_brd": {"enabled": True}}},
        "config_sha256": "b" * 64, "env_lock": {"pixi": "pixi.lock"},
        "gpu": "1x RTX 4090 (cuda:0)", "seed": 20260914,
        "started_at": "2026-09-14T00:00:00+00:00",
        "ended_at": "2026-09-15T00:00:00+00:00", "outer_fold": "fold0",
        "outer_train_patient_hash": "c" * 16, "outer_test_patient_hash": "d" * 16,
        "inner_split_hash": "e" * 16,
        "mean_checkpoint": "checkpoints/mean_fold0.pt",
        "mean_checkpoint_sha256": "f" * 64,
        "diffusion_checkpoint": "checkpoints/m_fold0.pt",
        "diffusion_checkpoint_sha256": "1" * 64,
        "contract_artifact": "artifacts/fold0/recoverability_contract.json",
        "contract_sha256": "2" * 64,
        "patient_metrics_table": "results/fold0/patient_metrics.csv",
        "per_fold_summary": "results/fold0/summary.json",
        "prediction_manifest": "results/fold0/predictions.json",
        "checkpoint_policy": "fixed_final_ema", "nfe": 50,
        "sampling_seed": 20260914, "deviation_log": [],
    }


class TestRunLedger:
    def test_complete_ledger_passes(self, tmp_path):
        report = validate_run_ledger(_full_ledger())
        assert report["all_ok"] is True and report["n_failed"] == 0
        assert report["n_required"] == len(REQUIRED_ARTIFACTS)
        path = tmp_path / "ledger.json"
        path.write_text(json.dumps(_full_ledger()), encoding="utf-8")
        assert ledger_main(["--ledger", str(path)]) == 0
        assert load_ledger(path)["run_id"] == "rc_brd_fold0_m"

    @pytest.mark.parametrize("key", [spec[0] for spec in REQUIRED_ARTIFACTS])
    def test_missing_any_required_key_fails(self, key, tmp_path):
        ledger = _full_ledger()
        del ledger[key]
        report = validate_run_ledger(ledger)
        assert report["all_ok"] is False and key in report["missing_required"]
        entry = report["checks"][key]
        assert entry["ok"] is False and entry["missing"] is True
        assert isinstance(entry["detail"], str) and entry["detail"]
        path = tmp_path / "ledger.json"
        path.write_text(json.dumps(ledger), encoding="utf-8")
        assert ledger_main(["--ledger", str(path)]) == 1

    def test_invalid_values_fail(self):
        bad_values = [
            ("run_id", "   "), ("config_sha256", "z" * 64), ("config_sha256", "abc"),
            ("nfe", 0), ("nfe", "50"), ("git_dirty", "false"), ("seed", 42.0),
            ("deviation_log", "none"), ("gpu", None), ("env_lock", {}),
            ("outer_fold", ""),
        ]
        for key, bad in bad_values:
            ledger = _full_ledger()
            ledger[key] = bad
            report = validate_run_ledger(ledger, required=[key])
            assert report["all_ok"] is False, (key, bad)
            assert report["checks"][key] == {
                "ok": False, "missing": False, "detail": report["checks"][key]["detail"]}

    def test_require_artifacts_subset_passes_despite_other_missing(self, tmp_path):
        ledger = _full_ledger()
        del ledger["nfe"]
        subset = validate_run_ledger(ledger, required=["run_id", "git_commit"])
        assert subset["all_ok"] is True
        path = tmp_path / "ledger.json"
        path.write_text(json.dumps(ledger), encoding="utf-8")
        rc = ledger_main(["--ledger", str(path),
                          "--require-artifacts", "run_id,git_commit"])
        assert rc == 0
        assert ledger_main(["--ledger", str(path)]) == 1  # default: full list

    def test_unknown_artifact_name_is_usage_error(self, tmp_path):
        with pytest.raises(ValueError, match="unknown artifact names"):
            validate_run_ledger(_full_ledger(), required=["not_a_key"])
        path = tmp_path / "ledger.json"
        path.write_text(json.dumps(_full_ledger()), encoding="utf-8")
        assert ledger_main([
            "--ledger", str(path), "--require-artifacts", "not_a_key",
        ]) == 2

    def test_missing_ledger_file_fails_closed(self, tmp_path):
        assert ledger_main(["--ledger", str(tmp_path / "absent.json")]) == 1
