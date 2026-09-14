"""CLI integration tests for the feature-emergence / causality audit scripts.

These exercise the fail-closed behaviours the formal validation depends on:
  - refusing to write into a non-empty run directory (stale COMPLETE.json hazard)
  - refusing to run A2 on val/test without train-frozen quartiles
  - refusing fallback patient IDs when meta.patient_id is missing
"""

from __future__ import annotations

import json

import pytest

from src.mechanism_validation.feature_emergence import _patient_keys


# ---------------------------------------------------------------------------
# Patient identity fail-closed
# ---------------------------------------------------------------------------


def test_patient_keys_accepts_default_collate_dict_of_lists():
    batch = {
        "ct": None,
        "meta": {
            "patient_id": ["p1", "p2", "p3"],
            "slice_id": [0, 0, 0],
        },
    }
    assert _patient_keys(batch) == ["p1", "p2", "p3"]


def test_patient_keys_accepts_list_of_dicts():
    batch = {
        "meta": [
            {"patient_id": "p1"},
            {"patient_id": "p2"},
        ],
    }
    assert _patient_keys(batch) == ["p1", "p2"]


def test_patient_keys_refuses_missing_meta():
    batch = {"ct": "placeholder"}
    with pytest.raises(ValueError, match="patient_id"):
        _patient_keys(batch)


def test_patient_keys_refuses_empty_patient_id():
    batch = {"meta": {"patient_id": [""]}}
    with pytest.raises(ValueError, match="patient_id"):
        _patient_keys(batch)


def test_patient_keys_refuses_none_patient_id_in_list_of_dicts():
    batch = {"meta": [{"patient_id": None}]}
    with pytest.raises(ValueError, match="patient_id"):
        _patient_keys(batch)


# ---------------------------------------------------------------------------
# Frozen quartiles fail-closed
# ---------------------------------------------------------------------------


def test_load_frozen_quartiles_requires_q1_q2_q3(tmp_path):
    import sys
    sys.path.insert(0, "scripts")
    from eval_feature_emergence import _load_frozen_quartiles

    class _Args:
        frozen_quartiles_json = None
        fake_data = False
        split = "val"

    with pytest.raises(RuntimeError, match="frozen-quartiles-json"):
        _load_frozen_quartiles(_Args(), None, None)


def test_load_frozen_quartiles_accepts_json(tmp_path):
    import sys
    sys.path.insert(0, "scripts")
    from eval_feature_emergence import _load_frozen_quartiles

    payload = {"q1": 10, "q2": 20, "q3": 40}
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    class _Args:
        frozen_quartiles_json = str(path)
        fake_data = False
        split = "val"

    q = _load_frozen_quartiles(_Args(), None, None)
    assert q == [10.0, 20.0, 40.0]


def test_load_frozen_quartiles_ok_on_train_without_json():
    import sys
    sys.path.insert(0, "scripts")
    from eval_feature_emergence import _load_frozen_quartiles

    class _Args:
        frozen_quartiles_json = None
        fake_data = False
        split = "train"

    assert _load_frozen_quartiles(_Args(), None, None) is None
