from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.eval_oracle_destination import (
    _resolve_conditional_mean_checkpoint,
    action_map_code,
    enumerate_action_maps,
    select_oracle_rows,
)
from tests.test_identifiable_router_repair import _repair_config


def test_action_map_enumeration_is_exact_and_stable() -> None:
    binary = enumerate_action_maps(("native", "shallow"))
    ternary = enumerate_action_maps(("native", "shallow", "null"))

    assert len(binary) == 2**6
    assert len(ternary) == 3**6
    assert action_map_code(binary[0]) == "NNN-NNN"
    assert action_map_code(ternary[-1]) == "000-000"
    assert len({action_map_code(actions) for actions in ternary}) == 3**6


def test_missing_external_mean_self_bootstraps_from_evaluated_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "ckpt_best_combined.pt"
    checkpoint = {
        "model": {
            "mean_predictor.example": torch.tensor(1.0),
            "unet.example": torch.tensor(2.0),
        }
    }
    torch.save(checkpoint, checkpoint_path)
    config = {
        "modules": {
            "conditional_mean": {
                "enabled": True,
                "checkpoint": "results/missing/mean.pt",
                "checkpoint_format": "mean",
            }
        }
    }

    resolution = _resolve_conditional_mean_checkpoint(
        config,
        evaluated_checkpoint_path=checkpoint_path,
        evaluated_checkpoint=checkpoint,
    )

    mean = config["modules"]["conditional_mean"]
    assert resolution["mode"] == "evaluated_checkpoint_self_bootstrap"
    assert mean["checkpoint"] == str(checkpoint_path.resolve())
    assert mean["checkpoint_format"] == "full_model"


def test_oracle_selection_uses_complete_per_sample_action_maps() -> None:
    baseline = [
        {"sample_id": "a", "patient_id": "p1", "loss": 1.0},
        {"sample_id": "b", "patient_id": "p2", "loss": 1.0},
        {"sample_id": "c", "patient_id": "p3", "loss": 0.1},
    ]
    native = [
        {"sample_id": "a", "patient_id": "p1", "loss": 0.6},
        {"sample_id": "b", "patient_id": "p2", "loss": 1.2},
        {"sample_id": "c", "patient_id": "p3", "loss": 0.3},
    ]
    shallow = [
        {"sample_id": "a", "patient_id": "p1", "loss": 0.8},
        {"sample_id": "b", "patient_id": "p2", "loss": 0.4},
        {"sample_id": "c", "patient_id": "p3", "loss": 0.2},
    ]
    action_maps = {
        "NNN-NNN": ("native",) * 6,
        "SSS-SSS": ("shallow",) * 6,
    }

    forced, fallback = select_oracle_rows(
        baseline,
        {
            "NNN-NNN": native,
            "SSS-SSS": shallow,
        },
        action_maps,
        objective="loss",
        lower_is_better=True,
        learned_fallback=True,
    )

    assert [row["oracle_selected_code"] for row in forced] == [
        "NNN-NNN",
        "SSS-SSS",
        "SSS-SSS",
    ]
    assert [row["oracle_objective_value"] for row in forced] == pytest.approx(
        [0.6, 0.4, 0.2]
    )
    assert [row["oracle_selected_code"] for row in fallback] == [
        "NNN-NNN",
        "SSS-SSS",
        "LEARNED",
    ]
    assert fallback[-1]["oracle_objective_improvement"] == pytest.approx(0.0)


def test_per_band_route_action_intervention_is_exact_and_state_free(
    tmp_path: Path,
) -> None:
    from src.model.slmf_bbdm import SLMFBBDM

    model = SLMFBBDM.from_config(_repair_config(tmp_path))
    router = model.residual_preconditioner
    router.set_training_epoch(5)
    router.eval()
    state_before = {
        key: value.detach().clone()
        for key, value in router.state_dict().items()
    }
    residual = torch.randn(2, 1, 32, 32)
    ct = torch.randn_like(residual)
    with torch.no_grad():
        _, baseline = router(
            residual,
            torch.tensor([20, 20]),
            model.noise_schedule,
            ct,
        )
        router.set_inference_route_action_intervention(
            actions_l2=("native", "shallow", "null"),
            actions_l1=("null", "native", "shallow"),
        )
        _, diagnostics = router(
            residual,
            torch.tensor([20, 20]),
            model.noise_schedule,
            ct,
        )

    baseline_active_l2 = baseline["routes_l2"][..., :2].sum(dim=-1)
    baseline_active_l1 = baseline["routes_l1"][..., :2].sum(dim=-1)
    applied_l2 = diagnostics["routes_l2"]
    applied_l1 = diagnostics["routes_l1"]
    torch.testing.assert_close(applied_l2[:, 0, 0], baseline_active_l2[:, 0])
    torch.testing.assert_close(applied_l2[:, 0, 1], torch.zeros(2))
    torch.testing.assert_close(applied_l2[:, 1, 0], torch.zeros(2))
    torch.testing.assert_close(applied_l2[:, 1, 1], baseline_active_l2[:, 1])
    torch.testing.assert_close(
        applied_l2[:, 2],
        torch.tensor([0.0, 0.0, 1.0]).expand(2, -1),
    )
    torch.testing.assert_close(
        applied_l1[:, 0],
        torch.tensor([0.0, 0.0, 1.0]).expand(2, -1),
    )
    torch.testing.assert_close(applied_l1[:, 1, 0], baseline_active_l1[:, 1])
    torch.testing.assert_close(applied_l1[:, 1, 1], torch.zeros(2))
    torch.testing.assert_close(applied_l1[:, 2, 0], torch.zeros(2))
    torch.testing.assert_close(applied_l1[:, 2, 1], baseline_active_l1[:, 2])
    torch.testing.assert_close(
        applied_l2.sum(dim=-1),
        torch.ones_like(applied_l2[..., 0]),
    )
    torch.testing.assert_close(
        applied_l1.sum(dim=-1),
        torch.ones_like(applied_l1[..., 0]),
    )
    assert router.inference_route_action_intervention() == {
        "enabled": True,
        "actions_l2": ["native", "shallow", "null"],
        "actions_l1": ["null", "native", "shallow"],
        "band_order": ["LH", "HL", "HH"],
    }

    state_after = router.state_dict()
    assert state_after.keys() == state_before.keys()
    for key, value in state_before.items():
        torch.testing.assert_close(state_after[key], value)

    router.set_inference_route_action_intervention()
    assert router.inference_route_action_intervention() == {"enabled": False}
    with pytest.raises(ValueError, match="both actions_l2 and actions_l1"):
        router.set_inference_route_action_intervention(
            actions_l2=("native", "native", "native"),
        )
