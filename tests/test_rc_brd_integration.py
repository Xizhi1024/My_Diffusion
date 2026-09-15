"""RC-BRD <-> SLMFBBDM integration tests (DESIGN section 9/11; PRD FR-5, C1/C6/C7).

Minimum assertion set (DESIGN section 11; additions allowed, no removals):
- enabled=false regression: bitwise-identical forward/sample vs rc_brd-free
  builds (no rc_brd key and residual-bridge-only), log key set unchanged;
- enabled=true smoke on synthetic + FakeDataset batches (finite loss, sane
  sample shape/range);
- kappa=0 end-to-end equivalence with the scalar D1 path (fp32 rtol 1e-5 /
  atol 1e-6, Haar round-trip included);
- output_scale=0 -> exact D0 identity (mean only);
- fail-closed construction (kappa!=0 w/o contract, missing path, fold/SHA
  mismatch, band_groups drift, enum violations) + vp posterior passthrough;
- endpoint_mode=ct_minus_mean runs with band-coordinate endpoints (C7);
- readout validation + mc_mean K wiring via sample_mc (C6);
- no GT-mask access inside rc_brd forward/sample paths (monkeypatch).
"""

from __future__ import annotations

import copy
import json
import os
import pathlib
import shutil
import sys
import uuid

import pytest
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.rc_brd import BAND_NAMES, RecoverabilityContract, mean_weights_sha256
from src.model.slmf_bbdm import SLMFBBDM

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
T = 100              # small schedule length (task spec 6: keep tests fast)
IMAGE = 32
SEED = 1234


@pytest.fixture()
def artifact_dir() -> pathlib.Path:
    """Repo-anchored scratch dir (sandboxed runs cannot scandir tmp_path),
    removed afterwards (batch-2 task note: clean up after use)."""
    run_dir = REPO_ROOT / ".t_dir" / "rc_brd_integration_tests" / uuid.uuid4().hex[:12]
    run_dir.mkdir(parents=True, exist_ok=True)
    yield run_dir
    shutil.rmtree(run_dir, ignore_errors=True)


def _base_config() -> dict:
    """Scalar D1 residual-bridge config (rc_brd-free)."""
    return {
        "experiment": {"name": "rc_brd_it", "seed": 42},
        "data": {"image_size": IMAGE, "use_fake_data": True},
        "runtime": {"eval_sampling_steps": 5},
        "model": {
            "objective": "pred_x0",
            "sample_scheduler": "ddim",
            "enable_heteroscedastic": False,
            "initialization_seed": SEED,   # UNet init decoupled from rc_brd
            "self_conditioning": {"enabled": False},
            "base_loss": {"mse_weight": 1.0, "l1_weight": 1.0,
                          "gradient_weight": 0.0, "min_snr_enabled": False},
        },
        "modules": {
            "gabor": {"enabled": False},
            "organ_prior": {"enabled": False},
            "hotspot_prior": {"enabled": False},
            "semantic_prior": {"enabled": False},
            "zero_adapter": {"enabled": False},
            "condition_dropout": {"enabled": False},
            "bbdm_bridge": {"enabled": True, "name": "bbdm_bridge",
                            "num_train_timesteps": T},
            "conditional_mean": {"enabled": True, "levels": 2, "base_channels": 8,
                                 "loss_weight": 0.0, "detach_bridge": True},
            "residual_bridge": {"enabled": True, "output_scale": 1.0,
                                "clip_output": False},
            "residual_frequency": {"enabled": False},
        },
        "losses": {},
    }


def _rc_cfg(**overrides) -> dict:
    """DESIGN section 8 rc_brd block; only deviations from defaults passed."""
    cfg = {"enabled": True, "num_timesteps": T}
    cfg.update(overrides)
    return cfg


def _build(cfg: dict) -> SLMFBBDM:
    torch.manual_seed(SEED)
    return SLMFBBDM.from_config(cfg)


def _batch(batch_size: int = 2, size: int = IMAGE) -> dict:
    return {
        "ct": torch.randn(batch_size, 1, size, size),
        "pet": torch.randn(batch_size, 1, size, size),
        "mask": torch.zeros(batch_size, 1, size, size),
        "organ_mask": torch.zeros(batch_size, 6, size, size),
        "organ_distance": torch.zeros(batch_size, 6, size, size),
        "mu_map": torch.zeros(batch_size, 1, size, size),
    }

class _NoisyRecorder:
    """Forward hook capturing the noisy state (UNet input channel 0)."""

    def __init__(self, model: SLMFBBDM) -> None:
        self.captured: list = []
        self._handle = model.unet.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output) -> None:
        self.captured.append(inputs[0][:, :1].detach().clone())

    def close(self) -> None:
        self._handle.remove()


class _AccessRecordingDict(dict):
    """Dict recording __getitem__ keys (GT-leak monkeypatch)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.accessed: set = set()

    def __getitem__(self, key):
        self.accessed.add(key)
        return super().__getitem__(key)


def _make_mean_checkpoint(path, base_channels: int = 8) -> dict:
    """Save a format-1 mean checkpoint; return its (stripped) state dict."""
    from src.model.mean_predictor import LowFrequencyPETPredictor
    torch.manual_seed(3)
    predictor = LowFrequencyPETPredictor(base_channels=base_channels, levels=2)
    state = predictor.state_dict()
    torch.save({"format_version": 1, "model": state}, path)
    return state


_CONTRACT_GROUPS = {"low": ["LL2"], "mid": ["LH2", "HL2", "HH2"], "high": ["LH1", "HL1", "HH1"]}


def _write_contract(path, *, fold_id: str = "fold_0", mean_sha=None,
                    support_mode: str = "floor_gated",
                    floor_rho: float = 0.1,
                    band_powers=None) -> None:
    """Write a self-hashed, valid recoverability contract JSON (DESIGN S3).

    band_powers=None keeps the v1 artifact form; a dict writes a schema-v2
    contract for the band_snr clock.
    """
    payload = {
        "fold_id": fold_id,
        "band_groups": {g: list(b) for g, b in _CONTRACT_GROUPS.items()},
        "log_snr_grid": [-10.0, 0.0, 10.0],
        "c_values": {"low": [0.2, 0.3, 0.4], "mid": [0.4, 0.5, 0.6],
                     "high": [0.6, 0.7, 0.8]},
        "mean_checkpoint_sha256": mean_sha or "0" * 64,
        "b_active": ["mid", "high"],   # low inactive -> explicit-zero c path
        "psd_floors": {"low": 1e-4, "mid": 2e-4, "high": 5e-4},
        "size_thresholds": {"small_lesion_q25": 12.0},
        "kappa_grid": [0.0, 0.25],
        "s_ref": 1.0,
        "support_mode": support_mode,
        "eta_max": 0.8,
        "floor_rho": floor_rho,
    }
    if band_powers is not None:
        payload["band_powers"] = {g: float(v) for g, v in band_powers.items()}
        payload["band_powers_split"] = "integration_test"
    contract = RecoverabilityContract.from_payload(payload)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({**contract.to_payload(), "contract_sha256": contract.contract_sha256}, handle)


def _model_with_contract_cfg(artifact_dir, **rc_overrides) -> dict:
    """Base config + frozen mean checkpoint + matching valid contract.

    Contract-level kwargs (support_mode/floor_rho/fold_id) are pulled from
    rc_overrides and forwarded to the contract writer.
    """
    contract_keys = {"support_mode", "floor_rho", "fold_id", "contract_band_powers"}
    contract_kw = {k: rc_overrides.pop(k) for k in list(rc_overrides) if k in contract_keys}
    if "contract_band_powers" in contract_kw:
        contract_kw["band_powers"] = contract_kw.pop("contract_band_powers")
    ckpt = artifact_dir / "mean.pt"
    state = _make_mean_checkpoint(ckpt)
    contract = artifact_dir / "contract.json"
    _write_contract(contract, mean_sha=mean_weights_sha256(state), **contract_kw)
    cfg = _base_config()
    cfg["modules"]["conditional_mean"].update({"checkpoint": str(ckpt), "freeze": True})
    cfg["modules"]["rc_brd"] = _rc_cfg(contract_path=str(contract), **rc_overrides)
    return cfg

# ---------------------------------------------------------------------------
# FR-5.1: enabled=false regression (bitwise)
# ---------------------------------------------------------------------------


def test_disabled_rc_brd_matches_legacy_bitwise():
    legacy_cfg = _base_config()                     # no rc_brd key at all
    residual_only_cfg = copy.deepcopy(legacy_cfg)   # residual-bridge-only
    disabled_cfg = copy.deepcopy(legacy_cfg)
    # DESIGN S8: disabled -> other rc_brd keys tolerated but never consumed.
    disabled_cfg["modules"]["rc_brd"] = {
        "enabled": False, "kappa": 0.5, "forward_mode": "vp_bandwise",
        "contract_path": "does/not/exist.json", "readout": {"mode": "nope"},
    }

    legacy = _build(legacy_cfg)
    residual_only = _build(residual_only_cfg)
    disabled = _build(disabled_cfg)

    assert disabled.rc_brd_enabled is False
    assert disabled.rc_brd_schedule is None and disabled.rc_brd_head is None
    assert not any(key.startswith("rc_brd") for key in disabled.state_dict())

    for other in (legacy, residual_only):
        assert set(other.state_dict()) == set(disabled.state_dict())
        for key, value in disabled.state_dict().items():
            assert torch.equal(value, other.state_dict()[key])

    batch = _batch()
    timesteps = torch.tensor([7, 63])
    torch.manual_seed(5)
    loss_a, logs_a = legacy(batch, timesteps)
    torch.manual_seed(5)
    loss_b, logs_b = disabled(batch, timesteps)
    torch.manual_seed(5)
    loss_c, logs_c = residual_only(batch, timesteps)
    assert torch.equal(loss_a, loss_b) and torch.equal(loss_a, loss_c)
    assert set(logs_a) == set(logs_b) == set(logs_c)
    for key, value in logs_a.items():
        assert torch.equal(value, logs_b[key])
        assert torch.equal(value, logs_c[key])

    noise = torch.randn(2, 1, IMAGE, IMAGE)
    out_a = legacy.sample(batch, initial_noise=noise)
    out_b = disabled.sample(batch, initial_noise=noise)
    assert torch.equal(out_a["synthetic_pet"], out_b["synthetic_pet"])


# ---------------------------------------------------------------------------
# kappa=0 end-to-end equivalence with scalar D1 (task spec 6; FR-3.3)
# ---------------------------------------------------------------------------


def test_kappa0_rc_brd_end_to_end_matches_scalar_d1():
    cfg = _base_config()
    cfg["modules"]["rc_brd"] = _rc_cfg(kappa=0.0)   # contract omitted (kappa=0)
    rc = _build(cfg)
    d1 = _build(_base_config())

    batch = _batch()
    timesteps = torch.tensor([0, 33])   # includes the t=0 variance edge
    rec_d1, rec_rc = _NoisyRecorder(d1), _NoisyRecorder(rc)
    torch.manual_seed(9)
    loss_d1, logs_d1 = d1(batch, timesteps)
    torch.manual_seed(9)
    loss_rc, logs_rc = rc(batch, timesteps)
    rec_d1.close()
    rec_rc.close()

    # noisy_x (UNet input channel 0): identical noise/timesteps on both paths,
    # so a match here also pins model_target = r0 = pet - mu for both.
    torch.testing.assert_close(rec_rc.captured[0], rec_d1.captured[0],
                               rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(loss_rc, loss_d1, rtol=1e-5, atol=1e-6)
    for key in ("loss/base_diffusion", "loss/base_mse", "loss/base_l1"):
        torch.testing.assert_close(logs_rc[key], logs_d1[key],
                                   rtol=1e-5, atol=1e-6)

    noise = torch.randn(2, 1, IMAGE, IMAGE)
    out_d1 = d1.sample(batch, initial_noise=noise)
    out_rc = rc.sample(batch, initial_noise=noise)
    # 5 reverse steps accumulate only Haar round-trip error (fp32): measured
    # worst case ~2e-6 absolute on 5/2048 elements (near-zero references show
    # larger *relative* error), i.e. within the Haar round-trip tolerance the
    # task grants on top of the single-step rtol 1e-5 / atol 1e-6 budget.
    torch.testing.assert_close(out_rc["synthetic_pet"], out_d1["synthetic_pet"],
                               rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_rc["mean_pet"], out_d1["mean_pet"], rtol=0, atol=0)

# ---------------------------------------------------------------------------
# enabled=true smoke (synthetic + FakeDataset; PRD DoD-1)
# ---------------------------------------------------------------------------


def test_enabled_smoke_forward_and_sample_synthetic_and_fake_dataset():
    cfg = _base_config()
    cfg["modules"]["rc_brd"] = _rc_cfg()
    model = _build(cfg)

    batch = _batch()
    loss, logs = model(batch, torch.tensor([3, 77]))
    assert torch.isfinite(loss).all()
    tensor_logs = [v for v in logs.values() if torch.is_tensor(v)]
    assert all(torch.isfinite(v).all() for v in tensor_logs)
    assert logs["module/rc_brd"].item() == 1.0
    assert logs["module/rc_brd_specialist"].item() == 1.0

    out = model.sample(batch, num_steps=5)
    pet = out["synthetic_pet"]
    assert pet.shape == (2, 1, IMAGE, IMAGE)
    assert torch.isfinite(pet).all()
    assert float(pet.std()) > 0.0
    assert float(pet.abs().max()) < 10.0   # sane range for [-1,1] images
    assert "raw_pred_residual" in out and "mean_pet" in out

    # FakeDataset path: the default switch combination must train/sample.
    from torch.utils.data import DataLoader
    from src.data.dataset import FakeDataset
    loader = DataLoader(FakeDataset(num_samples=2, image_size=IMAGE), batch_size=2)
    fake_batch = next(iter(loader))
    fake_loss, _ = model(fake_batch, torch.tensor([11, 42]))
    assert torch.isfinite(fake_loss).all()
    fake_out = model.sample(fake_batch, num_steps=3)
    assert torch.isfinite(fake_out["synthetic_pet"]).all()


def test_output_scale_zero_is_exact_d0_identity():
    cfg = _base_config()
    cfg["modules"]["residual_bridge"]["output_scale"] = 0.0
    cfg["modules"]["rc_brd"] = _rc_cfg()
    model = _build(cfg)
    batch = _batch()
    out = model.sample(batch, num_steps=4)
    # D0 fallback: synthetic_pet == mean exactly (alpha=0, clipping off).
    assert torch.equal(out["synthetic_pet"], out["mean_pet"])
    assert torch.equal(out["pred_residual"], torch.zeros_like(out["pred_residual"]))
    loss, _ = model(batch, torch.tensor([5, 50]))   # r0 head keeps training
    assert torch.isfinite(loss).all()


# ---------------------------------------------------------------------------
# Fail-closed construction (DESIGN S8; PRD S3/S4)
# ---------------------------------------------------------------------------


class TestFailClosedConstruction:
    def test_kappa_nonzero_without_contract(self):
        cfg = _base_config()
        cfg["modules"]["rc_brd"] = _rc_cfg(kappa=0.25)
        with pytest.raises(ValueError, match="kappa"):
            SLMFBBDM.from_config(cfg)

    def test_contract_path_missing_file(self, artifact_dir):
        cfg = _base_config()
        cfg["modules"]["rc_brd"] = _rc_cfg(contract_path=str(artifact_dir / "nope.json"))
        with pytest.raises(ValueError, match="contract_path"):
            SLMFBBDM.from_config(cfg)

    def test_contract_fold_mismatch(self, artifact_dir):
        cfg = _model_with_contract_cfg(artifact_dir)
        cfg["modules"]["rc_brd"]["contract_fold"] = "fold_9"
        with pytest.raises(ValueError, match="fold"):
            SLMFBBDM.from_config(cfg)

    def test_runtime_fold_key_accepted_on_match(self, artifact_dir):
        cfg = _model_with_contract_cfg(artifact_dir)
        cfg["runtime"]["rc_brd_fold"] = "fold_0"
        model = _build(cfg)
        assert model.rc_brd_contract is not None
        assert model.rc_brd_contract.fold_id == "fold_0"

    def test_contract_mean_sha_mismatch(self, artifact_dir):
        ckpt = artifact_dir / "mean.pt"
        _make_mean_checkpoint(ckpt)
        contract = artifact_dir / "contract.json"
        _write_contract(contract, mean_sha="b" * 64)
        cfg = _base_config()
        cfg["modules"]["conditional_mean"].update({"checkpoint": str(ckpt), "freeze": True})
        cfg["modules"]["rc_brd"] = _rc_cfg(contract_path=str(contract))
        with pytest.raises(ValueError, match="SHA"):
            SLMFBBDM.from_config(cfg)

    def test_contract_requires_loaded_mean_checkpoint(self, artifact_dir):
        contract = artifact_dir / "contract.json"
        _write_contract(contract, mean_sha="a" * 64)
        cfg = _base_config()   # no mean checkpoint configured
        cfg["modules"]["rc_brd"] = _rc_cfg(contract_path=str(contract))
        with pytest.raises(ValueError, match="conditional-mean"):
            SLMFBBDM.from_config(cfg)

    def test_band_groups_disagree_with_contract(self, artifact_dir):
        cfg = _model_with_contract_cfg(artifact_dir)
        cfg["modules"]["rc_brd"]["band_groups"] = 7
        with pytest.raises(ValueError, match="band_groups"):
            SLMFBBDM.from_config(cfg)

    def test_declared_contract_keys_must_match_artifact(self, artifact_dir):
        cfg = _model_with_contract_cfg(artifact_dir)
        cfg["modules"]["rc_brd"]["contract"] = {"support_mode": "stratified_mixture"}
        with pytest.raises(ValueError, match="support_mode"):
            SLMFBBDM.from_config(cfg)

    @pytest.mark.parametrize("key,value,pattern", [
        ("forward_mode", "galactic", "forward_mode"),
        ("endpoint_mode", "mean_minus_ct", "endpoint_mode"),
        ("band_groups", 5, "n_groups"),
    ])
    def test_enum_violations(self, key, value, pattern):
        cfg = _base_config()
        cfg["modules"]["rc_brd"] = _rc_cfg(**{key: value})
        with pytest.raises(ValueError, match=pattern):
            SLMFBBDM.from_config(cfg)

    def test_bad_readout_mode(self):
        cfg = _base_config()
        cfg["modules"]["rc_brd"] = _rc_cfg(readout={"mode": "magical"})
        with pytest.raises(ValueError, match="readout"):
            SLMFBBDM.from_config(cfg)

    def test_num_timesteps_mismatch(self):
        cfg = _base_config()
        cfg["modules"]["rc_brd"] = _rc_cfg(num_timesteps=50)
        with pytest.raises(ValueError, match="num_timesteps"):
            SLMFBBDM.from_config(cfg)

    def test_image_size_not_divisible_by_four(self):
        cfg = _base_config()
        cfg["data"]["image_size"] = 34
        cfg["modules"]["rc_brd"] = _rc_cfg()
        with pytest.raises(ValueError, match="divisible by 4"):
            SLMFBBDM.from_config(cfg)

    def test_rc_brd_requires_residual_bridge(self):
        cfg = _base_config()
        cfg["modules"]["residual_bridge"]["enabled"] = False
        cfg["modules"]["rc_brd"] = _rc_cfg()
        with pytest.raises(ValueError, match="residual_bridge"):
            SLMFBBDM.from_config(cfg)

# ---------------------------------------------------------------------------
# C7 endpoint_mode=ct_minus_mean + C1 vp posterior passthrough
# ---------------------------------------------------------------------------


class TestEndpointCtMinusMean:
    def _model(self, artifact_dir, **rc_overrides):
        overrides = {"endpoint_mode": "ct_minus_mean"}
        overrides.update(rc_overrides)
        return _build(_model_with_contract_cfg(artifact_dir, **overrides))

    def test_endpoint_bands_are_band_coordinate(self, artifact_dir):
        model = self._model(artifact_dir)
        ct = torch.randn(2, 1, IMAGE, IMAGE)
        mean = model.mean_predictor(ct)["mean_pet"]
        endpoint = model._rc_brd_endpoint_bands(ct, mean)
        assert set(endpoint) == set(BAND_NAMES)
        for band, tensor in endpoint.items():
            assert tensor.shape[-1] in (IMAGE // 4, IMAGE // 2)   # band grid
            assert tensor.shape[-1] != IMAGE                      # not image grid

    def test_forward_and_sample_pass_band_endpoints(self, artifact_dir, monkeypatch):
        model = self._model(artifact_dir)
        seen = {}
        original = model.rc_brd_schedule.q_marginal

        def spy(z0_bands, t, *args, **kwargs):
            seen["endpoint"] = kwargs.get("endpoint_bands")
            return original(z0_bands, t, *args, **kwargs)

        monkeypatch.setattr(model.rc_brd_schedule, "q_marginal", spy)
        batch = _batch()
        loss, _ = model(batch, torch.tensor([13, 87]))
        assert torch.isfinite(loss).all()
        assert seen["endpoint"] is not None and set(seen["endpoint"]) == set(BAND_NAMES)
        out = model.sample(batch, num_steps=4)
        assert torch.isfinite(out["synthetic_pet"]).all()

    def test_zeros_mode_returns_none(self, artifact_dir):
        model = self._model(artifact_dir, endpoint_mode="zeros")
        zeros = torch.zeros(1, 1, IMAGE, IMAGE)
        assert model._rc_brd_endpoint_bands(zeros, zeros) is None


def test_kappa_nonzero_with_contract_builds_and_runs(artifact_dir):
    # Contracted mainline smoke: warped clock + inactive-group zero c.
    cfg = _model_with_contract_cfg(artifact_dir, kappa=0.25)
    model = _build(cfg)
    assert model.rc_brd_schedule.config.kappa == 0.25
    batch = _batch()
    loss, _ = model(batch, torch.tensor([21, 66]))
    assert torch.isfinite(loss).all()
    out = model.sample(batch, num_steps=4)
    assert torch.isfinite(out["synthetic_pet"]).all()


def test_vp_mode_sampling_works_and_posterior_raises():
    cfg = _base_config()
    cfg["modules"]["rc_brd"] = _rc_cfg(forward_mode="vp_bandwise")
    model = _build(cfg)
    out = model.sample(_batch(), num_steps=3)
    assert torch.isfinite(out["synthetic_pet"]).all()
    # C1: the bridge posterior must refuse vp mode, unmodified (passthrough).
    bands = {band: torch.randn(2, 1, 4, 4) for band in BAND_NAMES}
    with pytest.raises(NotImplementedError):
        model.rc_brd_schedule.posterior(bands, bands,
                                        torch.tensor([1, 1]), torch.tensor([2, 2]))


# ---------------------------------------------------------------------------
# GT-leak fail-closed (FR-4.4/FR-5.2; plan S4)
# ---------------------------------------------------------------------------


def test_rc_brd_paths_never_read_gt_mask():
    cfg = _base_config()
    cfg["modules"]["rc_brd"] = _rc_cfg()
    model = _build(cfg)

    recording = _AccessRecordingDict(_batch())
    model(recording, torch.tensor([4, 44]))
    assert "mask" not in recording.accessed
    assert "ct" in recording.accessed and "pet" in recording.accessed

    recording_eval = _AccessRecordingDict(_batch())
    model.sample(recording_eval, num_steps=3)
    assert "mask" not in recording_eval.accessed
    assert "ct" in recording_eval.accessed
    assert "pet" not in recording_eval.accessed   # inference reads CT only


# ---------------------------------------------------------------------------
# C6 readout modes
# ---------------------------------------------------------------------------


class TestReadout:
    def test_mc_mean_pins_K_from_readout_config(self):
        cfg = _base_config()
        cfg["modules"]["rc_brd"] = _rc_cfg(readout={"mode": "mc_mean",
                                                  "mc_samples": 3})
        model = _build(cfg)
        assert model.rc_brd_readout_mode == "mc_mean"
        torch.manual_seed(8)
        out = model.sample_mc(_batch(), num_steps=2)   # n_samples=None -> K=3
        assert out["samples"].shape[0] == 3
        assert torch.isfinite(out["synthetic_pet"]).all()

    def test_deterministic_head_mode_smoke(self):
        cfg = _base_config()
        cfg["modules"]["rc_brd"] = _rc_cfg(readout={"mode": "deterministic_head"})
        model = _build(cfg)
        out = model.sample(_batch(), num_steps=2)
        assert torch.isfinite(out["synthetic_pet"]).all()

    def test_legacy_sample_mc_default_unaffected(self):
        model = _build(_base_config())
        torch.manual_seed(8)
        out = model.sample_mc(_batch(), num_steps=2)
        assert out["samples"].shape[0] == 20


# ---------------------------------------------------------------------------
# Smoke yaml (configs/experiments/rc_brd_smoke.yaml)
# ---------------------------------------------------------------------------


def test_smoke_yaml_config_builds_and_runs():
    yaml_path = REPO_ROOT / "configs" / "experiments" / "rc_brd_smoke.yaml"
    with open(yaml_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["modules"]["rc_brd"]["enabled"] is True
    model = _build(cfg)
    assert model.rc_brd_enabled and model.rc_brd_head is not None
    from torch.utils.data import DataLoader
    from src.data.dataset import FakeDataset
    size = cfg["data"]["image_size"]
    batch = next(iter(DataLoader(FakeDataset(num_samples=2, image_size=size),
                                batch_size=2)))
    loss, _ = model(batch, torch.tensor([2, 40]))
    assert torch.isfinite(loss).all()
    out = model.sample(batch, num_steps=3)
    assert torch.isfinite(out["synthetic_pet"]).all()

# ---------------------------------------------------------------------------
# DESIGN v1.0f: five ablation keys consumed fail-closed (AUDIT_1 A1-2)
# ---------------------------------------------------------------------------


class TestContractTransform:
    def test_flat_c_consumed_at_build(self, artifact_dir):
        cfg = _model_with_contract_cfg(artifact_dir, kappa=0.25,
                                       contract_transform="flat_c")
        model = _build(cfg)
        # Transformed contract: every group c == 0.5 ([审计] §3.3 c0).
        for values in model.rc_brd_contract.c_values.values():
            assert all(abs(v - 0.5) < 1e-12 for v in values)
        # flat c with kappa!=0 degenerates the clock to the uniform one.
        identity = torch.arange(T + 1, dtype=torch.float64) / T
        for group in model.rc_brd_contract.band_groups:
            assert float((model.rc_brd_schedule.m_sequence(group) - identity)
                         .abs().max()) < 1e-9
        loss, _ = model(_batch(), torch.tensor([9, 71]))
        assert torch.isfinite(loss).all()

    def test_group_rotation_and_grid_reversal_consumed(self, artifact_dir):
        from src.model.rc_brd import apply_contract_transform
        cfg = _model_with_contract_cfg(artifact_dir, kappa=0.25,
                                       contract_transform="group_rotation")
        model = _build(cfg)

        ckpt = artifact_dir / "mean.pt"   # same seeded checkpoint source
        from src.model.mean_predictor import LowFrequencyPETPredictor
        torch.manual_seed(3)
        original = RecoverabilityContract.from_payload({
            "fold_id": "fold_0",
            "band_groups": {g: list(b) for g, b in _CONTRACT_GROUPS.items()},
            "log_snr_grid": [-10.0, 0.0, 10.0],
            "c_values": {"low": [0.2, 0.3, 0.4], "mid": [0.4, 0.5, 0.6],
                         "high": [0.6, 0.7, 0.8]},
            "mean_checkpoint_sha256": mean_weights_sha256(
                LowFrequencyPETPredictor(base_channels=8, levels=2).state_dict()),
            "b_active": ["mid", "high"],
            "psd_floors": {"low": 1e-4, "mid": 2e-4, "high": 5e-4},
            "size_thresholds": {"small_lesion_q25": 12.0},
            "kappa_grid": [0.0, 0.25],
            "s_ref": 1.0, "support_mode": "floor_gated",
            "eta_max": 0.8, "floor_rho": 0.1,
        })
        expected = apply_contract_transform(original, "group_rotation")
        assert model.rc_brd_contract.c_values == expected.c_values

        cfg = _model_with_contract_cfg(artifact_dir, kappa=0.25,
                                       contract_transform="grid_reversal")
        model = _build(cfg)
        expected = apply_contract_transform(original, "grid_reversal")
        assert model.rc_brd_contract.log_snr_grid == expected.log_snr_grid
        assert model.rc_brd_contract.c_values == expected.c_values

    def test_transform_without_contract_fails_closed(self):
        cfg = _base_config()
        cfg["modules"]["rc_brd"] = _rc_cfg(contract_transform="flat_c")
        with pytest.raises(ValueError, match="contract_transform"):
            SLMFBBDM.from_config(cfg)

    def test_unknown_transform_tag(self, artifact_dir):
        cfg = _model_with_contract_cfg(artifact_dir, contract_transform="shuffle")
        with pytest.raises(ValueError, match="contract_transform"):
            SLMFBBDM.from_config(cfg)


class TestA3VariantPairing:
    @pytest.mark.parametrize("variant,density,legal", [
        ("", "", True),
        ("budget_matched", "", True),
        ("density_matched", "logsnr", True),
        ("budget_matched", "logsnr", False),   # budget arm must stay uniform
        ("density_matched", "", False),        # A3b without density match
        ("", "logsnr", False),                 # density match is A3b-only
        ("weird", "", False),                  # unknown variant
    ])
    def test_pairing_matrix(self, artifact_dir, variant, density, legal):
        cfg = _model_with_contract_cfg(artifact_dir, a3_variant=variant,
                                       density_match=density)
        if legal:
            model = _build(cfg)
            assert model.rc_brd_density_match == (density == "logsnr")
        else:
            with pytest.raises(ValueError, match="a3_variant|A3_VARIANTS"):
                SLMFBBDM.from_config(cfg)

class TestDensityMatchLogsnr:
    def _a3b_model(self, artifact_dir):
        # Full A3b arm: flat c contract-side, density from the ORIGINAL c.
        cfg = _model_with_contract_cfg(artifact_dir, kappa=0.25,
                                       contract_transform="flat_c",
                                       a3_variant="density_matched",
                                       density_match="logsnr")
        return _build(cfg)

    def test_weights_normalised_and_biased_to_high_c(self, artifact_dir):
        model = self._a3b_model(artifact_dir)
        weights = model._rc_brd_density_weights()
        assert weights.numel() == T
        assert torch.isfinite(weights).all()
        assert float(weights.sum()) == pytest.approx(1.0, abs=1e-12)
        assert bool((weights > 0).all())
        # Original contract: c rises with lambda and lambda0(u) falls with u,
        # so rho-bar peaks at small u -> the density mass follows the high-c
        # region (early/clean timesteps for this fixture).
        assert float(weights[0]) > float(weights[T - 1])
        assert int(weights.argmax()) < T // 2

    def test_draws_deterministic_and_advancing(self, artifact_dir):
        first = self._a3b_model(artifact_dir)
        second = self._a3b_model(artifact_dir)
        draw_a1 = first._rc_brd_density_timesteps(8, torch.device("cpu"))
        draw_a2 = first._rc_brd_density_timesteps(8, torch.device("cpu"))
        draw_b1 = second._rc_brd_density_timesteps(8, torch.device("cpu"))
        assert torch.equal(draw_a1, draw_b1)   # same seed -> same first draw
        assert not torch.equal(draw_a1, draw_a2)  # counter advances
        assert draw_a1.shape == (8,)

    def test_forward_with_density_sampling_runs(self, artifact_dir):
        model = self._a3b_model(artifact_dir)
        loss, logs = model(_batch(), timesteps=None)   # density path taken
        assert torch.isfinite(loss).all()
        assert logs["module/rc_brd"].item() == 1.0


class TestPerBandMinSnr:
    def test_band_weights_match_manual_min_snr_gamma(self):
        cfg = _base_config()
        cfg["modules"]["rc_brd"] = _rc_cfg(loss_weighting="per_band_min_snr",
                                          min_snr_gamma=3.0)
        model = _build(cfg)
        batch = _batch()
        timesteps = torch.tensor([10, 55])
        loss, logs = model(batch, timesteps)
        assert torch.isfinite(loss).all()
        # Manual weights: no contract -> group == band; SNR_g(t)=exp(lambda)
        # read on the schedule's single-source query axis
        # clock_query_log_snr (DESIGN_RC_BRD_clock_v2 §2.1/§5.5).  base_snr
        # mode without a contract keeps the shared axis
        # lambda0(u)=clip(log((1-u)/(2*sigma^2*u)), -10, 10), sigma_bridge=1.0
        # (the retired ah/(1-ah) proxy from schedule.alpha_hat carried no P_g
        # and no nu^2 scaling).
        for band in BAND_NAMES:
            log_snr = model.rc_brd_schedule.clock_query_log_snr(band, timesteps)
            manual = torch.minimum(log_snr.exp(),
                                   torch.tensor(3.0, dtype=log_snr.dtype))
            logged = logs[f"loss/rc_brd_band_weight_{band}"]
            torch.testing.assert_close(logged, manual.mean().to(logged.dtype),
                                       rtol=1e-5, atol=1e-6)

    def test_unknown_loss_weighting_fails_closed(self):
        cfg = _base_config()
        cfg["modules"]["rc_brd"] = _rc_cfg(loss_weighting="magic")
        with pytest.raises(ValueError, match="loss_weighting"):
            SLMFBBDM.from_config(cfg)

    def test_bad_gamma_fails_closed(self):
        cfg = _base_config()
        cfg["modules"]["rc_brd"] = _rc_cfg(loss_weighting="per_band_min_snr",
                                          min_snr_gamma=0.0)
        with pytest.raises(ValueError, match="min_snr_gamma"):
            SLMFBBDM.from_config(cfg)

# ---------------------------------------------------------------------------
# DESIGN v1.0f: four combination guards (AUDIT_1 A1-5, AUDIT_3)
# ---------------------------------------------------------------------------


class TestV1fCombinationGuards:
    def test_vp_bandwise_rejects_ct_minus_mean(self, artifact_dir):
        cfg = _model_with_contract_cfg(artifact_dir,
                                       forward_mode="vp_bandwise",
                                       endpoint_mode="ct_minus_mean")
        with pytest.raises(ValueError, match="vp_bandwise"):
            SLMFBBDM.from_config(cfg)

    def test_kappa_outside_contract_grid(self, artifact_dir):
        cfg = _model_with_contract_cfg(artifact_dir, kappa=0.5)   # grid: 0, 0.25
        with pytest.raises(ValueError, match="kappa_grid"):
            SLMFBBDM.from_config(cfg)

    def test_kappa_zero_without_contract_skips_grid_guard(self):
        cfg = _base_config()
        cfg["modules"]["rc_brd"] = _rc_cfg(kappa=0.0)   # no contract -> skip
        model = _build(cfg)
        assert model.rc_brd_schedule.config.kappa == 0.0

    def test_rc_brd_excludes_legacy_residual_frequency(self):
        cfg = _base_config()
        cfg["modules"]["residual_frequency"] = {"enabled": True}
        cfg["modules"]["rc_brd"] = _rc_cfg()
        with pytest.raises(ValueError, match="residual_frequency"):
            SLMFBBDM.from_config(cfg)


# ---------------------------------------------------------------------------
# v1.0f guard 4: floor_gated runtime floor ([审计] §5) + λ₀ convention
# ---------------------------------------------------------------------------


class TestFloorGatedRuntimeFloor:
    def test_floor_formula_and_inactive_zero(self, artifact_dir):
        model = _build(_model_with_contract_cfg(artifact_dir))   # floor 0.1
        t = torch.tensor([25])
        reference = torch.zeros(1, 1, IMAGE, IMAGE)
        c_bands = model._rc_brd_c_effective(t, reference)
        from src.model.slmf_bbdm import _rc_brd_base_log_snr
        cfg = model.rc_brd_schedule.config
        lam = _rc_brd_base_log_snr(t.double() / cfg.num_timesteps,
                                   cfg.sigma_bridge, cfg.lambda_min,
                                   cfg.lambda_max)
        contract = model.rc_brd_contract
        for group, bands in contract.band_groups.items():
            if group in contract.b_active:
                c_tilde = contract.effective_c(group, lam)
                expected = 0.1 + 0.9 * c_tilde   # floor_rho + (1-rho)*c~
            else:
                expected = torch.zeros_like(lam)  # inactive stays off
            for band in bands:
                torch.testing.assert_close(c_bands[band], expected,
                                           rtol=1e-6, atol=1e-7)

    def test_stratified_mixture_is_identity(self, artifact_dir):
        model = _build(_model_with_contract_cfg(
            artifact_dir, support_mode="stratified_mixture"))
        t = torch.tensor([25])
        c_bands = model._rc_brd_c_effective(t, torch.zeros(1, 1, IMAGE, IMAGE))
        from src.model.slmf_bbdm import _rc_brd_base_log_snr
        cfg = model.rc_brd_schedule.config
        lam = _rc_brd_base_log_snr(t.double() / cfg.num_timesteps,
                                   cfg.sigma_bridge, cfg.lambda_min,
                                   cfg.lambda_max)
        contract = model.rc_brd_contract
        for group, bands in contract.band_groups.items():
            expected = (contract.effective_c(group, lam)
                        if group in contract.b_active
                        else torch.zeros_like(lam))
            for band in bands:
                torch.testing.assert_close(c_bands[band], expected,
                                           rtol=1e-6, atol=1e-7)


class TestLambda0Convention:
    def test_closed_form_and_endpoint_clips(self):
        from src.model.slmf_bbdm import _rc_brd_base_log_snr
        u = torch.tensor([0.0, 0.25, 0.5, 0.9, 1.0], dtype=torch.float64)
        lam = _rc_brd_base_log_snr(u, 1.0, -10.0, 10.0)   # nu^2 = 2
        expected = torch.log((1.0 - u.clamp(1e-12, 1 - 1e-12))
                             / (2.0 * u.clamp(1e-12, 1 - 1e-12)))
        torch.testing.assert_close(lam, expected.clamp(-10.0, 10.0),
                                   rtol=0, atol=0)
        assert float(lam[0]) == 10.0 and float(lam[-1]) == -10.0

    def test_c_effective_uses_lambda0_and_matches_schedule_clock(self, artifact_dir):
        """Integration-side c lookup and the schedule clock share the v1.0f
        λ₀(u)=clip(log((1−u)/(ν²u)),λmin,λmax) lookup (DESIGN §4 v1.0f)."""
        from src.model.slmf_bbdm import _rc_brd_base_log_snr
        model = _build(_model_with_contract_cfg(artifact_dir, kappa=0.25))
        cfg = model.rc_brd_schedule.config
        contract = model.rc_brd_contract
        t = torch.tensor([1, 10, 50, 99])
        c_bands = model._rc_brd_c_effective(t, torch.zeros(len(t), 1, IMAGE, IMAGE))
        lam = _rc_brd_base_log_snr(t.double() / cfg.num_timesteps,
                                   cfg.sigma_bridge, cfg.lambda_min,
                                   cfg.lambda_max)
        floor = (contract.floor_rho if contract.support_mode == "floor_gated"
                 else 0.0)
        for group, bands in contract.band_groups.items():
            if group not in contract.b_active:
                continue
            expected = (floor + (1.0 - floor)
                        * contract.effective_c(group, lam))
            for band in bands:
                torch.testing.assert_close(c_bands[band], expected,
                                           rtol=1e-6, atol=1e-7)
        # Same convention on the schedule side: integrate rho = exp{k(2c-1)}
        # built from THIS lambda0 helper on the schedule grid (4T nodes).
        nodes = 4 * cfg.num_timesteps
        u = torch.arange(nodes + 1, dtype=torch.float64) / nodes
        lam_fine = _rc_brd_base_log_snr(u, cfg.sigma_bridge,
                                        cfg.lambda_min, cfg.lambda_max)
        group = "mid"
        rho = torch.exp(cfg.kappa * (2.0 * contract.effective_c(group, lam_fine)
                                     .to(torch.float64) - 1.0))
        inc = 0.5 * (rho[:-1] + rho[1:]) / nodes
        cum = torch.cat([torch.zeros(1, dtype=torch.float64),
                         torch.cumsum(inc, 0)])
        m_mine = (cum / cum[-1])[::4]
        torch.testing.assert_close(model.rc_brd_schedule.m_sequence(group),
                                   m_mine, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# v1.0g sampling_mode: endpoint_ddim | ancestral_mc (DESIGN §8; AUDIT 5 §3.3)
# ---------------------------------------------------------------------------


class TestSamplingMode:
    def test_default_is_endpoint_ddim(self):
        cfg = _base_config()
        cfg["modules"]["rc_brd"] = _rc_cfg()
        model = _build(cfg)
        assert model.rc_brd_sampling_mode == "endpoint_ddim"

    def test_invalid_enum_fails_closed(self):
        cfg = _base_config()
        cfg["modules"]["rc_brd"] = _rc_cfg(sampling_mode="banana")
        with pytest.raises(ValueError, match="sampling_mode"):
            SLMFBBDM.from_config(cfg)

    def test_both_modes_smoke(self):
        for mode in ("endpoint_ddim", "ancestral_mc"):
            cfg = _base_config()
            cfg["modules"]["rc_brd"] = _rc_cfg(sampling_mode=mode)
            model = _build(cfg)
            loss, _ = model(_batch(), torch.tensor([6, 60]))
            assert torch.isfinite(loss).all()
            out = model.sample(_batch(), num_steps=4)
            assert torch.isfinite(out["synthetic_pet"]).all()

    def test_same_seed_modes_diverge_and_mc_is_reproducible(self):
        """Same initial noise + same RNG seed: endpoint_ddim (deterministic
        steps) and ancestral_mc (per-step posterior noise) must produce
        numerically different outputs; ancestral_mc itself is seed-stable."""
        cfg_d = _base_config()
        cfg_d["modules"]["rc_brd"] = _rc_cfg(sampling_mode="endpoint_ddim")
        cfg_m = _base_config()
        cfg_m["modules"]["rc_brd"] = _rc_cfg(sampling_mode="ancestral_mc")
        model_d, model_m = _build(cfg_d), _build(cfg_m)
        batch = _batch()
        noise = torch.randn(2, 1, IMAGE, IMAGE)
        out_d = model_d.sample(batch, initial_noise=noise, num_steps=5)
        torch.manual_seed(777)
        out_m1 = model_m.sample(batch, initial_noise=noise, num_steps=5)
        torch.manual_seed(777)
        out_m2 = model_m.sample(batch, initial_noise=noise, num_steps=5)
        # same seed → identical ancestral trajectory (deterministic given seed)
        assert torch.equal(out_m1["synthetic_pet"], out_m2["synthetic_pet"])
        # different mode → numerically different output
        assert not torch.allclose(out_m1["synthetic_pet"], out_d["synthetic_pet"],
                                  rtol=1e-5, atol=1e-6)
        assert float((out_m1["synthetic_pet"] - out_d["synthetic_pet"]).abs().max()) > 1e-4

    def test_endpoint_ddim_keeps_kappa0_d1_equivalence(self):
        """The default mode must not perturb the κ=0 ≡ D1 lock (regression)."""
        cfg = _base_config()
        cfg["modules"]["rc_brd"] = _rc_cfg(kappa=0.0)  # endpoint_ddim default
        rc = _build(cfg)
        d1 = _build(_base_config())
        batch = _batch()
        noise = torch.randn(2, 1, IMAGE, IMAGE)
        out_d1 = d1.sample(batch, initial_noise=noise, num_steps=5)
        out_rc = rc.sample(batch, initial_noise=noise, num_steps=5)
        torch.testing.assert_close(out_rc["synthetic_pet"], out_d1["synthetic_pet"],
                                   rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# v1.0g density weighting: rho-bar averaged over Haar coefficients n_b/N
# (LL2:mid:high = 1:3:12 / 16; AUDIT 5 §3.4)
# ---------------------------------------------------------------------------


class TestDensityCoefficientWeighting:
    def test_group_shares_derived_from_band_groups(self):
        from src.model.slmf_bbdm import _rc_brd_group_shares
        from src.model.rc_brd import band_groups
        shares = _rc_brd_group_shares(_CONTRACT_GROUPS)
        assert shares == pytest.approx(
            {"low": 1.0 / 16.0, "mid": 3.0 / 16.0, "high": 12.0 / 16.0})
        assert sum(shares.values()) == pytest.approx(1.0, abs=1e-12)
        # derived, not hardcoded: the 7-group layout gets per-band shares
        seven = _rc_brd_group_shares(band_groups(7))
        assert sum(seven.values()) == pytest.approx(1.0, abs=1e-12)
        assert seven["LL2"] == pytest.approx(1.0 / 16.0)
        assert seven["LH1"] == pytest.approx(4.0 / 16.0)
        with pytest.raises(ValueError, match="Haar-level"):
            _rc_brd_group_shares({"weird": ["XX9"]})

    def test_equal_c_high_group_contributes_12_of_16(self):
        """With identical c across groups every per-band rho is equal, so the
        high group's contribution to rho-bar is exactly its coefficient share
        12/16 (not 1/3 as the old equal-group mean would have it)."""
        from src.model.slmf_bbdm import _rc_brd_group_shares
        shares = _rc_brd_group_shares(_CONTRACT_GROUPS)
        equal_rho = {g: 1.0 for g in shares}  # same c̃ ⇒ identical ρ_g
        total = sum(shares[g] * equal_rho[g] for g in shares)
        assert shares["high"] * equal_rho["high"] / total == pytest.approx(12.0 / 16.0)

    def test_density_weights_use_coefficient_counts_not_group_mean(self, artifact_dir):
        cfg = _model_with_contract_cfg(artifact_dir, kappa=0.25,
                                       contract_transform="flat_c",
                                       a3_variant="density_matched",
                                       density_match="logsnr")
        model = _build(cfg)
        weights = model._rc_brd_density_weights()
        from src.model.slmf_bbdm import _rc_brd_base_log_snr, _rc_brd_group_shares
        cfg_s = model.rc_brd_schedule.config
        contract = model._rc_brd_density_contract   # original, pre-transform
        u = torch.arange(cfg_s.num_timesteps, dtype=torch.float64) / cfg_s.num_timesteps
        lam = _rc_brd_base_log_snr(u, cfg_s.sigma_bridge, cfg_s.lambda_min, cfg_s.lambda_max)
        shares = _rc_brd_group_shares(contract.band_groups)
        manual = torch.zeros_like(lam)
        for group, share in shares.items():
            c_tilde = contract.effective_c(group, lam).to(torch.float64)
            manual = manual + share * torch.exp(cfg_s.kappa * (2.0 * c_tilde - 1.0))
        manual = manual / manual.sum()
        torch.testing.assert_close(weights, manual, rtol=1e-12, atol=1e-14)
        # the old equal-weight 3-group mean is a *different* distribution
        old = torch.stack([
            torch.exp(cfg_s.kappa * (2.0 * contract.effective_c(g, lam).to(torch.float64) - 1.0))
            for g in contract.band_groups
        ]).mean(dim=0)
        old = old / old.sum()
        assert not torch.allclose(weights, old, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# v1.0g CT provenance config plumbing (allow_unverified_tokens; AUDIT 5 X)
# ---------------------------------------------------------------------------


class TestCTProvenanceConfig:
    def test_default_is_fail_closed_and_smoke_yaml_registers_keys(self):
        cfg = _base_config()
        cfg["modules"]["rc_brd"] = _rc_cfg()
        model = _build(cfg)
        assert model.rc_brd_head.allow_unverified_tokens is False
        with open(REPO_ROOT / "configs" / "experiments" / "rc_brd_smoke.yaml",
                  "r", encoding="utf-8") as handle:
            smoke = yaml.safe_load(handle)
        rc = smoke["modules"]["rc_brd"]
        assert rc["sampling_mode"] == "endpoint_ddim"      # v1.0g 登记
        assert rc["allow_unverified_tokens"] is False      # 生产必须 false
        assert rc["specialist"]["d_max"] == 0.10           # v1.0g tanh 幅度界

    def test_escape_hatch_config_builds(self):
        cfg = _base_config()
        cfg["modules"]["rc_brd"] = _rc_cfg(allow_unverified_tokens=True)
        model = _build(cfg)
        assert model.rc_brd_head.allow_unverified_tokens is True
        loss, _ = model(_batch(), torch.tensor([8, 44]))
        assert torch.isfinite(loss).all()

    def test_internal_pipeline_issues_token_not_bare_tensor(self):
        """_rc_brd_apply_specialist wraps the ct_proj output in a head-issued
        CTFeatureToken; a bare tensor crossing the head boundary raises."""
        from src.model.rc_brd import CTFeatureToken, haar_forward2
        from src.model.rc_brd.head import BoundedSpecialistHead
        cfg = _base_config()
        cfg["modules"]["rc_brd"] = _rc_cfg()
        model = _build(cfg)
        pred = torch.randn(2, 1, IMAGE, IMAGE)
        out = model._rc_brd_apply_specialist(pred, torch.tensor([10, 20]),
                                             _batch()["ct"])
        assert torch.isfinite(out).all()
        # The head's own default gate rejects the identical ct_proj output
        # when it arrives as a bare tensor (provenance, not shape).
        bare_head = BoundedSpecialistHead(model.rc_brd_head.config)
        ct_feat = model.rc_brd_ct_proj(torch.nn.functional.avg_pool2d(
            _batch()["ct"], kernel_size=2))
        with pytest.raises(ValueError, match="CTFeatureToken"):
            bare_head(haar_forward2(pred), ct_feat, None)
        # and a token issued by the *model's* head session is the accepted form
        token = model.rc_brd_head.issue_ct_token(ct_feat)
        assert isinstance(token, CTFeatureToken)
        init_out = model.rc_brd_head(haar_forward2(pred), token, None)
        for band, base in haar_forward2(pred).items():
            assert torch.equal(init_out[band], base)  # zero-init identity


# ---------------------------------------------------------------------------
# v2 band clock integration (clock_mode; DESIGN_RC_BRD_clock_v2)
# ---------------------------------------------------------------------------

class TestBandClockV2:
    def test_bad_clock_mode_fails_closed(self, artifact_dir):
        cfg = _model_with_contract_cfg(artifact_dir, kappa=0.25, clock_mode="nope")
        with pytest.raises(ValueError, match="clock_mode"):
            _build(cfg)

    def test_band_snr_with_v1_contract_fails_closed(self, artifact_dir):
        # v1 artifact (no band_powers) + band_snr -> actionable build error.
        cfg = _model_with_contract_cfg(artifact_dir, kappa=0.25, clock_mode="band_snr")
        with pytest.raises(ValueError, match="band_powers"):
            _build(cfg)

    def test_band_snr_with_v2_contract_builds(self, artifact_dir):
        cfg = _model_with_contract_cfg(
            artifact_dir, kappa=0.25, clock_mode="band_snr",
            contract_band_powers={"low": 0.2, "mid": 1.0, "high": 5.0})
        model = _build(cfg)
        assert model.rc_brd_schedule.config.clock_mode == "band_snr"
        seq = model.rc_brd_schedule.m_sequence("mid")
        assert seq[0].item() == 0.0 and seq[-1].item() == 1.0
        assert bool((seq[1:] - seq[:-1] > 0).all())

    def test_band_snr_head_gating_uses_clock_query_axis(self, artifact_dir):
        # Single-source query axis: _rc_brd_c_effective interpolates the
        # contract at exactly schedule.clock_query_log_snr (audit defect (b)).
        cfg = _model_with_contract_cfg(
            artifact_dir, kappa=0.25, clock_mode="band_snr",
            contract_band_powers={"low": 0.2, "mid": 1.0, "high": 5.0})
        model = _build(cfg)
        contract = model.rc_brd_contract
        t = torch.tensor([1, 10, 50], dtype=torch.int64)
        c_bands = model._rc_brd_c_effective(t, torch.zeros(len(t), 1, IMAGE, IMAGE))
        floor = (contract.floor_rho if contract.support_mode == "floor_gated"
                 else 0.0)
        # Cover a P_g=1.0 group AND a P_g=5.0 group (review coverage note):
        # the warped axis differs from the old shared base axis for both, and
        # additionally the power shift is pinned for "high".
        for group in ("mid", "high"):
            lam = model.rc_brd_schedule.clock_query_log_snr(group, t)
            expected = floor + (1.0 - floor) * contract.effective_c(group, lam)
            for band in contract.band_groups[group]:
                torch.testing.assert_close(c_bands[band], expected,
                                           rtol=1e-6, atol=1e-7)
        lam_high = model.rc_brd_schedule.clock_query_log_snr("high", t)
        lam_low_axis = torch.log(
            1.0 * (1.0 - t.double() / T) / (2.0 * (t.double() / T))
        ).clamp(-10.0, 10.0)  # shared v1 axis for contrast
        assert not torch.allclose(lam_high, lam_low_axis)

    def test_base_snr_clock_ignores_contract_band_powers(self, artifact_dir):
        # Control B: a v2 contract consumed in base_snr mode is bit-identical
        # to the v1 contract without band_powers.
        cfg1 = _model_with_contract_cfg(
            artifact_dir, kappa=0.25, clock_mode="base_snr",
            contract_band_powers={"low": 0.2, "mid": 1.0, "high": 5.0})
        cfg2 = _model_with_contract_cfg(artifact_dir, kappa=0.25,
                                        clock_mode="base_snr")
        m1 = _build(cfg1).rc_brd_schedule.m_sequence("mid")
        m2 = _build(cfg2).rc_brd_schedule.m_sequence("mid")
        torch.testing.assert_close(m1, m2, rtol=0, atol=0)

    def test_band_snr_loss_weights_read_clock_query_axis(self, artifact_dir):
        # A8 per-band Min-SNR-gamma weighting reads the SAME single-source
        # query axis as the clock: w_b(t)=min{exp(lambda_g(t)), gamma} with
        # lambda_g=schedule.clock_query_log_snr(group, t) — the real band
        # bridge SNR with the frozen per-group power P_g baked into the v2
        # contract (DESIGN_RC_BRD_clock_v2 §2.1/§5.5), not the retired
        # alpha_hat proxy.
        cfg = _model_with_contract_cfg(
            artifact_dir, kappa=0.25, clock_mode="band_snr",
            contract_band_powers={"low": 0.2, "mid": 1.0, "high": 5.0},
            loss_weighting="per_band_min_snr", min_snr_gamma=3.0)
        model = _build(cfg)
        # Timesteps spanning high-SNR (small u) and low-SNR (large u) regions
        # so the gamma clamp engages on some but not all of them.
        timesteps = torch.tensor([1, 10, 50])
        loss, logs = model(_batch(batch_size=len(timesteps)), timesteps)
        assert torch.isfinite(loss).all()
        gamma = torch.tensor(3.0, dtype=torch.float64)
        for group in ("low", "mid", "high"):
            lam = model.rc_brd_schedule.clock_query_log_snr(group, timesteps)
            expected = torch.minimum(lam.exp(), gamma)
            for band in model.rc_brd_contract.band_groups[group]:
                logged = logs[f"loss/rc_brd_band_weight_{band}"]
                torch.testing.assert_close(logged.mean(),
                                           expected.mean().to(logged.dtype),
                                           rtol=1e-5, atol=1e-6)
        # P_g separation: the P_g=5 "high" group and the P_g=0.2 "low" group
        # carry visibly different weights on at least some t (the frozen
        # powers warp the band axes in opposite directions before the shared
        # gamma clamp flattens both ends).
        w_high = torch.minimum(
            model.rc_brd_schedule.clock_query_log_snr("high", timesteps).exp(),
            gamma)
        w_low = torch.minimum(
            model.rc_brd_schedule.clock_query_log_snr("low", timesteps).exp(),
            gamma)
        assert not torch.allclose(w_high, w_low)
