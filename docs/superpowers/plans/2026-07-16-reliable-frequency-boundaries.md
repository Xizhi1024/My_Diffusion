# Reliable Frequency Injection and Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bounded SNR/edge-aware four-level residual-frequency injection plus lesion/anatomy boundary supervision and metrics, without modifying the noisy residual or gating existing organ/SUV paths.

**Architecture:** A new reliable injector coexists with the V2 legacy preconditioner behind a `mode` switch. It builds exact Haar bands, combines analytic and spatial reliability, optionally gates each directional subband separately, and returns zero-initialized skip additions. A separate boundary loss and evaluator functions operate on reconstructed PET.

**Tech Stack:** Python 3.12, PyTorch, NumPy/SciPy, pytest

---

### Task 1: Reliable band construction and bounded gates

**Files:**
- Create: `src/model/frequency/reliable_injector.py`
- Create: `tests/test_reliable_frequency.py`

- [ ] **Step 1: Write failing reliable-injector contracts**

```python
import torch


def _inputs(batch_size=2, size=32):
    from src.model.noise.base import BBDMBridgeSchedule

    return (
        torch.randn(batch_size, 1, size, size),
        torch.randn(batch_size, 1, size, size),
        torch.tensor([999, 700][:batch_size], dtype=torch.long),
        BBDMBridgeSchedule(num_train_timesteps=1000),
        torch.rand(batch_size, 8, size, size),
    )


def test_reliable_injector_is_zero_init_bounded_and_decoder_ordered():
    residual, ct, timesteps, schedule, orientation = _inputs()
    module = ReliableResidualFrequencyInjector(
        output_channels=(256, 256, 128, 64),
        band_scales=(1.0, 0.5, 0.25, 0.25),
        use_directional_gate=True,
        gabor_orientations=8,
    )
    injections, diagnostics = module(residual, timesteps, schedule, ct, orientation)
    assert [x.shape for x in injections] == [
        (2, 256, 4, 4), (2, 256, 8, 8),
        (2, 128, 16, 16), (2, 64, 32, 32),
    ]
    assert all(torch.count_nonzero(x) == 0 for x in injections)
    assert torch.isfinite(diagnostics["reliability"]).all()
    assert diagnostics["reliability"].min() >= 0
    assert diagnostics["reliability"].max() <= 1


def test_reliability_increases_with_bridge_snr_for_same_content():
    residual, ct, _, schedule, _ = _inputs()
    module = ReliableResidualFrequencyInjector(use_directional_gate=False)
    _, low_snr = module(residual, torch.tensor([999, 999]), schedule, ct)
    _, high_snr = module(residual, torch.tensor([1, 1]), schedule, ct)
    assert high_snr["analytic_reliability"].mean() > low_snr["analytic_reliability"].mean()


def test_full_resolution_high_band_uses_separate_inverse_haar_directions():
    module = ReliableResidualFrequencyInjector(use_directional_gate=False)
    bands = module.build_bands(torch.randn(1, 1, 32, 32))
    assert bands[-1].shape == (1, 3, 32, 32)
    assert not torch.allclose(bands[-1][:, 0], bands[-1][:, 1])
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `python -m pytest tests/test_reliable_frequency.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement exact bands and analytic reliability**

Implement:

```python
def build_bands(self, residual):
    ll1, d1 = haar_dwt2(residual)
    ll2, d2 = haar_dwt2(ll1)
    ll3, _ = haar_dwt2(ll2)
    zeros = torch.zeros_like(ll1)
    full = torch.cat([
        haar_idwt2(zeros, tuple(d if j == i else torch.zeros_like(d) for j, d in enumerate(d1)))
        for i in range(3)
    ], dim=1)
    return [ll3, torch.cat(d2, 1), torch.cat(d1, 1), full]
```

Use `tanh(band / positive_scale)` before projections. Derive four log-SNR values from schedule progress and sigma, then calculate:

```python
analytic = floor + (1.0 - floor) * torch.sigmoid(
    (log_snr - snr_center) / snr_temperature
)
```

Validate four positive scales, positive temperature, and `0 <= floor < 1`.

- [ ] **Step 4: Add spatial and directional gates**

Compute finite-difference CT gradient magnitude, normalize it per sample, and resize with area/bilinear interpolation only as a conditioning map. For each level concatenate band magnitude, CT edge, optional orientation concentration, timestep, and log-SNR into a small convolutional sigmoid gate.

Map Gabor orientations to three detail factors using a zero-initialized 1x1 convolution:

```python
factor = 1.0 + direction_strength * torch.tanh(direction_conv(orientation))
```

Apply distinct factors to the three detail channels only. Keep the low-pass factor at one.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest tests/test_reliable_frequency.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/model/frequency/reliable_injector.py tests/test_reliable_frequency.py
git commit -m "feat: add reliable residual-frequency injector"
```

### Task 2: Mode routing and organ/SUV regression protection

**Files:**
- Modify: `src/model/slmf_bbdm.py`
- Modify: `tests/test_reliable_frequency.py`
- Modify: `tests/test_residual_frequency.py`

- [ ] **Step 1: Write failing routing tests**

```python
def test_reliable_mode_does_not_expose_noisy_state_modulation():
    from src.model.slmf_bbdm import SLMFBBDM

    cfg = _residual_config(frequency=True, gabor=False)
    cfg["modules"]["residual_frequency"].update({
        "mode": "reliable",
        "use_directional_gate": False,
        "band_scales": [1.0, 0.5, 0.25, 0.25],
    })
    model = SLMFBBDM.from_config(cfg)
    assert model.residual_frequency_mode == "reliable"
    assert not hasattr(model.residual_preconditioner, "modulate_residual")


def test_reliable_frequency_is_added_without_gating_adapter_output(monkeypatch):
    from src.model.interfaces import ConditionBundle
    from src.model.slmf_bbdm import SLMFBBDM

    cfg = _residual_config(frequency=True, gabor=False)
    cfg["modules"]["residual_frequency"].update({
        "mode": "reliable",
        "use_directional_gate": False,
        "band_scales": [1.0, 0.5, 0.25, 0.25],
    })
    cfg["modules"]["zero_adapter"]["enabled"] = True
    cfg["modules"]["organ_prior"] = {"enabled": True, "organ_channels": 6}
    model = SLMFBBDM.from_config(cfg)
    shapes = [(1, 256, 4, 4), (1, 256, 8, 8), (1, 128, 16, 16), (1, 64, 32, 32)]
    adapter = [torch.full(shape, 2.0) for shape in shapes]
    frequency = [torch.full(shape, 3.0) for shape in shapes]
    monkeypatch.setattr(model, "_build_adapter_injections", lambda *a, **k: adapter)
    monkeypatch.setattr(model, "_build_frequency_injections", lambda *a, **k: frequency)
    combined = model._build_skip_injections(
        ConditionBundle.empty(), torch.tensor([10]),
        noisy_residual=torch.randn(1, 1, 32, 32),
    )
    assert all(torch.all(x == 5.0) for x in combined)
```

Place these tests in `tests/test_residual_frequency.py`, which already defines `_residual_config`.

- [ ] **Step 2: Run tests and verify mode-routing failure**

Run: `python -m pytest tests/test_reliable_frequency.py tests/test_residual_frequency.py -q`

Expected: FAIL because only the legacy preconditioner is constructed.

- [ ] **Step 3: Implement explicit legacy/reliable routing**

In `SLMFBBDM.__init__`:

```python
self.residual_frequency_mode = str(frequency_cfg.get("mode", "legacy"))
if self.residual_frequency_mode == "legacy":
    self.residual_preconditioner = ResidualFrequencyPreconditioner(...)
elif self.residual_frequency_mode == "reliable":
    self.residual_preconditioner = ReliableResidualFrequencyInjector(...)
else:
    raise ValueError("modules.residual_frequency.mode must be legacy or reliable")
```

Pass raw CT to reliable mode. Restrict noisy-state modulation to `mode == "legacy" and inject_wavelet == false`. Extract `_build_adapter_injections` so adapter results and frequency results are visibly separate, then add them exactly once.

Directional gating requires enabled Gabor; reliable mode without direction does not. Legacy configuration defaults and checkpoints remain unchanged.

- [ ] **Step 4: Run integration tests**

Run: `python -m pytest tests/test_reliable_frequency.py tests/test_residual_frequency.py tests/test_smoke.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/model/slmf_bbdm.py tests/test_reliable_frequency.py tests/test_residual_frequency.py
git commit -m "feat: route reliable injection independently of clinical priors"
```

### Task 3: Boundary-frequency loss

**Files:**
- Create: `src/model/loss_terms/boundary_frequency.py`
- Modify: `src/model/loss_terms/__init__.py`
- Modify: `src/model/slmf_bbdm.py`
- Create: `tests/test_boundary_frequency.py`

- [ ] **Step 1: Write failing lesion/anatomy/optional-organ tests**

```python
import torch

from src.model.interfaces import ConditionBundle, LossContext


def _square_image(shift=0, size=32):
    image = torch.zeros(1, 1, size, size)
    image[:, :, 10 + shift:22 + shift, 10:22] = 1.0
    return image


def _square_mask(shift=0, size=32):
    return _square_image(shift=shift, size=size)


def _ctx(pred, target, mask, organ_mask=None):
    batch = {"ct": target.clone(), "mask": mask}
    if organ_mask is not None:
        batch["organ_mask"] = organ_mask
    return LossContext(
        model_pred=pred,
        loss_target=target,
        target_pet=target,
        pred_x0=pred,
        timesteps=torch.tensor([1]),
        tau=torch.tensor([0.1]),
        batch=batch,
        condition=ConditionBundle.empty(),
    )


def test_boundary_frequency_loss_penalizes_misaligned_lesion_edge():
    target = _square_image(shift=0)
    aligned = _ctx(pred=target.clone(), target=target, mask=_square_mask(shift=0))
    shifted = _ctx(pred=_square_image(shift=2), target=target, mask=_square_mask(shift=0))
    term = BoundaryFrequencyLoss(boundary_radius=1, active_tau_max=1.0)
    aligned_loss, _ = term(aligned)
    shifted_loss, shifted_logs = term(shifted)
    assert shifted_loss > aligned_loss
    assert shifted_logs["boundary_frequency/lesion_available"] == 1


def test_boundary_frequency_loss_works_without_organ_mask():
    target = _square_image()
    ctx = _ctx(_square_image(shift=1), target, _square_mask())
    loss, logs = BoundaryFrequencyLoss(active_tau_max=1.0)(ctx)
    assert torch.isfinite(loss)
    assert logs["boundary_frequency/organ_available"] == 0


def test_nonempty_organ_mask_activates_optional_organ_boundary():
    target = _square_image()
    organ = torch.zeros(1, 6, 32, 32)
    organ[:, 1, 6:26, 6:26] = 1.0
    ctx = _ctx(_square_image(shift=1), target, _square_mask(), organ_mask=organ)
    _, logs = BoundaryFrequencyLoss(active_tau_max=1.0)(ctx)
    assert logs["boundary_frequency/organ_available"] == 1
```

- [ ] **Step 2: Run tests and verify missing-class failure**

Run: `python -m pytest tests/test_boundary_frequency.py -q`

Expected: FAIL because `BoundaryFrequencyLoss` is missing.

- [ ] **Step 3: Implement morphology, gradients, consensus, and Haar term**

Use max-pooling dilation and complement max-pooling erosion to produce a differentiability-independent binary boundary band. Use finite-difference gradient magnitude for prediction, target, and CT. Normalize CT/target edge maps per sample and form anatomy consensus by multiplication.

Calculate weighted Charbonnier gradient errors for lesion, anatomy consensus, and optional non-empty organ boundaries. Downsample their union with adaptive max pooling and weight absolute level-1 Haar detail error. Return weighted total and detached component logs.

- [ ] **Step 4: Register configuration**

Add a `boundary_frequency` branch to `_build_loss` with every constructor parameter from the design spec. Export the class from `loss_terms/__init__.py`.

- [ ] **Step 5: Run focused and integration tests**

Run: `python -m pytest tests/test_boundary_frequency.py tests/test_residual_frequency.py tests/test_smoke.py -q`

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/model/loss_terms/boundary_frequency.py src/model/loss_terms/__init__.py src/model/slmf_bbdm.py tests/test_boundary_frequency.py
git commit -m "feat: supervise lesion and anatomy boundary frequencies"
```

### Task 4: Boundary evaluation metrics

**Files:**
- Modify: `scripts/evaluate.py`
- Modify: `tests/test_boundary_frequency.py`

- [ ] **Step 1: Write failing pure-NumPy metric tests**

```python
def _square_np(shift=0, size=32):
    image = np.zeros((1, size, size), dtype=np.float32)
    image[:, 10 + shift:22 + shift, 10:22] = 1.0
    return image


def test_boundary_metrics_reward_aligned_lesion_edges_and_report_anatomy_proxy():
    target = _square_np(shift=0)
    mask = _square_np(shift=0)
    ct = _square_np(shift=0)
    aligned = compute_boundary_metrics(target, target, ct, mask, np.zeros((6, 32, 32)))
    shifted = compute_boundary_metrics(_square_np(shift=2), target, ct, mask, np.zeros((6, 32, 32)))
    assert aligned["lesion_boundary_gradient_mae_norm"] < shifted["lesion_boundary_gradient_mae_norm"]
    assert np.isfinite(aligned["anatomy_edge_gradient_mae_norm"])
    assert np.isnan(aligned["organ_boundary_gradient_mae_norm"])
```

- [ ] **Step 2: Run tests and verify missing-function failure**

Run: `python -m pytest tests/test_boundary_frequency.py -q`

Expected: FAIL because `compute_boundary_metrics` is missing.

- [ ] **Step 3: Implement and integrate metrics**

Add `compute_boundary_metrics(pred, target, ct, lesion_mask, organ_mask, boundary_radius=2, anatomy_quantile=0.80)`. Use NumPy/SciPy morphology and finite differences. Return NaN for lesion or organ metrics when their masks are empty.

Call it for every evaluation sample using model-space normalized CT/PET. Existing aggregation already omits NaN values. Add the new metrics to the printed `Boundary Fidelity` group.

- [ ] **Step 4: Run evaluator tests**

Run: `python -m pytest tests/test_boundary_frequency.py tests/test_trainer_monitoring.py tests/test_smoke.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```powershell
git add scripts/evaluate.py tests/test_boundary_frequency.py
git commit -m "feat: report lesion and anatomy boundary fidelity"
```
