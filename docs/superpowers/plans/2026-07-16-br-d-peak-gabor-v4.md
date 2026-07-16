# BR-D Peak Preservation and Safe Gabor V4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add normalized lesion Top-Q supervision, scale-specific CT reliability floors, a bounded shallow Gabor agreement gate, and a two-stage V4 ablation runner while preserving BR-D, organ/SUV interfaces, and all V1-V3 outputs.

**Architecture:** Keep BR-D's additive two-level Haar branch unchanged except for independently configurable CT floors and an optional L1 shared Gabor agreement multiplier. Add peak supervision as a loss-only training route and extend validation with normalized peak diagnostics. Use a new V4 orchestrator over the existing generic runner helpers so legacy plans remain compatible.

**Tech Stack:** Python 3.11, PyTorch, NumPy/SciPy, PyYAML, pytest, Pixi.

---

### Task 1: Normalized lesion Top-Q loss

**Files:**
- Create: `src/model/loss_terms/normalized_lesion_peak.py`
- Modify: `src/model/loss_terms/__init__.py`
- Modify: `src/model/slmf_bbdm.py`
- Test: `tests/test_normalized_lesion_peak.py`

- [ ] **Step 1: Write failing loss tests**

Cover independent prediction/target Top-Q selection, `ceil(q*n)` clamping,
per-sample timestep gating, empty-mask exclusion, `pred_x0` preference, and
gradient flow. Use a helper that constructs `LossContext` with two samples so
one empty mask cannot dilute the valid sample.

```python
loss = NormalizedLesionPeakLoss(
    topk_percent=0.10, min_k=3, max_k=16,
    beta=0.02, active_tau_max=0.25, weight=1.0,
)
value, logs = loss(ctx)
assert torch.isfinite(value)
assert value.requires_grad
value.backward()
assert ctx.pred_x0.grad is not None
assert logs["normalized_lesion_peak/valid_count"].item() == 1
```

- [ ] **Step 2: Verify tests fail**

Run: `pixi run python -m pytest tests/test_normalized_lesion_peak.py -q`

Expected: import failure because `NormalizedLesionPeakLoss` does not exist.

- [ ] **Step 3: Implement the loss**

Implement constructor validation and a per-sample loop. For each valid mask:

```python
count = int(selected_pred.numel())
k = min(max(math.ceil(count * self.topk_percent), self.min_k), self.max_k, count)
pred_peak = torch.topk(selected_pred, k=k).values.mean()
target_peak = torch.topk(selected_target, k=k).values.mean()
per_sample = F.smooth_l1_loss(pred_peak, target_peak, beta=self.beta)
```

Multiply each valid sample by its own `smooth_tau_gate`, then divide by the
number of valid masks. Return unweighted diagnostics for loss, predicted peak,
target peak, signed bias, gate mean, and valid count.

Register `normalized_lesion_peak` in `SLMFBBDM._build_loss` with all six config
fields and export the class from `loss_terms.__init__`.

- [ ] **Step 4: Run focused tests**

Run: `pixi run python -m pytest tests/test_normalized_lesion_peak.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/model/loss_terms/normalized_lesion_peak.py src/model/loss_terms/__init__.py src/model/slmf_bbdm.py tests/test_normalized_lesion_peak.py
git commit -m "feat: add normalized lesion peak loss"
```

### Task 2: Scale-specific CT reliability floors

**Files:**
- Modify: `src/model/frequency/boundary_reliable.py`
- Modify: `src/model/slmf_bbdm.py`
- Test: `tests/test_reliable_frequency.py`

- [ ] **Step 1: Write failing floor tests**

Add tests for constructor validation, exact endpoints, monotonicity, separate
L2/L1 floors, and L0 inheriting the floored L1 details.

```python
injector = _injector(ct_reliability_floors=(0.25, 0.50))
raw = torch.tensor([0.0, 0.4, 1.0])
assert torch.allclose(
    injector.apply_reliability_floor(raw, 0.50),
    torch.tensor([0.50, 0.70, 1.00]),
)
```

- [ ] **Step 2: Verify focused failure**

Run: `pixi run python -m pytest tests/test_reliable_frequency.py -q`

Expected: failure for the missing floor API/configuration.

- [ ] **Step 3: Implement floors**

Add `ct_reliability_floors: Sequence[float] = (0.0, 0.0)` in native order
`[L2, L1]`, validate both values in `[0, 1]`, and register them as a buffer.

```python
@staticmethod
def apply_reliability_floor(reliability, floor):
    return floor + (1.0 - floor) * reliability
```

Pass `level_index` into cross-modal reliability and apply the corresponding
floor after the exponential reliability. A floor of `1.0` must produce exact
ones. Parse `ct_reliability_floor_l2` and `ct_reliability_floor_l1` in
`SLMFBBDM` without changing legacy mode.

- [ ] **Step 4: Run focused and compatibility tests**

Run: `pixi run python -m pytest tests/test_reliable_frequency.py tests/test_residual_frequency.py -q`

Expected: all tests pass, including organ-adapter and ROI-SUV compatibility.

- [ ] **Step 5: Commit**

```powershell
git add src/model/frequency/boundary_reliable.py src/model/slmf_bbdm.py tests/test_reliable_frequency.py
git commit -m "feat: add CT reliability floors"
```

### Task 3: Shared shallow Gabor agreement gate

**Files:**
- Modify: `src/model/frequency/boundary_reliable.py`
- Modify: `src/model/slmf_bbdm.py`
- Test: `tests/test_reliable_frequency.py`
- Test: `tests/test_residual_frequency.py`

- [ ] **Step 1: Write failing agreement tests**

Test agreement bounds, identical/disagreeing distributions, exact no-op when
disabled/missing/isotropic, descriptor detachment, L1-only behavior, and one
shared multiplier across LH/HL/HH.

```python
agreement = injector.gabor_agreement(q_g, q_h)
assert torch.all((agreement >= 0) & (agreement <= 1))
assert torch.allclose(agreement_same, torch.ones_like(agreement_same))

multiplier = injector.gabor_agreement_reliability(
    orientation, anisotropy, residual_details, level_index=1,
)
assert multiplier.shape[1] == 1
assert multiplier.min() >= 0.90
assert multiplier.max() <= 1.00
```

- [ ] **Step 2: Verify tests fail**

Run: `pixi run python -m pytest tests/test_reliable_frequency.py tests/test_residual_frequency.py -q`

Expected: missing agreement configuration and anisotropy route.

- [ ] **Step 3: Implement detached shared agreement**

Add configuration fields:

```python
use_gabor_agreement: bool = False
gabor_agreement_alpha: float = 0.10
gabor_agreement_l2: bool = False
gabor_agreement_l1: bool = True
detach_gabor_descriptor: bool = True
```

Map orientation energy to three bands with the current orientation weights,
normalize across bands, and form residual probabilities from `avg_pool2d` of
absolute Haar details. Compute:

```python
agreement = torch.sqrt(q_g * q_h).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
reliability = (1.0 - alpha * anisotropy * (1.0 - agreement)).clamp(1.0 - alpha, 1.0)
```

Multiply this one-channel reliability into all three BR-D subband gates only
at enabled levels. Keep the existing `use_directional_reliability` path intact
as the G1 control. Pass `gabor_anisotropy` from the condition bundle and detach
both Gabor tensors for G2.

- [ ] **Step 4: Run focused tests**

Run: `pixi run python -m pytest tests/test_reliable_frequency.py tests/test_residual_frequency.py -q`

Expected: all tests pass and BR-D initial no-op remains exact.

- [ ] **Step 5: Commit**

```powershell
git add src/model/frequency/boundary_reliable.py src/model/slmf_bbdm.py tests/test_reliable_frequency.py tests/test_residual_frequency.py
git commit -m "feat: add safe Gabor agreement reliability"
```

### Task 4: Peak diagnostics and checkpoint monitoring

**Files:**
- Modify: `scripts/evaluate.py`
- Modify: `src/model/trainer.py`
- Test: `tests/test_trainer_monitoring.py`
- Create: `tests/test_peak_diagnostics.py`

- [ ] **Step 1: Write failing diagnostic tests**

Use synthetic lesions with known core/ring peaks to test signed single-pixel
bias, signed/absolute Top-Q error, under/over flags, core fallback, ring/core
ratio, peak-to-boundary distance, and lesion size.

```python
metrics = compute_normalized_lesion_metrics(
    pred, target, mask, organ, topk_percent=0.10, min_k=3, max_k=16,
)
assert metrics["lesion_topq_peak_error_norm"] == pytest.approx(expected)
assert metrics["lesion_topq_peak_signed_bias_norm"] == pytest.approx(expected_bias)
assert metrics["lesion_size"] == mask.sum()
```

Add trainer tests proving `val/lesion_topq_peak_error_norm` is present and a
worse Top-Q error lowers the lesion/model-selection score.

- [ ] **Step 2: Verify tests fail**

Run: `pixi run python -m pytest tests/test_peak_diagnostics.py tests/test_trainer_monitoring.py -q`

Expected: missing V4 metrics.

- [ ] **Step 3: Implement diagnostics**

Extend `_compute_pet_sample_metrics` and `compute_normalized_lesion_metrics`
with deterministic Top-Q selection in unit-interval normalized space. Build a
one-pixel eroded core and a two-pixel dilated ring using SciPy morphology;
record `core_fallback=1` when erosion is empty.

Aggregate the new scalar fields through the evaluator's existing generic
summary and per-patient paths. Extend trainer validation and selection:

```python
lesion_score = -(
    topq_peak + 0.5 * peak + 0.02 * centroid + 0.5 * failure
)
```

Keep original peak metrics and failure logic unchanged.

- [ ] **Step 4: Run focused tests**

Run: `pixi run python -m pytest tests/test_peak_diagnostics.py tests/test_trainer_monitoring.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add scripts/evaluate.py src/model/trainer.py tests/test_peak_diagnostics.py tests/test_trainer_monitoring.py
git commit -m "feat: diagnose normalized lesion peaks"
```

### Task 5: V4 presets, plan, and two-stage runner

**Files:**
- Modify: `configs/experiments/ablations.yaml`
- Create: `configs/experiments/slmf_png_boundary_reliable_v4.yaml`
- Create: `configs/experiments/boundary_reliable_ablation_plan_v4.yaml`
- Modify: `scripts/run_frequency_ablations.py`
- Create: `scripts/run_boundary_reliable_v4.py`
- Create: `scripts/compare_v4_results.py`
- Modify: `tests/test_frequency_ablation_runner.py`

- [ ] **Step 1: Write failing runner/config tests**

Test D0/D1/D3/D4 preset distinctions, fixed 64-sample screens, dynamic Stage B
inheritance, G0/G1/G2 route differences, stage stop when no D candidate passes,
at most two promotion runs, V4 scoring with Top-Q, absolute gates, and mandatory
`ckpt_epoch0300.pt` completion. Add deterministic paired-report tests using
10,000 patient bootstrap resamples, percentile bounds, and seed 42.

```python
assert plan["stage_a"]["max_samples"] == 64
assert plan["stage_b"]["max_samples"] == 64
assert manifest["promotion_runs"][0]["completion_checkpoint"].endswith("ckpt_epoch0300.pt")
```

- [ ] **Step 2: Verify tests fail**

Run: `pixi run python -m pytest tests/test_frequency_ablation_runner.py -q`

Expected: missing V4 plan/orchestrator support.

- [ ] **Step 3: Extend generic helpers without changing V1-V3 behavior**

Add optional `min_value`/`max_value` hard gates, optional variant-specific CLI
overrides, V4 composite scoring when Top-Q is present, and an optional
`require_final_checkpoint` entry. When required, `_run_entries` skips training
only if both the best checkpoint and the final epoch checkpoint exist.

- [ ] **Step 4: Implement V4 orchestrator and configuration**

The new runner executes:

```text
Stage A D0/D1/D3/D4 screen
  -> gate and select highest passing D candidate
  -> Stage B G0/G1/G2 fresh screen using selected D preset
  -> gate and promote at most two
  -> fresh exact-300 training
  -> final hard-gate decision
```

Write outputs only under `results/boundary_reliable_ablations_v4` and
checkpoints with `br_v4_` prefixes. Reuse the frozen V2 mean checkpoint. The
V4 base config keeps boundary loss, gate-TV, complete Wavelet U-Net, direct
Gabor routes, organ prior, and ROI-SUV disabled.

`compare_v4_results.py` reads promoted result JSON files, intersects patient
IDs, and writes win/tie/loss, median paired difference, and deterministic 95%
bootstrap intervals for every V4 ranking metric. Fewer than two finite paired
patients produces a count and null interval. The orchestrator writes this to
`results/boundary_reliable_ablations_v4/paired_comparison.json` after promoted
evaluations complete.

- [ ] **Step 5: Run runner/config tests and dry-run**

Run:

```powershell
pixi run python -m pytest tests/test_frequency_ablation_runner.py -q
pixi run python scripts/run_boundary_reliable_v4.py --plan configs/experiments/boundary_reliable_ablation_plan_v4.yaml --stage all --dry-run
```

Expected: tests pass; dry-run emits four Stage A, three Stage B templates, and
no V1-V3 output paths.

- [ ] **Step 6: Commit**

```powershell
git add configs/experiments/ablations.yaml configs/experiments/slmf_png_boundary_reliable_v4.yaml configs/experiments/boundary_reliable_ablation_plan_v4.yaml scripts/run_frequency_ablations.py scripts/run_boundary_reliable_v4.py scripts/compare_v4_results.py tests/test_frequency_ablation_runner.py
git commit -m "feat: orchestrate two-stage V4 ablations"
```

### Task 6: Integration verification and handoff

**Files:**
- Modify only if verification reveals a defect in the files above.

- [ ] **Step 1: Run focused V4 suite**

Run:

```powershell
pixi run python -m pytest tests/test_normalized_lesion_peak.py tests/test_peak_diagnostics.py tests/test_reliable_frequency.py tests/test_residual_frequency.py tests/test_frequency_ablation_runner.py tests/test_trainer_monitoring.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run complete suite**

Run: `pixi run python -m pytest -q`

Expected: all tests pass with no regression from the 244-test V3 baseline.

- [ ] **Step 3: Verify repository boundaries**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; `wuzhe/` and `wuzhe_data/` remain untouched and
untracked unless the user independently changes them.

- [ ] **Step 4: Provide the cloud command**

```powershell
pixi run python scripts/run_boundary_reliable_v4.py --plan configs/experiments/boundary_reliable_ablation_plan_v4.yaml --stage all
```
