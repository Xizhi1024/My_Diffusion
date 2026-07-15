# PNG Baseline Validation Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair signed-range lesion metrics, deterministic validation sampling, checkpoint selection, early stopping, and loss observability without changing the model or training loss formulas.

**Architecture:** Keep orchestration in `Trainer`, but add pure signed-range metric and stratified-index helpers in `src/model/trainer.py` so they can be tested without a model. Cache one fixed-seed sampled tensor per evaluation epoch and reuse it for metrics and the sample grid. Let checkpoint saving return the single combined-improvement decision consumed by epoch-based early stopping.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, pytest, YAML.

**Execution note:** Work in the current repository because the user manually syncs this exact working tree to the cloud. Preserve the unrelated uncommitted changes in `src/data/dataset.py` and `src/data/png_cache.py` and never stage them in monitoring commits.

---

### Task 1: Signed-range-safe lesion metrics

**Files:**
- Modify: `src/model/trainer.py:28-48,422-485`
- Create: `tests/test_trainer_monitoring.py`

- [ ] **Step 1: Write the failing signed-range and empty-mask tests**

Create `tests/test_trainer_monitoring.py`:

```python
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.trainer import _compute_pet_sample_metrics


def test_signed_pet_metrics_do_not_select_zeroed_mask_background():
    pred = np.full((5, 5), -1.0, dtype=np.float32)
    target = np.full((5, 5), -1.0, dtype=np.float32)
    mask = np.zeros((5, 5), dtype=np.float32)
    mask[2, 2] = 1.0
    pred[2, 2] = -0.2
    target[2, 2] = 0.6
    pred[0, 0] = -0.6

    metrics = _compute_pet_sample_metrics(pred, target, mask)

    assert metrics is not None
    assert metrics["lesion_centroid_distance"] == pytest.approx(0.0)
    assert metrics["lesion_peak_error_norm"] == pytest.approx(0.4)
    assert metrics["outside_inside_peak_ratio"] == pytest.approx(0.5)
    assert metrics["failure"] == 0.0
    assert metrics["lesion_roi_l1"] == pytest.approx(0.4)


def test_signed_pet_metrics_skip_empty_mask():
    pred = np.zeros((4, 4), dtype=np.float32)
    target = np.zeros((4, 4), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=np.float32)
    assert _compute_pet_sample_metrics(pred, target, mask) is None
```

- [ ] **Step 2: Run the tests and verify RED**

Run `python -m pytest tests/test_trainer_monitoring.py -q`.

Expected: collection fails because `_compute_pet_sample_metrics` does not exist.

- [ ] **Step 3: Implement the pure metric helper**

Add below `_stripe_score` in `src/model/trainer.py`:

```python
def _to_unit_interval(array: np.ndarray) -> np.ndarray:
    return np.clip((array.astype(np.float32) + 1.0) * 0.5, 0.0, 1.0)


def _compute_pet_sample_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> Optional[Dict[str, float]]:
    valid = mask > 0.5
    if not np.any(valid):
        return None
    pred_unit = _to_unit_interval(pred)
    target_unit = _to_unit_interval(target)
    outside = ~valid
    pred_in_peak = float(pred_unit[valid].max())
    target_in_peak = float(target_unit[valid].max())
    out_peak = float(pred_unit[outside].max()) if np.any(outside) else 0.0
    ys, xs = np.nonzero(valid)
    peak_index = int(np.argmax(pred_unit[valid]))
    py, px = float(ys[peak_index]), float(xs[peak_index])
    cy, cx = float(ys.mean()), float(xs.mean())
    return {
        "lesion_peak_error_norm": abs(pred_in_peak - target_in_peak),
        "lesion_centroid_distance": float(np.hypot(py - cy, px - cx)),
        "outside_inside_peak_ratio": out_peak / max(pred_in_peak, 1e-6),
        "failure": float(out_peak > pred_in_peak),
        "lesion_roi_l1": float(np.abs(pred_unit[valid] - target_unit[valid]).mean()),
    }
```

Replace mask multiplication in `_compute_val_sample_metrics` with this helper. Accumulate only non-`None` results, divide failure rate by the valid-lesion sample count, add `val/lesion_sample_count`, and average the helper's per-sample `lesion_roi_l1` values instead of the old tensor-wide calculation.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run `python -m pytest tests/test_trainer_monitoring.py -q`.

Expected: `2 passed`.

- [ ] **Step 5: Commit the metric fix**

```powershell
git add -- src/model/trainer.py tests/test_trainer_monitoring.py
git commit -m "fix: make lesion monitoring signed-range safe"
```

### Task 2: Deterministic stratified validation subset

**Files:**
- Modify: `src/model/trainer.py:88-158,306-340,351-420`
- Modify: `tests/test_trainer_monitoring.py`

- [ ] **Step 1: Write failing tests for stratification and RNG isolation**

Append:

```python
import torch

from src.model.trainer import Trainer, _stratified_indices


def test_stratified_indices_are_deterministic_unique_and_cover_endpoints():
    indices = _stratified_indices(total=30, count=16)
    assert indices == _stratified_indices(total=30, count=16)
    assert len(indices) == 16
    assert len(set(indices)) == 16
    assert indices[0] == 0
    assert indices[-1] == 29


def test_eval_sampling_is_repeatable_without_advancing_outer_rng():
    class RandomSampleModel:
        def sample(self, batch):
            return {"synthetic_pet": torch.randn_like(batch["ct"])}

    trainer = object.__new__(Trainer)
    trainer.device = "cpu"
    trainer.eval_seed = 123
    trainer.model = RandomSampleModel()
    batch = {"ct": torch.zeros(2, 1, 4, 4)}
    torch.manual_seed(999)
    before = torch.random.get_rng_state().clone()
    first = trainer._sample_with_eval_seed(batch)["synthetic_pet"]
    after = torch.random.get_rng_state().clone()
    second = trainer._sample_with_eval_seed(batch)["synthetic_pet"]
    assert torch.equal(before, after)
    assert torch.equal(first, second)
```

- [ ] **Step 2: Run the new tests and verify RED**

Run `python -m pytest tests/test_trainer_monitoring.py -q`.

Expected: import or attribute failures for `_stratified_indices` and `_sample_with_eval_seed`.

- [ ] **Step 3: Add configuration and stratified selection**

In `Trainer.__init__`, add:

```python
self.eval_num_samples = int(run_cfg.get("eval_num_samples", 16))
self.eval_seed = int(run_cfg.get("eval_seed", config.get("experiment", {}).get("seed", 42)))
```

Add the pure helper:

```python
def _stratified_indices(total: int, count: int) -> List[int]:
    if total <= 0 or count <= 0:
        return []
    count = min(total, count)
    return np.rint(np.linspace(0, total - 1, num=count)).astype(int).tolist()
```

In `_select_tracked_batch`, use `_stratified_indices(n, self.eval_num_samples)` when no explicit IDs are configured. Explicit IDs keep every match. Keep `_save_sample_grid` at four rows and update the selection log to `fixed validation samples`.

- [ ] **Step 4: Add isolated deterministic sampling**

Add:

```python
@torch.no_grad()
def _sample_with_eval_seed(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    cuda_devices: List[int] = []
    device = torch.device(self.device)
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(self.eval_seed)
        return self.model.sample(_to_device(batch, self.device))
```

Change `_compute_val_sample_metrics` to accept `synth: Optional[torch.Tensor] = None`; when omitted, obtain it from `_sample_with_eval_seed`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run `python -m pytest tests/test_trainer_monitoring.py -q`.

Expected: `4 passed`.

- [ ] **Step 6: Commit deterministic validation**

```powershell
git add -- src/model/trainer.py tests/test_trainer_monitoring.py
git commit -m "fix: make validation subset and sampling deterministic"
```

### Task 3: Shared checkpoint improvement and epoch-based early stopping

**Files:**
- Modify: `src/model/trainer.py:141-154,312-328,514-562`
- Modify: `tests/test_trainer_monitoring.py`

- [ ] **Step 1: Write failing early-stop tests**

Append:

```python
def test_early_stopping_patience_is_measured_in_epochs():
    trainer = object.__new__(Trainer)
    trainer.early_stopping_enabled = True
    trainer.early_stopping_patience = 40
    trainer.early_stopping_min_epochs = 50
    trainer._last_combined_improvement_epoch = None
    trainer._epochs_since_improve = 0
    trainer.epoch_count = 50
    assert trainer._check_early_stopping(improved=True) is False
    trainer.epoch_count = 80
    assert trainer._check_early_stopping(improved=False) is False
    assert trainer._epochs_since_improve == 30
    trainer.epoch_count = 90
    assert trainer._check_early_stopping(improved=False) is True


def test_early_stopping_improvement_resets_epoch_origin():
    trainer = object.__new__(Trainer)
    trainer.early_stopping_enabled = True
    trainer.early_stopping_patience = 40
    trainer.early_stopping_min_epochs = 0
    trainer._last_combined_improvement_epoch = 50
    trainer._epochs_since_improve = 30
    trainer.epoch_count = 80
    assert trainer._check_early_stopping(improved=True) is False
    assert trainer._last_combined_improvement_epoch == 80
    assert trainer._epochs_since_improve == 0
```

- [ ] **Step 2: Run the early-stop tests and verify RED**

Run `python -m pytest tests/test_trainer_monitoring.py -q`.

Expected: failure because `_check_early_stopping` still expects a score and counts calls.

- [ ] **Step 3: Return the combined checkpoint improvement**

Change `_save_best_checkpoints` to return `bool`. Compute `combined_improved = combined > self._best_combined_score` before updating scores. Always update score state; guard only file writes with `best_ckpts_enabled`, then `return combined_improved`. This keeps early stopping valid when checkpoint file creation is disabled.

- [ ] **Step 4: Implement epoch-based early stopping**

Initialize `self._last_combined_improvement_epoch: Optional[int] = None`. Replace the method with:

```python
def _check_early_stopping(self, improved: bool) -> bool:
    if not self.early_stopping_enabled:
        return False
    if improved or self._last_combined_improvement_epoch is None:
        self._last_combined_improvement_epoch = self.epoch_count
        self._epochs_since_improve = 0
        return False
    self._epochs_since_improve = self.epoch_count - self._last_combined_improvement_epoch
    if self.epoch_count < self.early_stopping_min_epochs:
        return False
    if self._epochs_since_improve >= self.early_stopping_patience:
        print(
            f"  Early stopping: no combined-score improvement for "
            f"{self._epochs_since_improve} epochs "
            f"(patience={self.early_stopping_patience})."
        )
        return True
    return False
```

In `run`, call `_save_best_checkpoints` once and pass its returned boolean to `_check_early_stopping`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run `python -m pytest tests/test_trainer_monitoring.py -q`.

Expected: `6 passed`.

- [ ] **Step 6: Commit stopping fix**

```powershell
git add -- src/model/trainer.py tests/test_trainer_monitoring.py
git commit -m "fix: share checkpoint improvement with early stopping"
```

### Task 4: Reuse sampled output and expose validation loss components

**Files:**
- Modify: `src/model/trainer.py:292-343`
- Modify: `configs/experiments/slmf_png_baseline.yaml:40-58`
- Modify: `tests/test_trainer_monitoring.py`

- [ ] **Step 1: Write a failing configuration test**

Append:

```python
import yaml


def test_png_baseline_declares_deterministic_eval_seed():
    with open("configs/experiments/slmf_png_baseline.yaml", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["runtime"]["eval_num_samples"] == 16
    assert cfg["runtime"]["eval_seed"] == cfg["experiment"]["seed"]
```

- [ ] **Step 2: Run the configuration test and verify RED**

Run `python -m pytest tests/test_trainer_monitoring.py::test_png_baseline_declares_deterministic_eval_seed -q`.

Expected: failure because `runtime.eval_seed` is absent.

- [ ] **Step 3: Reuse one sampled output per evaluation epoch**

In `Trainer.run`, create `tracked_sample_result = None` at the start of each epoch. During the EMA evaluation scope, sample once with `_sample_with_eval_seed`, compute metrics from `tracked_sample_result["synthetic_pet"]`, and save best checkpoints while EMA weights remain applied. At sample time, only sample if no evaluation result exists. Run early stopping after leaving the EMA scope with the returned `combined_improved` boolean.

- [ ] **Step 4: Print existing loss components**

After `Eval Loss`, print:

```python
print(
    "  Eval components: "
    f"base={avg_eval.get('loss/base_diffusion', float('nan')):.4f}  "
    f"roi={avg_eval.get('loss/lesion_roi_l1/loss', float('nan')):.4f}  "
    f"topk={avg_eval.get('loss/topk_lesion/loss', float('nan')):.4f}  "
    f"ranking={avg_eval.get('loss/outside_peak_ranking/loss', float('nan')):.4f}"
)
```

These are the existing diagnostic values; no loss formula changes.

- [ ] **Step 5: Add the fixed evaluation seed to YAML**

Add under the runtime evaluation settings:

```yaml
  eval_seed: 42                     # fixed sampling noise for cross-epoch comparison
```

Keep `eval_num_samples: 16`, `eval_interval: 50`, and all loss weights unchanged.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run `python -m pytest tests/test_trainer_monitoring.py -q`.

Expected: `7 passed`.

- [ ] **Step 7: Commit orchestration and config**

```powershell
git add -- src/model/trainer.py tests/test_trainer_monitoring.py configs/experiments/slmf_png_baseline.yaml
git commit -m "feat: stabilize PNG baseline validation monitoring"
```

### Task 5: Regression verification and cloud handoff

**Files:**
- Verify: `src/model/trainer.py`
- Verify: `tests/test_trainer_monitoring.py`
- Verify: `tests/test_png_baseline.py`
- Verify: `tests/test_smoke.py`

- [ ] **Step 1: Run static diff checks**

Run `git diff --check HEAD~4..HEAD` and `git status --short`.

Expected: no whitespace errors; only the user's pre-existing `src/data/dataset.py` and `src/data/png_cache.py` remain uncommitted.

- [ ] **Step 2: Run focused tests**

Run `python -m pytest tests/test_trainer_monitoring.py tests/test_png_baseline.py -q`.

Expected: all tests pass.

- [ ] **Step 3: Run the full smoke suite**

Run `python -m pytest tests/test_smoke.py -q`.

Expected: all tests pass with no new warning or error.

- [ ] **Step 4: Inspect final scope**

Run `git log --oneline -5`, `git show --stat --oneline HEAD~3..HEAD`, and a scoped diff for `src/model/trainer.py`, `tests/test_trainer_monitoring.py`, and `configs/experiments/slmf_png_baseline.yaml`.

Expected: no model, cache, dataset, or loss-formula changes.

- [ ] **Step 5: Prepare cloud handoff**

Report the exact files to sync. Preserve the cloud checkpoint directory and first run checkpoint-only evaluation of `checkpoints/slmf_png_baseline/ckpt_best_combined.pt` under the repaired fixed-16 protocol; do not launch a new 300-epoch training run until those corrected metrics are reviewed.
