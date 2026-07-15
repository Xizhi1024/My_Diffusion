# Reliable Wavelet Ablation V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible V3 ablations for the wavelet backbone, reliable injector, directional refinement, and boundary loss, with fixed 300-epoch promotion training and boundary-aware ranking.

**Architecture:** Existing generic command builders and artifact gates are retained. New presets expose eight scientifically identifiable variants. A new plan reuses the frozen V2 mean, screens all variants, promotes at most two, disables early stopping for full runs, and evaluates EMA best-combined checkpoints on fixed subsets.

**Tech Stack:** YAML, Python subprocess orchestration, pytest

---

### Task 1: Boundary-aware promotion score

**Files:**
- Modify: `scripts/run_frequency_ablations.py`
- Modify: `tests/test_frequency_ablation_runner.py`

- [ ] **Step 1: Write failing ranking tests**

```python
def test_composite_score_rewards_boundary_fidelity_after_artifact_metrics():
    good = _metrics(
        lesion_boundary_gradient_mae_norm_mean=0.03,
        anatomy_edge_gradient_mae_norm_mean=0.04,
    )
    bad = _metrics(
        lesion_boundary_gradient_mae_norm_mean=0.12,
        anatomy_edge_gradient_mae_norm_mean=0.15,
    )
    assert composite_score(good) > composite_score(bad)


def test_composite_score_requires_new_boundary_metrics():
    metrics = _metrics()
    metrics.pop("lesion_boundary_gradient_mae_norm_mean")
    with pytest.raises(KeyError):
        composite_score(metrics)
```

Update `_metrics` defaults with both new boundary keys.

- [ ] **Step 2: Run tests and verify score does not distinguish boundaries**

Run: `python -m pytest tests/test_frequency_ablation_runner.py -q`

Expected: FAIL because `composite_score` ignores boundary metrics.

- [ ] **Step 3: Add bounded boundary terms**

Use:

```python
lesion_boundary = metric_value(metrics, "lesion_boundary_gradient_mae_norm_mean")
anatomy_boundary = metric_value(metrics, "anatomy_edge_gradient_mae_norm_mean")
lesion = -(peak + 0.02 * centroid + 0.50 * failure + 0.20 * lesion_boundary)
image = ssim - mae - 0.20 * stripe - 100.0 * hotspot - 0.10 * anatomy_boundary
```

Do not require the unavailable organ-boundary metric.

- [ ] **Step 4: Run runner tests**

Run: `python -m pytest tests/test_frequency_ablation_runner.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add scripts/run_frequency_ablations.py tests/test_frequency_ablation_runner.py
git commit -m "feat: rank ablations with boundary fidelity"
```

### Task 2: V3 model config and eight ablation presets

**Files:**
- Create: `configs/experiments/slmf_png_reliable_wavelet.yaml`
- Modify: `configs/experiments/ablations.yaml`
- Modify: `tests/test_frequency_ablation_runner.py`

- [ ] **Step 1: Write failing preset-resolution test**

```python
@pytest.mark.parametrize("preset", [
    "rw_r2", "rw_lf4", "rw_w1", "rw_i1",
    "rw_wi", "rw_wi_g", "rw_wi_l", "rw_wi_f",
])
def test_reliable_wavelet_presets_resolve_and_construct(preset):
    from src.model.config_utils import load_full_config, validate_png_baseline_config
    from src.model.slmf_bbdm import SLMFBBDM

    config = load_full_config(
        "configs/experiments/slmf_png_reliable_wavelet.yaml",
        ablation=preset,
        ablation_config_path="configs/experiments/ablations.yaml",
        overrides=["data.image_size=32", "runtime.eval_sampling_steps=2"],
    )
    validate_png_baseline_config(config)
    model = SLMFBBDM.from_config(config)
    assert model is not None
```

- [ ] **Step 2: Run test and verify missing-preset failure**

Run: `python -m pytest tests/test_frequency_ablation_runner.py -q`

Expected: FAIL because the V3 base config and presets do not exist.

- [ ] **Step 3: Create the V3 base config**

Copy the validated residual-bridge/training/loss settings from `slmf_png_residual_frequency.yaml`, then set the recommended full model:

```yaml
modules:
  wavelet_unet: {enabled: true, mix_kernel_size: 3}
  residual_frequency:
    enabled: true
    mode: reliable
    output_channels: [256, 256, 128, 64]
    band_scales: [1.0, 0.5, 0.25, 0.25]
    snr_center: 0.0
    snr_temperature: 2.0
    reliability_floor: 0.02
    use_directional_gate: true
    gabor_orientations: 8
    direction_strength: 0.1
losses:
  boundary_frequency:
    enabled: true
    weight: 0.05
    lesion_weight: 1.0
    anatomy_weight: 0.5
    organ_weight: 0.5
    wavelet_weight: 0.25
    boundary_radius: 2
    active_tau_max: 0.7
```

Keep `organ_prior`, `roi_suv`, and physical SUV evaluation disabled for current PNG data, but retain their unchanged config blocks.

- [ ] **Step 4: Add exact preset overrides**

Create the eight IDs from the design table. `rw_lf4` must set `mode: legacy` and reproduce F4. R2/W1 must disable residual frequency. I1/WI/WI-L must set `use_directional_gate: false`. Only WI-L/WI-F enable boundary loss. Disable legacy residual-wavelet/Gabor losses in every reliable-mode variant.

- [ ] **Step 5: Run preset construction and forward tests**

Run: `python -m pytest tests/test_frequency_ablation_runner.py tests/test_reliable_frequency.py tests/test_wavelet_unet.py -q`

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```powershell
git add configs/experiments/slmf_png_reliable_wavelet.yaml configs/experiments/ablations.yaml tests/test_frequency_ablation_runner.py
git commit -m "config: add reliable wavelet ablation presets"
```

### Task 3: Fixed-budget V3 cloud plan

**Files:**
- Create: `configs/experiments/reliable_wavelet_ablation_plan_v3.yaml`
- Modify: `tests/test_frequency_ablation_runner.py`
- Modify: `scripts/run_frequency_ablations.py`

- [ ] **Step 1: Write failing plan invariants**

```python
def test_v3_plan_uses_frozen_mean_and_fixed_300_epoch_promotions():
    plan = yaml.safe_load(Path("configs/experiments/reliable_wavelet_ablation_plan_v3.yaml").read_text())
    assert plan["reference_id"] == "R2"
    assert plan["mean_pretrain"]["enabled"] is False
    assert plan["common_train_overrides"]["modules.conditional_mean.freeze"] is True
    assert plan["screen"] == {**plan["screen"], "epochs": 50, "eval_interval": 10}
    assert plan["promote"]["epochs"] == 300
    assert plan["promote"]["eval_interval"] == 20
    assert plan["promote"]["early_stopping"] is False
    assert plan["top_k"] == 2
    assert plan["output_dir"] == "results/reliable_wavelet_ablations_v3"


def test_v3_promotion_commands_explicitly_disable_early_stopping():
    manifest = build_execution_manifest(plan, python="python", stage="promote", promoted_ids=["WI-F"])
    command = manifest["promotion_runs"][0]["train_command"]
    assert "runtime.early_stopping.enabled=false" in command
    assert "training.num_epochs=300" in command
    assert "runtime.eval_interval=20" in command
```

- [ ] **Step 2: Run test and verify missing-plan failure**

Run: `python -m pytest tests/test_frequency_ablation_runner.py -q`

Expected: FAIL because the V3 plan is missing.

- [ ] **Step 3: Create the V3 plan**

Use:

```yaml
base_config: configs/experiments/slmf_png_reliable_wavelet.yaml
ablation_config: configs/experiments/ablations.yaml
output_dir: results/reliable_wavelet_ablations_v3
reference_id: R2
top_k: 2
mean_pretrain:
  enabled: false
  checkpoint: checkpoints/freq_mean_pretrain_v2/mean_best.pt
common_train_overrides:
  model.initialization_seed: 4242
  modules.conditional_mean.checkpoint: checkpoints/freq_mean_pretrain_v2/mean_best.pt
  modules.conditional_mean.freeze: true
  modules.conditional_mean.loss_weight: 0.0
variants:
  - {id: R2, preset: rw_r2}
  - {id: LF4, preset: rw_lf4}
  - {id: W1, preset: rw_w1}
  - {id: I1, preset: rw_i1}
  - {id: WI, preset: rw_wi}
  - {id: WI-G, preset: rw_wi_g}
  - {id: WI-L, preset: rw_wi_l}
  - {id: WI-F, preset: rw_wi_f}
screen: {experiment_prefix: rw_v3_screen, epochs: 50, eval_interval: 10, split: val, max_samples: 16, seed: 42, mc_steps: 20}
promote: {experiment_prefix: rw_v3_full, epochs: 300, eval_interval: 20, split: val, max_samples: 64, seed: 42, mc_steps: 20, early_stopping: false}
```

Retain V2 artifact gates. Add no hard organ-boundary gate because the metric is unavailable on PNG.

- [ ] **Step 4: Record actual checkpoint epoch in result records**

Add a single helper and call it when `_run_entries` resolves `ckpt_best_combined.pt`:

```python
def checkpoint_epoch(path: Path) -> int:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or "epoch" not in payload:
        raise ValueError(f"Checkpoint {path} is missing integer epoch metadata")
    return int(payload["epoch"])
```

Store the result as `record["checkpoint_epoch"]` before leaderboard writing. Add a unit test that saves `{"epoch": 120}` with `torch.save` and asserts the helper returns `120`; add a second test for missing epoch metadata.

- [ ] **Step 5: Run runner tests and dry run**

Run: `python -m pytest tests/test_frequency_ablation_runner.py -q`

Run: `python scripts/run_frequency_ablations.py --plan configs/experiments/reliable_wavelet_ablation_plan_v3.yaml --stage all --dry-run`

Expected: tests pass; dry-run emits eight screen commands, no promotion commands before screening, and writes only under `results/reliable_wavelet_ablations_v3`.

- [ ] **Step 6: Commit**

```powershell
git add configs/experiments/reliable_wavelet_ablation_plan_v3.yaml scripts/run_frequency_ablations.py tests/test_frequency_ablation_runner.py
git commit -m "config: add fixed-budget reliable wavelet ablation v3"
```

### Task 4: Integrated verification and cloud handoff

**Files:**
- Modify: `docs/superpowers/specs/2026-07-16-reliable-wavelet-bbdm-design.md` only if implementation differs

- [ ] **Step 1: Run complete test suite**

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Run fake-data forwards for all eight presets**

Run: `python -m pytest tests/test_frequency_ablation_runner.py::test_every_reliable_wavelet_preset_completes_a_real_model_forward -q`

Expected: eight parametrized cases pass with finite losses and no CUDA requirement.

- [ ] **Step 3: Run dry-run manifest audit**

Confirm every command pins seed 42, initialization seed 4242, frozen mean checkpoint, EMA evaluation, split, maximum samples, and output directory. Confirm promotion commands contain `early_stopping.enabled=false`.

- [ ] **Step 4: Review compatibility**

Run: `python -m pytest tests/test_residual_frequency.py::TestResidualBBDMIntegration tests/test_wavelet_unet.py tests/test_reliable_frequency.py -q`

Expected: legacy V2 R2/F4 construction remains valid and the wavelet + reliable + optional organ/SUV forward is finite.

- [ ] **Step 5: Commit final verification adjustments**

```powershell
git add -A
git commit -m "test: verify reliable wavelet experiment end to end"
```
