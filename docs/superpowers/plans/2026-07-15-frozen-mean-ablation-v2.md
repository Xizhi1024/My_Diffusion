# Frozen-Mean Residual Ablation V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pretrain one CT-to-PET LL2 mean predictor, freeze the same checkpoint for every residual variant, match core UNet initialization across ablations, and orchestrate the complete V2 screen from one command.

**Architecture:** A focused `MeanPretrainer` trains only `LowFrequencyPETPredictor` and writes a versioned checkpoint. `SLMFBBDM` strictly loads and freezes that checkpoint, while an isolated component seed makes UNet initialization invariant to optional frequency modules. The existing ablation runner gains an optional M0 stage and common V2 overrides without breaking V1 plans.

**Tech Stack:** Python 3.11+, PyTorch, PyYAML, pytest, existing PNG data loaders and SLMF-BBDM CLI stack.

---

### Task 1: Mean-pretraining math and checkpoint contract

**Files:**
- Create: `src/model/mean_pretraining.py`
- Create: `tests/test_mean_pretraining.py`

- [ ] **Step 1: Write failing tests for the target, loss, update, and checkpoint schema**

```python
def test_mean_target_is_pet_ll2():
    pet = torch.randn(2, 1, 32, 32)
    expected_ll1, _ = haar_dwt2(pet)
    expected_ll2, _ = haar_dwt2(expected_ll1)
    assert torch.equal(mean_target_ll2(pet), expected_ll2)

def test_mean_pretrainer_updates_only_predictor_and_saves_versioned_checkpoint(tmp_path):
    predictor = LowFrequencyPETPredictor(base_channels=8)
    trainer = MeanPretrainer(predictor, train_loader, val_loader, config, device="cpu")
    before = {name: value.clone() for name, value in predictor.state_dict().items()}
    history = trainer.run(epochs=1, output_dir=tmp_path)
    checkpoint = torch.load(tmp_path / "mean_best.pt", weights_only=True)
    assert checkpoint["format_version"] == 1
    assert set(checkpoint) == {"format_version", "model", "mean_config", "epoch", "val_loss"}
    assert any(not torch.equal(before[name], value) for name, value in predictor.state_dict().items())
    assert len(history) == 1
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m pytest -q tests/test_mean_pretraining.py
```

Expected: import failure because `src.model.mean_pretraining` does not exist.

- [ ] **Step 3: Implement focused mean-pretraining primitives**

Implement these public interfaces:

```python
def mean_target_ll2(pet: torch.Tensor) -> torch.Tensor:
    ll1, _ = haar_dwt2(pet)
    ll2, _ = haar_dwt2(ll1)
    return ll2

def mean_charbonnier_loss(
    pred_ll2: torch.Tensor,
    target_ll2: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    return torch.sqrt((pred_ll2 - target_ll2).square() + epsilon ** 2).mean()
```

`MeanPretrainer.__init__` accepts predictor, train loader, validation loader,
configuration, and optional device. Its public methods are `train_epoch() ->
float`, `validate() -> float`, and `run(epochs, output_dir) -> list[dict[str,
float]]`.

The optimizer must receive exactly `predictor.parameters()`. `run()` must save
`mean_best.pt`, `mean_last.pt`, and `history.json`, selecting best strictly by
finite validation LL2 loss.

- [ ] **Step 4: Re-run focused tests and verify GREEN**

Expected: all `test_mean_pretraining.py` tests pass.

- [ ] **Step 5: Commit the isolated pretraining core**

```powershell
git add src/model/mean_pretraining.py tests/test_mean_pretraining.py
git commit -m "feat: add conditional mean pretraining core"
```

### Task 2: Standalone M0 CLI

**Files:**
- Create: `scripts/pretrain_conditional_mean.py`
- Modify: `tests/test_mean_pretraining.py`

- [ ] **Step 1: Write a failing CLI command-construction/smoke test**

The test must call the script with fake data, one epoch, image size 32, and a
temporary output directory, then assert that all four M0 artifacts exist and
the best checkpoint has finite `val_loss`.

- [ ] **Step 2: Run the test and verify RED because the script is missing**

- [ ] **Step 3: Implement the CLI**

Required arguments:

```text
--config PATH
--output-dir PATH
--epochs N
--seed N
--override key=value  (repeatable)
```

The script must use `load_full_config`, `resolve_runtime_profile`,
`validate_png_baseline_config`, `build_dataloaders`, and
`save_resolved_config`. It must fail when no validation loader exists.

- [ ] **Step 4: Run the CLI smoke test and focused suite**

- [ ] **Step 5: Commit**

```powershell
git add scripts/pretrain_conditional_mean.py tests/test_mean_pretraining.py
git commit -m "feat: add conditional mean pretraining command"
```

### Task 3: Strict frozen-mean loading in SLMFBBDM

**Files:**
- Modify: `src/model/slmf_bbdm.py`
- Modify: `tests/test_residual_frequency.py`

- [ ] **Step 1: Add failing model tests**

Cover these contracts:

```python
with pytest.raises(ValueError, match="checkpoint"):
    SLMFBBDM.from_config(config_with_freeze_true_but_no_checkpoint)

model = SLMFBBDM.from_config(config_with_valid_mean_checkpoint)
assert all(not parameter.requires_grad for parameter in model.mean_predictor.parameters())
model.train()
assert model.mean_predictor.training is False
assert model.mean_loss_weight == 0.0
```

Also alter saved weights to a known constant and assert strict loading restores
that constant.

- [ ] **Step 2: Run tests and verify RED**

- [ ] **Step 3: Implement checkpoint loading and freeze behavior**

Parse `checkpoint` and `freeze` from `conditional_mean_config`. Load only the
checkpoint's `model` entry with `strict=True`. Reject unsupported
`format_version`, missing files, and freeze-without-checkpoint. Override
`SLMFBBDM.train(mode)` so a frozen predictor stays in eval mode.

- [ ] **Step 4: Re-run focused residual tests and PNG baseline tests**

- [ ] **Step 5: Commit**

```powershell
git add src/model/slmf_bbdm.py tests/test_residual_frequency.py
git commit -m "feat: load and freeze pretrained PET mean"
```

### Task 4: Component-stable UNet initialization

**Files:**
- Modify: `src/model/slmf_bbdm.py`
- Modify: `tests/test_residual_frequency.py`

- [ ] **Step 1: Add a failing exact-equality test**

Construct R2, F1, F2, F3, and F4 with the same
`model.initialization_seed=4242`. Assert every key in `model.unet.state_dict()`
is bitwise equal across variants. Also assert the caller RNG stream is restored
after model construction.

- [ ] **Step 2: Verify RED because optional modules currently perturb UNet initialization**

- [ ] **Step 3: Isolate UNet construction RNG**

Add optional `initialization_seed` to `SLMFBBDM.__init__` and `from_config`.
When set, construct `BBDMUNet` inside:

```python
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(initialization_seed)
    self.unet = BBDMUNet(
        in_channels=3 if self.self_conditioning else 2,
        enable_heteroscedastic=enable_heteroscedastic,
        ca_kv_dim=64,
        meta_dim=meta_dim,
    )
```

When unset, preserve the existing construction behavior for compatibility.

- [ ] **Step 4: Run initialization, residual, and baseline tests**

- [ ] **Step 5: Commit**

```powershell
git add src/model/slmf_bbdm.py tests/test_residual_frequency.py
git commit -m "fix: match UNet initialization across ablations"
```

### Task 5: M0-aware ablation runner

**Files:**
- Modify: `scripts/run_frequency_ablations.py`
- Modify: `tests/test_frequency_ablation_runner.py`

- [ ] **Step 1: Add failing runner tests**

Require:

```python
manifest = build_execution_manifest(v2_plan, python="python", stage="all")
assert manifest["mean_run"]["checkpoint"].endswith("mean_best.pt")
assert "scripts/pretrain_conditional_mean.py" in " ".join(manifest["mean_run"]["command"])
assert all("model.initialization_seed=4242" in " ".join(run["train_command"])
           for run in manifest["screen_runs"])
```

Retain the existing assertion that a V1 plan without `mean_pretrain` produces
`mean_run is None` and unchanged screen commands.

- [ ] **Step 2: Run runner tests and verify RED**

- [ ] **Step 3: Extend the runner**

Add `build_mean_pretrain_command`, optional common train overrides, manifest
field `mean_run`, and `--stage {mean,screen,promote,all}`. Non-dry `all` must
execute M0 before screen. Checkpoint existence controls skip/re-run using the
existing `--force` semantics.

- [ ] **Step 4: Re-run runner tests and V1 dry-run**

- [ ] **Step 5: Commit**

```powershell
git add scripts/run_frequency_ablations.py tests/test_frequency_ablation_runner.py
git commit -m "feat: orchestrate frozen mean before ablations"
```

### Task 6: V2 plan and project task

**Files:**
- Create: `configs/experiments/frequency_ablation_plan_v2.yaml`
- Modify: `pixi.toml`
- Modify: `tests/test_frequency_ablation_runner.py`

- [ ] **Step 1: Add a failing V2 configuration test**

Load the V2 YAML and assert:

```python
assert plan["output_dir"] == "results/frequency_ablations_v2"
assert plan["mean_pretrain"]["epochs"] == 30
assert plan["common_train_overrides"]["modules.conditional_mean.freeze"] is True
assert plan["common_train_overrides"]["modules.conditional_mean.loss_weight"] == 0.0
```

- [ ] **Step 2: Verify RED because the V2 plan is absent**

- [ ] **Step 3: Add the V2 YAML and pixi task**

Use new experiment prefixes `freq_v2_screen` and `freq_v2_full`, seed 42,
component seed 4242, 30 M0 epochs, 50 screen epochs, 300 promoted epochs, and
the unchanged V1 hard gates. Add:

```toml
train-frequency-ablations-v2 = "python scripts/run_frequency_ablations.py --plan configs/experiments/frequency_ablation_plan_v2.yaml --stage all"
```

- [ ] **Step 4: Run config tests and V2 dry-run**

- [ ] **Step 5: Commit**

```powershell
git add configs/experiments/frequency_ablation_plan_v2.yaml pixi.toml tests/test_frequency_ablation_runner.py
git commit -m "config: add frozen mean ablation v2"
```

### Task 7: End-to-end verification and publication

**Files:**
- Modify only files required by verified failures.

- [ ] **Step 1: Run the complete test suite**

```powershell
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run fake-data M0 pretraining**

```powershell
python scripts/pretrain_conditional_mean.py --config configs/experiments/slmf_png_residual_frequency.yaml --output-dir checkpoints/freq_mean_v2_smoke --epochs 1 --seed 42 --override data.use_fake_data=true --override data.image_size=32
```

Expected: `mean_best.pt` and finite validation loss.

- [ ] **Step 3: Run frozen-mean F4 train/evaluate smoke**

Use the smoke M0 checkpoint, one F4 epoch, two DDIM steps, and EMA standalone
evaluation. Verify finite image, lesion, false-hotspot, and stripe metrics.

- [ ] **Step 4: Run V1 and V2 dry-runs plus diff checks**

Verify V1 remains backward compatible, V2 lists M0 before six screens, and
`git diff --check` is clean.

- [ ] **Step 5: Push commits and update Draft PR #2**

```powershell
git push My_Diffusion agent/residual-frequency-bbdm
```

Update the PR body with the V1 evidence, V2 fix, test count, and the new cloud
command:

```powershell
pixi run python scripts/run_frequency_ablations.py --plan configs/experiments/frequency_ablation_plan_v2.yaml --stage all
```
