# Boundary-Reliable Ablation V3 Implementation Plan

> Execute inline with the `executing-plans` and `test-driven-development` skills.

**Goal:** Provide paired, fixed-budget experiments that identify state modulation, noise release, CT reliability, directional subband gating, Gabor reliability, and boundary supervision.

### Task 1: Add boundary-aware ranking

**Files:** `scripts/run_frequency_ablations.py`, `tests/test_frequency_ablation_runner.py`

- [ ] RED: lower lesion/anatomy boundary error and lower prediction-target directional spectrum error improve ranking after artifact gates.
- [ ] Add bounded score terms; do not require unavailable organ metrics.
- [ ] Run runner tests to GREEN.

### Task 2: Add base config and eight distinct presets

**Files:**
- Create: `configs/experiments/slmf_png_boundary_reliable.yaml`
- Modify: `configs/experiments/ablations.yaml`
- Modify: `tests/test_frequency_ablation_runner.py`

- [ ] RED: presets `br_r2`, `br_f2`, `br_f4`, `br_b`, `br_c`, `br_d`, `br_e`, `br_f` resolve and construct.
- [ ] Keep `modules.wavelet_unet.enabled=false` in every preset.
- [ ] `br_f2` reproduces the legacy state-modulation-only F2 control; `br_f4` reproduces legacy F4 without falsely attributing state modulation to it.
- [ ] `br_b` enables two-level Haar plus fixed noise release.
- [ ] `br_c` adds CT soft reliability.
- [ ] `br_d` adds independent bounded LH/HL/HH offsets.
- [ ] `br_e` adds Gabor directional-energy reliability while all direct Gabor adapter/noise routes stay off.
- [ ] `br_f` adds boundary and gate-TV losses.
- [ ] Disable legacy residual-wavelet/Gabor losses in all new-mode variants.
- [ ] Run real 32x32 fake-data forwards for all presets.

### Task 3: Add fixed-budget cloud plan

**Files:**
- Create: `configs/experiments/boundary_reliable_ablation_plan_v3.yaml`
- Modify: `scripts/run_frequency_ablations.py`
- Modify: `tests/test_frequency_ablation_runner.py`

- [ ] RED: plan reuses `checkpoints/freq_mean_pretrain_v2/mean_best.pt`, screens 50 epochs, promotes at most two, and trains promotion runs exactly 300 epochs with eval interval 20 and early stopping false.
- [ ] Set output to `results/boundary_reliable_ablations_v3` and reference to R2.
- [ ] Preserve V2 artifact hard gates and fixed evaluation subsets/seeds/EMA weights.
- [ ] Record checkpoint epoch in result records.
- [ ] Dry-run all commands and verify no V1/V2 output path is used.

### Task 4: Integrated verification and handoff

- [ ] Run full pytest suite.
- [ ] Run all eight fake-data forwards.
- [ ] Audit dry-run manifest.
- [ ] Run `git diff --check` and focused compatibility tests.
- [ ] Provide exact cloud sync file list and Pixi command.
