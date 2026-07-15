# Residual Frequency BBDM Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task by task.

**Goal:** Add an opt-in conditional-mean residual Brownian bridge, Haar residual preconditioning, phase-stable complex Gabor regulation, and a reproducible screen/promote ablation runner without changing the existing PNG baseline when the new modules are disabled.

**Architecture:** A low-pass PET predictor supplies the terminal mean of a residual BBDM. The denoiser predicts the PET residual, while a two-level Haar transform creates zero-initialized, bridge-aware skip injections. A Cartesian quadrature Gabor bank supplies only bounded orientation gates and final-image consistency statistics. A separate experiment runner screens R0/R2/F1-F4 under fixed evaluation conditions and promotes only artifact-safe candidates.

**Tech Stack:** Python 3.10, PyTorch, PyYAML, pytest, existing SLMF-BBDM training/evaluation stack, Git/GitHub CLI.

---

### Task 1: Lock down Haar and conditional-mean behavior

**Files:**
- Create: `tests/test_residual_frequency.py`
- Create: `src/model/frequency/__init__.py`
- Create: `src/model/frequency/haar.py`
- Create: `src/model/mean_predictor.py`

1. Add failing tests for two-level orthonormal Haar round-trip, low-pass-only reconstruction, output shapes, and mean-predictor gradient isolation.
2. Run the targeted test file and verify imports/tests fail for the missing implementation.
3. Implement exact Haar forward/inverse helpers and `LowFrequencyPETPredictor`.
4. Re-run the targeted tests and keep the implementation minimal until they pass.

### Task 2: Rebuild Gabor as a phase-stable Cartesian descriptor

**Files:**
- Modify: `tests/test_residual_frequency.py`
- Modify: `src/model/priors/gabor.py`

1. Add failing tests for `scales * orientations` filters, descriptor shapes, quadrature phase stability, bounded trainable deviations, and finite gradients.
2. Run the focused tests and confirm the legacy implementation fails the new contract.
3. Replace zipped real-only filters with Cartesian cosine/sine pairs, per-filter RMS normalization, orientation pooling, energy, and anisotropy outputs while retaining `gabor_feat` compatibility.
4. Re-run focused and existing prior tests.

### Task 3: Add zero-initialized residual wavelet preconditioning

**Files:**
- Modify: `tests/test_residual_frequency.py`
- Create: `src/model/frequency/residual_preconditioner.py`
- Modify: `src/model/frequency/__init__.py`

1. Add failing tests for deep-to-shallow output shapes, exact zero initialization, independent finite gates, and optional bounded Gabor modulation.
2. Implement fixed band scaling, analytic bridge-progress/log-SNR features, independent sigmoid gates, and zero-final-convolution projection heads.
3. Verify the module is an exact no-op at initialization and becomes trainable after an optimizer step.

### Task 4: Integrate the residual Brownian bridge without regressing baseline

**Files:**
- Modify: `tests/test_residual_frequency.py`
- Modify: `src/model/interfaces.py`
- Modify: `src/model/slmf_bbdm.py`
- Modify: `src/model/config_utils.py`

1. Add failing tests for invalid config combinations, residual endpoint algebra, PET reconstruction, residual training/sampling shapes, and legacy-disabled compatibility.
2. Extend `LossContext` with optional mean/residual tensors.
3. Construct and validate conditional mean, residual bridge, residual-frequency, and Gabor-route modules.
4. In training, diffuse `PET - stop_gradient(mean)` toward zero, predict the residual, and apply image/lesion terms to `mean + residual`.
5. In sampling, reverse the zero-endpoint residual bridge and return the reconstructed PET plus diagnostic tensors.
6. Merge new frequency skip injections elementwise with any existing adapter injections.
7. Run focused integration tests and the pre-existing model tests.

### Task 5: Add residual-frequency losses

**Files:**
- Modify: `tests/test_residual_frequency.py`
- Create: `src/model/loss_terms/residual_frequency.py`
- Modify: `src/model/slmf_bbdm.py`

1. Add failing tests for lesion-weighted residual wavelet loss and Gabor amplitude/orientation consistency with both empty and non-empty masks.
2. Implement robust band losses and target-matched orientation Jensen-Shannon divergence.
3. Register the loss names through the existing loss builder and expose component logs.
4. Run focused loss and model-forward tests.

### Task 6: Make artifact-aware evaluation measurable

**Files:**
- Modify: `tests/test_evaluate.py`
- Modify: `scripts/evaluate.py`

1. Add a failing evaluator test requiring prediction stripe score, target stripe score, and stripe excess in the saved summary.
2. Reuse the trainer stripe metric and add the three per-sample fields.
3. Verify existing JSON consumers remain compatible because the output schema is only extended.

### Task 7: Build the block-screening and automatic-promotion runner

**Files:**
- Create: `tests/test_frequency_ablation_runner.py`
- Create: `scripts/run_frequency_ablations.py`
- Create: `configs/experiments/slmf_png_residual_frequency.yaml`
- Create: `configs/experiments/frequency_ablation_plan.yaml`
- Modify: `configs/experiments/ablations.yaml`
- Modify: `pixi.toml`

1. Add failing unit tests for metric extraction, hard artifact gates, lesion-heavy composite scoring, deterministic top-K promotion, command construction, and dry-run manifests.
2. Implement sequential screen/evaluate/rank/promote execution using the current Python interpreter and explicit resolved configs.
3. Define R0, R2, F1 (Haar skip), F2 (Gabor-gated level-1 residual details), F3 (Haar + Gabor), and F4 (F3 + losses) presets with clean promoted retraining names.
4. Add the `train-frequency-ablations` project task.
5. Run runner unit tests and a dry-run that materializes the planned commands without starting full training.

### Task 8: End-to-end verification and documentation

**Files:**
- Modify as required by verified failures only.

1. Run all tests in the isolated worktree.
2. Run a fake-data one-step train smoke for the recommended F4 configuration.
3. Run a fake-data evaluation smoke and validate finite residual/frequency/artifact metrics.
4. Review `git diff --check`, configuration compatibility, CLI help, and the exact cloud synchronization file list.
5. Self-review the complete diff for baseline regressions, leakage, metric gaming, and invalid promotion behavior.

### Task 9: Publish the version

1. Stage only files owned by this feature; do not include the root worktree's user edits to `src/data/dataset.py` or `src/data/png_cache.py`.
2. Commit the verified implementation on `agent/residual-frequency-bbdm`.
3. Push the branch to the configured GitHub remote.
4. Create a Draft PR containing design rationale, ablation matrix, validation evidence, and the cloud execution command.
5. Report the PR URL, commit, copied-file list, and the exact cloud command; clearly separate local smoke execution from the remaining full RTX5080 ablation run.
