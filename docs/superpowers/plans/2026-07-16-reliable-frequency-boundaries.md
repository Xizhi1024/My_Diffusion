# Boundary-Reliable Frequency Injection Implementation Plan

> Execute inline with the `executing-plans` and `test-driven-development` skills. Each behavior starts with a failing test.

**Goal:** Implement a two-level Haar, decoder-only, subband-reliability branch without changing the diffusion state or existing clinical conditioning paths.

**Architecture:** `BoundaryReliableFrequencyInjector` analyzes only inference-available noisy state, CT, timestep, and optional phase-insensitive Gabor energy. It injects gated residual high frequency at L2/L1/L0, returns exact zero at L3, and exposes gate diagnostics/TV. Boundary supervision operates only on reconstructed PET.

### Task 1: Replace the uncommitted global router with two-level subband injection

**Files:**
- Delete: `src/model/frequency/time_scale_router.py`
- Create: `src/model/frequency/boundary_reliable.py`
- Replace: `tests/test_reliable_frequency.py`

- [ ] Write RED tests for exactly two DWT calls, L3 zero, native L2/L1 details, and L0 `IDWT(0, details)`.
- [ ] Test finite/capped `[B, 2, 3, H_l, W_l]` subband gates and increasing noise reliability with bridge SNR.
- [ ] Test signed CT coefficient flips do not change CT energy reliability.
- [ ] Test shared-band mode yields equal LH/HL/HH gates; subband mode can distinguish them.
- [ ] Test Gabor orientation energy changes directional gates but is never an injected content tensor.
- [ ] Test zero heads make all injections initially zero yet receive gradients.
- [ ] Implement `_ZeroProjection`, local energy normalization, explicit noise/cross/content/direction factors, and gate-TV diagnostics.
- [ ] Run `python -m pytest tests/test_reliable_frequency.py -q` to GREEN.

### Task 2: Integrate without touching adapter or bridge state

**Files:**
- Modify: `src/model/slmf_bbdm.py`
- Modify: `src/model/frequency/residual_preconditioner.py`
- Modify: `tests/test_residual_frequency.py`

- [ ] RED: `mode=boundary_reliable` constructs the new class and exposes no `modulate_residual` method.
- [ ] RED: adapter injections of two plus frequency injections of three produce five at every skip.
- [ ] RED: reliable forward logs L2/L1 LH/HL/HH mean gates and `gate_tv`; L3 injection remains zero.
- [ ] RED: legacy `state_modulation=false` suppresses only the old `modulate_residual` call, leaving old skip injections available.
- [ ] Add `state_modulation` to the legacy preconditioner/config, defaulting true for checkpoint/config compatibility.
- [ ] Route raw CT and optional `gabor_orientation` to the new branch. Do not pass mean PET, hotspot, lesion, or organ masks.
- [ ] Put differentiable `frequency_gate_tv` into the forward condition only for the loss stack; detach diagnostic logs.
- [ ] Run reliable/residual/smoke tests to GREEN.

### Task 3: Add predicted-x0 boundary and gate-TV losses

**Files:**
- Create: `src/model/loss_terms/boundary_frequency.py`
- Create: `src/model/loss_terms/frequency_gate_tv.py`
- Modify: `src/model/loss_terms/__init__.py`
- Modify: `src/model/slmf_bbdm.py`
- Create: `tests/test_boundary_frequency.py`

- [ ] RED: shifted lesion edges cost more than aligned edges.
- [ ] RED: loss uses `pred_x0`, not noisy state/model target.
- [ ] RED: missing/all-zero organ masks report unavailable and contribute no organ loss.
- [ ] RED: CT-only edges without target-PET consensus receive a low anatomy weight.
- [ ] RED: high-noise timesteps suppress boundary supervision relative to reliable timesteps.
- [ ] RED: `FrequencyGateTVLoss` consumes the differentiable gate-TV scalar and is zero when unavailable.
- [ ] Implement max-pool morphology, gradient magnitude, soft CT/target consensus, level-1 Haar boundary error, and analytic timestep release.
- [ ] Register both losses and run focused/integration tests to GREEN.

### Task 4: Add boundary/directional evaluation

**Files:**
- Modify: `scripts/evaluate.py`
- Modify: `tests/test_boundary_frequency.py`

- [ ] RED: aligned lesion boundary scores better than shifted prediction.
- [ ] RED: empty organs return NaN/are omitted.
- [ ] RED: directional spectrum difference is zero for identical images and larger for synthetic directional artifacts.
- [ ] Implement normalized lesion-boundary intensity/gradient MAE, CT-target consensus anatomy-edge MAE, optional organ-boundary MAE, and prediction-target Gabor orientation-energy difference.
- [ ] Integrate sample and aggregate reporting; run evaluator tests to GREEN.

### Task 5: Compatibility regression

- [ ] Verify R2 and legacy F4 still construct.
- [ ] Verify boundary-reliable mode leaves bridge `add_noise` and reverse update arguments unchanged.
- [ ] Verify organ adapter and ROI-SUV loss interfaces complete a real forward.
- [ ] Run full test suite.
