# Conditional-Mean Residual Frequency BBDM Design

**Date:** 2026-07-15

**Status:** Approved
**Scope:** PNG-only CT-to-PET training; no DICOM, physical SUV, or organ masks

## Goal

Replace unsafe raw CT Gabor injection with a PET-domain conditional-mean Brownian bridge whose residual score is explicitly preconditioned by wavelet bands. Preserve Gabor as a bounded, phase-stable local-direction gate and consistency loss. Provide reproducible block screening and automatic promotion for cloud training.

## Research position

The method does not claim that frequency decomposition makes diffusion work or reduces tensor dimensionality. Its claim is narrower:

> CT-to-PET translation is reparameterized as a conditional low-frequency PET estimate plus a stochastic residual. A bridge-progress-aware multiband preconditioner improves estimation of the residual score, while complex Gabor statistics weakly regulate local directional artifacts.

Residual diffusion and residual diffusion bridges already exist. The contribution tested here is the task-specific combination of:

1. a low-frequency PET-domain endpoint for CT-to-PET Brownian bridging;
2. lesion-residual wavelet standardization and level-matched zero injection;
3. complex Gabor anisotropy as a weak gate, never as raw PET texture;
4. lesion-weighted band and direction losses with explicit artifact gates.

## Non-goals

- Do not use DICOM metadata, SUV calibration, TotalSegmentator, or organ priors.
- Do not inject raw cosine Gabor responses into the decoder.
- Do not replace BBDM with a standard residual DDPM.
- Do not claim that CT uniquely determines metabolic lesions.
- Do not enable EqualSNR or scale-adaptive bridge noise in this change.

## Architecture

### 1. Conditional low-frequency PET endpoint

A small CNN predicts only the level-2 Haar low-pass coefficient of PET:

```text
CT [B,1,H,W]
  -> shallow strided CNN
  -> predicted LL2 [B,1,H/4,W/4]
  -> inverse Haar with all detail bands fixed to zero
  -> mean_pet mu(CT) [B,1,H,W]
```

The predictor cannot emit level-1 or level-2 detail coefficients. Its sole supervised objective is a robust L1/Charbonnier error against the target PET LL2 coefficient. The mean passed to the bridge is detached, so lesion and diffusion losses cannot turn this branch into a full-image shortcut.

### 2. Residual Brownian bridge

For target PET `y` and conditional mean `mu(c)`:

```text
r0 = y - stop_gradient(mu(c))
rt = (1 - mt) * r0 + sigma_t * epsilon
y_hat = mu(c) + r0_hat
```

This is a Brownian bridge from residual `r0` to zero. In PET space it is the equivalent bridge:

```text
xt = mu(c) + (1 - mt) * (y - mu(c)) + sigma_t * epsilon
```

so the terminal endpoint is the predicted coarse PET rather than raw CT. CT remains an explicit condition through the existing CT encoder and model-input concatenation.

Training predicts `r0`. The base diffusion loss is applied to `r0_hat` versus `r0`; all lesion and image losses are applied to `mu + r0_hat` versus PET. Self-conditioning, when enabled, uses the residual prediction.

Sampling computes `mu` once, initializes the residual bridge at its zero endpoint, runs the existing deterministic BBDM reverse interpolation with source zero, and returns `mu + final_r0_hat`.

When the feature is disabled, the existing full-PET CT-to-PET BBDM path is unchanged.

### 3. Residual wavelet preconditioner

The current noisy residual state is decomposed with a two-level orthonormal Haar transform:

```text
LL2                    -> deep L3 skip
LH2, HL2, HH2          -> middle L2 skip
LH1, HL1, HH1          -> L1 skip and upsampled L0 skip
```

Each band is divided by a configured robust training-set scale before projection. Independent learned sigmoid gates receive:

- normalized timestep;
- Brownian bridge progress `1 - mt`;
- analytic per-band log-SNR derived from `mt`, `sigma_t`, and band scale.

Each injection head ends in a zero-initialized 1x1 convolution. Therefore enabling the module starts as an exact no-op. Gates are independent rather than softmax-normalized because several bands may be useful at the same reverse step.

For the Gabor-only F2 ablation, wavelet skip injection is disabled. The same
identity-initialized bounded gate modulates only the level-1 Haar detail
coefficients of the noisy residual before they are reconstructed and passed to
the denoiser. LL1 is unchanged and no raw Gabor carrier response is injected.
This makes F2 identifiable from F1 and F3.

### 4. Complex Gabor descriptor

The Gabor bank is rebuilt as a Cartesian `4 scales x 8 orientations` bank. Every filter uses a cosine/sine quadrature pair. Its output is phase-stable amplitude:

```text
A_s_theta = sqrt(response_cos^2 + response_sin^2 + epsilon)
```

Frequency, orientation, envelope scale, and aspect ratio are bounded around their initialization. Per-filter RMS normalization replaces per-image global-maximum normalization.

The prior exposes:

- `gabor_feat`: 32 normalized amplitude maps for legacy diagnostics;
- `gabor_orientation`: 8 scale-pooled orientation maps;
- `gabor_energy`: one phase-stable energy map;
- `gabor_anisotropy`: one bounded directional-concentration map.

Only `gabor_orientation` may reach the new frequency preconditioner. A zero-initialized gate produces a bounded multiplicative factor around one for the high-frequency residual injection. Raw responses are never added to decoder features.

### 5. Losses

`ResidualWaveletLoss` compares predicted and target residual coefficients with lesion-projected weighting. Low, middle, and high bands have independent configuration weights.

`GaborConsistencyLoss` uses the same fixed quadrature descriptors on final predicted PET and target PET. It combines:

- lesion-weighted amplitude Charbonnier error;
- Jensen-Shannon divergence between target and prediction orientation-energy distributions.

Matching the target direction distribution avoids the incorrect assumption that every PET image must be isotropic. The previous focal Fourier loss stays available but is off in the recommended configuration.

## Configuration surface

```yaml
modules:
  conditional_mean:
    enabled: true
    levels: 2
    base_channels: 32
    loss_weight: 1.0
    detach_bridge: true
  residual_bridge:
    enabled: true
  residual_frequency:
    enabled: true
    inject_wavelet: true
    use_gabor_gate: true
    band_scales: [1.0, 0.5, 0.25]
    gate_strength: 0.1
  gabor:
    enabled: true
    scales: 4
    orientations: 8
    inject_adapter: false
    use_for_loss: true

losses:
  residual_wavelet:
    enabled: true
    weight: 0.05
  gabor_consistency:
    enabled: true
    weight: 0.02
    orientation_weight: 0.1
```

Invalid combinations fail at model construction:

- residual bridge requires conditional mean and BBDM schedule;
- residual-frequency preconditioning requires residual bridge;
- Gabor gating requires an enabled Gabor prior;
- Gabor consistency requires `gabor.use_for_loss=true`.

## Ablation and automatic promotion

The cloud runner executes fixed-seed screen runs, evaluates EMA `best_combined` on the same lesion-area-stratified validation subset, applies artifact gates, and retrains promoted configurations from scratch.

```text
R0  current full-PET BBDM reference
R2  conditional-mean residual BBDM
F1  R2 + wavelet preconditioner
F2  R2 + complex Gabor weak gate on level-1 residual details (no skip injection)
F3  R2 + wavelet + Gabor
F4  F3 + residual-wavelet and Gabor consistency losses
```

Promotion requires finite metrics and hard limits relative to R0 for failure rate, false-hotspot density, stripe excess, and SSIM. Among passing runs, a lesion-heavy composite score selects the configured top K. Promotion is a clean full retraining, not continuation from a short-run scheduler.

Outputs include resolved configs, checkpoints, per-run evaluation JSON, CSV/JSON leaderboards, and the promotion decision.

## Validation

- Exact Haar round-trip and low-pass-only reconstruction tests.
- Complex Gabor shape, phase invariance, bounded parameters, and finite-gradient tests.
- Zero-init residual-frequency injection tests.
- Residual bridge endpoint, target reconstruction, training, sampling, and legacy compatibility tests.
- Loss tests for empty and non-empty lesion masks.
- Ablation dry-run, score, hard-gate, and promotion tests.
- Full existing test suite and fake-data training/evaluation smoke tests.

## Risks and controls

- **Mean branch absorbs lesions:** it predicts only LL2 and is detached from diffusion/lesion losses.
- **Gabor recreates stripes:** only quadrature amplitude enters a bounded multiplicative gate initialized to identity.
- **Band amplification destabilizes training:** fixed scales, independent sigmoid gates, zero-init projections, gradient clipping.
- **Short-run metric noise:** fixed subset, fixed seed, EMA weights, hard artifact gates, clean retraining of promoted runs.
- **Residual bridge becomes DDPM:** construction requires the BBDM schedule and reverse sampling uses the zero residual endpoint.
