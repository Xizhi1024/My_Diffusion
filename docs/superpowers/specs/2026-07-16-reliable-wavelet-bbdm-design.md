# Reliable Wavelet BBDM Design

**Date:** 2026-07-16
**Status:** Approved direction; implementation pending
**Scope:** PNG CT/PET/lesion-mask training now, while retaining optional organ and physical-SUV interfaces

## Goal

Replace the artifact-prone V2 residual-frequency path with two independently switchable contributions:

1. a complete Haar-wavelet U-Net whose three encoder and decoder scale transitions preserve all four subbands; and
2. a reliability-controlled four-level residual-frequency injector that never rewrites the noisy diffusion state.

Add lesion-boundary and CT-aligned anatomy-edge supervision and evaluation so improvements are measured where the scientific claim is made. Preserve the existing conditional-mean residual bridge and every existing organ/SUV route.

## Evidence from V2

V2 established that the frozen conditional low-frequency mean plus zero-endpoint residual bridge is useful. It did not establish a general benefit from the existing frequency module:

- F1/F2 failed the stripe gate.
- F4 only traded slightly better SSIM, lesion peak error, and failure count for worse MAE, lesion mean error, and centroid distance than R2.
- The legacy preconditioner divides high bands by small constants, bilinearly upsamples signed Haar details, applies one shared Gabor factor to all three detail directions, and can directly alter the noisy residual state.

The new design therefore retains R2 and the legacy F4 path as controls rather than treating V2 frequency enhancement as established.

## Data and claim boundary

Current guaranteed inputs are PNG CT, normalized PET, and lesion masks. Consequently:

- lesion-boundary metrics are valid now;
- CT-aligned anatomy-edge metrics are valid proxy metrics now;
- organ-specific metrics remain unavailable until non-empty organ masks exist;
- physical SUV claims remain unavailable until calibrated SUV tensors and metadata exist.

No lesion mask is consumed by the denoiser or sampler. Masks are used only by training losses and evaluation. Optional organ masks may strengthen the same boundary loss when present, but all-zero or absent masks are treated as unavailable.

## Architecture

### 1. Complete Haar-wavelet U-Net

`WaveletBBDMUNet` keeps the public `BBDMUNet.forward` contract and the same four skip widths `[64, 128, 256, 256]`.

Every encoder transition performs:

```text
h_l
  -> orthonormal Haar DWT
  -> concat [LL, LH, HL, HH]
  -> learned cross-subband projection
  -> h_(l+1)
```

Every decoder transition performs:

```text
h_(l+1)
  -> learned expansion to [LL, LH, HL, HH]
  -> orthonormal Haar IWT
  -> h_l
```

There are three DWT transitions and three IWT transitions. Bilinear interpolation is not used for backbone scale changes. Encoder skips, cross-attention, time/meta conditioning, heteroscedastic output, and external skip injections retain their existing semantics.

The standard `BBDMUNet` remains the default. With `modules.wavelet_unet.enabled=false` or the field absent, construction and checkpoint keys are unchanged.

### 2. Reliable four-level residual-frequency injector

The legacy class remains available only for the V2 F4 control. The new `ReliableResidualFrequencyInjector` builds decoder-ordered bands without bilinear interpolation of signed detail coefficients:

```text
L3 (H/8): LL3
L2 (H/4): [LH2, HL2, HH2]
L1 (H/2): [LH1, HL1, HH1]
L0 (H):   three independent IWT reconstructions of LH1, HL1, HH1
```

Each band is robustly bounded as `tanh(coefficient / scale)` before projection. This replaces unbounded high-band amplification.

For every level, reliability is the product of:

1. **analytic bridge reliability** — a sigmoid of band log-SNR derived from Brownian bridge progress, schedule sigma, and configured residual-band scale;
2. **spatial evidence** — a learned bounded gate from residual-band magnitude, multiscale CT gradient magnitude, optional Gabor concentration, normalized timestep, and log-SNR;
3. **optional directional agreement** — a learned `orientations -> 3 subbands` factor bounded to `[1-strength, 1+strength]`, so LH/HL/HH are not scaled identically.

Every output projection ends in a zero-initialized convolution, making the enabled injector an exact initial no-op. The module only returns skip injections; it has no method that changes the noisy residual supplied to the denoiser.

### 3. Injection compatibility boundary

The existing `ZeroConvAdapter` remains responsible for CT, organ, legacy Gabor-adapter, and hotspot conditioning. The new frequency branch is computed separately and added after the adapter:

```text
decoder_skip = encoder_skip + existing_adapter_injection + reliable_frequency_injection
```

Reliability gates never multiply the existing adapter output. Therefore:

- optional organ features remain injectable at L1-L3;
- CT/hotspot adapter routes remain unchanged;
- metadata FiLM and semantic cross-attention remain unchanged;
- `roi_suv`, `organ_consistency`, and other loss interfaces remain constructible;
- a disabled reliable injector cannot suppress an enabled organ/SUV route.

### 4. Boundary-frequency supervision

`BoundaryFrequencyLoss` operates on reconstructed PET, not on the noisy bridge state.

It contains three independently reported terms:

- **lesion boundary:** Charbonnier difference between prediction and target gradient magnitude inside a morphological band around the lesion mask;
- **anatomy consensus boundary:** the same error on a soft training-only weight formed from CT edge strength and target-PET edge strength;
- **optional organ boundary:** the same error on boundaries extracted from non-empty organ masks.

A level-1 Haar-detail error is also weighted by the downsampled union of the available boundary maps. This explicitly tests the user's edge-frequency hypothesis while avoiding the assumption that every CT edge must create a PET edge.

The loss is timestep gated and off by default in legacy configurations.

### 5. Boundary evaluation

The evaluator reports normalized-space metrics:

- `lesion_boundary_intensity_mae_norm`;
- `lesion_boundary_gradient_mae_norm`;
- `anatomy_edge_gradient_mae_norm` on high-gradient CT locations;
- `organ_boundary_gradient_mae_norm` only when a non-empty organ mask exists.

The first three work with the current PNG dataset. The organ metric is omitted from aggregates when unavailable rather than replaced with zero.

## Configuration

```yaml
modules:
  wavelet_unet:
    enabled: true
    mix_kernel_size: 3

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

Invalid modes, non-positive scales, invalid reliability temperatures/floors, and directional gating without an enabled Gabor prior fail at construction.

## V3 ablation

All variants reuse the same frozen V2 conditional-mean checkpoint and initialization seed.

| ID | Wavelet U-Net | Reliable injector | Direction gate | Boundary loss | Purpose |
|---|---:|---:|---:|---:|---|
| R2 | off | off | off | off | validated residual-bridge reference |
| LF4 | off | legacy F4 | legacy | legacy losses | previous frequency control |
| W1 | on | off | off | off | wavelet backbone contribution |
| I1 | off | on | off | off | reliable injection contribution |
| WI | on | on | off | off | two core modules together |
| WI-G | on | on | on | off | directional refinement contribution |
| WI-L | on | on | off | on | boundary supervision contribution |
| WI-F | on | on | on | on | complete proposed method |

Screening runs every variant for 50 epochs without early stopping and evaluates the same 16-sample stratified validation subset. Artifact gates remain mandatory. Edge metrics enter ranking after the artifact gates.

At most two non-reference variants are retrained from scratch for exactly 300 epochs. Promotion uses `eval_interval=20` and `early_stopping=false`; the final report evaluates each run's EMA `best_combined` checkpoint on 64 validation samples and records the checkpoint epoch.

## Acceptance criteria

- Legacy configs instantiate the original `BBDMUNet` and legacy preconditioner unchanged.
- Wavelet U-Net uses three DWT and three IWT scale transitions and accepts existing skip injections.
- Reliable injections have exact decoder shapes, are finite/bounded, and start at zero.
- Analytic reliability is lower at low SNR than high SNR for the same band.
- Directional gating can produce distinct LH/HL/HH factors while remaining bounded.
- Reliable mode never modifies the noisy residual tensor.
- Organ adapter routing still affects skips when wavelet and reliable modules are enabled.
- Boundary loss is finite with lesion-only PNG batches and uses optional organ masks only when non-empty.
- Evaluation emits lesion/anatomy edge metrics and omits unavailable organ aggregates.
- V3 promotion commands disable early stopping and pin fixed seed, frozen mean, evaluation subset, and EMA weights.

## Scientific interpretation

The complete model is considered effective only if it passes artifact gates and improves lesion/boundary metrics without materially degrading MAE/SSIM. CT-edge results are described as anatomy-aligned proxy results, not organ-specific results. Organ and physical-SUV claims are deferred until those inputs become available.
