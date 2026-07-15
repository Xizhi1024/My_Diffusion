# Boundary-Reliable Frequency BBDM Design

**Date:** 2026-07-16  
**Status:** Approved, revised after architecture review  
**Scope:** PNG CT/PET/lesion-mask training; optional organ and physical-SUV interfaces remain compatible

## Goal

Replace the artifact-prone V2 frequency path with a conservative decoder-side residual branch:

```text
noisy bridge state + CT + timestep + Gabor energy
    -> two-level Haar analysis
    -> noise/cross-modal/content reliability
    -> LH/HL/HH subband gates
    -> zero-initialized skip residuals
```

The branch must not modify the diffusion state, replace a skip, gate an existing adapter, or consume a ground-truth lesion mask. It is intended to improve lesion and anatomy boundaries without recreating periodic Gabor textures.

## Evidence and constraints

V2 supports the frozen conditional mean plus zero-endpoint residual bridge. It does not establish a general benefit from the old frequency module: F1/F2 failed the stripe gate and F4 traded small SSIM/peak-error gains for worse MAE, lesion-mean error, and centroid distance.

The likely structural failure is repeated propagation of noisy high-frequency content through decoder skips. The first implementation is therefore deliberately smaller than a complete wavelet backbone or a global time-layer-lesion router.

Current guaranteed inputs are PNG CT, normalized PET, and lesion masks. Lesion-boundary and CT-aligned proxy metrics are valid now. Organ-specific and physical-SUV claims remain unavailable until non-empty organ masks and calibrated SUV data are guaranteed.

## Architecture

### 1. Two-level Haar source mapping

For U-Net levels `L0=H`, `L1=H/2`, `L2=H/4`, `L3=H/8`:

| Decoder level | Frequency source | Rule |
|---|---|---|
| L3 | none | exact zero; do not invent deep high frequency |
| L2 | `LH2, HL2, HH2` | use at native resolution |
| L1 | `LH1, HL1, HH1` | use at native resolution |
| L0 | `IDWT(0, gated LH1, gated HL1, gated HH1)` | reconstruct pure full-resolution high-frequency increment |

There is no third Haar level and no bilinear interpolation of signed wavelet coefficients. Haar is described as **subband-aligned inverse reconstruction**, not as strictly phase-consistent: critically sampled Haar remains shift-variant.

The previously implemented `WaveletBBDMUNet` remains disabled by default and outside this first ablation. It is retained only as a future standalone experiment.

### 2. Explicit subband reliability

For each native high-frequency level and subband `b in {LH, HL, HH}`:

```text
g_l,b = cap * g_noise_l * g_cross_l,b * g_content_l,b * g_direction_l,b
```

- `g_noise` is an analytic sigmoid of bridge log-SNR and releases high frequency only as the current state becomes reliable.
- `g_cross` compares local normalized high-frequency energy between CT and the current state. It is soft and never treats every CT edge as a PET edge.
- `g_content` is predicted from global band statistics and timestep by a two-layer MLP with few channels.
- `g_direction` is optional and derives from phase-insensitive Gabor quadrature energy. It changes reliability only; raw signed Gabor responses are never injected.

The gates share an overall reliability envelope while retaining bounded LH/HL/HH offsets. This avoids three unrelated spatial masks making contradictory decisions. A weak total-variation penalty is exposed from the spatial gates to discourage checkerboard/noisy gate maps.

Configuration switches make the ablation identifiable:

- `use_noise_release`
- `use_ct_reliability`
- `use_subband_gates`
- `use_directional_reliability`

When subband gates are disabled, all three bands share the same reliability scalar. Directional reliability requires an enabled Gabor prior.

### 3. Safe fusion and compatibility

The original adapter and the new frequency branch are built independently:

```text
S'_l = S_l + Adapter_l(CT, organ, hotspot, legacy routes) + Z_l(F_l)
```

Each `Z_l` ends with a zero-initialized `1x1` convolution. Initial behavior is therefore strictly equivalent to the same R2 model:

1. original skips are unchanged;
2. no new normalization is applied to them;
3. concat order is unchanged;
4. gates never multiply skips or adapter outputs;
5. only the new additive residual starts at zero.

Consequently organ injection, metadata FiLM, semantic conditioning, `roi_suv`, and organ-consistency loss interfaces remain usable. A degraded frequency branch can be disabled without changing bridge forward/reverse equations or sampling state.

### 4. Gabor role

The existing Gabor prior already computes quadrature amplitude:

```text
E_theta(x) = sqrt((G_cos*x)^2 + (G_sin*x)^2 + eps)
```

The new branch maps these non-negative orientation energies to LH/HL/HH reliability offsets. Gabor content is not projected into PET features. Legacy direct adapter/noise routes remain available only for reproducing old controls and are explicitly disabled in every new reliable variant.

### 5. Boundary and gate supervision

`BoundaryFrequencyLoss` operates only on predicted `x0` and target PET.

- A lesion boundary ring is formed by dilation minus erosion of the training mask.
- Gradient-magnitude error is measured inside that ring.
- An anatomy proxy uses soft consensus between CT edge strength and target-PET edge strength, avoiding the assumption that every bone edge implies a metabolic edge.
- Optional organ boundaries are used only when a non-empty organ mask is present.
- A level-1 Haar-detail error is weighted by the available boundary union.
- The whole boundary loss is released in middle/late denoising by bridge reliability.
- A small `gate_tv` term regularizes only the newly generated spatial gates.

Lesion/organ masks never enter the denoiser or sampler.

For directional artifacts, evaluation compares prediction and target Gabor orientation-energy spectra rather than imposing isotropy on anatomically directional images.

## Configuration sketch

```yaml
modules:
  wavelet_unet:
    enabled: false
  residual_frequency:
    enabled: true
    mode: boundary_reliable
    output_channels: [256, 256, 128, 64]
    band_scales: [0.5, 0.25]
    use_noise_release: true
    use_ct_reliability: true
    use_subband_gates: true
    use_directional_reliability: true
    gate_max: 0.25
    cross_temperature: 1.0
    content_hidden_channels: 16

losses:
  boundary_frequency:
    enabled: true
    weight: 0.05
    lesion_weight: 1.0
    anatomy_weight: 0.5
    organ_weight: 0.5
    wavelet_weight: 0.25
    boundary_radius: 2
  frequency_gate_tv:
    enabled: true
    weight: 0.001
```

## V3 ablation

`R2-A` as originally suggested is not a valid comparison in this repository because R2 has frequency disabled and never calls `modulate_residual()`. The state-modulation question is isolated with paired legacy controls instead.

| ID | Change from prior row | Purpose |
|---|---|---|
| R2 | frozen mean + residual bridge, frequency off | validated reference |
| F4 | legacy F4 unchanged | old frequency control |
| F4-NM | F4 with `modulate_residual()` disabled | isolate direct state modification |
| BR-B | two-level Haar + fixed noise release | safe high-frequency injection |
| BR-C | add CT edge soft reliability | cross-modal boundary confidence |
| BR-D | add independent bounded LH/HL/HH offsets | subband selectivity |
| BR-E | add Gabor directional-energy reliability | directional reliability, no texture injection |
| BR-F | add boundary-ring and gate-TV losses | complete proposed method |

All variants reuse the same frozen V2 mean and paired U-Net initialization. Screening is 50 epochs. At most two eligible non-reference variants are retrained from scratch for exactly 300 epochs with early stopping disabled and EMA evaluation every 20 epochs.

## Acceptance criteria

- New band construction uses exactly two Haar levels.
- L3 injection is exactly zero.
- L2/L1 use native details and L0 uses `IDWT(0, details)`.
- Gates are finite, non-negative, capped, and distinguish LH/HL/HH only when enabled.
- Noise reliability increases with bridge SNR for identical content.
- CT reliability compares energy, not signed cross-modal coefficients.
- Gabor affects gates only; injected content is unchanged when only Gabor energy changes and gates are held fixed.
- New residual heads are zero-initialized and receive gradients.
- Boundary-reliable mode has no noisy-state modulation method or call.
- Existing adapter output is added independently and remains effective.
- Boundary loss uses predicted `x0`; masks are loss/evaluation-only.
- Empty organ masks do not create fake zero metrics.
- Legacy R2/F4 construction and checkpoints remain valid.

## Scientific claim boundary

The contribution is framed as a CT-to-PET **subband-level boundary-frequency injection mechanism jointly controlled by bridge-noise reliability, CT/PET edge-energy agreement, and Gabor directional reliability**, without modifying the diffusion state or main U-Net. It is not framed as the first use of timestep-adaptive frequency gating or generic wavelet diffusion.
