# BR-D Peak Preservation and Safe Gabor V4 Design

**Date:** 2026-07-16

**Status:** Approved

**Base:** PR #4, branch `agent/reliable-wavelet-ablation`

**Scope:** Normalized PNG CT/PET training with lesion masks; physical SUV is unavailable

## Objective

Retain BR-D as the structural backbone, recover lesion peak fidelity, relax
shallow CT filtering where PET-only uptake has no matching CT edge, and retain
Gabor as an inference-time reliability module without recreating directional
stripe artifacts.

The target model is:

```text
BR-D
  + normalized Top-Q lesion peak supervision
  + scale-specific CT reliability floors
  + shallow shared Gabor agreement reliability
```

F4 remains the peak-preservation comparator. It is not the new structural
backbone.

## V3 Evidence

The V3 promotion evaluated 64 validation images from 26 patients. Physical SUV
was unavailable in every result.

BR-D compared with F4:

| Metric | BR-D | F4 | Interpretation |
|---|---:|---:|---|
| SSIM | 0.9537 | 0.9467 | BR-D better |
| PSNR | 28.78 | 26.30 | BR-D better |
| Lesion boundary intensity error | 0.2885 | 0.5781 | BR-D better |
| Lesion boundary gradient error | 0.0947 | 0.1275 | BR-D better |
| Anatomy edge gradient error | 0.0401 | 0.0413 | BR-D slightly better |
| Lesion centroid distance | 3.11 px | 4.66 px | BR-D better |
| Lesion mean error | 0.1149 | 0.2402 | BR-D better |
| False-hotspot density | 0.000107 | 0.000309 | BR-D better |
| Lesion peak error | 0.0956 | 0.0406 | F4 better |
| Failure rate | 9.38% | 3.13% | F4 better |

Patient-paired results also favor BR-D for structure: it wins on SSIM in 23 of
26 patients, lesion boundary gradient in 21 of 26, centroid distance in 19 of
26, and false-hotspot density in all 26. It wins lesion peak error in only 7 of
26 patients.

The V3 ablation identifies these effects:

- BR-D validates two-level Haar injection, analytic noise release, CT local
  reliability, global content reliability, and independent LH/HL/HH gates.
- BR-E adds current per-subband Gabor reliability. It improves boundary
  metrics but raises stripe excess from 0.0846 to 0.0975 and fails the gate.
- BR-F adds boundary and gate-TV losses. The current combination worsens peak
  fidelity and failure rate and remains disabled in V4.
- BR-C shows that CT reliability without content/subband calibration is
  unstable. It does not prove CT reliability is universally harmful.

The 16-image V3 screen underestimated BR-D's promoted peak error. V4 therefore
uses the same fixed 64-image validation subset for every screening decision.

## Confirmed Problems and Mechanistic Hypotheses

### Confirmed: the current lesion loss is not a peak loss

When a lesion mask is present, `TopKLesionLoss` supervises all masked pixels
with focal L1. It does not select the brightest pixels inside the mask.
`OutsidePeakRankingLoss` only enforces an inside-versus-outside ordering.
`ROISUVLoss` is disabled for PNG and would return zero without calibrated SUV
metadata. No current loss directly matches a normalized lesion Top-Q peak.

### Hypothesis: bounded details favor shape over amplitude

BR-D uses:

```text
H_bounded = tanh(H / band_scale)
```

with scales `[0.5, 0.25]`. This is a plausible reason that weak and strong
detail responses become less distinguishable. It affects only the new skip
residual, not the original U-Net amplitude path, so V4 tests the hypothesis
instead of treating it as a confirmed root cause. The first V4 screen does not
change `tanh` or `band_scales`.

### Hypothesis: shallow CT reliability suppresses PET-only peaks

Current cross-modal reliability is a soft multiplicative gate with no floor:

```text
r_ct = exp(-abs(E_residual - E_ct) / temperature)
```

It may approach zero where a metabolic PET peak has no CT high-frequency
counterpart. The global content gate can correct a band for a whole sample but
cannot locally reopen a PET-only lesion core. V4 tests scale-specific floors
without changing the original skip or diffusion state.

## Architecture

### 1. Preserve the BR-D backbone

The following remain unchanged:

- exactly two Haar levels;
- exact-zero L3 injection;
- native L2 and L1 detail tensors;
- L0 constructed by `IDWT(0, LH1, HL1, HH1)`;
- additive zero-convolution skip residuals;
- no denoiser-state or sampler-state modification;
- independent adapter, organ, metadata, and SUV interfaces;
- no inference-time lesion mask.

Boundary-frequency loss, gate-TV, complete Wavelet U-Net, legacy Gabor state
modulation, direct Gabor adapter injection, and Gabor consistency loss remain
disabled.

### 2. Normalized Top-Q lesion peak loss

For each non-empty lesion mask `M`, select a bounded number of the brightest
pixels independently from predicted `x0` and target `x0`:

```text
k = clamp(ceil(q * count(M)), min_k, max_k)
p_pred   = mean(topk(pred_x0[M], k))
p_target = mean(topk(target_x0[M], k))
L_peak   = SmoothL1(p_pred, p_target, beta)
```

The loss is computed per sample and averaged only over valid samples. Empty
masks are skipped. The timestep gate is applied per sample, not as one batch
mean. The mask is available only to the training loss and is not consumed by
the denoiser or sampler.

Initial configuration:

```yaml
losses:
  normalized_lesion_peak:
    enabled: true
    weight: 0.05
    topk_percent: 0.10
    min_k: 3
    max_k: 16
    beta: 0.02
    active_tau_max: 0.25
```

The name deliberately says `normalized`, not `SUVmax`, because PNG values do
not provide calibrated physical SUV.

### 3. Scale-specific CT reliability floors

The floor mapping must be monotonic and preserve `r_ct=1`:

```text
r_ct_floored = floor + (1 - floor) * r_ct
```

The minus-sign form is invalid because it reverses reliability and can produce
negative values.

Configuration is explicit per native Haar level:

```yaml
modules:
  residual_frequency:
    ct_reliability_floor_l2: 0.25
    ct_reliability_floor_l1: 0.50
```

Semantics:

- floor `0.0` preserves current BR-D behavior;
- floor `1.0` disables CT attenuation at that level;
- L0 inherits the L1 decision because it is reconstructed from gated L1
  details;
- floors affect only the new frequency residual, never the original skip.

The L2-only CT variant uses floor `0.0` at L2 and floor `1.0` at L1.

### 4. Safe inference-time Gabor agreement gate

The current BR-E gate assigns separate `[0.5, 1.0]` factors to LH, HL, and HH.
That can preserve one direction while suppressing the others. V4 retains the
Gabor descriptor but replaces per-band redistribution with one shared spatial
reliability multiplier.

The Gabor descriptor is computed from CT with its parameters detached for this
route. Existing orientation energy is mapped to three bands with the current
cosine-squared, sine-squared, and diagonal weights, then normalized across
bands to form `q_g`. A `3x3` average pool of each absolute residual Haar detail
is normalized across bands to form `q_h`. Their bounded agreement is the
Bhattacharyya coefficient:

```text
A = sum_b sqrt(q_g[b] * q_h[b])
```

The existing phase-insensitive Gabor anisotropy, resized to the Haar level,
provides confidence `C` in `[0, 1]`. The shared reliability is:

```text
r_g = clamp(1 - alpha * C * (1 - A), 1 - alpha, 1)
```

Initial constraints:

```yaml
modules:
  residual_frequency:
    gabor_agreement:
      enabled: true
      level_l2: false
      level_l1: true
      alpha: 0.10
      detach_descriptor: true
      shared_across_subbands: true
```

Properties:

- Gabor remains active during inference.
- It uses direction through `A` but never injects a signed or amplitude Gabor
  response.
- It can only suppress a directionally inconsistent new residual by at most
  10%; it cannot amplify injection above BR-D.
- One multiplier is shared by LH/HL/HH, so Gabor cannot create subband energy
  imbalance.
- It is applied only at L1; L0 inherits the resulting L1 details, and L2 is
  unchanged.
- Missing Gabor data, non-finite descriptors, or negligible anisotropy must
  return exact reliability `1`.
- Organ and SUV routes remain independent.

## Evaluation Diagnostics

V4 retains existing metrics and adds normalized PNG diagnostics:

- single-pixel lesion peak signed and absolute error;
- Top-Q peak signed and absolute error;
- underestimation and overestimation fractions;
- core Top-Q, boundary-ring Top-Q, and ring/core ratio;
- peak-to-boundary distance;
- lesion size;
- patient-level win/tie/loss and median paired difference.

Core is an erosion of one pixel. The boundary ring is dilation by two pixels
minus the core. If erosion empties a small lesion, core metrics fall back to
the original lesion mask and record the fallback count.

Reports must keep single-pixel max and Top-Q separate. Physical-SUV fields
remain unavailable and must not be inferred from normalized ratios.

## Two-Stage V4 Ablation

Every screen uses the same frozen conditional mean, core initialization seed,
training seed, 50-epoch budget, EMA weights, 20 sampling steps, validation
split, and fixed 64-image subset.

### Stage A: peak and CT mechanism

| ID | Peak loss | CT L2 | CT L1/L0 | Purpose |
|---|---|---|---|---|
| D0 | off | original | original | exact BR-D reproduction |
| D1 | on | original | original | isolate peak supervision |
| D3 | on | floor 0.25 | floor 0.50 | test bounded CT attenuation |
| D4 | on | original | disabled, floor 1.0 | test CT only at medium scale |

D0 is the reference. D1, D3, and D4 are eligible. If none passes all safety
gates, Stage B stops and no Gabor model is promoted.

### Stage B: retain Gabor safely

Stage B starts from the highest-ranked eligible Stage A configuration and
repeats fresh training with matched initialization:

| ID | Gabor route | Purpose |
|---|---|---|
| G0 | off | matched Stage A winner |
| G1 | current per-subband directional gate | mechanism control |
| G2 | shared shallow agreement gate | proposed safe Gabor module |

At most two gate-passing Stage B models are retrained from scratch for exactly
300 epochs with early stopping disabled. F4 is evaluated as the fixed legacy
peak comparator but is not a V4 promotion candidate.

## Selection and Safety Gates

Model selection continues to include peak error, centroid distance, failure
rate, SSIM, MAE, and stripe score. V4 adds Top-Q peak error to checkpoint and
promotion reporting. The single-pixel peak remains visible so Top-Q cannot
hide spikes.

After all hard gates pass, variants are ranked by the predeclared score:

```text
lesion = -(
    topq_peak_error
    + 0.5 * single_peak_error
    + 0.02 * centroid_distance
    + 0.5 * failure_rate
    + 0.2 * lesion_boundary_gradient_error
)
image = (
    ssim
    - mae
    - 0.2 * max(stripe_excess, 0)
    - 100 * false_hotspot_density
    - 0.1 * anatomy_edge_gradient_error
    - 0.1 * directional_spectrum_error
)
score = 0.7 * lesion + 0.3 * image
```

On the fixed 64-image validation set, a full candidate must satisfy:

```text
SSIM                              >= 0.950
lesion boundary intensity error  <= 0.320
lesion centroid distance         <= 3.50 px
false-hotspot density            <= 0.00015
single-pixel lesion peak error   <= 0.065
Top-Q lesion peak error          <= D0 Top-Q error
failure rate                     <= 0.0625
```

These are promotion safety gates, not final publication thresholds. Ranking is
performed only after every gate passes. Reports include 26-patient paired
win/tie/loss, median paired differences, and bootstrap 95% confidence
intervals. Confidence intervals use 10,000 patient-level bootstrap resamples,
percentile bounds, and seed 42. Metrics with fewer than two non-missing
patients report the count and no interval.

## Failure Handling

- Empty lesion masks produce zero peak loss and are excluded from peak
  denominators.
- Non-finite Top-Q values fail the candidate gate instead of being converted
  to zero.
- CT floors are validated in `[0, 1]`.
- Missing Gabor orientation or anisotropy returns exact Gabor reliability `1`.
- Gabor descriptor parameters receive no gradient from the agreement route.
- Stage B cannot start without one eligible Stage A candidate.
- Promotion training must produce an epoch-300 checkpoint. A best checkpoint
  alone is not proof of completed training.
- Existing V1, V2, and V3 result and checkpoint directories are not reused.

## Verification

- Unit tests for per-sample Top-Q selection, `min_k`/`max_k`, empty masks,
  timestep gating, and gradient flow through predicted `x0`.
- Unit tests proving CT floor monotonicity, exact endpoints, and correct L0
  inheritance from L1.
- Unit tests for Gabor agreement bounds, exact no-op fallbacks, descriptor
  detachment, L1-only application, and shared LH/HL/HH multiplier.
- Exact initial equivalence to BR-D through zero-initialized projection heads.
- Forward and sampling tests proving no mask or noisy-state modification.
- Compatibility tests for independent adapter, organ, and ROI-SUV interfaces.
- Runner tests for Stage A stop conditions, Stage B dependency, fixed 64-image
  evaluation, at most two promotions, and mandatory epoch-300 completion.
- Full test suite and fake-data end-to-end runner smoke test.

## Claim Boundary

If G2 passes the gates and improves over G0, the supported contribution is a
two-level Haar boundary-frequency injection mechanism with scale-specific CT
reliability, normalized lesion-peak preservation, and phase-insensitive Gabor
direction agreement used as a bounded inference-time reliability veto.

The study does not claim physical SUV recovery, organ-specific improvement
without non-empty organ masks, or a benefit from complete Wavelet U-Net. It
also does not attribute BR-D's V3 gains to Gabor; Gabor is evaluated separately
in Stage B.
