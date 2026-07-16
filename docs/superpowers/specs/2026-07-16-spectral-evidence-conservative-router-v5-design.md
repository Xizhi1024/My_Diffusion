# Spectral-Evidence Conservative Frequency Router V5 Design

**Date:** 2026-07-16

**Status:** Approved in conversation; awaiting written-spec review

**Scope:** PNG CT-to-PET residual Brownian bridge with the frozen V2 conditional mean

## Goal

Extend the validated D3/G0 boundary-reliable frequency branch with a safe, learnable Gabor-DCT decision mechanism and explicit cross-level frequency routing:

```text
noisy residual + CT + bridge timestep
    -> two-level Haar packets (the only injectable content)
    -> constrained Gabor and selected-DCT descriptors (evidence only)
    -> bounded trust amplitude + conservative destination router
    -> L2/L1/L0 decoder residuals or an explicit null route
```

The method must preserve the current diffusion state, frozen conditional mean, Top-Q supervision, CT-reliability floors, and zero-initialized residual fusion. Gabor and DCT must never inject signed carriers or reconstructed frequency content directly.

## Scientific motivation

V4 produced a Pareto split. G0 retained better lesion Top-Q error, centroid distance, failure rate, and directional-spectrum error. G2 improved MAE, SSIM, and false-hotspot density, but its fixed L1-only Gabor agreement shifted lesion reliability in the wrong direction for some patients.

The V5 hypothesis is that Gabor is useful evidence but not universally reliable content. A model should learn:

1. whether a Haar packet is supported by CT, current residual, Gabor orientation, and DCT spectral composition;
2. how much of that packet is safe to inject at the current bridge timestep;
3. which adjacent decoder level should receive it;
4. when unsupported or artifact-like evidence should be rejected entirely.

The core contribution is not the simultaneous use of Haar, Gabor, and DCT. It is a reliability-constrained, conservative allocation rule for spatially aligned Haar packets, with Gabor-DCT features used only as routing evidence.

## Design principles

### 1. Separate injectable content from spectral evidence

- **Haar:** produces local, multiscale, spatially aligned detail packets. Haar packets are the only new frequency content passed to decoder projections.
- **Gabor:** describes local orientation, scale energy, and anisotropy. Parameters remain learnable only within bounded neighborhoods of their analytic initialization.
- **DCT:** describes the spectral composition of each Haar subband. It generates a small real-valued descriptor and never reconstructs an image or feature map.
- **FFT:** remains outside V5. It may later become a veto-only narrow-band artifact detector if V5 passes the lesion gates.

This separation prevents the condition descriptors from becoming a second uncontrolled image-generation path.

### 2. Use selected feature-level DCT descriptors

For each native Haar detail band from residual and CT:

```text
detail map -> absolute/log energy -> adaptive 8x8 pooling -> fixed DCT basis
           -> 12 selected coefficient groups -> learnable normalized weighting
```

The 12 groups emphasize middle and middle-high frequencies. DC and extreme corner frequencies receive low prior mass. Learnable frequency weights are a softmax residual around that prior, not an unconstrained full-spectrum filter.

For level `k` and band `b`, the descriptor contains:

```text
e_dct_residual, e_dct_ct, abs(e_dct_residual - e_dct_ct)
```

Global 192x192 DCT and overlapping block-DCT maps are excluded from the first V5 implementation. Spatial localization continues to come from the Haar energy reliability maps.

### 3. Keep Gabor learnable but bounded

The existing quadrature Gabor bank remains phase-insensitive. Frequency, orientation, and scale parameters use bounded residual parameterizations around their initialization. The router consumes only:

- orientation and scale energy;
- anisotropy;
- Gabor-Haar agreement;
- Gabor-DCT agreement.

No signed Gabor response, carrier, phase, adapter feature, diffusion-state modulation, or direct Gabor skip injection is allowed.

## Router architecture

### 1. Evidence vector

For Haar packet `H_(k,b)`, where `k in {L2,L1}` and `b in {LH,HL,HH}`:

```text
z_(k,b) = [
    timestep embedding,
    bridge log-SNR and analytic noise release,
    CT reliability and content reliability,
    residual/CT DCT energies and their absolute difference,
    Gabor orientation/scale energy and anisotropy,
    Gabor-Haar agreement,
    Gabor-DCT agreement,
    local residual high-energy and anisotropy statistics
]
```

Lesion masks, target PET, organ masks, and physical-SUV labels are prohibited router inputs. They remain loss/evaluation-only data.

### 2. Trust amplitude

The existing G0 gate is retained as a safety envelope:

```text
g_base_(k,b) = gate_max * noise * CT_reliability * content_reliability
```

V5 learns only a bounded residual correction:

```text
delta_(k,b) = bounded_mlp(z_(k,b)) in [-0.05, +0.10]
amplitude_(k,b) = clip(g_base_(k,b) * (1 + delta_(k,b)), 0, gate_max)
```

The output bias is calibrated so `delta=0` at initialization. Gabor-DCT evidence can therefore make a limited positive or negative adjustment but cannot bypass the validated noise release, CT floors, content gate, or global cap.

### 3. Conservative destination routing

The destination router is a masked softmax over the native level, the next shallower level, and an explicit null destination:

```text
H_L2 -> {L2, L1, null}
H_L1 -> {L1, L0, null}
```

All-to-all routing is intentionally excluded. It would require repeated resampling, increase aliasing risk, and weaken interpretability.

For every packet:

```text
pi_(k,b->d) = masked_softmax(router_mlp(z_(k,b)))
sum_d pi_(k,b->d) = 1
```

The injected residual at decoder level `d` is:

```text
I_d = sum_(k,b) amplitude_(k,b) * pi_(k,b->d) * R_(k->d)(H_(k,b))
```

`R` is either native subband projection or one adjacent pure-high-frequency inverse Haar reconstruction. Signed Haar coefficients are never bilinearly resized. L3 remains exact zero.

The null route is essential. Without it, a softmax would force an unsupported or stripe-like packet into some decoder layer instead of suppressing it.

## Safe initialization and fallback

### Exact no-op initial function

Every decoder projection still ends with an exact-zero `1x1` convolution. Therefore the first forward pass is exactly equivalent to disabling the frequency branch, regardless of the initial router probabilities.

### Null-biased but trainable route prior

The training router must not use a hard one-hot all-null matrix. Exact zero probability on every active destination can block useful gradients and cause permanent null collapse. Instead:

```text
initial null probability: 0.90
initial active probability: 0.10, divided across allowed non-null destinations
```

The projection heads provide the exact no-op guarantee, while the small active probability preserves a trainable path. Router gradients may be zero on the first update because the projections are zero, but become available once the zero projections begin learning.

### Exact operational fallback

The module remains a purely additive residual branch and exposes an `enabled` switch. When disabled or configured as hard all-null, the forward path short-circuits before every projection and returns exact zero tensors. It must not rely on multiplying projection inputs by zero, because learned convolution biases could otherwise produce a nonzero residual. Thus a failed V5 experiment can fall back to the same non-frequency model without changing the bridge state, U-Net inputs, conditional mean, or sampler equations.

This is an operational fail-safe, not a claim that a trained enabled router is mathematically guaranteed never to reduce validation performance. Promotion hard gates determine whether the enabled model is accepted.

## Regularization and diagnostics

No new image-domain loss is added in the first V5 screen. Existing Top-Q and reconstruction losses provide the optimization signal.

The router exposes:

- trust amplitude by level and Haar band;
- destination probability by source level, band, and destination;
- null probability;
- active route entropy;
- per-level injection norm relative to the native skip norm;
- DCT selected-frequency weights;
- Gabor parameter offsets and anisotropy;
- Gabor-Haar and Gabor-DCT agreement;
- timestep-binned route statistics.

Weak regularizers are limited to:

1. the existing spatial gate-TV penalty;
2. a small route temporal-smoothness term between adjacent timestep bins;
3. a small parameter-offset penalty for learnable Gabor and DCT frequency weights.

There is no target usage regularizer forcing non-null injection. Null collapse is treated as evidence that the branch adds no value, not as a behavior that must be overcome artificially.

## Configuration sketch

```yaml
modules:
  residual_frequency:
    enabled: true
    mode: spectral_evidence_router
    gate_max: 0.25
    ct_reliability_floor_l2: 0.25
    ct_reliability_floor_l1: 0.50
    projection_zero_init: true
    amplitude_delta_min: -0.05
    amplitude_delta_max: 0.10

    dct_descriptor:
      enabled: true
      pooled_size: 8
      selected_frequencies: 12
      learnable_weights: true
      weight_temperature: 1.0

    gabor_descriptor:
      enabled: true
      learnable_parameters: true
      inject_content: false
      parameter_offset_penalty: 1.0e-4

    cross_level_router:
      enabled: true
      adjacent_only: true
      null_destination: true
      initial_null_probability: 0.90
      temporal_smoothness_weight: 1.0e-4
```

## Fixed-budget V5 ablation

All variants use the same frozen conditional mean, U-Net initialization seed, Top-Q loss, CT floors, training subset, validation subset, EMA evaluation, 50-epoch screen, and no new image-domain loss.

### Stage A: identify spectral evidence value

Cross-level routing remains disabled so descriptor value is isolated.

| ID | Gabor evidence | DCT evidence | Gate form | Purpose |
|---|---:|---:|---|---|
| S0 | no | no | current G0 | validated control |
| S1 | yes | no | bounded residual per-level gate | learnable Gabor value |
| S2 | no | yes | bounded residual per-level gate | selected-DCT value |
| S3 | yes | yes | bounded residual per-level gate | evidence complementarity |

At most one eligible evidence configuration advances to Stage B. If none improves or preserves the lesion gates, S0 remains the evidence reference.

### Stage B: identify conservative routing value

| ID | Evidence | Routing | Purpose |
|---|---|---|---|
| N0 | none | hard all-null | exact frequency-off safety control |
| T0 | selected Stage-A evidence | independent timestep/level gate | TAFG-style non-conservative control |
| C0 | Haar/CT/content only | conservative adjacent-level routing + null | isolate routing contribution |
| C1 | selected Gabor-DCT evidence | conservative adjacent-level routing + null | complete V5 method |

At most two eligible non-reference candidates are retrained from scratch for exactly 300 epochs. Non-uniform timestep sampling, FFT artifact veto, new boundary losses, and block-DCT maps remain off throughout V5 so architectural effects stay identifiable.

## Promotion and success criteria

All V4 hard gates remain mandatory. A V5 candidate must additionally target the G0/G2 Pareto region on the fixed 64-sample validation subset:

| Metric | Required target |
|---|---:|
| Top-Q peak error | <= 0.080 |
| failure rate | <= 0.0625 |
| centroid distance | <= 3.15 |
| directional-spectrum error | <= 0.0095 |
| SSIM | >= 0.947 |
| MAE | <= 0.0365 |
| false-hotspot density | <= 0.00014 |

Final comparison uses paired per-patient bootstrap intervals. Aggregate thresholds alone do not establish superiority.

## Acceptance criteria

- Gabor and DCT affect only evidence, trust amplitude, and routing probabilities.
- Haar remains the only injected frequency content.
- Initial frequency output is exact zero through zero-initialized projections.
- Training initialization is null-biased but retains nonzero active-route probability.
- Hard all-null mode is exactly equivalent to disabling frequency injection.
- Every packet's routing probabilities, including null, sum to one.
- Only native and one adjacent shallower destination are permitted.
- L3 is exact zero and signed Haar coefficients are never bilinearly resized.
- Trust amplitude cannot exceed the existing `gate_max` or bypass base reliability.
- No training mask or target-derived value enters the router or sampler.
- Router probabilities and selected spectral weights receive gradients after projection heads leave zero.
- Legacy G0/G2 construction and result reproduction remain available.
- The ablation runner records initialization seed, checkpoint epoch, routing diagnostics, and all gate decisions.

## Scientific claim boundary

The method is described as a **spectral-evidence-guided conservative router for CT-to-PET diffusion**. It uses constrained Gabor-DCT descriptors to decide the trust and adjacent decoder destination of spatially aligned Haar packets, with an explicit rejection path for unsupported frequencies.

It is not claimed as the first use of Gabor, DCT attention, timestep frequency gating, multiscale frequency diffusion, or learnable U-Net block weighting. Any priority claim about conservative cross-level routing requires a dedicated final literature search before manuscript submission.
