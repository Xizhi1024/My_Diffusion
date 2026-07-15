# Frozen-Mean Residual Ablation V2 Design

**Date:** 2026-07-15

**Status:** Approved through the user's instruction to continue the proposed two-stage fix

## Evidence and objective

The first R0/R2/F1-F4 screen completed correctly but promoted nothing. R0
retained normal image quality (`SSIM=0.9216`, `MAE=0.0778`), while every
residual variant collapsed to approximately `SSIM=0.54-0.57` and
`MAE=0.58-0.66`. R2 nevertheless improved centroid distance and failure rate,
and F4 improved lesion peak error and suppressed stripe excess. The frequency
ideas therefore cannot yet be judged: the residual target changed throughout
joint training because the conditional mean was learning at the same time.

V2 fixes that confound before repeating the same block screen. It does not
weaken the artifact gates and does not promote any V1 checkpoint.

## Considered approaches

1. **Separate mean pretraining and frozen residual training (selected).** A
   small standalone job trains only the LL2 predictor, after which every
   residual variant loads the same checkpoint and freezes it. This produces a
   stationary residual target and is cheap to run.
2. **Warm-up and freeze inside the full Trainer.** This uses one process but
   requires optimizer, scheduler, EMA, checkpoint, and resume transitions at
   the phase boundary. It adds state that is unrelated to the research test.
3. **Longer joint training.** This changes no code but spends the most GPU time
   while retaining the moving-target failure mode.

## Stage M0: conditional-mean pretraining

`scripts/pretrain_conditional_mean.py` loads the same PNG configuration and
patient split as the diffusion experiments. It constructs only
`LowFrequencyPETPredictor`; the UNet, priors, residual preconditioner, and
diffusion schedule are never constructed.

For CT `c` and PET `y`, the target is the level-2 orthonormal Haar coefficient
`LL2(y)`. The loss is the configured Charbonnier error:

```text
L_mean = mean(sqrt((mu_LL2(c) - LL2(y))^2 + epsilon^2))
```

Training uses AdamW, the existing data loader, AMP when CUDA is available, and
validation LL2 loss for checkpoint selection. The output directory contains:

```text
mean_best.pt     best validation LL2 checkpoint
mean_last.pt     final epoch checkpoint
history.json     train/validation loss per epoch
resolved_config.yaml
```

The checkpoint schema is explicit:

```python
{
    "format_version": 1,
    "model": predictor.state_dict(),
    "mean_config": conditional_mean_config,
    "epoch": completed_epoch,
    "val_loss": best_or_current_validation_loss,
}
```

M0 defaults to 30 epochs. It is a small CNN-only job and must finish before any
residual screen begins.

## Frozen-mean model loading

The conditional-mean configuration gains two fields:

```yaml
modules:
  conditional_mean:
    enabled: true
    checkpoint: checkpoints/freq_mean_pretrain_v2/mean_best.pt
    freeze: true
    loss_weight: 0.0
```

When `checkpoint` is set, `SLMFBBDM` loads the checkpoint's `model` state
strictly. `freeze: true` requires a checkpoint, sets every mean-predictor
parameter to `requires_grad=false`, and keeps the predictor in evaluation mode
when the parent model enters training mode. Frozen residual training computes:

```text
mu = frozen_mean(CT)
r0 = PET - mu
PET_hat = mu + residual_hat
```

No residual, lesion, Gabor, or image loss can update `mu`. The explicit mean
loss is disabled in V2 residual runs because M0 already selected the mean
checkpoint.

R0 disables the conditional mean and remains a full-PET BBDM reference.

## Matched core initialization

The model configuration gains `model.initialization_seed`. Construction of
the core UNet occurs inside an isolated CPU RNG context seeded with this value.
The caller's RNG state is restored afterward. Consequently R0/R2/F1-F4 receive
bitwise-identical initial values for every shape-compatible UNet parameter,
regardless of whether a frequency module exists or how many parameters it
creates. Frequency-module parameters remain independently initialized as part
of their own tested intervention.

This is preferable to relying on one global experiment seed because optional
module constructors consume different numbers of random values.

## V2 experiment orchestration

The runner remains backward compatible with the V1 plan. A V2 plan adds an
optional `mean_pretrain` section and accepts `--stage mean`:

```yaml
mean_pretrain:
  enabled: true
  experiment: freq_mean_pretrain_v2
  epochs: 30
  checkpoint: checkpoints/freq_mean_pretrain_v2/mean_best.pt
```

`--stage all` executes this order:

```text
M0 mean pretraining
  -> R0/R2/F1/F2/F3/F4 fixed-seed screen
  -> artifact gates and ranking
  -> clean top-K full retraining
```

The M0 command is skipped only when its checkpoint already exists; `--force`
reruns it. Every residual resolved config points to the same M0 checkpoint.
V2 uses new experiment names and `results/frequency_ablations_v2`, so V1
results remain untouched.

The screen definitions remain:

```text
R0  full-PET BBDM reference
R2  frozen-mean residual BBDM
F1  R2 + Haar level-matched skip injection
F2  R2 + bounded complex-Gabor gate on level-1 residual details
F3  R2 + Haar injection + Gabor gate
F4  F3 + residual-wavelet and target-direction Gabor losses
```

The existing failure, false-hotspot, stripe-excess, and SSIM hard gates remain
unchanged. No variant is promoted merely because it ranks first.

## Failure handling

- A frozen mean without a checkpoint fails model construction.
- A missing or incompatible checkpoint fails before diffusion training.
- Mean pretraining requires a validation loader; it does not silently select
  a checkpoint on training loss.
- The runner does not use any V1 screen checkpoint or result path.
- If no V2 variant passes the gates, full retraining again stops automatically.

## Verification

- Haar/mean pretraining loss and optimizer-step tests.
- Mean checkpoint round-trip, strict loading, freezing, and evaluation-mode
  persistence tests.
- Exact UNet initialization equality across R2/F1/F2/F3/F4 at one seed.
- Runner manifest ordering and V1-plan backward compatibility tests.
- V2 preset construction and one-batch forward tests.
- Complete pytest suite.
- Fake-data M0 pretraining followed by frozen-mean F4 train/evaluate smoke.

## Success interpretation

The first question is whether frozen-mean R2 restores image quality. If R2
still fails the SSIM/MAE gate, the residual bridge or sampling path must be
revisited and F1-F4 must not be interpreted. If R2 recovers near R0, differences
among F1-F4 can be attributed to the tested frequency interventions with much
stronger confidence.
