# H3 direct-PNG prior preview

`scripts/estimate_h3_prior_from_png.py` estimates the six H3
`level x Haar-band` active-prior curves directly from PNG files. It never reads
or writes a tensor cache.

## Scope boundary

- Local execution is for code changes, numerical checks, and an exploratory
  curve preview.
- Every artifact is marked `PREVIEW_ONLY`; it is not a frozen H3-v2 schedule
  and the formal schedule loader must reject it.
- The authoritative run remains the cloud H3-v2 calibration, with the cloud
  dataset lineage, pathology-excluded checkpoint, and configured cache.
- Paths in commands and serialized artifacts are repository-relative. Runtime
  code may resolve them in memory, but it does not persist machine paths.

The direct-PNG worker uses:

- `main_data/split_manifest.csv` as the split authority;
- all physical `train/val/test` directories only as file stores indexed by
  `sample_id`;
- CT grayscale resize and `x / 127.5 - 1`;
- PET grayscale resize, `255 - x`, then `/ 127.5 - 1`;
- nearest-neighbor mask resize and `uint8 > 127`;
- one deterministic noise image per `(role, sample_id)`, reused for all 1000
  timesteps;
- patient-equal aggregation followed by a non-increasing isotonic projection.

## Local preview

The local checkpoint below is the historical pathology-excluded H2 endpoint.
Its embedded lineage is recorded but is deliberately not declared verified for
the new direct-PNG run.

```powershell
pixi run python scripts/estimate_h3_prior_from_png.py `
  --root . `
  --png-root Data/data `
  --manifest main_data/split_manifest.csv `
  --mean-checkpoint results/mechanism_validation/01_h2_residual_enrichment/checkpoints/mean_excluded_fixed.pt `
  --output-dir results/h3_png_prior_preview `
  --batch-size 16 `
  --device cpu
```

Expected local inventory:

- 1191 indexed CT/PET/mask triplets;
- 766 mechanism-train slices from 99 patients;
- 188 calibration slices from 25 patients;
- 237 validation slices from 31 patients excluded before pixel decoding.

Outputs:

- `results/h3_png_prior_preview/h3_prior_preview.json`
- `results/h3_png_prior_preview/h3_prior_curves.csv`
- `results/h3_png_prior_preview/h3_prior_patient_curves.npz`
- `results/h3_png_prior_preview/h3_prior_curves.png`

## Cloud execution

The same direct-PNG implementation can be rerun against the cloud copy without
changing code or embedding a mount path:

```powershell
pixi run python scripts/estimate_h3_prior_from_png.py `
  --root . `
  --png-root Data/data `
  --manifest main_data/split_manifest.csv `
  --mean-checkpoint checkpoints/freq_mean_excluded_v1/mean_best.pt `
  --output-dir results/h3_png_prior_cloud_preview `
  --batch-size 32 `
  --device cuda
```

That result is still a preview. The formal, inference-consumable H3-v2 schedule
must instead be produced by the preregistered cloud worker after all frozen
hashes and lineage checks pass:

```powershell
pixi run python scripts/calibrate_h3_v2_full_timestep_native_null.py `
  --root . `
  --config configs/h3_v2_full_timestep_native_null_v1.json `
  --output-dir results/h3_v2_full_timestep_native_null `
  --require-cuda
```

Do not copy a local preview JSON into an H3 schedule path, and do not substitute
`cache/tensors` for the configured cloud `cache/tensors_main`.
