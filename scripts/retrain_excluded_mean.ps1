# Retrain the pathology-EXCLUDED conditional mean on the production (main_data) split.
#
# Why: the current production checkpoint checkpoints/freq_mean_pretrain_v2/mean_best.pt
# is pathology-INCLUDED (MeanPretrainer previously had no mask path), so it is NOT the
# lesion-enriched residual endpoint validated by H2 (see docs/current_model_status.md
# section 3.4). This script retrains an excluded mean using the SAME validated exclusion
# as scripts/validate_h2_pathology_excluded_residual.py (guard_radius_px=8, Charbonnier
# on background LL2 pixels only), on the full main_data split used by production.
#
# Stage V2-03 hardening: the cloud run is FORCED to fail-closed cache lineage so the
# resulting checkpoint is independently auditable:
#   data.require_cache_lineage = true
#   data.dataset_contract      = configs/dataset_contract_stage0a_v1.json
#   data.cache_lineage         = cache/tensors_main/cache_lineage.json
# If the sealed cache lineage is missing or inconsistent with the locked Stage 0A
# contract, src/data/lineage.py raises before training starts (fail-closed). This
# script does NOT overwrite the included production mean and does NOT touch the v5
# checkpoint path; a separate cutover step is required after the new checkpoint is
# generated and audited.
#
# Run on cloud Win from the repo root (D:\ECPC-IDS-SEVEN-Work3\My_diffusion):
#   pixi run pwsh scripts/retrain_excluded_mean.ps1
#   # or override defaults (output dir / epochs / seed / guard px):
#   pixi run pwsh scripts/retrain_excluded_mean.ps1 -OutputDir checkpoints/freq_mean_excluded_v1 -Epochs 30 -Seed 42

param(
    [string]$Config          = "configs/experiments/slmf_png_residual_frequency.yaml",
    [string]$OutputDir       = "checkpoints/freq_mean_excluded_v1",
    [int]   $Epochs          = 30,
    [int]   $Seed            = 42,
    [int]   $GuardPx         = 8,
    [string]$DatasetContract = "configs/dataset_contract_stage0a_v1.json",
    [string]$CacheLineage    = "cache/tensors_main/cache_lineage.json"
)

$ErrorActionPreference = "Stop"

if ($GuardPx -lt 0) { throw "GuardPx must be non-negative" }
if (-not (Test-Path -LiteralPath $DatasetContract)) {
    throw "Dataset contract not found: $DatasetContract"
}
# The cache lineage file itself does not have to exist on this (local) machine, but
# on the cloud run it MUST exist; lineage.load_checkpoint_data_lineage fails closed.
if (-not (Test-Path -LiteralPath $Config)) { throw "Base config not found: $Config" }

# Match the INCLUDED production mean's regimen (same base config, seed, epochs); the ONLY
# intended difference is the loss: included -> pathology-excluded. This isolates the effect.
# The three lineage overrides below are forced regardless of what the base config says.
$cmdArgs = @(
    "run", "python", "scripts/pretrain_conditional_mean.py",
    "--config", $Config,
    "--output-dir", $OutputDir,
    "--epochs", $Epochs,
    "--seed", $Seed,
    "--override", "experiment.name=freq_mean_excluded_v1",
    "--override", "modules.conditional_mean.pathology_exclusion.enabled=true",
    "--override", "modules.conditional_mean.pathology_exclusion.guard_radius_px=$GuardPx",
    "--override", "data.require_cache_lineage=true",
    "--override", "data.dataset_contract=$DatasetContract",
    "--override", "data.cache_lineage=$CacheLineage",
    "--override", "data.use_fake_data=false"
)
& pixi @cmdArgs
if ($LASTEXITCODE -ne 0) {
    throw "Pretraining failed with exit code $LASTEXITCODE"
}

$bestPath = Join-Path $OutputDir "mean_best.pt"
$fingerprintPath = "$bestPath.fingerprint.json"
if (-not (Test-Path -LiteralPath $bestPath)) {
    throw "Expected checkpoint was not written: $bestPath"
}

Write-Host ""
Write-Host "Done. New excluded mean: $bestPath"
Write-Host "Audit the fail-closed policy + lineage + checkpoint SHA-256 before any cutover:"
if (Test-Path -LiteralPath $fingerprintPath) {
    Write-Host "  Get-Content $fingerprintPath"
} else {
    Write-Host "  (sidecar missing) $fingerprintPath"
}
Write-Host "Confirm the checkpoint embeds verified cache lineage:"
Write-Host "  python -c `"import torch; c=torch.load(r'$bestPath', map_location='cpu'); print(c.get('data_lineage', {}).get('cache_metadata_sha256', 'NO_LINEAGE')); print(c['mean_config'].get('pathology_exclusion'))`""
Write-Host "Independent SHA-256 (to contrast with the included a2035873... checkpoint):"
Write-Host "  Get-FileHash -Algorithm SHA256 $bestPath"
Write-Host ""
Write-Host "NOTE: the production v5 checkpoint path is intentionally NOT changed by this script."
Write-Host "      Cutover requires a separate, recorded decision after auditing the sidecar."
