# One-command, fail-closed cloud runner for V2-03 pathology-excluded mean.
#
# This script intentionally stops after checkpoint policy + lineage auditing.
# It never runs CT support, curriculum, artifact safety, H5-v2, or H6-v2.

param(
    [string]$OutputDir = "checkpoints/freq_mean_excluded_v1",
    [int]$Epochs = 30,
    [int]$Seed = 42,
    [int]$GuardPx = 8,
    [switch]$AuditExisting
)

$ErrorActionPreference = "Stop"
$script:RunId = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmss.fffZ")
$script:RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$script:RunOutput = Join-Path $script:RepoRoot (
    "results/mechanism_validation_v2/03_excluded_mean_production/cloud_runs/" +
    $script:RunId
)
$script:CanonicalStage = Join-Path $script:RepoRoot (
    "results/mechanism_validation_v2/03_excluded_mean_production"
)
$script:StartedAt = (Get-Date).ToUniversalTime().ToString("o")
$script:ScriptPath = $PSCommandPath

function Resolve-RepoPath([string]$PathValue) {
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath(
        (Join-Path $script:RepoRoot $PathValue)
    )
}

function Write-JsonAtomic([string]$PathValue, $Payload) {
    $parent = Split-Path -Parent $PathValue
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $temporary = "$PathValue.tmp"
    $Payload |
        ConvertTo-Json -Depth 30 |
        Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $PathValue -Force
}

function Assert-File([string]$PathValue, [string]$Label) {
    if (-not (Test-Path -LiteralPath $PathValue -PathType Leaf)) {
        throw "$Label not found: $PathValue"
    }
}

function Invoke-NativeChecked(
    [string]$Label,
    [string]$Executable,
    [string[]]$CommandArguments
) {
    Write-Host ""
    Write-Host "==== $Label ===="
    & $Executable @CommandArguments
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "$Label failed with exit code $code"
    }
}

$checkpointPath = Resolve-RepoPath (Join-Path $OutputDir "mean_best.pt")
$sidecarPath = "$checkpointPath.fingerprint.json"
$cacheLineagePath = Resolve-RepoPath "cache/tensors_main/cache_lineage.json"
$contractPath = Resolve-RepoPath "configs/dataset_contract_stage0a_v1.json"
$manifestPath = Resolve-RepoPath "main_data/split_manifest.csv"
$stage0cPath = Resolve-RepoPath (
    "results/mechanism_validation/00C_cloud_data_gate/decision.json"
)
$canonicalDecisionPath = Join-Path $script:CanonicalStage "decision.json"
$lineageOutput = Join-Path $script:RunOutput "checkpoint_lineage"

New-Item -ItemType Directory -Force -Path $script:RunOutput | Out-Null
Push-Location $script:RepoRoot
try {
    Assert-File $contractPath "Locked dataset contract"
    Assert-File $manifestPath "Authoritative manifest"
    Assert-File $cacheLineagePath "Sealed cache lineage"
    Assert-File $stage0cPath "Stage 0C decision"

    $stage0c = Get-Content -LiteralPath $stage0cPath -Encoding utf8 -Raw |
        ConvertFrom-Json
    if ($stage0c.data_gate -ne "PASS") {
        throw "Stage 0C data_gate is not PASS"
    }
    if ($stage0c.cloud_training_gate -ne "PASS") {
        throw "Stage 0C cloud_training_gate is not PASS"
    }

    $drive = (Get-Item -LiteralPath $script:RepoRoot).PSDrive
    if ($null -ne $drive.Free -and $drive.Free -lt 2GB) {
        throw "Less than 2 GiB free on repository drive"
    }

    $freezeCall = @{
        Label = "Frozen H4-v2 integrity audit"
        Executable = "pixi"
        CommandArguments = @(
            "run",
            "python",
            "scripts/audit_v2_freeze_integrity.py"
        )
    }
    Invoke-NativeChecked @freezeCall

    $cudaCall = @{
        Label = "CUDA availability"
        Executable = "pixi"
        CommandArguments = @(
            "run",
            "python",
            "-c",
            (
                "import sys,torch; " +
                "print('cuda_available=',torch.cuda.is_available()); " +
                "print('device=',torch.cuda.get_device_name(0) " +
                "if torch.cuda.is_available() else 'NONE'); " +
                "sys.exit(0 if torch.cuda.is_available() else 3)"
            )
        )
    }
    Invoke-NativeChecked @cudaCall

    if ($AuditExisting) {
        Assert-File $checkpointPath "Existing excluded-mean checkpoint"
        Assert-File $sidecarPath "Existing excluded-mean fingerprint"
        Write-Host "AuditExisting selected; training is skipped."
    }
    else {
        if (Test-Path -LiteralPath $checkpointPath) {
            throw (
                "Refusing to overwrite existing checkpoint: $checkpointPath. " +
                "Use -AuditExisting to audit it without retraining, or choose " +
                "a new -OutputDir."
            )
        }
        $trainingCall = @{
            Label = "V2-03 pathology-excluded mean retraining"
            Executable = "powershell"
            CommandArguments = @(
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                "scripts/retrain_excluded_mean.ps1",
                "-OutputDir",
                $OutputDir,
                "-Epochs",
                [string]$Epochs,
                "-Seed",
                [string]$Seed,
                "-GuardPx",
                [string]$GuardPx
            )
        }
        Invoke-NativeChecked @trainingCall
    }

    Assert-File $checkpointPath "Excluded-mean checkpoint"
    Assert-File $sidecarPath "Excluded-mean fingerprint"
    $sidecar = Get-Content -LiteralPath $sidecarPath -Encoding utf8 -Raw |
        ConvertFrom-Json
    if (-not [bool]$sidecar.pathology_exclusion.enabled) {
        throw "Fingerprint says pathology_exclusion.enabled is false"
    }
    if ([int]$sidecar.pathology_exclusion.guard_radius_px -ne $GuardPx) {
        throw (
            "Fingerprint guard radius mismatch: expected $GuardPx, got " +
            "$($sidecar.pathology_exclusion.guard_radius_px)"
        )
    }
    if (-not [bool]$sidecar.lineage_present) {
        throw "Fingerprint says checkpoint data lineage is absent"
    }
    $requiredLineage = @(
        "manifest_semantic_sha256",
        "raw_png_combined_sha256",
        "preprocessing_config_sha256",
        "dataset_contract_sha256",
        "cache_payload_sha256",
        "cache_metadata_sha256"
    )
    foreach ($field in $requiredLineage) {
        $value = [string]$sidecar.lineage_fingerprints.$field
        if ($value -notmatch "^[0-9a-f]{64}$") {
            throw "Fingerprint lineage field is missing/invalid: $field"
        }
    }
    $actualHash = (
        Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if ($actualHash -ne ([string]$sidecar.checkpoint_sha256).ToLowerInvariant()) {
        throw "Checkpoint SHA-256 differs from fingerprint sidecar"
    }

    $lineageCall = @{
        Label = "Checkpoint lineage against sealed Stage 0C cache"
        Executable = "pixi"
        CommandArguments = @(
            "run",
            "python",
            "scripts/audit_checkpoint_lineage.py",
            "--checkpoint",
            $checkpointPath,
            "--cache-dir",
            "cache/tensors_main",
            "--contract",
            "configs/dataset_contract_stage0a_v1.json",
            "--output",
            $lineageOutput,
            "--hash-checkpoints"
        )
    }
    Invoke-NativeChecked @lineageCall

    $lineageDecisionPath = Join-Path $lineageOutput "decision.json"
    Assert-File $lineageDecisionPath "Checkpoint-lineage decision"
    $lineageDecision = Get-Content -LiteralPath $lineageDecisionPath -Encoding utf8 -Raw |
        ConvertFrom-Json
    if ($lineageDecision.decision -ne "PASS") {
        throw "Checkpoint-lineage decision is not PASS"
    }
    if ($lineageDecision.checkpoint_lineage -ne "PASS") {
        throw "checkpoint_lineage is not PASS"
    }

    $decision = [ordered]@{
        schema_version = 2
        stage = "03_v2_pathology_excluded_mean_production"
        pipeline_id = "V2_EXCLUDED_MEAN_PRODUCTION_V1"
        decision = "PASS"
        decision_scope = "checkpoint_policy_and_lineage_readiness_only"
        checkpoint = $checkpointPath
        checkpoint_sha256 = $actualHash
        pathology_exclusion = [ordered]@{
            enabled = $true
            guard_radius_px = $GuardPx
        }
        checkpoint_lineage = "PASS"
        sealed_cache_lineage = "PASS"
        training_executed = (-not $AuditExisting.IsPresent)
        patients_or_slices_excluded = 0
        model_mechanism_claims_allowed = $false
        production_cutover_allowed = $false
        next_stage_allowed = $false
        stop_rule = (
            "Stop after V2-03. Do not run CT support, curriculum, artifact " +
            "safety, H5-v2, or H6-v2 until the leakage-free context provider " +
            "and true band-by-timestep H3 route fallback exist."
        )
        lineage_audit = $lineageDecisionPath
        run_id = $script:RunId
        created_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    }
    Write-JsonAtomic (Join-Path $script:RunOutput "decision.json") $decision

    $prepDecision = Join-Path $script:CanonicalStage (
        "prep_decision_before_cloud_run.json"
    )
    if (
        (Test-Path -LiteralPath $canonicalDecisionPath) -and
        -not (Test-Path -LiteralPath $prepDecision)
    ) {
        $copyDecision = @{
            LiteralPath = $canonicalDecisionPath
            Destination = $prepDecision
        }
        Copy-Item @copyDecision
    }
    Write-JsonAtomic $canonicalDecisionPath $decision
    $scriptHashArgs = @{
        LiteralPath = $script:ScriptPath
        Algorithm = "SHA256"
    }
    $metadata = [ordered]@{
        run_id = $script:RunId
        started_at_utc = $script:StartedAt
        completed_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        script = $script:ScriptPath
        script_sha256 = (
            Get-FileHash @scriptHashArgs
        ).Hash.ToLowerInvariant()
        audit_existing = $AuditExisting.IsPresent
        output_dir = $OutputDir
        epochs = $Epochs
        seed = $Seed
        guard_radius_px = $GuardPx
    }
    Write-JsonAtomic (
        Join-Path $script:RunOutput "execution_metadata.json"
    ) $metadata

    Write-Host ""
    Write-Host "V2-03 PASS: checkpoint policy and lineage are ready."
    Write-Host "Checkpoint: $checkpointPath"
    Write-Host "Decision:   $canonicalDecisionPath"
    Write-Host "STOP: downstream V2 stages remain deferred."
    exit 0
}
catch {
    $failure = [ordered]@{
        schema_version = 2
        stage = "03_v2_pathology_excluded_mean_production"
        pipeline_id = "V2_EXCLUDED_MEAN_PRODUCTION_V1"
        decision = "FAIL"
        failure_phase = "CLOUD_STAGE03_FAIL_CLOSED"
        error = $_.Exception.Message
        checkpoint = $checkpointPath
        model_mechanism_claims_allowed = $false
        production_cutover_allowed = $false
        next_stage_allowed = $false
        stop_rule = "Quarantine incomplete output and do not run downstream stages."
        run_id = $script:RunId
        created_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    }
    Write-JsonAtomic (Join-Path $script:RunOutput "decision.json") $failure
    Write-Error $_
    exit 2
}
finally {
    Pop-Location
}
