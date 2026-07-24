[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$TargetRoot = 'D:\ECPC-IDS-SEVEN-Work3\My_diffusion',
    [ValidateRange(1, 10000)]
    [int]$Epochs = 50,
    [switch]$SkipFullTests,
    [switch]$Foreground
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$sourceRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$target = [System.IO.Path]::GetFullPath($TargetRoot)
$bundle = Join-Path $sourceRoot 'h3_v2_overnight_bundle_dc8f4178.zip'
$expectedBundleSha256 = (
    'c81fb08c9ee8a23f0da001bdc8bc47058f1629ba159efbf32dac9b0de4f66e26'
)

if (-not (Test-Path -LiteralPath $bundle -PathType Leaf)) {
    throw "H3-v2 bundle not found: $bundle"
}
if (-not (Test-Path -LiteralPath $target -PathType Container)) {
    throw "GPU repository root not found: $target"
}

$observedBundleSha256 = (
    Get-FileHash -LiteralPath $bundle -Algorithm SHA256
).Hash.ToLowerInvariant()
if ($observedBundleSha256 -ne $expectedBundleSha256) {
    throw (
        "H3-v2 bundle SHA-256 mismatch. Expected $expectedBundleSha256; " +
        "observed $observedBundleSha256"
    )
}

$locks = @(
    (Join-Path $target (
        'results\mechanism_validation_v2\05C_h3_v2_full_timestep_native_null' +
        '\.overnight.lock'
    )),
    (Join-Path $target 'results\mechanism_validation_v2\.development_gpu.lock')
)
$presentLocks = @($locks | Where-Object { Test-Path -LiteralPath $_ })
if ($presentLocks.Count -gt 0) {
    throw (
        "A development run lock exists; refusing to replace code while a run " +
        "may be active. Resume/inspect that run first:`n  " +
        ($presentLocks -join "`n  ")
    )
}

Write-Host "Verified bundle SHA-256: $observedBundleSha256"
Write-Host "Deploying the frozen H3-v2 code bundle to: $target"
Expand-Archive -LiteralPath $bundle -DestinationPath $target -Force

$requiredInputs = @(
    'cache\tensors_main\cache_lineage.json',
    'checkpoints\freq_mean_excluded_v1\mean_best.pt',
    'checkpoints\freq_mean_excluded_v1\mean_best.pt.fingerprint.json',
    'results\mechanism_validation_v2\03_excluded_mean_production\decision.json',
    'results\mechanism_validation_v2\05A_inference_admissibility_audit\decision.json',
    'results\mechanism_validation_v2\05B_h3_fixed_schedule_inference\decision.json'
)
$missingInputs = @(
    $requiredInputs |
        Where-Object {
            -not (Test-Path -LiteralPath (Join-Path $target $_) -PathType Leaf)
        }
)
if ($missingInputs.Count -gt 0) {
    throw (
        "Required local GPU-repository inputs are missing:`n  " +
        ($missingInputs -join "`n  ")
    )
}

$pixiCommand = Get-Command pixi -CommandType Application -ErrorAction Stop
$pixi = $pixiCommand.Source
$protocolCheck = (
    "from scripts.run_h3_v2_overnight import validate_protocol; " +
    "c,s=validate_protocol(); " +
    "print(c['config_sha256']); print(f'{len(s)} runtime sources verified')"
)

Push-Location $target
try {
    & $pixi @('run', '--locked', 'python', '-c', $protocolCheck)
    $protocolExitCode = $LASTEXITCODE
    if ($protocolExitCode -ne 0) {
        throw "Frozen H3-v2 protocol validation failed with exit code $protocolExitCode"
    }

    $launcher = Join-Path $target 'scripts\run_h3_v2_overnight.ps1'
    $runnerArguments = @('--epochs', [string]$Epochs)
    if ($SkipFullTests) {
        $runnerArguments += '--skip-full-tests'
    }
    $powershellExe = Join-Path $PSHOME 'powershell.exe'

    if ($Foreground) {
        $foregroundArguments = @(
            '-NoProfile',
            '-ExecutionPolicy',
            'Bypass',
            '-File',
            $launcher
        ) + $runnerArguments
        & $powershellExe @foregroundArguments
        exit $LASTEXITCODE
    }

    $launchLogDir = Join-Path $target 'logs'
    New-Item -ItemType Directory -Path $launchLogDir -Force | Out-Null
    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $stdoutPath = Join-Path $launchLogDir "h3_v2_launcher_${timestamp}.out.log"
    $stderrPath = Join-Path $launchLogDir "h3_v2_launcher_${timestamp}.err.log"
    $quotedLauncher = '"' + $launcher.Replace('"', '\"') + '"'
    $childArguments = @(
        '-NoProfile',
        '-ExecutionPolicy',
        'Bypass',
        '-File',
        $quotedLauncher
    ) + $runnerArguments
    $startParameters = @{
        FilePath = $powershellExe
        ArgumentList = $childArguments
        WorkingDirectory = $target
        WindowStyle = 'Hidden'
        RedirectStandardOutput = $stdoutPath
        RedirectStandardError = $stderrPath
        PassThru = $true
    }
    $process = Start-Process @startParameters

    Start-Sleep -Milliseconds 1500
    if ($process.HasExited) {
        $tail = @()
        if (Test-Path -LiteralPath $stderrPath) {
            $tail += Get-Content -LiteralPath $stderrPath -Tail 20
        }
        if (Test-Path -LiteralPath $stdoutPath) {
            $tail += Get-Content -LiteralPath $stdoutPath -Tail 20
        }
        throw (
            "Detached H3-v2 launcher exited immediately with code " +
            "$($process.ExitCode). Recent output:`n" + ($tail -join "`n")
        )
    }

    Write-Host "H3-v2 is now running locally in a hidden process."
    Write-Host "Launcher PID: $($process.Id)"
    Write-Host "stdout: $stdoutPath"
    Write-Host "stderr: $stderrPath"
    Write-Host (
        "The detailed run appears under results\mechanism_validation_v2\" +
        "05C_h3_v2_full_timestep_native_null\overnight_runs."
    )
}
finally {
    Pop-Location
}
