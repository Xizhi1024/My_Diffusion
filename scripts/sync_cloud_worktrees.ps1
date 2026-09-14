<#
    sync_cloud_worktrees.ps1 — convenience wrapper around scripts/sync_cloud_worktrees.py

    One upload, manifest-driven distribution into the router and feature
    worktrees.  Dry-run by default; only --apply writes anything.

    Examples:
        .\scripts\sync_cloud_worktrees.ps1 -Target all -Mode dry
        .\scripts\sync_cloud_worktrees.ps1 -Target all -Mode apply
        .\scripts\sync_cloud_worktrees.ps1 -Target router -Mode verify
#>
param(
    [ValidateSet('router', 'feature', 'all')]
    [string]$Target = 'all',

    [ValidateSet('dry', 'apply', 'verify')]
    [string]$Mode = 'dry',

    [string]$Config = 'configs/cloud_worktree_sync.yaml',

    [string]$RunId = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

$argList = @(
    'run',
    'python',
    'scripts/sync_cloud_worktrees.py',
    '--config', $Config,
    '--target', $Target
)

if ($Mode -eq 'dry') {
    $argList += '--dry-run'
}
elseif ($Mode -eq 'apply') {
    $argList += '--apply'
}
else {
    $argList += '--verify'
}

if ($RunId) {
    $argList += '--run-id', $RunId
}

Push-Location $repoRoot
try {
    & pixi @argList
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
