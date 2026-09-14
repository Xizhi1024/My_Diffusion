# Thin Windows launcher for the fail-closed H3 overnight audit.
#
# All state, heartbeat, logging, resume, timeout, and sleep-inhibition logic
# lives in run_h3_fixed_schedule_overnight.py. Unknown arguments are forwarded
# verbatim, for example:
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File `
#     scripts/run_h3_fixed_schedule_overnight.ps1 `
#     --skip-full-tests
#
# Resume:
#   powershell -NoProfile -ExecutionPolicy Bypass -File `
#     scripts/run_h3_fixed_schedule_overnight.ps1 `
#     --resume --run-dir results/.../overnight_runs/<run-id>

[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunnerArguments
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$repoRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "..")
)
$runner = Join-Path $PSScriptRoot "run_h3_fixed_schedule_overnight.py"

if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Python overnight runner not found: $runner"
}

$pixiCommand = Get-Command pixi -CommandType Application -ErrorAction Stop
$pixi = $pixiCommand.Source

# Keep redirected Python output deterministic and immediately visible in both
# the console and the runner's UTF-8 log files.
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PIXI_LOCKED = "1"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$exitCode = 2
Push-Location $repoRoot
try {
    $commandArguments = @(
        "run",
        "--locked",
        "python",
        "-u",
        $runner
    )
    if ($null -ne $RunnerArguments) {
        $commandArguments += $RunnerArguments
    }

    & $pixi @commandArguments
    $exitCode = $LASTEXITCODE
}
catch {
    Write-Error $_
    $exitCode = 2
}
finally {
    Pop-Location
}

exit $exitCode
