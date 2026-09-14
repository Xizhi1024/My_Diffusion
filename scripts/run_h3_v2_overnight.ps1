# Thin Windows launcher for the isolated H3-v2 overnight pipeline.

[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunnerArguments
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$runner = Join-Path $PSScriptRoot "run_h3_v2_overnight.py"
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Python overnight runner not found: $runner"
}

$pixiCommand = Get-Command pixi -CommandType Application -ErrorAction Stop
$pixi = $pixiCommand.Source
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
