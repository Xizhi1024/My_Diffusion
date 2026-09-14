# Cloud runner for the perceptual-x0 five-arm ablation (Stage A).
#
# Fails closed: audits the plan + required checkpoints BEFORE any training
# starts.  It will not launch an arm whose dry-run audit is BLOCKED.
#
# Usage (on the cloud machine, from the worktree root):
#   pixi run powershell -ExecutionPolicy Bypass -File scripts/cloud_run_perceptual_x0_ablation.ps1
#
# You can edit the variables below (Project, CloudPlan, RunOutput, Arms)
# without touching the loop logic.
#
# WARNING: P1_SEG_OUT needs a real segmenter checkpoint
# (checkpoints/tiny_segmenter_v1/segmenter.pt).  It is not produced yet
# (the segmenter pretrain GATE_FAILed with recall 0.0), so P1 will SKIP.
# P0/P2/P3/P4 audit READY and will train.

param(
    [string]$Plan = 'configs/experiments/perceptual_x0_ablation_plan_v1.yaml',
    [int]$Seed = 42,
    [switch]$AuditOnly
)

$ErrorActionPreference = 'Stop'

if ($AuditOnly) {
    Write-Host ""
    Write-Host "AUDIT-ONLY 模式：只审计，不启动任何训练（纯 CPU，几秒）。"
    Write-Host "确认部署链路正确后，机器空闲时去掉 -AuditOnly 重跑即可真实训练。"
}

$Project = Split-Path -Parent $PSScriptRoot
$CloudPlan = Join-Path $Project $Plan
$RunOutput = Join-Path $Project 'results\perceptual_x0_few_step_v1\formal_seed42'

if (-not (Test-Path $CloudPlan)) {
    throw "计划文件不存在: $CloudPlan (请检查是否已部署)"
}

Set-Location $Project
$env:PYTHONUNBUFFERED = '1'

$Arms = @(
    'P0_PIXEL'
    'P1_SEG_OUT'
    'P2_FEAT_GLOBAL'
    'P3_FEAT_LESION_BALANCED'
    'P4_FEAT_RANDOM'
)

# The runner's dry-run returns exit 1 whenever ANY arm is BLOCKED, so we cannot
# rely on $LASTEXITCODE to decide per-arm readiness.  The authoritative status
# lives in resolved_runs.json under each arm's "audit.status".  We read it with
# python, NOT PowerShell's ConvertFrom-Json, because PS 5.1 chokes on the JSON
# (Chinese paths + backslashes in the resolved configs).
$ResolvedRuns = Join-Path $RunOutput 'resolved_runs.json'

function Get-ArmAuditStatus {
    param([string]$Arm)
    # Backslashes in the path are fine for python's open(); no -replace needed.
    $py = @"
import json, sys
try:
    d = json.load(open(r"$ResolvedRuns", encoding="utf-8"))
except Exception:
    print("UNKNOWN"); sys.exit(0)
a = d.get("$Arm")
if a is None:
    print("UNKNOWN"); sys.exit(0)
print(a.get("audit", {}).get("status", "UNKNOWN"))
"@
    $status = ($py | python -).Trim()
    if ([string]::IsNullOrWhiteSpace($status)) { return 'UNKNOWN' }
    return $status
}

Write-Host ""
Write-Host "===== DRY-RUN AUDIT : $(Get-Date) ====="

pixi run python scripts/run_perceptual_x0_ablation.py `
    --config $CloudPlan `
    --dry-run `
    --output $RunOutput
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "DRY-RUN 审计发现 BLOCKED 项（信息性，不中断）。"
    Write-Host "各臂 READY/BLOCKED 状态见下方逐臂审计。"
} else {
    Write-Host "DRY-RUN 全臂审计通过。开始训练循环。"
}

foreach ($Arm in $Arms) {
    $CheckpointDir = Join-Path $Project "checkpoints\perceptual_x0_$Arm"

    if (
        (Test-Path $CheckpointDir) -and
        @(Get-ChildItem $CheckpointDir -File -ErrorAction SilentlyContinue).Count -gt 0
    ) {
        Write-Host ""
        Write-Host "SKIP $Arm : 已有训练产物，拒绝覆盖 $CheckpointDir"
        continue
    }

    Write-Host ""
    Write-Host "===== AUDIT $Arm : $(Get-Date) ====="

    pixi run python scripts/run_perceptual_x0_ablation.py `
        --config $CloudPlan `
        --variant $Arm `
        --seed $Seed `
        --output $RunOutput `
        --dry-run

    $ArmStatus = Get-ArmAuditStatus $Arm
    if ($ArmStatus -ne 'READY') {
        Write-Host "SKIP $Arm : 审计状态=$ArmStatus（BLOCKED/UNKNOWN，原因见上方输出）"
        continue
    }

    if ($AuditOnly) {
        Write-Host "AUDIT-ONLY : $Arm 审计 READY（可训练），但本轮不启动训练。"
        Write-Host "            机器空闲时去掉 -AuditOnly 重跑即可真实训练。"
        continue
    }

    Write-Host "===== START $Arm : $(Get-Date) ====="

    pixi run python scripts/run_perceptual_x0_ablation.py `
        --config $CloudPlan `
        --variant $Arm `
        --seed $Seed `
        --output $RunOutput

    $ArmExit = $LASTEXITCODE
    if ($ArmExit -ne 0) {
        Write-Host "FAIL $Arm : 训练失败，退出码 $ArmExit"
        continue
    }

    Copy-Item `
        (Join-Path $RunOutput 'resolved_runs.json') `
        (Join-Path $RunOutput "resolved_runs_$Arm.json") `
        -Force

    Write-Host "===== COMPLETE $Arm : $(Get-Date) ====="
}

Write-Host ""
Write-Host 'PFM_ALL_FIVE_ARMS_DONE'
