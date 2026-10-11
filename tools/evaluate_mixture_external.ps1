param(
    [int]$BatchSize = 8,
    [ValidateSet('auto', 'cuda:0', 'cpu')][string]$Device = 'auto',
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) { throw "Missing Python: $pythonExe" }
$evalArgs = @(
    '-u', (Join-Path $PSScriptRoot 'evaluate_mixture_external.py'),
    '--report', 'F:\Discovery AI\Evaluation arm a and arm b report\mixture_evaluation_report.json',
    '--arm_a', 'F:\Discovery AI\Arm a expirement RGB model\best.pth',
    '--arm_b', 'F:\Discovery AI\arm b expirement RGB model\best.pth',
    '--data_root', 'F:\val to be deleted 2',
    '--batch_size', [string]$BatchSize, '--device', $Device
)
if ($CheckOnly) { $evalArgs += '--check_only' }
& $pythonExe @evalArgs
if ($LASTEXITCODE -ne 0) { throw "Pilot evaluation failed with exit code $LASTEXITCODE." }
