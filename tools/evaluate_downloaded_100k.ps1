param(
    [int]$BatchSize = 8,
    [ValidateSet('auto', 'cuda:0', 'cpu')]
    [string]$Device = 'auto'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
$experimentRoot = 'F:\Discovery AI\First expirement 100K models'
$testRoot = 'F:\val to be deleted 2'
$fusionDir = Join-Path $experimentRoot 'train_100000_seed42_06482e0771\token_seed42\checkpoints'
$fusionCheckpoint = Join-Path $fusionDir 'best.pth'
$thresholdFile = Join-Path $fusionDir 'threshold.json'
$rgbCheckpoint = Join-Path $experimentRoot 'train_100000_seed42_27cf24839a\rgb128_seed42\checkpoints\best.pth'
$waveletCheckpoint = Join-Path $experimentRoot 'train_100000_seed42_f5c32ae782\wavelet128_seed42\checkpoints\best.pth'

if ($BatchSize -lt 1) { throw 'BatchSize must be positive.' }
foreach ($path in @($pythonExe, $fusionCheckpoint, $thresholdFile, $rgbCheckpoint, $waveletCheckpoint)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing file: $path" }
}
foreach ($className in @('real', 'fake')) {
    $classDir = Join-Path $testRoot $className
    if (-not (Test-Path -LiteralPath $classDir -PathType Container)) { throw "Missing class folder: $classDir" }
}

$runTag = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$outputDir = Join-Path $repoRoot "output\external_evaluation\token_100k_$runTag"
$evalArgs = @(
    '-u', (Join-Path $repoRoot 'evaluate.py'),
    '--checkpoint', $fusionCheckpoint,
    '--arch', 'MHA_128',
    '--dataroot', $testRoot,
    '--rgb_model_path', $rgbCheckpoint,
    '--wavelet_model_path', $waveletCheckpoint,
    '--threshold_file', $thresholdFile,
    '--output_dir', $outputDir,
    '--batch_size', [string]$BatchSize,
    '--bootstrap', '0',
    '--no_tsne', '--no_gradcam', '--no_profiling'
)
if ($Device -ne 'auto') { $evalArgs += @('--device', $Device) }

Write-Host "Evaluating saved fusion checkpoint on: $testRoot"
Write-Host 'Labels: real=0, fake=1. Using the saved development threshold.'
Write-Host "Results will be saved in: $outputDir"
Push-Location -LiteralPath $repoRoot
try {
    & $pythonExe @evalArgs
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed with exit code $LASTEXITCODE. See the error above." }
} finally {
    Pop-Location
}

$reportPath = Join-Path $outputDir 'MHA_128\generalization_report.json'
$report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
$overall = $report.overall
Write-Host "`nRESULTS (fixed development threshold)"
Write-Host ("Images: {0} | Real: {1} | Fake: {2}" -f $overall.samples, $overall.real, $overall.fake)
Write-Host ("Accuracy:          {0:P2}" -f $overall.accuracy)
Write-Host ("Balanced accuracy: {0:P2}" -f $overall.balanced_accuracy)
Write-Host ("ROC AUC:           {0:F4}" -f $overall.roc_auc)
Write-Host ("Fake precision:    {0:P2}" -f $overall.per_class.fake.precision)
Write-Host ("Fake recall:       {0:P2}" -f $overall.per_class.fake.recall)
Write-Host ("Fake F1:           {0:F4}" -f $overall.per_class.fake.f1)
Write-Host ("Real recall:       {0:P2}" -f $overall.per_class.real.recall)
Write-Host ("Macro F1:          {0:F4}" -f $overall.macro_f1)
Write-Host "Full report: $reportPath"
