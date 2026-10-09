param(
    [string]$Checkpoint = 'F:\Discovery AI\Second Expirement 100K RGB model\best.pth',
    [string]$DataRoot = 'F:\val to be deleted 2',
    [string]$ThresholdFile = '',
    [int]$BatchSize = 8,
    [ValidateSet('auto', 'cuda:0', 'cpu')]
    [string]$Device = 'auto'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
if ($BatchSize -lt 1) { throw 'BatchSize must be positive.' }
foreach ($path in @($pythonExe, $Checkpoint)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing file: $path" }
}
foreach ($className in @('real', 'fake')) {
    $classDir = Join-Path $DataRoot $className
    if (-not (Test-Path -LiteralPath $classDir -PathType Container)) { throw "Missing class folder: $classDir" }
}

# A supplied threshold must exist. Otherwise discover one beside this checkpoint;
# never borrow a threshold from the previous fusion experiment.
if ($ThresholdFile) {
    if (-not (Test-Path -LiteralPath $ThresholdFile -PathType Leaf)) {
        throw "Missing requested development threshold: $ThresholdFile"
    }
} else {
    $candidate = Join-Path (Split-Path -Parent $Checkpoint) 'threshold.json'
    if (Test-Path -LiteralPath $candidate -PathType Leaf) { $ThresholdFile = $candidate }
}

$tag = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$outputDir = Join-Path $repoRoot "output\external_evaluation\rgb_second_100k_$tag"
$evalArgs = @(
    '-u', (Join-Path $repoRoot 'evaluate.py'),
    '--checkpoint', $Checkpoint, '--arch', 'Wang2020_128',
    '--val_root', $DataRoot, '--split', 'external_dev',
    '--output_dir', $outputDir,
    '--batch_size', [string]$BatchSize, '--num_workers', '0',
    '--eval_precision', 'fp32', '--bootstrap', '0',
    '--no_plots', '--no_tsne', '--no_gradcam', '--no_profiling'
)
if ($ThresholdFile) {
    $evalArgs += @('--threshold_file', $ThresholdFile)
    $thresholdDescription = 'saved development threshold (checkpoint linkage checked by evaluator)'
} else {
    $thresholdDescription = 'fixed 0.5; development threshold.json was not provided'
}
if ($Device -ne 'auto') { $evalArgs += @('--device', $Device) }

Write-Host "RGB checkpoint: $Checkpoint"
Write-Host "External development dataset: $DataRoot"
Write-Host 'Labels: real=0, fake=1. No threshold fitting on this dataset.'
Write-Host "Decision rule: $thresholdDescription"
Write-Host "Reports: $outputDir"
Push-Location -LiteralPath $repoRoot
try {
    & $pythonExe @evalArgs
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed with exit code $LASTEXITCODE. Inspect the error above." }
} finally {
    Pop-Location
}

$reportPath = Join-Path $outputDir 'Wang2020_128\generalization_report.json'
$report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
$overall = $report.overall
Write-Host "`nRESULTS ($thresholdDescription)"
Write-Host ("Images: {0} | Real: {1} | Fake: {2}" -f $overall.samples, $overall.real, $overall.fake)
Write-Host ("Accuracy:          {0:P2}" -f $overall.accuracy)
Write-Host ("Balanced accuracy: {0:P2}" -f $overall.balanced_accuracy)
Write-Host ("ROC AUC:           {0:F4}" -f $overall.roc_auc)
Write-Host ("Fake precision:    {0:P2}" -f $overall.per_class.fake.precision)
Write-Host ("Fake recall:       {0:P2}" -f $overall.per_class.fake.recall)
Write-Host ("Real recall:       {0:P2}" -f $overall.per_class.real.recall)
Write-Host ("Fake F1:           {0:F4}" -f $overall.per_class.fake.f1)
Write-Host ("Macro F1:          {0:F4}" -f $overall.macro_f1)
Write-Host "Full report: $reportPath"
