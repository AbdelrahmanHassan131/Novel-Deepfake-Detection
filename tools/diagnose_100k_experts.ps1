param(
    [int]$BatchSize = 8,
    [ValidateSet('auto', 'cuda:0', 'cpu')]
    [string]$Device = 'auto'
)

$ErrorActionPreference = 'Stop'
if ($BatchSize -lt 1) { throw 'BatchSize must be positive.' }
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
$experimentRoot = 'F:\Discovery AI\First expirement 100K models'
$testRoot = 'F:\val to be deleted 2'
$experts = @(
    @{ Name = 'RGB'; Arch = 'Wang2020_128'; Relative = 'train_100000_seed42_27cf24839a\rgb128_seed42\checkpoints' },
    @{ Name = 'Wavelet'; Arch = 'WolterWavelet2021_128'; Relative = 'train_100000_seed42_f5c32ae782\wavelet128_seed42\checkpoints' }
)
if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) { throw "Missing Python: $pythonExe" }
foreach ($className in @('real', 'fake')) {
    if (-not (Test-Path -LiteralPath (Join-Path $testRoot $className) -PathType Container)) {
        throw "Missing test class folder: $className"
    }
}
foreach ($expert in $experts) {
    $expert.Directory = Join-Path $experimentRoot $expert.Relative
    foreach ($fileName in @('best.pth', 'dev_predictions.csv')) {
        $filePath = Join-Path $expert.Directory $fileName
        if (-not (Test-Path -LiteralPath $filePath -PathType Leaf)) { throw "Missing file: $filePath" }
    }
}
$runTag = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$outputDir = Join-Path $repoRoot "output\external_evaluation\expert_diagnosis_$runTag"
$null = New-Item -ItemType Directory -Path $outputDir
$summary = @()
Write-Host "Results: $outputDir"
Write-Host 'Evaluating two existing experts; no training or external-threshold fitting.'
Push-Location -LiteralPath $repoRoot
try {
    foreach ($expert in $experts) {
        $threshold = Join-Path $outputDir ($expert.Name + '_dev_threshold.json')
        Write-Host ("`nSelecting {0} threshold from saved DEVELOPMENT predictions..." -f $expert.Name)
        & $pythonExe -u (Join-Path $repoRoot 'analyze_predictions.py') calibrate --predictions (Join-Path $expert.Directory 'dev_predictions.csv') --output $threshold
        if ($LASTEXITCODE -ne 0) { throw "Development calibration failed for $($expert.Name)." }
        $evalArgs = @(
            '-u', (Join-Path $repoRoot 'evaluate.py'),
            '--checkpoint', (Join-Path $expert.Directory 'best.pth'),
            '--arch', $expert.Arch, '--dataroot', $testRoot,
            '--threshold_file', $threshold, '--output_dir', $outputDir,
            '--batch_size', [string]$BatchSize, '--bootstrap', '0',
            '--no_tsne', '--no_gradcam', '--no_profiling'
        )
        if ($Device -ne 'auto') { $evalArgs += @('--device', $Device) }
        Write-Host ("Evaluating {0} on external images..." -f $expert.Name)
        & $pythonExe @evalArgs
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $($expert.Name). Earlier reports are preserved in $outputDir" }
        $reportPath = Join-Path $outputDir ($expert.Arch + '\generalization_report.json')
        $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        $m = $report.overall
        $summary += [pscustomobject]@{
            Model = $expert.Name; Samples = $m.samples; Threshold = $m.threshold
            Accuracy = $m.accuracy; BalancedAccuracy = $m.balanced_accuracy
            AUC = $m.roc_auc; RealRecall = $m.per_class.real.recall
            FakeRecall = $m.per_class.fake.recall; MacroF1 = $m.macro_f1
            CheckpointSha256 = $report.checkpoint.checkpoint_sha256
            Report = $reportPath
        }
        $summary | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $outputDir 'expert_summary.json') -Encoding UTF8
    }
} finally {
    Pop-Location
}
$summary | Format-Table Model, Samples, Accuracy, BalancedAccuracy, AUC, RealRecall, FakeRecall -AutoSize
Write-Host "Send expert_summary.json from: $outputDir"
