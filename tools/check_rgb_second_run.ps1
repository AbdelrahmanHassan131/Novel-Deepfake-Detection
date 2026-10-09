param(
    [string]$Checkpoint = 'F:\Discovery AI\Second Expirement 100K RGB model\best.pth',
    [ValidateSet('cpu', 'cuda:0')]
    [string]$Device = 'cuda:0'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
$inputs = Join-Path $repo 'output\review\rgb_r2_diagnostic_inputs'
foreach ($file in @($python, $Checkpoint, (Join-Path $inputs 'dev_replay_64.csv'),
                    (Join-Path $inputs 'saved_dev_predictions_64.csv'),
                    (Join-Path $inputs 'train_profile_256.csv'),
                    (Join-Path $inputs 'external_profile_256.csv'))) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing diagnostic input: $file" }
}
$tag = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$output = Join-Path $repo "output\review\rgb_diagnosis_$tag"
New-Item -ItemType Directory -Path $output | Out-Null
Write-Host 'Replaying 64 original development images against the saved Colab probabilities.'
Write-Host 'The diagnostic verifies their image hashes before model inference. No retraining.'
Push-Location -LiteralPath $repo
try {
    & $python -u tools/diagnose_rgb_parity.py --checkpoint $Checkpoint --arch Wang2020_128 `
      --manifest (Join-Path $inputs 'dev_replay_64.csv') --manifest_split dev `
      --saved_predictions (Join-Path $inputs 'saved_dev_predictions_64.csv') `
      --device $Device --max_samples 64 --tolerance 0.0001 `
      --output_report (Join-Path $output 'dev_replay.json')
    if ($LASTEXITCODE -ne 0) {
        $reportPath = Join-Path $output 'dev_replay.json'
        if (-not (Test-Path -LiteralPath $reportPath)) {
            throw 'Replay failed before producing a report. Resolve this error before continuing.'
        }
        $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        $otherFailures = @($report.general_discrepancies | Where-Object { $_ -notlike 'Saved probability mismatch:*' })
        if ($otherFailures.Count -gt 0 -or $report.max_tensor_diff -gt 0.0001 -or
            $report.max_probability_diff -gt 0.0001 -or $report.max_logit_diff -gt 0.0001) {
            throw 'Local-path or identity checks failed. Inspect dev_replay.json before continuing.'
        }
        Write-Warning 'Saved Colab probabilities differ beyond tolerance. This remains unresolved; image profiling can still proceed independently.'
    }
    foreach ($cohort in @('train', 'external')) {
        & $python -u tools/profile_image_cohort.py `
          --manifest (Join-Path $inputs ($cohort + '_profile_256.csv')) `
          --sample_size 256 --seed 42 --output (Join-Path $output ($cohort + '_profile.json'))
        if ($LASTEXITCODE -ne 0) { throw "Bounded $cohort image profile failed." }
    }
} finally {
    Pop-Location
}
Write-Host "Diagnostics completed: $output"
Write-Host 'Profiles are deliberately stratified diagnostic samples, not population-frequency estimates.'
Write-Host 'No test-fitted threshold, new model, or improved generalization result was produced.'
Write-Host 'Read dev_replay.json for replay status; completion of profiling does not imply replay passed.'
