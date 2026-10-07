# Evaluate the downloaded 100K fusion model on Windows

Run this in **PowerShell**, not a Kaggle notebook. It evaluates all supported images under `F:\val to be deleted 2\real` and `F:\val to be deleted 2\fake`, including subfolders. It does not train or modify the checkpoints or images.

```powershell
& "G:\Master's Of Science Computer Engineering\thesis deepfake detection\Thesis Revisions SP\First revesion\Novel Deepfake Detection\tools\evaluate_downloaded_100k.ps1"
```

The script uses the repository's existing `.venv\Scripts\python.exe`. That environment must have the project's evaluation dependencies installed. It automatically uses CUDA if supported by the installed PyTorch; otherwise it uses CPU. CPU evaluation can take much longer. An inference progress bar appears after checkpoint loading and dataset setup.

If GPU memory is insufficient, rerun with a smaller batch:

```powershell
& "G:\Master's Of Science Computer Engineering\thesis deepfake detection\Thesis Revisions SP\First revesion\Novel Deepfake Detection\tools\evaluate_downloaded_100k.ps1" -BatchSize 2
```

To explicitly use CPU, append `-Device cpu`. Each run creates a fresh output folder.

## Verified input paths

Root: `F:\Discovery AI\First expirement 100K models`

| Input | Path relative to that root |
| --- | --- |
| Fusion checkpoint | `train_100000_seed42_06482e0771\token_seed42\checkpoints\best.pth` |
| Development threshold | `train_100000_seed42_06482e0771\token_seed42\checkpoints\threshold.json` |
| RGB checkpoint | `train_100000_seed42_27cf24839a\rgb128_seed42\checkpoints\best.pth` |
| Wavelet checkpoint | `train_100000_seed42_f5c32ae782\wavelet128_seed42\checkpoints\best.pth` |

The fusion run's saved options reference these same expert runs. Its saved threshold is `0.26453450322151184`. Evaluation checks that the threshold's checkpoint hash matches the loaded fusion checkpoint. The threshold is not fitted again on the external images.

## Results

Under the repository, each run writes:

```text
output/external_evaluation/token_100k_<timestamp>_<id>/MHA_128/
    generalization_report.json
    evaluation_report.json
    evaluation_report.md
    predictions.csv
    plots/
```

The console summary includes accuracy, balanced accuracy, ROC AUC, fake precision/recall/F1, real recall, and macro F1. The reports also include per-class metrics, confusion matrix, false positive/negative rates, and EER. ROC, precision-recall, and confusion-matrix plots are enabled. Grad-CAM, t-SNE, and profiling are disabled for this metrics run.

Folder labels are interpreted as **real=0, fake=1**. The evaluator restores preprocessing and fusion configuration from the checkpoint. The two-stream fusion checkpoint can contain embedded expert weights; the explicit local expert paths also replace the saved Kaggle paths.

This folder-based evaluation reports image-level performance. It does not verify training/test overlap, near-duplicates, or identity/video independence. Source and group metadata remain unknown; confidence intervals are not requested because verified independent groups are unavailable. If this is the same dataset previously used to investigate the 60% result, treat it as external development evidence and retain another untouched dataset for the final paper evaluation.

Only paths, saved JSON metadata, and source code were inspected while preparing this command. No checkpoint was loaded and no inference was run locally by the assistant. Send `generalization_report.json` after your run for interpretation.

## Windows worker startup fix

The error `Can't pickle local object 'FusionDataset.__init__.<locals>.<lambda>'` is caused by local transform functions being sent to Windows DataLoader workers. The RGB, wavelet, and fusion loaders now use module-level functions and `functools.partial` with the same preprocessing operations. Rerun the same command from this updated checkout; no retraining or recalibration is needed. The trailing worker `EOFError` follows the failed parent serialization.

A regression test covers dataset serialization and comparison of spawned-worker output against single-process output on synthetic images. It has been authored but not run; only static syntax checks were performed locally.

## Diagnose the failed external result before retraining

The first external fusion run obtained 50.22% accuracy and 0.8177 AUC, recognizing only 131 of 30,000 real images. A fixed 0.5 threshold on its saved scores gives 50.38% accuracy. This is not resolved by simply reverting to 0.5. It does not prove the project cannot improve, nor establish an attainable 85% accuracy.

Run the two existing experts separately:

```powershell
& "G:\Master's Of Science Computer Engineering\thesis deepfake detection\Thesis Revisions SP\First revesion\Novel Deepfake Detection\tools\diagnose_100k_experts.ps1"
```

This runs two image-evaluation passes, with no training and no repeat fusion inference. Each expert gets its own threshold selected solely from its saved development predictions; the evaluator verifies the threshold/checkpoint match. Results are saved under `output/external_evaluation/expert_diagnosis_<timestamp>_<id>`, including `expert_summary.json`. Values in that summary are fractions, so accuracy 0.80 means 80%. The script accepts `-BatchSize 2` or `-Device cpu` as needed. Syntax was checked; inference was not executed by the assistant.

Decide the next experiment from those results:

- If an expert substantially outperforms fusion, investigate fusion behavior and compare a simpler combination using development data for selection.
- If both experts also fail, prioritize verifying preprocessing, label/source metadata, and training-source coverage before changing the fusion architecture.
- If ranking remains useful but scores shift across datasets, investigate calibration on separately designated external development sources. Freeze all choices before evaluating another untouched dataset.

Any next training pilot should have a stated hypothesis, known source composition, and a source-held-out development split. Keep a separate final test dataset. Do not assume more images from the same pool will resolve this observed failure.
