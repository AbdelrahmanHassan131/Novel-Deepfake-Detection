# RGB pilot: corrected execution guide

Updated 2026-10-08. Commands below are prepared for the user; they have **not been run during this review**. No GPU speedup or unseen accuracy has been established by these code changes.

## 1. What this run can establish

Your recovered manifest contains 98,590 training and 209,871 development images. It can be reused without inventory/hashing again while the dataset and verified preparation artifacts remain unchanged. Its aggregate `diffgan` source metadata does not establish source-held-out generalization. The commands below explicitly label that run as an engineering pilot. Keep the preparation directory, including the manifest, verification sidecar, audit/recovery reports and conflict exclusions.

Do not rename `diffgan` to a made-up source. For a scientific source-held-out pilot, prepare a separate manifest with verified original-source/generator/group metadata, hold out declared development sources from training, preserve final-test components, and complete its existing preparation/audit gate. Dataset provenance cannot be reconstructed from the code alone. Keep previously inspected `F:\val to be deleted 2` as external DEVELOPMENT data; reserve a new untouched final test.

## 2. Setup cell (once per new run)

Use your updated repository and restore the existing prepared-data folder in Kaggle. Change paths if mounted elsewhere. R2 is a candidate, not a proven winner. Start with the existing 100K cohort; do not expand to 1M until development comparisons justify it.

```python
import os, json
from pathlib import Path
from uuid import uuid4

GPU_COUNT = 2                 # 2 for Kaggle 2xT4; 1 for one T4 or L4
VARIANT = "R2"               # R0 / R1 / R2 / R3; new run for each
SCIENCE_MODE = "engineering" # "heldout" only with verified source metadata
ELIGIBLE_SOURCES = ""         # comma-separated held-out dev sources in heldout mode
DATA_ROOT = "/kaggle/input/datasets/abdelrahmanhassani/prepareddatasetdiffgan/Preapred Dataset"
MANIFEST = "/kaggle/working/prepared_data/recovered_100k_seed42_v2/selected_manifest.csv"
policies = {
    "R0": ("128d", "full", "train", "1.0", "legacy", "0.5"),
    "R1": ("128d", "full", "train", "1.0", "rgb_v1", "0.5"),
    "R2": ("128d", "layer4_and_head", "frozen", "0.1", "rgb_v1", "0.5"),
    "R3": ("linear", "head_only", "frozen", "0.0", "rgb_v1", "0.0"),
}
head, policy, bn, multiplier, recipe, dropout = policies[VARIANT]
run_id = "seed42_" + uuid4().hex[:8]
values = dict(
    REPO_ROOT="/kaggle/working/Novel-Deepfake-Detection",
    DATA_ROOT=DATA_ROOT, MANIFEST=MANIFEST, VAL_MANIFEST=MANIFEST,
    TRAIN_SPLIT="train", DEV_SPLIT="dev", GPU_COUNT=str(GPU_COUNT),
    GPU_IDS="0,1" if GPU_COUNT == 2 else "0",
    BATCH="16", ACCUM="2" if GPU_COUNT == 2 else "4", WORKERS="2",
    OUTPUT_BASE="/kaggle/working/rgb_pilots", RUN_ID=run_id,
    RUN_NAME="rgb_" + VARIANT.lower(), RGB_HEAD=head, RGB_DROPOUT=dropout,
    FINE_TUNE=policy, BN_POLICY=bn, LR_MULT=multiplier, AUG_RECIPE=recipe,
    SCIENCE_MODE=SCIENCE_MODE, ELIGIBLE_SOURCES=ELIGIBLE_SOURCES,
)
assert GPU_COUNT in (1, 2)
assert Path(MANIFEST).is_file(), "Restore the prepared manifest and its sidecar first"
run_dir = Path(values["OUTPUT_BASE"]) / (values["RUN_NAME"] + "_" + run_id)
# ExperimentManager creates run_dir itself; do not create it in advance.
Path(values["OUTPUT_BASE"]).mkdir(parents=True, exist_ok=True)
values["RGB_RUN_DIR"] = str(run_dir)
values["RGB_CKPT"] = str(run_dir / "checkpoints/best.pth")
os.environ.pop("RESUME_CHECKPOINT", None)  # new runs must not inherit a resume request
os.environ.update(values)
settings = Path(values["OUTPUT_BASE"]) / (run_id + "_session.json")
settings.write_text(json.dumps(values, indent=2))
print("Save this settings file:", settings)
print("Expected run:", run_dir)
```

R0 uses the historical blur-only settings below (`0.5`, sigma `0–3`), with JPEG off. R1 adds the named blur/JPEG recipe. R2 changes fine-tuning/BN/LR policy. R3 is a diagnostic linear probe and cannot replace a 128D fusion expert.

## 3. Bounded checks before training

Run the regression checks first in the environment where dependencies are installed. These include CPU synthetic fixtures, not the full image dataset. Stop on failures.

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
python -m pytest -q tests/test_rgb_review_final_repairs.py tests/test_versioned_augmentations.py tests/test_calibration_and_source_readiness.py tests/test_round1_phase3_repairs.py
python tools/diagnose_rgb_parity.py --synthetic --arch Wang2020_128 --output_report "$OUTPUT_BASE/synthetic_parity.json"
torchrun --standalone --nproc_per_node="$GPU_COUNT" tools/verify_ddp_environment.py
```

Then run a bounded benchmark. It uses the production training step with random initialization and does not save trained checkpoints. It reads manifest metadata and probes training paths; only the requested batches are decoded. R0's legacy benchmark is unaugmented, so use R1/R2/R3 for recipe-matched timing.

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node="$GPU_COUNT" tools/benchmark_training_throughput.py \
  --manifest "$MANIFEST" --val_manifest "$VAL_MANIFEST" --dataroot "$DATA_ROOT" \
  --gpu_ids "$GPU_IDS" --batch_size "$BATCH" --grad_accum_steps "$ACCUM" \
  --num_workers "$WORKERS" --prefetch_factor 2 --persistent_workers --pin_memory \
  --use_amp --amp_dtype fp16 --aug_recipe "$AUG_RECIPE" \
  --fine_tune_policy "$FINE_TUNE" --bn_policy "$BN_POLICY" \
  --rgb_head_type "$RGB_HEAD" --rgb_dropout "$RGB_DROPOUT" --backbone_lr_mult "$LR_MULT" \
  --warmup_microbatches 20 --measure_microbatches 100 --val_samples 500 \
  --output "$OUTPUT_BASE/${RUN_ID}_benchmark.json"
```

Compare workers 0, 2 and 4 **per GPU**, and batch 16 versus 32 if memory permits. More workers can make a CPU-limited notebook slower. Preserve effective batch `batch × GPUs × accumulation` when comparing speed (64 for defaults). For two GPUs, try 32×2×1; for one GPU, 32×1×2. Choose by measured images/second and memory, not a utilization screenshot. The benchmark buffers one accumulation window and reports that limitation. End-to-end time also includes the large development set, startup checks and checkpoint exports.

## 4. Train the selected RGB candidate

The same cell supports one or two GPUs using the setup values. Both T4 and L4 can use FP16. Pretrained backbone download requires internet or cached weights. To use a local backbone, add `--backbone_weights /path/to/resnet50.pth` while keeping `--pretrained`.

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
source_args=()
if [[ "$SCIENCE_MODE" == "engineering" ]]; then
  source_args=(--allow_aggregate_sources --allow_source_overlap --monitor_metric auc)
elif [[ "$SCIENCE_MODE" == "heldout" ]]; then
  : "${ELIGIBLE_SOURCES:?Declare verified held-out development sources}"
  source_args=(--monitor_metric source_macro_auc --eligible_sources "$ELIGIBLE_SOURCES")
else
  echo "Unknown SCIENCE_MODE" >&2; exit 1
fi
aug_args=()
if [[ "$AUG_RECIPE" == "legacy" ]]; then
  aug_args=(--blur_prob 0.5 --blur_sig 0.0,3.0 --jpg_prob 0)
fi
resume_args=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  resume_args=(--continue_train --resume_checkpoint "$RESUME_CHECKPOINT")
fi
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node="$GPU_COUNT" train.py \
  --arch Wang2020_128 --name "$RUN_NAME" --run_id "$RUN_ID" --checkpoints_dir "$OUTPUT_BASE" \
  --dataroot "$DATA_ROOT" --manifest "$MANIFEST" --manifest_split "$TRAIN_SPLIT" \
  --val_manifest "$VAL_MANIFEST" --val_manifest_split "$DEV_SPLIT" \
  --rgb_head_type "$RGB_HEAD" --rgb_dropout "$RGB_DROPOUT" \
  --fine_tune_policy "$FINE_TUNE" --bn_policy "$BN_POLICY" --backbone_lr_mult "$LR_MULT" \
  --aug_recipe "$AUG_RECIPE" "${aug_args[@]}" \
  --gpu_ids "$GPU_IDS" --batch_size "$BATCH" --grad_accum_steps "$ACCUM" \
  --val_batch_size 32 --num_workers "$WORKERS" --prefetch_factor 2 --persistent_workers --pin_memory \
  --epochs 5 --epochs_decay 0 --optim adam --lr 0.0001 --lr_policy cosine \
  --use_amp --amp_dtype fp16 --val_precision fp32 --seed 42 --pretrained \
  --early_stopping --early_stopping_patience 2 --early_stopping_min_epochs 2 \
  --save_epoch_freq 0 "${source_args[@]}" "${resume_args[@]}"
```

For an interrupted run, restore its saved session JSON into `os.environ`, set `RESUME_CHECKPOINT` to its `checkpoints/last.pth`, and rerun the same training cell. Do not regenerate a run ID or change model/augmentation/scheduler policies during resume. `epochs` means the total target, not additional epochs. Old one-group full-model optimizers retain their original grouping; arbitrary group/policy changes are rejected.

## 5. Calibrate from this checkpoint's saved development predictions

This deliberately reuses checkpoint-selection development data; the artifact reports that limitation. Do not call it independent calibration. The full independent-cohort workflow requires additional bound membership evidence and is not certified by these commands.

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
python analyze_predictions.py calibrate \
  --predictions "$RGB_RUN_DIR/checkpoints/best_dev_predictions.csv" \
  --output "$RGB_RUN_DIR/checkpoints/threshold.json"
```

## 6. Evaluate external development data after downloading

Save the **whole RGB run folder**, session JSON and prepared-data artifacts. Keep `best.pth`, its linked prediction CSV and `threshold.json` together. For local PowerShell, set the exact downloaded run directory below. This FP32 command uses your existing external DEVELOPMENT folder and writes a new result directory. It does not recalibrate on that folder.

```powershell
$RgbRun = Read-Host 'Paste the downloaded RGB run folder (containing checkpoints)'
$ReportDir = Join-Path $RgbRun ('external_dev_' + [guid]::NewGuid().ToString('N'))
& .\.venv\Scripts\python.exe evaluate.py `
  --checkpoint (Join-Path $RgbRun 'checkpoints/best.pth') --arch Wang2020_128 `
  --val_root 'F:\val to be deleted 2' --split external_dev `
  --threshold_file (Join-Path $RgbRun 'checkpoints/threshold.json') `
  --batch_size 8 --num_workers 0 --device cuda:0 --eval_precision fp32 `
  --output_dir $ReportDir --no_plots --no_tsne --no_gradcam --no_profiling --bootstrap 0
if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed; inspect the error before continuing.' }
```

Results are under `$ReportDir/Wang2020_128/generalization_report.json`. Compare ROC AUC, balanced accuracy, real/fake recall and score distributions at the fixed development threshold. Folder-only evaluation has unknown groups, so it cannot justify group-bootstrap confidence intervals. Use a verified external manifest for those.

## 7. What to inspect and save

- `opt.json`: confirm the actual RGB head, fine-tuning, BN, augmentation and LR settings before trusting the run.
- `checkpoints/best.pth`, `last.pth`, `best_selection_metadata.json`, `best_dev_predictions.csv`, `threshold.json`: preserve their checkpoint linkage.
- `checkpoints/performance_rank0.json` and rank1 when present: cumulative startup/loading/training/validation/export timing and bounded CUDA phase samples.
- `metrics.csv`, `steps.csv`, external reports and predictions.

Readiness remains conditional on these user-run checks. Neither full GPU utilization nor >85% unseen accuracy can be promised from code review.
