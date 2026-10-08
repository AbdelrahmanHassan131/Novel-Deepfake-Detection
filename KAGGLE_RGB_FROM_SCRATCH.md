# Kaggle RGB from scratch
UNEXECUTED workflow. Enable T4 x2 (or one GPU), Internet and attach the original image dataset. Push latest local changes to GeneralizationDeepFakeDetection before cloning; this review did not push them.

This rebuilds nominally 100K training + 15K development images. Inventory lists the full image tree; content hashing reads selected images only. Existing folder splits are preserved. Cleanup may reduce counts. The smaller dev cohort differs from the old 209K dev cohort; scores are not directly comparable.

Per-image grouping is explicitly UNVERIFIED. This is an engineering pilot, not proof of source/video/identity independence. Exact hashes cannot exclude all near-duplicates. No old manifest/checkpoint is required. Stop on unexpected errors. Download preparation before training.

## 1. Clone

Once in the fresh notebook.

```bash
%%bash
set -euo pipefail
git clone --branch GeneralizationDeepFakeDetection \
  https://github.com/AbdelrahmanHassan131/Novel-Deepfake-Detection.git \
  /kaggle/working/Novel-Deepfake-Detection
cd /kaggle/working/Novel-Deepfake-Detection
git log -1 --oneline
test -f tests/test_rgb_review_final_repairs.py
python - <<'PY'
from pathlib import Path
assert "--image_level_groups" in Path("prepare_dataset.py").read_text(), "Push latest fresh-start changes first."
assert "label_conflict =" in Path("tools/recover_prepared_manifest.py").read_text(), "Recovery fix missing."
PY
```

## 2. Dependencies

Keep Kaggle's installed PyTorch/torchvision pair.

```python
%pip install -q pytest pytorch-wavelets PyWavelets tensorboard
```

## 3. Setup

Run once for a new experiment. GPU_COUNT=1 selects one GPU. Keep dataset/recipe/seed fixed when reusing this preparation directory. Do not regenerate the run ID to resume training.

```python
import os, json
from pathlib import Path
from uuid import uuid4

GPU_COUNT = 2                 # 2 for Kaggle 2xT4; 1 for one T4 or L4
VARIANT = "R2"               # R0 / R1 / R2 / R3; new run for each
SCIENCE_MODE = "engineering" # "heldout" only with verified source metadata
ELIGIBLE_SOURCES = ""         # comma-separated held-out dev sources in heldout mode
DATA_ROOT = "/kaggle/input/datasets/abdelrahmanhassani/prepareddatasetdiffgan/Preapred Dataset"
PREP_ROOT = Path("/kaggle/working/prepared_data/rgb_100k_dev15k_seed42")
PREP_ROOT.mkdir(parents=True, exist_ok=True)
MANIFEST = str(PREP_ROOT / "ready/selected_manifest.csv")
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
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() >= GPU_COUNT
assert Path(DATA_ROOT).is_dir(), f"Attach dataset at {DATA_ROOT}"
for i in range(GPU_COUNT):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
values["PREP_ROOT"] = str(PREP_ROOT)
values["POOL_MANIFEST"] = str(PREP_ROOT / "pool_manifest.csv")
values["PILOT_MANIFEST"] = str(PREP_ROOT / "pilot_full_dev.csv")
values["CANDIDATE_MANIFEST"] = str(PREP_ROOT / "candidate_100k_dev15k.csv")
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

## 4. Code and GPU checks

No full image scan. These runtime checks were NOT RUN locally; fix failures before continuing.

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
python -m pytest -q tests/test_prepared_manifest_recovery.py tests/test_rgb_review_final_repairs.py \
  tests/test_versioned_augmentations.py tests/test_calibration_and_source_readiness.py \
  tests/test_round1_phase3_repairs.py
python -u tools/diagnose_rgb_parity.py --synthetic --arch Wang2020_128 \
  --output_report "$OUTPUT_BASE/${RUN_ID}_synthetic_parity.json"
torchrun --standalone --nproc_per_node="$GPU_COUNT" tools/verify_ddp_environment.py
```

## 5. Inventory and 100K selection

No images copied. Directory enumeration takes time. Existing train/val partitions are preserved. Retry reuses the manifests below; keep dataset/settings unchanged.

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
if [[ ! -f "$POOL_MANIFEST" ]]; then
  python -u prepare_dataset.py inventory \
    --root "$DATA_ROOT" --source diffgan --output "$POOL_MANIFEST" --image_level_groups
fi
if [[ ! -f "$PILOT_MANIFEST" ]]; then
  python -u prepare_dataset.py pilot \
    --manifest "$POOL_MANIFEST" --root "$DATA_ROOT" --output "$PILOT_MANIFEST" \
    --train_size 100000 --frame_cap 0 \
    --dev_ratio 0.1 --test_ratio 0.1 --require_groups --seed 42
fi
```

## 6. Choose 15K development rows

Metadata only. Unused dev rows never enter training; original manifests remain saved. Known connected groups stay together. Unknown relationships cannot be inferred.

```python
import os, sys, json
from pathlib import Path
if os.environ["REPO_ROOT"] not in sys.path:
    sys.path.insert(0, os.environ["REPO_ROOT"])
from data.manifest import read_manifest, write_manifest, select_representative_dev_cohort, clear_manifest_cache
candidate = Path(os.environ["CANDIDATE_MANIFEST"])
if candidate.exists():
    print("Reusing:", candidate)
else:
    rows = read_manifest(os.environ["PILOT_MANIFEST"], root=os.environ["DATA_ROOT"], check_files=False)
    train_rows = [r for r in rows if r["split"] == "train"]
    dev_rows = [r for r in rows if r["split"] == "dev"]
    assert len(train_rows) == 100000, f"Unexpected train size: {len(train_rows)}"
    selected_dev, summary = select_representative_dev_cohort(dev_rows, target_size=15000, seed=42)
    assert {int(r["label"]) for r in selected_dev} == {0, 1}
    write_manifest(candidate, train_rows + selected_dev)
    (Path(os.environ["PREP_ROOT"]) / "dev_selection_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Candidate: {len(train_rows):,} train + {len(selected_dev):,} dev")
    del rows, train_rows, dev_rows, selected_dev
    clear_manifest_cache()
```

## 7. Hash selected images and quarantine

Main one-time cost: roughly 115K files, not the full million. Progress prints every 5,000 rows; keep the periodically saved SQLite cache after interruptions.

Only a completed matching hash report can feed recovery. Audit exit 1 is handled for supported overlaps/label contradictions; other errors stop. Recovery excludes ambiguous components without guessing labels, removes training overlap with dev and records exclusions/counts. immutable_dataset means mounted images remain unchanged between audit and recovery.

```python
import os, sys, json, hashlib, subprocess
from pathlib import Path
from uuid import uuid4
from data.manifest import verify_manifest_gate
prep = Path(os.environ["PREP_ROOT"])
ready = prep / "ready"
if (ready / "selected_manifest.verified.json").exists():
    verify_manifest_gate(ready / "selected_manifest.csv", require_hashes=True)
    print("Already prepared:", ready)
else:
    report_path = prep / ("audit_" + uuid4().hex[:10] + ".json")
    result = subprocess.run([
        sys.executable, "-u", "prepare_dataset.py", "audit",
        "--manifest", os.environ["CANDIDATE_MANIFEST"],
        "--root", os.environ["DATA_ROOT"], "--output", str(report_path),
        "--hashes", "--hash_cache", str(prep / "hashes.sqlite")
    ], cwd=os.environ["REPO_ROOT"])
    if result.returncode not in (0, 1) or not report_path.is_file():
        raise RuntimeError("Audit did not complete. Fix error; retain hashes.sqlite for retry.")
    report = json.loads(report_path.read_text())
    digest = hashlib.sha256(Path(os.environ["CANDIDATE_MANIFEST"]).read_bytes()).hexdigest()
    if report.get("hashes_verified") is not True or report.get("manifest_sha256") != digest:
        raise RuntimeError("Audit evidence is incomplete or mismatched.")
    subprocess.run([
        sys.executable, "-u", "tools/recover_prepared_manifest.py",
        "--manifest", os.environ["CANDIDATE_MANIFEST"],
        "--audit_report", str(report_path), "--output_dir", str(ready),
        "--immutable_dataset", "--allow_shortfall", "--quarantine_label_conflicts"
    ], cwd=os.environ["REPO_ROOT"], check=True)
    verify_manifest_gate(ready / "selected_manifest.csv", require_hashes=True)
os.environ["MANIFEST"] = str(ready / "selected_manifest.csv")
os.environ["VAL_MANIFEST"] = os.environ["MANIFEST"]
summary = json.loads((ready / "recovery_report.json").read_text())
print("FINAL COHORT:", summary["after"])
print("Training images:", summary["actual_training_samples"])
print("Manifest:", os.environ["MANIFEST"])
```

## 8. Download preparation NOW

This ZIP contains manifests, verification sidecar, audit/recovery/exclusion reports and hash cache, not images. Future notebooks can restore it under PREP_ROOT and reuse ready/selected_manifest.csv while images are unchanged. Changed images/cohorts need renewed verification.

```python
import os, shutil
from IPython.display import FileLink, display
archive = shutil.make_archive("/kaggle/working/rgb_prepared_100k_dev15k_seed42", "zip",
                              root_dir=os.environ["PREP_ROOT"])
display(FileLink(archive))
```

## 9. Bounded benchmark

20 warmup + 100 measured batches; at most 500 validation images. Choose settings by throughput. Effective batch = batch per GPU × GPUs × accumulation.

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

## 10. Train RGB R2

Up to five epochs, early stopping and at most 15K dev images before cleanup. Keep SCIENCE_MODE=engineering with aggregate/per-image records. Pretrained weights require cache or Internet. More than 85% unseen accuracy is not guaranteed.

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

## 11. Development threshold

Reuses selection-dev predictions and labels that limitation; never calibrate on final-test data.

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
python analyze_predictions.py calibrate \
  --predictions "$RGB_RUN_DIR/checkpoints/best_dev_predictions.csv" \
  --output "$RGB_RUN_DIR/checkpoints/threshold.json"
```

## 12. Download results

Save BOTH ZIPs before ending the session. Follow RGB_PILOT_RUN_GUIDE.md section 6 for external-development evaluation. Previously inspected external data is development evidence; reserve a fresh final test.

```python
import os, json, shutil
from pathlib import Path
from IPython.display import FileLink, display
settings = Path(os.environ["OUTPUT_BASE"]) / (os.environ["RUN_ID"] + "_session.json")
settings.write_text(json.dumps({key: os.environ[key] for key in values}, indent=2))
archive = shutil.make_archive("/kaggle/working/rgb_pilot_results", "zip",
                              root_dir=os.environ["OUTPUT_BASE"])
display(FileLink(archive))
```
