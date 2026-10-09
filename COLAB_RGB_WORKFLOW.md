**2026-10-09 update:** This collection contains repeated `vid_<hash>_face_...` filename families. The original per-image manifests permit family overlap. Follow `RGB_GENERALIZATION_REPAIR.md` before another training run; keep existing hashes/backups. The repaired split does not by itself remove source/format bias.

# Colab: prepare once on CPU, train later on one GPU

Use **COLAB_01_PREPARE_CPU.ipynb** first, then **COLAB_02_TRAIN_RGB_GPU.ipynb** in a fresh session. All commands are included below.

- No GPU is needed for session 1. Large RAM and enough local disk for the image dataset matter more.
- Supply the original images with real/fake class folders under `/content/dataset/Preapred Dataset/train`. No old manifest or trained checkpoint is needed.
- This train-only workflow reserves 15K development images, audits the whole supplied pool once, and saves four choices: 100K, 500K, 1M-target and all. Label-conflict cleanup/group integrity can reduce counts. A literal 1M training cohort needs at least 1M eligible training images AFTER dev reservation and cleanup, including 500K in each class for the balanced 1M target. Reports never mislabel actual counts as the requested target.
- The same cleaned dev set is shared across all choices. Nested smaller training cohorts are prefixes of the same deterministic whole-group ordering. No unseen/final-test images are added.
- Save `MyDrive/deepfake_rgb/prepared_rgb_sizes_seed42.zip`, `prepared_rgb_code.zip`, and your ORIGINAL IMAGE ARCHIVE. Preparation ZIP contains pool/candidate/clean parent/subsets, hash cache, verification sidecars, audit/quarantine/selection reports. Never edit a CSV without renewing its verification.
- A second session must restore the original images at the exact same path, then restore ZIPs. Select TRAIN_SIZE and train. No full-image hashing repeats for completed preparation on unchanged data. Interrupted audits can resume from the cache, but re-extracted files with changed mtimes may be rehashed safely.
- Runtime checks/training/preparation were NOT executed by the author. The notebooks and helper were statically checked. Push/upload these changes before cloning in Colab.

## Additional information needed only for a dataset download cell

The local extraction path is known; the download source is not. Provide the dataset/archive location (Drive archive path or download source) if you want its exact download/extraction command. For this workflow, separate val images are optional: dev is reserved from train. For stronger scientific claims, supply original generator/source and video/identity metadata, plus untouched external test data later; image hashes cannot establish those relationships.


---

# Colab session 1 — CPU preparation

Run this notebook with **CPU**; no GPU is used for hashing or selection. Use a high-RAM runtime if available: the current audit/recovery tools hold large metadata tables in RAM. Runtime/disk availability varies; monitor RAM while preparing a million images.

Push/upload the new repository code before cloning. These cells were authored and syntax-checked, NOT executed here.

Input is exactly `/content/dataset/Preapred Dataset/train`. This workflow intentionally uses ONLY that folder, reserves a balanced 15K development set from it, hashes the entire supplied pool once, quarantines label conflicts and removes train/dev overlap, then derives nested 100K, 500K, 1M-target and `all` training manifests. Every size shares the same cleaned dev. No images are copied by manifest preparation. Sizes report actual counts: 1M total input cannot yield 1M train after reserving dev. Target-sized manifests are balanced where available; `all` includes all eligible training rows and may be imbalanced.

Groups are explicitly image-level and unverified for people/videos. Hashes detect exact duplicates, not all near-duplicates. This remains an engineering experiment, not verified source-held-out evidence. Reserve untouched external data for final generalization evaluation. Do not add an external test to this pool.

Keep dataset bytes/relative names unchanged throughout preparation and future sessions. If you re-extract after interruption, mtime changes may require rehashing for an INCOMPLETE audit; a completed saved preparation does not need hashing again under the unchanged-dataset assumption.

Google Drive saves live under `MyDrive/deepfake_rgb`. Images must be restored to local `/content` each session; manifests do not contain images. Avoid training from hundreds of thousands of small Drive files; copy/extract an archive locally ([Colab guidance](https://research.google.com/colaboratory/faq.html)).

## 1. Mount Drive and set paths

Run at the start of the CPU session.

```python
from google.colab import drive
drive.mount('/content/drive')
import os, json, shutil, subprocess, sys, hashlib, zipfile
from pathlib import Path
DRIVE_ROOT = Path('/content/drive/MyDrive/deepfake_rgb')
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
REPO_ROOT = Path('/content/Novel-Deepfake-Detection')
DATA_ROOT = Path('/content/dataset/Preapred Dataset/train')
PREP_ROOT = Path('/content/prepared_data/rgb_sizes_seed42')
os.environ.update(REPO_ROOT=str(REPO_ROOT), DATA_ROOT=str(DATA_ROOT), PREP_ROOT=str(PREP_ROOT))
print('Persistent storage:', DRIVE_ROOT)
print('Required images:', DATA_ROOT)
```

## 2. Get updated repository

Latest Colab helper must be on the branch, or upload the updated repository at REPO_ROOT.

```python
if not REPO_ROOT.exists():
    subprocess.run(['git', 'clone', '--branch', 'GeneralizationDeepFakeDetection',
                    'https://github.com/AbdelrahmanHassan131/Novel-Deepfake-Detection.git', str(REPO_ROOT)], check=True)
assert (REPO_ROOT / 'tools/colab_manifest_sizes.py').is_file(), 'Push/upload the latest Colab changes first.'
subprocess.run(['git', '-C', str(REPO_ROOT), 'log', '-1', '--oneline'], check=True)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

## 3. Dependencies

Keep Colab’s installed torch/torchvision. CPU preparation does not require CUDA.

```python
%pip install -q numpy pillow pytorch-wavelets PyWavelets tensorboard pytest
```

## 4. Metadata helper checks

Run these small metadata-only tests before scanning images. No GPU or image dataset is used by these tests.

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
python -m pytest -q tests/test_colab_manifest_sizes.py tests/test_prepared_manifest_recovery.py
```

## 5. Provide the images

Download/extract your original dataset before this check. The download command depends on where you store the dataset; do not rename class folders or recompress images. No old checkpoint or manifest is needed.

```python
# Put/download/extract your original dataset here before running this cell.
# Required: /content/dataset/Preapred Dataset/train/real and .../fake
# Also supported by inventory: 0_real and 1_fake instead of real and fake.
assert DATA_ROOT.is_dir(), f'Extract the image dataset first: {DATA_ROOT}'
class_dirs = [p.name for p in DATA_ROOT.iterdir() if p.is_dir()]
assert ({'real', 'fake'} <= set(class_dirs) or {'0_real', '1_fake'} <= set(class_dirs)), class_dirs
print('Class folders:', class_dirs)
print('Free local disk (GiB):', round(shutil.disk_usage('/content').free / 2**30, 1))
print('CPU count:', os.cpu_count())
print('RAM:', next(line.strip() for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:')))
```

## 6. Restore interrupted work and define backups

Safe only for this same dataset and seed. Completed-stage backups are separate from periodic hash-cache snapshots.

```python
# Optional recovery after an interrupted CPU session; keep the same images/settings.
PREP_ROOT.mkdir(parents=True, exist_ok=True)
backup = DRIVE_ROOT / 'preparation_progress.zip'
if backup.exists() and not any(PREP_ROOT.iterdir()):
    with zipfile.ZipFile(backup) as archive:
        archive.extractall(PREP_ROOT)
    print('Restored completed preparation stages.')
cache_backup = DRIVE_ROOT / 'hashes_progress.sqlite'
if cache_backup.exists() and not (PREP_ROOT / 'hashes.sqlite').exists():
    shutil.copy2(cache_backup, PREP_ROOT / 'hashes.sqlite')
    print('Restored hash cache. Changed file sizes/mtimes trigger safe rehashing.')

def save_preparation(name='preparation_progress.zip', include_cache=False):
    local = Path('/content') / name
    with zipfile.ZipFile(local, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for file in PREP_ROOT.rglob('*'):
            if file.is_file() and (include_cache or not file.name.startswith('hashes.sqlite')):
                archive.write(file, file.relative_to(PREP_ROOT))
    destination = DRIVE_ROOT / name
    temporary = destination.with_suffix('.uploading')
    shutil.copy2(local, temporary)
    temporary.replace(destination)
    print('Saved to Drive:', destination)
```

## 7. Inventory and reserve development data

Scans all directory entries but does not hash images yet. Class mapping: real=0, fake=1. Provisional per-image split is cleaned after the full audit.

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
if [[ ! -f "$PREP_ROOT/pool_manifest.csv" ]]; then
  python -u prepare_dataset.py inventory --root "$DATA_ROOT" --source diffgan \
    --output "$PREP_ROOT/pool_manifest.csv.tmp" --image_level_groups
  mv "$PREP_ROOT/pool_manifest.csv.tmp" "$PREP_ROOT/pool_manifest.csv"
fi
if [[ ! -f "$PREP_ROOT/candidate.csv" ]]; then
  python -u tools/colab_manifest_sizes.py partition \
    --inventory "$PREP_ROOT/pool_manifest.csv" --output "$PREP_ROOT/candidate.csv" \
    --dev_size 15000 --seed 42
fi
```

## 8. Save inventory before the long audit

Prevents losing inventory/partition selection if the CPU runtime disconnects.

```python
save_preparation()
```

## 9. Hash the entire pool ONCE

Main CPU/disk cost. Progress every 5,000 images; transaction-consistent hash-cache snapshots go to Drive every five minutes. Audit exit 1 is not ignored: recovery below accepts only supported duplicate/label-conflict cases. Other failures stop.

```python
import threading, sqlite3
from uuid import uuid4

# SQLite stays on local disk. Make transaction-consistent snapshots for Drive.
stop_backup = threading.Event()
def snapshot_hashes():
    cache = PREP_ROOT / 'hashes.sqlite'
    if not cache.exists():
        return
    local = Path('/content/hashes_snapshot.sqlite')
    with sqlite3.connect(str(cache), timeout=60) as source:
        with sqlite3.connect(str(local)) as destination:
            source.backup(destination, pages=1024)
    remote = DRIVE_ROOT / 'hashes_progress.sqlite'
    temporary = remote.with_suffix('.uploading')
    shutil.copy2(local, temporary)
    temporary.replace(remote)
    print('Hash progress saved to Drive.', flush=True)

def backup_worker():
    while not stop_backup.wait(300):
        try:
            snapshot_hashes()
        except Exception as exc:
            print('Drive snapshot failed; local hashing continues:', exc, flush=True)

def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

candidate = PREP_ROOT / 'audited_candidate.csv'
report_path = PREP_ROOT / 'full_audit.json'
completed = False
if report_path.exists() and candidate.exists():
    evidence = json.loads(report_path.read_text())
    completed = evidence.get('hashes_verified') is True and evidence.get('manifest_sha256') == file_digest(candidate)
if completed:
    print('Reusing completed full-pool hash audit; no images reopened.')
else:
    shutil.copy2(PREP_ROOT / 'candidate.csv', candidate)
    worker = threading.Thread(target=backup_worker, daemon=True)
    worker.start()
    fresh_report = PREP_ROOT / ('audit_' + uuid4().hex[:8] + '.json')
    try:
        result = subprocess.run([sys.executable, '-u', 'prepare_dataset.py', 'audit',
            '--manifest', str(candidate), '--root', str(DATA_ROOT), '--output', str(fresh_report),
            '--hashes', '--hash_cache', str(PREP_ROOT / 'hashes.sqlite')], cwd=REPO_ROOT)
    finally:
        stop_backup.set()
        worker.join()
        snapshot_hashes()
    if result.returncode not in (0, 1) or not fresh_report.is_file():
        raise RuntimeError('Audit interrupted/failed. Retain cache and retry; do not proceed.')
    evidence = json.loads(fresh_report.read_text())
    assert evidence.get('hashes_verified') is True
    assert evidence.get('manifest_sha256') == file_digest(candidate), 'Incomplete/mismatched evidence'
    shutil.copy2(fresh_report, report_path)
save_preparation()
print('Completed audit saved. Next: quarantine and derive all training sizes.')
```

## 10. Clean and derive every size

No images opened. Contradictory components are excluded, never relabeled. Whole duplicate groups remain together; the same dev membership is used across all sizes. A shortfall is reported, never padded with repeated images.

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
python -u tools/recover_prepared_manifest.py \
  --manifest "$PREP_ROOT/audited_candidate.csv" --audit_report "$PREP_ROOT/full_audit.json" \
  --output_dir "$PREP_ROOT/clean_parent" \
  --immutable_dataset --allow_shortfall --quarantine_label_conflicts
python -u tools/colab_manifest_sizes.py subsets \
  --parent "$PREP_ROOT/clean_parent/selected_manifest.csv" --output_dir "$PREP_ROOT/sizes" \
  --sizes 100000 500000 1000000 all --seed 42 --immutable_dataset
```

## 11. Save final preparation and exact code to Drive

Wait for READY ON DRIVE for both ZIPs before ending the CPU session. Save the image archive separately; the preparation archive is metadata only.

```python
summary = json.loads((PREP_ROOT / 'sizes/sizes_summary.json').read_text())
for size, row in summary.items():
    print(size, '-> actual train:', row['actual_training_samples'], 'dev:', row['dev_samples'], 'shortfall:', row['shortfall'])
save_preparation('prepared_rgb_sizes_seed42.zip', include_cache=True)

# Preserve the exact code used, so session 2 does not accidentally use a newer checkout.
files = subprocess.check_output(['git', '-C', str(REPO_ROOT), 'ls-files', '-co', '--exclude-standard', '-z']).decode().split('\0')
code_zip = Path('/content/prepared_rgb_code.zip')
with zipfile.ZipFile(code_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for relative in sorted(set(files)):
        file = REPO_ROOT / relative
        if file.is_file() and file.suffix.lower() in {'.py', '.toml', '.yaml', '.yml', '.json', '.md', '.txt', '.ipynb'} and not any(part in {'output', 'checkpoints', '.venv', '.git', 'logs'} for part in Path(relative).parts):
            archive.write(file, relative)
shutil.copy2(code_zip, DRIVE_ROOT / code_zip.name)
for name in ('prepared_rgb_sizes_seed42.zip', 'prepared_rgb_code.zip'):
    path = DRIVE_ROOT / name
    assert path.is_file() and path.stat().st_size > 0
    print('READY ON DRIVE:', path, 'bytes:', path.stat().st_size)
print('CPU session finished. Keep your original image dataset/archive too.')
```

---

# Colab session 2 — single-GPU RGB training

Start a NEW Colab session with one CUDA GPU (for example T4, L4, or A100). If “G4” was a typo for T4, T4 works too; the setup detects the actual device and VRAM. GPU availability varies ([Colab FAQ](https://research.google.com/colaboratory/faq.html)). No distributed/two-GPU launch is needed.

Restore the SAME image dataset at `/content/dataset/Preapred Dataset/train` each session. Restoring manifests alone is insufficient. These cells reuse completed SHA256 evidence under the explicit unchanged-dataset assumption; they do not claim to reverify current bytes. Dataset changes require renewed verification.

Select `TRAIN_SIZE='100000'` first; later use `500000`, `1000000`, or `all`. Read actual counts before training. Use the same shared development cohort and a fresh run directory for each experiment. No old model checkpoint is required. CPU preparation never needs to run again for these saved cohorts while image bytes and paths remain unchanged.

The RGB R2 recipe is a starting experiment, not an accuracy guarantee. Full GPU utilization cannot be guaranteed: benchmark on the assigned CPU/GPU and local dataset first. Code/runtime tests below were not executed locally.

## 1. Mount Drive and set paths

Same paths as the CPU notebook.

```python
from google.colab import drive
drive.mount('/content/drive')
import os, json, shutil, subprocess, sys, hashlib, zipfile
from pathlib import Path
DRIVE_ROOT = Path('/content/drive/MyDrive/deepfake_rgb')
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
REPO_ROOT = Path('/content/Novel-Deepfake-Detection')
DATA_ROOT = Path('/content/dataset/Preapred Dataset/train')
PREP_ROOT = Path('/content/prepared_data/rgb_sizes_seed42')
os.environ.update(REPO_ROOT=str(REPO_ROOT), DATA_ROOT=str(DATA_ROOT), PREP_ROOT=str(PREP_ROOT))
print('Persistent storage:', DRIVE_ROOT)
print('Required images:', DATA_ROOT)
```

## 2. Restore prepared manifests and exact code

Do not clone another branch over this code snapshot. Use a fresh /content session.

```python
PREP_ROOT.mkdir(parents=True, exist_ok=True)
for archive_name, destination in [('prepared_rgb_sizes_seed42.zip', PREP_ROOT),
                                  ('prepared_rgb_code.zip', REPO_ROOT)]:
    archive_path = DRIVE_ROOT / archive_name
    assert archive_path.is_file(), f'Complete CPU preparation first: {archive_path}'
    assert not destination.exists() or not any(destination.iterdir()), f'Use a fresh session/empty destination: {destination}'
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
print('Restored preparation and the exact saved code. No inventory or hash run required.')
```

## 3. Dependencies

Keep Colab’s preinstalled compatible torch/torchvision pair.

```python
%pip install -q numpy pillow pytorch-wavelets PyWavelets tensorboard pytest
```

## 4. Restore original images

Repeat your original dataset download/extraction now. No hash/inventory command is needed for unchanged images.

```python
# Put/download/extract your original dataset here before running this cell.
# Required: /content/dataset/Preapred Dataset/train/real and .../fake
# Also supported by inventory: 0_real and 1_fake instead of real and fake.
assert DATA_ROOT.is_dir(), f'Extract the image dataset first: {DATA_ROOT}'
class_dirs = [p.name for p in DATA_ROOT.iterdir() if p.is_dir()]
assert ({'real', 'fake'} <= set(class_dirs) or {'0_real', '1_fake'} <= set(class_dirs)), class_dirs
print('Class folders:', class_dirs)
print('Free local disk (GiB):', round(shutil.disk_usage('/content').free / 2**30, 1))
print('CPU count:', os.cpu_count())
print('RAM:', next(line.strip() for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:')))
```

## 5. Choose size and detect one GPU

Batch/accumulation keep effective batch 64. For an out-of-memory error, halve BATCH and double ACCUM before benchmarking again. Change os.environ values too if adjusting after this cell.

```python
import torch
from uuid import uuid4
from data.manifest import verify_manifest_gate

TRAIN_SIZE = '100000'  # '500000', '1000000', or 'all' in later sessions
UNCHANGED_DATASET = True  # True only for the SAME image bytes and paths as session 1
assert UNCHANGED_DATASET, 'Changed data needs renewed preparation/audit.'
assert torch.cuda.is_available(), 'Select a GPU runtime first.'
manifest = PREP_ROOT / 'sizes' / TRAIN_SIZE / 'selected_manifest.csv'
verify_manifest_gate(manifest, require_hashes=True)
summary = json.loads((manifest.parent / 'selection_report.json').read_text())
print('Training size actually available:', summary['actual_training_samples'])
print('Shared dev size:', summary['dev_samples'], 'Training shortfall:', summary['shortfall'])
props = torch.cuda.get_device_properties(0)
vram_gib = props.total_memory / 2**30
# Starting points, not throughput guarantees; use the benchmark below.
batch = 64 if vram_gib >= 35 else (32 if vram_gib >= 20 else 16)
accum = 64 // batch  # same effective batch of 64 on one GPU
workers = min(4, max(1, (os.cpu_count() or 2) - 1))
run_id = 'seed42_' + uuid4().hex[:8]
output = Path('/content/rgb_pilots')
output.mkdir(exist_ok=True)
values = dict(REPO_ROOT=str(REPO_ROOT), DATA_ROOT=str(DATA_ROOT), MANIFEST=str(manifest),
    VAL_MANIFEST=str(manifest), TRAIN_SPLIT='train', DEV_SPLIT='dev', GPU_COUNT='1', GPU_IDS='0',
    BATCH=str(batch), ACCUM=str(accum), WORKERS=str(workers), OUTPUT_BASE=str(output),
    RUN_ID=run_id, RUN_NAME='rgb_r2_' + TRAIN_SIZE, RGB_HEAD='128d', RGB_DROPOUT='0.5',
    FINE_TUNE='layer4_and_head', BN_POLICY='frozen', LR_MULT='0.1', AUG_RECIPE='rgb_v1',
    SCIENCE_MODE='engineering', ELIGIBLE_SOURCES='')
run_dir = output / (values['RUN_NAME'] + '_' + run_id)
values.update(RGB_RUN_DIR=str(run_dir), RGB_CKPT=str(run_dir / 'checkpoints/best.pth'))
os.environ.pop('RESUME_CHECKPOINT', None)
os.environ.update(values)
(output / (run_id + '_session.json')).write_text(json.dumps(values, indent=2))
print('GPU:', props.name, '| VRAM GiB:', round(vram_gib, 1))
print('Batch:', batch, '| accumulation:', accum, '| workers:', workers)
print('Run:', run_dir)
print('Do not rerun this setup cell during the same experiment; it creates a new run ID.')
```

## 6. Focused code checks

Stop if checks fail; synthetic parity uses a tiny synthetic input, not your external test.

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
python -m pytest -q tests/test_colab_manifest_sizes.py tests/test_prepared_manifest_recovery.py tests/test_rgb_review_final_repairs.py \
  tests/test_versioned_augmentations.py tests/test_calibration_and_source_readiness.py \
  tests/test_round1_phase3_repairs.py
python -u tools/diagnose_rgb_parity.py --synthetic --arch Wang2020_128 \
  --output_report "$OUTPUT_BASE/${RUN_ID}_synthetic_parity.json"
```

## 7. Benchmark

Inspect throughput, data-wait and GPU timing before full training. If CPU loading dominates, try WORKERS=2 versus 4 within available CPUs; rerun this bounded benchmark. Keep inputs on local disk. Update os.environ["BATCH"], ["ACCUM"], ["WORKERS"] to change settings. No performance promise from a GPU name alone.

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
PYTHONUNBUFFERED=1 python -u tools/benchmark_training_throughput.py \
  --manifest "$MANIFEST" --val_manifest "$VAL_MANIFEST" --dataroot "$DATA_ROOT" \
  --gpu_ids "$GPU_IDS" --batch_size "$BATCH" --grad_accum_steps "$ACCUM" \
  --num_workers "$WORKERS" --prefetch_factor 2 --persistent_workers --pin_memory \
  --use_amp --amp_dtype fp16 --aug_recipe "$AUG_RECIPE" \
  --fine_tune_policy "$FINE_TUNE" --bn_policy "$BN_POLICY" \
  --rgb_head_type "$RGB_HEAD" --rgb_dropout "$RGB_DROPOUT" --backbone_lr_mult "$LR_MULT" \
  --warmup_microbatches 20 --measure_microbatches 100 --val_samples 500 \
  --output "$OUTPUT_BASE/${RUN_ID}_benchmark.json"
```

## 8. Start periodic Drive backups

Best effort every five minutes; a sudden runtime loss can lose work since the last backup. Final ZIP below is the complete saved result. Stop training before restoring a backup; do not rerun setup to resume.

```python
# Periodically sync stable completed files; run before training.
# Local images/checkpoints stay fast; copies go to Drive in the background.
import threading, time
if 'training_backup_stop' in globals():
    training_backup_stop.set()
    training_backup_thread.join()
training_backup_stop = threading.Event()
remote_run = DRIVE_ROOT / 'runs' / (values['RUN_NAME'] + '_' + values['RUN_ID'])
remote_run.mkdir(parents=True, exist_ok=True)

def sync_results():
    local_run = Path(os.environ['RGB_RUN_DIR'])
    if not local_run.exists():
        return
    for source in local_run.rglob('*'):
        if not source.is_file() or source.name.endswith(('.tmp', '.temp', '.uploading')):
            continue
        before = source.stat()
        if time.time() - before.st_mtime < 15:
            continue
        target = remote_run / source.relative_to(local_run)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == before.st_size and abs(target.stat().st_mtime - before.st_mtime) < 1:
            continue
        tmp = target.with_name(target.name + '.uploading')
        shutil.copy2(source, tmp)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns):
            tmp.replace(target)
    shutil.copy2(Path(os.environ['OUTPUT_BASE']) / (values['RUN_ID'] + '_session.json'), remote_run / 'session.json')

def training_backup_worker():
    while not training_backup_stop.wait(300):
        try:
            sync_results()
            print('Stable training outputs copied to Drive.', flush=True)
        except Exception as exc:
            print('Training backup failed; local training continues:', exc, flush=True)
training_backup_thread = threading.Thread(target=training_backup_worker, daemon=True)
training_backup_thread.start()
print('Backup destination:', remote_run)
```

## 9. Train RGB

One GPU, up to five epochs, early stopping, pretrained RGB backbone. Development images never enter training. Output logs remain visible.

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
PYTHONUNBUFFERED=1 python -u train.py \
  --arch Wang2020_128 --name "$RUN_NAME" --run_id "$RUN_ID" --checkpoints_dir "$OUTPUT_BASE" \
  --dataroot "$DATA_ROOT" --manifest "$MANIFEST" --manifest_split "$TRAIN_SPLIT" \
  --val_manifest "$VAL_MANIFEST" --val_manifest_split "$DEV_SPLIT" \
  --rgb_head_type "$RGB_HEAD" --rgb_dropout "$RGB_DROPOUT" \
  --fine_tune_policy "$FINE_TUNE" --bn_policy "$BN_POLICY" --backbone_lr_mult "$LR_MULT" \
  --aug_recipe "$AUG_RECIPE" "${aug_args[@]}" \
  --gpu_ids "$GPU_IDS" --batch_size "$BATCH" --grad_accum_steps "$ACCUM" \
  --val_batch_size "$BATCH" --num_workers "$WORKERS" --prefetch_factor 2 --persistent_workers --pin_memory \
  --epochs 5 --epochs_decay 0 --optim adam --lr 0.0001 --lr_policy cosine \
  --use_amp --amp_dtype fp16 --val_precision fp32 --seed 42 --pretrained \
  --early_stopping --early_stopping_patience 2 --early_stopping_min_epochs 2 \
  --save_epoch_freq 0 "${source_args[@]}" "${resume_args[@]}"
```

## 10. Calibrate on development predictions

Never choose a threshold using final external-test labels. The same selection dev is used here; this is not independent calibration evidence.

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
python analyze_predictions.py calibrate \
  --predictions "$RGB_RUN_DIR/checkpoints/best_dev_predictions.csv" \
  --output "$RGB_RUN_DIR/checkpoints/threshold.json"
```

## 11. Save complete results to Drive

Wait for the final saved confirmation before disconnecting. Then evaluate the RGB best checkpoint on unseen data using its saved threshold, without tuning on the final test.

```python
training_backup_stop.set()
training_backup_thread.join()
local_run = Path(os.environ['RGB_RUN_DIR'])
assert (local_run / 'checkpoints/best.pth').is_file()
assert (local_run / 'checkpoints/threshold.json').is_file()
archive = shutil.make_archive('/content/' + local_run.name, 'zip', root_dir=local_run)
destination = DRIVE_ROOT / 'runs' / (local_run.name + '.zip')
temporary = destination.with_suffix('.uploading')
shutil.copy2(archive, temporary)
temporary.replace(destination)
shutil.copy2(Path(os.environ['OUTPUT_BASE']) / (values['RUN_ID'] + '_session.json'), remote_run / 'session.json')
for report in Path(os.environ['OUTPUT_BASE']).glob(values['RUN_ID'] + '_*.json'):
    shutil.copy2(report, remote_run / report.name)
print('FINAL MODEL, THRESHOLD, LOGS AND PREDICTIONS SAVED:', destination)
print('Keep the prepared data ZIP and original image dataset for future training sizes.')
```
