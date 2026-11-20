# ConvNeXt-128: fresh Colab notebook with visible Python training

Verified against this repository on 2026-10-15. Select one GPU runtime. Execute cells in order. All cells are Python; training runs inside the notebook process so its prints are visible.

Confirmed architecture: `ConvNeXt_128`, torchvision ConvNeXt-Base, classifier `1024 -> 128 -> ReLU -> Dropout(0.5) -> 1 raw logit`. Labels: real=0, fake=1. It fine-tunes the entire network; this trainer does not implement Wang's head/freezing/dropout/LR-multiplier options. No diffusion reconstructions or CLIP package are needed for this model. Its 128-feature interface exists, but this guide does not claim it can replace the RGB expert in every fusion loader without changes. Generalization improvement must be measured.

## 1. Mount Drive, use the latest dataset paths, clone

The archive below is the actual multi-domain ZIP extracted in your last notebook. There is deliberately no fallback to the old dataset ZIP. The folder *inside* that archive still has the name `DatesetNoVideosNoGAN`.

```python
from google.colab import drive
drive.mount('/content/drive')
import os, sys, json, csv, shutil, subprocess, zipfile, runpy, hashlib, gc
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

DATA_CANDIDATES = [
    Path('/content/drive/MyDrive/DataesetNoVideoMultipleDomainAndRealImages/DatasetNoVideoMultipleDomainAndRealImages.zip'),
    Path('/content/drive/MyDrive/DatasetNoVideoMultipleDomainAndRealImages/DatasetNoVideoMultipleDomainAndRealImages.zip'),
    Path('/content/drive/MyDrive/DatasetNoVideoMultipleDomainAndRealImages.zip'),
    Path('/content/drive/MyDrive/DatasetNoVideoNoGan/DatesetNoVideosNoGAN.zip'),
]
DATA_ZIP = next((p for p in DATA_CANDIDATES if p.is_file()), None)
DATA_ROOT = Path('/content/dataset/DatesetNoVideosNoGAN')
UNSEEN_ROOT = Path('/content/unseenDataset/val to be deleted 2')
DRIVE_ROOT = Path('/content/drive/MyDrive/deepfake_convnext_128')
OUTPUT_BASE = DRIVE_ROOT / 'runs'  # checkpoints and logs saved DIRECTLY to Drive
PREP_ROOT = Path('/content/convnext_prepared')
MANIFEST = PREP_ROOT / 'selected_manifest.csv'
SAVED_PREP = DRIVE_ROOT / 'prepared'
REPO_ROOT = Path('/content/Novel-Deepfake-Detection')
for p in (DRIVE_ROOT, OUTPUT_BASE, PREP_ROOT, SAVED_PREP, DRIVE_ROOT / 'commands'):
    p.mkdir(parents=True, exist_ok=True)

if not REPO_ROOT.exists():
    subprocess.run(['git', 'clone', '--branch', 'GeneralizationDeepFakeDetection',
        'https://github.com/AbdelrahmanHassan131/Novel-Deepfake-Detection.git', str(REPO_ROOT)], check=True)
else:
    subprocess.run(['git', 'pull', '--ff-only', 'origin', 'GeneralizationDeepFakeDetection'], cwd=REPO_ROOT, check=True)
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
COMMIT = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
print('Code commit:', COMMIT)

def run_python(script, args):
    # Same Python process as Colab: no hidden subprocess stdout for training.
    previous_argv = sys.argv[:]
    sys.argv = [str(REPO_ROOT / script), *map(str, args)]
    try:
        runpy.run_path(str(REPO_ROOT / script), run_name='__main__')
    finally:
        sys.argv = previous_argv
        gc.collect()
```

## 2. Dependencies and extract the training dataset locally

Keep Colab's matched torch/torchvision. Do not install the repository's entire dependency file over them.

```python
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'timm', 'PyWavelets', 'pytorch-wavelets', 'scikit-learn', 'scipy',
    'pillow', 'opencv-python-headless', 'matplotlib', 'pyyaml', 'tensorboard', 'tqdm'], check=True)
import torch, torchvision
assert torch.cuda.is_available(), 'Select a GPU runtime first.'
print('GPU:', torch.cuda.get_device_name(0))
print('Torch:', torch.__version__, 'Torchvision:', torchvision.__version__)

def extract_archive(archive, destination):
    assert archive.is_file(), f'Archive not found: {archive}'
    destination.mkdir(parents=True, exist_ok=True)
    print('Extracting:', archive, flush=True)
    if shutil.which('7z'):
        subprocess.run(['7z', 'x', str(archive), '-o' + str(destination), '-y'],
                       check=True, stdout=subprocess.DEVNULL)
    else:
        with zipfile.ZipFile(archive) as z:
            z.extractall(destination)

if not all((DATA_ROOT / s / c).is_dir() for s in ('train', 'val') for c in ('real', 'fake')):
    assert DATA_ZIP is not None, f"Dataset archive not found in candidates: {DATA_CANDIDATES}"
    extract_archive(DATA_ZIP, Path('/content/dataset'))
for split in ('train', 'val'):
    for label in ('real', 'fake'):
        folder = DATA_ROOT / split / label
        assert folder.is_dir(), f'Expected the latest dataset here: {folder}'
        print(folder)
```

Your previous folder counts were 225,483 per training class and 59,802 per development class, before training-duplicate exclusion. Thus roughly 451k are training images; the roughly 570k total includes development images.

## 3. Reuse the cleaned manifest, or prepare it once

The previous notebook saved its cleaned originals manifest at `deepfake_drct_128/base_inventory.csv`. This is usable for ConvNeXt because paths, labels and splits are shared. Only reuse it with the SAME original archive and extraction paths. The previous cell saved that CSV without its final verification sidecar, so a one-time hash audit may be needed. After this cell, both are saved for future sessions.

```python
from tools.drct_colab_workflow import copy_stable, digest
from data.drct_preparation import inventory_dataset_directory, generate_gate_sidecar
from data.drct_manifest import read_drct_manifest
from data.manifest import verify_manifest_gate

previous_manifest = Path('/content/drive/MyDrive/deepfake_drct_128/base_inventory.csv')
saved_manifest = SAVED_PREP / 'selected_manifest.csv'
saved_gate = saved_manifest.with_suffix('.verified.json')

if saved_manifest.is_file() and saved_gate.is_file():
    copy_stable(saved_manifest, MANIFEST)
    copy_stable(saved_gate, MANIFEST.with_suffix('.verified.json'))
    verify_manifest_gate(MANIFEST, require_hashes=True)
    print('Restored verified preparation; no image rehash.')
else:
    if previous_manifest.is_file():
        print('Reusing the previous cleaned originals manifest:', previous_manifest)
        copy_stable(previous_manifest, MANIFEST)
    else:
        result = inventory_dataset_directory(DATA_ROOT, MANIFEST, duplicate_policy='drop_train')
        print(json.dumps(result, indent=2))
        for suffix in ('.inventory_report.json', '.excluded_duplicates.csv'):
            copy_stable(MANIFEST.with_suffix(suffix), SAVED_PREP / MANIFEST.with_suffix(suffix).name)
    generate_gate_sidecar(MANIFEST, require_hashes=True)
    copy_stable(MANIFEST, saved_manifest)
    copy_stable(MANIFEST.with_suffix('.verified.json'), saved_gate)

rows = read_drct_manifest(MANIFEST)
assert all(r['sample_kind'].endswith('_original') for r in rows), 'Use an originals-only manifest for this baseline.'
assert all(Path(r['path']).resolve().is_relative_to(DATA_ROOT.resolve()) for r in rows), 'Manifest points to a different dataset root.'
counts = {}
for r in rows:
    key = f"{r['split']}/{r['label']}"
    counts[key] = counts.get(key, 0) + 1
assert all(counts.get(k, 0) > 0 for k in ('train/0', 'train/1', 'dev/0', 'dev/1'))
print('Manifest counts (0=real, 1=fake):', counts)
del rows
```

## 4. Configure a NEW ConvNeXt run

Do not rerun this cell to resume: it creates a new identity. These are conservative starting batch sizes, not claims of maximum throughput. L4: batch16 with accumulation2. A100: batch32 with accumulation1. T4: start batch8 with accumulation4. All have effective batch32 on one GPU. If memory is insufficient, reduce batch size before starting a new run.

```python
gpu_name = torch.cuda.get_device_name(0).upper()
BATCH = 32 if 'A100' in gpu_name else (8 if 'T4' in gpu_name else 16)
TARGET_EFFECTIVE_BATCH = 32
ACCUM = max(1, (TARGET_EFFECTIVE_BATCH + BATCH - 1) // BATCH)
WORKERS = min(2, os.cpu_count() or 1)
EPOCHS = 3
RUN_NAME = 'convnext128_multidomain'
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
EXP_DIR = OUTPUT_BASE / f'{RUN_NAME}_{RUN_ID}'
BEST = EXP_DIR / 'checkpoints/best.pth'

training_args = [
    '--arch', 'ConvNeXt_128', '--pretrained',
    '--name', RUN_NAME, '--run_id', RUN_ID, '--checkpoints_dir', str(OUTPUT_BASE),
    '--dataroot', str(DATA_ROOT / 'train'), '--val_root', str(DATA_ROOT / 'val'),
    '--manifest', str(MANIFEST), '--manifest_split', 'train',
    '--val_manifest', str(MANIFEST), '--val_manifest_split', 'dev',
    '--allow_source_overlap', '--allow_aggregate_sources',
    '--gpu_ids', '0', '--batch_size', str(BATCH), '--grad_accum_steps', str(ACCUM),
    '--val_batch_size', str(BATCH), '--num_workers', str(WORKERS),
    '--prefetch_factor', '2', '--persistent_workers', '--pin_memory',
    '--image_size', '256', '--crop_size', '224', '--crop_policy', 'scale_and_crop',
    '--aug_recipe', 'rgb_v1', '--class_bal',
    '--optim', 'adam', '--lr', '0.0001', '--lr_policy', 'cosine',
    '--epochs', str(EPOCHS), '--epochs_decay', '0', '--monitor_metric', 'auc',
    '--use_amp', '--amp_dtype', 'fp16', '--val_precision', 'fp32',
    '--seed', '42', '--log_freq', '25', '--val_epoch_freq', '1', '--save_epoch_freq', '0',
]
COMMAND_FILE = DRIVE_ROOT / 'commands' / f'{RUN_NAME}_{RUN_ID}.json'
COMMAND_FILE.write_text(json.dumps({
    'commit': COMMIT, 'experiment_dir': str(EXP_DIR), 'args': training_args,
    'manifest_sha256': digest(MANIFEST),
}, indent=2))
print('SAVE THIS COMMAND FILE FOR RESUME:', COMMAND_FILE)
print('Checkpoints and logs:', EXP_DIR)
print('Batch:', BATCH, 'Accumulation:', ACCUM, 'Workers:', WORKERS)
print('Actual effective batch on one GPU:', BATCH * ACCUM)
```

The recipe uses training-only random cropping/flipping, JPEG and blur augmentation; evaluation uses the saved deterministic preprocessing. This is a baseline experiment, not a guaranteed generalization fix. No Wang-specific label smoothing, layer4 policy or backbone LR multiplier is passed. Adam in the current ConvNeXt trainer does not apply the CLI weight-decay option, so this command makes no weight-decay claim. `save_epoch_freq=0` suppresses numbered checkpoint copies; `best.pth` and `last.pth` are still saved.

## 5. TRAIN — direct Python execution with live progress

This is the requested training cell: no Bash, no hidden subprocess, no background training. First use downloads torchvision's pretrained ConvNeXt-Base weights. You should see startup, first-batch, 25-batch progress, epoch and validation messages. Hash/path checks happen before GPU batches. Drive checkpoint writes can pause at epoch boundaries. A sudden runtime loss can lose the unfinished epoch; completed best/last checkpoints live on Drive without a separate backup thread.

```python
assert not EXP_DIR.exists(), 'Existing experiment: use the resume cell instead.'
print('Starting ConvNeXt-128 training; progress will appear below.', flush=True)
run_python('train.py', training_args)
assert BEST.is_file(), f'Training returned without best checkpoint: {BEST}'
torch.cuda.empty_cache()
print('Training completed. Exact best checkpoint:', BEST, flush=True)
```

If you interrupt, do not pick the first checkpoint from a directory search. Resume the exact experiment below, or explicitly report it as an interrupted run.

## 6. Optional RESUME — replaces cells 4–5

In a new session, restore the same code commit and dataset/preparation using cells 1–3. Set the exact saved command-file path. Do not change batch size, total epoch budget or configuration during this controlled resume.

```python
COMMAND_FILE = Path('/content/drive/MyDrive/deepfake_convnext_128/commands/PASTE_EXACT_FILENAME.json')
saved = json.loads(COMMAND_FILE.read_text())
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip() == saved['commit'], 'Restore the saved code commit, then restart the kernel.'
assert digest(MANIFEST) == saved['manifest_sha256'], 'The prepared manifest changed.'
EXP_DIR = Path(saved['experiment_dir'])
LAST = EXP_DIR / 'checkpoints/last.pth'
BEST = EXP_DIR / 'checkpoints/best.pth'
assert LAST.is_file(), f'No completed epoch checkpoint to resume: {LAST}'
run_python('train.py', saved['args'] + ['--continue_train', '--resume_checkpoint', str(LAST)])
assert BEST.is_file(), BEST
torch.cuda.empty_cache()
```

## 7. Evaluate internal development and freeze its logit threshold

```python
assert BEST.is_file(), BEST
DEV_OUT = EXP_DIR / 'internal_eval'
run_python('evaluate.py', [
    '--arch', 'ConvNeXt_128', '--checkpoint', str(BEST),
    '--manifest', str(MANIFEST), '--split', 'dev', '--device', 'cuda:0',
    '--batch_size', '16', '--num_workers', '0', '--eval_precision', 'fp32',
    '--no_plots', '--no_tsne', '--no_gradcam', '--no_profiling', '--output_dir', str(DEV_OUT),
])
THRESHOLD = BEST.parent / 'threshold_logit.json'
run_python('analyze_predictions.py', [
    'calibrate', '--predictions', str(DEV_OUT / 'ConvNeXt_128/predictions.csv'),
    '--score_space', 'logit', '--eval_precision', 'fp32', '--output', str(THRESHOLD),
])
print('Frozen internal-development threshold:', THRESHOLD)
```

## 8. External development evaluation

This is the previously inspected 60,000-image collection; do not call it a fresh independent test. Extract it only when needed. The candidates below are all locations from your last notebook for the same external archive.

```python
external_archives = [
    Path('/content/drive/MyDrive/UnseenData/val to be deleted 2.zip'),
    Path('/content/drive/MyDrive/unseenData/val to be deleted 2.zip'),
    Path('/content/drive/MyDrive/UnseenData/val_to_be_deleted_2.zip'),
    Path('/content/drive/MyDrive/shared_project_files/UnseenData/val to be deleted 2.zip'),
]
if not all((UNSEEN_ROOT / c).is_dir() for c in ('real', 'fake')):
    archive = next((p for p in external_archives if p.is_file()), None)
    assert archive is not None, 'External archive not found in the known Drive locations.'
    extract_archive(archive, Path('/content/unseenDataset'))
assert all((UNSEEN_ROOT / c).is_dir() for c in ('real', 'fake')), UNSEEN_ROOT
EXTERNAL_OUT = EXP_DIR / 'external_eval'
run_python('evaluate.py', [
    '--arch', 'ConvNeXt_128', '--checkpoint', str(BEST),
    '--val_root', str(UNSEEN_ROOT), '--split', 'external_dev',
    '--threshold_file', str(THRESHOLD), '--device', 'cuda:0',
    '--batch_size', '16', '--num_workers', '0', '--eval_precision', 'fp32',
    '--no_plots', '--no_tsne', '--no_gradcam', '--no_profiling', '--output_dir', str(EXTERNAL_OUT),
])
report_path = EXTERNAL_OUT / 'ConvNeXt_128/evaluation_report.json'
gen_report_path = EXTERNAL_OUT / 'ConvNeXt_128/generalization_report.json'
if report_path.is_file():
    report = json.loads(report_path.read_text())
    metrics = report.get('classification_metrics', {})
elif gen_report_path.is_file():
    report = json.loads(gen_report_path.read_text())
    overall = report.get('overall', {})
    metrics = {
        'accuracy': overall.get('accuracy', 0.0),
        'roc_auc': overall.get('auc', 0.0),
        'pr_auc': overall.get('ap', 0.0),
        'eer': overall.get('eer', 0.0),
        'f1_score': overall.get('f1', 0.0),
        'specificity': overall.get('specificity', 0.0),
        'sensitivity': overall.get('sensitivity', 0.0),
    }
else:
    metrics = {}

if metrics:
    print("\n" + "=" * 60)
    print("      CONVNEXT-128 EXTERNAL UNSEEN EVALUATION")
    print("=" * 60)
    print(f"  Accuracy (Calibrated Threshold): {metrics.get('accuracy', 0.0) * 100:.2f}%")
    print(f"  ROC AUC:                         {metrics.get('roc_auc', 0.0):.4f}")
    print(f"  Average Precision (PR AUC):      {metrics.get('pr_auc', metrics.get('average_precision', 0.0)):.4f}")
    print(f"  Equal Error Rate (EER):          {metrics.get('eer', 0.0) * 100:.2f}%")
    print(f"  F1 Score:                        {metrics.get('f1_score', 0.0):.4f}")
    print(f"  Specificity (True Real Rate):    {metrics.get('specificity', 0.0) * 100:.2f}%")
    print(f"  Sensitivity (True Fake Rate):    {metrics.get('sensitivity', 0.0) * 100:.2f}%")
    print("=" * 60)
    print('Report available at:', report_path if report_path.is_file() else gen_report_path)
```

Keep the complete `deepfake_convnext_128` Drive folder: prepared manifests/gates, command descriptors, checkpoints, logs, predictions and thresholds. No production ConvNeXt training or pretrained download was performed while writing this guide.
