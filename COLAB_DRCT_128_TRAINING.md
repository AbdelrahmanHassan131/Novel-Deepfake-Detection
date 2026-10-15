# Colab: DRCT-128 preparation, training and resume

Updated 2026-10-15. Push these fixes before cloning. Output is one raw binary logit (real=0, fake=1) with a learned 128D feature for MHA. This is an adaptation, not a verified paper reproduction or an accuracy guarantee.

**Full DRCT needs diffusion reconstructions of both real and fake training images, with actual parent/provenance metadata. Original real/fake folders alone are insufficient.** This repository imports precomputed reconstructions; it does not currently provide a verified diffusion reconstruction generator. Do not use copied/augmented images or unrelated fakes as reconstructions.

If you have no reconstructions, explicitly select `rgb_control` below for pretrained CLIP + 128D + binary BCE training. That is a useful control, not full DRCT. Preparation runs on CPU when reconstructions already exist; training needs one GPU.

## 1. Fresh session: mount and clone
On a resume, restore the code commit recorded by your previous run. Do not advance code silently.

```python
from google.colab import drive
drive.mount('/content/drive')
import os, sys, json, subprocess
from pathlib import Path

REPO_ROOT = Path('/content/Novel-Deepfake-Detection')
DRIVE_ROOT = Path('/content/drive/MyDrive/deepfake_drct_128')
DATA_ROOT = Path('/content/dataset/DatesetNoVideosNoGAN')  # train/real, train/fake, val/real, val/fake
REC_ROOT = Path('/content/dataset/reconstructions')
PREP_ROOT = Path('/content/drct_prepared')
MANIFEST = PREP_ROOT / 'selected_manifest.csv'
BUNDLE = DRIVE_ROOT / 'preparation'
OUTPUT_BASE = Path('/content/drct_experiments')
DRIVE_RUNS = DRIVE_ROOT / 'runs'
TRAINING_MODE = 'drct'  # explicitly change to 'rgb_control' for originals-only BCE
assert TRAINING_MODE in ('drct', 'rgb_control')

if not REPO_ROOT.exists():
    subprocess.run(['git', 'clone', '--branch', 'GeneralizationDeepFakeDetection',
        'https://github.com/AbdelrahmanHassan131/Novel-Deepfake-Detection.git', str(REPO_ROOT)], check=True)
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
for p in (DRIVE_ROOT, PREP_ROOT, REC_ROOT):
    p.mkdir(parents=True, exist_ok=True)
print('Commit:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
```

## 2. Dependencies and original data
Keep Colab's matched PyTorch/torchvision. Scientific wheels are resolved for its Python version, rather than forcing old SciPy builds on Python 3.13. The CLIP source is pinned; each run saves its installed environment.

```python
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33',
    'pytorch-metric-learning==2.8.1', 'scikit-learn', 'scipy', 'pillow', 'timm'], check=True)
import torch, clip
print('Python:', sys.version, 'Torch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU preparation only')
for split in ('train', 'val'):
    for label in ('real', 'fake'):
        assert (DATA_ROOT / split / label).is_dir(), DATA_ROOT / split / label
```

## 3. Prepare once OR restore
For full DRCT, put reconstruction files (nested folders allowed) in `DRIVE_ROOT/reconstructions` and `reconstruction_mapping.csv` at DRIVE_ROOT. Mapping columns:

```text
parent_id,path,parent_sha256,sha256,rec_model_revision,rec_settings,rec_seed
```

Use original **train** IDs from base_inventory.csv, actual SHA256 values, actual generation revision/settings/seed. Paths are relative to REC_ROOT or absolute local reconstruction paths. rec_settings is a CSV-quoted JSON object. Both real and fake parents need mappings. Do not invent metadata. The cell saves the base inventory before stopping if mapping is absent, so a reconstruction pipeline can use those IDs.

```python
from data.drct_preparation import (inventory_dataset_directory, import_precomputed_reconstructions,
                                   generate_gate_sidecar, check_drct_readiness)
from tools.drct_colab_workflow import (copy_stable, mirror_tree, save_preparation,
                                      restore_preparation, create_run, read_run, run_training)

if (BUNDLE / 'bundle.json').is_file():
    # True ONLY after extracting the SAME unchanged original dataset at DATA_ROOT.
    SAME_UNCHANGED_DATASET = True
    restore_preparation(BUNDLE, MANIFEST, REC_ROOT, immutable_dataset=SAME_UNCHANGED_DATASET)
    print('Restored preparation. No original-image inventory/rehash; reconstruction hashes checked while copying.')
else:
    base = PREP_ROOT / 'base_inventory.csv'
    if not base.is_file():
        inventory_dataset_directory(DATA_ROOT, base)  # val on disk becomes dev
    copy_stable(base, DRIVE_ROOT / 'base_inventory.csv')
    if TRAINING_MODE == 'drct':
        mapping = DRIVE_ROOT / 'reconstruction_mapping.csv'
        if not mapping.is_file():
            raise FileNotFoundError(f'Provide paired reconstructions and {mapping}; base inventory is saved on Drive.')
        mirror_tree(DRIVE_ROOT / 'reconstructions', REC_ROOT)
        import_precomputed_reconstructions(base, REC_ROOT, MANIFEST,
                                           reconstruct_fakes=True, mapping_csv=mapping)
        if not check_drct_readiness(MANIFEST)['is_ready_for_full_drct']:
            raise RuntimeError('Incomplete reconstruction pairs: correct the mapping before saving a reusable bundle.')
    else:
        copy_stable(base, MANIFEST)
    generate_gate_sidecar(MANIFEST, require_hashes=True)
    save_preparation(MANIFEST, BUNDLE, REC_ROOT)

readiness = check_drct_readiness(MANIFEST)
print(json.dumps(readiness, indent=2))
if TRAINING_MODE == 'drct' and not readiness['is_ready_for_full_drct']:
    raise RuntimeError('Complete verified real AND fake reconstruction pairs before full DRCT training.')
if TRAINING_MODE == 'rgb_control':
    from data.drct_manifest import read_drct_manifest
    if any(r['sample_kind'].endswith('_reconstruction') for r in read_drct_manifest(MANIFEST)):
        raise RuntimeError('Use a separate originals-only preparation for the RGB control.')
```

You can now end the CPU session. Keep the preparation bundle (CSV, verification JSON, bundle.json, reconstructions) and original dataset archive. Original images are NOT in the bundle. In a new session, extract the same original archive to DATA_ROOT, then run Cells 1-3. Reconstruction checksums are verified during the required copy from Drive; original images are not rehashed. A changed dataset requires a new preparation, not the unchanged-data flag.

## 4. Bounded GPU benchmark
This uses pretrained CLIP, matching training. Its first use downloads authentic weights to the Colab cache; complete checkpoint restore does not download them.

```python
assert torch.cuda.is_available(), 'Select a GPU runtime.'
BATCH = 32
WORKERS = min(2, os.cpu_count() or 1)
ACCUM = 1
ALPHA = 0.3 if TRAINING_MODE == 'drct' else 0.0
subprocess.run([sys.executable, '-u', 'tools/benchmark_drct.py',
    '--manifest', str(MANIFEST), '--dataroot', str(DATA_ROOT),
    '--drct_backbone', 'clip_vit_l14', '--pretrained', '--drct_allow_download',
    '--drct_alpha', str(ALPHA), '--batch_size', str(BATCH), '--grad_accum_steps', str(ACCUM),
    '--num_workers', str(WORKERS), '--use_amp', '--warmup_steps', '5', '--measure_steps', '20',
    '--output', str(DRIVE_ROOT / 'benchmark.json')], check=True)
```

Compare throughput at batch 32 and 64 if memory permits. GPU memory usage alone is not utilization: the frozen encoder needs less activation memory. Keep images local and measure before adding workers.

## 5. NEW training run (skip when resuming)
The helper records the complete command, code commit, data digests and actual unique `name_runid/checkpoints` path. Checkpoints are written atomically. It mirrors the whole experiment every minute and on normal exit/caught interruption. Sudden runtime loss can lose work since the last completed backup.

```python
EPOCHS = 5
training_args = [
    '--arch', 'DRCT_128', '--drct_backbone', 'clip_vit_l14', '--pretrained', '--drct_allow_download',
    '--embed_dim', '128', '--drct_alpha', str(ALPHA), '--drct_pos_margin', '0', '--drct_neg_margin', '1',
    '--dataroot', str(DATA_ROOT), '--manifest_split', 'train', '--val_manifest_split', 'dev',
    '--allow_source_overlap', '--allow_aggregate_sources', '--monitor_metric', 'auc', '--class_bal',
    '--gpu_ids', '0', '--batch_size', str(BATCH), '--grad_accum_steps', str(ACCUM),
    '--num_workers', str(WORKERS), '--pin_memory', '--use_amp', '--amp_dtype', 'fp16',
    '--val_precision', 'fp32', '--lr', '0.0001', '--optim', 'adam',
    '--epochs', str(EPOCHS), '--epochs_decay', '0', '--seed', '42', '--save_epoch_freq', '1',
]
run = create_run(REPO_ROOT, MANIFEST, OUTPUT_BASE, DRIVE_RUNS,
                 'drct128' if TRAINING_MODE == 'drct' else 'clip128_control', training_args)
RUN_BACKUP = Path(run['drive_dir'])
(RUN_BACKUP / 'environment.txt').write_text(
    subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
print('SAVE THIS RUN PATH:', RUN_BACKUP)
BEST = run_training(run, REPO_ROOT)
print('Best checkpoint:', BEST)
```

The overlap flags explicitly identify an engineering pilot with internal development from the same collection. They do not certify source independence. Training uses binary class balancing. The preprocessing is the versioned adaptation recipe, not verified upstream reproduction.

## 6. RESUME (skip Cell 5)
Restore original data and preparation first. Set the exact previous backup directory. Do not create a new descriptor. All training settings, including total epochs and AMP, come from the saved command. Changed data/code identity fails.

```python
RUN_BACKUP = Path('/content/drive/MyDrive/deepfake_drct_128/runs/PASTE_EXACT_RUN_DIRECTORY')
run = read_run(RUN_BACKUP)
print('Required code commit:', run['commit'])
# If a FRESH clean clone differs, check out this commit and restart the kernel before importing project code:
# subprocess.run(['git', 'checkout', '--detach', run['commit']], cwd=REPO_ROOT, check=True)
BEST = run_training(run, REPO_ROOT, resume=True)
```

## 7. Internal dev evaluation and logit calibration
Use explicit dev metadata. One FP32 pass, with expensive plotting disabled.

```python
EVAL_ROOT = Path(run['experiment_dir']) / 'internal_eval'
subprocess.run([sys.executable, '-u', 'evaluate.py', '--checkpoint', str(BEST),
    '--arch', 'DRCT_128', '--manifest', str(MANIFEST), '--split', 'dev',
    '--batch_size', str(BATCH), '--num_workers', '0', '--eval_precision', 'fp32',
    '--no_plots', '--no_tsne', '--no_gradcam', '--no_profiling',
    '--output_dir', str(EVAL_ROOT)], check=True)
DEV_PREDS = EVAL_ROOT / 'DRCT_128' / 'predictions.csv'
THRESHOLD = BEST.parent / 'threshold_logit.json'
subprocess.run([sys.executable, 'analyze_predictions.py', 'calibrate',
    '--predictions', str(DEV_PREDS), '--score_space', 'logit',
    '--eval_precision', 'fp32', '--output', str(THRESHOLD)], check=True)
mirror_tree(Path(run['experiment_dir']), Path(run['drive_dir']))
```

## 8. External development scoring
This collection has informed past experiments; report it as external development. Keep the threshold frozen from internal dev.

```python
EXTERNAL_ROOT = Path('/content/unseenDataset/val to be deleted 2')
assert EXTERNAL_ROOT.is_dir(), EXTERNAL_ROOT
subprocess.run([sys.executable, '-u', 'evaluate.py', '--checkpoint', str(BEST),
    '--arch', 'DRCT_128', '--val_root', str(EXTERNAL_ROOT), '--split', 'external_dev',
    '--threshold_file', str(THRESHOLD), '--batch_size', str(BATCH), '--num_workers', '0',
    '--eval_precision', 'fp32', '--no_plots', '--no_tsne', '--no_gradcam', '--no_profiling',
    '--output_dir', str(Path(run['experiment_dir']) / 'external_eval')], check=True)
mirror_tree(Path(run['experiment_dir']), Path(run['drive_dir']))
```

## 9. Optional MHA, after the standalone result
Use a compatible trained wavelet128 checkpoint. Train a new fusion head; an old Wang-based head cannot simply be reused. Keep the standalone backup. Experts are frozen and embedded in the fusion checkpoint.

```python
WAVELET = Path('/content/drive/MyDrive/wavelet_checkpoints/best.pth')
assert WAVELET.is_file(), WAVELET
fusion_args = [
    '--arch', 'MHA_128', '--fusion_type', 'token_attention', '--embed_dim', '128',
    '--rgb_model_path', str(BEST), '--wavelet_model_path', str(WAVELET),
    '--drct_backbone', 'clip_vit_l14', '--dataroot', str(DATA_ROOT),
    '--manifest_split', 'train', '--val_manifest_split', 'dev',
    '--allow_source_overlap', '--allow_aggregate_sources', '--monitor_metric', 'auc', '--class_bal',
    '--gpu_ids', '0', '--batch_size', str(BATCH), '--num_workers', str(WORKERS),
    '--use_amp', '--amp_dtype', 'fp16', '--val_precision', 'fp32',
    '--optim', 'adam', '--lr', '0.0001', '--epochs', '5', '--epochs_decay', '0', '--seed', '42',
]
fusion_run = create_run(REPO_ROOT, MANIFEST, OUTPUT_BASE, DRIVE_RUNS, 'mha_drct128', fusion_args)
print('Fusion backup:', fusion_run['drive_dir'])
FUSION_BEST = run_training(fusion_run, REPO_ROOT)
```

Resume fusion through the same helper; restore its source expert paths if the current fusion constructor requires them. Evaluate/calibrate with `--arch MHA_128`, its own dev predictions and its own threshold. Do not apply the standalone threshold to fusion scores.
