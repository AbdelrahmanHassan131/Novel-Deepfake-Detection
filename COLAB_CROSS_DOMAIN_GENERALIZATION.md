# Colab L4 / T4 Playbook: Cross-Domain Deepfake Generalization (EFFORT-128 & Paired-Reconstruction DRCT-128)

This guide runs the **two-stage cross-domain generalization pipeline** designed to generalize from **CelebA + ADM/DDIM/DDPM/DiffSwap** to **unseen generative domains (e.g., LDM, SDXL, DFDC)** on a single **NVIDIA L4 (24 GB)** or **T4 (16 GB)** GPU.

- **Stage 1 (Fast, works immediately on your existing dataset — ~20–30 min on L4):**
  Trains **`EFFORT_128`** — OpenAI CLIP ViT-L/14 with **RINE-style Multi-Layer Intermediate Feature Fusion** + **EFFORT Orthogonal SVD Residual Adaptation** + **Symmetric Forensic Augmentations** (native CLIP norm, mixed native-scale/resized crops, symmetric JPEG/blur/resampling/color jitter, label smoothing, AdamW).
- **Stage 2 (Biggest boost for CelebA → LDM — ~10 min prep + ~30 min train on L4):**
  Uses `tools/generate_reconstructions.py` to generate **paired Stable Diffusion 1.5 + SDXL VAE reconstructions** of your CelebA real images (same face, same crop, same identity, zero dataset shortcut!), then trains **Paired-Reconstruction `EFFORT_128` (contrastive)** and **`DRCT_128` (with `--drct_aug_mode forensic`)**.
- Both models output the exact **128-D embedding + single raw binary logit (`real=0, fake=1`)**, so either can plug directly into **`MHA_128`** fusion and your existing `evaluate.py` / `analyze_predictions.py` calibration pipeline.

---

## Cell 1: Mount Drive, Clone Repo, Install Dependencies

```python
from google.colab import drive
drive.mount('/content/drive')

import os, sys, json, subprocess, runpy, gc, shutil
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

REPO_ROOT = Path('/content/Novel-Deepfake-Detection')
if not REPO_ROOT.exists():
    subprocess.run([
        'git', 'clone', '--branch', 'GeneralizationDeepFakeDetection',
        'https://github.com/AbdelrahmanHassan131/Novel-Deepfake-Detection.git',
        str(REPO_ROOT),
    ], check=True)
else:
    subprocess.run(['git', 'pull', '--ff-only', 'origin', 'GeneralizationDeepFakeDetection'], cwd=REPO_ROOT, check=True)

os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Install dependencies (keep Colab's matched PyTorch/torchvision)
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q',
    'git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33',
    'diffusers>=0.27.0', 'accelerate', 'safetensors',
    'scikit-learn', 'scipy', 'pillow', 'opencv-python-headless', 'tqdm', 'timm',
], check=True)

import torch
assert torch.cuda.is_available(), 'Select an L4 or T4 GPU runtime first!'
GPU_NAME = torch.cuda.get_device_name(0)
print('=' * 65)
print(f'GPU:    {GPU_NAME}')
print(f'Torch:  {torch.__version__}')
print(f'Commit: {subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}')
print('=' * 65)

def run_python(script, args):
    previous_argv = sys.argv[:]
    sys.argv = [str(REPO_ROOT / script), *map(str, args)]
    try:
        runpy.run_path(str(REPO_ROOT / script), run_name='__main__')
    finally:
        sys.argv = previous_argv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

---

## Cell 2: Extract Dataset & Build a Clean Manifest (or Leave-One-Generator-Out Split)

> **Critical Rule:** Never pick `best.pth` on an in-domain validation split if you have generator subfolders (`ADM`, `DDIM`, `DDPM`, `DiffSwap`). Hold out one unseen generator (e.g. `DDIM`) as your `dev` split so `best.pth` is selected for **cross-generator** generalization!

```python
DATA_ROOT = Path('/content/dataset/DatesetNoVideosNoGAN')
BASE_ZIP = Path('/content/drive/MyDrive/DatasetNoVideoNoGan/DatesetNoVideosNoGAN.zip')
DRIVE_OUT = Path('/content/drive/MyDrive/deepfake_cross_domain_generalization')
PREP_DIR = Path('/content/generalization_prep')
for p in (DRIVE_OUT, DRIVE_OUT / 'runs', PREP_DIR):
    p.mkdir(parents=True, exist_ok=True)

if not (DATA_ROOT / 'train/real').is_dir():
    assert BASE_ZIP.is_file(), f'Update BASE_ZIP to your dataset archive: {BASE_ZIP}'
    Path('/content/dataset').mkdir(parents=True, exist_ok=True)
    subprocess.run(['7z', 'x', str(BASE_ZIP), '-o/content/dataset/', '-y'], check=True, stdout=subprocess.DEVNULL)

# Option A (RECOMMENDED if train/fake has generator subfolders or generator filename prefixes):
# Build a Leave-One-Generator-Out manifest holding out e.g. DDIM for dev and LDM for final test.
USE_LOGO_SPLIT = False  # Set True if your fake folder has generator names like ADM, DDPM, DDIM, DiffSwap

BASE_MANIFEST = PREP_DIR / 'base_inventory.csv'
if USE_LOGO_SPLIT:
    run_python('tools/build_logo_manifest.py', [
        '--real_dir', str(DATA_ROOT / 'train/real'),
        '--fake_dir', str(DATA_ROOT / 'train/fake'),
        '--output_manifest', str(BASE_MANIFEST),
        '--dev_generators', 'DDIM',
        '--test_generators', 'LDM',
        '--seed', '42',
    ])
else:
    from data.drct_preparation import inventory_dataset_directory, generate_gate_sidecar
    if not BASE_MANIFEST.is_file():
        inventory_dataset_directory(DATA_ROOT, BASE_MANIFEST, duplicate_policy='drop_train')
        generate_gate_sidecar(BASE_MANIFEST, require_hashes=True)

print('Verified base manifest ready:', BASE_MANIFEST)
```

---

## Cell 3 (Stage 1 — Immediate): Train `EFFORT_128` (Multi-Layer + Orthogonal SVD CLIP ViT-L/14)

Works **immediately** on your existing dataset with zero reconstruction preprocessing:

- Taps 4 evenly spaced CLIP ViT-L/14 blocks (`[5, 11, 17, 23]`) + final visual token
- Applies Orthogonal SVD Residual Adaptation (`rank=16` on last 12 blocks, ~1.9M trainable params)
- Applies symmetric forensic augmentations (native CLIP norm + native/resized crop mix + JPEG + blur + downscale-upscale + mild color jitter)
- Uses `AdamW` + cosine schedule + label smoothing (`0.05`) + orthogonality regularization (`0.1`)

```python
is_t4 = 'T4' in GPU_NAME.upper()
BATCH = 16 if is_t4 else 32
ACCUM = 2 if is_t4 else 1  # Effective batch = 32
AMP_DTYPE = 'fp16' if is_t4 else 'bf16'
WORKERS = min(2, os.cpu_count() or 1)

RUN_NAME = 'effort128_stage1_clip_vitl14'
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
EXP_DIR_S1 = DRIVE_OUT / 'runs' / f'{RUN_NAME}_{RUN_ID}'
BEST_S1 = EXP_DIR_S1 / 'checkpoints/best.pth'

run_python('train.py', [
    '--arch', 'EFFORT_128',
    '--drct_backbone', 'clip_vit_l14', '--pretrained', '--drct_allow_download',
    '--embed_dim', '128',
    '--effort_svd_rank', '16', '--effort_svd_blocks', '12',
    '--effort_multi_layer', '--effort_num_layers', '4',
    '--effort_ortho_weight', '0.1', '--effort_label_smoothing', '0.05',
    '--effort_dropout', '0.3',
    '--drct_alpha', '0.0',  # Pure smoothed BCE + ortho loss in Stage 1
    '--dataroot', str(DATA_ROOT),
    '--manifest', str(BASE_MANIFEST), '--manifest_split', 'train',
    '--val_manifest', str(BASE_MANIFEST), '--val_manifest_split', 'dev',
    '--allow_source_overlap', '--allow_aggregate_sources',
    '--name', RUN_NAME, '--run_id', RUN_ID, '--checkpoints_dir', str(DRIVE_OUT / 'runs'),
    '--gpu_ids', '0', '--batch_size', str(BATCH), '--grad_accum_steps', str(ACCUM),
    '--val_batch_size', str(BATCH), '--num_workers', str(WORKERS), '--pin_memory',
    '--class_bal', '--optim', 'adamw', '--lr', '0.0002', '--backbone_lr_mult', '0.2',
    '--weight_decay', '0.01', '--lr_policy', 'cosine',
    '--epochs', '5', '--epochs_decay', '0', '--monitor_metric', 'auc',
    '--use_amp', '--amp_dtype', AMP_DTYPE, '--val_precision', 'fp32',
    '--seed', '42', '--log_freq', '50', '--val_epoch_freq', '1', '--save_epoch_freq', '1',
])

assert BEST_S1.is_file(), f'Missing Stage 1 checkpoint: {BEST_S1}'
print('Stage 1 EFFORT_128 best checkpoint:', BEST_S1)
```

---

## Cell 4 (Stage 2 — Game Changer for CelebA → LDM): Generate Paired SD-1.5 + SDXL VAE Reconstructions

Passing your CelebA real images through **SD-1.5 VAE + SDXL VAE** creates paired fakes of the **exact same faces** (~5–8 minutes on L4 for 20,000 per class!). This breaks the _"is this CelebA?"_ shortcut and teaches Latent Diffusion VAE artifacts directly.

```python
REC_DIR = Path('/content/dataset/reconstructions_multi_vae')
MAPPING_CSV = DRIVE_OUT / 'reconstruction_mapping_multi_vae.csv'
DRCT_MANIFEST = PREP_DIR / 'paired_recon_manifest.csv'

# Start with 20,000 real + 20,000 fake pairs (~5-8 min on L4; set 0 to reconstruct all)
run_python('tools/generate_reconstructions.py', [
    '--inventory', str(BASE_MANIFEST),
    '--output_dir', str(REC_DIR),
    '--mapping_csv', str(MAPPING_CSV),
    '--backend', 'multi_vae',       # 50% SD-1.5 VAE + 50% SDXL VAE (use 'sd_turbo' for full 2-step diffusion)
    '--max_per_class', '20000',
    '--reconstruct_fakes',
    '--batch_size', '32',
    '--work_size', '256',
    '--save_format', 'match_parent',
    '--device', 'cuda', '--dtype', 'fp16',
    '--import_manifest', str(DRCT_MANIFEST),
])

print('Verified paired-reconstruction manifest:', DRCT_MANIFEST)
```

---

## Cell 5 (Stage 2 Training): Train Paired-Reconstruction `EFFORT_128` (Contrastive) & `DRCT_128` (Forensic)

```python
# 5A. EFFORT_128 + Paired Reconstructions + Contrastive Loss (Recommended Champion)
RUN_NAME_S2 = 'effort128_stage2_paired_contrastive'
RUN_ID_S2 = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
EXP_DIR_S2 = DRIVE_OUT / 'runs' / f'{RUN_NAME_S2}_{RUN_ID_S2}'
BEST_S2 = EXP_DIR_S2 / 'checkpoints/best.pth'

run_python('train.py', [
    '--arch', 'EFFORT_128',
    '--drct_backbone', 'clip_vit_l14', '--pretrained', '--drct_allow_download',
    '--embed_dim', '128',
    '--effort_svd_rank', '16', '--effort_svd_blocks', '12',
    '--effort_multi_layer', '--effort_num_layers', '4',
    '--effort_ortho_weight', '0.1', '--effort_label_smoothing', '0.05',
    '--drct_alpha', '0.3', '--drct_pos_margin', '0.0', '--drct_neg_margin', '1.0',
    '--dataroot', str(DATA_ROOT),
    '--manifest', str(DRCT_MANIFEST), '--manifest_split', 'train',
    '--val_manifest', str(DRCT_MANIFEST), '--val_manifest_split', 'dev',
    '--allow_source_overlap', '--allow_aggregate_sources',
    '--name', RUN_NAME_S2, '--run_id', RUN_ID_S2, '--checkpoints_dir', str(DRIVE_OUT / 'runs'),
    '--gpu_ids', '0', '--batch_size', str(BATCH), '--grad_accum_steps', str(ACCUM),
    '--val_batch_size', str(BATCH), '--num_workers', str(WORKERS), '--pin_memory',
    '--class_bal', '--optim', 'adamw', '--lr', '0.0002', '--backbone_lr_mult', '0.2',
    '--weight_decay', '0.01', '--lr_policy', 'cosine',
    '--epochs', '5', '--epochs_decay', '0', '--monitor_metric', 'auc',
    '--use_amp', '--amp_dtype', AMP_DTYPE, '--val_precision', 'fp32',
    '--seed', '42', '--log_freq', '50', '--val_epoch_freq', '1', '--save_epoch_freq', '1',
])

# 5B. Full DRCT_128 with Forensic Augmentations (Thesis Ablation Comparator)
# Note: DRCT_128 requires 100% of train originals in its manifest to have reconstructions.
# Since Cell 4 used --max_per_class 20000, build a clean 100%-paired 80K train subset (20K real + 20K fake + 40K recon) + full dev split:
from data.drct_manifest import read_drct_manifest, write_drct_manifest
from data.drct_preparation import generate_gate_sidecar, check_drct_readiness

all_rows = read_drct_manifest(DRCT_MANIFEST)
reconstructed_parents = {
    r['parent_id'] for r in all_rows
    if r['split'] == 'train' and r['sample_kind'] in ('real_reconstruction', 'fake_reconstruction')
}
drct_paired_rows = [
    r for r in all_rows
    if r['split'] != 'train' or r['sample_id'] in reconstructed_parents or r['parent_id'] in reconstructed_parents
]
DRCT_PAIRED_MANIFEST = PREP_DIR / 'drct_fully_paired_manifest.csv'
write_drct_manifest(DRCT_PAIRED_MANIFEST, drct_paired_rows)
generate_gate_sidecar(DRCT_PAIRED_MANIFEST, require_hashes=False)
print('DRCT 100%-paired readiness:', check_drct_readiness(DRCT_PAIRED_MANIFEST, split='train')['is_ready_for_full_drct'])

RUN_NAME_DRCT = 'drct128_stage2_paired_forensic'
RUN_ID_DRCT = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
EXP_DIR_DRCT = DRIVE_OUT / 'runs' / f'{RUN_NAME_DRCT}_{RUN_ID_DRCT}'
BEST_DRCT = EXP_DIR_DRCT / 'checkpoints/best.pth'

run_python('train.py', [
    '--arch', 'DRCT_128',
    '--drct_backbone', 'clip_vit_l14', '--pretrained', '--drct_allow_download',
    '--drct_aug_mode', 'forensic',  # Native CLIP norm + symmetric JPEG/blur/crop augmentations
    '--embed_dim', '128',
    '--drct_alpha', '0.3', '--drct_pos_margin', '0.0', '--drct_neg_margin', '1.0',
    '--dataroot', str(DATA_ROOT),
    '--manifest', str(DRCT_PAIRED_MANIFEST), '--manifest_split', 'train',
    '--val_manifest', str(DRCT_PAIRED_MANIFEST), '--val_manifest_split', 'dev',
    '--allow_source_overlap', '--allow_aggregate_sources',
    '--name', RUN_NAME_DRCT, '--run_id', RUN_ID_DRCT, '--checkpoints_dir', str(DRIVE_OUT / 'runs'),
    '--gpu_ids', '0', '--batch_size', str(BATCH), '--grad_accum_steps', str(ACCUM),
    '--val_batch_size', str(BATCH), '--num_workers', str(WORKERS), '--pin_memory',
    '--class_bal', '--optim', 'adamw', '--lr', '0.0001', '--weight_decay', '0.01', '--lr_policy', 'cosine',
    '--epochs', '5', '--epochs_decay', '0', '--monitor_metric', 'auc',
    '--use_amp', '--amp_dtype', AMP_DTYPE, '--val_precision', 'fp32',
    '--seed', '42', '--log_freq', '50', '--val_epoch_freq', '1', '--save_epoch_freq', '1',
])
```

---

## Cell 6: Calibrate on Dev & Evaluate on Unseen Domain (LDM / DFDC / External Test)

```python
# Point UNSEEN_ROOT to your unseen domain folder (containing real/ and fake/, e.g. LDM or DFDC)
UNSEEN_ROOT = Path('/content/unseenDataset/val to be deleted 2')

for label, arch, exp_dir, manifest_used in [
    ('Stage1_EFFORT_128', 'EFFORT_128', EXP_DIR_S1, BASE_MANIFEST),
    ('Stage2_EFFORT_128_Recon', 'EFFORT_128', EXP_DIR_S2, DRCT_MANIFEST),
    ('Stage2_DRCT_128_Forensic', 'DRCT_128', EXP_DIR_DRCT, DRCT_MANIFEST),
]:
    best_ckpt = exp_dir / 'checkpoints/best.pth'
    if not best_ckpt.is_file():
        continue

    dev_out = exp_dir / 'dev_eval'
    run_python('evaluate.py', [
        '--arch', arch, '--checkpoint', str(best_ckpt),
        '--manifest', str(manifest_used), '--split', 'dev',
        '--device', 'cuda:0', '--batch_size', str(BATCH), '--num_workers', '0',
        '--eval_precision', 'fp32', '--no_plots', '--no_tsne', '--no_gradcam', '--no_profiling',
        '--output_dir', str(dev_out),
    ])

    thresh_json = best_ckpt.parent / 'threshold_logit.json'
    run_python('analyze_predictions.py', [
        'calibrate', '--predictions', str(dev_out / arch / 'predictions.csv'),
        '--score_space', 'logit', '--eval_precision', 'fp32', '--output', str(thresh_json),
    ])

    if UNSEEN_ROOT.is_dir():
        ext_out = exp_dir / 'external_unseen_eval'
        run_python('evaluate.py', [
            '--arch', arch, '--checkpoint', str(best_ckpt),
            '--val_root', str(UNSEEN_ROOT), '--split', 'external_test',
            '--threshold_file', str(thresh_json),
            '--device', 'cuda:0', '--batch_size', str(BATCH), '--num_workers', '0',
            '--eval_precision', 'fp32', '--no_plots', '--no_tsne', '--no_gradcam', '--no_profiling',
            '--output_dir', str(ext_out),
        ])
        rep = json.loads((ext_out / arch / 'generalization_report.json').read_text())['overall']
        print(
            f"[{label}] ROC AUC={rep['roc_auc']:.4f} | BalAcc={rep['balanced_accuracy']*100:.2f}% | "
            f"EER={rep['eer']*100:.2f}% | RealRec={rep['per_class']['real']['recall']*100:.2f}% | "
            f"FakeRec={rep['per_class']['fake']['recall']*100:.2f}%"
        )
```

---

## Cell 7 (Breakthrough Upgrade): Frozen Multi-Layer CLIP (`--effort_svd_rank 0`) + Resolution-Harmonized `DRCT_128`

When `train/real` is `178x218 JPG` (CelebA) and `train/fake` is `256x256 PNG` (ADM/DDIM/DDPM):
1. Keeping CLIP ViT-L/14 **100% frozen** (`DRCT_128` or `EFFORT_128 --effort_svd_rank 0`) prevents the backbone from memorizing the `178x218` vs `256x256` shortcut (achieving **0.9365 ROC AUC / 14.02% EER** on 60K unseen images!).
2. Updating `CLIPForensicTrainTransform` to harmonize effective resolution (`160..224`) + `80%` JPEG compression removes the residual threshold shift so `RealRec` and `FakeRec` both stay high at the dev-calibrated threshold.

```python
# Pull the resolution-harmonized transform fix first!
subprocess.run(['git', 'pull', '--ff-only', 'origin', 'GeneralizationDeepFakeDetection'], cwd=REPO_ROOT, check=True)

# 7A. Frozen Multi-Layer CLIP (RINE-128 style: effort_svd_rank=0) on the 100%-paired 80K manifest
RUN_NAME_RINE = 'effort128_frozen_multilayer_paired80k'
RUN_ID_RINE = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
EXP_DIR_RINE = DRIVE_OUT / 'runs' / f'{RUN_NAME_RINE}_{RUN_ID_RINE}'

run_python('train.py', [
    '--arch', 'EFFORT_128',
    '--drct_backbone', 'clip_vit_l14', '--pretrained', '--drct_allow_download',
    '--embed_dim', '128',
    '--effort_svd_rank', '0',           # 100% FROZEN CLIP ViT-L/14 (zero shortcut memorization!)
    '--effort_multi_layer', '--effort_num_layers', '4',
    '--effort_ortho_weight', '0.0', '--effort_label_smoothing', '0.05', '--effort_dropout', '0.3',
    '--drct_alpha', '0.3', '--drct_pos_margin', '0.0', '--drct_neg_margin', '1.0',
    '--dataroot', str(DATA_ROOT),
    '--manifest', str(DRCT_PAIRED_MANIFEST), '--manifest_split', 'train',
    '--val_manifest', str(DRCT_PAIRED_MANIFEST), '--val_manifest_split', 'dev',
    '--allow_source_overlap', '--allow_aggregate_sources',
    '--name', RUN_NAME_RINE, '--run_id', RUN_ID_RINE, '--checkpoints_dir', str(DRIVE_OUT / 'runs'),
    '--gpu_ids', '0', '--batch_size', str(BATCH), '--grad_accum_steps', str(ACCUM),
    '--val_batch_size', str(BATCH), '--num_workers', str(WORKERS), '--pin_memory',
    '--class_bal', '--optim', 'adamw', '--lr', '0.0002', '--weight_decay', '0.01', '--lr_policy', 'cosine',
    '--epochs', '5', '--epochs_decay', '0', '--monitor_metric', 'auc',
    '--use_amp', '--amp_dtype', AMP_DTYPE, '--val_precision', 'fp32',
    '--seed', '42', '--log_freq', '50', '--val_epoch_freq', '1', '--save_epoch_freq', '1',
])

# 7B. Pure Paired-Real-Only DRCT_128 (20K Real CelebA 178x218 vs 20K Multi-VAE Reconstructed CelebA 178x218)
# ZERO 178x218-vs-256x256 and ZERO JPG-vs-PNG difference between Real and Fake!
from data.drct_manifest import read_drct_manifest, write_drct_manifest
from data.drct_preparation import generate_gate_sidecar

paired_rows = read_drct_manifest(DRCT_PAIRED_MANIFEST)
pure_real_pair_train = [
    r for r in paired_rows
    if r['split'] == 'train' and r['sample_kind'] in ('real_original', 'real_reconstruction')
]
# Split 10% of paired real+recon as matched dev set (zero resolution shift in threshold calibration!)
pair_ids = sorted({r['parent_id'] for r in pure_real_pair_train})
dev_pair_ids = set(pair_ids[:max(1, len(pair_ids) // 10)])
pure_manifest_rows = []
for r in pure_real_pair_train:
    rc = dict(r)
    rc['split'] = 'dev' if r['parent_id'] in dev_pair_ids else 'train'
    pure_manifest_rows.append(rc)

PURE_PAIRED_MANIFEST = PREP_DIR / 'pure_real_vs_recon_manifest.csv'
write_drct_manifest(PURE_PAIRED_MANIFEST, pure_manifest_rows)
generate_gate_sidecar(PURE_PAIRED_MANIFEST, require_hashes=False)

RUN_NAME_PURE = 'effort128_frozen_multilayer_pure_recon_only'
RUN_ID_PURE = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
EXP_DIR_PURE = DRIVE_OUT / 'runs' / f'{RUN_NAME_PURE}_{RUN_ID_PURE}'

run_python('train.py', [
    '--arch', 'EFFORT_128',
    '--drct_backbone', 'clip_vit_l14', '--pretrained', '--drct_allow_download',
    '--embed_dim', '128',
    '--effort_svd_rank', '0',
    '--effort_multi_layer', '--effort_num_layers', '4',
    '--effort_ortho_weight', '0.0', '--effort_label_smoothing', '0.05', '--effort_dropout', '0.3',
    '--drct_alpha', '0.3', '--drct_pos_margin', '0.0', '--drct_neg_margin', '1.0',
    '--dataroot', str(DATA_ROOT),
    '--manifest', str(PURE_PAIRED_MANIFEST), '--manifest_split', 'train',
    '--val_manifest', str(PURE_PAIRED_MANIFEST), '--val_manifest_split', 'dev',
    '--allow_source_overlap', '--allow_aggregate_sources',
    '--name', RUN_NAME_PURE, '--run_id', RUN_ID_PURE, '--checkpoints_dir', str(DRIVE_OUT / 'runs'),
    '--gpu_ids', '0', '--batch_size', str(BATCH), '--grad_accum_steps', str(ACCUM),
    '--val_batch_size', str(BATCH), '--num_workers', str(WORKERS), '--pin_memory',
    '--class_bal', '--optim', 'adamw', '--lr', '0.0002', '--weight_decay', '0.01', '--lr_policy', 'cosine',
    '--epochs', '5', '--epochs_decay', '0', '--monitor_metric', 'auc',
    '--use_amp', '--amp_dtype', AMP_DTYPE, '--val_precision', 'fp32',
    '--seed', '42', '--log_freq', '50', '--val_epoch_freq', '1', '--save_epoch_freq', '1',
])
```
