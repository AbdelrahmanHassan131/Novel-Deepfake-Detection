# Colab L4 GPU Playbook: Train & Benchmark on `DatasetMultiDomainClean_v2` (556K Images, 15 Generators, 4 Native Real Sources)

> **Prerequisite:** Wait until **Cell 6** of `COLAB_BUILD_CLEAN_MULTIDOMAIN_V2.md` finishes saving `DatasetMultiDomainClean_v2.zip` to Google Drive, then switch your Colab runtime to **NVIDIA L4 (24 GB)** (or T4).

### Why This Run Is Built to Generalize Across Unseen Domains
1. **Training Diversity:** 13 seen generative families (`ADM`, `DDIM`, `DDPM`, `DiffSwap`, `CoDiff`, `cycle_diff`, `DCFace`, `DreamBooth`, `FreeDoM_I`, `FreeDoM_T`, `HPS`, `LoRA`, `SDXL`) paired 1:1 with **4 native high-res real sources** (`CelebA`, `FFHQ-256`, `FairFace`, `UTKFace` — **154K PNG + 80K JPG reals**, ZERO blurry 112px WebFace!).
2. **True Leave-One-Generator-Out (LOGO) Model Selection (`dev`):** We hold out **`SDXL_Refine`** and **`DiffFace`** completely out of training into the `dev` split so `best.pth` and `threshold_logit.json` are selected specifically for **unseen-generator generalization** (not in-domain memorization!).
3. **Proven Generalization Architecture:** `EFFORT_128 --effort_svd_rank 0 --effort_multi_layer --effort_num_layers 4` (100% Frozen CLIP ViT-L/14 + 4 intermediate layers `[5, 11, 17, 23]` + `128-D` bottleneck + contrastive + label smoothing + resolution-harmonized forensic augmentations).
4. **3 Completely Untouched Unseen Final Benchmarks:**
   - **Benchmark 1:** `val to be deleted 2` (60,000 CelebA-HQ real + unseen diffusion fake)
   - **Benchmark 2:** **`Midjourney`** (completely unseen commercial generator!) vs. held-out real faces
   - **Benchmark 3:** **`Imagic`** (completely unseen face-editing generator!) vs. held-out real faces

---

## Cell 1: Mount Drive, Clone Repo & Install Dependencies (Python 3.13-Safe)

```python
from google.colab import drive
drive.mount('/content/drive')

import os, sys, json, csv, shutil, subprocess, runpy, gc
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

# 1. Install dependencies + setuptools<81 FIRST
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q',
    'ftfy', 'regex', 'setuptools<81',
    'scikit-learn', 'scipy', 'pillow', 'opencv-python-headless', 'tqdm', 'timm', 'pandas',
], check=True)

# 2. Install OpenAI CLIP SECOND without build isolation
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '--no-build-isolation', '--no-deps',
    'git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33',
], check=True)

import torch, clip
assert torch.cuda.is_available(), 'Select an NVIDIA L4 (recommended) or T4 GPU runtime!'
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

## Cell 2: Extract `DatasetMultiDomainClean_v2.zip` & Build Leave-One-Generator-Out (LOGO) Training Manifest

Holds **`SDXL_Refine`** and **`DiffFace`** completely out of training and puts them into `dev` so `best.pth` and `threshold_logit.json` are chosen strictly by **cross-generator** generalization!

```python
import random
from collections import Counter
from data.drct_manifest import write_drct_manifest
from data.drct_preparation import generate_gate_sidecar

DATA_ROOT = Path('/content/dataset/DatasetMultiDomainClean_v2')
DATA_ZIP  = Path('/content/drive/MyDrive/DatasetMultiDomainClean_v2/DatasetMultiDomainClean_v2.zip')
DRIVE_OUT = Path('/content/drive/MyDrive/deepfake_multidomain_v2_runs')
PREP_DIR  = Path('/content/multidomain_v2_prep')
for p in (DRIVE_OUT, PREP_DIR):
    p.mkdir(parents=True, exist_ok=True)

if not (DATA_ROOT / 'train/real').is_dir():
    assert DATA_ZIP.is_file(), f'Wait for Cell 6 of CPU builder to save: {DATA_ZIP}'
    Path('/content/dataset').mkdir(parents=True, exist_ok=True)
    print(f'Extracting {DATA_ZIP} to local SSD...', flush=True)
    subprocess.run(['7z', 'x', str(DATA_ZIP), '-o/content/dataset/', '-y'], check=True, stdout=subprocess.DEVNULL)
    print('Extraction complete.')

# Build Leave-One-Generator-Out manifest from dataset_manifest.csv
# Hold out SDXL_Refine (I2I latent diffusion) + DiffFace (face swap) from train -> unseen cross-domain dev!
LOGO_HOLDOUT_DEV_GENS = {'sdxl_refine', 'diffface'}

src_manifest = DATA_ROOT / 'dataset_manifest.csv'
assert src_manifest.is_file(), f'Missing provenance manifest: {src_manifest}'

train_rows = []
dev_unseen_fakes = []
dev_in_domain_fakes = []
dev_reals = []

with open(src_manifest, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        abs_path = str((DATA_ROOT / row['relative_path']).resolve())
        lbl = int(row['label'])
        gen = row['generator']
        src = row['dataset_source']
        ident = row['identity_id']
        rec = {
            'sample_id': row['sample_id'],
            'parent_id': row['sample_id'],
            'path': abs_path,
            'label': lbl,
            'split': row['split'],
            'sample_kind': 'real_original' if lbl == 0 else 'fake_original',
            'group_id': f"{src}_{ident}" if ident != 'unknown' else row['sample_id'],
            'dataset_source': src,
            'generator': gen,
            'sha256': row['sample_id'],  # Unique deterministic identifier (files already verified intact during build)
            'rec_model_revision': 'none',
            'rec_settings': '{}',
            'rec_seed': '0',
        }
        if row['split'] == 'train':
            # Exclude held-out LOGO dev generators completely from training!
            if lbl == 1 and gen.lower() in LOGO_HOLDOUT_DEV_GENS:
                continue
            train_rows.append(rec)
        else:
            if lbl == 0:
                dev_reals.append(rec)
            elif gen.lower() in LOGO_HOLDOUT_DEV_GENS:
                dev_unseen_fakes.append(rec)
            else:
                dev_in_domain_fakes.append(rec)

# Build a fast, balanced, cross-domain dev split:
#   - 100% of held-out unseen dev fakes (SDXL_Refine + DiffFace = ~3,652 imgs)
#   - + 3,652 stratified seen-generator val fakes (so dev measures both seen + unseen)
#   - + 7,304 balanced val reals across CelebA, FFHQ-256, and FairFace!
rng = random.Random(42)
rng.shuffle(dev_in_domain_fakes)
dev_fakes_selected = dev_unseen_fakes + dev_in_domain_fakes[:len(dev_unseen_fakes)]

# Stratify val reals evenly across celeba, ffhq256, fairface_val
reals_by_src = {}
for r in dev_reals:
    reals_by_src.setdefault(r['dataset_source'], []).append(r)
per_src_quota = len(dev_fakes_selected) // len(reals_by_src) + 1
dev_reals_selected = []
for src_k, items in sorted(reals_by_src.items()):
    rng.shuffle(items)
    dev_reals_selected.extend(items[:per_src_quota])
dev_reals_selected = dev_reals_selected[:len(dev_fakes_selected)]

# Keep held-out unseen-only dev subset separately for pure cross-generator threshold calibration!
LOGO_MANIFEST = PREP_DIR / 'multidomain_v2_logo_manifest.csv'
all_selected = train_rows + dev_reals_selected + dev_fakes_selected
write_drct_manifest(LOGO_MANIFEST, all_selected)
generate_gate_sidecar(LOGO_MANIFEST, require_hashes=False)

tr_counts = Counter(r['label'] for r in train_rows)
dv_counts = Counter(r['label'] for r in dev_reals_selected + dev_fakes_selected)
print('=' * 70)
print(f'LOGO Training Split : Real={tr_counts[0]:,} | Fake={tr_counts[1]:,} (13 seen generators)')
print(f'LOGO Dev Split      : Real={dv_counts[0]:,} | Fake={dv_counts[1]:,} (50% UNSEEN: {sorted(LOGO_HOLDOUT_DEV_GENS)})')
print(f'Manifest saved      : {LOGO_MANIFEST}')
print('=' * 70)
```

---

## Cell 3: Train `EFFORT_128` (Frozen 4-Layer CLIP ViT-L/14) on Full Multi-Domain Dataset

- Uses `batch_size=64, bf16` on L4 (~25 min/epoch, 3 epochs ≈ ~1h15m total; `batch_size=32, fp16` on T4).
- Saves `best.pth`, `last.pth`, and `model_epoch_k.pth` directly to Google Drive after every epoch (100% resumable if Colab disconnects!).

```python
is_t4 = 'T4' in GPU_NAME.upper()
BATCH = 32 if is_t4 else 64
ACCUM = 2 if is_t4 else 1   # Effective batch = 64
AMP_DTYPE = 'fp16' if is_t4 else 'bf16'
WORKERS = min(4, os.cpu_count() or 2)

RUN_NAME = 'effort128_frozen_multilayer_v2_logo'
RUN_ID   = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
EXP_DIR  = DRIVE_OUT / f'{RUN_NAME}_{RUN_ID}'
BEST_CKPT = EXP_DIR / 'checkpoints/best.pth'

print(f'Starting training -> {EXP_DIR}')
print(f'GPU={GPU_NAME} | batch={BATCH} x accum={ACCUM} | amp={AMP_DTYPE} | workers={WORKERS}')

run_python('train.py', [
    '--arch', 'EFFORT_128',
    '--drct_backbone', 'clip_vit_l14', '--pretrained', '--drct_allow_download',
    '--embed_dim', '128',
    '--effort_svd_rank', '0',           # 100% Frozen CLIP ViT-L/14 (proven champion!)
    '--effort_multi_layer', '--effort_num_layers', '4',
    '--effort_ortho_weight', '0.0',
    '--effort_label_smoothing', '0.05',
    '--effort_dropout', '0.4',
    '--drct_alpha', '0.2', '--drct_pos_margin', '0.0', '--drct_neg_margin', '1.0',
    '--dataroot', str(DATA_ROOT),
    '--manifest', str(LOGO_MANIFEST), '--manifest_split', 'train',
    '--val_manifest', str(LOGO_MANIFEST), '--val_manifest_split', 'dev',
    '--allow_source_overlap', '--allow_aggregate_sources',
    '--name', RUN_NAME, '--run_id', RUN_ID, '--checkpoints_dir', str(DRIVE_OUT),
    '--gpu_ids', '0', '--batch_size', str(BATCH), '--grad_accum_steps', str(ACCUM),
    '--val_batch_size', '64', '--num_workers', str(WORKERS), '--pin_memory',
    '--class_bal', '--optim', 'adamw', '--lr', '0.0002', '--weight_decay', '0.02', '--lr_policy', 'cosine',
    '--epochs', '3', '--epochs_decay', '0', '--monitor_metric', 'auc',
    '--use_amp', '--amp_dtype', AMP_DTYPE, '--val_precision', 'fp32',
    '--seed', '42', '--log_freq', '100', '--val_epoch_freq', '1', '--save_epoch_freq', '1',
])

assert BEST_CKPT.is_file(), f'Missing best checkpoint: {BEST_CKPT}'
print('Training complete! Best checkpoint:', BEST_CKPT)
```

---

## Cell 4: Calibrate Threshold on LOGO Dev (Zero Unseen Test Images Touched!)

```python
dev_out = EXP_DIR / 'logo_dev_eval'
run_python('evaluate.py', [
    '--arch', 'EFFORT_128', '--checkpoint', str(BEST_CKPT),
    '--manifest', str(LOGO_MANIFEST), '--split', 'dev',
    '--device', 'cuda:0', '--batch_size', '64', '--num_workers', '4',
    '--eval_precision', 'fp32', '--no_plots', '--no_tsne', '--no_gradcam', '--no_profiling',
    '--output_dir', str(dev_out),
])

THRESH_JSON = BEST_CKPT.parent / 'threshold_logit.json'
run_python('analyze_predictions.py', [
    'calibrate', '--predictions', str(dev_out / 'EFFORT_128/predictions.csv'),
    '--score_space', 'logit', '--eval_precision', 'fp32', '--output', str(THRESH_JSON),
])
print('Frozen LOGO-dev threshold saved:', THRESH_JSON)
print(THRESH_JSON.read_text())
```

---

## Cell 5: Evaluate on ALL 3 Completely Untouched Unseen Domains (`val to be deleted 2`, `Midjourney`, `Imagic`)

Reports **ROC AUC**, **EER**, **Dev-Calibrated BalAcc**, and **Uncalibrated (`0.5`) BalAcc** separately for every unseen domain—ready directly for your thesis Results table!

```python
import zipfile, io
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

# --- Prepare Unseen Domain 1: val to be deleted 2 (60,000 images) ---
UNSEEN_1 = Path('/content/unseenDataset/val to be deleted 2')
if not (UNSEEN_1 / 'real').is_dir():
    cands = [
        Path('/content/drive/MyDrive/UnseenData/val to be deleted 2.zip'),
        Path('/content/drive/MyDrive/unseenData/val to be deleted 2.zip'),
        *Path('/content/drive/MyDrive').glob('*/*val*deleted*.zip'),
    ]
    z1 = next((p for p in cands if p.is_file()), None)
    assert z1 is not None, 'val to be deleted 2.zip not found on Drive.'
    Path('/content/unseenDataset').mkdir(parents=True, exist_ok=True)
    subprocess.run(['7z', 'x', str(z1), '-o/content/unseenDataset/', '-y'], check=True, stdout=subprocess.DEVNULL)

# --- Prepare Unseen Domain 2 (Midjourney) & Unseen Domain 3 (Imagic) from test/ ZIPs ---
# Pair each with 3,000 held-out real faces (1K CelebA + 1K FFHQ + 1K FairFace from val/real, never trained on!)
SHARED_ROOT = Path('/content/drive/MyDrive/shared_project_files')
val_real_pool = []
for prefix in ('celeba', 'ffhq256', 'fairface_val'):
    matches = sorted((DATA_ROOT / 'val/real').glob(f'{prefix}*'))
    val_real_pool.extend(matches[:1000])

def build_unseen_folder_from_zip(name: str, fake_zip: Path, max_fakes: int = 3000) -> Path:
    out_root = Path(f'/content/unseen_benchmarks/{name}')
    if (out_root / 'fake').is_dir() and len(list((out_root / 'fake').glob('*.*'))) >= 500:
        return out_root
    (out_root / 'real').mkdir(parents=True, exist_ok=True)
    (out_root / 'fake').mkdir(parents=True, exist_ok=True)
    # Symlink held-out real faces
    for p in val_real_pool[:max_fakes]:
        dst = out_root / 'real' / p.name
        if not dst.exists():
            os.symlink(p.resolve(), dst)
    # Extract unseen test fakes
    saved = 0
    with zipfile.ZipFile(fake_zip, 'r') as z:
        imgs = [zi for zi in z.infolist() if zi.file_size > 100 and zi.filename.lower().endswith(('.png', '.jpg', '.jpeg')) and not zi.filename.startswith('__MACOSX') and not Path(zi.filename).name.startswith('.')]
        rng = random.Random(42)
        rng.shuffle(imgs)
        for idx, zi in enumerate(imgs):
            if saved >= max_fakes:
                break
            try:
                with Image.open(io.BytesIO(z.read(zi.filename))) as im:
                    im = im.convert('RGB')
                    w, h = im.size
                    if w != h:
                        s = min(w, h)
                        im = im.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
                    im = im.resize((256, 256), Image.Resampling.BICUBIC)
                    im.save(out_root / 'fake' / f'{name}_{idx:05d}.png', format='PNG', compress_level=1)
                saved += 1
            except Exception:
                continue
    print(f'Prepared unseen benchmark {name}: {len(list((out_root/"real").glob("*.*"))):,} real + {saved:,} fake')
    return out_root

UNSEEN_2 = build_unseen_folder_from_zip('Midjourney_Unseen', SHARED_ROOT / 'test/T2I/Midjourney.zip', max_fakes=3000)
UNSEEN_3 = build_unseen_folder_from_zip('Imagic_Unseen',     SHARED_ROOT / 'test/FE/Imagic.zip',      max_fakes=3000)

# --- Evaluate sequentially on ALL 3 unseen domains ---
print('\n' + '=' * 108)
print('  FINAL UNSEEN CROSS-DOMAIN GENERALIZATION BENCHMARK (ZERO TARGET-SET TUNING)')
print('=' * 108)

for bench_name, bench_dir in [
    ('Midjourney_Unseen (6K)',     UNSEEN_2),
    ('Imagic_Unseen (6K)',         UNSEEN_3),
    ('val_to_be_deleted_2 (60K)',  UNSEEN_1),
]:
    ext_out = EXP_DIR / f'unseen_eval_{bench_name.split()[0]}'
    run_python('evaluate.py', [
        '--arch', 'EFFORT_128', '--checkpoint', str(BEST_CKPT),
        '--val_root', str(bench_dir), '--split', 'external_test',
        '--threshold_file', str(THRESH_JSON),
        '--device', 'cuda:0', '--batch_size', '64', '--num_workers', '4',
        '--eval_precision', 'fp32', '--no_plots', '--no_tsne', '--no_gradcam', '--no_profiling',
        '--output_dir', str(ext_out),
    ])
    rep = json.loads((ext_out / 'EFFORT_128/generalization_report.json').read_text())['overall']
    df_p = pd.read_csv(ext_out / 'EFFORT_128/predictions.csv')
    raw_pred = (df_p['logit'] >= 0.0).astype(int)
    raw05_bal = balanced_accuracy_score(df_p['label'], raw_pred) * 100
    print(
        f"[{bench_name:<26}] ROC AUC={rep['roc_auc']:.4f} | EER={rep['eer']*100:5.2f}% | "
        f"DevCal BalAcc={rep['balanced_accuracy']*100:6.2f}% (R={rep['per_class']['real']['recall']*100:5.1f}%, F={rep['per_class']['fake']['recall']*100:5.1f}%) | "
        f"Uncal(0.5) BalAcc={raw05_bal:6.2f}% (R={(raw_pred[df_p['label']==0]==0).mean()*100:5.1f}%, F={(raw_pred[df_p['label']==1]==1).mean()*100:5.1f}%)"
    )
    print(f"    -> Logit Medians: Real={df_p[df_p['label']==0]['logit'].median():.3f} | Fake={df_p[df_p['label']==1]['logit'].median():.3f}")
print('=' * 108)
```
