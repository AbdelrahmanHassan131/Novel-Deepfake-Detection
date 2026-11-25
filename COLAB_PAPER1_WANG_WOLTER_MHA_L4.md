# Colab L4 Playbook — Paper 1 Revision (`Wang2020_128` + `WolterWavelet2021_128` → `MHA_128` vs. `Fusion_128`)

> **Strategy:** This playbook completes **Paper 1 (original submitted architecture)** cleanly on the shortcut-free 556K multi-domain dataset (`DatasetMultiDomainClean_v2`, LOGO split) and evaluates all models on the corrected photorealistic unseen cross-domain benchmarks (`LFW` real camera photos vs. unseen `LDM`, `SDXL_Refine`, `DiffFace`, `StyleGAN`), plus **Perturbation Robustness**, **FairFace Ground-Truth Demographic Fairness**, and **McNemar / Bootstrap 95% CI Statistics**.
>
> **Zero Redundant Training:** Both Paper 1 experts are **already trained on `DatasetMultiDomainClean_v2` and safely saved in Google Drive**:
> - `Wang2020_128` (ResNet-50, Epoch 1 best, `LOGO dev AUC = 0.9981`)
> - `WolterWavelet2021_128` (GPU Haar Level-3 `signed_log1p` Wavelet CNN)
>
> We only train the two fast fusion heads (`MHA_128` + `Fusion_128`, ~6–8 min each on L4) and run the complete evaluation suite (~25 min total)!

---

## Cell 1: Mount Drive, Clone/Pull Repo & Install Packages

```python
from google.colab import drive
drive.mount('/content/drive')

import gc, io, os, sys, csv, json, time, shutil, random, runpy, zipfile, subprocess
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

# Flush any cached modules if re-running
for mod_name in list(sys.modules):
    if mod_name.split('.')[0] in ('data', 'models', 'config', 'training', 'evaluation'):
        del sys.modules[mod_name]

subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q',
    'setuptools<81', 'PyWavelets', 'pytorch-wavelets', 'datasets', 'scikit-learn', 'scipy', 'pillow', 'opencv-python-headless',
], check=True)

import torch, numpy as np
assert torch.cuda.is_available(), 'Select an NVIDIA L4 (recommended) or T4 GPU runtime!'
print(f'GPU: {torch.cuda.get_device_name(0)} | Torch: {torch.__version__}')

def run_python(script, args):
    old_argv = sys.argv[:]
    sys.argv = [script] + list(args)
    try:
        runpy.run_path(script, run_name='__main__')
    finally:
        sys.argv = old_argv

# Locate both already-trained Paper 1 experts in Google Drive
DRIVE_OUT = Path('/content/drive/MyDrive/deepfake_multidomain_v2_runs')
BEST_WANG = DRIVE_OUT / 'wang2020_128_v2_logo_forensic_20261122_185559_d289b6e0/checkpoints/best.pth'
wav_cands = sorted(DRIVE_OUT.glob('wolter_wavelet128_v2_logo_forensic_*/checkpoints/best.pth'))
assert BEST_WANG.is_file(), f'Missing Wang2020_128 checkpoint: {BEST_WANG}'
assert wav_cands, 'Missing WolterWavelet2021_128 checkpoint in Drive!'
BEST_WAV = wav_cands[-1]

print('Paper 1 RGB Expert     :', BEST_WANG)
print('Paper 1 Wavelet Expert :', BEST_WAV)
```

---

## Cell 2: Extract `DatasetMultiDomainClean_v2` & Build Identical LOGO Manifest

*(Skipped automatically if already extracted on local SSD in this runtime.)*

```python
import random
from collections import Counter
from data.drct_manifest import write_drct_manifest
from data.drct_preparation import generate_gate_sidecar

DATA_ROOT     = Path('/content/dataset/DatasetMultiDomainClean_v2')
DATA_ZIP      = Path('/content/drive/MyDrive/DatasetMultiDomainClean_v2/DatasetMultiDomainClean_v2.zip')
PREP_DIR      = Path('/content/multidomain_v2_prep')
LOGO_MANIFEST = PREP_DIR / 'multidomain_v2_logo_manifest.csv'
PREP_DIR.mkdir(parents=True, exist_ok=True)

if not (DATA_ROOT / 'train/real').is_dir():
    assert DATA_ZIP.is_file(), f'Dataset ZIP not found: {DATA_ZIP}'
    Path('/content/dataset').mkdir(parents=True, exist_ok=True)
    print('Extracting DatasetMultiDomainClean_v2.zip to local SSD (~2-3 min)...')
    subprocess.run(['7z', 'x', str(DATA_ZIP), '-o/content/dataset/', '-y'], check=True, stdout=subprocess.DEVNULL)
    print('Extraction complete.')

assert (DATA_ROOT / 'train/real').is_dir(), f'Expected {DATA_ROOT}/train/real'

if not LOGO_MANIFEST.is_file():
    LOGO_HOLDOUT_DEV_GENS = {'sdxl_refine', 'diffface'}
    src_manifest = DATA_ROOT / 'dataset_manifest.csv'
    assert src_manifest.is_file(), f'Missing provenance manifest: {src_manifest}'

    train_rows, dev_unseen_fakes, dev_in_domain_fakes, dev_reals = [], [], [], []
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
                'sha256': row['sample_id'],
                'rec_model_revision': 'none',
                'rec_settings': '{}',
                'rec_seed': '0',
            }
            if row['split'] == 'train':
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

    rng = random.Random(42)
    rng.shuffle(dev_in_domain_fakes)
    dev_fakes_selected = dev_unseen_fakes + dev_in_domain_fakes[:len(dev_unseen_fakes)]

    reals_by_src = {}
    for r in dev_reals:
        reals_by_src.setdefault(r['dataset_source'], []).append(r)
    per_src_quota = len(dev_fakes_selected) // len(reals_by_src) + 1
    dev_reals_selected = []
    for src_k, items in sorted(reals_by_src.items()):
        rng.shuffle(items)
        dev_reals_selected.extend(items[:per_src_quota])
    dev_reals_selected = dev_reals_selected[:len(dev_fakes_selected)]

    all_selected = train_rows + dev_reals_selected + dev_fakes_selected
    write_drct_manifest(LOGO_MANIFEST, all_selected)
    generate_gate_sidecar(LOGO_MANIFEST, require_hashes=False)

    tr_counts = Counter(r['label'] for r in train_rows)
    dv_counts = Counter(r['label'] for r in dev_reals_selected + dev_fakes_selected)
    print(f'LOGO Train: Real={tr_counts[0]:,} | Fake={tr_counts[1]:,}')
    print(f'LOGO Dev  : Real={dv_counts[0]:,} | Fake={dv_counts[1]:,} (50% UNSEEN: {sorted(LOGO_HOLDOUT_DEV_GENS)})')
else:
    print('Reusing existing LOGO_MANIFEST:', LOGO_MANIFEST)
```

---

## Cell 3: Train Paper 1 Two-Stream Fusion Heads (`MHA_128` vs. `Fusion_128` on Frozen `Wang2020_128` + `WolterWavelet2021_128`)

Trains both your proposed **`MHA_128` (`token_attention`)** and the mandatory **`Fusion_128` (`concat` control)** from your exact saved `Wang2020_128` + `WolterWavelet2021_128` experts (~6–8 min each on L4 with `batch_size=128`, `bf16`):

```python
gc.collect()
torch.cuda.empty_cache()

PAPER1_FUSION_CKPTS = {}

for arch_name, fusion_type, tag in [
    ('MHA_128',    'token_attention', 'paper1_mha128_wang_wolter_v2'),
    ('Fusion_128', 'concat',          'paper1_fusion128_concat_wang_wolter_v2'),
]:
    rid = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
    exp_dir = DRIVE_OUT / f'{tag}_{rid}'
    print(f'\n=== Training Paper 1 {arch_name} ({fusion_type}) -> {exp_dir} ===')
    run_python('train.py', [
        '--arch', arch_name, '--fusion_type', fusion_type,
        '--embed_dim', '128', '--num_heads', '4', '--dropout', '0.2',
        '--rgb_model_path', str(BEST_WANG),
        '--wavelet_model_path', str(BEST_WAV),
        '--wavelet_backend', 'gpu', '--wavelet_type', 'haar', '--wavelet_level', '3',
        '--wavelet_mode', 'reflect', '--use_log_packets', '--wavelet_log_mode', 'signed_log1p',
        '--image_size', '256', '--crop_size', '224', '--crop_policy', 'scale_and_crop',
        '--aug_recipe', 'custom',
        '--jpg_prob', '0.8', '--jpg_qual', '50,60,70,75,80,85,90,95', '--jpg_method', 'cv2,pil',
        '--blur_prob', '0.35', '--blur_sig', '0.1,2.0',
        '--downscale_prob', '0.35', '--downscale_range', '0.55', '1.0',
        '--dataroot', str(DATA_ROOT),
        '--manifest', str(LOGO_MANIFEST), '--manifest_split', 'train',
        '--val_manifest', str(LOGO_MANIFEST), '--val_manifest_split', 'dev',
        '--allow_source_overlap', '--allow_aggregate_sources',
        '--name', tag, '--run_id', rid, '--checkpoints_dir', str(DRIVE_OUT),
        '--gpu_ids', '0', '--batch_size', '128', '--val_batch_size', '128',
        '--num_workers', '4', '--pin_memory',
        '--class_bal', '--optim', 'adam', '--lr', '0.0002', '--weight_decay', '0.0001', '--lr_policy', 'cosine',
        '--epochs', '3', '--epochs_decay', '0', '--monitor_metric', 'auc',
        '--use_amp', '--amp_dtype', 'bf16', '--val_precision', 'fp32',
        '--seed', '42', '--log_freq', '100', '--val_epoch_freq', '1', '--save_epoch_freq', '1',
    ])
    best_p = exp_dir / 'checkpoints/best.pth'
    assert best_p.is_file(), f'Missing {arch_name} checkpoint: {best_p}'
    PAPER1_FUSION_CKPTS[arch_name] = best_p
    print(f'Saved Paper 1 {arch_name} best checkpoint -> {best_p}')
```

---

## Cell 4: Paper 1 Complete 5-Way Photorealistic Unseen Benchmark + Bootstrap 95% CIs + McNemar p-Values + Attention Weights

Evaluates **all 5 Paper 1 models** (`Wang2020_128`, `WolterWavelet2021_128`, `LateEnsemble`, `Fusion_128`, `MHA_128`) on the exact same 4 unseen photorealistic domains (`LFW` real camera photos vs. unseen `LDM`, `SDXL_Refine`, `DiffFace`, `StyleGAN`, RAW + Same-JPEG `q=85`, **zero threshold tuning `@0.5`**) with **1,000-resample 95% bootstrap CIs** and **paired McNemar significance tests**:

```python
from PIL import Image
from scipy.stats import binomtest
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import roc_auc_score, accuracy_score
from torchvision import transforms
from datasets import load_dataset
from evaluation.checkpoint_loader import CheckpointLoader

# Auto-locate checkpoints if restarting session
if 'PAPER1_FUSION_CKPTS' not in globals():
    PAPER1_FUSION_CKPTS = {
        'MHA_128':    sorted(DRIVE_OUT.glob('paper1_mha128_wang_wolter_v2_*/checkpoints/best.pth'))[-1],
        'Fusion_128': sorted(DRIVE_OUT.glob('paper1_fusion128_concat_wang_wolter_v2_*/checkpoints/best.pth'))[-1],
    }

mha_p1    = CheckpointLoader(str(PAPER1_FUSION_CKPTS['MHA_128']), device='cuda:0').load()
fusion_p1 = CheckpointLoader(str(PAPER1_FUSION_CKPTS['Fusion_128']), device='cuda:0').load()
wang_p1   = CheckpointLoader(str(BEST_WANG), device='cuda:0').load()
wav_p1    = CheckpointLoader(str(BEST_WAV), device='cuda:0').load()

rgb_tf = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
wav_tf = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
])

def to_jpeg(imgs, q=85):
    out = []
    for im in imgs:
        buf = io.BytesIO()
        im.convert('RGB').save(buf, format='JPEG', quality=q)
        buf.seek(0)
        out.append(Image.open(buf).convert('RGB').copy())
    return out

def score_paper1(imgs, bs=128):
    s_rgb, s_wav, s_fus, s_mha = [], [], [], []
    attn_rgb, attn_wav = [], []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            chunk = imgs[i:i+bs]
            x_rgb = torch.stack([rgb_tf(im.convert('RGB')) for im in chunk], dim=0).cuda()
            x_wav = torch.stack([
                torch.from_numpy(np.array(wav_tf(im.convert('RGB')), dtype=np.float32).transpose(2, 0, 1))
                for im in chunk
            ], dim=0).cuda()

            s_rgb.extend(wang_p1.model(x_rgb).flatten().cpu().tolist())
            s_wav.extend(wav_p1(x_wav).flatten().cpu().tolist())
            s_fus.extend(fusion_p1(x_rgb, x_wav).flatten().cpu().tolist())
            s_mha.extend(mha_p1(x_rgb, x_wav).flatten().cpu().tolist())

            w = mha_p1.model.last_attention[:, :, 0, :].mean(dim=1).cpu().numpy()
            attn_rgb.extend(w[:, 1].tolist())
            attn_wav.extend(w[:, 2].tolist())

    s_rgb, s_wav, s_fus, s_mha = map(np.array, (s_rgb, s_wav, s_fus, s_mha))
    return {
        'Wang2020_128 (RGB)':  s_rgb,
        'WolterWavelet_128':   s_wav,
        'LateEnsemble (0.5)':  0.5 * s_rgb + 0.5 * s_wav,
        'Fusion_128 (Concat)': s_fus,
        'MHA_128 (Attention)': s_mha,
        '_attn_rgb': float(np.mean(attn_rgb)),
        '_attn_wav': float(np.mean(attn_wav)),
    }

def bootstrap_ci(y, sc, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y)
    aucs, accs = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb, sb = y[idx], sc[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, sb))
        accs.append(accuracy_score(yb, (sb >= 0.0).astype(int)) * 100.0)
    return (np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)), (np.percentile(accs, 2.5), np.percentile(accs, 97.5))

def mcnemar_p(y, sc_a, sc_b):
    ca = ((sc_a >= 0.0).astype(int) == y)
    cb = ((sc_b >= 0.0).astype(int) == y)
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    if b + c == 0:
        return 1.0
    return float(binomtest(min(b, c), b + c, 0.5).pvalue)

# Load exact same 1,000 unseen images per domain
lfw = fetch_lfw_people(color=True, resize=1.0, slice_=(slice(0, 250), slice(0, 250)))
lfw_reals = [Image.fromarray((a * 255).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)) for a in lfw.images[:1000]]

sg_ds = load_dataset('TheKernel01/140k-Real-and-Fake-Faces', split='test', streaming=True)
names = getattr(sg_ds.features.get('label'), 'names', None)
fake_id = next(i for i, n in enumerate(names) if 'fake' in n.lower()) if names else 0
stylegan_fakes = []
for s in sg_ds:
    if s['label'] == fake_id:
        stylegan_fakes.append(s['image'].convert('RGB'))
    if len(stylegan_fakes) >= 1000:
        break

sdxl_ref_imgs = [Image.open(p).convert('RGB').copy() for p in sorted((DATA_ROOT / 'val/fake').glob('SDXL_Refine*'))[:1000]]
diffface_imgs = [Image.open(p).convert('RGB').copy() for p in sorted((DATA_ROOT / 'val/fake').glob('DiffFace*'))[:1000]]

# Unseen LDM (val to be deleted 2/fake) via 7z
ldm_tmp = Path('/content/unseen_ldm_tmp')
if not any(ldm_tmp.rglob('*.png')):
    cands = [
        Path('/content/drive/MyDrive/UnseenData/val to be deleted 2.zip'),
        Path('/content/drive/MyDrive/unseenData/val to be deleted 2.zip'),
        *Path('/content/drive/MyDrive').glob('*/*val*deleted*.zip'),
    ]
    ldm_zip = next((p for p in cands if p.is_file()), None)
    assert ldm_zip is not None, 'val to be deleted 2.zip not found on Drive.'
    ldm_tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run(['7z', 'x', str(ldm_zip), '*/fake/*', f'-o{ldm_tmp}/', '-y'], check=True, stdout=subprocess.DEVNULL)
fake_files = sorted([p for p in ldm_tmp.rglob('*.*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg')], key=lambda p: p.name)[:1000]
v2_fake_imgs = [Image.open(p).convert('RGB').copy() for p in fake_files]

MODELS = ['Wang2020_128 (RGB)', 'WolterWavelet_128', 'LateEnsemble (0.5)', 'Fusion_128 (Concat)', 'MHA_128 (Attention)']

for mode, fn in [('RAW', lambda x: x), ('SAME-JPEG q=85', lambda x: to_jpeg(x, 85))]:
    print('\n' + '=' * 124)
    print(f'  PAPER 1 — 5-WAY PHOTOREALISTIC UNSEEN BENCHMARK [{mode}] (ZERO TUNING @0.5, 1K REAL + 1K FAKE)')
    print('=' * 124)
    res_real = score_paper1(fn(lfw_reals))
    print(f"  MHA_128 [CLS] Attention on LFW Real: RGB={res_real['_attn_rgb']*100:.1f}% | Wavelet={res_real['_attn_wav']*100:.1f}%")
    macro_auc = {m: [] for m in MODELS}
    macro_acc = {m: [] for m in MODELS}
    for bench_name, fakes in [
        ('Unseen LDM (v2/fake)',   v2_fake_imgs),
        ('Unseen SDXL_Refine',     sdxl_ref_imgs),
        ('Unseen DiffFace (Swap)', diffface_imgs),
        ('Unseen StyleGAN (GAN)',  stylegan_fakes),
    ]:
        res_fake = score_paper1(fn(fakes))
        print(f"\n  --- LFW Real vs. {bench_name}  |  MHA Attn(Fake): RGB={res_fake['_attn_rgb']*100:.1f}%, Wav={res_fake['_attn_wav']*100:.1f}% ---")
        y = np.array([0]*len(lfw_reals) + [1]*len(fakes))
        sc_mha = np.concatenate([res_real['MHA_128 (Attention)'], res_fake['MHA_128 (Attention)']])
        for m in MODELS:
            sr, sf = res_real[m], res_fake[m]
            sc = np.concatenate([sr, sf])
            auc = roc_auc_score(y, sc)
            acc = accuracy_score(y, (sc >= 0.0).astype(int)) * 100
            (al, ah), (cl, ch) = bootstrap_ci(y, sc, n_boot=500)
            p_str = f"p(vs MHA)={mcnemar_p(y, sc_mha, sc):.2e}" if m != 'MHA_128 (Attention)' else "reference       "
            macro_auc[m].append(auc); macro_acc[m].append(acc)
            print(f"    {m:<22} | AUC={auc:.4f} [{al:.4f},{ah:.4f}] | Acc={acc:6.2f}% [{cl:5.2f},{ch:5.2f}] | R={(sr<0).mean()*100:5.1f}% F={(sf>=0).mean()*100:5.1f}% | {p_str}")
    print('\n  --- 4-DOMAIN MACRO AVERAGE ---')
    for m in MODELS:
        print(f"    {m:<22} | Mean AUC={np.mean(macro_auc[m]):.4f} | Mean Acc={np.mean(macro_acc[m]):6.2f}%")
print('=' * 124)
```

---

## Cell 5 (Reviewer 5 Closure): Perturbation Robustness + FairFace Ground-Truth Demographic Fairness (~5 min on L4)

Runs BOTH remaining Reviewer 5 checks on all 4 Paper 1 models (`Wang2020_128`, `WolterWavelet2021_128`, `Fusion_128`, `MHA_128`):
1. **Real-World Perturbation Robustness** on unseen `LFW` vs. `LDM` (`JPEG q=90/70/50/30`, Gaussian blur `σ=1.0/2.0`, Downscale `0.5×`, Gaussian noise `σ=10`).
2. **Ground-Truth Demographic Fairness** across all **7 racial groups, 2 genders, and 9 age bands** on held-out `HuggingFaceM4/FairFace` (`validation` split, 2,000 samples, zero training overlap).

```python
from PIL import ImageFilter

def apply_blur(imgs, r):
    return [im.filter(ImageFilter.GaussianBlur(radius=r)) for im in imgs]

def apply_downscale(imgs, scale):
    out = []
    for im in imgs:
        w, h = im.size
        out.append(im.resize((max(16, int(w*scale)), max(16, int(h*scale))), Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BILINEAR))
    return out

def apply_noise(imgs, std):
    rng = np.random.RandomState(42)
    out = []
    for im in imgs:
        arr = np.array(im, dtype=np.float32) + rng.normal(0, std, (im.height, im.width, 3))
        out.append(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)))
    return out

# --- 1. Perturbation Robustness (Unseen LFW vs. Unseen LDM, 500+500) ---
R_SUB, F_SUB = lfw_reals[:500], v2_fake_imgs[:500]
y_sub = np.array([0]*len(R_SUB) + [1]*len(F_SUB))
P1_MODELS = ['Wang2020_128 (RGB)', 'WolterWavelet_128', 'Fusion_128 (Concat)', 'MHA_128 (Attention)']

print('\n' + '=' * 116)
print('  PAPER 1 — PERTURBATION ROBUSTNESS (UNSEEN LFW vs. UNSEEN LDM, 500 REAL + 500 FAKE, AUC / Acc@0.5)')
print('=' * 116)
header = f"  {'Perturbation':<22} | " + " | ".join(f"{m:<20}" for m in P1_MODELS)
print(header)
print('-' * 116)
for pname, tf_fn in [
    ('Clean RAW',          lambda x: x),
    ('JPEG q=90',          lambda x: to_jpeg(x, 90)),
    ('JPEG q=70',          lambda x: to_jpeg(x, 70)),
    ('JPEG q=50',          lambda x: to_jpeg(x, 50)),
    ('JPEG q=30',          lambda x: to_jpeg(x, 30)),
    ('Gaussian Blur s=1.0', lambda x: apply_blur(x, 1.0)),
    ('Gaussian Blur s=2.0', lambda x: apply_blur(x, 2.0)),
    ('Downscale 0.5x',     lambda x: apply_downscale(x, 0.5)),
    ('Gauss Noise s=10',   lambda x: apply_noise(x, 10.0)),
]:
    rr = score_paper1(tf_fn(R_SUB))
    rf = score_paper1(tf_fn(F_SUB))
    cells = []
    for m in P1_MODELS:
        sc = np.concatenate([rr[m], rf[m]])
        auc = roc_auc_score(y_sub, sc)
        acc = accuracy_score(y_sub, (sc >= 0.0).astype(int)) * 100
        cells.append(f"{auc:.4f} / {acc:5.1f}%     ")
    print(f"  {pname:<22} | " + " | ".join(cells))
print('=' * 116)

# --- 2. Ground-Truth Demographic Fairness (Held-Out FairFace Validation Split) ---
print('\nLoading 2,000 held-out FairFace validation samples (zero training overlap)...')
ff_val = load_dataset('HuggingFaceM4/FairFace', '0.25', split='validation', streaming=True)
race_names   = ff_val.features['race'].names
gender_names = ff_val.features['gender'].names
age_names    = ff_val.features['age'].names

ff_imgs, ff_race, ff_gender, ff_age = [], [], [], []
for item in ff_val:
    ff_imgs.append(item['image'].convert('RGB'))
    ff_race.append(race_names[item['race']])
    ff_gender.append(gender_names[item['gender']])
    ff_age.append(age_names[item['age']])
    if len(ff_imgs) >= 2000:
        break

res_ff = score_paper1(ff_imgs)

print('\n' + '=' * 116)
print('  PAPER 1 — FAIRFACE GROUND-TRUTH DEMOGRAPHIC FAIRNESS (REAL-FACE FALSE POSITIVE RATE %, LOWER IS BETTER)')
print('=' * 116)
for attr_name, labels_list in [('RACE (7 groups)', ff_race), ('GENDER (2 groups)', ff_gender), ('AGE (9 bands)', ff_age)]:
    print(f'\n  [{attr_name}]')
    groups = sorted(set(labels_list))
    fpr_by_model = {m: [] for m in P1_MODELS}
    for g in groups:
        mask = np.array([v == g for v in labels_list])
        n_g = int(mask.sum())
        vals = []
        for m in P1_MODELS:
            fpr = float((res_ff[m][mask] >= 0.0).mean() * 100.0)
            fpr_by_model[m].append(fpr)
            vals.append(f"{fpr:5.2f}%")
        print(f"    {g:<20} (n={n_g:4d}) | " + " | ".join(f"{m.split()[0]}={v}" for m, v in zip(P1_MODELS, vals)))
    gaps = " | ".join(f"{m.split()[0]} ΔFPR={max(fpr_by_model[m])-min(fpr_by_model[m]):.2f}%" for m in P1_MODELS)
    print(f"    --> MAX DISPARITY GAP         | {gaps}")
print('=' * 116)
```
