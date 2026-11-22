# Colab L4 Playbook: Option A — Clean `WolterWavelet2021_128` (GPU Wavelets) + Two-Stream `MHA_128` Fusion

> **Prerequisite:** Run on the same L4 session where `DatasetMultiDomainClean_v2` and `LOGO_MANIFEST` (`/content/multidomain_v2_prep/multidomain_v2_logo_manifest.csv`) are already prepared (if you restarted Colab, run Cells 1–2 of `COLAB_TRAIN_MULTIDOMAIN_V2_L4.md` first).

### Why This `WolterWavelet2021_128` Run Is Completely Different From Your Old `52%` Run

- Your old Wavelet model trained with **`jpg_prob = 0.0`** on a JPG-vs-PNG shortcut dataset.
- Here we train on **`DatasetMultiDomainClean_v2` (556K images, 154K PNG + 80K JPG reals)** with the **exact same LOGO split** (holding out `SDXL_Refine` + `DiffFace`) and **symmetric forensic augmentations** (`jpg_prob=0.8`, `blur_prob=0.35`, `downscale_prob=0.35`) using fast **GPU-batched Haar Level-3 `signed_log1p` wavelets** (~3–4 min/epoch on L4!).

---

## Cell 1: Install Wavelet GPU Packages (`PyWavelets` + `pytorch-wavelets`)

```python
import sys, subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'setuptools<81', 'PyWavelets', 'pytorch-wavelets'], check=True)
import pywt
from pytorch_wavelets import DWTForward
print('Wavelet GPU stack ready: PyWavelets', pywt.__version__)
```

---

## Cell 2 (Option A): Train `WolterWavelet2021_128` on 556K LOGO Manifest (~15–20 min on L4)

```python
import os, json
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

DATA_ROOT     = Path('/content/dataset/DatasetMultiDomainClean_v2')
LOGO_MANIFEST = Path('/content/multidomain_v2_prep/multidomain_v2_logo_manifest.csv')
DRIVE_OUT     = Path('/content/drive/MyDrive/deepfake_multidomain_v2_runs')
assert LOGO_MANIFEST.is_file(), 'Run Cells 1-2 of COLAB_TRAIN_MULTIDOMAIN_V2_L4.md first to prepare LOGO_MANIFEST.'

RUN_NAME_WAV = 'wolter_wavelet128_v2_logo_forensic'
RUN_ID_WAV   = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
EXP_DIR_WAV  = DRIVE_OUT / f'{RUN_NAME_WAV}_{RUN_ID_WAV}'
BEST_WAV     = EXP_DIR_WAV / 'checkpoints/best.pth'

print('Starting WolterWavelet2021_128 GPU training ->', EXP_DIR_WAV)

run_python('train.py', [
    '--arch', 'WolterWavelet2021_128',
    '--embed_dim', '128',
    '--wavelet_backend', 'gpu',
    '--wavelet_type', 'haar', '--wavelet_level', '3', '--wavelet_mode', 'reflect',
    '--use_log_packets', '--wavelet_log_mode', 'signed_log1p',
    '--image_size', '256', '--crop_size', '224', '--crop_policy', 'scale_and_crop',
    '--aug_recipe', 'custom',
    '--jpg_prob', '0.8', '--jpg_qual', '50,60,70,75,80,85,90,95', '--jpg_method', 'cv2,pil',
    '--blur_prob', '0.35', '--blur_sig', '0.1,2.0',
    '--downscale_prob', '0.35', '--downscale_range', '0.55', '1.0',
    '--dataroot', str(DATA_ROOT),
    '--manifest', str(LOGO_MANIFEST), '--manifest_split', 'train',
    '--val_manifest', str(LOGO_MANIFEST), '--val_manifest_split', 'dev',
    '--allow_source_overlap', '--allow_aggregate_sources',
    '--name', RUN_NAME_WAV, '--run_id', RUN_ID_WAV, '--checkpoints_dir', str(DRIVE_OUT),
    '--gpu_ids', '0', '--batch_size', '128', '--grad_accum_steps', '1',
    '--val_batch_size', '128', '--num_workers', '4', '--pin_memory',
    '--class_bal', '--optim', 'adam', '--lr', '0.0005', '--weight_decay', '0.0001', '--lr_policy', 'cosine',
    '--epochs', '5', '--epochs_decay', '0', '--monitor_metric', 'auc',
    '--use_amp', '--amp_dtype', 'bf16', '--val_precision', 'fp32',
    '--seed', '42', '--log_freq', '100', '--val_epoch_freq', '1', '--save_epoch_freq', '1',
])

assert BEST_WAV.is_file(), f'Missing wavelet checkpoint: {BEST_WAV}'
print('WolterWavelet2021_128 training complete! Best checkpoint:', BEST_WAV)
```

---

## Cell 3: Evaluate `WolterWavelet2021_128` on the Exact Same 4 Photorealistic Unseen Benchmarks (`LFW` vs. `LDM`, `SDXL_Refine`, `DiffFace`, `StyleGAN`)

Tests both **RAW** and **Same-JPEG `q=85`** with **zero threshold tuning (`0.5 / logit = 0.0`)** so you can compare `WolterWavelet2021_128` directly against your saved `EFFORT_128` table!

```python
import io, torch, numpy as np
from pathlib import Path
from PIL import Image
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import roc_auc_score, accuracy_score
from torchvision import transforms
from datasets import load_dataset
from evaluation.checkpoint_loader import CheckpointLoader

wav_model = CheckpointLoader(str(BEST_WAV), device='cuda:0').load()

# Match WolterWavelet2021_128 deterministic eval preprocessing: Resize(256) -> CenterCrop(224) -> [0, 255] float32 -> GPU Wavelets
wav_prep = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
])

def score_wav(imgs, bs=128):
    scores = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            tensors = []
            for im in imgs[i:i+bs]:
                im_c = wav_prep(im.convert('RGB'))
                arr = np.array(im_c, dtype=np.float32).transpose(2, 0, 1)
                tensors.append(torch.from_numpy(arr))
            batch = torch.stack(tensors, dim=0).cuda()
            logits = wav_model(batch).squeeze(1).cpu().tolist()
            scores.extend(logits)
    return np.array(scores)

def to_jpeg(imgs, q=85):
    out = []
    for im in imgs:
        buf = io.BytesIO()
        im.convert('RGB').save(buf, format='JPEG', quality=q)
        buf.seek(0)
        out.append(Image.open(buf).convert('RGB').copy())
    return out

# Load exact same 1,000 unseen images per domain (reuse if already in RAM, else load)
if 'lfw_reals' not in globals():
    lfw = fetch_lfw_people(color=True, resize=1.0, slice_=(slice(0, 250), slice(0, 250)))
    lfw_reals = [Image.fromarray((a * 255).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)) for a in lfw.images[:1000]]

if 'stylegan_fakes' not in globals():
    sg_ds = load_dataset('TheKernel01/140k-Real-and-Fake-Faces', split='test', streaming=True)
    names = getattr(sg_ds.features.get('label'), 'names', None)
    fake_id = next(i for i, n in enumerate(names) if 'fake' in n.lower()) if names else 0
    stylegan_fakes = []
    for s in sg_ds:
        if s['label'] == fake_id:
            stylegan_fakes.append(s['image'].convert('RGB'))
        if len(stylegan_fakes) >= 1000:
            break

v2_fake_imgs  = [Image.open(p).convert('RGB') for p in sorted(Path('/content/unseenDataset/val to be deleted 2/fake').glob('*.*'))[:1000]]
sdxl_ref_imgs = [Image.open(p).convert('RGB') for p in sorted((DATA_ROOT / 'val/fake').glob('SDXL_Refine*'))[:1000]]
diffface_imgs = [Image.open(p).convert('RGB') for p in sorted((DATA_ROOT / 'val/fake').glob('DiffFace*'))[:1000]]

print('\n' + '=' * 112)
print('  WOLTER WAVELET-128 PHOTOREALISTIC UNSEEN BENCHMARK (ZERO TUNING, 1K REAL + 1K FAKE)')
print('=' * 112)
for bench_name, fakes in [
    ('LFW vs Unseen LDM (v2/fake)', v2_fake_imgs),
    ('LFW vs Unseen SDXL_Refine',   sdxl_ref_imgs),
    ('LFW vs Unseen DiffFace',      diffface_imgs),
    ('LFW vs Unseen StyleGAN (GAN)', stylegan_fakes),
]:
    for mode, r_in, f_in in [('RAW     ', lfw_reals, fakes), ('JPEG q85', to_jpeg(lfw_reals, 85), to_jpeg(fakes, 85))]:
        sr = score_wav(r_in)
        sf = score_wav(f_in)
        y  = np.array([0]*len(sr) + [1]*len(sf))
        sc = np.concatenate([sr, sf])
        auc = roc_auc_score(y, sc)
        acc = accuracy_score(y, (sc >= 0.0).astype(int)) * 100
        print(f"  [{mode}] {bench_name:<30} | AUC={auc:.4f} | Acc(@0.5)={acc:6.2f}% | RealRec={(sr<0).mean()*100:5.1f}% | FakeRec={(sf>=0).mean()*100:5.1f}% | Med(R/F)={np.median(sr):+.2f}/{np.median(sf):+.2f}")
print('=' * 112)
```

---

## Cell 4: Pull Two-Stream `EFFORT_128` Checkpoint Fix & Train Matched `MHA_128` vs. `Fusion_128`

> **Important:** First run `git pull` inside the cell below (it pulls our fix ensuring `EFFORT_128 (svd_rank=0)` weights and metadata embed cleanly inside `MHA_128`/`Fusion_128` checkpoints with strict reload parity) and clears any cached Python modules.

This trains **both** your proposed **`MHA_128` (`token_attention`)** and the mandatory **`Fusion_128` (`concat` control)** from your **exact saved frozen `EFFORT_128` + `WolterWavelet2021_128` checkpoints** (~12–15 min per model on L4):

```python
import sys, subprocess
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

# 1. Pull latest code (includes EFFORT_128 embedded checkpoint reload fix) & flush module cache
subprocess.run(['git', 'pull', '--ff-only'], cwd='/content/repo', check=True)
for mod_name in list(sys.modules):
    if mod_name.split('.')[0] in ('data', 'models', 'config', 'training', 'evaluation'):
        del sys.modules[mod_name]

DATA_ROOT     = Path('/content/dataset/DatasetMultiDomainClean_v2')
LOGO_MANIFEST = Path('/content/multidomain_v2_prep/multidomain_v2_logo_manifest.csv')
DRIVE_OUT     = Path('/content/drive/MyDrive/deepfake_multidomain_v2_runs')

BEST_RGB = Path('/content/drive/MyDrive/deepfake_multidomain_v2_runs/effort128_frozen_multilayer_v2_logo_20261121_150525_be2f666c/checkpoints/best.pth')
if 'BEST_WAV' not in globals() or not Path(BEST_WAV).is_file():
    wav_candidates = sorted(DRIVE_OUT.glob('wolter_wavelet128_v2_logo_forensic_*/checkpoints/best.pth'))
    assert wav_candidates, 'Could not locate WolterWavelet2021_128 checkpoint in Drive!'
    BEST_WAV = wav_candidates[-1]

assert BEST_RGB.is_file(), f'Missing RGB checkpoint: {BEST_RGB}'
assert Path(BEST_WAV).is_file(), f'Missing Wavelet checkpoint: {BEST_WAV}'
print('RGB expert    :', BEST_RGB)
print('Wavelet expert:', BEST_WAV)

FUSION_CKPTS = {}

for arch_name, fusion_type, tag in [
    ('MHA_128',    'token_attention', 'mha128_effort_wolter_v2'),
    ('Fusion_128', 'concat',          'fusion128_concat_effort_wolter_v2'),
]:
    rid = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
    exp_dir = DRIVE_OUT / f'{tag}_{rid}'
    print(f'\n=== Training {arch_name} ({fusion_type}) -> {exp_dir} ===')
    run_python('train.py', [
        '--arch', arch_name, '--fusion_type', fusion_type,
        '--embed_dim', '128', '--num_heads', '4', '--dropout', '0.2',
        '--rgb_model_path', str(BEST_RGB),
        '--wavelet_model_path', str(BEST_WAV),
        '--drct_backbone', 'clip_vit_l14',
        '--wavelet_backend', 'gpu', '--wavelet_type', 'haar', '--wavelet_level', '3',
        '--wavelet_mode', 'reflect', '--use_log_packets', '--wavelet_log_mode', 'signed_log1p',
        '--image_size', '256', '--crop_size', '224',
        '--aug_recipe', 'custom',
        '--jpg_prob', '0.8', '--jpg_qual', '50,60,70,75,80,85,90,95', '--jpg_method', 'cv2,pil',
        '--blur_prob', '0.35', '--blur_sig', '0.1,2.0',
        '--downscale_prob', '0.35', '--downscale_range', '0.55', '1.0',
        '--dataroot', str(DATA_ROOT),
        '--manifest', str(LOGO_MANIFEST), '--manifest_split', 'train',
        '--val_manifest', str(LOGO_MANIFEST), '--val_manifest_split', 'dev',
        '--allow_source_overlap', '--allow_aggregate_sources',
        '--name', tag, '--run_id', rid, '--checkpoints_dir', str(DRIVE_OUT),
        '--gpu_ids', '0', '--batch_size', '64', '--val_batch_size', '64',
        '--num_workers', '4', '--pin_memory',
        '--class_bal', '--optim', 'adam', '--lr', '0.0002', '--weight_decay', '0.0001', '--lr_policy', 'cosine',
        '--epochs', '3', '--epochs_decay', '0', '--monitor_metric', 'auc',
        '--use_amp', '--amp_dtype', 'bf16', '--val_precision', 'fp32',
        '--seed', '42', '--log_freq', '100', '--val_epoch_freq', '1', '--save_epoch_freq', '1',
    ])
    best_p = exp_dir / 'checkpoints/best.pth'
    assert best_p.is_file(), f'Missing {arch_name} checkpoint: {best_p}'
    FUSION_CKPTS[arch_name] = best_p
    print(f'Saved {arch_name} best checkpoint -> {best_p}')
```

---

## Cell 5: Complete 5-Way Photorealistic Unseen Benchmark + `MHA_128` Attention Interpretability

Evaluates **all 5 models** side-by-side on the exact same 4 photorealistic unseen benchmarks (`LFW` vs. `LDM`, `SDXL_Refine`, `DiffFace`, `StyleGAN`, RAW + Same-JPEG `q=85`, **zero threshold tuning `logit=0.0`**) and prints per-domain **`[CLS] -> RGB` vs. `[CLS] -> Wavelet` attention weights** from `MHA_128`:

```python
import io, torch, numpy as np
from pathlib import Path
from PIL import Image
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import roc_auc_score, accuracy_score
from torchvision import transforms
from datasets import load_dataset
from data.transforms.clip_forensic_transforms import CLIPForensicValTransform
from evaluation.checkpoint_loader import CheckpointLoader

# Locate checkpoints automatically if restarting
DRIVE_OUT = Path('/content/drive/MyDrive/deepfake_multidomain_v2_runs')
BEST_RGB  = Path('/content/drive/MyDrive/deepfake_multidomain_v2_runs/effort128_frozen_multilayer_v2_logo_20261121_150525_be2f666c/checkpoints/best.pth')
if 'BEST_WAV' not in globals():
    BEST_WAV = sorted(DRIVE_OUT.glob('wolter_wavelet128_v2_logo_forensic_*/checkpoints/best.pth'))[-1]
if 'FUSION_CKPTS' not in globals():
    FUSION_CKPTS = {
        'MHA_128':    sorted(DRIVE_OUT.glob('mha128_effort_wolter_v2_*/checkpoints/best.pth'))[-1],
        'Fusion_128': sorted(DRIVE_OUT.glob('fusion128_concat_effort_wolter_v2_*/checkpoints/best.pth'))[-1],
    }

mha_model    = CheckpointLoader(str(FUSION_CKPTS['MHA_128']), device='cuda:0').load()
fusion_model = CheckpointLoader(str(FUSION_CKPTS['Fusion_128']), device='cuda:0').load()
rgb_model    = CheckpointLoader(str(BEST_RGB), device='cuda:0').load()
wav_model    = CheckpointLoader(str(BEST_WAV), device='cuda:0').load()

rgb_tf = CLIPForensicValTransform(size=224)
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

def score_all(imgs, bs=64):
    s_rgb, s_wav, s_fus, s_mha = [], [], [], []
    attn_cls_rgb, attn_cls_wav = [], []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            chunk = imgs[i:i+bs]
            x_rgb = torch.stack([rgb_tf(im.convert('RGB')) for im in chunk], dim=0).cuda()
            x_wav = torch.stack([
                torch.from_numpy(np.array(wav_tf(im.convert('RGB')), dtype=np.float32).transpose(2, 0, 1))
                for im in chunk
            ], dim=0).cuda()

            s_rgb.extend(rgb_model(x_rgb).squeeze(1).cpu().tolist())
            s_wav.extend(wav_model(x_wav).squeeze(1).cpu().tolist())
            s_fus.extend(fusion_model(x_rgb, x_wav).squeeze(1).cpu().tolist())
            s_mha.extend(mha_model(x_rgb, x_wav).squeeze(1).cpu().tolist())

            # [B, num_heads, 3, 3] -> CLS row (index 0) over [CLS(0), RGB(1), Wavelet(2)]
            w = mha_model.model.last_attention[:, :, 0, :].mean(dim=1).cpu().numpy()  # [B, 3]
            attn_cls_rgb.extend(w[:, 1].tolist())
            attn_cls_wav.extend(w[:, 2].tolist())

    s_rgb, s_wav, s_fus, s_mha = map(np.array, (s_rgb, s_wav, s_fus, s_mha))
    return {
        'EFFORT_128 (RGB)':    s_rgb,
        'WolterWavelet_128':   s_wav,
        'LateEnsemble (0.5)':  0.5 * s_rgb + 0.5 * s_wav,
        'Fusion_128 (Concat)': s_fus,
        'MHA_128 (Attention)': s_mha,
        '_attn_rgb': float(np.mean(attn_cls_rgb)),
        '_attn_wav': float(np.mean(attn_cls_wav)),
    }

# Load exact same 1,000 images per unseen domain
DATA_ROOT = Path('/content/dataset/DatasetMultiDomainClean_v2')
if 'lfw_reals' not in globals():
    lfw = fetch_lfw_people(color=True, resize=1.0, slice_=(slice(0, 250), slice(0, 250)))
    lfw_reals = [Image.fromarray((a * 255).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)) for a in lfw.images[:1000]]
if 'stylegan_fakes' not in globals():
    sg_ds = load_dataset('TheKernel01/140k-Real-and-Fake-Faces', split='test', streaming=True)
    names = getattr(sg_ds.features.get('label'), 'names', None)
    fake_id = next(i for i, n in enumerate(names) if 'fake' in n.lower()) if names else 0
    stylegan_fakes = []
    for s in sg_ds:
        if s['label'] == fake_id:
            stylegan_fakes.append(s['image'].convert('RGB'))
        if len(stylegan_fakes) >= 1000:
            break

v2_fake_imgs  = [Image.open(p).convert('RGB') for p in sorted(Path('/content/unseenDataset/val to be deleted 2/fake').glob('*.*'))[:1000]]
sdxl_ref_imgs = [Image.open(p).convert('RGB') for p in sorted((DATA_ROOT / 'val/fake').glob('SDXL_Refine*'))[:1000]]
diffface_imgs = [Image.open(p).convert('RGB') for p in sorted((DATA_ROOT / 'val/fake').glob('DiffFace*'))[:1000]]

MODELS = ['EFFORT_128 (RGB)', 'WolterWavelet_128', 'LateEnsemble (0.5)', 'Fusion_128 (Concat)', 'MHA_128 (Attention)']

for mode, fn in [('RAW', lambda x: x), ('SAME-JPEG q=85', lambda x: to_jpeg(x, 85))]:
    print('\n' + '=' * 116)
    print(f'  5-WAY PHOTOREALISTIC UNSEEN BENCHMARK [{mode}] — ZERO TUNING (@0.5 / logit=0.0, 1K REAL + 1K FAKE)')
    print('=' * 116)
    res_real = score_all(fn(lfw_reals))
    print(f"  MHA_128 [CLS] Attention on LFW Real: RGB={res_real['_attn_rgb']*100:.1f}% | Wavelet={res_real['_attn_wav']*100:.1f}%")
    for bench_name, fakes in [
        ('Unseen LDM (v2/fake)',   v2_fake_imgs),
        ('Unseen SDXL_Refine',     sdxl_ref_imgs),
        ('Unseen DiffFace (Swap)', diffface_imgs),
        ('Unseen StyleGAN (GAN)',  stylegan_fakes),
    ]:
        res_fake = score_all(fn(fakes))
        print(f"\n  --- LFW Real vs. {bench_name}  |  MHA Attn(Fake): RGB={res_fake['_attn_rgb']*100:.1f}%, Wav={res_fake['_attn_wav']*100:.1f}% ---")
        for m in MODELS:
            sr, sf = res_real[m], res_fake[m]
            y  = np.array([0]*len(sr) + [1]*len(sf))
            sc = np.concatenate([sr, sf])
            auc = roc_auc_score(y, sc)
            acc = accuracy_score(y, (sc >= 0.0).astype(int)) * 100
            print(f"    {m:<22} | AUC={auc:.4f} | Acc(@0.5)={acc:6.2f}% | RealRec={(sr<0).mean()*100:5.1f}% | FakeRec={(sf>=0).mean()*100:5.1f}%")
print('=' * 116)
```
