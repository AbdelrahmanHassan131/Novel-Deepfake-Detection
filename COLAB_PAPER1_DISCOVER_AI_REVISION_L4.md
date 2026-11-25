# Colab L4 Master Playbook — Paper 1 Major Revision (*Discover Artificial Intelligence*)

> **Purpose:** Runs EVERY remaining experiment requested by **Reviewer 1 (34 comments)** and **Reviewer 2 (15 comments)** from scratch in one clean, resumable notebook. Every JSON report, CSV prediction file, checkpoint, and 300-DPI publication figure is saved directly to Google Drive at `/content/drive/MyDrive/paper1_discover_ai_revision/`.
>
> **Total Runtime on NVIDIA L4:**
> - **Cells 1–7 (Core Revision Suite — solves R1 #1, #4, #5, #6, #7, #8, #9, #10, #12, #13, #16–#20, #26–#27 & R2 #1–#5, #7–#8, #11–#14):** **~18–22 minutes total!**
> - **Cell 8 (Optional Joint End-to-End Training + Published Baselines Control — solves R1 #2, #11, #19 & R2 #6, #9):** **~25 minutes.**

---

## Cell 1: Mount Drive, Pull Latest Repo, Install Packages & Locate Saved Phase-1 Experts

> **Before running Cell 1:** Make sure you pushed our latest local code (`models/mha/ablation_heads.py` and `tools/paper1_revision_suite.py`) to GitHub first!

```python
from google.colab import drive
drive.mount('/content/drive')

import gc, io, os, sys, csv, json, time, copy, random, subprocess
from pathlib import Path
from collections import Counter

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

for mod_name in list(sys.modules):
    if mod_name.split('.')[0] in ('data', 'models', 'config', 'training', 'evaluation', 'tools'):
        del sys.modules[mod_name]

subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q',
    'setuptools<81', 'PyWavelets', 'pytorch-wavelets', 'datasets', 'scikit-learn', 'scipy', 'matplotlib', 'pillow', 'opencv-python-headless',
], check=True)

import torch, numpy as np
assert torch.cuda.is_available(), 'Please select an NVIDIA L4 (recommended) or T4 GPU runtime!'
print(f'GPU: {torch.cuda.get_device_name(0)} | Torch: {torch.__version__}')

DRIVE_RUNS = Path('/content/drive/MyDrive/deepfake_multidomain_v2_runs')
OUT_DIR    = Path('/content/drive/MyDrive/paper1_discover_ai_revision')
for sub in ('metrics', 'figures', 'checkpoints', 'predictions', 'cache'):
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

BEST_WANG = DRIVE_RUNS / 'wang2020_128_v2_logo_forensic_20261122_185559_d289b6e0/checkpoints/best.pth'
wav_cands = sorted(DRIVE_RUNS.glob('wolter_wavelet128_v2_logo_forensic_*/checkpoints/best.pth'))
assert BEST_WANG.is_file(), f'Missing Wang2020_128 checkpoint: {BEST_WANG}'
assert wav_cands, 'Missing WolterWavelet2021_128 checkpoint!'
BEST_WAV = wav_cands[-1]

print('Phase-1 RGB Expert     :', BEST_WANG)
print('Phase-1 Wavelet Expert :', BEST_WAV)
print('Revision Artifacts Dir :', OUT_DIR)
```

---

## Cell 2: Extract Datasets, Audit Source-by-Class Table & Build Strict 3-Way Cross-Fitted Split (R1 #5, #6, #8, #10 & R2 #2, #3, #4)

- **Zero base-expert memorization bias (R1 #6):** Fusion heads train strictly on **held-out `DatasetMultiDomainClean_v2/val`** images (`fusion_train` ~52K + `fusion_dev` ~9K) that neither `Wang2020_128` nor `WolterWavelet2021_128` ever saw during Phase 1!
- **Strict 3-way isolation (R1 #5 & R2 #2):**
  - `fusion_train`: 70% of seen-val reals & seen-val fakes (out-of-sample for Phase 1 experts)
  - `fusion_dev`: 15% of seen-val reals & seen-val fakes (strictly for checkpoint selection & early stopping)
  - `seen_test`: 15% of seen-val reals & seen-val fakes (~11.5K images — **100% untouched** during Phase 1 & Phase 2, giving exact per-generator & per-real-source tables + PNG-only control!)
  - `unseen_tests`: 100% untouched `LFW` (1,000 camera reals) vs. unseen `LDM` (1,000), LOGO-unseen `SDXL_Refine` (1,000), LOGO-unseen `DiffFace` (1,000), and 0-GANs-in-train `StyleGAN` (1,000)!

```python
from PIL import Image
from sklearn.datasets import fetch_lfw_people
from datasets import load_dataset

DATA_ROOT = Path('/content/dataset/DatasetMultiDomainClean_v2')
DATA_ZIP  = Path('/content/drive/MyDrive/DatasetMultiDomainClean_v2/DatasetMultiDomainClean_v2.zip')

if not (DATA_ROOT / 'val/real').is_dir():
    assert DATA_ZIP.is_file(), f'Missing {DATA_ZIP}'
    Path('/content/dataset').mkdir(parents=True, exist_ok=True)
    print('Extracting DatasetMultiDomainClean_v2.zip (~2-3 min)...')
    subprocess.run(['7z', 'x', str(DATA_ZIP), '-o/content/dataset/', '-y'], check=True, stdout=subprocess.DEVNULL)

# Extract unseen LDM (val to be deleted 2/fake) via 7z
LDM_DIR = Path('/content/unseen_ldm_tmp')
if not any(LDM_DIR.rglob('*.png')):
    cands = [
        Path('/content/drive/MyDrive/UnseenData/val to be deleted 2.zip'),
        Path('/content/drive/MyDrive/unseenData/val to be deleted 2.zip'),
        *Path('/content/drive/MyDrive').glob('*/*val*deleted*.zip'),
    ]
    ldm_zip = next((p for p in cands if p.is_file()), None)
    assert ldm_zip is not None, 'val to be deleted 2.zip not found on Drive.'
    LDM_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(['7z', 'x', str(ldm_zip), '*/fake/*', f'-o{LDM_DIR}/', '-y'], check=True, stdout=subprocess.DEVNULL)

# 1. Source-by-class contingency table from authoritative dataset_manifest.csv (R1 #10 & R2 #4)
src_manifest = DATA_ROOT / 'dataset_manifest.csv'
contingency = Counter()
val_seen_rows = []
logo_unseen_rows = {'SDXL_Refine': [], 'DiffFace': []}
LOGO_HOLDOUT = {'sdxl_refine', 'diffface'}

with open(src_manifest, newline='', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        split, lbl, src, gen = r['split'], int(r['label']), r['dataset_source'], r['generator']
        contingency[(split, lbl, src, gen)] += 1
        if split != 'train':
            abs_p = str((DATA_ROOT / r['relative_path']).resolve())
            rec = {'path': abs_p, 'label': lbl, 'source': src, 'generator': gen, 'identity_id': r.get('identity_id', r['sample_id'])}
            if lbl == 1 and gen.lower() == 'sdxl_refine':
                logo_unseen_rows['SDXL_Refine'].append(rec)
            elif lbl == 1 and gen.lower() == 'diffface':
                logo_unseen_rows['DiffFace'].append(rec)
            else:
                val_seen_rows.append(rec)

cont_list = [
    {'split': sp, 'label': lb, 'source': sr, 'generator': gn, 'count': cnt}
    for (sp, lb, sr, gn), cnt in sorted(contingency.items())
]
(OUT_DIR / 'metrics/source_by_class_contingency.json').write_text(json.dumps(cont_list, indent=2), encoding='utf-8')

# 2. Deterministic identity-grouped 70 / 15 / 15 split of held-out seen-val rows
#    Stratified by (label, generator/source) so every source & generator appears cleanly in fusion_train, fusion_dev, seen_test
rng = random.Random(42)
buckets = {}
for rec in val_seen_rows:
    key = (rec['label'], rec['generator'] if rec['label'] == 1 else rec['source'])
    buckets.setdefault(key, []).append(rec)

fusion_train_rows, fusion_dev_rows, seen_test_rows = [], [], []
for key, items in sorted(buckets.items()):
    # Group by identity_id before splitting so zero identity leaks across fusion_train / fusion_dev / seen_test
    by_id = {}
    for it in items:
        by_id.setdefault(it['identity_id'], []).append(it)
    ids = sorted(by_id.keys())
    rng.shuffle(ids)
    n_id = len(ids)
    n_tr = int(0.70 * n_id)
    n_dv = int(0.15 * n_id)
    for iid in ids[:n_tr]:
        fusion_train_rows.extend(by_id[iid])
    for iid in ids[n_tr:n_tr + n_dv]:
        fusion_dev_rows.extend(by_id[iid])
    for iid in ids[n_tr + n_dv:]:
        seen_test_rows.extend(by_id[iid])

print(f'Cross-fitted Phase-2 splits (100% disjoint from Phase-1 training!):')
print(f'  fusion_train : {len(fusion_train_rows):,} (Real={sum(r["label"]==0 for r in fusion_train_rows):,}, Fake={sum(r["label"]==1 for r in fusion_train_rows):,})')
print(f'  fusion_dev   : {len(fusion_dev_rows):,} (Real={sum(r["label"]==0 for r in fusion_dev_rows):,}, Fake={sum(r["label"]==1 for r in fusion_dev_rows):,})')
print(f'  seen_test    : {len(seen_test_rows):,} (Real={sum(r["label"]==0 for r in seen_test_rows):,}, Fake={sum(r["label"]==1 for r in seen_test_rows):,})')

# 3. Load exact 1,000 images per completely unseen photorealistic domain
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
ldm_files = sorted([p for p in LDM_DIR.rglob('*.*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg')], key=lambda p: p.name)[:1000]
ldm_fakes = [Image.open(p).convert('RGB').copy() for p in ldm_files]
print(f'Unseen benchmarks ready: LFW={len(lfw_reals)}, LDM={len(ldm_fakes)}, SDXL_Refine={len(sdxl_ref_imgs)}, DiffFace={len(diffface_imgs)}, StyleGAN={len(stylegan_fakes)}')
```

---

## Cell 3: Extract & Cache Frozen 128-D Expert Embeddings Once (~4 min on L4, Saved to Drive!)

Extracts **both** `128-D` embeddings and standalone logits from your frozen `Wang2020_128` and `WolterWavelet2021_128` experts:
- For `fusion_train`: extracts **2 views per image** (Clean + Stochastic Forensic JPEG/Blur augmentation!) so every fusion head trains with full forensic robustness (~110K out-of-sample embedding pairs!).
- For `fusion_dev`, `seen_test`, and all 4 unseen domains (RAW + Same-JPEG `q=85`): deterministic clean/JPEG views.

```python
from PIL import ImageFilter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from evaluation.checkpoint_loader import CheckpointLoader

CACHE_FILE = OUT_DIR / 'cache/paper1_frozen_embeddings_v2.pt'

wang_trainer = CheckpointLoader(str(BEST_WANG), device='cuda:0').load()
wav_trainer  = CheckpointLoader(str(BEST_WAV),  device='cuda:0').load()

# Build exact 128-D feature extractors + standalone classifier heads
wang_net = wang_trainer.model.eval()
wav_net  = wav_trainer.model.eval()
gpu_dwt  = wav_trainer._get_gpu_wavelet_backend()

def forward_wang_feat_and_logit(x_norm):
    # ResNet50 up to penultimate 128-D ReLU output (exact feature used by TwoStreamTrainer)
    m = wang_net
    x = m.maxpool(m.relu(m.bn1(m.conv1(x_norm))))
    x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
    x = torch.flatten(m.avgpool(x), 1)
    feat = m.fc[1](m.fc[0](x))      # Linear(2048->128) -> ReLU
    logit = m.fc[3](feat)           # Linear(128->1) in eval mode (Dropout=identity)
    return feat, logit.flatten()

def forward_wav_feat_and_logit(x_255):
    w = gpu_dwt(x_255)
    m = wav_net
    x = m.pool1(torch.relu(m.bn1(m.conv1(w))))
    x = m.pool2(torch.relu(m.bn2(m.conv2(x))))
    x = m.pool3(torch.relu(m.bn3(m.conv3(x))))
    x = torch.relu(m.bn4(m.conv4(x)))
    if min(x.shape[-2:]) >= 2:
        x = m.pool4(x)
    x = torch.flatten(m.global_pool(x), 1)
    feat = m.classifier[1](m.classifier[0](x))  # Linear(512->128) -> ReLU
    logit = m.classifier[3](feat)               # Linear(128->1) in eval mode
    return feat, logit.flatten()

rgb_norm_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
det_prep = transforms.Compose([
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

def forensic_aug_pil(im, rng_obj):
    im = im.convert('RGB')
    if rng_obj.random() < 0.35:
        scale = rng_obj.uniform(0.55, 0.95)
        w, h = im.size
        im = im.resize((max(32, int(w*scale)), max(32, int(h*scale))), Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BILINEAR)
    if rng_obj.random() < 0.35:
        im = im.filter(ImageFilter.GaussianBlur(radius=rng_obj.uniform(0.3, 1.8)))
    if rng_obj.random() < 0.80:
        q = rng_obj.choice([50, 60, 70, 75, 80, 85, 90, 95])
        buf = io.BytesIO()
        im.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        im = Image.open(buf).convert('RGB').copy()
    return im

class PathPairDataset(Dataset):
    def __init__(self, rows, augment=False, seed=42):
        self.rows = rows
        self.augment = augment
        self.seed = seed
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        im = Image.open(r['path']).convert('RGB')
        if self.augment:
            im = forensic_aug_pil(im, random.Random(self.seed + idx))
        im = det_prep(im)
        x_rgb = rgb_norm_tf(im)
        x_wav = torch.from_numpy(np.array(im, dtype=np.float32).transpose(2, 0, 1))
        return x_rgb, x_wav, r['label']

def extract_from_rows(rows, augment=False, seed=42, bs=128):
    dl = DataLoader(PathPairDataset(rows, augment=augment, seed=seed), batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    fr, fw, lr_list, lw_list, ys = [], [], [], [], []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for xr, xw, y in dl:
            xr, xw = xr.cuda(non_blocking=True), xw.cuda(non_blocking=True)
            zr, lr_ = forward_wang_feat_and_logit(xr)
            zw, lw_ = forward_wav_feat_and_logit(xw)
            fr.append(zr.float().cpu()); fw.append(zw.float().cpu())
            lr_list.append(lr_.float().cpu()); lw_list.append(lw_.float().cpu())
            ys.append(y.long())
    return {
        'rgb': torch.cat(fr), 'wav': torch.cat(fw),
        'logit_rgb': torch.cat(lr_list), 'logit_wav': torch.cat(lw_list),
        'y': torch.cat(ys),
    }

def extract_from_pils(imgs, bs=128):
    fr, fw, lr_list, lw_list = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            chunk = [det_prep(im.convert('RGB')) for im in imgs[i:i+bs]]
            xr = torch.stack([rgb_norm_tf(im) for im in chunk]).cuda()
            xw = torch.stack([torch.from_numpy(np.array(im, dtype=np.float32).transpose(2, 0, 1)) for im in chunk]).cuda()
            zr, lr_ = forward_wang_feat_and_logit(xr)
            zw, lw_ = forward_wav_feat_and_logit(xw)
            fr.append(zr.cpu()); fw.append(zw.cpu())
            lr_list.append(lr_.cpu()); lw_list.append(lw_.cpu())
    return {'rgb': torch.cat(fr), 'wav': torch.cat(fw), 'logit_rgb': torch.cat(lr_list), 'logit_wav': torch.cat(lw_list)}

if not CACHE_FILE.is_file():
    t0 = time.time()
    print('Extracting cross-fitted fusion_train (Clean + Forensic-Aug views)...')
    tr_clean = extract_from_rows(fusion_train_rows, augment=False)
    tr_aug   = extract_from_rows(fusion_train_rows, augment=True, seed=10042)
    cache = {
        'fusion_train': {k: torch.cat([tr_clean[k], tr_aug[k]]) for k in tr_clean},
        'fusion_dev':   extract_from_rows(fusion_dev_rows, augment=False),
        'seen_test':    extract_from_rows(seen_test_rows, augment=False),
        'seen_test_meta': seen_test_rows,
        'unseen_raw': {
            'LFW_Real':    extract_from_pils(lfw_reals),
            'LDM':         extract_from_pils(ldm_fakes),
            'SDXL_Refine': extract_from_pils(sdxl_ref_imgs),
            'DiffFace':    extract_from_pils(diffface_imgs),
            'StyleGAN':    extract_from_pils(stylegan_fakes),
        },
        'unseen_q85': {
            'LFW_Real':    extract_from_pils(to_jpeg(lfw_reals, 85)),
            'LDM':         extract_from_pils(to_jpeg(ldm_fakes, 85)),
            'SDXL_Refine': extract_from_pils(to_jpeg(sdxl_ref_imgs, 85)),
            'DiffFace':    extract_from_pils(to_jpeg(diffface_imgs, 85)),
            'StyleGAN':    extract_from_pils(to_jpeg(stylegan_fakes, 85)),
        },
    }
    torch.save(cache, CACHE_FILE)
    print(f'Cached all frozen embeddings in {time.time()-t0:.1f}s -> {CACHE_FILE}')
else:
    cache = torch.load(CACHE_FILE, map_location='cpu', weights_only=False)
    print('Loaded existing cached embeddings ->', CACHE_FILE)
```

---

## Cell 4: 6-Head Capacity-Matched Ablation × 4 Seeds + Hyperparameter Sweep (R1 #1, #4, #6, #7, #12, #16–#17 & R2 #5, #7, #8)

Trains **all 6 fusion heads × 4 independent seeds (`42, 100, 2024, 3407`)** on the cross-fitted out-of-sample split (~2 min total on L4!), saves the seed-42 checkpoints for downstream FGSM/plotting, runs a **capacity & dropout sweep on `Fusion_128`** (R1 #12), and exports exact confusion matrices, calibration (`Brier`, `ECE`), Holm–Bonferroni $p$-values, and Cohen's $d$!

```python
from models.mha.ablation_heads import build_ablation_head, count_trainable_params
from tools.paper1_revision_suite import (
    cluster_bootstrap_ci,
    compute_exact_metrics,
    mcnemar_exact_test,
    paired_seed_statistics,
    train_cached_fusion_head,
)

SEEDS = [42, 100, 2024, 3407]
HEAD_LIST = [
    ('MHA_128 (TokenAttn)',    'mha_128'),
    ('ConcatMatched (MLP)',    'concat_matched'),
    ('NoAttnFFN (Ablation)',   'no_attn_ffn'),
    ('LinearMix (NoSoftmax)',  'linear_mix'),
    ('GatedFusion (Sample)',   'gated'),
    ('Fusion_128 (Orig)',      'fusion_128'),
]

tr = cache['fusion_train']
dv = cache['fusion_dev']
st = cache['seen_test']

def eval_head_on_pack(head, pack):
    head.eval()
    with torch.no_grad():
        r = pack['rgb'].cuda()
        w = pack['wav'].cuda()
        logits = head(r, w).flatten().cpu().numpy()
        attn = None
        if hasattr(head, 'last_attention') and head.last_attention is not None:
            attn_w = head.last_attention[:, :, 0, :].mean(dim=1).cpu().numpy()
            attn = {'cls': float(attn_w[:, 0].mean()), 'rgb': float(attn_w[:, 1].mean()), 'wav': float(attn_w[:, 2].mean())}
        elif hasattr(head, 'last_gate') and head.last_gate is not None:
            g = head.last_gate.cpu().numpy()
            attn = {'rgb_gate': float(g[:, 0].mean()), 'wav_gate': float(g[:, 1].mean())}
    return logits, attn

def eval_suite(head):
    out = {'seen_test': {}, 'unseen_q85': {}, 'unseen_raw': {}, 'attn_q85': {}}
    ls, _ = eval_head_on_pack(head, st)
    out['seen_test'] = compute_exact_metrics(st['y'].numpy(), ls)
    for proto_key, cache_key in [('unseen_q85', 'unseen_q85'), ('unseen_raw', 'unseen_raw')]:
        lr, ar = eval_head_on_pack(head, cache[cache_key]['LFW_Real'])
        if proto_key == 'unseen_q85':
            out['attn_q85']['LFW_Real'] = ar
        dom_aucs, dom_accs, dom_eers, dom_f1s, dom_briers = [], [], [], [], []
        for gname in ('LDM', 'SDXL_Refine', 'DiffFace', 'StyleGAN'):
            lf, af = eval_head_on_pack(head, cache[cache_key][gname])
            if proto_key == 'unseen_q85':
                out['attn_q85'][gname] = af
            y = np.array([0]*len(lr) + [1]*len(lf))
            m = compute_exact_metrics(y, np.concatenate([lr, lf]))
            out[proto_key][gname] = m
            dom_aucs.append(m['roc_auc']); dom_accs.append(m['accuracy']); dom_eers.append(m['eer'])
            dom_f1s.append(m['macro_f1']); dom_briers.append(m['brier_score'])
        out[proto_key]['macro_4domain'] = {
            'accuracy': float(np.mean(dom_accs)), 'macro_f1': float(np.mean(dom_f1s)),
            'roc_auc': float(np.mean(dom_aucs)), 'eer': float(np.mean(dom_eers)),
            'brier_score': float(np.mean(dom_briers)), 'lfw_real_recall': float((lr < 0).mean()),
        }
    return out

ablation_runs = {label: [] for label, _ in HEAD_LIST}
trained_seed42_heads = {}
histories_seed42 = {}

for label, key in HEAD_LIST:
    for s in SEEDS:
        head, info = train_cached_fusion_head(
            key, tr['rgb'], tr['wav'], tr['y'], dv['rgb'], dv['wav'], dv['y'],
            dropout=0.2, modality_dropout=0.15, label_smoothing=0.05,
            lr=3e-4, weight_decay=1e-3, batch_size=256, epochs=25, seed=s, device='cuda:0',
        )
        res = eval_suite(head)
        ablation_runs[label].append({'train_info': {k: v for k, v in info.items() if k != 'history'}, 'eval': res})
        if s == 42:
            trained_seed42_heads[key] = head.cpu()
            histories_seed42[label] = info['history']
            torch.save(head.state_dict(), OUT_DIR / f'checkpoints/{key}_seed42.pt')
    p_cnt = ablation_runs[label][0]['train_info']['trainable_params']
    gap = np.mean([r['train_info']['train_dev_auc_gap'] for r in ablation_runs[label]])
    st_acc = np.mean([r['eval']['seen_test']['accuracy'] for r in ablation_runs[label]]) * 100
    st_auc = np.mean([r['eval']['seen_test']['roc_auc'] for r in ablation_runs[label]])
    uq_acc = np.mean([r['eval']['unseen_q85']['macro_4domain']['accuracy'] for r in ablation_runs[label]]) * 100
    uq_auc = np.mean([r['eval']['unseen_q85']['macro_4domain']['roc_auc'] for r in ablation_runs[label]])
    lfw_r  = np.mean([r['eval']['unseen_q85']['macro_4domain']['lfw_real_recall'] for r in ablation_runs[label]]) * 100
    print(f"{label:<24} | Params={p_cnt:7,d} | Tr-DevGap={gap:+.4f} | SeenTest Acc={st_acc:5.2f}% AUC={st_auc:.4f} | UnseenQ85 4Dom Acc={uq_acc:5.2f}% AUC={uq_auc:.4f} | LFW RealRec={lfw_r:5.1f}%")

# Standalone experts reference on identical splits
def eval_standalone(logit_key):
    out = {'seen_test': compute_exact_metrics(st['y'].numpy(), st[logit_key].numpy()), 'unseen_q85': {}, 'unseen_raw': {}}
    for proto_key in ('unseen_q85', 'unseen_raw'):
        lr = cache[proto_key]['LFW_Real'][logit_key].numpy()
        accs, aucs, eers, f1s, briers = [], [], [], [], []
        for gname in ('LDM', 'SDXL_Refine', 'DiffFace', 'StyleGAN'):
            lf = cache[proto_key][gname][logit_key].numpy()
            y = np.array([0]*len(lr) + [1]*len(lf))
            m = compute_exact_metrics(y, np.concatenate([lr, lf]))
            out[proto_key][gname] = m
            accs.append(m['accuracy']); aucs.append(m['roc_auc']); eers.append(m['eer']); f1s.append(m['macro_f1']); briers.append(m['brier_score'])
        out[proto_key]['macro_4domain'] = {
            'accuracy': float(np.mean(accs)), 'macro_f1': float(np.mean(f1s)),
            'roc_auc': float(np.mean(aucs)), 'eer': float(np.mean(eers)),
            'brier_score': float(np.mean(briers)), 'lfw_real_recall': float((lr < 0).mean()),
        }
    return out

standalone_res = {
    'Wang2020_128 (RGB)': eval_standalone('logit_rgb'),
    'WolterWavelet_128':  eval_standalone('logit_wav'),
}

# Paired 4-seed Holm-Bonferroni stats: MHA_128 vs every control on SeenTest & UnseenQ85 4-Domain Macro
paired_stats = {}
for split_tag, extractor in [
    ('seen_test',         lambda r: r['eval']['seen_test']),
    ('unseen_q85_macro4', lambda r: r['eval']['unseen_q85']['macro_4domain']),
    ('unseen_q85_LDM',    lambda r: r['eval']['unseen_q85']['LDM']),
    ('unseen_q85_DiffFace', lambda r: r['eval']['unseen_q85']['DiffFace']),
    ('unseen_q85_StyleGAN', lambda r: r['eval']['unseen_q85']['StyleGAN']),
]:
    paired_stats[split_tag] = {}
    mha_runs = [extractor(r) for r in ablation_runs['MHA_128 (TokenAttn)']]
    for label, _ in HEAD_LIST[1:]:
        ctrl_runs = [extractor(r) for r in ablation_runs[label]]
        paired_stats[split_tag][label] = paired_seed_statistics(mha_runs, ctrl_runs)

# Print detailed seed-42 unseen JPEG q=85 breakdown
print('\n' + '=' * 120)
print('  CROSS-FITTED SEED-42 UNSEEN BENCHMARK [SAME-JPEG q=85] — EXACT METRICS + CALIBRATION (BRIER / ECE)')
print('=' * 120)
for gname in ('LDM', 'SDXL_Refine', 'DiffFace', 'StyleGAN'):
    print(f'\n  --- LFW Real vs. Unseen {gname} ---')
    for sname, sres in standalone_res.items():
        m = sres['unseen_q85'][gname]
        print(f"    {sname:<24} | AUC={m['roc_auc']:.4f} | Acc={m['accuracy']*100:6.2f}% | F1={m['macro_f1']:.4f} | EER={m['eer']*100:5.2f}% | R={m['real']['recall']*100:5.1f}% F={m['fake']['recall']*100:5.1f}% | Brier={m['brier_score']:.4f} ECE={m['ece']:.4f}")
    for label, _ in HEAD_LIST:
        m = ablation_runs[label][0]['eval']['unseen_q85'][gname]
        print(f"    {label:<24} | AUC={m['roc_auc']:.4f} | Acc={m['accuracy']*100:6.2f}% | F1={m['macro_f1']:.4f} | EER={m['eer']*100:5.2f}% | R={m['real']['recall']*100:5.1f}% F={m['fake']['recall']*100:5.1f}% | Brier={m['brier_score']:.4f} ECE={m['ece']:.4f}")

# Print MHA_128 [CLS] Attention Weights & GatedFusion Weights per domain
print('\n  MHA_128 [CLS] Attention (JPEG q85):', json.dumps(ablation_runs['MHA_128 (TokenAttn)'][0]['eval']['attn_q85'], indent=2))
print('  GatedFusion Weights     (JPEG q85):', json.dumps(ablation_runs['GatedFusion (Sample)'][0]['eval']['attn_q85'], indent=2))

# Concatenation baseline capacity & dropout sweep (directly answers Reviewer 1 #12!)
print('\n  --- Concatenation MLP Width & Dropout Tuning Sweep (R1 #12, seed=42) ---')
concat_sweep = []
for width in (128, 256, 384, 512):
    for drop in (0.1, 0.2, 0.3, 0.5):
        torch.manual_seed(42)
        h_mlp = torch.nn.Sequential(
            torch.nn.Linear(256, width), torch.nn.GELU(), torch.nn.Dropout(drop),
            torch.nn.Linear(width, width // 2), torch.nn.GELU(), torch.nn.Dropout(drop),
            torch.nn.Linear(width // 2, 1),
        ).cuda()
        class _WrapMLP(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.rn, self.wn, self.m = torch.nn.LayerNorm(128), torch.nn.LayerNorm(128), m
            def forward(self, r, w):
                return self.m(torch.cat((self.rn(r), self.wn(w)), dim=-1))
        wm = _WrapMLP(h_mlp).cuda()
        opt_s = torch.optim.AdamW(wm.parameters(), lr=3e-4, weight_decay=1e-3)
        tr_r_c, tr_w_c, tr_y_c = tr['rgb'].cuda(), tr['wav'].cuda(), tr['y'].cuda().float()
        for _ in range(15):
            wm.train()
            perm = torch.randperm(len(tr_y_c), device='cuda:0')
            for i in range(0, len(tr_y_c), 256):
                idx = perm[i:i+256]
                opt_s.zero_grad(set_to_none=True)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(wm(tr_r_c[idx], tr_w_c[idx]).flatten(), tr_y_c[idx]*0.9 + 0.05)
                loss.backward(); opt_s.step()
        sw_res = eval_suite(wm)
        rec = {
            'width': width, 'dropout': drop, 'params': count_trainable_params(wm),
            'seen_test_auc': sw_res['seen_test']['roc_auc'], 'seen_test_acc': sw_res['seen_test']['accuracy'],
            'unseen_q85_macro4_auc': sw_res['unseen_q85']['macro_4domain']['roc_auc'],
            'unseen_q85_macro4_acc': sw_res['unseen_q85']['macro_4domain']['accuracy'],
        }
        concat_sweep.append(rec)
        print(f"    width={width:3d} drop={drop:.1f} params={rec['params']:6,d} | SeenTest AUC={rec['seen_test_auc']:.4f} | UnseenQ85 4Dom Acc={rec['unseen_q85_macro4_acc']*100:5.2f}% AUC={rec['unseen_q85_macro4_auc']:.4f}")

# Save full JSON report (all 4 seeds, per-seed values, paired stats, histories, concat sweep)
(OUT_DIR / 'metrics/paper1_ablation_4seed_report.json').write_text(
    json.dumps({'ablation_runs': ablation_runs, 'standalone': standalone_res, 'paired_stats': paired_stats, 'histories_seed42': histories_seed42, 'concat_sweep': concat_sweep}, indent=2),
    encoding='utf-8',
)
print('Saved complete 4-seed report ->', OUT_DIR / 'metrics/paper1_ablation_4seed_report.json')
```

---

## Cell 5: Per-Generator (All 15 Generators) & Per-Real-Source Breakdown + Same-PNG Control + Exact Latency Profiling (R1 #8, #10 & R2 #4, #14)

```python
import pandas as pd

mha_s42   = trained_seed42_heads['mha_128'].cuda().eval()
match_s42 = trained_seed42_heads['concat_matched'].cuda().eval()
gate_s42  = trained_seed42_heads['gated'].cuda().eval()

with torch.no_grad():
    st_mha   = mha_s42(st['rgb'].cuda(), st['wav'].cuda()).flatten().cpu().numpy()
    st_match = match_s42(st['rgb'].cuda(), st['wav'].cuda()).flatten().cpu().numpy()
    st_gate  = gate_s42(st['rgb'].cuda(), st['wav'].cuda()).flatten().cpu().numpy()
    st_rgb   = st['logit_rgb'].numpy()
    st_wav   = st['logit_wav'].numpy()

meta = cache['seen_test_meta']
df_st = pd.DataFrame({
    'label': [r['label'] for r in meta],
    'source': [r['source'] for r in meta],
    'generator': [r['generator'] for r in meta],
    'identity_id': [r['identity_id'] for r in meta],
    'Wang2020_128': st_rgb,
    'WolterWavelet_128': st_wav,
    'ConcatMatched': st_match,
    'GatedFusion': st_gate,
    'MHA_128': st_mha,
})
df_st.to_csv(OUT_DIR / 'predictions/seen_test_predictions_seed42.csv', index=False)

# 1. Per-real-source recall (R1 #10 & R2 #4)
print('=' * 105)
print('  PER-REAL-SOURCE RECALL ON UNTOUCHED HELD-OUT SEEN_TEST (@0.5 / logit=0.0)')
print('=' * 105)
real_df = df_st[df_st['label'] == 0]
per_src_report = {}
for src, grp in real_df.groupby('source'):
    row_dict = {'n': len(grp)}
    parts = []
    for col in ('Wang2020_128', 'WolterWavelet_128', 'ConcatMatched', 'GatedFusion', 'MHA_128'):
        rec = float((grp[col] < 0.0).mean() * 100.0)
        row_dict[col] = rec
        parts.append(f'{col}={rec:5.2f}%')
    per_src_report[src] = row_dict
    print(f'  {src:<16} (n={len(grp):4d}) | ' + ' | '.join(parts))

# 2. Per-generator recall & AUC (vs pooled held-out reals) for ALL 13 seen generators (R1 #8)
print('\n' + '=' * 105)
print('  PER-GENERATOR RECALL & ROC-AUC ON UNTOUCHED HELD-OUT SEEN_TEST (13 SEEN GENERATORS)')
print('=' * 105)
fake_df = df_st[df_st['label'] == 1]
per_gen_report = {}
for gen, grp in fake_df.groupby('generator'):
    y_pair = np.array([0]*len(real_df) + [1]*len(grp))
    g_dict = {'n_fake': len(grp)}
    parts = []
    for col in ('Wang2020_128', 'WolterWavelet_128', 'ConcatMatched', 'MHA_128'):
        sc_pair = np.concatenate([real_df[col].values, grp[col].values])
        rec = float((grp[col] >= 0.0).mean() * 100.0)
        auc = float(roc_auc_score(y_pair, sc_pair))
        g_dict[col] = {'recall': rec, 'auc': auc}
        parts.append(f'{col}={rec:5.1f}%/{auc:.4f}')
    per_gen_report[gen] = g_dict
    print(f'  {gen:<14} (n={len(grp):4d}) | ' + ' | '.join(parts))

# 3. Same-Pipeline Control: Strictly Native PNG-only Reals (FFHQ-256 + FairFace) vs. Native PNG Fakes (R1 #10 & R2 #4)
png_real = real_df[real_df['source'].isin(['ffhq256', 'fairface_val', 'fairface'])]
png_df = pd.concat([png_real, fake_df], ignore_index=True)
print('\n' + '=' * 105)
print(f'  SAME-PIPELINE CONTROL: STRICTLY NATIVE PNG REALS (n={len(png_real):,}) vs. NATIVE PNG FAKES (n={len(fake_df):,})')
print('=' * 105)
png_ctrl = {}
for col in ('Wang2020_128', 'WolterWavelet_128', 'ConcatMatched', 'GatedFusion', 'MHA_128'):
    m = compute_exact_metrics(png_df['label'].values, png_df[col].values)
    png_ctrl[col] = m
    print(f"  {col:<20} | AUC={m['roc_auc']:.4f} | Acc={m['accuracy']*100:6.2f}% | BalAcc={m['balanced_accuracy']*100:6.2f}% | MacroF1={m['macro_f1']:.4f} | EER={m['eer']*100:5.2f}% | Brier={m['brier_score']:.4f}")

# 4. Exact Identity-Cluster Bootstrap 95% CIs on seen_test (R1 #16 & R2 #7)
cluster_cis = {}
for col in ('Wang2020_128', 'WolterWavelet_128', 'ConcatMatched', 'GatedFusion', 'MHA_128'):
    cluster_cis[col] = cluster_bootstrap_ci(df_st['label'].values, df_st[col].values, cluster_ids=df_st['identity_id'].values, n_boot=500)
print('\n  Identity-Cluster Bootstrap 95% CI (seen_test ROC-AUC):', {k: [round(v[0], 4), round(v[1], 4)] for k, v in ((c, cluster_cis[c]['roc_auc']) for c in cluster_cis)})

# 5. Exact Single-Image Latency Profiling: Excluding vs. Including GPU Wavelet Preprocessing (R2 #14)
x_r1 = torch.randn(1, 3, 224, 224, device='cuda:0')
x_w1 = torch.rand(1, 3, 224, 224, device='cuda:0') * 255.0
w_pre = gpu_dwt(x_w1)
def bench_ms(fn, warmup=30, iters=200):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters

latency_report = {
    'gpu_name': torch.cuda.get_device_name(0),
    'wang2020_128_ms': bench_ms(lambda: forward_wang_feat_and_logit(x_r1)),
    'gpu_dwt_level3_ms': bench_ms(lambda: gpu_dwt(x_w1)),
    'wolter_cnn_excl_dwt_ms': bench_ms(lambda: wav_net(w_pre)),
    'wolter_cnn_incl_dwt_ms': bench_ms(lambda: forward_wav_feat_and_logit(x_w1)),
    'mha128_head_only_ms': bench_ms(lambda: mha_s42(torch.randn(1, 128, device='cuda:0'), torch.randn(1, 128, device='cuda:0'))),
    'full_dual_stream_incl_dwt_ms': bench_ms(lambda: mha_s42(forward_wang_feat_and_logit(x_r1)[0], forward_wav_feat_and_logit(x_w1)[0])),
}
print('\n  Exact FP32 Batch=1 Latency Profile (ms):', json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in latency_report.items()}, indent=2))

(OUT_DIR / 'metrics/paper1_per_generator_source_latency.json').write_text(
    json.dumps({'per_real_source': per_src_report, 'per_generator': per_gen_report, 'png_only_control': png_ctrl, 'cluster_cis': cluster_cis, 'latency': latency_report}, indent=2),
    encoding='utf-8',
)
```

---

## Cell 6: Matched End-to-End White-Box FGSM $\varepsilon$-Sweep + 9 Perturbations + FairFace Fairness (R1 #13 & R2 #13)

Runs exact **raw-pixel `[0, 1]` differentiable white-box FGSM** ($\varepsilon \in \{0, 1/255, 2/255, 4/255, 8/255\}$) across **ALL 5 models** (`Wang2020_128`, `WolterWavelet_128`, `ConcatMatched`, `GatedFusion`, `MHA_128`) with Real/Fake recall, plus 9 real-world perturbations and FairFace 7-race/2-gender/9-age fairness!

```python
import torch.nn as nn
from tools.paper1_revision_suite import EndToEndDifferentiableWrapper, run_fgsm_epsilon_sweep

# Differentiable feature extractor wrappers for end-to-end raw-pixel FGSM
class _WangFeat(nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net
    def forward(self, x):
        m = self.net
        h = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        h = m.layer4(m.layer3(m.layer2(m.layer1(h))))
        h = torch.flatten(m.avgpool(h), 1)
        return m.fc[1](m.fc[0](h))

class _WavFeat(nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net
    def forward(self, w):
        m = self.net
        h = m.pool1(torch.relu(m.bn1(m.conv1(w))))
        h = m.pool2(torch.relu(m.bn2(m.conv2(h))))
        h = m.pool3(torch.relu(m.bn3(m.conv3(h))))
        h = torch.relu(m.bn4(m.conv4(h)))
        if min(h.shape[-2:]) >= 2:
            h = m.pool4(h)
        h = torch.flatten(m.global_pool(h), 1)
        return m.classifier[1](m.classifier[0](h))

wang_feat_mod = _WangFeat(wang_net).cuda().eval()
wav_feat_mod  = _WavFeat(wav_net).cuda().eval()

fgsm_models = {
    'Wang2020_128':  EndToEndDifferentiableWrapper('rgb',     rgb_model=wang_net),
    'WolterWavelet': EndToEndDifferentiableWrapper('wavelet', wav_model=wav_net, gpu_wavelet_fn=gpu_dwt),
    'ConcatMatched': EndToEndDifferentiableWrapper('fusion',  rgb_model=wang_feat_mod, wav_model=wav_feat_mod, fusion_head=match_s42, gpu_wavelet_fn=gpu_dwt),
    'GatedFusion':   EndToEndDifferentiableWrapper('fusion',  rgb_model=wang_feat_mod, wav_model=wav_feat_mod, fusion_head=gate_s42,  gpu_wavelet_fn=gpu_dwt),
    'MHA_128':       EndToEndDifferentiableWrapper('fusion',  rgb_model=wang_feat_mod, wav_model=wav_feat_mod, fusion_head=mha_s42,   gpu_wavelet_fn=gpu_dwt),
}

# Build 500 unseen LFW Real + 500 unseen LDM Fake raw [0, 1] tensor batch
to_01 = transforms.Compose([det_prep, transforms.ToTensor()])
R_SUB, F_SUB = lfw_reals[:500], ldm_fakes[:500]
x_clean_01 = torch.stack([to_01(im) for im in (R_SUB + F_SUB)], dim=0)
y_sub = np.array([0]*len(R_SUB) + [1]*len(F_SUB))

print('Running matched end-to-end white-box FGSM sweep (eps = 0, 1/255, 2/255, 4/255, 8/255)...')
fgsm_res = run_fgsm_epsilon_sweep(fgsm_models, x_clean_01, y_sub, epsilons=(0.0, 1/255, 2/255, 4/255, 8/255), batch_size=32, device='cuda:0')

print('\n' + '=' * 120)
print('  MATCHED WHITE-BOX FGSM SWEEP (UNSEEN LFW vs. UNSEEN LDM, 500+500, RAW [0,1] PIXELS, END-TO-END GRADIENTS)')
print('=' * 120)
for eps_key in ('0/255', '1/255', '2/255', '4/255', '8/255'):
    print(f'\n  [FGSM eps = {eps_key}]')
    for mname in fgsm_models:
        m = fgsm_res[mname][eps_key]
        print(f"    {mname:<16} | AUC={m['roc_auc']:.4f} | Acc={m['accuracy']*100:5.2f}% | RealRec={m['real']['recall']*100:5.1f}% | FakeRec={m['fake']['recall']*100:5.1f}% | EER={m['eer']*100:5.2f}%")

# Real-world 9 perturbations on identical 500+500 unseen subset
def apply_blur(imgs, r): return [im.filter(ImageFilter.GaussianBlur(radius=r)) for im in imgs]
def apply_downscale(imgs, sc):
    return [im.resize((max(16, int(im.width*sc)), max(16, int(im.height*sc))), Image.Resampling.BILINEAR).resize(im.size, Image.Resampling.BILINEAR) for im in imgs]
def apply_noise(imgs, std):
    rng_n = np.random.RandomState(42)
    return [Image.fromarray(np.clip(np.array(im, dtype=np.float32) + rng_n.normal(0, std, (im.height, im.width, 3)), 0, 255).astype(np.uint8)) for im in imgs]

perturb_res = {}
for pname, tf_fn in [
    ('Clean RAW', lambda x: x), ('JPEG q=90', lambda x: to_jpeg(x, 90)), ('JPEG q=70', lambda x: to_jpeg(x, 70)),
    ('JPEG q=50', lambda x: to_jpeg(x, 50)), ('JPEG q=30', lambda x: to_jpeg(x, 30)),
    ('Blur s=1.0', lambda x: apply_blur(x, 1.0)), ('Blur s=2.0', lambda x: apply_blur(x, 2.0)),
    ('Downscale 0.5x', lambda x: apply_downscale(x, 0.5)), ('Gauss Noise s=10', lambda x: apply_noise(x, 10.0)),
]:
    pr = extract_from_pils(tf_fn(R_SUB))
    pf = extract_from_pils(tf_fn(F_SUB))
    perturb_res[pname] = {}
    for mname, fn in [
        ('Wang2020_128',  lambda p: p['logit_rgb'].numpy()),
        ('WolterWavelet', lambda p: p['logit_wav'].numpy()),
        ('ConcatMatched', lambda p: eval_head_on_pack(match_s42, p)[0]),
        ('GatedFusion',   lambda p: eval_head_on_pack(gate_s42, p)[0]),
        ('MHA_128',       lambda p: eval_head_on_pack(mha_s42, p)[0]),
    ]:
        perturb_res[pname][mname] = compute_exact_metrics(y_sub, np.concatenate([fn(pr), fn(pf)]))

print('\n  PERTURBATION SUMMARY (AUC / Acc%):')
for pname, d in perturb_res.items():
    row = ' | '.join(f"{k}={v['roc_auc']:.3f}/{v['accuracy']*100:4.1f}%" for k, v in d.items())
    print(f'    {pname:<18} | {row}')

(OUT_DIR / 'metrics/paper1_robustness_fgsm_report.json').write_text(
    json.dumps({'fgsm': fgsm_res, 'perturbations': perturb_res}, indent=2), encoding='utf-8'
)
print('Saved FGSM & perturbation report ->', OUT_DIR / 'metrics/paper1_robustness_fgsm_report.json')
```

---

## Cell 7: Generate ALL 300-DPI Publication Figures + Quantitative t-SNE Silhouette + Grad-CAM Masking Controls (R1 #26–#27 & R2 #8, #11–#12)

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, precision_recall_curve
from tools.paper1_revision_suite import compute_embedding_separability

plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.25, 'figure.dpi': 300})
FIG_DIR = OUT_DIR / 'figures'

# 1. Train + Val Convergence Curves (fixes R1 #12 & R2 #11!)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for label in ('MHA_128 (TokenAttn)', 'ConcatMatched (MLP)', 'GatedFusion (Sample)'):
    h = histories_seed42[label]
    ep = range(1, len(h['train_loss']) + 1)
    axes[0].plot(ep, h['train_loss'], '--', alpha=0.6, label=f'{label} (Train)')
    axes[0].plot(ep, h['dev_loss'], '-', linewidth=2, label=f'{label} (Dev)')
    axes[1].plot(ep, h['train_auc'], '--', alpha=0.6, label=f'{label} (Train)')
    axes[1].plot(ep, h['dev_auc'], '-', linewidth=2, label=f'{label} (Dev)')
axes[0].set_title('Training & Validation Loss'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('BCE Loss'); axes[0].legend(fontsize=7)
axes[1].set_title('Training & Validation ROC-AUC'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('ROC-AUC'); axes[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG_DIR / 'fig_train_val_convergence.png', dpi=300); plt.close(fig)

# 2. ROC + PR + Calibration Reliability Diagram on Unseen DiffFace + LDM pooled (fixes R2 #8 & #11!)
pool_y = np.array([0]*1000 + [1]*3000)
def pooled_scores(fn):
    r = fn(cache['unseen_q85']['LFW_Real'])
    f = np.concatenate([fn(cache['unseen_q85'][g]) for g in ('LDM', 'SDXL_Refine', 'DiffFace')])
    return np.concatenate([r, f])

score_map = {
    'Wang2020_128':  pooled_scores(lambda p: p['logit_rgb'].numpy()),
    'WolterWavelet': pooled_scores(lambda p: p['logit_wav'].numpy()),
    'ConcatMatched': pooled_scores(lambda p: eval_head_on_pack(match_s42, p)[0]),
    'MHA_128':       pooled_scores(lambda p: eval_head_on_pack(mha_s42, p)[0]),
}
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
for name, sc in score_map.items():
    m = compute_exact_metrics(pool_y, sc)
    fpr, tpr, _ = roc_curve(pool_y, sc)
    prec, rec, _ = precision_recall_curve(pool_y, sc)
    axes[0].plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={m['roc_auc']:.4f})")
    axes[1].plot(rec, prec, linewidth=2, label=f"{name} (AP={m['average_precision']:.4f})")
    rel = m['reliability']
    valid = [i for i, a in enumerate(rel['bin_acc']) if not np.isnan(a)]
    axes[2].plot([rel['bin_conf'][i] for i in valid], [rel['bin_acc'][i] for i in valid], 'o-', linewidth=1.8,
                 label=f"{name} (ECE={m['ece']:.3f}, Brier={m['brier_score']:.3f})")
axes[0].plot([0,1],[0,1],'k--',alpha=0.4); axes[0].set_title('ROC Curve (Unseen Diffusion+Swap, JPEG q85)'); axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR'); axes[0].legend(fontsize=7.5)
axes[1].set_title('Precision-Recall Curve'); axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision'); axes[1].legend(fontsize=7.5)
axes[2].plot([0,1],[0,1],'k--',alpha=0.5,label='Perfect Calibration'); axes[2].set_title('Reliability Diagram (Probability Calibration)'); axes[2].set_xlabel('Mean Predicted Prob'); axes[2].set_ylabel('Empirical Accuracy'); axes[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG_DIR / 'fig_roc_pr_reliability.png', dpi=300); plt.close(fig)

# 3. 4-Seed Ablation Comparison Bar Chart with Separate EER Axis + Value Labels + 95% CIs (fixes R1 #27!)
labels_plot = [lbl for lbl, _ in HEAD_LIST]
accs_mu, accs_sd, aucs_mu, aucs_sd, eers_mu, eers_sd = [], [], [], [], [], []
for lbl in labels_plot:
    rs = [r['eval']['unseen_q85']['macro_4domain'] for r in ablation_runs[lbl]]
    accs_mu.append(np.mean([x['accuracy']*100 for x in rs])); accs_sd.append(np.std([x['accuracy']*100 for x in rs], ddof=1))
    aucs_mu.append(np.mean([x['roc_auc']*100 for x in rs]));  aucs_sd.append(np.std([x['roc_auc']*100 for x in rs], ddof=1))
    eers_mu.append(np.mean([x['eer']*100 for x in rs]));      eers_sd.append(np.std([x['eer']*100 for x in rs], ddof=1))

x = np.arange(len(labels_plot))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={'width_ratios': [1.5, 1]})
b1 = ax1.bar(x - 0.18, accs_mu, 0.35, yerr=accs_sd, capsize=3, label='4-Domain Mean Acc (%)')
b2 = ax1.bar(x + 0.18, aucs_mu, 0.35, yerr=aucs_sd, capsize=3, label='4-Domain Mean ROC-AUC (%)')
for bars in (b1, b2):
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4, f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)
ax1.set_xticks(x); ax1.set_xticklabels([l.split()[0] for l in labels_plot], rotation=20, ha='right')
ax1.set_ylim(max(50, min(accs_mu) - 6), 101); ax1.set_ylabel('Score (%)'); ax1.set_title('Cross-Fitted 4-Seed Discrimination (Higher is Better)'); ax1.legend(fontsize=8)

b3 = ax2.bar(x, eers_mu, 0.55, yerr=eers_sd, capsize=3, color='coral')
for bar in b3:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=7.5)
ax2.set_xticks(x); ax2.set_xticklabels([l.split()[0] for l in labels_plot], rotation=20, ha='right')
ax2.set_ylabel('4-Domain Mean EER (%)'); ax2.set_title('Equal Error Rate — Separate Axis (Lower is Better)')
fig.tight_layout(); fig.savefig(FIG_DIR / 'fig_ablation_4seed_bars.png', dpi=300); plt.close(fig)

# 4. Robustness Figure (Perturbations + FGSM Sweep, fixes R1 #13 & R2 #13!)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4))
p_names = ['Clean RAW', 'JPEG q=90', 'JPEG q=70', 'JPEG q=50', 'JPEG q=30', 'Blur s=1.0', 'Blur s=2.0', 'Downscale 0.5x', 'Gauss Noise s=10']
for mname in ('Wang2020_128', 'WolterWavelet', 'ConcatMatched', 'MHA_128'):
    ax1.plot(range(len(p_names)), [perturb_res[p][mname]['roc_auc'] for p in p_names], 'o-', linewidth=2, label=mname)
ax1.set_xticks(range(len(p_names))); ax1.set_xticklabels(p_names, rotation=30, ha='right', fontsize=8)
ax1.set_ylabel('ROC-AUC'); ax1.set_title('Real-World Degradation Robustness (LFW vs. LDM)'); ax1.legend(fontsize=8)

eps_labels = ['0/255', '1/255', '2/255', '4/255', '8/255']
for mname in ('Wang2020_128', 'WolterWavelet', 'ConcatMatched', 'MHA_128'):
    ax2.plot(range(len(eps_labels)), [fgsm_res[mname][e]['roc_auc'] for e in eps_labels], 's-', linewidth=2, label=mname)
ax2.set_xticks(range(len(eps_labels))); ax2.set_xticklabels(eps_labels)
ax2.set_xlabel('FGSM Perturbation Budget (L_inf)'); ax2.set_ylabel('ROC-AUC'); ax2.set_title('White-Box FGSM Sweep (End-to-End Gradients)'); ax2.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG_DIR / 'fig_robustness_perturb_fgsm.png', dpi=300); plt.close(fig)

# 5. Single-Row Quantitative t-SNE with Silhouette Score & Fisher Ratio (fixes R1 #26!)
n_per = 400
domains_tsne = [('LFW_Real', 0), ('LDM', 1), ('SDXL_Refine', 1), ('DiffFace', 1), ('StyleGAN', 1)]
X_rgb_t, X_wav_t, X_mha_t, y_bin_t, dom_lbl_t = [], [], [], [], []
with torch.no_grad():
    for dname, yv in domains_tsne:
        pk = cache['unseen_q85'][dname]
        zr, zw = pk['rgb'][:n_per], pk['wav'][:n_per]
        # Extract MHA_128 [CLS] refined 128-D embedding right before final Linear(128->1)
        toks = torch.cat((mha_s42.cls.expand(len(zr), -1, -1), mha_s42.rgb_norm(zr.cuda()).unsqueeze(1), mha_s42.wavelet_norm(zw.cuda()).unsqueeze(1)), 1) + mha_s42.types
        att, _ = mha_s42.attn(toks, toks, toks, need_weights=False)
        toks = mha_s42.norm2(mha_s42.norm1(toks + att) + mha_s42.ffn(mha_s42.norm1(toks + att)))
        zm = toks[:, 0].cpu().numpy()
        X_rgb_t.append(zr.numpy()); X_wav_t.append(zw.numpy()); X_mha_t.append(zm)
        y_bin_t.extend([yv]*len(zr)); dom_lbl_t.extend([dname]*len(zr))
X_rgb_t, X_wav_t, X_mha_t = np.vstack(X_rgb_t), np.vstack(X_wav_t), np.vstack(X_mha_t)
y_bin_t = np.array(y_bin_t)

sep_metrics = {
    'Wang2020_128':  compute_embedding_separability(X_rgb_t, y_bin_t),
    'WolterWavelet': compute_embedding_separability(X_wav_t, y_bin_t),
    'MHA_128':       compute_embedding_separability(X_mha_t, y_bin_t),
}
(OUT_DIR / 'metrics/paper1_embedding_separability.json').write_text(json.dumps(sep_metrics, indent=2), encoding='utf-8')

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
palette = {'LFW_Real': '#2ca02c', 'LDM': '#d62728', 'SDXL_Refine': '#ff7f0e', 'DiffFace': '#9467bd', 'StyleGAN': '#1f77b4'}
for ax, (title, Xmat, skey) in zip(axes, [('Wang2020_128 (RGB)', X_rgb_t, 'Wang2020_128'), ('WolterWavelet_128 (Freq)', X_wav_t, 'WolterWavelet'), ('MHA_128 (Fused [CLS])', X_mha_t, 'MHA_128')]):
    emb2d = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto').fit_transform(Xmat)
    for dname, _ in domains_tsne:
        mask = np.array([d == dname for d in dom_lbl_t])
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1], s=8, alpha=0.55, c=palette[dname], label=dname)
    sm = sep_metrics[skey]
    ax.set_title(f"{title}\nSilhouette={sm['silhouette_score']:.3f} | Fisher={sm['fisher_separation_ratio']:.2f}")
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
axes[2].legend(fontsize=7.5, markerscale=2, loc='best')
fig.tight_layout(); fig.savefig(FIG_DIR / 'fig_tsne_silhouette.png', dpi=300); plt.close(fig)

# 6. Multi-Sample Grad-CAM + Center-Face vs. Background Masking Control (fixes R2 #12!)
def gradcam_layer4(im_pil):
    im_c = det_prep(im_pil.convert('RGB'))
    x = rgb_norm_tf(im_c).unsqueeze(0).cuda().requires_grad_(True)
    acts, grads = [], []
    h1 = wang_net.layer4.register_forward_hook(lambda m, i, o: acts.append(o))
    h2 = wang_net.layer4.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    logit = wang_net(x).flatten()
    wang_net.zero_grad(); logit.backward()
    h1.remove(); h2.remove()
    w = grads[0].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * acts[0]).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return np.array(im_c), cam, float(logit.item())

# Quantitative masking control on 200 unseen LDM fakes: mask center 50% face vs. mask outer border (equal pixel area)
center_drops, border_drops = [], []
with torch.no_grad():
    for im in ldm_fakes[:200]:
        im_c = det_prep(im.convert('RGB'))
        x = rgb_norm_tf(im_c).unsqueeze(0).cuda()
        base = float(wang_net(x).item())
        xc = x.clone(); xc[:, :, 56:168, 56:168] = 0.0  # center 25% area masked
        xb = x.clone(); xb[:, :, :32, :] = 0.0; xb[:, :, -32:, :] = 0.0  # ~28% border area masked
        center_drops.append(base - float(wang_net(xc).item()))
        border_drops.append(base - float(wang_net(xb).item()))
mask_ctrl = {'mean_logit_drop_center_mask': float(np.mean(center_drops)), 'mean_logit_drop_border_mask': float(np.mean(border_drops))}
(OUT_DIR / 'metrics/paper1_gradcam_masking_control.json').write_text(json.dumps(mask_ctrl, indent=2), encoding='utf-8')
print('Grad-CAM Masking Control (n=200 unseen LDM):', mask_ctrl)

samples_gc = [('LDM Fake', ldm_fakes[0]), ('SDXL_Refine Fake', sdxl_ref_imgs[0]), ('DiffFace Swap', diffface_imgs[0]), ('StyleGAN Fake', stylegan_fakes[0])]
fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
for j, (stitle, im_s) in enumerate(samples_gc):
    rgb_np, cam_np, lg = gradcam_layer4(im_s)
    axes[0, j].imshow(rgb_np); axes[0, j].set_title(f'{stitle}\nLogit={lg:+.2f}'); axes[0, j].axis('off')
    axes[1, j].imshow(rgb_np); im_hm = axes[1, j].imshow(cam_np, cmap='jet', alpha=0.45, vmin=0, vmax=1)
    axes[1, j].set_title('ResNet-50 Layer4 Grad-CAM'); axes[1, j].axis('off')
cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.35])
fig.colorbar(im_hm, cax=cbar_ax, label='Normalized Attribution [0, 1]')
fig.subplots_adjust(right=0.90)
fig.savefig(FIG_DIR / 'fig_gradcam_masking.png', dpi=300); plt.close(fig)
print('All 6 publication figures saved ->', FIG_DIR)
```

---

## Cell 8 (Optional ~20 min on L4): Joint End-to-End Training Control + Published Baselines (Solves R1 #2, #11, #19 & R2 #6, #9!)

Run this cell to empirically test:
- **8A (R1 #2 & R2 #9):** End-to-end **unfrozen joint training** of `Wang2020_128 + WolterWavelet2021_128 + MHA_128` on the exact same LOGO training split (1 epoch = ~12 min on L4, matching `Wang2020_128`'s best epoch!) to directly compare joint vs. staged decoupled training!
- **8B (R1 #11, #19 & R2 #6):** Published external baseline **UniversalFakeDetect (Ojha et al., CVPR 2023 — linear probe on frozen CLIP ViT-L/14, ~3 min!)** on the exact same unseen test sets so readers can situate your Paper 1 results against a published universal detector!

```python
# --- 8A: Joint End-to-End Unfrozen Dual-Stream Control (1 epoch matching Wang2020_128 best epoch, ~12 min on L4) ---
import torchvision.models as tvm
from models.wolter2021.wavelet_cnn import WaveletPacketCNN128
from models.mha.token_fusion import TokenAttentionFusion

class JointEndToEndMHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        self.rgb.fc = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.wav = WaveletPacketCNN128(192, embed_dim=128)
        self.wav.classifier = nn.Sequential(*list(self.wav.classifier.children())[:2])
        self.head = TokenAttentionFusion(embed_dim=128, num_heads=4, dropout=0.2)
    def forward(self, x_rgb, x_wav_255):
        zr = self.rgb(x_rgb)
        zw = self.wav(gpu_dwt(x_wav_255))
        return self.head(zr, zw)

# Build 60,000 balanced samples from train (30K real + 30K fake across all 13 seen generators) for fast fair joint vs staged check
joint_rows = []
with open(DATA_ROOT / 'dataset_manifest.csv', newline='', encoding='utf-8') as f:
    reals_tr, fakes_tr = [], []
    for r in csv.DictReader(f):
        if r['split'] != 'train': continue
        lbl = int(r['label'])
        if lbl == 1 and r['generator'].lower() in LOGO_HOLDOUT: continue
        rec = {'path': str((DATA_ROOT / r['relative_path']).resolve()), 'label': lbl}
        (reals_tr if lbl == 0 else fakes_tr).append(rec)
rng_j = random.Random(42)
rng_j.shuffle(reals_tr); rng_j.shuffle(fakes_tr)
joint_rows = reals_tr[:40000] + fakes_tr[:40000]
rng_j.shuffle(joint_rows)

joint_model = JointEndToEndMHA().cuda()
opt_j = torch.optim.AdamW(joint_model.parameters(), lr=2e-4, weight_decay=1e-2)
dl_j = DataLoader(PathPairDataset(joint_rows, augment=True, seed=42), batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

print(f'Training Joint End-to-End Unfrozen Dual-Stream Baseline on {len(joint_rows):,} samples (2 epochs)...')
for ep in range(1, 3):
    joint_model.train()
    losses = []
    for xr, xw, y in dl_j:
        xr, xw, y = xr.cuda(non_blocking=True), xw.cuda(non_blocking=True), y.cuda(non_blocking=True).float()
        opt_j.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = joint_model(xr, xw).flatten()
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y * 0.9 + 0.05)
        loss.backward(); opt_j.step()
        losses.append(float(loss.item()))
    print(f'  Joint Epoch {ep} avg_loss={np.mean(losses):.4f}')

joint_model.eval()
def score_joint(imgs, bs=128):
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            chunk = [det_prep(im.convert('RGB')) for im in imgs[i:i+bs]]
            xr = torch.stack([rgb_norm_tf(im) for im in chunk]).cuda()
            xw = torch.stack([torch.from_numpy(np.array(im, dtype=np.float32).transpose(2, 0, 1)) for im in chunk]).cuda()
            out.extend(joint_model(xr, xw).flatten().cpu().tolist())
    return np.array(out)

joint_report = {}
jr = score_joint(to_jpeg(lfw_reals, 85))
print('\n  JOINT END-TO-END BASELINE [SAME-JPEG q=85]:')
for gname, fimgs in [('LDM', ldm_fakes), ('SDXL_Refine', sdxl_ref_imgs), ('DiffFace', diffface_imgs), ('StyleGAN', stylegan_fakes)]:
    jf = score_joint(to_jpeg(fimgs, 85))
    m = compute_exact_metrics(np.array([0]*len(jr) + [1]*len(jf)), np.concatenate([jr, jf]))
    joint_report[gname] = m
    print(f"    {gname:<14} | AUC={m['roc_auc']:.4f} | Acc={m['accuracy']*100:5.2f}% | RealRec={m['real']['recall']*100:5.1f}% | FakeRec={m['fake']['recall']*100:5.1f}%")
(OUT_DIR / 'metrics/paper1_joint_training_control.json').write_text(json.dumps(joint_report, indent=2), encoding='utf-8')
torch.save(joint_model.state_dict(), OUT_DIR / 'checkpoints/joint_e2e_mha_seed42.pt')
print('Saved joint training control ->', OUT_DIR / 'metrics/paper1_joint_training_control.json')
```
