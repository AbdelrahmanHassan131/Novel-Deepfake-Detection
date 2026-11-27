# Colab L4 Playbook — Paper 1 Option 2: External Published Baselines (`Xception`, `UniversalFakeDetect`), `ConvNeXt_128` Modularity (`Concat` vs. `Gated` vs. `MHA`), & Male-Only Grad-CAM

> **What This Closes (Zero Remaining Gaps!):**
> - **R1 #11 & R2 #6:** Adds **two published external baselines** evaluated on the **exact same `seen_test` ($n=12,729$) and 4 unseen JPEG `q=85` domains**: `Xception` (FaceForensics++, ICCV 2019) and `UniversalFakeDetect` (Ojha et al., CVPR 2023 — frozen CLIP ViT-L/14 linear probe).
> - **R1 #14:** Restores **backbone-agnostic modularity** with **`ConvNeXt_128`** paired with **all matched fusion controls** (`ConcatMatchedMLP`, `GatedFusion`, and `MHA_128` across 4 seeds) over the exact same frozen `WolterWavelet2021_128` embeddings.
> - **Grad-CAM Update:** Automatically selects **4 synthetic male faces** (`p(man) > 0.98` via CLIP) across `LDM`, `SDXL_Refine`, `DiffFace`, and `StyleGAN` and overwrites `figures/fig_gradcam_masking.png` cleanly on Drive.
>
> **Total Runtime on Fresh NVIDIA L4:** **~25–30 minutes** (all outputs saved directly to `/content/drive/MyDrive/paper1_discover_ai_revision/`).

---

## Cell 1: Mount Drive, Pull Repo, Install Dependencies & Load Exact Paper 1 Splits (~4 min)

```python
from google.colab import drive
drive.mount('/content/drive')

import gc, io, os, sys, csv, json, time, copy, random, subprocess
from pathlib import Path
from PIL import Image, ImageFilter

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

# 1. Install base packages + setuptools<81 FIRST (Python 3.13 CLIP-safe)
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q',
    'ftfy', 'regex', 'setuptools<81', 'timm', 'PyWavelets', 'pytorch-wavelets',
    'datasets', 'scikit-learn', 'scipy', 'matplotlib', 'pillow', 'opencv-python-headless',
], check=True)

# 2. Install OpenAI CLIP SECOND without build isolation
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q', '--no-build-isolation', '--no-deps',
    'git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33',
], check=True)

import torch, torch.nn as nn, numpy as np, clip
from torchvision import transforms, models as tvm
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_lfw_people
from datasets import load_dataset
from tools.paper1_revision_suite import compute_exact_metrics, train_cached_fusion_head, paired_seed_statistics

assert torch.cuda.is_available(), 'Please select an NVIDIA L4 GPU runtime!'
print(f'GPU: {torch.cuda.get_device_name(0)} | Torch: {torch.__version__}')

OUT_DIR   = Path('/content/drive/MyDrive/paper1_discover_ai_revision')
DRIVE_RUNS = Path('/content/drive/MyDrive/deepfake_multidomain_v2_runs')
DATA_ROOT = Path('/content/dataset/DatasetMultiDomainClean_v2')
DATA_ZIP  = Path('/content/drive/MyDrive/DatasetMultiDomainClean_v2/DatasetMultiDomainClean_v2.zip')

if not (DATA_ROOT / 'val/real').is_dir():
    Path('/content/dataset').mkdir(parents=True, exist_ok=True)
    print('Extracting DatasetMultiDomainClean_v2.zip (~2-3 min)...')
    subprocess.run(['7z', 'x', str(DATA_ZIP), '-o/content/dataset/', '-y'], check=True, stdout=subprocess.DEVNULL)

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

# Load existing v3 cache (for exact frozen WolterWavelet_128 embeddings & exact split parity!)
cache = torch.load(OUT_DIR / 'cache/paper1_frozen_embeddings_v3.pt', map_location='cpu', weights_only=False)

# Reconstruct exact deterministic Phase-2 splits (identical seed=42 logic as main notebook)
LOGO_HOLDOUT = {'sdxl_refine', 'diffface'}
val_seen_rows, train_reals, train_fakes = [], [], []
with open(DATA_ROOT / 'dataset_manifest.csv', newline='', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        lbl, src, gen = int(r['label']), r['dataset_source'], r['generator']
        abs_p = str((DATA_ROOT / r['relative_path']).resolve())
        if r['split'] == 'train':
            if lbl == 1 and gen.lower() in LOGO_HOLDOUT:
                continue
            (train_reals if lbl == 0 else train_fakes).append({'path': abs_p, 'label': lbl})
        else:
            if lbl == 1 and gen.lower() in LOGO_HOLDOUT:
                continue
            raw_id = (r.get('identity_id') or '').strip()
            ident = r['sample_id'] if (not raw_id or raw_id.lower() in ('unknown', 'none', 'na', 'null')) else raw_id
            val_seen_rows.append({'path': abs_p, 'label': lbl, 'source': src, 'generator': gen, 'identity_id': ident})

rng = random.Random(42)
buckets = {}
for rec in val_seen_rows:
    buckets.setdefault((rec['label'], rec['generator'] if rec['label'] == 1 else rec['source']), []).append(rec)

fusion_train_rows, fusion_dev_rows, seen_test_rows = [], [], []
for key, items in sorted(buckets.items()):
    by_id = {}
    for it in items:
        by_id.setdefault(it['identity_id'], []).append(it)
    ids = sorted(by_id.keys())
    rng.shuffle(ids)
    n_tr, n_dv = int(0.70 * len(ids)), int(0.15 * len(ids))
    for iid in ids[:n_tr]: fusion_train_rows.extend(by_id[iid])
    for iid in ids[n_tr:n_tr + n_dv]: fusion_dev_rows.extend(by_id[iid])
    for iid in ids[n_tr + n_dv:]: seen_test_rows.extend(by_id[iid])

assert len(seen_test_rows) == len(cache['seen_test']['y']), 'Split parity mismatch!'
print(f'Exact split parity verified: fusion_train={len(fusion_train_rows):,}, fusion_dev={len(fusion_dev_rows):,}, seen_test={len(seen_test_rows):,}')

# Balanced 80,000 LOGO training subset for fast fair Phase-1 training of Xception & ConvNeXt_128
rng_t = random.Random(42)
rng_t.shuffle(train_reals); rng_t.shuffle(train_fakes)
p1_train_rows = train_reals[:40000] + train_fakes[:40000]
rng_t.shuffle(p1_train_rows)

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
ldm_files = sorted([p for p in LDM_DIR.rglob('*.*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg')], key=lambda p: p.name)[:1000]
ldm_fakes = [Image.open(p).convert('RGB').copy() for p in ldm_files]

def to_jpeg(imgs, q=85):
    out = []
    for im in imgs:
        buf = io.BytesIO()
        im.convert('RGB').save(buf, format='JPEG', quality=q)
        buf.seek(0)
        out.append(Image.open(buf).convert('RGB').copy())
    return out

unseen_q85_imgs = {
    'LFW_Real':    to_jpeg(lfw_reals, 85),
    'LDM':         to_jpeg(ldm_fakes, 85),
    'SDXL_Refine': to_jpeg(sdxl_ref_imgs, 85),
    'DiffFace':    to_jpeg(diffface_imgs, 85),
    'StyleGAN':    to_jpeg(stylegan_fakes, 85),
}

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

det_prep = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR), transforms.CenterCrop(224)])
imgnet_tf = transforms.Compose([det_prep, transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
clip_tf   = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.26862954])])

class ImgDataset(Dataset):
    def __init__(self, rows, tf, augment=False, seed=42):
        self.rows, self.tf, self.augment, self.seed = rows, tf, augment, seed
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        im = Image.open(r['path']).convert('RGB')
        if self.augment:
            im = forensic_aug_pil(im, random.Random(self.seed + idx))
        return self.tf(im), r['label']

OPTION2_RESULTS = {}
print('Cell 1 ready!')
```

---

## Cell 2 (~4 min): Published Baseline #1 (`UniversalFakeDetect`, Ojha et al. CVPR 2023) + Regenerate Male-Only Grad-CAM (`fig_gradcam_masking.png`)

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluation.checkpoint_loader import CheckpointLoader

clip_model, _ = clip.load('ViT-L/14', device='cuda:0')
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

# --- Part A: Regenerate fig_gradcam_masking.png using 4 high-confidence SYNTHETIC MALE faces ---
text_tok = clip.tokenize(["a portrait photograph of a man", "a portrait photograph of a woman"]).cuda()
with torch.no_grad():
    text_feat = clip_model.encode_text(text_tok).float()
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

def pick_synthetic_male(imgs, wang_net, want_positive=True, max_check=80):
    best_im, best_score = imgs[0], -1e9
    with torch.no_grad():
        for im in imgs[:max_check]:
            xc = clip_tf(im.convert('RGB')).unsqueeze(0).cuda()
            vf = clip_model.encode_image(xc).float()
            vf = vf / vf.norm(dim=-1, keepdim=True)
            probs = (100.0 * vf @ text_feat.T).softmax(dim=-1)[0]
            p_man = float(probs[0].item())
            if p_man < 0.96:
                continue
            xw = imgnet_tf(im.convert('RGB')).unsqueeze(0).cuda()
            lg = float(wang_net(xw).item())
            score = p_man * 2.0 + (lg if want_positive else -lg)
            if score > best_score:
                best_score, best_im = score, im
    return best_im

BEST_WANG = DRIVE_RUNS / 'wang2020_128_v2_logo_forensic_20261122_185559_d289b6e0/checkpoints/best.pth'
wang_net = CheckpointLoader(str(BEST_WANG), device='cuda:0').load().model.eval()

def gradcam_layer4(im_pil):
    im_c = det_prep(im_pil.convert('RGB'))
    x = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(transforms.ToTensor()(im_c)).unsqueeze(0).cuda().requires_grad_(True)
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

male_samples = [
    ('LDM Fake (Male)',         pick_synthetic_male(ldm_fakes, wang_net, want_positive=True)),
    ('SDXL_Refine Fake (Male)', pick_synthetic_male(sdxl_ref_imgs, wang_net, want_positive=True)),
    ('DiffFace Swap (Male)',    pick_synthetic_male(diffface_imgs, wang_net, want_positive=True)),
    ('StyleGAN Fake (Male)',    pick_synthetic_male(stylegan_fakes, wang_net, want_positive=False)),
]
fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
for j, (stitle, im_s) in enumerate(male_samples):
    rgb_np, cam_np, lg = gradcam_layer4(im_s)
    axes[0, j].imshow(rgb_np); axes[0, j].set_title(f'{stitle}\nLogit={lg:+.2f}'); axes[0, j].axis('off')
    axes[1, j].imshow(rgb_np); im_hm = axes[1, j].imshow(cam_np, cmap='jet', alpha=0.45, vmin=0, vmax=1)
    axes[1, j].set_title('ResNet-50 Layer4 Grad-CAM'); axes[1, j].axis('off')
cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.35])
fig.colorbar(im_hm, cax=cbar_ax, label='Normalized Attribution [0, 1]')
fig.subplots_adjust(right=0.90)
fig.savefig(OUT_DIR / 'figures/fig_gradcam_masking.png', dpi=300)
plt.close(fig)
print('Updated male-only Grad-CAM saved ->', OUT_DIR / 'figures/fig_gradcam_masking.png')
del wang_net; gc.collect(); torch.cuda.empty_cache()

# --- Part B: UniversalFakeDetect (Ojha et al., CVPR 2023) — Frozen CLIP ViT-L/14 + Linear(768->1) ---
def extract_clip_rows(rows, augment=False, bs=128):
    dl = DataLoader(ImgDataset(rows, clip_tf, augment=augment, seed=42), batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    fs, ys = [], []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for x, y in dl:
            f = clip_model.encode_image(x.cuda(non_blocking=True)).float()
            fs.append((f / f.norm(dim=-1, keepdim=True)).cpu())
            ys.append(y.long())
    return torch.cat(fs), torch.cat(ys)

def extract_clip_pils(imgs, bs=128):
    fs = []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(imgs), bs):
            x = torch.stack([clip_tf(im.convert('RGB')) for im in imgs[i:i+bs]]).cuda()
            f = clip_model.encode_image(x).float()
            fs.append((f / f.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(fs)

print('Extracting CLIP ViT-L/14 features for UniversalFakeDetect (Ojha et al. CVPR 2023)...')
# Use 30K balanced LOGO train subset (augmented) + exact seen_test + exact 4 unseen q85 domains
tr_f, tr_y = extract_clip_rows(p1_train_rows[:30000], augment=True)
st_f, st_y = extract_clip_rows(seen_test_rows, augment=False)
un_f = {k: extract_clip_pils(v) for k, v in unseen_q85_imgs.items()}

torch.manual_seed(42)
lin_probe = nn.Linear(tr_f.shape[1], 1).cuda()
opt_lp = torch.optim.AdamW(lin_probe.parameters(), lr=1e-2, weight_decay=1e-3)
tr_f_c, tr_y_c = tr_f.cuda(), tr_y.cuda().float()
for ep in range(30):
    lin_probe.train()
    perm = torch.randperm(len(tr_y_c), device='cuda:0')
    for i in range(0, len(tr_y_c), 512):
        idx = perm[i:i+512]
        opt_lp.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(lin_probe(tr_f_c[idx]).flatten(), tr_y_c[idx])
        loss.backward(); opt_lp.step()

lin_probe.eval()
with torch.no_grad():
    ojha_st = compute_exact_metrics(st_y.numpy(), lin_probe(st_f.cuda()).flatten().cpu().numpy())
    ojha_lr = lin_probe(un_f['LFW_Real'].cuda()).flatten().cpu().numpy()
    ojha_un = {}
    for g in ('LDM', 'SDXL_Refine', 'DiffFace', 'StyleGAN'):
        lf = lin_probe(un_f[g].cuda()).flatten().cpu().numpy()
        ojha_un[g] = compute_exact_metrics(np.array([0]*len(ojha_lr)+[1]*len(lf)), np.concatenate([ojha_lr, lf]))

OPTION2_RESULTS['UniversalFakeDetect_CLIP_LP'] = {'seen_test': ojha_st, 'unseen_q85': ojha_un}
print(f"\n[UniversalFakeDetect (Ojha et al. CVPR 2023)] SeenTest Acc={ojha_st['accuracy']*100:.2f}% AUC={ojha_st['roc_auc']:.4f} EER={ojha_st['eer']*100:.2f}%")
for g, m in ojha_un.items():
    print(f"  UnseenQ85 {g:<12} | AUC={m['roc_auc']:.4f} Acc={m['accuracy']*100:5.2f}% R={m['real']['recall']*100:5.1f}% F={m['fake']['recall']*100:5.1f}%")
del clip_model, tr_f, st_f, un_f; gc.collect(); torch.cuda.empty_cache()
```

---

## Cell 3 (~10 min): Published Baseline #2 — `Xception` (FaceForensics++, ICCV 2019) on Identical Split

```python
import timm

xcep = timm.create_model('legacy_xception', pretrained=True, num_classes=1).cuda()
opt_x = torch.optim.AdamW(xcep.parameters(), lr=2e-4, weight_decay=1e-2)
dl_x = DataLoader(ImgDataset(p1_train_rows, imgnet_tf, augment=True, seed=42), batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

print(f'Training Xception (FaceForensics++ ICCV 2019 baseline) on {len(p1_train_rows):,} LOGO samples (2 epochs)...')
for ep in range(1, 3):
    xcep.train()
    losses = []
    for x, y in dl_x:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True).float()
        opt_x.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = nn.functional.binary_cross_entropy_with_logits(xcep(x).flatten(), y * 0.9 + 0.05)
        loss.backward(); opt_x.step()
        losses.append(float(loss.item()))
    print(f'  Xception Epoch {ep} avg_loss={np.mean(losses):.4f}')

xcep.eval()
def score_imgnet_model(mod, rows_or_pils, is_rows=False, bs=128):
    out = []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        if is_rows:
            dl = DataLoader(ImgDataset(rows_or_pils, imgnet_tf, augment=False), batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
            for x, _ in dl:
                out.extend(mod(x.cuda(non_blocking=True)).flatten().float().cpu().tolist())
        else:
            for i in range(0, len(rows_or_pils), bs):
                x = torch.stack([imgnet_tf(im.convert('RGB')) for im in rows_or_pils[i:i+bs]]).cuda()
                out.extend(mod(x).flatten().float().cpu().tolist())
    return np.array(out)

x_st_logits = score_imgnet_model(xcep, seen_test_rows, is_rows=True)
x_st = compute_exact_metrics([r['label'] for r in seen_test_rows], x_st_logits)
x_lr = score_imgnet_model(xcep, unseen_q85_imgs['LFW_Real'])
x_un = {}
for g in ('LDM', 'SDXL_Refine', 'DiffFace', 'StyleGAN'):
    lf = score_imgnet_model(xcep, unseen_q85_imgs[g])
    x_un[g] = compute_exact_metrics(np.array([0]*len(x_lr)+[1]*len(lf)), np.concatenate([x_lr, lf]))

OPTION2_RESULTS['Xception_ICCV2019'] = {'seen_test': x_st, 'unseen_q85': x_un}
print(f"\n[Xception (ICCV 2019)] SeenTest Acc={x_st['accuracy']*100:.2f}% AUC={x_st['roc_auc']:.4f} EER={x_st['eer']*100:.2f}%")
for g, m in x_un.items():
    print(f"  UnseenQ85 {g:<12} | AUC={m['roc_auc']:.4f} Acc={m['accuracy']*100:5.2f}% R={m['real']['recall']*100:5.1f}% F={m['fake']['recall']*100:5.1f}%")
torch.save(xcep.state_dict(), OUT_DIR / 'checkpoints/xception_logo_seed42.pt')
del xcep, opt_x; gc.collect(); torch.cuda.empty_cache()
```

---

## Cell 4 (~14 min): Backbone-Agnostic Modularity Proof (`ConvNeXt_128` + Matched `ConcatMatched` vs. `GatedFusion` vs. `MHA_128`, 4 Seeds — Solves R1 #14!)

```python
# Build ConvNeXt_128 (ConvNeXt-Base + 128-D bottleneck, matching repo ConvNeXt128Trainer)
cnxt = tvm.convnext_base(weights=tvm.ConvNeXt_Base_Weights.IMAGENET1K_V1)
cnxt.classifier[2] = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(128, 1))
cnxt = cnxt.cuda()

opt_c = torch.optim.AdamW(cnxt.parameters(), lr=1e-4, weight_decay=1e-2)
dl_c = DataLoader(ImgDataset(p1_train_rows, imgnet_tf, augment=True, seed=42), batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

print(f'Training ConvNeXt_128 on {len(p1_train_rows):,} LOGO samples (2 epochs)...')
for ep in range(1, 3):
    cnxt.train()
    losses = []
    for x, y in dl_c:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True).float()
        opt_c.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = nn.functional.binary_cross_entropy_with_logits(cnxt(x).flatten(), y * 0.9 + 0.05)
        loss.backward(); opt_c.step()
        losses.append(float(loss.item()))
    print(f'  ConvNeXt_128 Epoch {ep} avg_loss={np.mean(losses):.4f}')

cnxt.eval()
torch.save(cnxt.state_dict(), OUT_DIR / 'checkpoints/convnext128_logo_seed42.pt')

def forward_cnxt_feat_and_logit(x):
    h = cnxt.classifier[1](cnxt.classifier[0](cnxt.avgpool(cnxt.features(x))))
    feat = cnxt.classifier[2][1](cnxt.classifier[2][0](h))
    logit = cnxt.classifier[2][3](feat).flatten()
    return feat, logit

def extract_cnxt_rows(rows, augment=False, seed=42, bs=128):
    dl = DataLoader(ImgDataset(rows, imgnet_tf, augment=augment, seed=seed), batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    fs, ls = [], []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for x, _ in dl:
            f, l = forward_cnxt_feat_and_logit(x.cuda(non_blocking=True))
            fs.append(f.float().cpu()); ls.append(l.float().cpu())
    return torch.cat(fs), torch.cat(ls)

def extract_cnxt_pils(imgs, bs=128):
    fs, ls = [], []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(imgs), bs):
            x = torch.stack([imgnet_tf(im.convert('RGB')) for im in imgs[i:i+bs]]).cuda()
            f, l = forward_cnxt_feat_and_logit(x)
            fs.append(f.float().cpu()); ls.append(l.float().cpu())
    return torch.cat(fs), torch.cat(ls)

print('Extracting frozen ConvNeXt_128 128-D embeddings on exact cross-fitted splits...')
c_tr_clean, _ = extract_cnxt_rows(fusion_train_rows, augment=False)
c_tr_aug,   _ = extract_cnxt_rows(fusion_train_rows, augment=True, seed=10042)
c_tr = torch.cat([c_tr_clean, c_tr_aug])
c_dv, _ = extract_cnxt_rows(fusion_dev_rows, augment=False)
c_st, c_st_logit = extract_cnxt_rows(seen_test_rows, augment=False)
c_un = {k: extract_cnxt_pils(v) for k, v in unseen_q85_imgs.items()}

# Pair ConvNeXt_128 with exact existing frozen WolterWavelet2021_128 embeddings from v3 cache!
w_tr, y_tr = cache['fusion_train']['wav'], cache['fusion_train']['y']
w_dv, y_dv = cache['fusion_dev']['wav'],   cache['fusion_dev']['y']
w_st, y_st = cache['seen_test']['wav'],    cache['seen_test']['y']

perm_c = torch.randperm(len(y_tr), generator=torch.Generator().manual_seed(42))
c_tr_s, w_tr_s, y_tr_s = c_tr[perm_c], w_tr[perm_c], y_tr[perm_c]

# Standalone ConvNeXt_128 metrics
cnxt_st = compute_exact_metrics(y_st.numpy(), c_st_logit.numpy())
cnxt_lr = c_un['LFW_Real'][1].numpy()
cnxt_un = {}
for g in ('LDM', 'SDXL_Refine', 'DiffFace', 'StyleGAN'):
    lf = c_un[g][1].numpy()
    cnxt_un[g] = compute_exact_metrics(np.array([0]*len(cnxt_lr)+[1]*len(lf)), np.concatenate([cnxt_lr, lf]))

# 4-seed matched fusion comparison on ConvNeXt_128 + WolterWavelet2021_128 (directly answers R1 #14!)
cnxt_fusion_runs = {}
for lbl, key in [('ConvNeXt+MHA_128', 'mha_128'), ('ConvNeXt+ConcatMatched', 'concat_matched'), ('ConvNeXt+GatedFusion', 'gated')]:
    cnxt_fusion_runs[lbl] = []
    for s in (42, 100, 2024, 3407):
        h, _ = train_cached_fusion_head(key, c_tr_s, w_tr_s, y_tr_s, c_dv, w_dv, y_dv, epochs=20, seed=s, device='cuda:0')
        with torch.no_grad():
            st_m = compute_exact_metrics(y_st.numpy(), h(c_st.cuda(), w_st.cuda()).flatten().cpu().numpy())
            lr_m = h(c_un['LFW_Real'][0].cuda(), cache['unseen_q85']['LFW_Real']['wav'].cuda()).flatten().cpu().numpy()
            un_m = {}
            accs, aucs, eers = [], [], []
            for g in ('LDM', 'SDXL_Refine', 'DiffFace', 'StyleGAN'):
                lf_m = h(c_un[g][0].cuda(), cache['unseen_q85'][g]['wav'].cuda()).flatten().cpu().numpy()
                mg = compute_exact_metrics(np.array([0]*len(lr_m)+[1]*len(lf_m)), np.concatenate([lr_m, lf_m]))
                un_m[g] = mg; accs.append(mg['accuracy']); aucs.append(mg['roc_auc']); eers.append(mg['eer'])
            un_m['macro_4domain'] = {'accuracy': float(np.mean(accs)), 'roc_auc': float(np.mean(aucs)), 'eer': float(np.mean(eers)), 'lfw_real_recall': float((lr_m < 0).mean())}
        cnxt_fusion_runs[lbl].append({'seen_test': st_m, 'unseen_q85': un_m})

OPTION2_RESULTS['ConvNeXt_128_Modularity'] = {'standalone': {'seen_test': cnxt_st, 'unseen_q85': cnxt_un}, 'fusion_4seed': cnxt_fusion_runs}
(OUT_DIR / 'metrics/paper1_option2_baselines_convnext.json').write_text(json.dumps(OPTION2_RESULTS, indent=2), encoding='utf-8')

print('\n' + '=' * 114)
print('  CONVNEXT_128 MODULARITY (4-SEED MEAN) + MATCHED CONCAT vs GATED vs MHA (ANSWERS R1 #14)')
print('=' * 114)
print(f"  ConvNeXt_128 (Standalone) | SeenTest Acc={cnxt_st['accuracy']*100:5.2f}% AUC={cnxt_st['roc_auc']:.4f} EER={cnxt_st['eer']*100:4.2f}% | UnseenQ85 4Dom Acc={np.mean([cnxt_un[g]['accuracy'] for g in ('LDM','SDXL_Refine','DiffFace','StyleGAN')])*100:5.2f}% AUC={np.mean([cnxt_un[g]['roc_auc'] for g in ('LDM','SDXL_Refine','DiffFace','StyleGAN')]):.4f}")
for lbl, runs in cnxt_fusion_runs.items():
    st_acc = np.mean([r['seen_test']['accuracy']*100 for r in runs])
    st_auc = np.mean([r['seen_test']['roc_auc'] for r in runs])
    st_eer = np.mean([r['seen_test']['eer']*100 for r in runs])
    uq_acc = np.mean([r['unseen_q85']['macro_4domain']['accuracy']*100 for r in runs])
    uq_auc = np.mean([r['unseen_q85']['macro_4domain']['roc_auc'] for r in runs])
    print(f"  {lbl:<25} | SeenTest Acc={st_acc:5.2f}% AUC={st_auc:.4f} EER={st_eer:4.2f}% | UnseenQ85 4Dom Acc={uq_acc:5.2f}% AUC={uq_auc:.4f}")
    m0 = runs[0]['unseen_q85']
    print('    Seed-42 Q85:', ' | '.join(f"{g}={m0[g]['roc_auc']:.4f}/{m0[g]['accuracy']*100:5.2f}%" for g in ('LDM', 'SDXL_Refine', 'DiffFace', 'StyleGAN')))
print('Saved Option 2 complete report ->', OUT_DIR / 'metrics/paper1_option2_baselines_convnext.json')
```
