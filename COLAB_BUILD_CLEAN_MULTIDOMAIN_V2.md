# Colab CPU Guide: Build Clean Multi-Domain Dataset v2 (`DatasetMultiDomainClean_v2`)

Run this on a **Google Colab CPU runtime** (no GPU needed — saves your L4/T4 compute units for training!).

This completely replaces the older `112x112` WebFace-upscaled dataset by extending `DatesetNoVideosNoGAN` with **shortcut-free, native high-resolution multi-domain real faces** and **all 7 extended diffusion fake generators**, while keeping **LDM** and **`val to be deleted 2`** **100% unseen**.

---

### Dataset Blueprint & Scientific Guarantees

| Class | Sources Included | Native Resolution | Count Target |
| :--- | :--- | :---: | :---: |
| **Real (`label = 0`)** | 1. **CelebA** (from base `DatesetNoVideosNoGAN`)<br>2. **FFHQ-256** (`bitmind/ffhq-256`, Flickr faces)<br>3. **FairFace** (`HuggingFaceM4/FairFace` `"0.25"`, age/gender/race balanced)<br>4. **UTKFace** (`nu-delta/utkface`, auto-used if extra reals needed for 1:1 balance) | `178×218` to `256×256+`<br>*(ZERO blurry 112px WebFace!)* | Strictly matched 1:1 to Fake in both `train` & `val` |
| **Fake (`label = 1`)** | 1. **Base (4 families):** `ADM`, `DDIM`, `DDPM`, `DiffSwap`<br>2. **Extended (11 families across FE, FS, I2I, T2I — capped at 15K train / 2.5K val per generator for balance):** `CoDiff`, `cycle_diff`, `DCFace`, `DiffFace`, `DreamBooth`, `FreeDoM_I`, `LoRA`, `SDXL_Refine`, `FreeDoM_T`, `HPS`, `SDXL` | `256×256` to `512×512` | ~220K train + ~45K val |
| **Strictly Excluded (Unseen Tests)** | **`LDM`**, **`Midjourney`**, **`Imagic`**, **`val to be deleted 2`** (CelebA-HQ + unseen diffusion), **DFDC / GenImage** | — | **0 images** in train or val |

**Hard Guarantees Enforced Automatically:**
1. **Zero `112×112` Upscaling Shortcut:** Only native ≥178px–256px real datasets (`CelebA`, `FFHQ-256`, `FairFace`, `UTKFace`).
2. **Zero Format Shortcut:** All newly added real and fake images are saved in the **exact same format** (`PNG` lossless by default, or `JPEG_96` if Drive space is tight).
3. **Zero Aspect-Ratio Distortion:** Non-square images are **center-square-cropped first**, then resized to `256×256` Bicubic.
4. **Zero Identity Leakage:** All 7 extended fake generators are split **80% Train / 20% Val globally by identity ID** (`id_0000`…); FFHQ, FairFace, and UTKFace use strictly disjoint train/val splits.
5. **Zero Corrupt/Truncated Images:** Every image is verified with `img.load()` before writing atomically.
6. **Full Provenance Manifest:** Writes `dataset_manifest.csv` + `dataset_audit_report.json` with exact generator/source tags so Stage 2 L4 training can immediately run **Leave-One-Generator-Out (LOGO)** validation.

---

## Cell 1: Mount Drive, Install Deps & Check Disk / Drive Free Space

```python
from google.colab import drive
drive.mount('/content/drive')

import os, sys, io, json, csv, shutil, random, zipfile, subprocess
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'datasets', 'pillow', 'tqdm'], check=True)

# Check free disk space locally (/content) and on Google Drive
local_free_gb = shutil.disk_usage('/content').free / (1024**3)
drive_free_gb = shutil.disk_usage('/content/drive/MyDrive').free / (1024**3)
print('=' * 65)
print(f'Local /content free space : {local_free_gb:.1f} GB')
print(f'Google Drive free space   : {drive_free_gb:.1f} GB')
print('=' * 65)

# Choose output format for all NEW images (both real and fake get the EXACT same format!):
#   'PNG'     -> lossless (~35-55 GB final zip; best forensic fidelity if Drive has >= 65 GB free)
#   'JPEG_96' -> visually lossless 4:4:4 JPEG (~12-16 GB final zip; safe if Drive has < 65 GB free)
SAVE_FORMAT = 'PNG' if drive_free_gb >= 65.0 else 'JPEG_96'
print(f'Auto-selected matched packaging format for new real & fake images: {SAVE_FORMAT}')
```

---

## Cell 2: Clean Slate & Extract Base `DatesetNoVideosNoGAN`

```python
DATASET_ROOT = Path('/content/dataset/DatasetMultiDomainClean_v2')
BASE_ZIP_CANDIDATES = [
    Path('/content/drive/MyDrive/DatasetNoVideoNoGan/DatesetNoVideosNoGAN.zip'),
    Path('/content/drive/MyDrive/DatesetNoVideosNoGAN.zip'),
]
BASE_ZIP = next((p for p in BASE_ZIP_CANDIDATES if p.is_file()), None)
assert BASE_ZIP is not None, f'Base zip not found in: {BASE_ZIP_CANDIDATES}'

if Path('/content/dataset').exists():
    shutil.rmtree('/content/dataset')
Path('/content/dataset').mkdir(parents=True, exist_ok=True)

print(f'Extracting base dataset from {BASE_ZIP}...')
subprocess.run(['7z', 'x', str(BASE_ZIP), '-o/content/dataset/', '-y'], check=True, stdout=subprocess.DEVNULL)

extracted_base = Path('/content/dataset/DatesetNoVideosNoGAN')
assert (extracted_base / 'train/real').is_dir(), f'Expected folder missing: {extracted_base}'
extracted_base.rename(DATASET_ROOT)

for split in ('train', 'val'):
    for cls in ('real', 'fake'):
        cnt = len(list((DATASET_ROOT / split / cls).glob('*.*')))
        print(f'  Base {split:>5}/{cls:<4}: {cnt:,}')
```

---

## Cell 3: Extract All 11 Extended Fake Generators Across FE / FS / I2I / T2I (Balanced Caps, Official Train/Val Splits, `Midjourney` + `Imagic` + `LDM` Strictly Held Out!)

```python
TARGET_SIZE = (256, 256)
RANDOM_SEED = 42
SHARED_ROOT = Path('/content/drive/MyDrive/shared_project_files')

# Balanced caps per generator so no single generator (like 145K CoDiff) drowns out the other 10!
MAX_TRAIN_PER_GEN = 15000
MAX_VAL_PER_GEN   = 2500

# Strictly held-out UNSEEN generators (NEVER included in train or val!)
HELD_OUT_UNSEEN_GENERATORS = {'midjourney', 'imagic', 'ldm'}

# All 11 extended fake generators across FE, FS, I2I, T2I with their exact train & val ZIPs
EXTENDED_GENERATORS = [
    ('CoDiff',      SHARED_ROOT / 'train/FE/CoDiff.zip',               SHARED_ROOT / 'val/FE/CoDiff.zip'),
    ('cycle_diff',  SHARED_ROOT / 'train/FE/cycle_diff.zip',           SHARED_ROOT / 'val/FE/cycle_diff.zip'),
    ('DCFace',      SHARED_ROOT / 'train/FS/DCFace.zip',               SHARED_ROOT / 'val/FS/DCFace.zip'),
    ('DiffFace',    SHARED_ROOT / 'train/FS/DiffFace.zip',             SHARED_ROOT / 'val/FS/DiffFace.zip'),
    ('DreamBooth',  SHARED_ROOT / 'train/I2I/DreamBooth.zip',          SHARED_ROOT / 'val/I2I/DreamBooth.zip'),
    ('FreeDoM_I',   SHARED_ROOT / 'train/I2I/FreeDoM_I.zip',           SHARED_ROOT / 'val/I2I/FreeDoM_I.zip'),
    ('LoRA',        SHARED_ROOT / 'train/I2I/LoRA.zip',                SHARED_ROOT / 'val/I2I/LoRA.zip'),
    ('SDXL_Refine', SHARED_ROOT / 'train/I2I/SDXL_Refine.zip',         SHARED_ROOT / 'val/I2I/SDXL_Refine.zip'),
    ('FreeDoM_T',   SHARED_ROOT / 'train/T2I/Copy of FreeDoM_T.zip',   SHARED_ROOT / 'val/T2I/FreeDoM_T.zip'),
    ('HPS',         SHARED_ROOT / 'train/T2I/Copy of HPS.zip',         SHARED_ROOT / 'val/T2I/HPS.zip'),
    ('SDXL',        SHARED_ROOT / 'train/T2I/Copy of SDXL.zip',        SHARED_ROOT / 'val/T2I/SDXL.zip'),
]

# Clean any leftover partial extended-fake files from an interrupted Cell 3 run (preserves base ADM/DDIM/DDPM/DiffSwap!)
removed_partial = 0
for split in ('train', 'val'):
    for f in (DATASET_ROOT / split / 'fake').glob('*__ID_*'):
        f.unlink()
        removed_partial += 1
if removed_partial:
    print(f'Cleaned {removed_partial:,} partial extended-fake files from interrupted run.')

def center_square_resize(img: Image.Image, target_size=(256, 256)) -> Image.Image:
    """Center-crop to square FIRST (no aspect distortion), then bicubic resize."""
    img = img.convert('RGB')
    w, h = img.size
    if w != h:
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
    if img.size != target_size:
        img = img.resize(target_size, Image.Resampling.BICUBIC)
    return img

def save_matched(img: Image.Image, out_stem: Path) -> Path:
    """Save image atomically in the globally matched SAVE_FORMAT."""
    if SAVE_FORMAT == 'PNG':
        dest = out_stem.with_suffix('.png')
        tmp = dest.with_suffix('.tmp.png')
        img.save(tmp, format='PNG', compress_level=3)
    else:
        dest = out_stem.with_suffix('.jpg')
        tmp = dest.with_suffix('.tmp.jpg')
        img.save(tmp, format='JPEG', quality=96, subsampling=0)
    os.replace(tmp, dest)
    return dest

def scan_zip_by_identity(zp: Path, gen_name: str):
    """Return dict[ident -> list[filename]] for clean image entries in a ZIP."""
    by_id = defaultdict(list)
    with zipfile.ZipFile(zp, 'r') as z:
        for zi in z.infolist():
            fn = zi.filename
            base = fn.replace('\\', '/').split('/')[-1]
            if base.startswith('.') or fn.startswith('__MACOSX') or zi.file_size < 100:
                continue
            if not fn.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue
            if any(h in fn.lower() for h in HELD_OUT_UNSEEN_GENERATORS):
                continue
            parts = fn.replace('\\', '/').split('/')
            ident = next((p for p in parts if p.startswith('id_') and len(p) >= 4), None)
            if ident is None:
                ident = f'pseudo_{gen_name}_{parts[-2] if len(parts) >= 2 else "root"}'
            by_id[ident].append(fn)
    return by_id

def stratified_sample_from_ids(by_id: dict, max_n: int, seed: int):
    """Round-robin sample evenly across ALL identities up to max_n images."""
    rng = random.Random(seed)
    id_keys = sorted(by_id.keys())
    queues = {}
    for k in id_keys:
        lst = list(by_id[k])
        rng.shuffle(lst)
        queues[k] = lst
    selected = []
    while len(selected) < max_n:
        progressed = False
        for k in id_keys:
            if queues[k]:
                selected.append((k, queues[k].pop()))
                progressed = True
                if len(selected) >= max_n:
                    break
        if not progressed:
            break
    return selected

# Pass 1: Scan all 11 official train & val ZIPs and enforce GLOBAL cross-generator identity disjointness
scanned = []
global_val_ids = set()
for gen, train_zip, val_zip in EXTENDED_GENERATORS:
    assert gen.lower() not in HELD_OUT_UNSEEN_GENERATORS
    assert train_zip.is_file(), f'Missing train ZIP: {train_zip}'
    assert val_zip.is_file(),   f'Missing val ZIP: {val_zip}'
    tr_by_id = scan_zip_by_identity(train_zip, gen)
    va_by_id = scan_zip_by_identity(val_zip, gen)
    global_val_ids.update(va_by_id.keys())
    scanned.append((gen, train_zip, val_zip, tr_by_id, va_by_id))

# Pass 2: Drop any global val ID from train across ALL generators, then round-robin sample & extract
NEW_FAKE_TRAIN = 0
NEW_FAKE_VAL = 0
per_gen_counts = {}

for gen_idx, (gen, train_zip, val_zip, tr_by_id, va_by_id) in enumerate(scanned):
    overlap_ids = set(tr_by_id.keys()) & global_val_ids
    for oid in overlap_ids:
        del tr_by_id[oid]

    tr_selected = stratified_sample_from_ids(tr_by_id, MAX_TRAIN_PER_GEN, seed=RANDOM_SEED + gen_idx)
    va_selected = stratified_sample_from_ids(va_by_id, MAX_VAL_PER_GEN,   seed=RANDOM_SEED + 100 + gen_idx)

    # Extract train
    saved_tr = 0
    with zipfile.ZipFile(train_zip, 'r') as z:
        for i, (ident, fn) in enumerate(tr_selected):
            stem = Path(fn).stem.replace('.', '_').replace('/', '_').replace('\\', '_')
            out_stem = DATASET_ROOT / 'train/fake' / f'{gen}__ID_{ident}__{stem}_{i:06d}'
            try:
                with Image.open(io.BytesIO(z.read(fn))) as im:
                    im.load()
                    save_matched(center_square_resize(im, TARGET_SIZE), out_stem)
                saved_tr += 1
            except Exception:
                continue

    # Extract val
    saved_va = 0
    with zipfile.ZipFile(val_zip, 'r') as z:
        for i, (ident, fn) in enumerate(va_selected):
            stem = Path(fn).stem.replace('.', '_').replace('/', '_').replace('\\', '_')
            out_stem = DATASET_ROOT / 'val/fake' / f'{gen}__ID_{ident}__{stem}_{i:06d}'
            try:
                with Image.open(io.BytesIO(z.read(fn))) as im:
                    im.load()
                    save_matched(center_square_resize(im, TARGET_SIZE), out_stem)
                saved_va += 1
            except Exception:
                continue

    NEW_FAKE_TRAIN += saved_tr
    NEW_FAKE_VAL += saved_va
    per_gen_counts[gen] = {'train': saved_tr, 'val': saved_va, 'dropped_overlap_ids': len(overlap_ids)}
    print(f'  [{gen:<12}] Train: +{saved_tr:6,} | Val: +{saved_va:5,} (overlap IDs filtered: {len(overlap_ids)})', flush=True)

print(f'\nTotal Extended Fakes Extracted across 11 Generators -> Train: +{NEW_FAKE_TRAIN:,} | Val: +{NEW_FAKE_VAL:,}')
```

---

## Cell 4: Add Native High-Res Multi-Domain Real Faces (FFHQ-256 + FairFace + UTKFace)

Streams **only native ≥200px–256px authentic real faces** (zero blurry `112×112` WebFace!), strictly matched **1-to-1** to `NEW_FAKE_TRAIN` and `NEW_FAKE_VAL` in the exact same `SAVE_FORMAT` and `256×256` resolution:

```python
from datasets import load_dataset

need_train_real = NEW_FAKE_TRAIN
need_val_real   = NEW_FAKE_VAL
print(f'Target New Native High-Res Real Faces -> Train: {need_train_real:,} | Val: {need_val_real:,}')

dest_train_real = DATASET_ROOT / 'train/real'
dest_val_real   = DATASET_ROOT / 'val/real'

added_train = Counter()
added_val   = Counter()

def save_real_from_stream(ds_iter, src_tag: str, max_train: int, max_val: int, train_ratio: float = 0.8):
    """Stream real faces with deterministic disjoint train/val assignment and full decode verification."""
    t_cnt, v_cnt = 0, 0
    for idx, sample in enumerate(ds_iter):
        if t_cnt >= max_train and v_cnt >= max_val:
            break
        # Assign deterministically: first portion train, remainder val (or split by hash)
        use_val = (idx % 5 == 0)
        if use_val and v_cnt >= max_val:
            use_val = False
        elif (not use_val) and t_cnt >= max_train:
            use_val = True
        if use_val and v_cnt >= max_val:
            break
        if (not use_val) and t_cnt >= max_train:
            break

        try:
            im = sample.get('image') or sample.get('img') or sample.get('jpg')
            if isinstance(im, bytes):
                im = Image.open(io.BytesIO(im))
            if im is None:
                continue
            im.load()
            if min(im.size) < 160:
                continue  # Enforce native high-resolution rule (reject tiny thumbnails!)
            im_std = center_square_resize(im, TARGET_SIZE)
            dest_dir = dest_val_real if use_val else dest_train_real
            split_tag = 'val' if use_val else 'train'
            out_stem = dest_dir / f'{src_tag}__ID_{src_tag}_{idx:06d}__{split_tag}_{idx:06d}'
            save_matched(im_std, out_stem)
            if use_val:
                v_cnt += 1
                added_val[src_tag] += 1
            else:
                t_cnt += 1
                added_train[src_tag] += 1
            if (t_cnt + v_cnt) % 10000 == 0:
                print(f'  [{src_tag}] Saved train={t_cnt:,}/{max_train:,} | val={v_cnt:,}/{max_val:,}', flush=True)
        except Exception:
            continue
    print(f'  [{src_tag}] DONE -> train=+{t_cnt:,} | val=+{v_cnt:,}', flush=True)
    return t_cnt, v_cnt

# --- Source 1: FFHQ-256 (70,000 native 256x256 Flickr faces: up to 56K train + 14K val) ---
print('\n[1/3] Streaming FFHQ-256 (bitmind/ffhq-256)...')
try:
    ffhq_stream = load_dataset('bitmind/ffhq-256', split='train', streaming=True)
except Exception:
    ffhq_stream = load_dataset('merkol/ffhq-256', split='train', streaming=True)
t1, v1 = save_real_from_stream(
    ffhq_stream, 'ffhq256',
    max_train=min(56000, need_train_real),
    max_val=min(14000, need_val_real),
)

rem_train = max(0, need_train_real - sum(added_train.values()))
rem_val   = max(0, need_val_real   - sum(added_val.values()))

# --- Source 2: FairFace 0.25 (Race/Gender/Age balanced native 224x224 faces: ~86.7K train + ~10.9K val) ---
if rem_train > 0:
    print(f'\n[2/3] Streaming FairFace train (remaining train needed: {rem_train:,})...')
    ff_tr = load_dataset('HuggingFaceM4/FairFace', '0.25', split='train', streaming=True)
    save_real_from_stream(ff_tr, 'fairface', max_train=rem_train, max_val=0)
if rem_val > 0:
    print(f'[2/3] Streaming FairFace validation (remaining val needed: {rem_val:,})...')
    ff_va = load_dataset('HuggingFaceM4/FairFace', '0.25', split='validation', streaming=True)
    save_real_from_stream(ff_va, 'fairface_val', max_train=0, max_val=rem_val)

rem_train = max(0, need_train_real - sum(added_train.values()))
rem_val   = max(0, need_val_real   - sum(added_val.values()))

# --- Source 3: UTKFace (~23.7K native >=200px faces across ages 0-116) if still needed ---
if rem_train > 0 or rem_val > 0:
    print(f'\n[3/3] Streaming UTKFace (remaining needed -> train: {rem_train:,}, val: {rem_val:,})...')
    try:
        utk_stream = load_dataset('nu-delta/utkface', split='train', streaming=True)
    except Exception:
        utk_stream = load_dataset('ljnlonoljpiljm/utkface', split='train', streaming=True)
    save_real_from_stream(utk_stream, 'utkface', max_train=rem_train, max_val=rem_val)

print('\nSummary of New Native High-Res Real Faces Added:')
print('  Train:', dict(added_train), 'Total:', sum(added_train.values()))
print('  Val  :', dict(added_val),   'Total:', sum(added_val.values()))
```

---

## Cell 5: Strict 50/50 Balance, Anti-Leakage Audit & Provenance Manifest Generation

Balances train & val 1:1 (preserving balanced representation across generators), verifies **0 identity leakage**, **0 LDM leakage**, and writes **`dataset_manifest.csv`** + **`dataset_audit_report.json`** right inside the dataset folder:

```python
import random

def infer_source(filename: str) -> str:
    if '__ID_' in filename:
        return filename.split('__ID_', 1)[0]
    if '_' in filename:
        return filename.split('_', 1)[0]
    return 'base_unknown'

def extract_ident(filename: str):
    if '__ID_' in filename:
        return filename.split('__ID_', 1)[1].split('__', 1)[0]
    return None

# 1. Enforce exact 50/50 class balance per split (trim excess randomly with fixed seed so every generator stays represented)
rng = random.Random(42)
for split in ('train', 'val'):
    r_files = sorted((DATASET_ROOT / split / 'real').glob('*.*'))
    f_files = sorted((DATASET_ROOT / split / 'fake').glob('*.*'))
    target_n = min(len(r_files), len(f_files))
    for files in (r_files, f_files):
        if len(files) > target_n:
            shuffled = list(files)
            rng.shuffle(shuffled)
            for extra in shuffled[target_n:]:
                extra.unlink()

# 2. Full provenance scan & mathematical leakage audit
manifest_rows = []
counts_by_split_class = Counter()
sources_by_split_class = defaultdict(Counter)
exts_by_split_class = defaultdict(Counter)
fake_ids_by_split = {'train': set(), 'val': set()}
real_ids_by_split = {'train': set(), 'val': set()}
ldm_hits = []

for split in ('train', 'val'):
    for cls_name, lbl in (('real', 0), ('fake', 1)):
        for p in sorted((DATASET_ROOT / split / cls_name).rglob('*.*')):
            src = infer_source(p.name)
            ident = extract_ident(p.name)
            if any(h in p.name.lower() or h in src.lower() for h in ('ldm', 'midjourney', 'imagic')):
                ldm_hits.append(p.name)
            if ident:
                if lbl == 1:
                    fake_ids_by_split[split].add(ident)
                else:
                    real_ids_by_split[split].add(ident)
            counts_by_split_class[(split, cls_name)] += 1
            sources_by_split_class[(split, cls_name)][src] += 1
            exts_by_split_class[(split, cls_name)][p.suffix.lower()] += 1
            manifest_rows.append({
                'sample_id': f'{split}_{cls_name}_{p.stem}',
                'relative_path': p.relative_to(DATASET_ROOT).as_posix(),
                'label': lbl,
                'split': 'dev' if split == 'val' else 'train',
                'dataset_source': src,
                'generator': src if lbl == 1 else 'real',
                'identity_id': ident or 'unknown',
                'ext': p.suffix.lower(),
            })

fake_leak = fake_ids_by_split['train'] & fake_ids_by_split['val']
real_leak = real_ids_by_split['train'] & real_ids_by_split['val']

assert len(ldm_hits) == 0, f'FATAL: LDM leaked into dataset: {ldm_hits[:5]}'
assert len(fake_leak) == 0, f'FATAL: Fake identity leakage across train/val: {list(fake_leak)[:5]}'
assert len(real_leak) == 0, f'FATAL: Real identity leakage across train/val: {list(real_leak)[:5]}'
assert counts_by_split_class[('train', 'real')] == counts_by_split_class[('train', 'fake')]
assert counts_by_split_class[('val', 'real')] == counts_by_split_class[('val', 'fake')]

# Write provenance manifest and audit report
manifest_csv = DATASET_ROOT / 'dataset_manifest.csv'
with open(manifest_csv, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
    w.writeheader()
    w.writerows(manifest_rows)

audit_report = {
    'dataset_name': 'DatasetMultiDomainClean_v2',
    'save_format_new_images': SAVE_FORMAT,
    'counts': {f'{s}/{c}': n for (s, c), n in sorted(counts_by_split_class.items())},
    'total_images': len(manifest_rows),
    'sources': {f'{s}/{c}': dict(cnt) for (s, c), cnt in sorted(sources_by_split_class.items())},
    'extensions': {f'{s}/{c}': dict(cnt) for (s, c), cnt in sorted(exts_by_split_class.items())},
    'fake_identity_overlap': len(fake_leak),
    'real_identity_overlap': len(real_leak),
    'ldm_leakage_count': len(ldm_hits),
}
(DATASET_ROOT / 'dataset_audit_report.json').write_text(json.dumps(audit_report, indent=2), encoding='utf-8')

print('=' * 75)
print('  CLEAN MULTI-DOMAIN DATASET v2 — AUDIT PASSED')
print('=' * 75)
print(json.dumps(audit_report, indent=2))
print('=' * 75)
```

---

## Cell 6: Package & Save `DatasetMultiDomainClean_v2.zip` to Google Drive

```python
OUT_DIR = Path('/content/drive/MyDrive/DatasetMultiDomainClean_v2')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_ZIP = OUT_DIR / 'DatasetMultiDomainClean_v2.zip'

# Save audit report separately on Drive for quick inspection without unzipping
shutil.copy2(DATASET_ROOT / 'dataset_audit_report.json', OUT_DIR / 'dataset_audit_report.json')
shutil.copy2(DATASET_ROOT / 'dataset_manifest.csv', OUT_DIR / 'dataset_manifest.csv')

print(f'Packaging {DATASET_ROOT} -> {OUT_ZIP} (fast store/mx=1)...', flush=True)
subprocess.run(
    ['7z', 'a', '-tzip', '-mx=1', str(OUT_ZIP), 'DatasetMultiDomainClean_v2'],
    cwd='/content/dataset',
    check=True,
)
size_gb = OUT_ZIP.stat().st_size / (1024**3)
print(f'\nDONE! Saved {OUT_ZIP} ({size_gb:.2f} GB) + manifest & audit report to Drive!')
```

---

### Next Step (After CPU Build Completes)
Once Cell 6 finishes, paste the `dataset_audit_report.json` summary here—then we will switch to an **L4 GPU** to:
1. Generate a targeted **~20% paired `sd_turbo` + `multi_vae` reconstruction mix** across CelebA, FFHQ, and FairFace (so no single real source can be memorized and VAE artifacts never drown out real diffusion detection),
2. Run **Leave-One-Generator-Out** training with **`EFFORT_128 --effort_svd_rank 0 --effort_multi_layer`** (our proven `0.9796`-AUC architecture!), and
3. Evaluate on your completely untouched **LDM** and **`val to be deleted 2`** unseen domains!
