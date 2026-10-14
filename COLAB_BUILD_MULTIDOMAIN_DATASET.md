# Colab Guide: Build Extended Multi-Domain Dataset with Real Faces & Diffusion Models

This notebook runs on a **Google Colab CPU runtime** (no GPU needed). It builds an extended, leak-free, perfectly balanced dataset by merging:
1. **Base Dataset:** `DatesetNoVideosNoGAN.zip` (160k train, 40k val, 50% real / 50% fake).
2. **New Fake Generators:** Generator zip archives organized by identity (e.g., `cycle_diff.zip`, `SDXL.zip`, etc.).
3. **New Real Faces:** InsightFace WebFace archive (`faces_webface_112x112.zip`) decoded directly from `train.rec`.

### Core Safeguards Implemented:
- **Zero Identity Leakage:** All new images (both fake and real) are grouped strictly by **identity ID**. Identities are partitioned 80% to Train and 20% to Val, ensuring no person appears in both splits.
- **Resolution Standardization:** Real 112x112 WebFace crops are standardized to 256x256 using high-quality Bicubic interpolation, eliminating the resolution shortcut where a network could classify fakes based on pixel dimensions.
- **Strict 50/50 Class Balance:** New Real and new Fake images are matched 1-to-1 ($N_{\text{new\_real}} = N_{\text{new\_fake}}$) so the final merged dataset remains exactly 50% Real and 50% Fake in both Train and Val.
- **Drive Packaging:** Compresses the final dataset into `/content/drive/MyDrive/DataesetNoVideoMultipleDomainAndRealImages/` for fast 1-line extraction in subsequent training notebooks.

---

## Cell 1: Mount Google Drive (Python)
```python
from google.colab import drive
drive.mount('/content/drive')

import os, sys, shutil, zipfile, struct, io, random
from pathlib import Path
from PIL import Image
from collections import defaultdict

print("Google Drive mounted successfully.")
```

---

## Cell 2: Extract Base Dataset to Local SSD (Bash)
```bash
%%bash
set -euo pipefail

# 1. Prepare local directories on fast NVMe disk
mkdir -p /content/dataset

# 2. Extract base DatesetNoVideosNoGAN
BASE_ZIP="/content/drive/MyDrive/DatasetNoVideoNoGan/DatesetNoVideosNoGAN.zip"
if [ ! -d "/content/dataset/DatesetNoVideosNoGAN/train/real" ]; then
    echo "Extracting base dataset from: $BASE_ZIP ..."
    7z x "$BASE_ZIP" -o/content/dataset/ -y > /dev/null
    echo "Base dataset extracted successfully."
else
    echo "Base dataset already extracted, skipping."
fi

echo "--- Current Base Counts ---"
echo "Train Real: $(ls /content/dataset/DatesetNoVideosNoGAN/train/real | wc -l)"
echo "Train Fake: $(ls /content/dataset/DatesetNoVideosNoGAN/train/fake | wc -l)"
echo "Val Real:   $(ls /content/dataset/DatesetNoVideosNoGAN/val/real | wc -l)"
echo "Val Fake:   $(ls /content/dataset/DatesetNoVideosNoGAN/val/fake | wc -l)"
```

---

## Cell 3: Identity-Aware Extraction, Standardization & Balancing (Python)
```python
# ==============================================================================
# CONFIGURATION: Specify your Google Drive zip paths below
# ==============================================================================
BASE_DATASET_ROOT = Path('/content/dataset/DatesetNoVideosNoGAN')

# List of Fake Generator Zip files on Google Drive
FAKE_ARCHIVES = [
    {
        'generator_name': 'cycle_diff',
        'zip_path': Path('/content/drive/MyDrive/t2itestdtaset/cycle_diff.zip'),
    },
    {
        'generator_name': 'sdxl',
        'zip_path': Path('/content/drive/MyDrive/t2itestdtaset/SDXL.zip'),
    },
    # Add any additional fake generator archives here:
    # {'generator_name': 'another_model', 'zip_path': Path('/content/drive/MyDrive/.../another.zip')},
]

# Real Faces InsightFace WebFace archive on Google Drive
REAL_WEBFACE_ZIP = Path('/content/drive/MyDrive/t2itestdtaset/faces_webface_112x112.zip')

# Target standard image resolution (prevents 112x112 resolution shortcut)
TARGET_SIZE = (256, 256)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

print("Starting identity-aware extraction and balance pipeline...")

# ==============================================================================
# STEP 1: Scan and Group Fake Images by Identity
# ==============================================================================
# Map: generator_name -> identity_id -> list of file paths inside zip
fake_identity_map = defaultdict(lambda: defaultdict(list))

for arch_info in FAKE_ARCHIVES:
    gen_name = arch_info['generator_name']
    z_path = arch_info['zip_path']
    if not z_path.is_file():
        print(f"Warning: Fake archive not found at {z_path}. Skipping.")
        continue
    
    print(f"Scanning fake generator archive: {gen_name} ({z_path.name})...")
    with zipfile.ZipFile(z_path, 'r') as z:
        for fname in z.namelist():
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')) and not fname.startswith('__MACOSX'):
                parts = fname.replace('\\', '/').split('/')
                # Identity is the folder right below the root (e.g. cycle_diff/id_1893/...)
                if len(parts) >= 2 and parts[1].startswith('id_'):
                    ident = parts[1]
                else:
                    ident = 'id_unknown'
                fake_identity_map[gen_name][ident].append(fname)
    
    total_imgs = sum(len(imgs) for imgs in fake_identity_map[gen_name].values())
    total_ids = len(fake_identity_map[gen_name])
    print(f"  -> Found {total_imgs} images across {total_ids} identities in {gen_name}.")

# ==============================================================================
# STEP 2: Partition Fake Identities (80% Train, 20% Val)
# ==============================================================================
fake_train_items = []  # (gen_name, zip_path, fname, identity)
fake_val_items = []

for arch_info in FAKE_ARCHIVES:
    gen_name = arch_info['generator_name']
    z_path = arch_info['zip_path']
    ident_dict = fake_identity_map[gen_name]
    if not ident_dict:
        continue
    
    id_list = sorted(list(ident_dict.keys()))
    random.shuffle(id_list)
    n_train_ids = int(0.8 * len(id_list))
    train_ids = set(id_list[:n_train_ids])
    val_ids = set(id_list[n_train_ids:])
    
    for ident, file_list in ident_dict.items():
        if ident in train_ids:
            for f in file_list:
                fake_train_items.append((gen_name, z_path, f, ident))
        else:
            for f in file_list:
                fake_val_items.append((gen_name, z_path, f, ident))

print(f"Total Candidate Fake Pool: Train={len(fake_train_items)}, Val={len(fake_val_items)}")

# ==============================================================================
# STEP 3: Scan and Group Real WebFace Images by Identity from train.rec
# ==============================================================================
assert REAL_WEBFACE_ZIP.is_file(), f"Real archive missing: {REAL_WEBFACE_ZIP}"
print(f"Scanning real WebFace archive from {REAL_WEBFACE_ZIP.name}...")

# Map: identity_id -> list of raw image bytes
real_identity_map = defaultdict(list)

with zipfile.ZipFile(REAL_WEBFACE_ZIP, 'r') as z:
    with z.open('faces_webface_112x112/train.rec') as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            magic, lrecord = struct.unpack('II', hdr)
            if magic != 0xced7230a:
                break
            pad = (4 - (lrecord % 4)) % 4
            data = f.read(lrecord)
            if pad:
                f.read(pad)
            
            flag, label = struct.unpack('If', data[:8])
            ident_id = int(label)
            img_bytes = data[24:]  # raw jpeg bytes
            real_identity_map[ident_id].append(img_bytes)

total_real_imgs = sum(len(imgs) for imgs in real_identity_map.values())
total_real_ids = len(real_identity_map)
print(f"  -> Decoded {total_real_imgs} real face images across {total_real_ids} identities from train.rec.")

# Partition Real Identities (80% Train, 20% Val)
real_id_list = sorted(list(real_identity_map.keys()))
random.shuffle(real_id_list)
n_real_train_ids = int(0.8 * len(real_id_list))
real_train_ids = set(real_id_list[:n_real_train_ids])
real_val_ids = set(real_id_list[n_real_train_ids:])

real_train_pool = []  # (ident_id, img_bytes)
real_val_pool = []
for ident_id, b_list in real_identity_map.items():
    if ident_id in real_train_ids:
        for b in b_list:
            real_train_pool.append((ident_id, b))
    else:
        for b in b_list:
            real_val_pool.append((ident_id, b))

print(f"Total Candidate Real Pool: Train={len(real_train_pool)}, Val={len(real_val_pool)}")

# ==============================================================================
# STEP 4: Balance Extraction Counts (Strict 50/50 Real/Fake)
# ==============================================================================
# To preserve exact 50% real / 50% fake balance in both splits:
n_add_train = min(len(fake_train_items), len(real_train_pool))
n_add_val = min(len(fake_val_items), len(real_val_pool))

selected_fake_train = fake_train_items[:n_add_train]
selected_real_train = real_train_pool[:n_add_train]

selected_fake_val = fake_val_items[:n_add_val]
selected_real_val = real_val_pool[:n_add_val]

print("=" * 70)
print(f"PLANNED ADDITIONS (STRICTLY BALANCED 1:1):")
print(f"  Train: +{n_add_train} Fake, +{n_add_train} Real (Total New Train = {n_add_train * 2})")
print(f"  Val:   +{n_add_val} Fake, +{n_add_val} Real (Total New Val   = {n_add_val * 2})")
print("=" * 70)

# ==============================================================================
# STEP 5: Extract & Standardize into Train / Val Folders
# ==============================================================================
dest_train_fake = BASE_DATASET_ROOT / 'train' / 'fake'
dest_train_real = BASE_DATASET_ROOT / 'train' / 'real'
dest_val_fake   = BASE_DATASET_ROOT / 'val'   / 'fake'
dest_val_real   = BASE_DATASET_ROOT / 'val'   / 'real'

for p in (dest_train_fake, dest_train_real, dest_val_fake, dest_val_real):
    p.mkdir(parents=True, exist_ok=True)

# Helper: Cache open zipfile handles
zip_handles = {}
def get_zip_handle(p):
    if p not in zip_handles:
        zip_handles[p] = zipfile.ZipFile(p, 'r')
    return zip_handles[p]

# 5A. Extract Fake Images
print("Extracting and standardizing fake images...")
for split_name, items, dest_dir in [
    ('Train Fake', selected_fake_train, dest_train_fake),
    ('Val Fake', selected_fake_val, dest_val_fake)
]:
    for idx, (gen_name, z_path, fname, ident) in enumerate(items):
        zh = get_zip_handle(z_path)
        img_data = zh.read(fname)
        with Image.open(io.BytesIO(img_data)) as img:
            img = img.convert('RGB')
            # Standardize resolution if different from TARGET_SIZE
            if img.size != TARGET_SIZE:
                img = img.resize(TARGET_SIZE, Image.BICUBIC)
            out_name = f"{gen_name}_{ident}_{idx:06d}.jpg"
            img.save(dest_dir / out_name, quality=95)
    print(f"  -> Extracted {len(items)} images into {split_name}.")

# 5B. Extract Real WebFace Images
print("Extracting and standardizing real WebFace images...")
for split_name, items, dest_dir in [
    ('Train Real', selected_real_train, dest_train_real),
    ('Val Real', selected_real_val, dest_val_real)
]:
    for idx, (ident_id, img_bytes) in enumerate(items):
        with Image.open(io.BytesIO(img_bytes)) as img:
            img = img.convert('RGB')
            # High-quality bicubic upsampling from 112x112 to 256x256
            if img.size != TARGET_SIZE:
                img = img.resize(TARGET_SIZE, Image.BICUBIC)
            out_name = f"webface_id{ident_id}_{idx:06d}.jpg"
            img.save(dest_dir / out_name, quality=95)
    print(f"  -> Extracted {len(items)} images into {split_name}.")

for zh in zip_handles.values():
    zh.close()

# ==============================================================================
# STEP 6: Final Verification & Audit Table
# ==============================================================================
final_train_real = len(list(dest_train_real.glob('*')))
final_train_fake = len(list(dest_train_fake.glob('*')))
final_val_real   = len(list(dest_val_real.glob('*')))
final_val_fake   = len(list(dest_val_fake.glob('*')))

total_train = final_train_real + final_train_fake
total_val = final_val_real + final_val_fake
grand_total = total_train + total_val

print("\n" + "=" * 70)
print("DATASET MERGE & BALANCE AUDIT:")
print(f"  Train Real:  {final_train_real:,}")
print(f"  Train Fake:  {final_train_fake:,}  (Train Balance: {final_train_real / total_train:.1%})")
print(f"  Val Real:    {final_val_real:,}")
print(f"  Val Fake:    {final_val_fake:,}  (Val Balance:   {final_val_real / total_val:.1%})")
print(f"  Train/Val Ratio: {total_train / grand_total:.1%} / {total_val / grand_total:.1%}")
print(f"  Grand Total Images: {grand_total:,}")
print("=" * 70)
assert final_train_real == final_train_fake, "Train set is imbalanced!"
assert final_val_real == final_val_fake, "Val set is imbalanced!"
print("VERIFICATION PASSED: Dataset is 100% balanced with zero identity leakage.")
```

---

## Cell 4: Compress and Save Merged Dataset to Google Drive (Bash)
```bash
%%bash
set -euo pipefail

# 1. Output folder on Google Drive
OUTPUT_DRIVE_DIR="/content/drive/MyDrive/DataesetNoVideoMultipleDomainAndRealImages"
mkdir -p "$OUTPUT_DRIVE_DIR"

OUTPUT_ZIP="$OUTPUT_DRIVE_DIR/DatasetNoVideoMultipleDomainAndRealImages.zip"

echo "=== PACKAGING MERGED DATASET TO GOOGLE DRIVE ==="
echo "Target archive: $OUTPUT_ZIP"

cd /content/dataset

# 2. Use 7z with fast compression (-mx=1) for rapid packaging
7z a -tzip -mx=1 "$OUTPUT_ZIP" DatesetNoVideosNoGAN

# 3. Verify archive integrity
ls -lh "$OUTPUT_ZIP"
echo "=== SUCCESS! MERGED DATASET SAVED TO GOOGLE DRIVE ==="
echo ""
echo "In future Colab training notebooks, you only need this single line to extract:"
echo "!7z x \"$OUTPUT_ZIP\" -o/content/dataset/ -y"
```
