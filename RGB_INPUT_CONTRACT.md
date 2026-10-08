# RGB Input and Preprocessing Contract

**Document Version:** 1.0  
**Target Architecture:** Wang2020_128 (ResNet-50 with 128-D embedding classification head)  
**Status:** Established & Verified Contract (Static verification complete; runtime checks deferred)

---

## 1. Pipeline Overview & Preprocessing Lifecycle

The Wang2020_128 detector processes standard 3-channel RGB imagery. The contract below defines the exact transformation sequence, numerical properties, and execution semantics that must be preserved between training, validation, and standalone evaluation.

```
Raw Image File (Disk)
      │
      ▼  (PIL Image.open via default_loader)
RGB Image [H, W, 3], uint8 [0, 255]
      │
      ▼  (custom_resize: PIL.Image.BILINEAR to loadSize=256)
Resized Image [256, 256, 3] (or scaled short side)
      │
      ▼  (data_augment: blur/jpeg/noise/downscale if isTrain, BYPASSED in eval)
Augmented Image (Train only)
      │
      ▼  (Crop: CenterCrop(cropSize=224) in eval, RandomCrop in train; Identity if no_crop)
Cropped Image [224, 224, 3]
      │
      ▼  (Horizontal Flip: RandomHorizontalFlip in train; Identity in eval or if no_flip)
Flipped Image (Train only)
      │
      ▼  (transforms.ToTensor())
Tensor [3, 224, 224], float32 in range [0.0, 1.0]
      │
      ▼  (transforms.Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
Normalized Tensor [3, 224, 224], float32
      │
      ▼  (model.forward())
Logit z in (-inf, +inf) -> Sigmoid -> Probability p in [0.0, 1.0]
```

---

## 2. Formal Stage Specification

### 2.1 Decode & Color Mode
- **Loader:** `torchvision.datasets.folder.default_loader` using `PIL.Image.open(path).convert('RGB')`.
- **Dtype & Color Space:** PIL Image in 8-bit RGB color mode (3 channels).
- **Format Handling:** Grayscale (`L`), transparency (`RGBA`), and CMYK images are automatically converted to 3-channel RGB during decode.
- **Path Resolution:** Strict matching by canonical `sample_id` and exact resolved file path. Never infer paths by filename basename.

### 2.2 Spatial Resizing
- **Function:** `data.transforms.resize.custom_resize`.
- **Target Size:** `opt.loadSize` (default: `256`).
- **Interpolation:**
  - **Validation & Standalone Evaluation:** `opt.rz_interp[0]` (default: `'bilinear'`, mapped to `PIL.Image.BILINEAR`).
  - **Training:** `sample_discrete(opt.rz_interp)` allows stochastic interpolation sampling across `['bilinear', 'bicubic', 'lanczos', 'nearest']` if configured.
- **Bypass Flag:** If `opt.no_resize` is `True`, resizing is an identity transform (`transforms.Lambda(identity_image)`).

### 2.3 Data Augmentation
- **Function:** `data.transforms.augmentations.data_augment`.
- **Evaluation Guarantee:** Bypassed when `not isTrain`. In validation and evaluation, blur, JPEG compression, additive noise, and downscaling probabilities are strictly `0.0`.
- **Training Mode:** Class-independent probabilistic transformations applied in fixed sequence:
  1. Gaussian blur (`blur_prob`, sigma sampled from `blur_sig`)
  2. JPEG compression (`jpg_prob`, method from `jpg_method`, quality from `jpg_qual`)
  3. Gaussian noise (`noise_prob`, sigma sampled from `noise_std`)
  4. Downscale & upscale (`downscale_prob`, scale from `downscale_range`)

### 2.4 Cropping
- **Function:**
  - **Validation & Standalone Evaluation:** `transforms.CenterCrop(opt.cropSize)` (default: `224`).
  - **Training:** `transforms.RandomCrop(opt.cropSize)` (default: `224`).
- **Bypass Flag:** If `opt.no_crop` is `True`, cropping is an identity transform (`transforms.Lambda(identity_image)`).

### 2.5 Horizontal Flipping
- **Function:**
  - **Validation & Standalone Evaluation:** Identity (`transforms.Lambda(identity_image)`).
  - **Training:** `transforms.RandomHorizontalFlip()` unless `opt.no_flip` is `True`.

### 2.6 Tensor Conversion & Normalization
- **ToTensor:** `torchvision.transforms.ToTensor()`. Converts PIL Image `[H, W, 3]` in `[0, 255]` to PyTorch FloatTensor `[3, H, W]` in range `[0.0, 1.0]`.
- **Normalization:** ImageNet constants:
  - `mean = [0.485, 0.456, 0.406]`
  - `std = [0.229, 0.224, 0.225]`
- **Calculation:** `x_norm = (x - mean) / std`.

---

## 3. Model Execution & Evaluation Semantics

### 3.1 Network Architecture
- **Backbone:** ResNet-50 (`models.shared.resnet.resnet50`).
- **Classification Head:**
  ```python
  nn.Sequential(
      nn.Linear(2048, 128),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(128, 1)
  )
  ```
- **Total Head Parameters:** 262,401.

### 3.2 Evaluation Mode & BatchNorm Contract
- **Mode Enforcement:** `model.eval()` must be called before validation or inference.
- **Dropout:** Disabled in eval mode.
- **BatchNorm2d:** In eval mode, running statistics (`running_mean` and `running_var`) are used; batch statistics are not updated.
- **Distributed Evaluation Buffer Sync:** In DDP validation, all BatchNorm buffers are synchronized across ranks via rank 0 before local evaluation.

### 3.3 Score Direction & Label Mapping
- **Canonical Mapping:** `real: 0`, `fake: 1`.
- **Logit Definition:** Binary output $z \in \mathbb{R}$ from final `Linear(128, 1)`.
- **Probability:** $p = \sigma(z) = \frac{1}{1 + e^{-z}}$.
- **Score Sign:** `score_sign = 1.0` if `fake == 1`.
  - $z > 0.0 \iff p > 0.5 \implies$ Predicted Fake (`1`).
  - $z < 0.0 \iff p < 0.5 \implies$ Predicted Real (`0`).
  - Score interpretation must never be silently inverted or guessed without explicit checkpoint label mapping metadata.

---

## 4. Checkpoint Protocol & Metadata Restoration

Checkpoints save complete metadata under `protocol`:
1. `format_version`: Integer schema version (e.g., 2).
2. `label_mapping`: Must explicitly declare `{'real': 0, 'fake': 1}`.
3. `options`: Full snapshot of training options, including preprocessing keys (`cropSize`, `loadSize`, `rz_interp`, `no_crop`, `no_resize`).
4. `manifests`: Bound paths and SHA-256 digests of training/validation manifests.
5. `checkpoint_sha256`: Checkpoint file content hash.

When restoring via `CheckpointLoader`, `opt.isTrain` is set to `False`, all preprocessing options are restored from the protocol, and legacy `load_networks` auto-loads are bypassed.

---

## 5. User-Run Diagnostic Instructions

A dedicated diagnostic tool is provided in `tools/diagnose_rgb_parity.py`. It runs in two modes:

### 5.1 Synthetic Mode (Safe, CPU-only, No Dependencies)
Validates transform pipelines, eval/BN modes, and forward pass parity on synthetic in-memory images:
```powershell
python tools/diagnose_rgb_parity.py --synthetic
```

Expected output:
- `Status : passed`
- `Mode Checks : BN eval=True, Dropout eval=True`
- `Max Tensor Diff : 0.00e+00`
- `Max Logit Diff : 0.00e+00`

### 5.2 Real Checkpoint Parity Mode (User-Run on GPU/CPU)
Evaluates an actual checkpoint against a small manifest and cross-checks with saved predictions:
```powershell
python tools/diagnose_rgb_parity.py `
    --checkpoint checkpoints/wang2020_128.pth `
    --manifest data/small_val_manifest.csv `
    --sample_ids sample_001,sample_002,sample_003 `
    --saved_predictions predictions_dev.csv `
    --output_report rgb_parity_report.json
```
If discrepancies exceed `--tolerance 1e-5`, the tool outputs the exact mismatch and exits with code 2. It will never auto-correct labels.
