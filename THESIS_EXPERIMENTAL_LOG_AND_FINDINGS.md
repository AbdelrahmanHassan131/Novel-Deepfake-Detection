# Thesis Experimental Log, Architecture, Dataset Card & Cross-Domain Findings

**Project:** Novel Deepfake Detection — Cross-Domain Generalization Study  
**Branch:** `GeneralizationDeepFakeDetection`  
**Last Updated:** 2026-11-22  
**Primary Compute:** Single NVIDIA L4 (24 GB VRAM, BF16 AMP) for training/evaluation; Google Colab CPU for dataset assembly & auditing.

---

## 1. Research Problem & Root-Cause Diagnosis of Prior Failures

### Initial Failure Mode
Early CNN and Wavelet baselines (`Wang2020_128` ResNet-50, `WolterWavelet2021_128`, `ConvNeXt_128`, and two-stream `MHA_128`) achieved **97%–99.6% accuracy in-domain**, yet collapsed to **~50% accuracy** when evaluated on unseen domains (`val to be deleted 2`, 60,000 images: 30K real + 30K fake).

### Four Empirically Verified Root Causes
1. **Packaging & Resolution Shortcuts:**
   - In `DatesetNoVideosNoGAN`, `train/real` was **100% `.jpg` at `178×218`** (standard CelebA), whereas `train/fake` was **100% `.png` at `256×256`** (`ADM`, `DDIM`, `DDPM`, `DiffSwap`).
   - In the earlier 570K attempt (`DatasetNoVideoMultipleDomainAndRealImages.zip`), added real faces came from **WebFace `112×112` bicubic-upscaled to `256×256`** (zero true frequency energy above the 112px Nyquist limit) paired against sharp native `256×256`/`512×512` diffusion fakes—and 5 of the 7 extended generator ZIPs were in `FS/`, `I2I/`, `T2I/` rather than `FE/` and had been skipped.
   - As a result, end-to-end fine-tuned CNNs (`Wang2020_128`, `ConvNeXt_128`) and unfrozen ViT adapters (`EFFORT_128` with `svd_rank=16`) memorized *"178×218 JPG / blurry = Real; 256×256 PNG / sharp = Fake"*, pushing unseen `256×256 PNG` reals to ~99% fake probability (`RealRec = 0.7%–6.0%`).
2. **Pixel-Space vs. Latent-Space Diffusion Gap:**
   - Training solely on pixel-space diffusion (`ADM`, `DDIM`, `DDPM`) did not cover Latent Diffusion / VAE-decoded synthesis artifacts (`LDM`, `SDXL`, etc.).
3. **In-Domain Model Selection & Unregularized Logit Saturation:**
   - Selecting `best.pth` on an in-domain validation split picked checkpoints that overfit training-generator shortcuts (`Dev AUC = 1.0000`) while degrading on unseen generators.
4. **The `CelebA-HQ` Neural Super-Resolution Trap (`val to be deleted 2 / real`):**
   - `val to be deleted 2 / real` consists of 30,000 `256×256` faces (`0.png`–`29999.png`) from **CelebA-HQ** (Karras et al., ICLR 2018), which NVIDIA synthesized from low-res CelebA JPEGs via a **neural JPEG-artifact removal autoencoder** followed by a **deep 4× super-resolution CNN**.
   - Every forensic detector trained to distinguish real camera photography from neural synthesis naturally flagged neurally super-resolved `CelebA-HQ` as AI-processed (`RealRec = 0.7%–13.4%`), while correctly classifying genuine unseen camera photos (`LFW`) as **98.0%–98.6% REAL**.

---

## 2. Proposed Architecture & Engineering Solutions Implemented

### A. Frozen Multi-Layer Foundation Expert (`EFFORT_128`, `svd_rank=0`)
Implemented in `models/effort/` (`effort_model.py`, `svd_linear.py`, `loss.py`, `trainer.py`):
- **Backbone:** OpenAI CLIP `ViT-L/14` visual encoder (`304.47M` total parameters), kept **100% frozen** (`effort_svd_rank=0`) to preserve universal visual representations and prevent dataset-shortcut memorization. *(Optional Orthogonal SVD Residual Adaptation `SVDResidualLinear` is also implemented for controlled ablations.)*
- **Multi-Layer Intermediate Block Fusion (RINE-style):** Taps `[CLS]` tokens from **4 evenly spaced intermediate ViT blocks** (`[5, 11, 17, 23]` in 24-layer ViT-L/14), applies per-layer `LayerNorm`, aggregates them via **learnable softmax layer weights** $\alpha_l$, and concatenates with the normalized final CLIP visual embedding.
  - **Empirical Interpretability Finding:** Across runs, the model automatically assigned **`39%–45%` weight to middle Block 11**, **`24%–25%` to Block 17**, **`15%–19%` to early Block 5**, and only **`15%–18%` to final semantic Block 23**—directly explaining why multi-layer fusion beat single-layer `DRCT_128` (`0.9796` vs. `0.9365` ROC AUC on identical data).
- **128-D Fusion-Compatible Head (`502,533` trainable parameters):**
  `Concat(1024 + 768) -> Linear(1792, 256) -> LayerNorm -> GELU -> Dropout(0.4) -> Linear(256, 128) -> LayerNorm(128) -> Dropout(0.4) -> Linear(128, 1)`
  - Strictly preserves the `128-D` penultimate embedding (`forward_features(x)`) + single raw binary logit (`real=0, fake=1`) required for `MHA_128` and `Fusion_128`.
- **Objective & Optimization:** Combined label-smoothed `BCEWithLogitsLoss` (`s = 0.05`) + normalized Euclidean `MarginContrastiveLoss` (`alpha = 0.2`, `pos_margin = 0.0`, `neg_margin = 1.0`), `AdamW` (`lr = 2e-4`, `weight_decay = 0.02`, no decay on biases/LayerNorm/layer weights), cosine LR schedule, `BF16` AMP on L4 (`FP32` validation).

### B. Resolution-Harmonized Symmetric Forensic Preprocessing (`clip_forensic_v1`)
Implemented in `data/transforms/clip_forensic_transforms.py`:
- Uses native OpenAI CLIP normalization (`mean=[0.48145466, 0.4578275, 0.40821073]`, `std=[0.26862954, 0.26130258, 0.26862954]`) and bicubic resampling.
- Applies identical square cropping + shared effective-resolution bandwidth harmonization (`160..224px`), `80%` stochastic JPEG recompression (`q ∈ [50..95]`, `cv2`/`pil`), `35%` Gaussian blur (`σ ∈ [0.1, 2.0]`), `35%` downscale-upscale resampling (`scale ∈ [0.55, 1.0]`), `30%` mild color/exposure jitter, and `50%` horizontal flip symmetrically to both classes.

### C. Resumable Paired Diffusion/VAE Reconstruction Pipeline
Implemented in `tools/generate_reconstructions.py`:
- Supports `sd15_vae` (`stabilityai/sd-vae-ft-mse`), `sdxl_vae` (`madebyollin/sdxl-vae-fp16-fix`), `multi_vae` (50/50 deterministic mix), and `sd_turbo` (2-step diffusion img2img), emitting SHA256-verified `reconstruction_mapping.csv` manifests compatible with `DRCT_128` and `EFFORT_128`.

---

## 3. Clean Multi-Domain Dataset Card (`DatasetMultiDomainClean_v2`)

Built on Colab CPU via `COLAB_BUILD_CLEAN_MULTIDOMAIN_V2.md` and archived on Google Drive at:
`/content/drive/MyDrive/DatasetMultiDomainClean_v2/DatasetMultiDomainClean_v2.zip`

### Authoritative Split & Source Counts (`556,082` Total Verified Images)
- **Train Split (`469,222` images — exact 50.0% / 50.0% balance):**
  - **Real (`234,611` across 4 native high-res sources; ZERO `112×112` WebFace):**
    - `fairface` (`HuggingFaceM4/FairFace` `"0.25"`, race/gender/age balanced): `86,744` (`.png`)
    - `celeba` (base): `80,000` (`.jpg`)
    - `ffhq256` (`bitmind/ffhq-256`, native 256×256 Flickr faces): `56,000` (`.png`)
    - `utkface` (`nu-delta/utkface`, ages 0–116): `11,867` (`.png`)
    - *Extension mix:* `154,611` `.png` (65.9%) + `80,000` `.jpg` (34.1%)
  - **Fake (`234,611` across 15 generators, capped evenly for balance, 100% `.png`):**
    - Base pixel-diffusion & face swap (`4`): `adm` (20,000), `ddim` (20,000), `ddpm` (20,000), `diffswap` (20,000)
    - Extended FE / FS / I2I / T2I (`11`): `CoDiff` (15,000), `cycle_diff` (15,000), `DCFace` (15,000), `DiffFace` (15,000), `DreamBooth` (15,000), `HPS` (15,000), `SDXL` (15,000), `SDXL_Refine` (15,000), `FreeDoM_T` (13,776), `FreeDoM_I` (12,434), `LoRA` (8,401)
- **Validation Split (`86,860` images — exact 50.0% / 50.0% balance):**
  - **Real (`43,430`):** `celeba` (20,000), `ffhq256` (14,000), `fairface_val` (9,430)
  - **Fake (`43,430`):** all 15 generators from official val splits
- **Automated Leakage Audit (`dataset_audit_report.json`):**
  - `fake_identity_overlap = 0` (globally enforced across all extended generators)
  - `real_identity_overlap = 0`
  - `ldm_leakage_count = 0` (`LDM`, `Midjourney`, `Imagic`, `StyleGAN`, `LFW`, and `val to be deleted 2` are **100% excluded** from train and val)
  - Every image verified intact via `PIL.Image.load()` and center-square-cropped before `256×256` Bicubic resizing.

### Leave-One-Generator-Out (LOGO) Training Protocol (`COLAB_TRAIN_MULTIDOMAIN_V2_L4.md`)
- Held **`SDXL_Refine`** (I2I latent diffusion) and **`DiffFace`** (diffusion face swap) **100% out of training** and placed them in the `dev` split (50% unseen LOGO fakes + 50% seen val fakes vs. stratified CelebA/FFHQ/FairFace reals) so `best.pth` and `threshold_logit.json` optimize for **unseen-generator generalization**.

---

## 4. Controlled Ablation & Generalization Results

### Table 1: Diagnostic Ablation on 80K–120K Pilot (`DatesetNoVideosNoGAN` → `val to be deleted 2`, 60K)
| Model / Experiment | Backbone Policy | Training Composition | Unseen ROC AUC | Unseen EER | Key Scientific Takeaway |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `Wang2020_128` (R2 baseline) | ResNet-50 fine-tuned | Legacy pool | `0.7242` | — | Format/source shortcut (`RealRec=1.1%`) |
| `WolterWavelet2021_128` | Wavelet CNN (scratch) | Legacy pool | `0.7507` | `31.51%` | Format/source shortcut (`RealRec=4.4%`) |
| `Stage1_EFFORT_128` | CLIP + SVD unfrozen (`r=16`) | Unpaired 160K | `0.5758` | `45.26%` | Unfrozen SVD memorized `178×218 JPG` vs `256×256 PNG` (`Dev AUC=1.0000`) |
| `Stage2_EFFORT_128_Recon` | CLIP + SVD unfrozen (`r=16`) | +40K Multi-VAE recon | `0.6234` | `41.14%` | Paired recon improved AUC (+4.8%), but unfrozen SVD still overfit |
| `Stage2_DRCT_128_Forensic` | **Frozen CLIP** (1 layer) | 80K 100%-paired + forensic aug | **`0.9365`** | **`14.02%`** | Freezing CLIP prevented shortcut memorization (+31.3% AUC!) |
| **`7A_FrozenMultiLayer_80K`** | **Frozen 4-Layer CLIP** (`r=0`) | 80K 100%-paired + harmonized aug | **`0.9796`** | **`6.80%`** | **+4.31% AUC & >50% EER drop** over 1-layer DRCT (`93.35%` separation!) |
| `7B_FrozenMultiLayer_Pure` | Frozen 4-Layer CLIP (`r=0`) | 40K Real vs Multi-VAE only | `0.6251` | `41.19%` | VAE-only reconstructions are insufficient without real diffusion diversity |

### Table 2: Champion Model (`556K DatasetMultiDomainClean_v2`, LOGO Training) — 100% Unseen Cross-Domain Evaluation
**Champion Checkpoint:**  
`/content/drive/MyDrive/deepfake_multidomain_v2_runs/effort128_frozen_multilayer_v2_logo_20261121_150525_be2f666c/checkpoints/best.pth`

All evaluations below use **ZERO target-domain threshold tuning** (uncalibrated `0.5` / `logit = 0.0`). Every test uses **1,000 real + 1,000 fake (50/50 balanced)**, so Normal Accuracy (`Acc`) equals Balanced Accuracy (`BalAcc`).

| Unseen Real Source (0 in Train) | Unseen Fake Generator (0 in Train) | Test Condition | ROC AUC | Normal Acc (`0.5`) | Real Recall | Fake Recall | Median Logits (R / F) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`LFW` (camera photos)** | **`LDM`** (`val_to_be_deleted_2/fake`) | RAW | `0.9990` | **`98.55%`** | `98.0%` | `99.1%` | `-2.92` / `+2.87` |
| **`LFW` (camera photos)** | **`LDM`** (`val_to_be_deleted_2/fake`) | **Same JPEG `q=85`** | **`0.9995`** | **`98.90%`** | **`98.6%`** | **`99.2%`** | `-2.92` / `+2.87` |
| **`LFW` (camera photos)** | **`SDXL_Refine`** (LOGO unseen I2I) | RAW | `0.9996` | **`98.80%`** | `98.0%` | `99.6%` | `-2.92` / `+2.91` |
| **`LFW` (camera photos)** | **`SDXL_Refine`** (LOGO unseen I2I) | **Same JPEG `q=85`** | **`0.9998`** | **`99.00%`** | **`98.6%`** | **`99.4%`** | `-2.92` / `+2.91` |
| **`LFW` (camera photos)** | **`DiffFace`** (LOGO unseen face swap) | RAW | `0.9887` | **`90.75%`** | `98.0%` | `83.5%` | `-2.92` / `+2.85` |
| **`LFW` (camera photos)** | **`DiffFace`** (LOGO unseen face swap) | **Same JPEG `q=85`** | **`0.9866`** | **`87.95%`** | **`98.6%`** | **`77.3%`** | `-2.92` / `+2.85` |
| **`LFW` (camera photos)** | **`StyleGAN`** (**UNSEEN GAN, 0 GANs in train!**) | RAW | `0.9541` | **`74.35%`** | `98.0%` | `50.7%` | `-2.92` / `+0.14` |
| **`LFW` (camera photos)** | **`StyleGAN`** (**UNSEEN GAN, 0 GANs in train!**) | **Same JPEG `q=85`** | **`0.9720`** | **`81.35%`** | **`98.6%`** | **`64.1%`** | `-2.92` / `+2.05` |
| **`LFW` (camera photos)** | **`Midjourney`** (unseen commercial T2I) | RAW | `0.9986` | **`98.10%`** | `98.0%` | `98.2%` | `-2.92` / `+2.89` |
| **`LFW` (camera photos)** | **`Imagic`** (unseen face editing) | RAW | `1.0000` | **`99.00%`** | `98.0%` | `100.0%` | `-2.92` / `+2.92` |

### Table 2B: Frequency Expert (`WolterWavelet2021_128`, 1.7M Scratch CNN) vs. RGB Expert (`EFFORT_128`, 304M Frozen 4-Layer CLIP) — Direct Complementarity Proof
Trained on the **exact same 556K `DatasetMultiDomainClean_v2` LOGO split** using GPU-batched Haar Level-3 `signed_log1p` wavelet packets and identical forensic augmentations (`jpg_prob=0.8`, `blur_prob=0.35`, `downscale_prob=0.35`). Evaluated with **zero threshold tuning (`0.5 / logit = 0.0`)** on `1,000` unseen `LFW` reals vs. `1,000` unseen fakes per generator:

| Unseen Photorealistic Benchmark | Condition | `EFFORT_128` (RGB) AUC / Acc | `WolterWavelet2021_128` (Freq) AUC / Acc | `WolterWavelet` RealRec / FakeRec | Stronger Modality |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`LFW` vs. Unseen `LDM`** (`v2/fake`) | RAW | **`0.9990` / `98.55%`** | `0.9940` / `96.05%` | `97.8%` / `94.3%` | RGB (`EFFORT_128`) |
| **`LFW` vs. Unseen `LDM`** (`v2/fake`) | **JPEG `q=85`** | **`0.9995` / `98.90%`** | `0.9967` / `96.50%` | `98.5%` / `94.5%` | RGB (`EFFORT_128`) |
| **`LFW` vs. Unseen `SDXL_Refine`** | RAW | **`0.9996` / `98.80%`** | `0.9993` / `98.80%` | `97.8%` / `99.8%` | Tie (`98.80%`) |
| **`LFW` vs. Unseen `SDXL_Refine`** | **JPEG `q=85`** | **`0.9998` / `99.00%`** | `0.9994` / `98.75%` | `98.5%` / `99.0%` | Tie (~`99%`) |
| **`LFW` vs. Unseen `DiffFace` (Swap)** | RAW | **`0.9887` / `90.75%`** | `0.9776` / `85.10%` | `97.8%` / `72.4%` | RGB (`EFFORT_128`) |
| **`LFW` vs. Unseen `DiffFace` (Swap)** | **JPEG `q=85`** | `0.9866` / `87.95%` | **`0.9901` / `90.00%`** | **`98.5%` / `81.5%`** | **Wavelet (+2.05% Acc!)** |
| **`LFW` vs. Unseen `StyleGAN` (GAN)** | RAW | **`0.9541` / `74.35%`** | `0.7714` / `52.45%` | `97.8%` / `7.1%` | **RGB (+21.9% Acc!)** |
| **`LFW` vs. Unseen `StyleGAN` (GAN)** | **JPEG `q=85`** | **`0.9720` / `81.35%`** | `0.8411` / `54.10%` | `98.5%` / `9.7%` | **RGB (+27.25% Acc!)** |

> **Key Thesis Finding — Genuine Cross-Modal Complementarity:**
> 1. **Clean data + forensic augmentations transformed `WolterWavelet2021_128`** from a `52%` / `0.75` AUC collapsed model into a **`96.5%–98.8%` (`0.994–0.999` AUC)** unseen-diffusion detector—proving our data curation & forensic augmentation pipeline generalizes across completely different inductive biases (not just CLIP!).
> 2. **True Complementary Strengths:** On JPEG-compressed unseen face swaps (**`DiffFace`**), the tiny 1.7M-parameter scratch Wavelet CNN **outperformed** the 304M CLIP `EFFORT_128` model (`90.00%` Acc / `0.9901` AUC vs. `87.95%` Acc / `0.9866` AUC, FakeRec `81.5%` vs. `77.3%`) because wavelet subbands isolate local face-swap blending boundaries even after JPEG compression. Conversely, on unseen **StyleGAN** (0 GANs in training), `EFFORT_128` vastly outperformed Wavelets (`81.35%` / `0.9720` vs. `54.10%` / `0.8411`). This empirical asymmetry directly motivates **input-dependent multi-token attention (`MHA_128`)** over static concatenation or naive averaging.

### Table 3: Forensic Finding on Neurally Super-Resolved "Real" Datasets (`CelebA-HQ`)
| Real Dataset Tested | Creation Pipeline | Real Recall (`@0.0`) | Median Logit | Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **`LFW`** (1,000 unseen) | Direct real-world camera photography | **`98.0%` (`98.6%` JPEG q85)** | **`-2.920`** | Cleanly identified as real camera imagery |
| **`CelebA / FFHQ / FairFace`** (held-out val) | Direct real-world camera photography | **`99.8%`** | **`-2.923`** | Cleanly identified as real camera imagery |
| **`CelebA-HQ`** (`val_to_be_deleted_2/real`) | Low-res CelebA → **neural JPEG-removal AE + neural 4× SR CNN** | `10.3%–13.4%` | `+2.848` | Detector captures genuine neural upscaling/denoising artifacts in CelebA-HQ |

---

## 5. Remaining Thesis Milestones (Reviewer 5 Closure Checklist)

1. **Frequency Expert & Two-Stream `MHA_128` Fusion:**
   - Train `WolterWavelet2021_128` on `DatasetMultiDomainClean_v2` (with forensic augmentations).
   - Freeze `EFFORT_128` (`128-D`) + `WolterWavelet2021_128` (`128-D`) and compare **`MHA_128` (multi-token attention)** vs. **`Fusion_128` (concatenation)** vs. **late score ensemble** vs. standalone experts—especially measuring gains on **unseen `StyleGAN`** (where wavelets excel at GAN frequency spectra) and **`DiffFace`**.
2. **Perturbation & Adversarial Robustness:**
   - End-to-end JPEG (`q=30..90`), Gaussian blur, additive noise, downscale-upscale, and corrected raw-pixel FGSM/PGD curves.
3. **Ground-Truth Demographic Fairness:**
   - Evaluate FPR, FNR, AUC, balanced accuracy, and maximum demographic gaps across FairFace's ground-truth **race (7 groups), gender (2 groups), and age (9 bands)** using held-out FairFace validation samples.
4. **Multi-Seed Stability & Additional External Benchmarks:**
   - Run 3 independent seeds (`42, 123, 777`) for 95% bootstrap CIs and paired statistical tests, plus evaluate on an untouched video deepfake dataset (e.g., DFDC / FaceForensics++ extracted frames) if available.
