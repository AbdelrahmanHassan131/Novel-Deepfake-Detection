# Controlled RGB Data-Mixture Pilot Protocol

**Document:** `CONTROLLED_RGB_MIXTURE_PROTOCOL.md`  
**Location:** `protocols/CONTROLLED_RGB_MIXTURE_PROTOCOL.md`  
**JSON Specification:** [controlled_rgb_mixture_protocol.json](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/protocols/controlled_rgb_mixture_protocol.json)  
**Status:** Frozen  

---

## 1. Hypothesis and Scientific Scope

### 1.1 The Engineering Hypothesis
The standard R2 training cohort of 100K images reflects an extreme internal class-stratum asymmetry:
- Authentic real data contains 76.7% numeric JPEG photographs and only 23.3% video PNG face crops.
- Manipulated fake data contains 94.5% video PNG crops and only 5.5% numeric PNG portraits.

**Hypothesis:** The severe disparity in real recall observed on development data (84.15% on video PNG real vs 99.91% on numeric JPEG real) is exacerbated by this sampling imbalance. Training a model on an **equal mixture** of the four observable strata (Arm B: 5,000 each of numeric real, video real, numeric fake, video fake) may improve worst-case stratum recall and overall balanced accuracy compared to an **R2-like mixture** (Arm A: 7,714 numeric real, 2,286 video real, 551 numeric fake, 9,449 video fake) under an identical 20,000-sample training budget.

### 1.2 Non-Claims and Guardrails
- **Not a Verified-Source Experiment:** The strata are defined purely by observable file naming conventions (`vid_` hex pattern vs numeric index), not confirmed dataset origins or identity labels.
- **No Promised Accuracy:** Equalizing representation does not remove container format confounding (numeric real remains 100% JPEG; numeric fake remains 100% PNG).
- **No Training Resumption:** Models in both arms start from standard ImageNet initialization; no weights from previous R2 runs are reused.

---

## 2. Experimental Controls and Invariants

Every variable outside the sampling mixture is strictly locked across both arms:

| Component | Invariant Specification |
|---|---|
| **Architecture** | `Wang2020_128` (ResNet50 backbone, 128-dim linear projection head, dropout 0.5) |
| **Backbone Init** | Standard PyTorch ImageNet pretrained weights |
| **Fine-Tuning Policy**| `layer4_and_head` (Backbone layers 1–3 frozen; BatchNorm frozen) |
| **Backbone LR Multiplier** | `0.1` (Backbone LR: `1e-5`, Head LR: `1e-4`) |
| **Optimizer** | Adam (`beta1=0.9`, `weight_decay=0.0`) with cosine annealing schedule |
| **Batch Budget** | Batch size `32`, `grad_accum_steps=2` (Effective batch size = `64`) |
| **Epoch Budget** | Exactly **5 fixed epochs** (No early stopping in either arm to guarantee equal sample exposures) |
| **Augmentation** | Exact production `rgb_v1` recipe: JPEG compression (`prob=0.5`, qualities 50..95), Gaussian blur (`prob=0.5`, sigma 0..3) |
| **Preprocessing** | Resize to 256 via bilinear interpolation, CenterCrop 224, standard ImageNet normalization |
| **Random Seed** | `42` |

---

## 3. Quota Design

### 3.1 Training Arms (20,000 samples each)
Both training arms maintain strict 50/50 Real/Fake class balance:

- **Arm A (R2-like Mixture):**
  - Numeric Real: **7,714** (77.14% of real class)
  - Video-Pattern Real: **2,286** (22.86% of real class)
  - Numeric Fake: **551** (5.51% of fake class)
  - Video-Pattern Fake: **9,449** (94.49% of fake class)
- **Arm B (Equal Stratum Mixture):**
  - Numeric Real: **5,000** (50.0% of real class)
  - Video-Pattern Real: **5,000** (50.0% of real class)
  - Numeric Fake: **5,000** (50.0% of fake class)
  - Video-Pattern Fake: **5,000** (50.0% of fake class)

### 3.2 Shared Development Partition (2,000 samples)
- Numeric Real: **500**
- Video-Pattern Real: **500**
- Numeric Fake: **500**
- Video-Pattern Fake: **500**
- **Strict Invariant:** Evaluated on the exact same sample IDs in both Arm A and Arm B. Zero overlap with training connected groups.

---

## 4. Evaluation and Decision Metric

1. **Primary Selection Metric:** Overall Development ROC AUC across the 5 epochs.
2. **Diagnostic Stratum Metrics Reported:**
   - Real recall on Numeric Real (`numeric_real`)
   - Real recall on Video Real (`video_real`)
   - Fake recall on Numeric Fake (`numeric_fake`)
   - Fake recall on Video Fake (`video_fake`)
   - Minimum stratum recall: $\min(R_{\text{num\_real}}, R_{\text{vid\_real}}, R_{\text{num\_fake}}, R_{\text{vid\_fake}})$
   - Numeric subcollection ROC AUC & Video subcollection ROC AUC.
3. **Threshold Selection:** Default metrics at fixed 0.5 threshold. A secondary optimal threshold is derived strictly from development predictions and bound to the checkpoint.
