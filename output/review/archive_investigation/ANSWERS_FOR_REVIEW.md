# Evidence-Based Archive Investigation Answers

**Report Directory:** [archive_investigation](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation)  
**Investigation Timestamp:** 2026-10-09T18:45:20Z  
**Primary Archive:** `E:\PreparedDataset\Preapred Dataset.zip` (17 volumes, 32.52 GiB compressed, 1,016,855 images)  
**External Diagnostic Set:** `F:\val to be deleted 2` (60,000 images)  

---

## Task Completion Summary

| Prompt / Task | Status | Commands / Tools Used | Storage Consumed | Primary Artifacts | Checks Not Performed |
|---|---|---|---|---|---|
| **Prompt 1: Access & Storage** | **Completed** | Discovered `7z.exe` (22.01 x64) via NVIDIA App; `shutil.disk_usage` | 0 MiB scratch | [access_report.json](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/access_report.json) | Full CRC across 32.5 GiB; software installation; training |
| **Prompt 2: Metadata & Provenance** | **Completed** | Stream listing via 7-Zip (`7z l`), streaming gzip parser | ~12.1 MiB (`archive_listing.tsv.gz`) | [archive_inventory_summary.json](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/archive_inventory_summary.json), [source_evidence.md](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/source_evidence.md) | Full payload extraction |
| **Prompt 3: Relationships & Overlap** | **Completed** | Token regex, hash joins to `clean_parent` and old 100K manifest | 0 MiB scratch | [relationships_report.json](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/relationships_report.json) | Hash calculation for entire 1M images; manifest repartitioning |
| **Prompt 4: Direct Sample Inspection** | **Completed** | Bounded 7-Zip extraction (`7z x @list`), PIL profiling, contact sheets | ~11.2 MiB (256 train images) | [sample_members.csv](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/sample_members.csv), [sample_profile.json](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/sample_profile.json), 4 Contact Sheets | Inference execution; checkpoint weight loading |
| **Prompt 5: Synthesis & Decision** | **Completed** | Synthesis across empirical evidence | ~3.8 MiB total in repo | [ANSWERS_FOR_REVIEW.md](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/ANSWERS_FOR_REVIEW.md), [NEXT_IMPLEMENTATION_PROMPTS.md](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/NEXT_IMPLEMENTATION_PROMPTS.md) | Speculative training recipes |

---

## Core Questions and Evidence-Based Answers

### 1. Which real datasets and fake generators are actually verified? What remains unknown?
* **Verified from explicit archive metadata:** **None.**  
  The archive contains 1,016,855 image files and 7 directories (`train/fake`, `train/real`, `val/fake`, `val/real`). There are zero READMEs, license files, source lists, generator manifests, or camera metadata files. Downstream CSV files tag images with `dataset_source: "diffgan"` and `generator: "authentic"` / `"unknown"`, but these are pipeline placeholders rather than verified dataset cards.
* **Suggested by naming patterns and visual contact sheets:**
  * **Video Face Manipulation:** 765,164 files (612k train, 153k val) match `vid_<64hex>_face_<digits>_<digits>.png`. Contact sheets confirm these are tight/medium face crops extracted across video frames of interview/talk-show footage, with blending and warping artifacts characteristic of FaceForensics++ video manipulations.
  * **Numeric Images:** 251,691 files (`<digits>.<ext>`) have flattened anonymous numbering. In `real`, all 162k numeric files are JPEG face crops; in `fake`, all 31k numeric files are PNG 256x256 crops.
  * **External Diagnostic Dataset (`F:\val to be deleted 2`):** Visual contact sheets confirm that `external/real` consists of 256x256 celebrity/studio still photography portraits (CelebA-HQ / FFHQ style), while `external/fake` consists of 256x256 whole-face synthetic GAN/diffusion portraits (StyleGAN / ProGAN / DiffGAN style).
* **Unknown:**
  * Exact generator architectures (DeepFakes vs Face2Face vs FaceSwap vs NeuralTextures vs StyleGAN vs Diffusion).
  * Original video dataset identity (FaceForensics++, Celeb-DF, DFDC).
  * Original identities/subjects.

---

### 2. Are training and external images the same detection task?
* **NO. Training and external images represent two fundamentally DIFFERENT tasks and domains:**
  1. **Task Disparity:**
     * **Training Task:** Video facial manipulation / face-swap detection on extracted video frame crops. The fake class consists of real videos where facial regions have been swapped or re-enacted, retaining video compression, motion blur, and variable aspect ratios (mean ~170x170).
     * **External Task:** Whole-face unconditional GAN/diffusion synthesis detection on studio still portraits. The fake class consists of fully synthesized portraits generated from scratch (aligned 256x256 faces with GAN texture artifacts), while the real class consists of high-resolution still photography portraits.
  2. **Resolution & Geometry Gap:**
     * Training images vary widely from 40x40 to 372x372 (aspect ratio ~0.92 to 1.09). Several training samples are even false-positive crops (ears, background cloth patches, foliage).
     * External images are 100% uniform 256x256, aspect ratio exactly 1.0, with standardized facial landmark centering.
  3. **Conclusion:** Evaluating the RGB detector on `F:\val to be deleted 2` is NOT an in-domain generalization test; it is an out-of-task domain transfer test across completely different deepfake modalities.

---

### 3. Which known relationships invalidate current partitions? What leakage remains possible after the existing family repair?
* **Archive-Level Partition (`train/` vs `val/` in `Preapred Dataset.zip`):**
  * The original archive contains 21,100 unique `vid_` families in `train` and 5,274 unique `vid_` families in `val`.
  * **Our audit verified that EXACTLY 0 families cross between archive `train` and archive `val`.** The original author properly partitioned videos across `train` and `val`.
* **Downstream Parent Manifest Defect (Colab/CPU pipeline):**
  * The downstream preparation pipeline discarded the archive's `val` partition entirely (0 val images were used).
  * It extracted 794,243 images strictly from `train/` and carved out an internal `dev` split (14,835 samples) **randomly per-image**, causing **6,926 crossing filename families** (affecting almost 100% of dev!).
* **What the Family Repair (`output/review/rgb_family_repaired_100k_seed42`) fixes:**
  * Reduced crossing `vid_<hex>` families in the 100K train/dev pool from 6,420 to **0**.
* **What leakage and defects remain unresolved:**
  1. **Numeric Files (193k in train, 58k in val):** Anonymous numeric filenames cannot be grouped by video token. Video frames named with integers may cross splits undetected.
  2. **Real-to-Fake Source Pairing:** If a manipulated video was created from an original real video, both share the underlying identity/scene. Because the archive lacks real-to-fake provenance links (`source_video_id` is unknown), the real and fake versions of the same video sequence could be split across train and dev.
  3. **Wasted Validation Partition:** The pristine 211,434-image `val/` partition inside `Preapred Dataset.zip` was completely omitted from model evaluation.

---

### 4. Which class-correlated format, resolution, crop or source differences were observed?
* **Empirical Observations:**
  1. **Extreme Class-Format Confounding in Training Archive:**
     * `train/fake`: **100% PNG (594,233 / 594,233). Exactly ZERO JPEG files exist in fake.**
     * `train/real`: **76.7% JPEG (162,078 / 211,188) and 23.3% PNG (49,110 / 211,188).**
     * In the entire 1,016,855-image archive, every single JPEG image is REAL. There is not a single JPEG fake image.
  2. **Repaired Dev Cohort Imbalance:**
     * `dev/real`: 6,950 JPEG, 550 PNG (92.7% JPEG).
     * `dev/fake`: 550 JPEG, 6,950 PNG (92.7% PNG).
  3. **External Diagnostic Dataset Discrepancy:**
     * `external/real`: 100% PNG (30,000 / 30,000).
     * `external/fake`: 100% PNG (30,000 / 30,000).
* **Separation of Observations from Causal Hypotheses:**
  * *Observation:* In training, JPEG is 100% real, and fake is 100% PNG. In external evaluation, 100% of images are PNG.
  * *Causal Hypothesis:* An RGB convolutional model trained on this archive easily exploits format/compression cues (e.g. absence of DCT blocking, PNG demosaicing artifacts) as a strong shortcut for fake. When tested on external real images that are 100% PNG, the model predicts them as fake.
  * *Nuance:* Because JPEG augmentation (`data_augment`) was used during training, format correlation cannot be claimed as the sole reason; however, the lack of JPEG fake training data combined with the radical task shift (video face swaps vs GAN synthesis) creates a severe compounding failure mode.

---

### 5. Can existing data support both classes across meaningful domains and a source-held-out development split?
* **NO, not within `Preapred Dataset` alone.**
  * Provenance is flattened: there are no generator tags, source video IDs, or domain labels.
  * Fake data is completely devoid of JPEG images.
  * A true source-held-out evaluation requires multi-generator annotations (e.g. holding out Face2Face while training on DeepFakes, or holding out StyleGAN while training on ProGAN). The archive does not preserve these distinctions.

---

### 6. What is the smallest controlled experiment justified by the evidence?
* **Hypothesis:**
  The detector's 50.55% external performance is caused by: (1) task domain mismatch (video face manipulation vs full GAN synthesis), and (2) format confounding (PNG vs JPEG). In-domain performance on true held-out video sequences will demonstrate strong detection, whereas cross-task transfer fails predictably.
* **Controlled Protocol:**
  1. **In-Domain Zero-Leakage Test:**
     * Extract a balanced 10,000-image evaluation cohort from the archive's own `val/` partition (`Preapred Dataset/val`), which has verified **0 crossing video families** with `train/`.
     * Stratify by format and pattern (5,000 real, 5,000 fake).
  2. **Format-Controlled Ablation:**
     * Evaluate on raw PNG/JPEG, then re-evaluate the same cohort re-encoded uniformly as quality-95 JPEG and uncompressed PNG.
  3. **Role of External Dataset:**
     * Retain `F:\val to be deleted 2` strictly as an out-of-domain diagnostic suite; do not tune thresholds on it or treat it as an unseen test benchmark.
* **Success/Failure Criteria:**
  * In-domain `val` AUC > 0.90 demonstrates that the detector successfully learned video manipulation features in-domain.
  * In-domain AUC remaining high after format standardization confirms features are not purely compression shortcuts.
  * Continued poor performance on external GAN data confirms that cross-task generalization between video face-swaps and GAN portraits requires specialized architectural or multi-task domain adaptation, not merely partition adjustment.

---

### 7. What exact additional information is still needed from the user?
1. **Raw dataset origins of `Preapred Dataset`:** What original datasets were downloaded and combined (e.g. FaceForensics++ c23/c40, Celeb-DF v2, DFDC)?
2. **Origin of numeric files (`<digits>.jpg` / `<digits>.png`):** Were these pulled from CelebA, FFHQ, or another benchmark?
3. **Primary Thesis Claim:** Is the thesis evaluating **video face-swap detection** (in which case `Preapred Dataset/val` is the proper test set), or **universal cross-generator synthesis detection** (which requires training on diverse GAN/diffusion generators)?
