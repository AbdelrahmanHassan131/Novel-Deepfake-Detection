# Review Response: Corrected Evidence and Constrained Pilot Recommendations

**Document:** `REVIEW_RESPONSE.md`  
**Supercedes:** `ANSWERS_FOR_REVIEW.md` and `NEXT_IMPLEMENTATION_PROMPTS.md`  
**Review Reference:** [ARCHIVE_REVIEW_CORRECTIONS_AND_NEXT_PROMPTS.md](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/ARCHIVE_REVIEW_CORRECTIONS_AND_NEXT_PROMPTS.md)  
**Date:** 2026-10-09  

---

## 1. What Corrections Were Made

All nine points identified in the review have been explicitly corrected in [CORRECTED_ANSWERS.md](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/CORRECTED_ANSWERS.md):

1. **Repaired Dev Counts Corrected:** Documented exact class and format totals: Real = 6,950 JPEG + 550 PNG; Fake = **7,500 PNG and 0 JPEG**. The typo claiming 550 fake JPEGs has been retracted.
2. **Provenance Guardrails Restored:** Speculative generator/dataset names (FaceForensics++, CelebA, DiffFace, StyleGAN) are strictly labeled as unverified leads from repository adapters and user context, rather than proven facts.
3. **Training Collection Breadth Acknowledged:** Recognized that the training set includes 31,111 numeric fake files (2,753 in the R2 100K training split), disproving the claim that training was exclusively video manipulation.
4. **Unbiased Visual Inspection:** Generated and inspected separate contact sheets for all four subcollection strata (numeric real, numeric fake, video real, video fake) in [subcollection_review.md](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/subcollection_review.md).
5. **Validation Status Bounded:** Acknowledged that zero recognized filename tokens between archive `train/` and `val/` does not prove zero identity or original-derivative leakage, and that archive `val/` was previously used by the project rather than universally ignored.
6. **External Overlap Status Corrected:** Reported cross-dataset overlap as **unknown**, not zero.
7. **Metric Interpretations Constrained:** Removed claims that validation AUC > 0.90 identifies learned features or proves absence of shortcuts.
8. **Physical Reality of Encoding Acknowledged:** Retracted "PNG demosaicing artifacts" and recognized that neural networks process decoded RGB tensors, while converting JPEG to PNG preserves original compression artifacts.
9. **Explicit Verification Scopes Enforced:** Separated population listing metrics from sample-level profiling observations.

---

## 2. What the Numeric Subset Contributes

Direct inspection of the numeric subcollection in [subcollection_review.md](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/subcollection_review.md) reveals:

- **Numeric Real (162,078 in train; 38,570 in R2 100K train):** 100% JPEG, uniform 178×218 resolution, centered portraits matching CelebA-style photographic crops.
- **Numeric Fake (31,111 in train; 2,753 in R2 100K train):** 100% PNG, uniform 256×256 resolution, full-face portraits with generative artifacts.
- **Contribution:** Demonstrates that the training archive contains still-portrait face synthesis alongside video face swaps. However, within the numeric subset, **format and resolution remain 100% confounded with the target class** (Real is 100% 178×218 JPEG; Fake is 100% 256×256 PNG).

---

## 3. What Existing Per-Subcollection Predictions Show

Analysis of R2 `best_dev_predictions.csv` (14,835 samples, checkpoint `e598e1b3...`) in [subcollection_metrics.json](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/subcollection_metrics.json) establishes:

| Diagnostic Stratum | Total Samples | Real Recall | Fake Recall | Balanced Accuracy | ROC AUC | Error Concentration |
|---|---|---|---|---|---|---|
| **Overall Development** | 14,835 | 96.35% (7,176 / 7,448) | 98.09% (7,246 / 7,387) | 97.22% | 0.9960 | 413 total errors (272 FP, 141 FN) |
| **All Numeric** (Real JPG + Fake PNG) | 6,155 | 99.91% (5,758 / 5,763) | 100.0% (392 / 392) | **99.96%** | **0.9998** | Only 5 errors (all FP on JPG) |
| **All Video** (Real PNG + Fake PNG) | 8,680 | 84.15% (1,418 / 1,685) | 97.98% (6,854 / 6,995) | 91.07% | 0.9823 | 408 errors (267 FP, 141 FN) |
| **Numeric Real** (100% JPG) | 5,763 | 99.91% (5,758 / 5,763) | N/A | N/A | Undefined | Only 5 False Positives |
| **Video Real** (100% PNG) | 1,685 | **84.15%** (1,418 / 1,685) | N/A | N/A | Undefined | **267 False Positives (98.2% of all FPs!)** |
| **Numeric Fake** (100% PNG) | 392 | N/A | **100.0%** (392 / 392) | N/A | Undefined | 0 False Negatives (median prob: 0.999998) |
| **Video Fake** (100% PNG) | 6,995 | N/A | 97.98% (6,854 / 6,995) | N/A | Undefined | 141 False Negatives |

### Interpretation
1. **The aggregate score (97.22%) conceals severe failure on PNG-real:** Out of 272 false positives in development, **267 (98.2%) occur on authentic PNG face crops**.
2. **Apparent near-perfection on numeric (99.96% bacc) is driven by container separation:** Real numeric images are 100% JPEG, and fake numeric images are 100% PNG. The model easily leverages format separation.
3. **When exposed to authentic PNG faces, the model's error rate spikes by 180×** compared to authentic JPEG faces (15.85% error rate vs 0.09%).

---

## 4. Reusable Paired-Preprocessing Diagnostic Utility

Implemented [tools/diagnose_format_sensitivity.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/tools/diagnose_format_sensitivity.py) and unit tests in [tests/test_diagnose_format_sensitivity.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/tests/test_diagnose_format_sensitivity.py).

### Test Suite Results
- Ran 5 unit tests verifying:
  - PNG RGB equality control (`np.max(abs(orig - png)) == 0.0`).
  - Strict preservation of original disk files (SHA256 identical before and after).
  - Hard manifest ceiling enforcement (rejects >128 samples).
  - Deterministic evaluation transform reproducibility.
- **Result:** `Ran 5 tests in 0.145s — OK (All 5 passed)`.

### Bounded Command for User Execution
To run the diagnostic on up to 128 images with an existing checkpoint on GPU/CPU:
```powershell
python tools/diagnose_format_sensitivity.py `
  --checkpoint "F:\Discovery AI\Second Expirement 100K RGB model\best.pth" `
  --manifest "output/review/rgb_r2_diagnostic_inputs/train_profile_256.csv" `
  --output_dir "output/review/archive_investigation/format_sensitivity" `
  --device cpu `
  --workers 0
```
*(Note: `train_profile_256.csv` can be sliced to <=128 rows, or a dedicated diagnostic manifest can be supplied).*

---

## 5. Constrained Next Pilot Proposal (Unexecuted)

Based on the empirical findings, any future pilot must adhere to the following bounded principles:

1. **Do Not Extract Archive Val or Retrain Yet:**
   - Archive extraction and retraining remain paused pending resolution of source provenance.
2. **Balanced Stratification Across Observable Subcollections:**
   - If a training cohort is constructed, do not allow the dominant video-fake subset (69.9%) to drown out the numeric subset. Stratify sampling across observable strata:
     - Stratum 1: Numeric Real (JPEG, 178×218)
     - Stratum 2: Numeric Fake (PNG, 256×256)
     - Stratum 3: Video-Pattern Real (PNG, variable)
     - Stratum 4: Video-Pattern Fake (PNG, variable)
3. **Format Balancing Intervention:**
   - Because all fake training images are PNG, while real images are predominantly JPEG, models trained on raw containers inherently learn format shortcuts. Any new pilot must evaluate format harmonization (e.g., uniform pre-encoding or paired format augmentation).
4. **No Premature Claim of Source-Held-Out Protocol:**
   - Without verified source annotations, a true source-held-out split is impossible. All evaluation on internal partitions must be reported as in-distribution stratified development.
5. **External Role Preserved:**
   - `F:\val to be deleted 2` remains strictly an external development diagnostic set.

---

## 6. Key Information Required from User

To advance from diagnostic observations to a defensible experimental plan, the following information is needed:
1. **The preparation/download script** used to assemble `Preapred Dataset.zip`.
2. **Original benchmark names** for the numeric fake subset (`0.png`–`31111.png`) and numeric real subset (`000001.jpg`–`199999.jpg`).
3. **The intended thesis task:** Video face-swap detection (where video face crops are the target domain), or general image synthesis detection (requiring multi-generator GAN/diffusion training data).
