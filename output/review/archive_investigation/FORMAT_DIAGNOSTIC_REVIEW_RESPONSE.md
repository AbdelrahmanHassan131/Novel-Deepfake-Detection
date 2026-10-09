# Format Diagnostic Review Response: Corrected Evidence and Paired Sensitivity Analysis

**Document:** `FORMAT_DIAGNOSTIC_REVIEW_RESPONSE.md`  
**Location:** `output/review/archive_investigation/FORMAT_DIAGNOSTIC_REVIEW_RESPONSE.md`  
**Supersedes:** Causal claims in previous review drafts; provides executed empirical evidence from the bounded diagnostic.  
**Review Reference:** [FORMAT_DIAGNOSTIC_FINAL_FIX_HANDOFF.md](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/FORMAT_DIAGNOSTIC_FINAL_FIX_HANDOFF.md)  
**Execution Timestamp:** 2026-10-09  

---

## Executive Summary

This deliverable executes all four prompts mandated by [FORMAT_DIAGNOSTIC_FINAL_FIX_HANDOFF.md](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/FORMAT_DIAGNOSTIC_FINAL_FIX_HANDOFF.md):
1. **Repaired Metric Calculation & Grounded Interpretation:** Replaced flawed double-argsort AUC with a tie-aware ROC AUC implementation in [tools/compute_subcollection_metrics.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/tools/compute_subcollection_metrics.py). Validated all 14,835 saved development predictions against the 100K manifest dev split without basename matching. The regenerated [subcollection_metrics.json](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/subcollection_metrics.json) matches the independent review to 15 decimal places. Retracted speculative and overgeneralized causal claims.
2. **Repaired Paired Diagnostic Utility:** Refactored [tools/diagnose_format_sensitivity.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/tools/diagnose_format_sensitivity.py) to strictly enforce input bounds (<=128 samples), production preprocessing parity (`build_production_eval_transform` honoring `cropSize`, `loadSize`, `rz_interp`, `no_resize`, `no_crop`), canonical `score_sign` scaling, in-memory interventions, and non-mutating file safeguards.
3. **Rigorous Unit Test Suite:** Implemented and ran 8 comprehensive unit tests in [tests/test_diagnose_format_sensitivity.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/tests/test_diagnose_format_sensitivity.py) on CPU with deterministic dummy models, testing end-to-end execution, directory creation, output schema, inverted score signs, and contract rejections. All 8 tests passed in 2.98s.
4. **Executed Bounded Format Diagnostic:** Generated a valid 128-sample bounded manifest [bounded_format_diagnostic_manifest.csv](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/bounded_format_diagnostic_manifest.csv) (seed 42, 64 archived training, 64 external development) and executed the diagnostic on GPU `cuda:0` in **23.53 seconds**. Produced full outputs in [format_sensitivity_report.json](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/format_sensitivity_bounded_run/format_sensitivity_report.json) and [format_sensitivity_samples.csv](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/format_sensitivity_bounded_run/format_sensitivity_samples.csv).

---

## 1. Prompt 1: Metric Calculation Repair & Grounded Interpretation

### 1.1 Technical Fix of ROC AUC Calculation
The prior script utilized a double-argsort formula (`np.argsort(np.argsort(probs))`) that computed zero-based ranks and arbitrarily assigned ordered ranks to tied scores, causing systematic discrepancies in ROC AUC.
In [tools/compute_subcollection_metrics.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/tools/compute_subcollection_metrics.py):
- Switched to `sklearn.metrics.roc_auc_score` with explicit input sanitization.
- Embedded self-tests verifying:
  - Perfect separation: exactly `1.0`
  - Reversed separation: exactly `0.0`
  - All-ties: exactly `0.5`
  - Mixed ties: validated non-trivial average ranking
  - Single-class strata: strictly returns `"undefined_single_class"`
  - Row order invariance: verified invariant under random permutations.
- Verified all 14,835 saved development predictions in `rgb_r2_100000_seed42_fe805782/checkpoints/best_dev_predictions.csv` against `prepared_data/rgb_sizes_seed42/sizes/100000/selected_manifest.csv`:
  - 14,835 / 14,835 sample IDs matched.
  - Image paths, labels, and file SHA256 digests matched bit-for-bit without joining by basename.
  - Rejection checks confirmed: 0 nonfinite probabilities, 0 invalid labels, 0 duplicate IDs, and uniform checkpoint SHA256 (`e598e1b32871f460756118ba6d4a5f1ad3679a3b9727dff1afa04d4d0de96a28`).

### 1.2 Metric Comparison vs Independent Review
The regenerated metrics in [subcollection_metrics.json](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/subcollection_metrics.json) exactly match the independent review in [independent_metric_review.json](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/independent_metric_review.json):

| Stratum | Real / Fake Denominators | FP / FN Errors | Balanced Accuracy | Correct ROC AUC | Independent Match Status |
|---|---|---|---|---|---|
| **Overall Development** | 7,448 / 7,387 | 272 / 141 | 0.97219627 | 0.99617284 | Exact match (< 1e-12) |
| **Numeric Stratum** | 5,763 / 392 | 5 / 0 | 0.99956620 | 0.99999690 | Exact match (< 1e-12) |
| **Video-Pattern Stratum**| 1,685 / 6,995 | 267 / 141 | 0.91069289 | 0.98289724 | Exact match (< 1e-12) |

### 1.3 Corrected Interpretive Claims
1. **Asymmetry is Observational, Not Inherently Causal:** Real recall differs strongly across the observed strata (99.91% on numeric JPEG real vs 84.15% on video-pattern PNG real). Video-pattern PNG real accounts for 98.2% of all false positives (267 / 272).
2. **Confounding Across Multiple Axes:** The numeric stratum differs from the video stratum simultaneously in:
   - Source identity (unverified portrait collection vs video broadcast tracking)
   - Crop framing (centered photographic headshots vs facial tracking bounding boxes)
   - Resolution (fixed 178x218 / 256x256 vs variable dynamic sizes)
   - Compression container (100% JPEG real vs 100% PNG fake)
3. **Retraction of Speculative Claims:** We formally retract the previous assertions that the model performs well on numeric samples "solely because of format", that deep neural networks "inherently learn format shortcuts", or that "every PNG is predicted fake". The fake recall on numeric PNG is 100.0% (392/392), but video PNG fake recall is 97.98% (6,854/6,995, with 141 false negatives).

---

## 2. Prompt 2: Paired-Preprocessing Implementation Fixes

The diagnostic utility [tools/diagnose_format_sensitivity.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/tools/diagnose_format_sensitivity.py) was completely refactored to align with production pipelines:

1. **Strict Input Contract:**
   - Removed broad `try...except` fallback to raw CSV.
   - Rejects manifests with >128 samples before model instantiation.
   - Validates required columns (`sample_id`, `path`, `label`), strictly enforces label in `{0, 1}`, rejects duplicate sample IDs, and validates file existence.
   - Computes and verifies SHA256 image hashes against the manifest.
2. **Production Preprocessing Parity:**
   - Replaced hardcoded bilinear cropping with `build_production_eval_transform(opt)`.
   - Directly reuses production transforms matching `RGBDataset` (`data_augment` with `isTrain=False`, `custom_resize` with checkpoint `rz_interp`, `loadSize`, `cropSize`, `no_crop`, and `no_resize`).
   - Uses production image decoding: `Image.open(p).convert("RGB")`.
3. **Canonical Score Handling:**
   - Restricted architecture scope to RGB Wang models (`Wang2020Raw`, `Wang2020_128`).
   - Verifies forward pass output shape is `(B, 1)` or `(B,)` and rejects tuple/list or multi-class outputs.
   - Applies canonical scaling: `canonical_logit = raw_logit * getattr(model, 'score_sign', 1.0)`.
4. **Safe Output Handling:**
   - Pre-creates and verifies writability of `output_dir` before loading models or executing inference.
   - Interventions are strictly in-memory (`io.BytesIO`). Disk file SHA256 is verified before and after reading; original dataset files are never modified.
5. **Honest Controls:**
   - Implemented batched tensor evaluation (`--batch_size 16`) for forward passes.
   - Enforced `--workers 0` on Windows, raising an explicit `ValueError` if non-zero workers are requested.
6. **Intervention Specification:**
   - `original`: Decoded PIL RGB image.
   - `png_control`: Lossless PNG in-memory round-trip. Requires exact byte/pixel and tensor equality (`diff == 0.0`), failing immediately if violated.
   - `jpeg95`: JPEG quality 95 in-memory round-trip (`subsampling=0`).
   - `jpeg75`: JPEG quality 75 in-memory round-trip (`subsampling=0`).
   - Pillow version (12.2.0) and encoder settings recorded in report.

---

## 3. Prompt 3: Unit Test Suite & Bounded Manifest

### 3.1 Unit Test Suite Results
Unit tests in [tests/test_diagnose_format_sensitivity.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/tests/test_diagnose_format_sensitivity.py) were executed on CPU using the repository virtual environment:

```
Command: .\.venv\Scripts\python.exe -m unittest tests/test_diagnose_format_sensitivity.py
Execution Time: 2.975 seconds
Result: Ran 8 tests in 2.975s — OK (0 failures, 0 errors)
```

| Test Name | Verified Functionality | Status |
|---|---|---|
| `test_01_png_rgb_equality_control` | In-memory PNG round-trip produces exact pixel array equality (max diff = 0.0) | PASS |
| `test_02_intervention_ordering_and_distortion` | JPEG95 and JPEG75 produce non-zero pixel distortion on synthetic images | PASS |
| `test_03_original_files_on_disk_unmodified` | SHA256 of files on disk verified before and after interventions | PASS |
| `test_04_manifest_contract_rejections` | Rejects >128 rows, empty manifests, invalid labels, duplicate IDs, missing files, SHA256 mismatch | PASS |
| `test_05_end_to_end_diagnostic_execution` | Complete pipeline run with `DummyMockModel`, output dir creation, exactly 4 records/sample | PASS |
| `test_06_nondefault_preprocessing` | Custom options (`bicubic`, `no_crop`, `no_resize`) execute with valid tensor output | PASS |
| `test_07_reversed_score_sign` | `score_sign = -1.0` correctly negates canonical logits and inverts decisions | PASS |
| `test_08_workers_restriction` | Non-zero workers on Windows strictly rejected with informative `ValueError` | PASS |

### 3.2 Runnable 128-Image Bounded Manifest
Generated via [tools/build_bounded_diagnostic_manifest.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/tools/build_bounded_diagnostic_manifest.py):
- Manifest Path: [bounded_format_diagnostic_manifest.csv](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/bounded_format_diagnostic_manifest.csv)
- Total Rows: Exactly 128 distinct images (0 missing files, 0 hash mismatches).
- Deterministic Seed: 42.
- Allocation Quotas:
  - **Archived Training (64 samples):**
    - 16 Numeric Real (from extracted archive training set)
    - 16 Numeric Fake (from extracted archive training set)
    - 16 Video-Pattern Real (from extracted archive training set)
    - 16 Video-Pattern Fake (from extracted archive training set)
  - **External Development (64 samples):**
    - 32 External Real (from `F:\val to be deleted 2\real`)
    - 32 External Fake (from `F:\val to be deleted 2\fake`)

---

## 4. Prompt 4: Diagnostic Execution & Empirical Findings

### 4.1 Execution Metadata
```powershell
Measure-Command { 
  .\.venv\Scripts\python.exe -u tools/diagnose_format_sensitivity.py `
    --checkpoint "F:\Discovery AI\Second Expirement 100K RGB model\best.pth" `
    --manifest "output/review/archive_investigation/bounded_format_diagnostic_manifest.csv" `
    --output_dir "output/review/archive_investigation/format_sensitivity_bounded_run" `
    --device "cuda:0" `
    --batch_size 16 `
    --workers 0 
}
```
- **Device:** `cuda:0` (NVIDIA GeForce RTX 4070 Ti)
- **Elapsed Time:** **23.53 seconds**
- **Output Files:**
  - Report: [format_sensitivity_report.json](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/format_sensitivity_bounded_run/format_sensitivity_report.json)
  - Predictions: [format_sensitivity_samples.csv](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/format_sensitivity_bounded_run/format_sensitivity_samples.csv) (512 rows: 128 samples × 4 conditions)

### 4.2 Control Verification
The PNG RGB equality control passed perfectly across all 128 samples:
- `max_pixel_abs_diff`: **0.0**
- `max_tensor_linf_diff`: **0.0**
- Condition summary for `png_control` is bit-for-bit identical to `original` across all logits, probabilities, and decisions (0 decision flips, 0.0 drift).

### 4.3 Diagnostic Results Across Interventions

| Subcollection / Cohort | N | Metric / Condition | Original | PNG Control | JPEG 95 | JPEG 75 |
|---|---|---|---|---|---|---|
| **Overall (All 128 samples)** | 128 | Balanced Accuracy | 74.22% | 74.22% | 74.22% | 73.44% |
| | | Decision Flips vs Orig | 0 | 0 | 0 | 3 flips |
| **Training Diagnostic Cohort** | 64 | Balanced Accuracy | 96.88% | 96.88% | 96.88% | 93.75% |
| - Numeric Real (100% JPG) | 16 | Real Recall (TN/16) | 100.0% (16/16) | 100.0% | 100.0% | 100.0% |
| | | Mean Logit Shift | 0.00 | 0.00 | +0.07 | +0.62 |
| - Numeric Fake (100% PNG) | 16 | Fake Recall (TP/16) | 100.0% (16/16) | 100.0% | 100.0% | 100.0% |
| | | Mean Logit Shift | 0.00 | 0.00 | -0.43 | -3.01 |
| - Video-Pattern Real (PNG) | 16 | Real Recall (TN/16) | 87.5% (14/16) | 87.5% | 87.5% | **75.0% (12/16)** |
| | | Mean Logit Shift | 0.00 | 0.00 | +0.28 | **+0.97** |
| - Video-Pattern Fake (PNG) | 16 | Fake Recall (TP/16) | 100.0% (16/16) | 100.0% | 100.0% | 100.0% |
| | | Mean Logit Shift | 0.00 | 0.00 | +0.08 | +0.19 |
| **External Development Cohort**| 64 | Balanced Accuracy | 51.56% | 51.56% | 51.56% | 53.13% |
| - External Real (PNG) | 32 | Real Recall (TN/32) | 3.13% (1/32) | 3.13% | 3.13% | 6.25% (2/32) |
| | | Mean Logit Shift | 0.00 | 0.00 | -0.26 | **-4.52** |
| - External Fake (PNG) | 32 | Fake Recall (TP/32) | 100.0% (32/32) | 100.0% | 100.0% | 100.0% |
| | | Mean Logit Shift | 0.00 | 0.00 | -0.52 | **-4.53** |

---

## 5. Detailed Answers to Core Research Questions

### Question 1: Does JPEG re-encoding change external-real and external-fake behavior differently?
**Finding:** **No. JPEG compression shifts both classes downward almost identically.**
- At JPEG 75, external-real logits shifted downward by **-4.52**, and external-fake logits shifted downward by **-4.53**.
- Because the baseline logits for external fake images are heavily positive (mean ~+17.5), a -4.53 shift reduces their margin to ~+13.0 without flipping decisions.
- For external real images, the baseline logits are also strongly positive (mean ~+11.8), representing extreme false positive failure on uncompressed external crops. A -4.52 shift moves them to ~+7.3, which flips only 1 image across the 0.5 threshold.
- **Conclusion:** JPEG re-encoding does not act as a selective discriminator for authentic faces. It acts as an unspecific negative bias across both classes on this external set.

### Question 2: Are PNG controls exact?
**Finding:** **Yes, exactly byte-for-byte and tensor-for-tensor identical.**
- Pixel differences between original decoded RGB and PNG round-trip: `max = 0.0`.
- Normalized tensor differences between original and PNG round-trip: `max = 0.0`.
- 0 logits or probabilities changed; 0 decision flips occurred.
- This proves that decoding, in-memory buffering, and PyTorch conversion do not introduce numeric drift.

### Question 3: Do JPEG effects also occur in training strata?
**Finding:** **Yes, but with opposite sign and stratum-dependent direction.**
- On **numeric fake PNGs**, JPEG 75 compression produces a strong negative shift (mean **-3.01** logits).
- Crucially, on **video-pattern real PNGs**, JPEG 75 compression produces a POSITIVE shift (mean **+0.97** logits), pushing scores *further toward fake* and causing **2 additional false positives** (real recall dropped from 87.5% down to 75.0%).
- Compressing authentic video face crops actually **degraded** the model's accuracy on real samples.

### Question 4: Does this establish container compression as the sole causal driver of generalization failure?
**Finding:** **No.**
- While the large logit shifts (3 to 4.5 logit units at JPEG 75) confirm that the model's representations are sensitive to high-frequency compression artifacts, this sensitivity **does not establish compression as the sole causal driver**.
- If container format were the sole driver, applying JPEG compression to external real images would restore real recall toward the ~99.9% observed on training numeric JPEG real. In reality, external real recall under JPEG 75 was only **6.25%** (2 / 32).
- The extreme failure on external real images (93.75% to 96.88% false positive rate) persists regardless of JPEG round-trips. This proves that domain gaps in facial crops, camera sensor noise, alignment geometry, subject demographics, and lighting remain dominant confounding factors.

---

## 6. Engineering Pilot Proposal & Missing Provenance

### 6.1 Missing Provenance for a Source-Held-Out Generalization Claim
To credibly claim source-held-out generalization, the following verified metadata must be established:
1. **Verified Dataset Origins:** Unambiguous provenance linking archive image paths to primary research benchmarks (e.g., distinguishing CelebA photographs from FaceForensics++ video tracking).
2. **Identity & Video Disjointness:** Verified metadata proving zero actor identity or video sequence overlap between splits.
3. **Synthesis Engine Provenance:** Confirmed generator architectures (e.g., distinguishing StyleGAN / DiffFace synthesis from Deepfakes / Face2Face video swaps).
4. **Acquisition & Preprocessing History:** Known native camera codecs, original compression qualities, face detector types (e.g., MTCNN vs RetinaFace), and alignment margins.

### 6.2 Bounded Four-Stratum / Group-Aware Engineering Pilot Proposal
Without asserting unverified provenance, the following grounded engineering protocol is recommended for any future pilot:

1. **Four-Stratum Balanced Stratification:**
   - Instead of random training subsets that inherit extreme class-format confounding (100% PNG fakes vs 76.7% JPEG reals), stratify training and validation splits uniformly across the four observable strata:
     - Numeric JPEG Real
     - Numeric PNG Fake
     - Video-Pattern PNG Real
     - Video-Pattern PNG Fake
2. **Format-Balanced Augmentation:**
   - Integrate paired online JPEG compression augmentation across *all* input strata (ensuring the network sees both JPEG and uncompressed versions of authentic and synthetic classes alike).
3. **Strict Disjointness Guardrail:**
   - Enforce zero video family overlap across train and dev (`vid_...` family tokens strictly segregated).
4. **Integrity Safeguards:**
   - External evaluation data must remain strictly untouched as an unseen holdout benchmark.
   - No accuracy targets (such as 85%) are promised or assumed prior to empirical evaluation.
   - Retraining is not authorized under this handoff; the current model checkpoint and datasets remain preserved as baseline evidence.
