# Controlled RGB Data-Mixture Pilot: Implementation & Verification Report

**Date:** October 9, 2026  
**Repository:** `G:\Master's Of Science Computer Engineering\thesis deepfake detection\Thesis Revisions SP\First revesion\Novel Deepfake Detection`  
**Protocol File:** `protocols/controlled_rgb_mixture_protocol.json`  
**Colab Workflow Guide:** `COLAB_RGB_MIXTURE_PILOT.md`  

---

## 1. Executive Summary & Review Qualifications

Following the bounded format diagnostic review, universal JPEG conversion was rejected as a standalone solution for external failure. The diagnostic confirmed that external real recall was essentially near chance (1/32 original, 2/32 at JPEG75), while training real recall remained ~90% (30/32 original, 28/32 at JPEG75).

### Key Qualifications Adopted
1. **External Development vs. Holdout**: `F:\val to be deleted 2` is classified as repeatedly inspected **external development**, not an untouched holdout.
2. **Retraction of Speculative Claims**: Speculative hypotheses regarding demographics, facial geometry, lighting, or specific frequency bands have been formally retracted in `output/review/archive_investigation/CORRECTED_DIAGNOSTIC_INTERPRETATION_NOTE.md`. No demographic metadata was analyzed.
3. **Existing JPEG Augmentation in R2**: R2 already utilized the `rgb_v1` recipe with online JPEG augmentation (probability 0.5, qualities 50–95) and Gaussian blur (probability 0.5). JPEG augmentation is therefore retained unchanged in this pilot, not treated as a new intervention.
4. **Engineering Hypothesis**: This pilot tests the hypothesis that rebalancing observable training strata (equalizing numeric and video-pattern representations within real and fake classes) improves within-distribution balance and robustness. It makes **no guaranteed claim of out-of-distribution generalization or 85% accuracy**. Numeric real remains predominantly JPEG while numeric fake remains predominantly PNG; format confounding is not eliminated by data re-balancing alone.

---

## 2. Protocol Freeze (Prompt 1)

The evaluation and training configuration is frozen in `protocols/controlled_rgb_mixture_protocol.json` and documented in `protocols/CONTROLLED_RGB_MIXTURE_PROTOCOL.md`:

| Parameter | Specification | Rationale |
|---|---|---|
| **Architecture** | `Wang2020_128` | ResNet-50 backbone with 128-d projection head, exact parity with R2 |
| **Initialization** | ImageNet pretrained | Clean start; no fine-tuning from contaminated R2 checkpoints |
| **Fine-tuning Policy** | `layer4_and_head` | Backbone lower layers frozen; layer4 and head trained with LR mult 0.1 |
| **Batch Normalization** | `frozen` (eval mode) | Eliminates batch statistics shift across small batches |
| **Augmentation** | `rgb_v1` | Gaussian blur prob 0.5 (sig 0.0–3.0), JPEG prob 0.5 (qual 50, 60, 70, 80, 90, 95) |
| **Training Budget** | Exactly 5 epochs | Fixed exposure; early stopping explicitly disabled (`--no-early_stopping`) |
| **Optimizer** | Adam (`lr=0.0001`, `lr_policy=cosine`) | Matches R2 optimizer and schedule |
| **Batching** | Batch size 32, grad accum 2 | Effective batch size 64 |
| **Checkpoint Selection** | Monitored development AUC | Evaluated on shared dev; all 4 stratum recalls reported alongside |

---

## 3. Connected Grouping, Partitioning, and Manifest Build (Prompt 2)

Derived manifests were generated from the audited `clean_parent` metadata (794,243 rows, SHA-256: `6a968a75432490941467201fa7e1e777dd9dbfe9415a26e5bd821b842833d75b`).

### Grouping & Isolation Mechanics
- **Single Component Formation**: Groups connected once across `group_id`, `source_video_id`, `original_id`, `identity_id`, `sha256`, path, and filename family (`vid_<64hex>`).
- **Connected Components Count**: Exactly **213,127 disjoint components** identified.
- **Group-Level Partitioning**: Components assigned to `dev` (fraction < 0.15) or `train` based on deterministic hash of component ID.
- **Deduplication & Caps**: Duplicate content hashes dropped; at most **8 images per connected component per class** retained.
- **Token Leak Check**: **0 tokens, hashes, paths, or families** cross between train and dev in either arm.

### Capacity Audit & Target Quotas

The post-grouping capacity audit verified that all requested quotas are met without shortfall:

| Stratum | Post-Grouping Dev Pool | Shared Dev Target | Post-Grouping Train Pool | Arm A Train Target (R2-like) | Arm B Train Target (Equal) |
|---|---:|---:|---:|---:|---:|
| **Numeric Real** | 24,347 | 500 | 137,645 | 7,714 | 5,000 |
| **Video-Pattern Real** | 1,830 | 500 | 11,270 | 2,286 | 5,000 |
| **Numeric Fake** | 4,623 | 500 | 26,488 | 551 | 5,000 |
| **Video-Pattern Fake** | 21,706 | 500 | 123,894 | 9,449 | 5,000 |
| **Total per Cohort** | **52,506** | **2,000** | **299,297** | **20,000** | **20,000** |

### Derived Manifest Files & Cryptographic Gates

All manifests and verification sidecars have been written to `output/controlled_mixture/manifests/`:

1. **Arm A Manifest**: `output/controlled_mixture/manifests/arm_a/selected_manifest.csv`
   - Total rows: 22,000 (20,000 train + 2,000 dev)
   - SHA-256: `21922d71b8dcc77d3d1fa896e8d2ba69a69c6fffe9fd9392fbd524f15dfc8bea`
   - Gate status: `verified: true`, `hashes_verified: true`
2. **Arm B Manifest**: `output/controlled_mixture/manifests/arm_b/selected_manifest.csv`
   - Total rows: 22,000 (20,000 train + 2,000 dev)
   - SHA-256: `1b316052b2fcae5d713b344dbd345afad77b42e45869aa31c21964a23f94765d`
   - Gate status: `verified: true`, `hashes_verified: true`
3. **Shared Dev Manifest**: `output/controlled_mixture/manifests/shared_dev/selected_manifest.csv`
   - Total rows: 2,000 (exactly 500 per stratum, 100% identical between Arm A and Arm B)
   - SHA-256: `1e16a9dbca55f39a7ac42ca51d3de80b8540da7158e57554ee08478bcff6f528`
   - Gate status: `verified: true`, `hashes_verified: true`
4. **Union Member Lists for Colab Selective Extraction**:
   - `output/controlled_mixture/manifests/union_members_list.txt` (full paths, 29,163 entries)
   - `output/controlled_mixture/manifests/union_archive_members.txt` (archive-relative paths, 29,163 entries)

---

## 4. Per-Stratum Development Reporting Tool (Prompt 3)

Implemented in `tools/evaluate_mixture_predictions.py`:
- **Strict Input Validation**: Validates unique sample IDs, labels in {0, 1}, finite probabilities in [0.0, 1.0], logits, and homogeneous checkpoint hashes.
- **Manifest Cross-Referencing**: Ensures 1-to-1 match with the 2,000 shared dev samples.
- **Stratum-Level Diagnostics**:
  - Reports denominator, errors, real/fake recall, and probability/logit distribution quantiles.
  - Explicitly leaves ROC AUC as `"undefined_single_class"` for single-class strata (no fabricated 0.0 or 1.0).
- **Two-Class Subcollections**: Reports tie-aware ROC AUC and balanced accuracy on `numeric_both` and `video_both`.
- **Diagnostic Aggregate**: Reports overall balanced accuracy and `min_stratum_recall` (minimum of the 4 stratum recalls).
- **Threshold Policy**:
  - `fixed_0_5`: Reference fixed 0.5 threshold.
  - `dev_selected`: Threshold tuned strictly on development predictions (maximizing Youden's J statistic) and bound to checkpoint metadata. Never tuned on external labels.
- **Paired Comparative Analysis**:
  - 2x2 contingency matrix (both correct, A right / B wrong, A wrong / B right, both wrong).
  - McNemar's test with continuity correction and exact p-value.
  - Per-stratum paired net error shifts ($B - A$).

---

## 5. Colab Single-GPU Workflow Guide (Prompt 4)

Delivered in `COLAB_RGB_MIXTURE_PILOT.md`:
- Self-contained, executable notebook cells.
- 7-Zip selective extraction using `union_archive_members.txt` to extract only 29,163 images (~1.0 GiB) rather than the 33.8 GiB archive.
- Path/gate verification preflight check.
- Bounded throughput benchmark (`tools/benchmark_training_throughput.py`).
- Precise training commands matching repo argument parser.
- Drive checkpoint sync and paired evaluation execution.

---

## 6. Diagnostic Utility Maintenance Summary

Prior to this pilot, maintenance was applied to `tools/diagnose_format_sensitivity.py`:
1. **Non-Empty Directory Rejection**: Fails closed if output directory exists and is non-empty.
2. **Explicit Cohort Requirement**: Requires explicit cohort metadata column in the input manifest.
3. **Input Parameter Validation**: Rejects `batch_size <= 0`.
4. **Model Output Validation**: Validates prediction shape, finite numbers, and length against input records.
5. **Transform Tensor Parity Test**: Added exact test in `tests/test_diagnose_format_sensitivity.py` verifying tensor parity against production inference pipeline.

---

## 7. Verification Test Suite Execution Results

All 28 automated tests in the test suite passed cleanly with 0 failures and 0 errors:

```
Ran 28 tests in 0.814s

OK
- tests/test_diagnose_format_sensitivity.py: 10 tests passed (tensor parity, cohort, batch_size, non-empty dir, etc.)
- tests/test_controlled_mixture_manifests.py: 8 tests passed (determinism, caps, hash grouping, shared dev identity, shortfalls, split rejection)
- tests/test_evaluate_mixture_predictions.py: 6 tests passed (schema validation, stratum classification, AUC edge cases, McNemar test, pipeline)
- tests/test_rgb_family_repair.py: 1 test passed (family repair regression)
- tests/test_rgb_review_final_repairs.py: 2 tests passed (relationship & gate verification)
- tests/test_calibration_and_source_readiness.py: 1 test passed (source readiness regression)
```

---

## 8. Changed & Created Files Inventory

| File Path | Description |
|---|---|
| `protocols/controlled_rgb_mixture_protocol.json` | Versioned machine-readable protocol specification |
| `protocols/CONTROLLED_RGB_MIXTURE_PROTOCOL.md` | Human-readable protocol description |
| `output/review/archive_investigation/CORRECTED_DIAGNOSTIC_INTERPRETATION_NOTE.md` | Retraction of overclaims & clarification of external development status |
| `tools/diagnose_format_sensitivity.py` | Maintained diagnostic tool with fail-closed validation |
| `tests/test_diagnose_format_sensitivity.py` | 10 unit tests for format diagnostic utility |
| `tools/build_controlled_mixture_manifests.py` | Grouped manifest builder with cap=8 and zero leak |
| `tests/test_controlled_mixture_manifests.py` | 8 unit tests for manifest builder |
| `output/controlled_mixture/manifests/arm_a/` | Arm A manifest (22K) & verified cryptographic gate |
| `output/controlled_mixture/manifests/arm_b/` | Arm B manifest (22K) & verified cryptographic gate |
| `output/controlled_mixture/manifests/shared_dev/` | Shared Dev manifest (2K) & verified cryptographic gate |
| `output/controlled_mixture/manifests/union_members_list.txt` | 29,163 full paths for selective extraction |
| `output/controlled_mixture/manifests/union_archive_members.txt` | 29,163 archive-relative paths for 7-Zip selective extraction |
| `output/controlled_mixture/manifests/mixture_manifest_report.json` | Machine-readable manifest audit & capacity report |
| `tools/evaluate_mixture_predictions.py` | Per-stratum development evaluation & paired comparison tool |
| `tests/test_evaluate_mixture_predictions.py` | 6 unit tests for mixture prediction evaluator |
| `COLAB_RGB_MIXTURE_PILOT.md` | Step-by-step Colab GPU execution guide |
| `output/review/archive_investigation/MIXTURE_PILOT_IMPLEMENTATION_REPORT.md` | This report |

---

## 9. User Action Steps in Colab

1. Open a Google Colab notebook with GPU runtime (T4 or higher).
2. Follow the cell-by-cell instructions in `COLAB_RGB_MIXTURE_PILOT.md`.
3. Train **Arm A** (approx. 20K train samples, ~30–45 min on T4).
4. Train **Arm B** (approx. 20K train samples, ~30–45 min on T4).
5. Run Step 8 in `COLAB_RGB_MIXTURE_PILOT.md` to produce the comparative report.
6. If Arm B improves `min_stratum_recall` without regressing overall AUC, confirm with additional seeds before making generalization claims. If both arms fail, provenance and genuine dataset diversity must be prioritized over expanding sample volume to 500K/1M.
