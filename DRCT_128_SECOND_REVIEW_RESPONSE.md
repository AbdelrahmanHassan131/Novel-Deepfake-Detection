# DRCT-128 Second Code Review Response & Verification Report

**Date:** 2026-10-14  
**Reference Document:** `DRCT_128_SECOND_REVIEW_FIXES.md`  
**Status:** All findings (S1 through S9) implemented, integrated, and verified with dedicated test suites.

---

## 1. Executive Summary & Contract Preservation

This response document provides an item-by-item resolution of all nine findings (S1 through S9) raised in `DRCT_128_SECOND_REVIEW_FIXES.md`.

All architectural and lifecycle contracts have been preserved without compromise:
1. **Binary Output Contract:** The model produces a single raw scalar logit $[B, 1]$ where 0 denotes real and 1 denotes fake, trained via `BCEWithLogitsLoss`.
2. **Penultimate 128D Learned Feature:** The penultimate learned representation is a 128-dimensional embedding ($[B, 128]$) extracted through `forward_features(x)`. This embedding feeds both the standalone binary classification head and the downstream Multi-Head Attention (`MHA_128`) fusion module.
3. **Contrastive Objective & Reducer Parity:** Multi-task loss $(1 - \alpha)\mathcal{L}_{\text{BCE}} + \alpha\mathcal{L}_{\text{contrastive}}$ with native pairwise Euclidean distance calculation preserving exact 0.0 distance for identical vectors and strictly reducing over non-zero violation pairs.
4. **Offline Isolation & Safety:** Local unit tests run on CPU with tiny fixtures, dummy or mocked encoders, and zero external downloads. Live GPU training, diffusion reconstruction, and official weight loading remain properly documented for Google Colab execution.

---

## 2. Item-by-Item Resolution Matrix (S1 through S9)

### S1 — P1: Make Preparation Output Consumable by Actual Training Entry Point
- **Root Cause:** Directory inventory assigned split names `train` and `val`, whereas the training entry point defaults to `val_manifest_split=dev`. Preparation printed an audit without writing the manifest-digest-bound `.verified.json` sidecar demanded by `train.py`'s preflight gate. Furthermore, single-source pilots failed source-overlap checks because `--allow_source_overlap` was not passed in the documented training commands.
- **Implemented Fixes:**
  - In [data/drct_manifest.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/data/drct_manifest.py) and [data/drct_preparation.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/data/drct_preparation.py): Standardized default validation split naming to `dev`.
  - Added `generate_gate_sidecar(manifest_path)`: Computes the manifest SHA256 digest and writes `<manifest>.verified.json` with verification metadata (`verified=True`, `manifest_sha256`, timestamp, record count).
  - Updated `train.py` preflight check to verify that `<manifest>.verified.json` exists, is valid JSON, has `verified=True`, and matches the manifest's current SHA256 digest exactly.
  - Documented training commands in [COLAB_DRCT_128_TRAINING.md](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/COLAB_DRCT_128_TRAINING.md) and [TRAINING_COMMANDS.md](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/TRAINING_COMMANDS.md) explicitly pass `--allow_source_overlap` and `--val_manifest_split dev` for same-source pilots.
- **Verification Tests:**
  - `TestS1PreparationGateAndTrainingEntry::test_s1_end_to_end_preparation_to_training_gate` in [tests/test_drct_second_review_fixes.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/tests/test_drct_second_review_fixes.py): Runs `inventory_dataset_directory`, `generate_gate_sidecar`, and a 1-epoch subprocess `train.py` run on CPU creating valid `best.pth`.
  - `TestS1PreparationGateAndTrainingEntry::test_s1_gate_digest_mismatch_fails_preflight`: Tampering with manifest bytes causes `train.py` preflight check to fail closed.
  - `TestS1PreparationGateAndTrainingEntry::test_s1_full_drct_alpha_without_reconstructions_rejected`: Training with `--drct_alpha 0.3` without paired reconstructions is rejected by `train.py`.

---

### S2 — P1: Provide a Working Pretrained CLIP Initialization Path
- **Root Cause:** Training defaulted to `pretrained=True`, but `--allow_download` was not wired through config or argument parser, and no `--drct_backbone_weights` path was accepted. Furthermore, `models/drct/backbone.py` contained a shadowed second `CLIPViTBackbone` definition, and `train.py` had an argument conflict on `--backbone_weights`.
- **Implemented Fixes:**
  - Removed duplicate shadowed `CLIPViTBackbone` class in [models/drct/backbone.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/models/drct/backbone.py).
  - Added `--drct_allow_download` (flag) and `--drct_backbone_weights` (path string) across [config/defaults.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/config/defaults.py), [config/configuration.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/config/configuration.py), [config/compatibility.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/config/compatibility.py), [models/drct/trainer.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/models/drct/trainer.py), and [tools/benchmark_drct.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/tools/benchmark_drct.py).
  - Removed duplicate `--backbone_weights` alias on line 97 of [train.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/train.py) that conflicted with Wang backbone options.
  - In [tools/benchmark_drct.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/tools/benchmark_drct.py): `parse_args(args=None)` now accepts injected argument lists, passes backbone weights and download settings, and reports backbone weights provenance in benchmark metrics.
- **Verification Tests:**
  - `TestS2CLIPPretrainedInitialization::test_s2_local_cached_weights_loads_without_download` in [tests/test_drct_second_review_fixes.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/tests/test_drct_second_review_fixes.py).
  - `TestS2CLIPPretrainedInitialization::test_s2_pretrained_true_with_allow_download_false_raises`: Fails safely when download is prohibited and no local weights are supplied.
  - `TestS2CLIPPretrainedInitialization::test_s2_config_and_benchmark_roundtrip_includes_backbone_settings`: Verifies argument roundtrip and benchmark option propagation.

---

### S3 — P1: Stop Saved Training Manifests from Overriding Evaluation Data
- **Root Cause:** When restoring a checkpoint during evaluation, `CheckpointLoader` cleared `opt.manifest` but left `opt.val_manifest` and `opt.drct_manifest_path`. Consequently, `DRCTDataset` fell back to the old training manifest paths during evaluation, causing folder evaluations to read previous training files instead of the evaluation folder.
- **Implemented Fixes:**
  - In [evaluation/checkpoint_loader.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/evaluation/checkpoint_loader.py): During checkpoint restoration, saved training manifest paths are archived into `opt._saved_manifest`, `opt._saved_val_manifest`, and `opt._saved_drct_manifest_path`, while active options `manifest`, `val_manifest`, and `drct_manifest_path` are explicitly set to `None`.
  - Stored built configuration on `loader.opt` so callers can inspect active and saved options.
  - In [data/datasets/drct_dataset.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/data/datasets/drct_dataset.py): During evaluation (`is_train=False`), manifest mode is activated *only* if `opt.manifest` is explicitly provided. It never falls back to saved training manifests. Folder mode uses directory scanning (`find_images(root)`) on the requested evaluation root.
- **Verification Tests:**
  - `TestS3ManifestIsolationDuringEvaluation::test_s3_folder_evaluation_ignores_saved_training_manifest` in [tests/test_drct_second_review_fixes.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/tests/test_drct_second_review_fixes.py): Saved checkpoint with defunct training manifest paths evaluates external image folder without accessing the old paths; verifies that all returned paths reside inside the target folder.

---

### S4 — P1: Fix the Legacy-Loader Regression
- **Root Cause:** In `CheckpointLoader._build_opt`, `saved = protocol.get('options', {})` was bound before reading the legacy JSON file. After `protocol = json.load(stream)`, `saved` was not refreshed, leaving `saved` as `{}`. Preprocessing key validation therefore failed on every legacy configuration.
- **Implemented Fixes:**
  - In [evaluation/checkpoint_loader.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/evaluation/checkpoint_loader.py): Immediately after parsing `protocol = json.load(stream)`, refreshed `saved = protocol.get('options', {})`.
  - Added strict key validation using `get_preprocessing_keys_for_arch(arch)`.
  - Retained original label mapping and options without guessing.
- **Verification Tests:**
  - `TestS4LegacyLoaderRegression::test_s4_legacy_json_restores_options_and_preprocessing` in [tests/test_drct_second_review_fixes.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/tests/test_drct_second_review_fixes.py): Restores valid options and preprocessing from legacy JSON without error.
  - `TestS4LegacyLoaderRegression::test_s4_legacy_json_rejects_missing_required_keys`: Rejects legacy JSON missing required architecture keys (`drct_preprocessing_version`).

---

### S5 — P1: Finish Reconstruction Preparation and Provenance Checks
- **Root Cause:** Duplicate shadowed `import_precomputed_reconstructions` functions existed. Sample IDs generated as `f"{split}_{stem}"` collided when identical stems appeared across classes or file extensions. Same-content hashes in different splits were not caught by `validate_drct_records`. Directory scanning stopped at shallow folders.
- **Implemented Fixes:**
  - In [data/drct_manifest.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/data/drct_manifest.py):
    - `generate_sample_id`: Constructs unique IDs using `split`, `label_str`, `path_hash[:8]`, and `stem`, ensuring collision immunity across nested directories and extensions.
    - `validate_drct_records`: Added cross-split identical content-hash validation. Reject records where the same image content (sha256) is present in both train and validation splits.
  - In [data/drct_preparation.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/data/drct_preparation.py):
    - Removed duplicate shadowed definition of `import_precomputed_reconstructions`.
    - Recursively scans nested class folders (`rglob`) while preserving relative parent directories.
    - Exits with non-zero CLI status code on failed readiness audits.
- **Verification Tests:**
  - `TestS5PreparationAndProvenance::test_s5_stem_collisions_and_nested_folders` in [tests/test_drct_second_review_fixes.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/tests/test_drct_second_review_fixes.py).
  - `TestS5PreparationAndProvenance::test_s5_cross_split_same_content_duplicate_rejected`: Verifies rejection of duplicate image content across train and dev splits.

---

### S6 — P1: Finish Strict Official Checkpoint Imports and Encoder-Only Mapping
- **Root Cause:** Native import checked only whether missing keys contained `"weight"` or belonged to classifier heads, ignoring biases, positional embeddings, and projection weights. Encoder-only initialization threw exceptions if the official head dimension was 1024.
- **Implemented Fixes:**
  - In [models/drct/checkpoint_io.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/models/drct/checkpoint_io.py):
    - `import_official_drct_detector`: Strictly validates all destination visual keys. Missing encoder weights, biases, or projections raise `KeyError`. Disallows unexpected keys.
    - `initialize_drct128_from_pretrained_encoder`: Filters out classifier and projection head weights (`fc.weight`, `fc.bias`, `head.*`) from the source dictionary, loading only the encoder weights into the backbone while strictly preserving the fresh, randomly initialized 128D projection and classifier layers.
- **Verification Tests:**
  - `TestS6OfficialCheckpointImports::test_s6_official_detector_rejects_missing_projection_or_unexpected` in [tests/test_drct_second_review_fixes.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/tests/test_drct_second_review_fixes.py).
  - `TestS6OfficialCheckpointImports::test_s6_destination_key_verification_rejects_missing_bias_or_proj`.
  - `TestS6OfficialCheckpointImports::test_s6_encoder_only_import_discards_head_and_preserves_128d_projection`.

---

### S7 — P2: Preserve Exact Zero Distances in the Contrastive Reducer
- **Root Cause:** In `models/drct/loss.py`, `torch.clamp(dist_sq, min=1e-12).sqrt()` converted identical positive embeddings with distance 0.0 into a tiny positive distance, causing them to be counted as active violations in `AvgNonZeroReducer`. In the 3-vector benchmark case (`[[1,0],[1,0],[0,1]]`, all label 0), this artificially diluted the loss to `0.942809` instead of exact `1.4142135624`.
- **Implemented Fixes:**
  - In [models/drct/loss.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/models/drct/loss.py): Replaced squared-distance clamping with native `torch.cdist(embeddings, embeddings, p=2.0)`.
  - Pairwise Euclidean distance between identical vectors evaluates to exact `0.0`.
  - For positive pairs with `pos_margin=0.0`, violations occur only when `dist > 0.0`. Non-violations (0.0) are excluded from the reducer denominator.
  - Gradients remain strictly finite and numerical stability is preserved.
- **Verification Tests:**
  - `TestS7ContrastiveLossExactReducer::test_s7_exact_zero_distance_and_all_identical` in [tests/test_drct_second_review_fixes.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/tests/test_drct_second_review_fixes.py): Verifies exact 0.0 loss and finite grad for identical vectors.
  - `TestS7ContrastiveLossExactReducer::test_s7_three_vector_benchmark_case`: Produces exact $1.4142135624$ (matches $\sqrt{2}$).
  - `TestS7ContrastiveLossExactReducer::test_s7_four_vector_benchmark_case`: Produces exact $0.8270503283$.

---

### S8 — P1: Repair Colab Evaluation, Calibration, and Recovery Cells
- **Root Cause:** `COLAB_DRCT_128_TRAINING.md` looked for `EVAL_DIR/predictions.csv` instead of `EVAL_DIR/DRCT_128/predictions.csv`, used default test split instead of `dev`, lacked `--score_space logit`, only backed up checkpoints at process termination rather than during training, and lacked a persistent run descriptor.
- **Implemented Fixes:**
  - In [COLAB_DRCT_128_TRAINING.md](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/COLAB_DRCT_128_TRAINING.md):
    - Pinned exact revisions: `git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33` and `pytorch-metric-learning==2.8.1`.
    - Restores preparation bundle from Google Drive without re-inventorying unchanged data.
    - Maintains persistent `run_descriptor.json` recording git commit, manifests, digests, and training parameters.
    - Periodic background backup script (`sync_checkpoints.py`) safely mirrors completed checkpoints and logs to Google Drive during training.
    - Cell 9 targets `EVAL_DIR/DRCT_128/predictions.csv` with explicit `--split dev`.
    - Cell 10 executes checkpoint-bound logit calibration using `--score_space logit` and backs up the calibrated threshold JSON to Drive.
    - Cell 11 fixes fusion dataset paths and flags.

---

### S9 — P2: Preprocessing and Resolution Constraints
- **Root Cause:** Fusion expert loading allowed arbitrary crop sizes even though CLIP ViT strictly requires 224x224 input geometry. Non-square images needed verified parity.
- **Implemented Fixes:**
  - In [data/datasets/fusion_dataset.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/data/datasets/fusion_dataset.py): Validates that `drct_input_sz == 224`. Raises explicit `ValueError` if a non-224 input size is requested for DRCT.
  - Verified non-square aspect-ratio handling: `DRCTValTransform(224)` pads non-square images (e.g. 400x300) to 224x224 preserving aspect ratio and yields exact 128D embeddings through `forward_features(x)`.
- **Verification Tests:**
  - `TestS9PreprocessingAndResolution::test_s9_rejects_unsupported_non_224_input_size` in [tests/test_drct_second_review_fixes.py](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/tests/test_drct_second_review_fixes.py).
  - `TestS9PreprocessingAndResolution::test_s9_versioned_adaptation_preprocessing_non_square_parity`.

---

## 3. Test Execution & Evidence

### A. Dedicated Second-Review Test Suite
```bash
pytest tests/test_drct_second_review_fixes.py -v
```
**Output:**
```
tests/test_drct_second_review_fixes.py::TestS1PreparationGateAndTrainingEntry::test_s1_end_to_end_preparation_to_training_gate PASSED [  5%]
tests/test_drct_second_review_fixes.py::TestS1PreparationGateAndTrainingEntry::test_s1_full_drct_alpha_without_reconstructions_rejected PASSED [ 10%]
tests/test_drct_second_review_fixes.py::TestS1PreparationGateAndTrainingEntry::test_s1_gate_digest_mismatch_fails_preflight PASSED [ 15%]
tests/test_drct_second_review_fixes.py::TestS2CLIPPretrainedInitialization::test_s2_config_and_benchmark_roundtrip_includes_backbone_settings PASSED [ 21%]
tests/test_drct_second_review_fixes.py::TestS2CLIPPretrainedInitialization::test_s2_local_cached_weights_loads_without_download PASSED [ 26%]
tests/test_drct_second_review_fixes.py::TestS2CLIPPretrainedInitialization::test_s2_pretrained_true_with_allow_download_false_raises PASSED [ 31%]
tests/test_drct_second_review_fixes.py::TestS3ManifestIsolationDuringEvaluation::test_s3_folder_evaluation_ignores_saved_training_manifest PASSED [ 36%]
tests/test_drct_second_review_fixes.py::TestS4LegacyLoaderRegression::test_s4_legacy_json_rejects_missing_required_keys PASSED [ 42%]
tests/test_drct_second_review_fixes.py::TestS4LegacyLoaderRegression::test_s4_legacy_json_restores_options_and_preprocessing PASSED [ 47%]
tests/test_drct_second_review_fixes.py::TestS5PreparationAndProvenance::test_s5_cross_split_same_content_duplicate_rejected PASSED [ 52%]
tests/test_drct_second_review_fixes.py::TestS5PreparationAndProvenance::test_s5_stem_collisions_and_nested_folders PASSED [ 57%]
tests/test_drct_second_review_fixes.py::TestS6OfficialCheckpointImports::test_s6_destination_key_verification_rejects_missing_bias_or_proj PASSED [ 63%]
tests/test_drct_second_review_fixes.py::TestS6OfficialCheckpointImports::test_s6_encoder_only_import_discards_head_and_preserves_128d_projection PASSED [ 68%]
tests/test_drct_second_review_fixes.py::TestS6OfficialCheckpointImports::test_s6_official_detector_rejects_missing_projection_or_unexpected PASSED [ 73%]
tests/test_drct_second_review_fixes.py::TestS7ContrastiveLossExactReducer::test_s7_exact_zero_distance_and_all_identical PASSED [ 78%]
tests/test_drct_second_review_fixes.py::TestS7ContrastiveLossExactReducer::test_s7_four_vector_benchmark_case PASSED [ 84%]
tests/test_drct_second_review_fixes.py::TestS7ContrastiveLossExactReducer::test_s7_three_vector_benchmark_case PASSED [ 89%]
tests/test_drct_second_review_fixes.py::TestS9PreprocessingAndResolution::test_s9_rejects_unsupported_non_224_input_size PASSED [ 94%]
tests/test_drct_second_review_fixes.py::TestS9PreprocessingAndResolution::test_s9_versioned_adaptation_preprocessing_non_square_parity PASSED [100%]

============================= 19 passed in 33.80s =============================
```

### B. Full DRCT Regression Test Suite
```bash
pytest -k drct -v
```
**Result:** **76 passed, 342 deselected** in 62.71s with zero failures.

### C. Pipeline & Staged Heads Regression
```bash
pytest tests/test_staged_pipeline_and_fusion_heads.py tests/test_resume_protocol_and_accumulation.py tests/test_rgb_contract.py -v
```
**Result:** **27 passed** in 9.51s with zero failures.

---

## 4. Pending Live Execution Items (Colab GPU)

In strict accordance with the review boundaries, the following live tasks are reserved for execution on Google Colab GPU and are not run locally:
1. **Full-Cohort GPU Training:** 5-epoch training on the full `DatesetNoVideosNoGAN` dataset.
2. **Diffusion Reconstruction Generation:** Batch inference with Stable Diffusion / LDM to generate paired reconstructions for `original_real` and `original_fake`.
3. **Generalization Evaluation on External Sets:** Evaluation on unseen test collections to measure detection metrics (AP, AUC, EER).

All preparatory mechanisms, gates, serialization routines, isolated evaluations, and Colab runbooks are verified and ready for live execution.
