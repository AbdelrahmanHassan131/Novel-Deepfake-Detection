# DRCT-128 Code Review Fixes: Formal Response & Evidence Report

This document provides a systematic, finding-by-finding response to all 12 review findings (R1 through R12) specified in `DRCT_128_CODE_REVIEW_FIXES.md`. All fixes preserve the foundational contract established in `DRCT_128_IMPLEMENTATION_HANDOFF.md`:
- **Model Output:** Single raw binary logit $[B, 1]$ (real = 0, fake = 1).
- **Penultimate Learned Embedding:** 128-dimensional feature vector ($[B, 128]$) extracted via `forward_features(x)` prior to classification.
- **Fusion Integration:** Two-stream Multi-Head Attention (`MHA_128`) and Controlled Fusion with self-contained embedded metadata.

All local unit and regression tests (57 DRCT tests, 27 existing regression tests) pass with zero errors on Windows (Python 3.11.3, PyTorch 2.5.1+cu121). Real GPU dataset training, diffusion reconstruction generation, and downloads of large official weights remain honestly marked pending execution on GPU.

---

## Detailed Finding-by-Finding Response

### R1 — [P1] Train/val leakage in the dataset and broad exception swallowing
- **Review Finding:** `DRCTDataset` defaulted missing splits to `'train'`, swallowed corrupted rows or missing keys silently, lacked on-disk image file verification, and evaluated on checkpoint-persisted training paths when evaluation manifests were specified.
- **Implemented Fixes:**
  - In `data/datasets/drct_dataset.py`:
    - Enforced strict split checking: if the requested split is absent from the manifest, raised `ValueError` listing all available manifest splits (preventing silent fallback to `'train'`).
    - Enforced explicit schema validation via `DRCTManifest.validate()` without broad exception catching.
    - Verified that every referenced image file exists on disk and is a valid file (`os.path.isfile`), raising `FileNotFoundError` if absent.
    - During evaluation (`not is_train`), explicitly prioritized CLI-provided `opt.manifest` or `opt.val_manifest` over saved checkpoint paths.
- **Test Evidence:**
  - `tests/test_drct_preparation.py`: `test_preservation_of_protected_splits_and_cross_split_rejection`
  - `tests/test_drct_benchmark.py`: `test_benchmark_real_manifest_smoke[0]`, `test_benchmark_real_manifest_smoke[1]`
- **Status:** **RESOLVED**

---

### R2 — [P1] Preparation allows contradictory data and crosses split boundaries
- **Review Finding:** Reconstruction generation allowed identical content hashes to have opposing labels, let groups cross split boundaries, and permitted training reconstructions to be generated from external development parents.
- **Implemented Fixes:**
  - In `data/drct_manifest.py`:
    - Added hash-to-labels tracking: rejected any manifest containing duplicate SHA-256 digests with contradictory labels ($[0, 1]$).
    - Added group-to-splits tracking: rejected any manifest where a single `group_id` spans multiple splits.
    - Added training parent exclusion: prohibited samples from external development splits (`val to be deleted`, `val_unseen`, `external_dev`, `external_test`) from serving as training parents for reconstructions.
- **Test Evidence:**
  - `tests/test_drct_preparation.py`: `test_contradictory_hash_rejection`
  - `tests/test_drct_preparation.py`: `test_cross_split_group_rejection`
  - `tests/test_drct_preparation.py`: `test_preservation_of_protected_splits_and_cross_split_rejection`
- **Status:** **RESOLVED**

---

### R3 — [P1] Silent non-generation of reconstructions and non-atomic preparation
- **Review Finding:** Preparation wrote incomplete manifests without verifying generated files, lacked directory inventory and precomputed reconstruction import tools, and lacked atomic write safety. Training had no preflight readiness check.
- **Implemented Fixes:**
  - In `data/drct_preparation.py`:
    - Implemented `inventory_dataset_directory` to inventory standard `train/val` and `real/fake` folder structures into valid DRCT CSV manifests.
    - Implemented `import_precomputed_reconstructions` to link precomputed reconstructed images to authentic parent records.
    - Implemented atomic file writing: outputs are generated to `.tmp_{pid}_{key}.png`, verified with `PIL.Image.verify()`, and atomically renamed to destination.
    - Added seed propagation: `seed` is passed explicitly in settings to reconstruction generators.
    - In `check_drct_readiness`: verified that declared reconstruction files physically exist and are non-zero byte decodable images.
  - In `train.py`:
    - Added preflight check calling `check_drct_readiness` before training when `arch == 'DRCT_128'` and `drct_alpha > 0`.
- **Test Evidence:**
  - `tests/test_drct_preparation.py`: `test_parent_linkage_and_opposite_labels`
  - `tests/test_drct_preparation.py`: `test_interrupted_preparation_resumes_without_recomputing`
  - `tests/test_drct_preparation.py`: `test_inventory_dataset_directory`
  - `tests/test_drct_preparation.py`: `test_import_precomputed_reconstructions`
  - `tests/test_drct_preparation.py`: `test_drct_readiness_audit`
- **Status:** **RESOLVED**

---

### R4 — [P1] Inverted contrastive loss math and diverging fallback implementation
- **Review Finding:** Euclidean distance formula was inverted ($m - D$ for positive pairs instead of $D - m_{\text{pos}}$), violations were squared, reduction divided by total pairs rather than averaging non-zero violations (`AvgNonZeroReducer`), and exception handling silently swapped loss functions.
- **Implemented Fixes:**
  - In `models/drct/loss.py`:
    - Embeddings are $L_2$-normalized: $z = \frac{x}{\|x\|_2}$.
    - Upper-triangular pairwise Euclidean distances: $D_{ij} = \|z_i - z_j\|_2$.
    - Unsquared hinge margin violations:
      - Positive pairs ($y_i = y_j$): $V_{ij}^+ = \max(0, D_{ij} - m_{\text{pos}})$.
      - Negative pairs ($y_i \neq y_j$): $V_{ij}^- = \max(0, m_{\text{neg}} - D_{ij})$.
    - Reduction: `AvgNonZeroReducer` averaging only non-zero violation terms ($V > 0$). If zero violations exist, returns $0.0$ connected to the autograd graph (`(embeddings * 0.0).sum()`).
    - Pinned arithmetic: matches the review reference example `[[1, 0], [1, 0.5], [0, 1], [0.5, 1]]` with labels `[0, 0, 1, 1]` producing exact loss `0.8270503283`.
    - Removed arbitrary exception catching and dynamic switching.
- **Test Evidence:**
  - `tests/test_drct_training_objective.py`: `test_drct_margin_contrastive_loss_arithmetic_and_pml_parity`
  - `tests/test_drct_training_objective.py`: `test_combined_loss_bce_and_contrastive_weighting`
  - `tests/test_drct_training_objective.py`: `test_homogeneous_batch_contrastive_safety`
  - `tests/test_drct_training_objective.py`: `test_zero_violation_margin_case`
- **Status:** **RESOLVED**

---

### R5 — [P1] Offline restore invokes CLIP's weight download path
- **Review Finding:** Backbone loading entered an empty pass block and called `clip.load` without honoring `pretrained=False`, risking network calls or failures when restoring self-contained checkpoints.
- **Implemented Fixes:**
  - In `models/drct/backbone.py`:
    - Implemented alias resolution: `clip_vit_b16`, `ViT-B/16`, `clip_vit_l14`, `ViT-L/14`, `clip-ViT-L/14`. Default: `clip_vit_l14` (ViT-L/14).
    - When `pretrained=False`, constructs the model architecture directly without invoking network downloads.
    - Enforced `allow_download=False` guard: raises `PermissionError` if weight download is attempted when downloads are disabled.
- **Test Evidence:**
  - `tests/test_drct_architecture.py`: `test_backbone_aliases_and_architecture_only_construction`
  - `tests/test_drct_architecture.py`: `test_allow_download_guard_prevents_unauthorized_fetching`
- **Status:** **RESOLVED**

---

### R6 — [P1] Partial/mismapped checkpoint imports are accepted
- **Review Finding:** Official reference import mapped all `model.model.*` tensors indiscriminately into `encoder.*`, retaining full text-side tensors. Incomplete checkpoints missing encoder weights were accepted under `strict=False`.
- **Implemented Fixes:**
  - In `models/drct/checkpoint_io.py`:
    - Added explicit text tensor exclusion prefix list (`token_embedding`, `text_projection`, `ln_final`, `transformer`).
    - Implemented strict visual encoder mapping (`model.model.visual.` -> `encoder.`).
    - Rejected incomplete checkpoints: raises `ValueError` if required visual encoder, projection (`model.fc`), or classification (`fc`) weights are missing.
- **Test Evidence:**
  - `tests/test_drct_checkpoint_and_transforms.py`: `test_import_official_drct_detector_and_single_logit_parity`
  - `tests/test_drct_checkpoint_and_transforms.py`: `test_import_official_drct_detector_rejects_missing_visual_weights`
  - `tests/test_drct_mha_expert.py`: `test_drct_expert_rejects_missing_encoder_weights`
- **Status:** **RESOLVED**

---

### R7 — [P1] Embedded MHA checkpoints lose the DRCT projection definition
- **Review Finding:** CheckpointManager only recorded architecture name in `expert_metadata`. When reloaded in a clean process without source files, non-default projection activation (e.g., ReLU) or dropout reverted to default linear projection.
- **Implemented Fixes:**
  - In `training/checkpoint_manager.py`:
    - Enhanced `expert_metadata` serialization to record `projection_activation`, `projection_dropout`, `input_size`, `embed_dim`, `preprocessing_version`, and `label_mapping`.
  - In `models/shared/expert_loading.py`:
    - `DRCT128FeatureExtractor` instantiates the projection using the exact saved activation and dropout metadata.
    - Strictly loads weights with `strict=True`, raising `ValueError` if encoder or projection keys are missing.
- **Test Evidence:**
  - `tests/test_drct_mha_expert.py`: `test_embedded_drct_preserves_non_default_activation_and_dropout`
  - `tests/test_drct_mha_expert.py`: `test_embedded_drct_fusion_checkpoint_roundtrip`
- **Status:** **RESOLVED**

---

### R8 — [P1] Fusion dataset decides RGB branch from path substrings and uses wavelet dimensions
- **Review Finding:** `FusionDataset` identified DRCT if `'drct'` was present anywhere in the file path (misidentifying Wang models in a folder named `drct`). DRCT input size was taken from global `cropSize` (which could be overridden by wavelet settings). Adding DRCT keys to `PREPROCESSING_KEYS` broke legacy Wang config validation.
- **Implemented Fixes:**
  - In `data/datasets/fusion_dataset.py`:
    - Removed path substring matching (`'drct' in path`).
    - Resolved RGB expert architecture strictly from checkpoint metadata (`ckpt['protocol']['options']['arch']` or `expert_metadata`).
    - Bound DRCT image transform size to `expert_metadata['input_size']` or `drct_crop_size` ($224 \times 224$), isolated from wavelet global crop settings.
  - In `config/protocol.py` & `evaluation/checkpoint_loader.py`:
    - Implemented `get_preprocessing_keys_for_arch(arch)`: legacy Wang checkpoints only require legacy keys (`cropSize`, `loadSize`, etc.) and do not acquire DRCT key requirements.
- **Test Evidence:**
  - `tests/test_drct_mha_expert.py`: `test_wang_in_drct_named_path_not_misidentified`
  - `tests/test_drct_mha_expert.py`: `test_branch_aware_transforms_fusion_dataset`
  - `tests/test_drct_mha_expert.py`: `test_wang_rgb_expert_backward_compatibility`
- **Status:** **RESOLVED**

---

### R9 — [P1] The Colab guide contains commands that cannot execute and incorrect backup paths
- **Review Finding:** The Colab guide used invalid CLI arguments (`--n_epochs`, `--epoch latest`, `--manifest_split`, `*.json`), lacked dependency installation for CLIP and PML, had non-atomic file copy loops, and described post-training backup as interruption recovery.
- **Implemented Fixes:**
  - Completely rewrote `COLAB_DRCT_128_TRAINING.md`:
    - Updated dependencies to install `pytorch-metric-learning`, official OpenAI CLIP, `scikit-learn`, `scipy`, `timm`, `pillow`.
    - Pinned CLI flags to real parser arguments: `--epochs`, `--continue_train --resume_checkpoint <path>`.
    - Specified CSV manifest paths throughout (`*.csv`).
    - Added tested preparation cells (`inventory_dir`, `import_rec`, `audit`).
    - Added internal validation evaluation and threshold calibration (`analyze_predictions.py calibrate`).
    - Added external development scoring using calibrated threshold artifacts (`--threshold_file`).
    - Implemented atomic backup routines with file verification and reporting.
- **Test Evidence:**
  - Parser argument compatibility verified against `train.py`, `evaluate.py`, `analyze_predictions.py`, and `data/drct_preparation.py`.
- **Status:** **RESOLVED**

---

### R10 — [P2] The benchmark's real-data path fails, and accumulation is ignored
- **Review Finding:** Benchmark created loader options without `mode`, causing real-data loading to fail. `grad_accum_steps` was accepted but ignored (every microbatch stepped optimizer). Benchmark silently defaulted to synthetic data when manifest was missing.
- **Implemented Fixes:**
  - In `tools/benchmark_drct.py`:
    - Set `loader_opt.mode = 'binary'` for proper dataset factory dispatch.
    - Implemented real gradient accumulation: microbatch losses are divided by `grad_accum_steps`, gradients accumulate across steps, and optimizer steps only on accumulation boundaries.
    - Enforced explicit mode requirement: raises `SystemExit` if neither `--manifest` nor `--synthetic` is specified.
    - Extracted and propagated target device index (`gpu_ids = [device.index]`).
    - Returned true observability metrics: `data_source`, `encoder`, `grad_accum_steps`, `effective_batch_size`, `optimizer_steps`.
- **Test Evidence:**
  - `tests/test_drct_benchmark.py`: `test_benchmark_gradient_accumulation`
  - `tests/test_drct_benchmark.py`: `test_benchmark_requires_explicit_mode`
  - `tests/test_drct_benchmark.py`: `test_benchmark_real_manifest_smoke`
- **Status:** **RESOLVED**

---

### R11 — [P2] Feature gradients and encoder-unfreeze options contradict their contracts
- **Review Finding:** `forward_features()` wrapped extraction in `torch.no_grad()` and toggled module modes. Encoder parameters were unfreezable via CLI but `trainable_parameters()` excluded them, preventing optimizer updates.
- **Implemented Fixes:**
  - In `models/drct/drct_model.py`:
    - Removed `no_grad()` and global mode toggles from `forward_features(x)`: autograd graph is fully preserved.
    - In `trainable_parameters()`: yields encoder parameters when `freeze_encoder=False`.
  - In `models/drct/trainer.py`:
    - Configured multi-group optimizer with `backbone_lr_mult` (default: 0.1) when encoder is unfrozen.
- **Test Evidence:**
  - `tests/test_drct_architecture.py`: `test_forward_features_preserves_autograd`
  - `tests/test_drct_architecture.py`: `test_encoder_unfreeze_and_optimizer_groups`
- **Status:** **RESOLVED**

---

### R12 — [P2] Replace completion claims with evidence covering the real workflow
- **Review Finding:** Documentation claimed unsupported aliases and verified readiness without full test evidence.
- **Implemented Fixes:**
  - Updated `DRCT_128_IMPLEMENTATION_REPORT.md` and created this comprehensive response document.
  - Documented exact test suite counts (57 DRCT tests, 27 existing regression tests).
  - Explicitly separated implemented CPU/synthetic tests from pending GPU operations (real-data 5-epoch training, diffusion reconstruction generation, official weight validation).
- **Status:** **RESOLVED**

---

## Verification Test Summary

```
tests\test_drct_architecture.py ..........                               [ 17%]
tests\test_drct_training_objective.py .....                              [ 26%]
tests\test_drct_benchmark.py ........                                    [ 40%]
tests\test_drct_preparation.py .........                                 [ 56%]
tests\test_drct_mha_expert.py ..........                                 [ 73%]
tests\test_drct_config_and_lifecycle.py ........                         [ 87%]
tests\test_drct_checkpoint_and_transforms.py .......                     [100%]
======================= 57 passed in 36.23s =======================
```
Existing regression suite (`test_staged_pipeline_and_fusion_heads.py`, `test_resume_protocol_and_accumulation.py`, `test_rgb_contract.py`):
```
======================= 27 passed in 10.39s =======================
```
All tasks from `DRCT_128_CODE_REVIEW_FIXES.md` are fully completed and verified.
