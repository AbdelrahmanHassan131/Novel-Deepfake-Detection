# Google Colab Execution Guide: 100K Deepfake Detection Pilot Pipeline

**Completed audit reports SHA256 train/dev overlaps?** Follow [KAGGLE_PREPARATION_RECOVERY.md](KAGGLE_PREPARATION_RECOVERY.md). Reuse the completed hash evidence and preserve validation; do not rerun inventory to repair actual duplicate content. The main notebook now accepts `PREPARED_MANIFEST` to reuse the recovered selection.

This guide details the exact manual steps for training and evaluating the 100K pilot pipeline on Google Colab using an **NVIDIA L4 (24 GB)** or **two NVIDIA T4 (16 GB)** GPUs.

---

## Choose the training size from one image folder

The main notebook `colab_pilot_pipeline.ipynb` now supports any positive even training-image count or all eligible training images. In its first setup cell, change:

```python
TRAIN_SIZE = 100_000  # or 200_000, 400_000, or "all"
```

Keep `DATA_ROOT` pointing to the same full image folder and `POOL_MANIFEST` pointing to the same enriched full-pool CSV for every experiment. Nothing copies, moves, or deletes images. Each run writes a small selection CSV and experiment outputs in its own directory under `OUTPUT_BASE`. Setup creates a new run directory; do not rerun setup to resume an existing run.

- Numeric sizes target half real and half fake (100K means 50K of each, excluding validation/test counts). Selection is reproducible using seeded connected groups, rather than taking the first filenames, which might all belong to one class/source.
- `"all"` selects all eligible training images, disables the training frame cap, and retains the natural real/fake distribution. It does not put reserved validation/test or held-out sources into training. Thus a folder containing 1M images may provide fewer than 1M training images.
- The full-pool split logic runs before size selection, so validation/test membership stays the same across sizes for the same pool and split settings. Numeric subsets respect `FRAME_CAP`; `all` intentionally does not. If you want identical frame-cap policy for a size study, set `FRAME_CAP = 0` for all its runs. Group-based selection does not promise that every smaller subset is a strict prefix of a larger one.
- The requested count may be impossible because whole groups cannot fit, one class is too small, or frame caps reduce availability. Preparation stops and reports the shortage. Set `ALLOW_SHORTFALL = True` only when you accept the reported smaller count. The report always records the actual training count.

For direct CLI use, the `pilot` action now handles either subset or full selection:

```bash
python prepare_dataset.py pilot --manifest /content/data/pool_manifest.csv --root /content/data --output /content/experiments/train_100k.csv --train_size 100000 --frame_cap 15 --seed 42
python prepare_dataset.py pilot --manifest /content/data/pool_manifest.csv --root /content/data --output /content/experiments/train_all.csv --train_size all --seed 42
```

Replace `100000` with `200000` or `400000` as needed. Use `--train_size` instead of the older per-class target flags; those flags remain available when `--train_size` is omitted. `all` rejects source/generator quotas because they would exclude training images. Audit the chosen CSV before training (the notebook performs this automatically). The original full-pool CSV cannot be overwritten by pilot selection.

This change does not relax the separate smoke notebook's 300-row limit. Selection runtime tests have been written but NOT RUN; no dataset preparation or training was executed during implementation.

---

## First: the bounded smoke notebook

Use `colab_smoke_pipeline.ipynb` before the 100K pilot. Its code has been checked statically; training, inference, and deferred tests have NOT RUN. Begin with one GPU actually available in your runtime; the two-GPU instructions below apply only when two devices are available.

1. Upload the repository and a separate `smoke_input.csv`. Set `REPO_ROOT`, `DATA_ROOT`, `SMOKE_INPUT_MANIFEST`, and `OUTPUT_BASE` in the setup cell. The default output under `/content` is temporary. To persist outputs, run `from google.colab import drive` and `drive.mount('/content/drive')` in a separate Python cell first, then set `OUTPUT_BASE` under `/content/drive/MyDrive`.
2. Build the small CSV from verified metadata, preserving existing assignments and all identity/video/original-derivative links. Select independent groups from train and dev; never move a group across splits to fill a quota. Include both real and fake in each. Preserve all available identity/original/video/hash columns and every required column listed in Step 0 below. Paths must be absolute Colab paths or relative to `DATA_ROOT`, including each source's directory. Explicitly relocate Windows paths first. If a group is too large, choose another independent group; do not invent metadata.
3. Use at most 50 real + 50 fake training rows, at most 100 development rows, and optionally at most 100 internal-test rows (300 total maximum). Internal-test rows are only audited by this workflow. Smaller train/dev sets are allowed when both classes are present. Supply preassigned `train`, `dev`, and optional `internal_test` splits; the smoke workflow does not partition the full pool or run pilot selection. If you lack verified grouping metadata, obtain it before training.
4. Run notebook cells in order. The CSV preflight stops oversized inputs before any image files are opened; the mandatory existing audit then checks relationships and hashes. Fresh RGB/wavelet/fusion training follows, using batch size 8, no pretrained initialization, no AMP, and zero loader workers. The notebook evaluates the fresh fusion checkpoint on dev, writes `predictions.csv`, and calibrates the threshold through the actual analysis CLI.
5. The final cell resumes the same fusion run with the same settings and checks saved manifest digests, state presence, logged starting counters, and advancement from epoch 1 to 2. It does not prove bitwise optimizer/RNG continuation. Preserve `resolved_commands.json`, `resume.log`, `resume_evidence.json`, manifests/audits, checkpoints, predictions, and threshold output.

Setup creates a unique run directory. Re-running setup starts another isolated run; it does not resume an old one. Re-running an already-completed training cell is expected to fail on a directory collision. The explicit resume cell is the intended continuation path.

The generator `tools/make_smoke_notebook.py` owns both the notebook and `config/experiments/smoke_test_run.json`. The notebook is the plan's consumer. `tools/check_smoke_static.py` validates entry points and CLI options without importing training/evaluation modules. It is included as a notebook preflight.

After dependency setup and before an expensive pilot, run the deferred relevant tests in a separate Colab cell, for example:

```python
import subprocess, sys
for pattern in ('test_smoke_manifest_bounds.py', 'test_round3_data_contracts.py',
                'test_resume_protocol_and_accumulation.py'):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', pattern], check=True)
```

These commands are provided for later user execution, not reported as passed. Run the distributed correctness suites separately before relying on two GPUs. Start the 100K pilot only after relevant tests and the bounded smoke workflow succeed; smoke accuracy is not a scientific result.

---

## 1. Hardware & Memory Sizing Rules

| Hardware Setup | Micro-Batch Size | Gradient Accumulation Steps | GPUs / Ranks | Effective Batch Size | Notes |
|---|---:|---:|---:|---:|---|
| **Single NVIDIA L4 (24 GB)** | 32 | 2 | 1 | **64** | Recommended; highest memory headroom and simplest execution. |
| **Single NVIDIA T4 (16 GB)** | 16 | 4 | 1 | **64** | Resource-constrained; uses 4 accumulation steps to reach effective batch 64. |
| **Dual NVIDIA T4 (2x 16 GB)** | 16 | 2 | 2 (DDP) | **64** | Two independent devices in DDP via `torchrun`. **Never** treated as one 32 GB shared-memory GPU. |

> [!IMPORTANT]
> **No Pre-Measurement Promises**: Multi-GPU throughput and batch capacity must be measured on the actual hardware instance before claiming wall-clock times. The pilot commands below request mixed precision with `--use_amp`; the smoke notebook leaves it disabled.

---

## 2. Directory Layout & Mount Setup

On Google Colab, configure the following persistent and local scratch paths:

```bash
# 1. Mount Google Drive for persistent artifact and checkpoint storage
# CRITICAL: /content is ephemeral scratch storage that is deleted when your Colab runtime disconnects!
# To preserve trained checkpoints, calibration thresholds, and evaluation logs, use a mounted Google Drive directory.
from google.colab import drive
drive.mount('/content/drive')

# 2. Recommended Directory Mapping
export REPO_ROOT="/content/Novel-Deepfake-Detection"
export DATA_ROOT="/content/data"
export POOL_MANIFEST="${DATA_ROOT}/pool_manifest.csv"
export OUTPUT_ROOT="/content/drive/MyDrive/deepfake_experiments"  # Persistent on Drive
```

---

## 3. Step-by-Step Training Protocol

### Step 0: Pool Manifest Input Contract & Enrichment
Before running pilot selection, an enriched pool manifest must be present at `${POOL_MANIFEST}`.
Required CSV columns:
- `sample_id`: Globally namespaced, source-prefixed unique identifier (e.g., `hashlib.sha256(f"{source}:{rel_path}").hexdigest()[:16]`).
- `path`: Image path (relative to source root or absolute with explicit relocation).
- `label`: Binary classification target (`0` = authentic real, `1` = manipulated fake).
- `split`: Partition hint (`train`, `dev`, `internal_test`, `external_dev`, or `unassigned`).
- `dataset_source`: Immutable verified dataset source identifier (e.g. `FaceForensics`, `Celeb-DF`, `DFDC`).
- `group_id`: Connected component grouping token linking pristine originals and manipulated derivatives.
- `source_video_id`: Verified parent video identifier (or `none` for independent photos).

To create it honestly:
1. **Inventory raw sources**:
   ```bash
   python prepare_dataset.py inventory --root ${DATA_ROOT}/FaceForensics --source FaceForensics --output ${DATA_ROOT}/ff_inv.csv
   python prepare_dataset.py inventory --root ${DATA_ROOT}/CelebDF --source CelebDF --output ${DATA_ROOT}/celeb_inv.csv
   ```
2. **Enrich metadata**: Fill verified `group_id`, `source_video_id`, and `generator` values.
3. **Combine**: Merge the enriched manifests into `${POOL_MANIFEST}`.

### Step 1: 100K Pilot Dataset Preparation
Generates an auditable dataset split enforcing connected-component grouping (pristine real and manipulated fakes remain in the same split) and per-video frame capping:

```bash
# 1. Build 100K pilot dataset split (or specify smaller targets / --allow_shortfall if using a limited pool)
python prepare_dataset.py pilot \
    --manifest ${POOL_MANIFEST} \
    --root ${DATA_ROOT} \
    --output ${OUTPUT_ROOT}/pilot_100k_manifest.csv \
    --target_real 50000 \
    --target_fake 50000 \
    --dev_ratio 0.1 \
    --test_ratio 0.1 \
    --frame_cap 15 \
    --require_groups \
    --seed 42

# 2. Mandatory split audit and hash verification
# Automatically writes ${OUTPUT_ROOT}/pilot_100k_manifest.verified.json upon passing.
python prepare_dataset.py audit \
    --manifest ${OUTPUT_ROOT}/pilot_100k_manifest.csv \
    --output ${OUTPUT_ROOT}/audit_report.json \
    --hashes
```

> [!IMPORTANT]
> **Audit Gate Requirement**: `train.py` enforces `--require_verified_manifest` by default. It verifies that `${OUTPUT_ROOT}/pilot_100k_manifest.verified.json` exists, records `verified: true`, covers both classes (real=0 and fake=1), and that the SHA-256 digest of `pilot_100k_manifest.csv` matches the gate file. If the audit failed or was modified, training fails closed immediately. If `val_manifest` is a separate file, its gate is verified as well.


### Step 2: Stage 1 — Standalone RGB Expert (`Wang2020_128`)
Trains the RGB expert on the training split with development-AUC checkpoint selection:

```bash
python train.py \
    --arch Wang2020_128 \
    --dataroot ${DATA_ROOT} \
    --manifest ${OUTPUT_ROOT}/pilot_100k_manifest.csv \
    --manifest_split train \
    --val_manifest ${OUTPUT_ROOT}/pilot_100k_manifest.csv \
    --val_manifest_split dev \
    --name stage1_rgb_expert \
    --run_id seed42 \
    --checkpoints_dir ${OUTPUT_ROOT}/stage1_rgb \
    --batch_size 32 \
    --grad_accum_steps 2 \
    --epochs 10 \
    --lr 0.0001 \
    --monitor_metric auc \
    --seed 42 \
    --use_amp
```

### Step 3: Stage 2 — Standalone Wavelet Expert (`WolterWavelet2021_128`)
Trains the frequency expert using Haar wavelets at level 3 with numerically stable `signed_log1p` scaling:

```bash
python train.py \
    --arch WolterWavelet2021_128 \
    --dataroot ${DATA_ROOT} \
    --manifest ${OUTPUT_ROOT}/pilot_100k_manifest.csv \
    --manifest_split train \
    --val_manifest ${OUTPUT_ROOT}/pilot_100k_manifest.csv \
    --val_manifest_split dev \
    --name stage2_wavelet_expert \
    --run_id seed42 \
    --checkpoints_dir ${OUTPUT_ROOT}/stage2_wavelet \
    --wavelet_type haar \
    --wavelet_level 3 \
    --wavelet_log_mode signed_log1p \
    --batch_size 32 \
    --grad_accum_steps 2 \
    --epochs 10 \
    --lr 0.0001 \
    --monitor_metric auc \
    --seed 42 \
    --use_amp
```

### Step 4: Stage 3 — Controlled Fusion Heads on Frozen Experts
Both experts are frozen. The fusion heads are evaluated under identical data and expert features.

Define paths to newly produced expert checkpoints:
```bash
export RGB_CKPT="${OUTPUT_ROOT}/stage1_rgb/stage1_rgb_expert_seed42/checkpoints/best.pth"
export WAV_CKPT="${OUTPUT_ROOT}/stage2_wavelet/stage2_wavelet_expert_seed42/checkpoints/best.pth"
```

#### Head A: Corrected Multi-Key Token Attention (`MHA_128`)
```bash
python train.py \
    --arch MHA_128 \
    --fusion_type token_attention \
    --rgb_model_path ${RGB_CKPT} \
    --wavelet_model_path ${WAV_CKPT} \
    --dataroot ${DATA_ROOT} \
    --manifest ${OUTPUT_ROOT}/pilot_100k_manifest.csv \
    --manifest_split train \
    --val_manifest ${OUTPUT_ROOT}/pilot_100k_manifest.csv \
    --val_manifest_split dev \
    --name stage3_token_attention \
    --run_id seed42 \
    --checkpoints_dir ${OUTPUT_ROOT}/stage3_token_attention \
    --batch_size 32 \
    --grad_accum_steps 2 \
    --epochs 10 \
    --monitor_metric auc \
    --seed 42 \
    --use_amp
```

#### Head B: Gated Fusion Control (`Fusion_128`)
```bash
python train.py \
    --arch Fusion_128 \
    --fusion_type gated \
    --rgb_model_path ${RGB_CKPT} \
    --wavelet_model_path ${WAV_CKPT} \
    --dataroot ${DATA_ROOT} \
    --manifest ${OUTPUT_ROOT}/pilot_100k_manifest.csv \
    --manifest_split train \
    --val_manifest ${OUTPUT_ROOT}/pilot_100k_manifest.csv \
    --val_manifest_split dev \
    --name stage3_gated \
    --run_id seed42 \
    --checkpoints_dir ${OUTPUT_ROOT}/stage3_gated \
    --batch_size 32 \
    --grad_accum_steps 2 \
    --epochs 10 \
    --monitor_metric auc \
    --seed 42 \
    --use_amp
```

#### Head C: Concatenation MLP Control (`Fusion_128`)
```bash
python train.py \
    --arch Fusion_128 \
    --fusion_type concat \
    --rgb_model_path ${RGB_CKPT} \
    --wavelet_model_path ${WAV_CKPT} \
    --dataroot ${DATA_ROOT} \
    --manifest ${OUTPUT_ROOT}/pilot_100k_manifest.csv \
    --manifest_split train \
    --val_manifest ${OUTPUT_ROOT}/pilot_100k_manifest.csv \
    --val_manifest_split dev \
    --name stage3_concat \
    --run_id seed42 \
    --checkpoints_dir ${OUTPUT_ROOT}/stage3_concat \
    --batch_size 32 \
    --grad_accum_steps 2 \
    --epochs 10 \
    --monitor_metric auc \
    --seed 42 \
    --use_amp
```

---

## 4. Dual-GPU Distributed Execution (2x T4 via `torchrun`)

When a multi-GPU runtime is available, launch using `torchrun`:

```bash
torchrun --nproc_per_node=2 train.py \
    --arch Wang2020_128 \
    --dataroot ${DATA_ROOT} \
    --manifest ${OUTPUT_ROOT}/pilot_100k_manifest.csv \
    --manifest_split train \
    --val_manifest ${OUTPUT_ROOT}/pilot_100k_manifest.csv \
    --val_manifest_split dev \
    --name stage1_rgb_ddp \
    --run_id seed42 \
    --checkpoints_dir ${OUTPUT_ROOT}/stage1_rgb_ddp \
    --batch_size 16 \
    --grad_accum_steps 2 \
    --epochs 10 \
    --use_amp
```

---

## 5. Resuming an Interrupted Run

To resume an interrupted stage safely, point directly to the full-protocol checkpoint:

```bash
python train.py \
    --continue_train \
    --resume_checkpoint ${OUTPUT_ROOT}/stage1_rgb/stage1_rgb_expert_seed42/checkpoints/last.pth \
    --dataroot ${DATA_ROOT} \
    --manifest ${OUTPUT_ROOT}/pilot_100k_manifest.csv
```

---

## 6. Threshold Calibration & Model Comparison

Calibrate the deployment threshold strictly on development predictions using Youden's J:

```bash
# 1. Calibrate threshold on dev predictions
python analyze_predictions.py calibrate \
    --predictions ${OUTPUT_ROOT}/stage3_token_attention/stage3_token_attention_seed42/checkpoints/dev_predictions.csv \
    --output ${OUTPUT_ROOT}/stage3_token_attention/threshold_calibrated.json

python analyze_predictions.py calibrate \
    --predictions ${OUTPUT_ROOT}/stage3_gated/stage3_gated_seed42/checkpoints/dev_predictions.csv \
    --output ${OUTPUT_ROOT}/stage3_gated/threshold_calibrated.json

# 2. Paired statistical comparison with cluster bootstrap using calibrated thresholds
python analyze_predictions.py compare \
    --predictions ${OUTPUT_ROOT}/stage3_token_attention/stage3_token_attention_seed42/checkpoints/dev_predictions.csv \
                  ${OUTPUT_ROOT}/stage3_gated/stage3_gated_seed42/checkpoints/dev_predictions.csv \
    --threshold_file ${OUTPUT_ROOT}/stage3_token_attention/threshold_calibrated.json \
    --threshold_file_other ${OUTPUT_ROOT}/stage3_gated/threshold_calibrated.json \
    --bootstrap 1000 \
    --output ${OUTPUT_ROOT}/paired_comparison_token_vs_gated.json
```

---

## 7. Multi-Seed Reporting (3 Independent Seeds)

Repeat the entire pipeline for seeds 42, 43, and 44 (re-training experts and heads independently per seed):

```bash
python analyze_predictions.py seeds \
    --predictions ${OUTPUT_ROOT}/seed42_preds.csv \
                  ${OUTPUT_ROOT}/seed43_preds.csv \
                  ${OUTPUT_ROOT}/seed44_preds.csv \
    --thresholds 0.50 0.51 0.49 \
    --output ${OUTPUT_ROOT}/three_seed_summary.json
```

---

## 8. Scaling Ladder (100K -> 200K -> 400K -> 1M)

Do not jump directly to 1M images. Follow this evidence-gated scaling policy:

1. **100K Pilot**: Complete Stage 1-3 training, 3 seeds, and external-development evaluation.
2. **Evaluation Gate**: Expand to 200K only if external-development ROC AUC / balanced accuracy demonstrates generalization gain over standalone experts.
3. **400K & 1M**: Maintain identical fixed development and test partitions. Scale data only if gains justify computation cost.
