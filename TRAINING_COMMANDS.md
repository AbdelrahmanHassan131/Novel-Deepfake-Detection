# Training commands: Kaggle / Colab, one GPU or two GPUs

Updated 2026-10-02. Commands checked against the current source interfaces; training and runtime tests have NOT been run locally. Use the corrected smoke notebook first. This file contains training commands, not inference commands.

If independent-image inventory fails with `Conflicting fixed partition assignments ... ['dev', 'train']`, update `prepare_dataset.py` and follow [KAGGLE_MANIFEST_FIX.md](KAGGLE_MANIFEST_FIX.md). Older inventory code incorrectly grouped identical filename stems from different folders. Regenerate only the affected automatically generated inventory; preserve verified metadata and overlap checks.

## Reuse completed preparation; recover a failed duplicate audit

If your audit finished with SHA256 overlaps, use [KAGGLE_PREPARATION_RECOVERY.md](KAGGLE_PREPARATION_RECOVERY.md). It uses your saved completed hash evidence, preserves validation by default and excludes linked training groups without rereading images. Its explicit label-conflict quarantine option creates a documented new cohort and can exclude ambiguous validation groups too. Do not rerun inventory to fix real duplicates.

For a new model/run using an existing verified selection, set `PREPARED_MANIFEST` in the setup below. Keep the dataset contents and paths unchanged. The preparation cell will verify the saved CSV/gate and skip selection and hashing. Only checkpoints/results need a new `RUN_ROOT`. This fast reuse does not assert that changed image contents have been rechecked.

## 1. How to use this file

1. Put the updated repository in your notebook's working directory and install its dependencies as in the existing Colab/smoke guide.
2. Run the Python setup cell below from the directory containing `train.py`.
3. Select and audit the dataset once using Section 3.
4. For each desired model, run **either its one-GPU command or its two-GPU command**, not both in the same experiment directory.
5. For the main fusion experiment: train RGB-128, then Wavelet-128, then the desired fusion head(s). Raw models are optional standalone comparisons.

Each `%%bash` block is a complete Kaggle/Colab notebook cell. Paste it with `%%bash` on the first line. Do not add `!` before commands inside it. Variables are exported by the Python setup cell, so later cells can use them. For a terminal, omit `%%bash` and export equivalent variables in your shell.

## 2. Setup: dataset size, paths, epochs, and GPU settings

All size options use the SAME image directory and full-pool CSV. Only selected-row CSVs and experiment outputs are created. No image files are copied.

| `TRAIN_SIZE` | Training selection |
|---|---|
| `100_000` | Target 50K real + 50K fake |
| `200_000` | Target 100K real + 100K fake |
| `400_000` | Target 200K real + 200K fake |
| `"all"` | Every eligible training image, with natural class counts |

Numeric sizes use reproducible group-based selection, not the first filenames. Validation/test/held-out data remains excluded from training in all modes. Full mode disables training frame caps. With a fixed pool and split settings, evaluation membership stays fixed across sizes. Group constraints can prevent an exact requested count; preparation reports this rather than silently using fewer images. Set `ALLOW_SHORTFALL` only if you accept the reported smaller set.

```python
# Run from the uploaded repository directory containing train.py.
import os
from pathlib import Path
from uuid import uuid4

REPO_ROOT = Path.cwd().resolve()
if not (REPO_ROOT / 'train.py').is_file():
    raise FileNotFoundError('Change the notebook working directory to the repository containing train.py first.')

# Your existing Kaggle folder, including BOTH train/ and val/.
DATA_ROOT = Path('/kaggle/input/datasets/abdelrahmanhassani/prepareddatasetdiffgan/Preapred Dataset')
POOL_MANIFEST = Path('/kaggle/working/pool_manifest.csv')
PREPARED_MANIFEST = None  # Or Path('/kaggle/working/prepared_data/recovered_100k_seed42_v2/selected_manifest.csv')
OUTPUT_BASE = Path('/kaggle/working/deepfake_experiments')
# On Colab, change these paths to your actual /content/... dataset and mounted Drive output.

TRAIN_SIZE = 100_000       # 200_000, 400_000, or 'all'
EPOCHS = 2                # Matches your old short run; increase deliberately for later experiments.
SEED = 42
FRAME_CAP = 15            # Ignored for 'all'. Use 0 for uncapped subset size comparisons too.
ALLOW_SHORTFALL = False
INIT_FLAG = '--pretrained' # ImageNet RGB initialization; '--no-pretrained' starts RGB from scratch.

# Starting settings only: memory/throughput have not been measured for your run.
SINGLE_GPU = 'L4'          # Change to 'T4' for a single T4.
SINGLE_BATCH = 32 if SINGLE_GPU == 'L4' else 16
SINGLE_ACCUM = 2 if SINGLE_GPU == 'L4' else 4
DUAL_BATCH = 16            # Per GPU for two T4s.
DUAL_ACCUM = 2

size_text = str(TRAIN_SIZE).strip().lower()
if size_text != 'all' and (not size_text.isdecimal() or int(size_text) <= 0 or int(size_text) % 2):
    raise ValueError('TRAIN_SIZE must be a positive even count or all')
if not DATA_ROOT.is_dir() or (PREPARED_MANIFEST is None and not POOL_MANIFEST.is_file()):
    raise FileNotFoundError('Set DATA_ROOT and supply the enriched full-pool CSV at POOL_MANIFEST; see below.')
RUN_ROOT = OUTPUT_BASE / f'train_{size_text}_seed{SEED}_{uuid4().hex[:10]}'
RUN_ROOT.mkdir(parents=True, exist_ok=False)
MANIFEST = Path(PREPARED_MANIFEST).resolve() if PREPARED_MANIFEST is not None else RUN_ROOT / 'selected_manifest.csv'
RUN_ID = f'seed{SEED}'
values = dict(REPO_ROOT=REPO_ROOT, DATA_ROOT=DATA_ROOT, POOL_MANIFEST=POOL_MANIFEST, OUTPUT_BASE=OUTPUT_BASE,
              RUN_ROOT=RUN_ROOT, MANIFEST=MANIFEST, RUN_ID=RUN_ID, TRAIN_SIZE=size_text,
              REUSE_PREPARED=int(PREPARED_MANIFEST is not None),
              EPOCHS=EPOCHS, SEED=SEED, FRAME_CAP=FRAME_CAP,
              ALLOW_SHORTFALL=int(ALLOW_SHORTFALL), INIT_FLAG=INIT_FLAG,
              SINGLE_BATCH=SINGLE_BATCH, SINGLE_ACCUM=SINGLE_ACCUM,
              DUAL_BATCH=DUAL_BATCH, DUAL_ACCUM=DUAL_ACCUM,
              RGB_CKPT=RUN_ROOT / f'rgb128_{RUN_ID}/checkpoints/best.pth',
              WAV_CKPT=RUN_ROOT / f'wavelet128_{RUN_ID}/checkpoints/best.pth')
os.environ.update({key: str(value) for key, value in values.items()})
print(f'Run directory: {RUN_ROOT}')
print(f'Training size: {size_text}; no image copies will be made.')
```

The `POOL_MANIFEST` file is a required metadata input; these commands do not assume it already exists. It must cover your full image pool, with unique `sample_id`, valid `path`, `label` (real=0, fake=1), `split`, verified `dataset_source`, and verified `group_id`, plus available original/video/identity links. See `COLAB_RUN_GUIDE.md` for inventory and enrichment. Relative paths should include `train/...` or `val/...` under the shared `DATA_ROOT`. Development rows use the CSV split `dev` even when the physical folder is called `val`. Preserve verified assignments and relationships; do not make every frame an independent group or infer dataset origin from the combined folder name.

`--pretrained` initializes RGB from ImageNet and can download weights when the command is later run; it does not require your old deepfake checkpoint. If internet is unavailable, explicitly choose `--no-pretrained` before starting, or configure local backbone weights for RGB-128 using its supported `--backbone_weights` argument. Scratch training and ImageNet initialization are different experiments. Keep the same initialization policy across comparisons. Wavelet experts train fresh; fusion uses the experts produced below.

When `PREPARED_MANIFEST` is set, it determines the cohort and `TRAIN_SIZE` is ignored. To select a genuinely different size, set `PREPARED_MANIFEST = None`. Changing `TRAIN_SIZE` then means starting another experiment: rerun setup and preparation, then train new experts and fusion. Do not resume a checkpoint using a different selection manifest. Re-running setup always generates a fresh output directory; keep the printed directory for later continuation. Save/download Kaggle outputs or use persistent Colab storage before ending a session.

### Batch-size meaning

| Runtime | Batch per GPU | Accumulation | Effective batch |
|---|---:|---:|---:|
| One L4 | 32 | 2 | 64 |
| One T4 | 16 | 4 | 64 |
| Two T4s | 16 | 2 | 64 |

Effective batch = batch per GPU × GPU count × accumulation. Your previous `batch_size=64` on two GPUs corresponds to an effective batch of 128 when accumulation is 1. Two GPUs do not combine their memory into one allocation. If a model runs out of memory, halve its batch and double accumulation, then start a fresh run with those settings. These are initial settings, not measured capacity guarantees.

## 3. Select the size and audit it (same commands for one or two GPUs)

Run once before training. All model stages below use this exact selected manifest.

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
if [[ "${REUSE_PREPARED:-0}" == "1" ]]; then
  python - <<'PY'
import os
from data.manifest import verify_manifest_gate
result = verify_manifest_gate(os.environ['MANIFEST'], enforce_class_coverage=True, require_hashes=True)
print(f"Reusing verified manifest: {os.environ['MANIFEST']} ({result['samples']:,} total rows)")
print('No selection or image hashing performed; source images must be unchanged.')
PY
  exit 0
fi
extra=()
if [[ "$ALLOW_SHORTFALL" == "1" ]]; then extra+=(--allow_shortfall); fi
python prepare_dataset.py pilot \
  --manifest "$POOL_MANIFEST" \
  --root "$DATA_ROOT" \
  --output "$MANIFEST" \
  --train_size "$TRAIN_SIZE" \
  --frame_cap "$FRAME_CAP" \
  --dev_ratio 0.1 --test_ratio 0.1 \
  --require_groups --seed "$SEED" "${extra[@]}"
python prepare_dataset.py audit \
  --manifest "$MANIFEST" \
  --root "$DATA_ROOT" \
  --output "$RUN_ROOT/audit_report.json" \
  --hash_cache "$OUTPUT_BASE/hashes.sqlite" \
  --hashes
```

Inspect `selected_manifest.report.json` for actual training counts and source/class coverage. The hash audit writes `selected_manifest.verified.json`; stop if selection or auditing fails. Audit all selected splits once; development/test sizes are not capped by `TRAIN_SIZE`, so auditing a full pool can take time.

## 4. Standalone model commands

The blur settings below preserve your old `--blur_prob 0.5 --blur_sig 0.0,3.0`. They are a reproducible starting policy, not proof of improved generalization. Numeric subsets already target balanced classes, so `--class_bal` is omitted. In full mode this leaves natural class counts; adding class-balanced sampling would change sampling frequencies and may sample with replacement.

The main experiment uses **B then C**, followed by Section 5. **A and D** are optional raw-model comparisons; their checkpoints cannot replace the 128-dimensional experts in fusion.


### A. Your original RGB architecture

**One GPU:**

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
python train.py \
  --arch Wang2020Raw \
  --name wangraw --run_id "$RUN_ID" \
  --checkpoints_dir "$RUN_ROOT" \
  --dataroot "$DATA_ROOT" \
  --manifest "$MANIFEST" --manifest_split train \
  --val_manifest "$MANIFEST" --val_manifest_split dev \
  --gpu_ids 0 --batch_size "$SINGLE_BATCH" --grad_accum_steps "$SINGLE_ACCUM" \
  --num_workers 2 --epochs "$EPOCHS" --epochs_decay 0 \
  --optim adam --lr 0.0001 --monitor_metric auc --seed "$SEED" \
  --use_amp --blur_prob 0.5 --blur_sig 0.0,3.0 \
  --wavelet_type haar --wavelet_level 3 --wavelet_log_mode signed_log1p \
  "$INIT_FLAG"
```

**Two GPUs:**

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
torchrun --standalone --nproc_per_node=2 train.py \
  --arch Wang2020Raw \
  --name wangraw --run_id "$RUN_ID" \
  --checkpoints_dir "$RUN_ROOT" \
  --dataroot "$DATA_ROOT" \
  --manifest "$MANIFEST" --manifest_split train \
  --val_manifest "$MANIFEST" --val_manifest_split dev \
  --gpu_ids 0,1 --batch_size "$DUAL_BATCH" --grad_accum_steps "$DUAL_ACCUM" \
  --num_workers 2 --epochs "$EPOCHS" --epochs_decay 0 \
  --optim adam --lr 0.0001 --monitor_metric auc --seed "$SEED" \
  --use_amp --blur_prob 0.5 --blur_sig 0.0,3.0 \
  --wavelet_type haar --wavelet_level 3 --wavelet_log_mode signed_log1p \
  "$INIT_FLAG"
```


### B. RGB expert for fusion

**One GPU:**

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
python train.py \
  --arch Wang2020_128 \
  --name rgb128 --run_id "$RUN_ID" \
  --checkpoints_dir "$RUN_ROOT" \
  --dataroot "$DATA_ROOT" \
  --manifest "$MANIFEST" --manifest_split train \
  --val_manifest "$MANIFEST" --val_manifest_split dev \
  --gpu_ids 0 --batch_size "$SINGLE_BATCH" --grad_accum_steps "$SINGLE_ACCUM" \
  --num_workers 2 --epochs "$EPOCHS" --epochs_decay 0 \
  --optim adam --lr 0.0001 --monitor_metric auc --seed "$SEED" \
  --use_amp --blur_prob 0.5 --blur_sig 0.0,3.0 \
  --wavelet_type haar --wavelet_level 3 --wavelet_log_mode signed_log1p \
  "$INIT_FLAG"
```

**Two GPUs:**

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
torchrun --standalone --nproc_per_node=2 train.py \
  --arch Wang2020_128 \
  --name rgb128 --run_id "$RUN_ID" \
  --checkpoints_dir "$RUN_ROOT" \
  --dataroot "$DATA_ROOT" \
  --manifest "$MANIFEST" --manifest_split train \
  --val_manifest "$MANIFEST" --val_manifest_split dev \
  --gpu_ids 0,1 --batch_size "$DUAL_BATCH" --grad_accum_steps "$DUAL_ACCUM" \
  --num_workers 2 --epochs "$EPOCHS" --epochs_decay 0 \
  --optim adam --lr 0.0001 --monitor_metric auc --seed "$SEED" \
  --use_amp --blur_prob 0.5 --blur_sig 0.0,3.0 \
  --wavelet_type haar --wavelet_level 3 --wavelet_log_mode signed_log1p \
  "$INIT_FLAG"
```


### C. Wavelet expert for fusion

**One GPU:**

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
python train.py \
  --arch WolterWavelet2021_128 \
  --name wavelet128 --run_id "$RUN_ID" \
  --checkpoints_dir "$RUN_ROOT" \
  --dataroot "$DATA_ROOT" \
  --manifest "$MANIFEST" --manifest_split train \
  --val_manifest "$MANIFEST" --val_manifest_split dev \
  --gpu_ids 0 --batch_size "$SINGLE_BATCH" --grad_accum_steps "$SINGLE_ACCUM" \
  --num_workers 2 --epochs "$EPOCHS" --epochs_decay 0 \
  --optim adam --lr 0.0001 --monitor_metric auc --seed "$SEED" \
  --use_amp --blur_prob 0.5 --blur_sig 0.0,3.0 \
  --wavelet_type haar --wavelet_level 3 --wavelet_log_mode signed_log1p \
  --no-pretrained
```

**Two GPUs:**

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
torchrun --standalone --nproc_per_node=2 train.py \
  --arch WolterWavelet2021_128 \
  --name wavelet128 --run_id "$RUN_ID" \
  --checkpoints_dir "$RUN_ROOT" \
  --dataroot "$DATA_ROOT" \
  --manifest "$MANIFEST" --manifest_split train \
  --val_manifest "$MANIFEST" --val_manifest_split dev \
  --gpu_ids 0,1 --batch_size "$DUAL_BATCH" --grad_accum_steps "$DUAL_ACCUM" \
  --num_workers 2 --epochs "$EPOCHS" --epochs_decay 0 \
  --optim adam --lr 0.0001 --monitor_metric auc --seed "$SEED" \
  --use_amp --blur_prob 0.5 --blur_sig 0.0,3.0 \
  --wavelet_type haar --wavelet_level 3 --wavelet_log_mode signed_log1p \
  --no-pretrained
```


### D. Optional raw wavelet baseline

**One GPU:**

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
python train.py \
  --arch WolterWavelet2021Raw \
  --name waveletraw --run_id "$RUN_ID" \
  --checkpoints_dir "$RUN_ROOT" \
  --dataroot "$DATA_ROOT" \
  --manifest "$MANIFEST" --manifest_split train \
  --val_manifest "$MANIFEST" --val_manifest_split dev \
  --gpu_ids 0 --batch_size "$SINGLE_BATCH" --grad_accum_steps "$SINGLE_ACCUM" \
  --num_workers 2 --epochs "$EPOCHS" --epochs_decay 0 \
  --optim adam --lr 0.0001 --monitor_metric auc --seed "$SEED" \
  --use_amp --blur_prob 0.5 --blur_sig 0.0,3.0 \
  --wavelet_type haar --wavelet_level 3 --wavelet_log_mode signed_log1p \
  --no-pretrained
```

**Two GPUs:**

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
torchrun --standalone --nproc_per_node=2 train.py \
  --arch WolterWavelet2021Raw \
  --name waveletraw --run_id "$RUN_ID" \
  --checkpoints_dir "$RUN_ROOT" \
  --dataroot "$DATA_ROOT" \
  --manifest "$MANIFEST" --manifest_split train \
  --val_manifest "$MANIFEST" --val_manifest_split dev \
  --gpu_ids 0,1 --batch_size "$DUAL_BATCH" --grad_accum_steps "$DUAL_ACCUM" \
  --num_workers 2 --epochs "$EPOCHS" --epochs_decay 0 \
  --optim adam --lr 0.0001 --monitor_metric auc --seed "$SEED" \
  --use_amp --blur_prob 0.5 --blur_sig 0.0,3.0 \
  --wavelet_type haar --wavelet_level 3 --wavelet_log_mode signed_log1p \
  --no-pretrained
```


## 5. Fusion commands: train experts first

These commands load the fresh `rgb128` and `wavelet128` best checkpoints from this experiment. Both must use the same selected manifest and preprocessing. No historical checkpoint is needed. Each head has a separate output name so you can compare all three using the same frozen experts. Choose one GPU mode consistently for the experiment.


### E. Token-attention fusion

**One GPU:**

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
test -f "$RGB_CKPT"
test -f "$WAV_CKPT"
python train.py \
  --arch MHA_128 \
  --name token --run_id "$RUN_ID" \
  --checkpoints_dir "$RUN_ROOT" \
  --dataroot "$DATA_ROOT" \
  --manifest "$MANIFEST" --manifest_split train \
  --val_manifest "$MANIFEST" --val_manifest_split dev \
  --gpu_ids 0 --batch_size "$SINGLE_BATCH" --grad_accum_steps "$SINGLE_ACCUM" \
  --num_workers 2 --epochs "$EPOCHS" --epochs_decay 0 \
  --optim adam --lr 0.0001 --monitor_metric auc --seed "$SEED" \
  --use_amp --blur_prob 0.5 --blur_sig 0.0,3.0 \
  --wavelet_type haar --wavelet_level 3 --wavelet_log_mode signed_log1p \
  --no-pretrained \
  --fusion_type token_attention \
  --freeze_base_models \
  --rgb_model_path "$RGB_CKPT" --wavelet_model_path "$WAV_CKPT"
```

**Two GPUs:**

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
test -f "$RGB_CKPT"
test -f "$WAV_CKPT"
torchrun --standalone --nproc_per_node=2 train.py \
  --arch MHA_128 \
  --name token --run_id "$RUN_ID" \
  --checkpoints_dir "$RUN_ROOT" \
  --dataroot "$DATA_ROOT" \
  --manifest "$MANIFEST" --manifest_split train \
  --val_manifest "$MANIFEST" --val_manifest_split dev \
  --gpu_ids 0,1 --batch_size "$DUAL_BATCH" --grad_accum_steps "$DUAL_ACCUM" \
  --num_workers 2 --epochs "$EPOCHS" --epochs_decay 0 \
  --optim adam --lr 0.0001 --monitor_metric auc --seed "$SEED" \
  --use_amp --blur_prob 0.5 --blur_sig 0.0,3.0 \
  --wavelet_type haar --wavelet_level 3 --wavelet_log_mode signed_log1p \
  --no-pretrained \
  --fusion_type token_attention \
  --freeze_base_models \
  --rgb_model_path "$RGB_CKPT" --wavelet_model_path "$WAV_CKPT"
```


### F. Gated fusion comparison

**One GPU:**

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
test -f "$RGB_CKPT"
test -f "$WAV_CKPT"
python train.py \
  --arch Fusion_128 \
  --name gated --run_id "$RUN_ID" \
  --checkpoints_dir "$RUN_ROOT" \
  --dataroot "$DATA_ROOT" \
  --manifest "$MANIFEST" --manifest_split train \
  --val_manifest "$MANIFEST" --val_manifest_split dev \
  --gpu_ids 0 --batch_size "$SINGLE_BATCH" --grad_accum_steps "$SINGLE_ACCUM" \
  --num_workers 2 --epochs "$EPOCHS" --epochs_decay 0 \
  --optim adam --lr 0.0001 --monitor_metric auc --seed "$SEED" \
  --use_amp --blur_prob 0.5 --blur_sig 0.0,3.0 \
  --wavelet_type haar --wavelet_level 3 --wavelet_log_mode signed_log1p \
  --no-pretrained \
  --fusion_type gated \
  --freeze_base_models \
  --rgb_model_path "$RGB_CKPT" --wavelet_model_path "$WAV_CKPT"
```

**Two GPUs:**

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
test -f "$RGB_CKPT"
test -f "$WAV_CKPT"
torchrun --standalone --nproc_per_node=2 train.py \
  --arch Fusion_128 \
  --name gated --run_id "$RUN_ID" \
  --checkpoints_dir "$RUN_ROOT" \
  --dataroot "$DATA_ROOT" \
  --manifest "$MANIFEST" --manifest_split train \
  --val_manifest "$MANIFEST" --val_manifest_split dev \
  --gpu_ids 0,1 --batch_size "$DUAL_BATCH" --grad_accum_steps "$DUAL_ACCUM" \
  --num_workers 2 --epochs "$EPOCHS" --epochs_decay 0 \
  --optim adam --lr 0.0001 --monitor_metric auc --seed "$SEED" \
  --use_amp --blur_prob 0.5 --blur_sig 0.0,3.0 \
  --wavelet_type haar --wavelet_level 3 --wavelet_log_mode signed_log1p \
  --no-pretrained \
  --fusion_type gated \
  --freeze_base_models \
  --rgb_model_path "$RGB_CKPT" --wavelet_model_path "$WAV_CKPT"
```


### G. Concatenation fusion comparison

**One GPU:**

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
test -f "$RGB_CKPT"
test -f "$WAV_CKPT"
python train.py \
  --arch Fusion_128 \
  --name concat --run_id "$RUN_ID" \
  --checkpoints_dir "$RUN_ROOT" \
  --dataroot "$DATA_ROOT" \
  --manifest "$MANIFEST" --manifest_split train \
  --val_manifest "$MANIFEST" --val_manifest_split dev \
  --gpu_ids 0 --batch_size "$SINGLE_BATCH" --grad_accum_steps "$SINGLE_ACCUM" \
  --num_workers 2 --epochs "$EPOCHS" --epochs_decay 0 \
  --optim adam --lr 0.0001 --monitor_metric auc --seed "$SEED" \
  --use_amp --blur_prob 0.5 --blur_sig 0.0,3.0 \
  --wavelet_type haar --wavelet_level 3 --wavelet_log_mode signed_log1p \
  --no-pretrained \
  --fusion_type concat \
  --freeze_base_models \
  --rgb_model_path "$RGB_CKPT" --wavelet_model_path "$WAV_CKPT"
```

**Two GPUs:**

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
test -f "$RGB_CKPT"
test -f "$WAV_CKPT"
torchrun --standalone --nproc_per_node=2 train.py \
  --arch Fusion_128 \
  --name concat --run_id "$RUN_ID" \
  --checkpoints_dir "$RUN_ROOT" \
  --dataroot "$DATA_ROOT" \
  --manifest "$MANIFEST" --manifest_split train \
  --val_manifest "$MANIFEST" --val_manifest_split dev \
  --gpu_ids 0,1 --batch_size "$DUAL_BATCH" --grad_accum_steps "$DUAL_ACCUM" \
  --num_workers 2 --epochs "$EPOCHS" --epochs_decay 0 \
  --optim adam --lr 0.0001 --monitor_metric auc --seed "$SEED" \
  --use_amp --blur_prob 0.5 --blur_sig 0.0,3.0 \
  --wavelet_type haar --wavelet_level 3 --wavelet_log_mode signed_log1p \
  --no-pretrained \
  --fusion_type concat \
  --freeze_base_models \
  --rgb_model_path "$RGB_CKPT" --wavelet_model_path "$WAV_CKPT"
```


## 6. Resume an interrupted or completed stage

Use the original environment values and existing run directory. Do not rerun setup or preparation. Keep the architecture, manifest, expert paths, batch/accumulation, preprocessing, and initialization settings unchanged. `--epochs` is the desired **total**, not the number of extra epochs.

For example, after token fusion has trained for 2 epochs, set a total of 4:

```python
os.environ['EPOCHS'] = '4'
```

Then run ONE of the following. A `last.pth` file from that stage must exist.


**Resume token fusion on one GPU:**

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
test -f "$RGB_CKPT"
test -f "$WAV_CKPT"
python train.py \
  --arch MHA_128 \
  --name token --run_id "$RUN_ID" \
  --checkpoints_dir "$RUN_ROOT" \
  --dataroot "$DATA_ROOT" \
  --manifest "$MANIFEST" --manifest_split train \
  --val_manifest "$MANIFEST" --val_manifest_split dev \
  --gpu_ids 0 --batch_size "$SINGLE_BATCH" --grad_accum_steps "$SINGLE_ACCUM" \
  --num_workers 2 --epochs "$EPOCHS" --epochs_decay 0 \
  --optim adam --lr 0.0001 --monitor_metric auc --seed "$SEED" \
  --use_amp --blur_prob 0.5 --blur_sig 0.0,3.0 \
  --wavelet_type haar --wavelet_level 3 --wavelet_log_mode signed_log1p \
  --no-pretrained \
  --fusion_type token_attention \
  --freeze_base_models \
  --rgb_model_path "$RGB_CKPT" --wavelet_model_path "$WAV_CKPT" \
  --continue_train \
  --resume_checkpoint "$RUN_ROOT/token_$RUN_ID/checkpoints/last.pth"
```


**Resume token fusion on two GPUs:**

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
test -f "$RGB_CKPT"
test -f "$WAV_CKPT"
torchrun --standalone --nproc_per_node=2 train.py \
  --arch MHA_128 \
  --name token --run_id "$RUN_ID" \
  --checkpoints_dir "$RUN_ROOT" \
  --dataroot "$DATA_ROOT" \
  --manifest "$MANIFEST" --manifest_split train \
  --val_manifest "$MANIFEST" --val_manifest_split dev \
  --gpu_ids 0,1 --batch_size "$DUAL_BATCH" --grad_accum_steps "$DUAL_ACCUM" \
  --num_workers 2 --epochs "$EPOCHS" --epochs_decay 0 \
  --optim adam --lr 0.0001 --monitor_metric auc --seed "$SEED" \
  --use_amp --blur_prob 0.5 --blur_sig 0.0,3.0 \
  --wavelet_type haar --wavelet_level 3 --wavelet_log_mode signed_log1p \
  --no-pretrained \
  --fusion_type token_attention \
  --freeze_base_models \
  --rgb_model_path "$RGB_CKPT" --wavelet_model_path "$WAV_CKPT" \
  --continue_train \
  --resume_checkpoint "$RUN_ROOT/token_$RUN_ID/checkpoints/last.pth"
```


For RGB, wavelet, gated, or concat resume: copy that stage's original command unchanged, set the new total epochs, and append `--continue_train` plus `--resume_checkpoint` using its `last.pth` below. Keep the same GPU count for these continuation commands; cross-device-count reproducibility is not established.

| Stage | Last checkpoint (under `RUN_ROOT`) |
|---|---|
| Original RGB | `wangraw_seed42/checkpoints/last.pth` |
| RGB-128 | `rgb128_seed42/checkpoints/last.pth` |
| Wavelet-128 | `wavelet128_seed42/checkpoints/last.pth` |
| Raw wavelet | `waveletraw_seed42/checkpoints/last.pth` |
| Token fusion | `token_seed42/checkpoints/last.pth` |
| Gated fusion | `gated_seed42/checkpoints/last.pth` |
| Concat fusion | `concat_seed42/checkpoints/last.pth` |

The table assumes `SEED=42`; substitute the recorded `RUN_ID` otherwise. `best.pth` is the development-selected checkpoint for later evaluation; `last.pth` is the latest complete epoch for continuation. Do not resume from an old options placeholder.

## 7. What changed from your old command

- `Wang2020Raw` remains available in Section 4A. The fusion pipeline uses `Wang2020_128`, not the raw checkpoint.
- `DATA_ROOT` now points to the shared parent folder containing train and val. The audited manifest determines which files and explicit labels belong to each split. `--val_root` is replaced here by `--val_manifest` and `--val_manifest_split dev`.
- One GPU uses `python`; two GPUs use `torchrun --standalone --nproc_per_node=2`. `--gpu_ids 0,1` alone does not launch distributed workers.
- Per-GPU batch and accumulation are explicit. Training still uses AMP and your blur settings.
- Size selection happens once in `prepare_dataset.py`; do not pass `--train_size` to `train.py`.
- The setup creates an isolated experiment directory so 100K, 200K, 400K, and full-data checkpoints do not overwrite one another.

Only the dataset size changes between size experiments; reuse the same original images and full-pool manifest. Development/test files stay separate from training. Run the relevant deferred correctness tests before expensive one-/two-GPU training; command validation alone does not establish numerical correctness or external accuracy.
