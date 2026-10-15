# Colab: Multi-Domain Training & Evaluation on L4 GPU

This notebook executes training and evaluation on the **newly created 570,570-image multi-domain dataset** (`DatasetNoVideoMultipleDomainAndRealImages.zip`) using an **NVIDIA L4 GPU (24 GB VRAM)**.

### Model Arms:
- **Arm B (Recommended for Generalization):** `Wang2020_128` with **Label Smoothing ($s = 0.05$)**. Prevents logit overconfidence and bounds network activations to improve out-of-domain generalization.
- **Arm A (Baseline Control):** `Wang2020_128` with hard labels ($s = 0.0$). Standard baseline to measure exact gains against.

---

## Cell 1: Environment Setup & Fast Extraction (Python)

```python
from google.colab import drive
drive.mount('/content/drive')

import os, sys, json, subprocess
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

# 0. Dependencies check
for dep in ('torch', 'torchvision', 'sklearn', 'scipy', 'matplotlib', 'PIL'):
    __import__(dep)

# 1. Repository Setup
REPO_URL = 'https://github.com/AbdelrahmanHassan131/Novel-Deepfake-Detection.git'
BRANCH = 'GeneralizationDeepFakeDetection'
candidates = [
    Path('/content/Novel_Deepfake_Detection'),
    Path('/content/code/Novel-Deepfake-Detection'),
    Path('/content/Novel-Deepfake-Detection'),
    Path('/content/drive/MyDrive/Generalization_First_Try_RGB/Novel-Deepfake-Detection'),
]
REPO_ROOT = next((p for p in candidates if (p / 'train.py').is_file()), None)
if REPO_ROOT is None:
    REPO_ROOT = Path('/content/Novel_Deepfake_Detection')
    print(f'Cloning repository from {REPO_URL} (branch: {BRANCH})...')
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', BRANCH, REPO_URL, str(REPO_ROOT)], check=True)
else:
    print(f'Using existing repository at {REPO_ROOT}...')
    subprocess.run(['git', 'fetch', 'origin', BRANCH], cwd=REPO_ROOT, check=True)
    subprocess.run(['git', 'checkout', BRANCH], cwd=REPO_ROOT, check=True)
    subprocess.run(['git', 'pull', '--ff-only', 'origin', BRANCH], cwd=REPO_ROOT, check=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 2. Extract New 570k Multi-Domain Dataset
DATA_ROOT = Path('/content/dataset/DatesetNoVideosNoGAN')
MULTI_ZIP = Path('/content/drive/MyDrive/DataesetNoVideoMultipleDomainAndRealImages/DatasetNoVideoMultipleDomainAndRealImages.zip')

if not (DATA_ROOT / 'train/real').is_dir():
    print(f'Extracting multi-domain dataset from {MULTI_ZIP}...')
    assert MULTI_ZIP.is_file(), f'Multi-domain zip missing: {MULTI_ZIP}'
    Path('/content/dataset').mkdir(parents=True, exist_ok=True)
    subprocess.run(['7z', 'x', str(MULTI_ZIP), '-o/content/dataset/', '-y'], check=True, stdout=subprocess.DEVNULL)
    print('Multi-domain dataset extracted successfully.')

# 3. Extract External Unseen Dataset (60,000 images)
UNSEEN_ROOT = Path('/content/unseenDataset/val to be deleted 2')
UNSEEN_ZIP = Path('/content/drive/MyDrive/UnseenData/val to be deleted 2.zip')

if not (UNSEEN_ROOT / 'real').is_dir():
    print(f'Extracting external unseen dataset from {UNSEEN_ZIP}...')
    assert UNSEEN_ZIP.is_file(), f'Unseen zip missing: {UNSEEN_ZIP}'
    Path('/content/unseenDataset').mkdir(parents=True, exist_ok=True)
    subprocess.run(['7z', 'x', str(UNSEEN_ZIP), '-o/content/unseenDataset/', '-y'], check=True, stdout=subprocess.DEVNULL)
    print('External unseen dataset extracted successfully.')

# Verify dataset directories
for split_path in (DATA_ROOT / 'train', DATA_ROOT / 'val', UNSEEN_ROOT):
    for label in ('real', 'fake'):
        assert (split_path / label).is_dir(), f'Missing directory: {split_path / label}'

print('--- Verified Dataset Image Counts ---')
print(f"Train Real: {len(list((DATA_ROOT / 'train/real').glob('*'))):,}")
print(f"Train Fake: {len(list((DATA_ROOT / 'train/fake').glob('*'))):,}")
print(f"Val Real:   {len(list((DATA_ROOT / 'val/real').glob('*'))):,}")
print(f"Val Fake:   {len(list((DATA_ROOT / 'val/fake').glob('*'))):,}")
print(f"Unseen Real: {len(list((UNSEEN_ROOT / 'real').glob('*'))):,}")
print(f"Unseen Fake: {len(list((UNSEEN_ROOT / 'fake').glob('*'))):,}")

# 4. Checkpoints & Run Config
OUTPUT_BASE = Path('/content/drive/MyDrive/deepfake_checkpoints_Gen_Third_Try/multidomain_smoothing_ab')
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
(OUTPUT_BASE / 'launch_logs').mkdir(exist_ok=True)

EXP_TIMESTAMP = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
SHARED_SEED = 42

RUN_NAME_SMOOTH = 'multidomain_smooth_s05_3ep'
RUN_ID_SMOOTH = f'seed{SHARED_SEED}_{EXP_TIMESTAMP}_smth_{uuid4().hex[:6]}'
RUN_ROOT_SMOOTH = OUTPUT_BASE / f'{RUN_NAME_SMOOTH}_{RUN_ID_SMOOTH}'

RUN_NAME_CONTROL = 'multidomain_ctrl_s00_3ep'
RUN_ID_CONTROL = f'seed{SHARED_SEED}_{EXP_TIMESTAMP}_ctrl_{uuid4().hex[:6]}'
RUN_ROOT_CONTROL = OUTPUT_BASE / f'{RUN_NAME_CONTROL}_{RUN_ID_CONTROL}'

env_vars = {
    'REPO_ROOT': str(REPO_ROOT),
    'DATA_ROOT': str(DATA_ROOT),
    'UNSEEN_ROOT': str(UNSEEN_ROOT),
    'OUTPUT_BASE': str(OUTPUT_BASE),
    'EXP_TIMESTAMP': EXP_TIMESTAMP,
    'RUN_NAME_SMOOTH': RUN_NAME_SMOOTH,
    'RUN_ID_SMOOTH': RUN_ID_SMOOTH,
    'RUN_ROOT_SMOOTH': str(RUN_ROOT_SMOOTH),
    'RUN_NAME_CONTROL': RUN_NAME_CONTROL,
    'RUN_ID_CONTROL': RUN_ID_CONTROL,
    'RUN_ROOT_CONTROL': str(RUN_ROOT_CONTROL),
}
os.environ.update(env_vars)

import torch
assert torch.cuda.is_available(), 'GPU runtime is required. Select L4 GPU.'
gpu_name = torch.cuda.get_device_name(0)

print('=' * 70)
print('SETUP VERIFICATION SUCCESSFUL')
print(f'GPU:             {gpu_name}')
print(f'Smoothing Arm B: {RUN_ROOT_SMOOTH}')
print(f'Control Arm A:   {RUN_ROOT_CONTROL}')
print('=' * 70)
```

---

## Cell 2: Train Arm B — Label Smoothing ($s = 0.05$, Recommended) (Bash)

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"

echo "=== STARTING ARM B: MULTI-DOMAIN + LABEL SMOOTHING (s = 0.05, 3 Epochs) ==="
PYTHONUNBUFFERED=1 python -u train.py \
  --arch Wang2020_128 --pretrained \
  --dataroot "$DATA_ROOT/train" --val_root "$DATA_ROOT/val" \
  --allow_folder_training \
  --name "$RUN_NAME_SMOOTH" --run_id "$RUN_ID_SMOOTH" --checkpoints_dir "$OUTPUT_BASE" \
  --rgb_head_type 128d --rgb_dropout 0.5 \
  --fine_tune_policy layer4_and_head --bn_policy frozen --backbone_lr_mult 0.1 \
  --rgb_label_smoothing 0.05 \
  --gpu_ids 0 --batch_size 64 --grad_accum_steps 1 --val_batch_size 64 \
  --num_workers 4 --prefetch_factor 2 --persistent_workers --pin_memory \
  --epochs 3 --epochs_decay 0 --no-early_stopping \
  --optim adam --beta1 0.9 --weight_decay 0 --lr 0.0001 --lr_policy none \
  --aug_recipe rgb_v1 --val_precision fp32 --seed 42 \
  --use_amp --amp_dtype fp16 \
  --monitor_metric auc --val_epoch_freq 1 --save_epoch_freq 1 --log_freq 100 \
  2>&1 | tee "$OUTPUT_BASE/launch_logs/${RUN_NAME_SMOOTH}_${RUN_ID_SMOOTH}.train.log"

test -f "$RUN_ROOT_SMOOTH/checkpoints/best.pth"
echo "Arm B training completed successfully!"
```

---

## Cell 3 (Optional): Train Arm A — Control ($s = 0.0$) (Bash)

> Run this cell if you wish to conduct a matched A/B comparison between standard hard labels ($s = 0.0$) and smoothed labels ($s = 0.05$).

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"

echo "=== STARTING ARM A: MULTI-DOMAIN + CONTROL (s = 0.0, 3 Epochs) ==="
PYTHONUNBUFFERED=1 python -u train.py \
  --arch Wang2020_128 --pretrained \
  --dataroot "$DATA_ROOT/train" --val_root "$DATA_ROOT/val" \
  --allow_folder_training \
  --name "$RUN_NAME_CONTROL" --run_id "$RUN_ID_CONTROL" --checkpoints_dir "$OUTPUT_BASE" \
  --rgb_head_type 128d --rgb_dropout 0.5 \
  --fine_tune_policy layer4_and_head --bn_policy frozen --backbone_lr_mult 0.1 \
  --rgb_label_smoothing 0.0 \
  --gpu_ids 0 --batch_size 64 --grad_accum_steps 1 --val_batch_size 64 \
  --num_workers 4 --prefetch_factor 2 --persistent_workers --pin_memory \
  --epochs 3 --epochs_decay 0 --no-early_stopping \
  --optim adam --beta1 0.9 --weight_decay 0 --lr 0.0001 --lr_policy none \
  --aug_recipe rgb_v1 --val_precision fp32 --seed 42 \
  --use_amp --amp_dtype fp16 \
  --monitor_metric auc --val_epoch_freq 1 --save_epoch_freq 1 --log_freq 100 \
  2>&1 | tee "$OUTPUT_BASE/launch_logs/${RUN_NAME_CONTROL}_${RUN_ID_CONTROL}.train.log"

test -f "$RUN_ROOT_CONTROL/checkpoints/best.pth"
echo "Arm A training completed successfully!"
```

---

## Cell 4: Calibrate Decision Thresholds on Internal Dev Logits (Bash)

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"

echo "=== CALIBRATING LOGIT THRESHOLDS ON INTERNAL DEV SET ==="

# Calibrate Arm B (Smoothing)
if [ -f "$RUN_ROOT_SMOOTH/checkpoints/best_dev_predictions.csv" ]; then
  python analyze_predictions.py calibrate \
    --predictions "$RUN_ROOT_SMOOTH/checkpoints/best_dev_predictions.csv" \
    --output "$RUN_ROOT_SMOOTH/checkpoints/threshold_logit.json" \
    --score_space logit \
    --eval_precision fp32
  echo "Arm B threshold calibrated."
fi

# Calibrate Arm A (Control, if trained)
if [ -f "$RUN_ROOT_CONTROL/checkpoints/best_dev_predictions.csv" ]; then
  python analyze_predictions.py calibrate \
    --predictions "$RUN_ROOT_CONTROL/checkpoints/best_dev_predictions.csv" \
    --output "$RUN_ROOT_CONTROL/checkpoints/threshold_logit.json" \
    --score_space logit \
    --eval_precision fp32
  echo "Arm A threshold calibrated."
fi
```

---

## Cell 5: External Inference on 60,000 Unseen Images in FP32 (Bash)

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"

echo "=== EVALUATING ON EXTERNAL UNSEEN DATASET (val to be deleted 2) ==="

# 1. Arm B Inference
if [ -f "$RUN_ROOT_SMOOTH/checkpoints/best.pth" ]; then
  echo "Running Arm B External Evaluation..."
  PYTHONUNBUFFERED=1 python -u evaluate.py \
    --checkpoint "$RUN_ROOT_SMOOTH/checkpoints/best.pth" --arch Wang2020_128 \
    --val_root "$UNSEEN_ROOT" --split external_dev \
    --output_dir "$RUN_ROOT_SMOOTH/external_eval" --device cuda:0 \
    --batch_size 64 --num_workers 2 --eval_precision fp32 --bootstrap 0 \
    --no_plots --no_tsne --no_gradcam --no_profiling \
    2>&1 | tee "$OUTPUT_BASE/launch_logs/${RUN_NAME_SMOOTH}_${RUN_ID_SMOOTH}.eval.log"
fi

# 2. Arm A Inference (if trained)
if [ -f "$RUN_ROOT_CONTROL/checkpoints/best.pth" ]; then
  echo "Running Arm A External Evaluation..."
  PYTHONUNBUFFERED=1 python -u evaluate.py \
    --checkpoint "$RUN_ROOT_CONTROL/checkpoints/best.pth" --arch Wang2020_128 \
    --val_root "$UNSEEN_ROOT" --split external_dev \
    --output_dir "$RUN_ROOT_CONTROL/external_eval" --device cuda:0 \
    --batch_size 64 --num_workers 2 --eval_precision fp32 --bootstrap 0 \
    --no_plots --no_tsne --no_gradcam --no_profiling \
    2>&1 | tee "$OUTPUT_BASE/launch_logs/${RUN_NAME_CONTROL}_${RUN_ID_CONTROL}.eval.log"
fi

echo "External evaluation complete."
```

---

## Cell 6: Compare Saved Predictions Using Verified Development Thresholds (Python)

For completed runs, replace ONLY the old summary cell with this cell. Keep the current
`RUN_ROOT_CONTROL` and `RUN_ROOT_SMOOTH`; do not rerun setup, training, calibration or inference.
This reads saved CSVs and checkpoint files on CPU. It does not open images or fit new thresholds.
If the notebook session restarted, restore the exact run paths first; do not choose a run implicitly.

The old cell used `prob` instead of `probability`, read `threshold_logit` instead of
`logit_threshold`, and silently substituted zero. This version uses the repository's
shared comparison function, verifies checkpoint/threshold linkage and reports raw-logit AUC.
The existing 60K collection is external development data, not an untouched final test.

```python
import os, sys
from pathlib import Path
from uuid import uuid4

required = ('REPO_ROOT', 'OUTPUT_BASE', 'RUN_ROOT_CONTROL', 'RUN_ROOT_SMOOTH')
missing = [key for key in required if not os.environ.get(key)]
if missing:
    raise RuntimeError(f'Restore the exact completed-run environment variables: {missing}')
repo = Path(os.environ['REPO_ROOT'])
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))
from evaluation.generalization import compare_evaluation_runs

control = Path(os.environ['RUN_ROOT_CONTROL'])
smoothing = Path(os.environ['RUN_ROOT_SMOOTH'])
output = Path(os.environ['OUTPUT_BASE']) / f'corrected_comparison_{uuid4().hex}'
output.mkdir(parents=True, exist_ok=False)

comparison = compare_evaluation_runs(
    control_pred_path=str(control / 'external_eval/Wang2020_128/predictions.csv'),
    smooth_pred_path=str(smoothing / 'external_eval/Wang2020_128/predictions.csv'),
    control_thresh_path=str(control / 'checkpoints/threshold_logit.json'),
    smooth_thresh_path=str(smoothing / 'checkpoints/threshold_logit.json'),
    control_checkpoint_path=str(control / 'checkpoints/best.pth'),
    smooth_checkpoint_path=str(smoothing / 'checkpoints/best.pth'),
    output_json_path=str(output / 'comparison.json'),
    output_md_path=str(output / 'comparison.md'),
    control_name='multidomain_control_s00',
    smooth_name='multidomain_smoothing_s05',
    bootstrap_repeats=0,
    eval_precision='fp32',
)
print(comparison['markdown_summary'])
print('Saved:', output / 'comparison.json')
print('Send back the table above. No training or image inference was rerun.')
```
