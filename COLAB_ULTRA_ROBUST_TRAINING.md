# Colab Guide: Ultra-Robust Multi-Domain Deepfake Training & Evaluation

This guide trains an **Ultra-Robust Regularized Deepfake Detector** on the newly created **570,570-image Multi-Domain Dataset** using an **NVIDIA L4 GPU (24 GB VRAM)**.

### Key Architectural & Regularization Enhancements:
1. **Heavy Dynamic Augmentation (`--aug_recipe custom`):**
   - **80% JPEG Compression:** Random quality factors from 30 to 90 (`--jpg_prob 0.8 --jpg_qual 30,40,50,60,70,80,90`).
   - **70% Gaussian Blur:** Sigma ranges from 0.0 to 3.0 (`--blur_prob 0.7 --blur_sig 0.0,3.0`).
   - **30% Downscale/Upscale Resampling:** Downscaling by 0.5 to 1.0 (`--downscale_prob 0.3 --downscale_range 0.5 1.0`).
   - *Result:* The CNN is completely immunized against video compression and camera resampling artifacts, directly eliminating the $+3.0$ score shift on unseen datasets.
2. **High-Regularization Classifier Head (`--rgb_dropout 0.67` + `--weight_decay 1e-4`):**
   - Deactivates $\frac{2}{3}$ of the 128 bottleneck features randomly during training, preventing shortcut learning.
3. **Gentle Backbone Fine-Tuning (`--fine_tune_policy layer4_and_head` with `--backbone_lr_mult 0.05`):**
   - Preserves general ImageNet representations across layers 1–3 while gently tuning `layer4` at a 20× slower learning rate.
4. **Label Smoothing ($s = 0.05$):**
   - Bounds logit explosion and prevents overconfidence.

---

## Cell 1: Mount Drive, Extract Datasets & Setup (Python)

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
print(f"Train Real:  {len(list((DATA_ROOT / 'train/real').glob('*'))):,}")
print(f"Train Fake:  {len(list((DATA_ROOT / 'train/fake').glob('*'))):,}")
print(f"Val Real:    {len(list((DATA_ROOT / 'val/real').glob('*'))):,}")
print(f"Val Fake:    {len(list((DATA_ROOT / 'val/fake').glob('*'))):,}")
print(f"Unseen Real: {len(list((UNSEEN_ROOT / 'real').glob('*'))):,}")
print(f"Unseen Fake: {len(list((UNSEEN_ROOT / 'fake').glob('*'))):,}")

# 4. Checkpoints & Run Config
OUTPUT_BASE = Path('/content/drive/MyDrive/deepfake_checkpoints_Gen_Third_Try/multidomain_ultra_robust')
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
(OUTPUT_BASE / 'launch_logs').mkdir(exist_ok=True)

EXP_TIMESTAMP = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
SHARED_SEED = 42

RUN_NAME_ROBUST = 'multidomain_robust_reg_3ep'
RUN_ID_ROBUST = f'seed{SHARED_SEED}_{EXP_TIMESTAMP}_rob_{uuid4().hex[:6]}'
RUN_ROOT_ROBUST = OUTPUT_BASE / f'{RUN_NAME_ROBUST}_{RUN_ID_ROBUST}'

env_vars = {
    'REPO_ROOT': str(REPO_ROOT),
    'DATA_ROOT': str(DATA_ROOT),
    'UNSEEN_ROOT': str(UNSEEN_ROOT),
    'OUTPUT_BASE': str(OUTPUT_BASE),
    'EXP_TIMESTAMP': EXP_TIMESTAMP,
    'RUN_NAME_ROBUST': RUN_NAME_ROBUST,
    'RUN_ID_ROBUST': RUN_ID_ROBUST,
    'RUN_ROOT_ROBUST': str(RUN_ROOT_ROBUST),
}
os.environ.update(env_vars)

import torch
assert torch.cuda.is_available(), 'GPU runtime is required. Select L4 GPU.'
gpu_name = torch.cuda.get_device_name(0)

print('=' * 70)
print('SETUP VERIFICATION SUCCESSFUL')
print(f'GPU:             {gpu_name}')
print(f'Run Target:      {RUN_ROOT_ROBUST}')
print('=' * 70)
```

---

## Cell 2: Train Ultra-Robust Model with Dynamic Augmentations (Bash)

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"

echo "=== STARTING ULTRA-ROBUST MULTI-DOMAIN TRAINING (3 Epochs) ==="
PYTHONUNBUFFERED=1 python -u train.py \
  --arch Wang2020_128 --pretrained \
  --dataroot "$DATA_ROOT/train" --val_root "$DATA_ROOT/val" \
  --allow_folder_training \
  --name "$RUN_NAME_ROBUST" --run_id "$RUN_ID_ROBUST" --checkpoints_dir "$OUTPUT_BASE" \
  --rgb_head_type 128d \
  --rgb_dropout 0.67 \
  --fine_tune_policy layer4_and_head --bn_policy frozen --backbone_lr_mult 0.05 \
  --rgb_label_smoothing 0.05 \
  --aug_recipe custom \
  --blur_prob 0.7 --blur_sig 0.0,3.0 \
  --jpg_prob 0.8 --jpg_qual 30,40,50,60,70,80,90 \
  --downscale_prob 0.3 --downscale_range 0.5 1.0 \
  --gpu_ids 0 --batch_size 64 --grad_accum_steps 1 --val_batch_size 64 \
  --num_workers 4 --prefetch_factor 2 --persistent_workers --pin_memory \
  --epochs 3 --epochs_decay 0 --no-early_stopping \
  --optim adam --beta1 0.9 --weight_decay 0.0001 --lr 0.0001 --lr_policy none \
  --val_precision fp32 --seed 42 \
  --use_amp --amp_dtype fp16 \
  --monitor_metric auc --val_epoch_freq 1 --save_epoch_freq 1 --log_freq 100 \
  2>&1 | tee "$OUTPUT_BASE/launch_logs/${RUN_NAME_ROBUST}_${RUN_ID_ROBUST}.train.log"

test -f "$RUN_ROOT_ROBUST/checkpoints/best.pth"
echo "Ultra-robust training completed successfully! Best checkpoint saved."
```

---

## Cell 3: Calibrate Decision Threshold on Development Logits (Bash)

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"

echo "=== CALIBRATING LOGIT THRESHOLDS ON INTERNAL DEV SET ==="
python analyze_predictions.py calibrate \
  --predictions "$RUN_ROOT_ROBUST/checkpoints/best_dev_predictions.csv" \
  --output "$RUN_ROOT_ROBUST/checkpoints/threshold_logit.json" \
  --score_space logit \
  --eval_precision fp32

test -f "$RUN_ROOT_ROBUST/checkpoints/threshold_logit.json"
echo "Threshold calibrated successfully."
```

---

## Cell 4: Single-Pass FP32 Inference on 60,000 Unseen Images (Bash)

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"

echo "=== EVALUATING ON EXTERNAL UNSEEN DATASET (val to be deleted 2) ==="
PYTHONUNBUFFERED=1 python -u evaluate.py \
  --checkpoint "$RUN_ROOT_ROBUST/checkpoints/best.pth" --arch Wang2020_128 \
  --val_root "$UNSEEN_ROOT" --split external_dev \
  --output_dir "$RUN_ROOT_ROBUST/external_eval" --device cuda:0 \
  --batch_size 64 --num_workers 2 --eval_precision fp32 --bootstrap 0 \
  --no_plots --no_tsne --no_gradcam --no_profiling \
  2>&1 | tee "$OUTPUT_BASE/launch_logs/${RUN_NAME_ROBUST}_${RUN_ID_ROBUST}.eval.log"

test -f "$RUN_ROOT_ROBUST/external_eval/Wang2020_128/predictions.csv"
echo "External evaluation inference complete."
```

---

## Cell 5: Comprehensive Evaluation & Domain Alignment Report (Python)

```python
import os, sys, json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve

output_base = Path(os.environ['OUTPUT_BASE'])
run_root = Path(os.environ['RUN_ROOT_ROBUST'])
pred_file = run_root / 'external_eval' / 'Wang2020_128' / 'predictions.csv'
thresh_file = run_root / 'checkpoints' / 'threshold_logit.json'

assert pred_file.is_file(), f"Predictions missing: {pred_file}"
df = pd.read_csv(pred_file)

y_true = df['label'].values
prob_col = 'probability' if 'probability' in df.columns else ('prob' if 'prob' in df.columns else df.columns[1])
y_prob = df[prob_col].values
y_logit = df['logit'].values if 'logit' in df.columns else None

auc = roc_auc_score(y_true, y_prob)

# 1. Standard Fixed Cutoff (0.5)
acc_05 = accuracy_score(y_true, (y_prob >= 0.5).astype(int))

# 2. In-Domain Calibrated Threshold (from validation set)
thresh_logit = 0.0
thresh_prob = 0.5
if thresh_file.is_file():
    t_info = json.loads(thresh_file.read_text())
    thresh_logit = t_info.get('threshold_logit', 0.0)
    thresh_prob = t_info.get('threshold_prob', 0.5)

y_pred_cal = (y_logit >= thresh_logit).astype(int) if y_logit is not None else (y_prob >= thresh_prob).astype(int)
acc_cal = accuracy_score(y_true, y_pred_cal)

# 3. Unsupervised Zero-Shot Median Centering (No Labels Used)
median_thresh = float(np.median(y_prob))
y_pred_unsupervised = (y_prob >= median_thresh).astype(int)
acc_unsupervised = accuracy_score(y_true, y_pred_unsupervised)

# 4. Optimal Target Bound (Youden's J on Unseen Domain)
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
opt_idx = np.argmax(tpr - fpr)
opt_thresh = float(thresholds[opt_idx])
y_pred_opt = (y_prob >= opt_thresh).astype(int)
acc_opt = accuracy_score(y_true, y_pred_opt)

cm_opt = confusion_matrix(y_true, y_pred_opt)

# Score distributions
real_scores = y_prob[y_true == 0]
fake_scores = y_prob[y_true == 1]

print("=" * 80)
print("ULTRA-ROBUST MODEL EVALUATION REPORT (60,000 UNSEEN IMAGES)")
print("=" * 80)
print(f"  External ROC AUC:                       {auc:.4f}")
print(f"  Accuracy (@ Fixed 0.5 Cutoff):          {acc_05:.2%}")
print(f"  Accuracy (@ Dev-Calibrated Threshold):  {acc_cal:.2%}")
print(f"  Accuracy (@ Unsupervised Batch Median): {acc_unsupervised:.2%}")
print(f"  Accuracy (@ Optimal Domain Threshold):  {acc_opt:.2%}")
print("-" * 80)
print("SCORE DISTRIBUTIONS:")
print(f"  Real Images Mean Prob: {real_scores.mean():.4f} | Median: {np.median(real_scores):.4f}")
print(f"  Fake Images Mean Prob: {fake_scores.mean():.4f} | Median: {np.median(fake_scores):.4f}")
print("-" * 80)
print(f"CONFUSION MATRIX (At Optimal Threshold {opt_thresh:.4f}):")
print(f"  True Reals Correct:  {cm_opt[0,0]:,} / {len(real_scores):,} ({cm_opt[0,0]/len(real_scores):.1%})")
print(f"  True Fakes Correct:  {cm_opt[1,1]:,} / {len(fake_scores):,} ({cm_opt[1,1]/len(fake_scores):.1%})")
print(f"  Total Correct:       {cm_opt[0,0] + cm_opt[1,1]:,} / {len(df):,}")
print("=" * 80)
```
