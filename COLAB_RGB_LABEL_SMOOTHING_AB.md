# Colab: Matched 3-Epoch A/B Evaluation (RGB Label Smoothing vs Control)

This guide executes a matched, controlled 3-epoch A/B comparison between:
- **Arm A (Control):** `Wang2020_128` baseline with hard targets ($s = 0.0$).
- **Arm B (Smoothing):** `Wang2020_128` with label smoothing ($s = 0.05$, target shrinkage $y_{\text{smooth}} = y(1 - 2s) + s$).

Both arms run for **3 epochs** under an identical configuration on `DatesetNoVideosNoGAN`, use internal dev logit AUC for checkpoint selection, calibrate a decision threshold on their own internal development logits, and evaluate **once** on the 60,000 external development images in FP32.

> [!IMPORTANT]
> **Push Before Running:** Local code edits must be committed and pushed to the remote repository branch (`GeneralizationDeepFakeDetection`) by the user before running in Google Colab. The assistant cannot push to GitHub without authorization.
>
> **Evidence Boundary:** This experiment measures whether label smoothing alleviates logit overconfidence and improves generalization on the previously inspected external collection (`val to be deleted 2`). It does not establish that the training split is free of identity leakage or repair dataset source bias. No 85% accuracy is promised.

---

## Cell 1: Environment Setup and Verification (Python)

Run this cell first in a GPU runtime. It mounts Google Drive, verifies the GPU, updates or clones the repository, checks data paths, and validates all CLI options.

```python
from google.colab import drive
drive.mount('/content/drive')

import os, sys, json, subprocess
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

# 0. Preflight dependency verification
for dep in ('torch', 'torchvision', 'sklearn', 'scipy', 'matplotlib', 'PIL'):
    __import__(dep)

# 1. Repository checkout
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
    if REPO_ROOT.exists() and any(REPO_ROOT.iterdir()):
        raise RuntimeError(f'Existing non-repository directory: {REPO_ROOT}')
    print(f'Cloning fresh checkout from {REPO_URL} (branch: {BRANCH})...')
    subprocess.run([
        'git', 'clone', '--depth', '1', '--branch', BRANCH,
        REPO_URL, str(REPO_ROOT)
    ], check=True)
else:
    print(f'Using existing repository at {REPO_ROOT}; verifying status and updating...')
    # Fail fast if there are uncommitted changes
    status_res = subprocess.run(['git', 'status', '--porcelain'], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    if status_res.stdout.strip():
        raise RuntimeError(f'Repository at {REPO_ROOT} has uncommitted changes:\n{status_res.stdout.strip()}\nCommit or stash before continuing.')
    # Verify/switch branch
    branch_res = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    curr_branch = branch_res.stdout.strip()
    if curr_branch != BRANCH:
        subprocess.run(['git', 'checkout', BRANCH], cwd=REPO_ROOT, check=True)
    # Fast-forward only; fail fast if fetch or pull cannot fast-forward
    subprocess.run(['git', 'fetch', 'origin', BRANCH], cwd=REPO_ROOT, check=True)
    subprocess.run(['git', 'pull', '--ff-only', 'origin', BRANCH], cwd=REPO_ROOT, check=True)

# Ensure repository root is in Python kernel path
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 2. Dataset validation
DATA_ROOT = Path('/content/dataset/DatesetNoVideosNoGAN')
UNSEEN_ROOT = Path('/content/unseenDataset/val to be deleted 2')
for split_path in (DATA_ROOT / 'train', DATA_ROOT / 'val', UNSEEN_ROOT):
    for label in ('real', 'fake'):
        assert (split_path / label).is_dir(), f'Missing required directory: {split_path / label}'

# 3. Output directory setup
OUTPUT_BASE = Path('/content/drive/MyDrive/deepfake_checkpoints_Gen_Third_Try/label_smoothing_ab')
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
(OUTPUT_BASE / 'launch_logs').mkdir(exist_ok=True)

EXP_TIMESTAMP = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
SHARED_SEED = 42

RUN_NAME_CONTROL = 'rgb_control_s00_3ep'
RUN_ID_CONTROL = f'seed{SHARED_SEED}_{EXP_TIMESTAMP}_ctrl_{uuid4().hex[:6]}'
RUN_ROOT_CONTROL = OUTPUT_BASE / f'{RUN_NAME_CONTROL}_{RUN_ID_CONTROL}'

RUN_NAME_SMOOTH = 'rgb_smooth_s05_3ep'
RUN_ID_SMOOTH = f'seed{SHARED_SEED}_{EXP_TIMESTAMP}_smth_{uuid4().hex[:6]}'
RUN_ROOT_SMOOTH = OUTPUT_BASE / f'{RUN_NAME_SMOOTH}_{RUN_ID_SMOOTH}'

assert not RUN_ROOT_CONTROL.exists(), f'Control run directory already exists: {RUN_ROOT_CONTROL}'
assert not RUN_ROOT_SMOOTH.exists(), f'Smoothing run directory already exists: {RUN_ROOT_SMOOTH}'

env_vars = {
    'REPO_ROOT': str(REPO_ROOT),
    'DATA_ROOT': str(DATA_ROOT),
    'UNSEEN_ROOT': str(UNSEEN_ROOT),
    'OUTPUT_BASE': str(OUTPUT_BASE),
    'EXP_TIMESTAMP': EXP_TIMESTAMP,
    'RUN_NAME_CONTROL': RUN_NAME_CONTROL,
    'RUN_ID_CONTROL': RUN_ID_CONTROL,
    'RUN_ROOT_CONTROL': str(RUN_ROOT_CONTROL),
    'RUN_NAME_SMOOTH': RUN_NAME_SMOOTH,
    'RUN_ID_SMOOTH': RUN_ID_SMOOTH,
    'RUN_ROOT_SMOOTH': str(RUN_ROOT_SMOOTH),
}
os.environ.update(env_vars)

# 4. Environment & Flag verification
import torch
assert torch.cuda.is_available(), 'GPU runtime is required. Switch runtime type to GPU.'
gpu_name = torch.cuda.get_device_name(0)

help_output = subprocess.check_output([sys.executable, 'train.py', '--help'], cwd=REPO_ROOT, text=True)
for req_flag in ('--rgb_label_smoothing', '--fine_tune_policy', '--bn_policy', '--backbone_lr_mult', '--use_amp', '--amp_dtype'):
    assert req_flag in help_output, f'Repository is missing required flag: {req_flag}. Ensure code is pushed!'

commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
print('=' * 70)
print('SETUP VERIFICATION SUCCESSFUL')
print(f'GPU:             {gpu_name}')
print(f'Git Commit:      {commit_hash}')
print(f'Repository:      {REPO_ROOT}')
print(f'Output Base:     {OUTPUT_BASE}')
print(f'Control Target:  {RUN_ROOT_CONTROL}')
print(f'Smoothing Target:{RUN_ROOT_SMOOTH}')
print('=' * 70)

setup_metadata = dict(env_vars, gpu_name=gpu_name, git_commit=commit_hash, timestamp=EXP_TIMESTAMP)
(OUTPUT_BASE / 'launch_logs' / f'setup_{EXP_TIMESTAMP}.json').write_text(json.dumps(setup_metadata, indent=2))
(OUTPUT_BASE / 'launch_logs' / 'latest_active_run_ids.json').write_text(json.dumps(setup_metadata, indent=2))
```

---

## Cell 2: Train Arm A — Control ($s = 0.0$, 3 Epochs) (Bash)

This trains the 3-epoch baseline with hard targets ($s = 0.0$). Unbuffered streaming output is displayed in real-time.

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"

echo "=== STARTING ARM A: CONTROL (s = 0.0, 3 Epochs) ==="
PYTHONUNBUFFERED=1 python -u train.py \
  --arch Wang2020_128 --pretrained \
  --dataroot "$DATA_ROOT/train" --val_root "$DATA_ROOT/val" \
  --allow_folder_training \
  --name "$RUN_NAME_CONTROL" --run_id "$RUN_ID_CONTROL" --checkpoints_dir "$OUTPUT_BASE" \
  --rgb_head_type 128d --rgb_dropout 0.5 \
  --fine_tune_policy layer4_and_head --bn_policy frozen --backbone_lr_mult 0.1 \
  --rgb_label_smoothing 0.0 \
  --gpu_ids 0 --batch_size 64 --grad_accum_steps 1 --val_batch_size 64 \
  --num_workers 2 --prefetch_factor 2 --persistent_workers --pin_memory \
  --epochs 3 --epochs_decay 0 --no-early_stopping \
  --optim adam --beta1 0.9 --weight_decay 0 --lr 0.0001 --lr_policy none \
  --aug_recipe rgb_v1 --val_precision fp32 --seed 42 \
  --use_amp --amp_dtype fp16 \
  --monitor_metric auc --val_epoch_freq 1 --save_epoch_freq 1 --log_freq 50 \
  2>&1 | tee "$OUTPUT_BASE/launch_logs/${RUN_NAME_CONTROL}_${RUN_ID_CONTROL}.train.log"

test -f "$RUN_ROOT_CONTROL/checkpoints/best.pth"
echo "Arm A completed successfully. Best checkpoint: $RUN_ROOT_CONTROL/checkpoints/best.pth"
```

> [!TIP]
> **OOM Fallback:** If GPU Out-Of-Memory occurs, do not overwrite failed run directories. Re-run Cell 1 to generate fresh run IDs (`RUN_ID_CONTROL`, `RUN_ID_SMOOTH`), preserving failed-run evidence on Drive. Update the batch settings to:
> `--batch_size 32 --grad_accum_steps 2 --val_batch_size 32`. Apply the exact same batch and accumulation settings to both arms.

---

## Cell 3: Train Arm B — Label Smoothing ($s = 0.05$, 3 Epochs) (Bash)

This trains the matched model with training-only label smoothing ($s = 0.05$). Executed in a clean, separate subshell process to ensure identical initial state.

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"

echo "=== STARTING ARM B: LABEL SMOOTHING (s = 0.05, 3 Epochs) ==="
PYTHONUNBUFFERED=1 python -u train.py \
  --arch Wang2020_128 --pretrained \
  --dataroot "$DATA_ROOT/train" --val_root "$DATA_ROOT/val" \
  --allow_folder_training \
  --name "$RUN_NAME_SMOOTH" --run_id "$RUN_ID_SMOOTH" --checkpoints_dir "$OUTPUT_BASE" \
  --rgb_head_type 128d --rgb_dropout 0.5 \
  --fine_tune_policy layer4_and_head --bn_policy frozen --backbone_lr_mult 0.1 \
  --rgb_label_smoothing 0.05 \
  --gpu_ids 0 --batch_size 64 --grad_accum_steps 1 --val_batch_size 64 \
  --num_workers 2 --prefetch_factor 2 --persistent_workers --pin_memory \
  --epochs 3 --epochs_decay 0 --no-early_stopping \
  --optim adam --beta1 0.9 --weight_decay 0 --lr 0.0001 --lr_policy none \
  --aug_recipe rgb_v1 --val_precision fp32 --seed 42 \
  --use_amp --amp_dtype fp16 \
  --monitor_metric auc --val_epoch_freq 1 --save_epoch_freq 1 --log_freq 50 \
  2>&1 | tee "$OUTPUT_BASE/launch_logs/${RUN_NAME_SMOOTH}_${RUN_ID_SMOOTH}.train.log"

test -f "$RUN_ROOT_SMOOTH/checkpoints/best.pth"
echo "Arm B completed successfully. Best checkpoint: $RUN_ROOT_SMOOTH/checkpoints/best.pth"
```

---

## Cell 4: Calibrate Decision Thresholds on Internal Dev Logits (Bash)

Each run exported its best development validation predictions (with canonical logits and checkpoint hash) into `best_dev_predictions.csv`. Here, we calibrate an optimal Youden J decision boundary in logit space for each checkpoint independently.

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"

echo "=== CALIBRATING LOGIT THRESHOLDS ON INTERNAL DEV SET ==="

# 1. Calibrate Arm A
python analyze_predictions.py calibrate \
  --predictions "$RUN_ROOT_CONTROL/checkpoints/best_dev_predictions.csv" \
  --output "$RUN_ROOT_CONTROL/checkpoints/threshold_logit.json" \
  --score_space logit \
  --eval_precision fp32

# 2. Calibrate Arm B
python analyze_predictions.py calibrate \
  --predictions "$RUN_ROOT_SMOOTH/checkpoints/best_dev_predictions.csv" \
  --output "$RUN_ROOT_SMOOTH/checkpoints/threshold_logit.json" \
  --score_space logit \
  --eval_precision fp32

echo "Arm A threshold artifact: $RUN_ROOT_CONTROL/checkpoints/threshold_logit.json"
echo "Arm B threshold artifact: $RUN_ROOT_SMOOTH/checkpoints/threshold_logit.json"
```

---

## Cell 5: External Inference — Single Pass per Model in FP32 (Bash)

Each model runs forward-pass inference **exactly once** on the 60,000 external development images in FP32 precision.

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"

echo "=== RUNNING SINGLE-PASS FP32 INFERENCE ON EXTERNAL DATASET ==="

# 1. Arm A Inference
test -f "$RUN_ROOT_CONTROL/checkpoints/best.pth"
test ! -e "$RUN_ROOT_CONTROL/external_eval"
PYTHONUNBUFFERED=1 python -u evaluate.py \
  --checkpoint "$RUN_ROOT_CONTROL/checkpoints/best.pth" --arch Wang2020_128 \
  --val_root "$UNSEEN_ROOT" --split external_dev \
  --output_dir "$RUN_ROOT_CONTROL/external_eval" --device cuda:0 \
  --batch_size 32 --num_workers 0 --eval_precision fp32 --bootstrap 0 \
  --no_plots --no_tsne --no_gradcam --no_profiling \
  2>&1 | tee "$OUTPUT_BASE/launch_logs/${RUN_NAME_CONTROL}_${RUN_ID_CONTROL}.eval.log"

# 2. Arm B Inference
test -f "$RUN_ROOT_SMOOTH/checkpoints/best.pth"
test ! -e "$RUN_ROOT_SMOOTH/external_eval"
PYTHONUNBUFFERED=1 python -u evaluate.py \
  --checkpoint "$RUN_ROOT_SMOOTH/checkpoints/best.pth" --arch Wang2020_128 \
  --val_root "$UNSEEN_ROOT" --split external_dev \
  --output_dir "$RUN_ROOT_SMOOTH/external_eval" --device cuda:0 \
  --batch_size 32 --num_workers 0 --eval_precision fp32 --bootstrap 0 \
  --no_plots --no_tsne --no_gradcam --no_profiling \
  2>&1 | tee "$OUTPUT_BASE/launch_logs/${RUN_NAME_SMOOTH}_${RUN_ID_SMOOTH}.eval.log"

test -f "$RUN_ROOT_CONTROL/external_eval/Wang2020_128/predictions.csv"
test -f "$RUN_ROOT_SMOOTH/external_eval/Wang2020_128/predictions.csv"
echo "External predictions exported for both arms."
```

---

## Cell 6: CPU Re-Scoring and Unified A/B Comparison (Python)

This cell loads the exported predictions, computes metrics at both fixed 0.5 and each model's frozen development logit threshold, evaluates logit saturation fractions, generates paired difference bootstraps, and prints the unified side-by-side comparison table.

```python
import os, sys
from pathlib import Path

# Ensure repository root is in Python kernel path
repo_root = os.environ.get('REPO_ROOT')
if repo_root and repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from evaluation.generalization import compare_evaluation_runs

output_base = Path(os.environ['OUTPUT_BASE'])
ctrl_root = Path(os.environ['RUN_ROOT_CONTROL'])
smth_root = Path(os.environ['RUN_ROOT_SMOOTH'])
timestamp = os.environ.get('EXP_TIMESTAMP', 'latest')

json_path = output_base / f'comparison_{timestamp}.json'
md_path = output_base / f'comparison_{timestamp}.md'

comparison = compare_evaluation_runs(
    control_pred_path=str(ctrl_root / 'external_eval/Wang2020_128/predictions.csv'),
    smooth_pred_path=str(smth_root / 'external_eval/Wang2020_128/predictions.csv'),
    control_thresh_path=str(ctrl_root / 'checkpoints/threshold_logit.json'),
    smooth_thresh_path=str(smth_root / 'checkpoints/threshold_logit.json'),
    control_checkpoint_path=str(ctrl_root / 'checkpoints/best.pth'),
    smooth_checkpoint_path=str(smth_root / 'checkpoints/best.pth'),
    output_json_path=str(json_path),
    output_md_path=str(md_path),
    control_name=os.environ['RUN_NAME_CONTROL'],
    smooth_name=os.environ['RUN_NAME_SMOOTH'],
    bootstrap_repeats=500,
    seed=42,
    eval_precision='fp32',
)

print(comparison['markdown_summary'])
print("=" * 70)
print("WHAT TO SEND BACK TO THE ASSISTANT:")
print("1. Copy and paste the Markdown table above.")
print(f"2. Confirm the path of: {json_path}")
print("3. Epoch 3 validation log line for Arm A and Arm B.")
print("=" * 70)
```
