# Colab: one-epoch conservative RGB retry

Use the current GPU runtime and the existing `DatesetNoVideosNoGAN` dataset. This tests a less aggressive fine-tuning policy on the same folder split; it does not establish that the split is leakage-free or fix source bias. Do not rebuild images or use external evaluation images in training.

The new policy trains only the last ResNet block and classifier, freezes BatchNorm statistics, and uses a backbone learning rate of 0.00001. Keep the previous run's constant learning rate, batch size 64, FP32 training, and rgb_v1 augmentation. One epoch matches the completed checkpoint reported by the user. Seed 42 makes this new run repeatable; the earlier run did not specify a seed, so this is an exploratory comparison, not a perfectly controlled ablation. No 85% accuracy is promised.

## 1. Setup in a Python cell

Mount Drive if needed. Use the existing checkout; this experiment requires no new model code. Only if none of these standard checkout locations exists, clone the pushed branch. Images must already be extracted at the paths below.

```python
from google.colab import drive
drive.mount('/content/drive')

import os, sys, json, subprocess
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

candidates = [
    Path('/content/code/Novel-Deepfake-Detection'),
    Path('/content/Novel_Deepfake_Detection'),
    Path('/content/Novel-Deepfake-Detection'),
    Path('/content/drive/MyDrive/Generalization_First_Try_RGB/Novel-Deepfake-Detection'),
]
REPO_ROOT = next((p for p in candidates if (p / 'train.py').is_file()), None)
if REPO_ROOT is None:
    REPO_ROOT = Path('/content/code/Novel-Deepfake-Detection')
    if REPO_ROOT.exists() and any(REPO_ROOT.iterdir()):
        raise RuntimeError(f'Existing non-repository directory: {REPO_ROOT}')
    subprocess.run([
        'git', 'clone', '--depth', '1', '--branch', 'GeneralizationDeepFakeDetection',
        'https://github.com/AbdelrahmanHassan131/Novel-Deepfake-Detection.git',
        str(REPO_ROOT)
    ], check=True)
else:
    # Update existing clone to latest commit
    subprocess.run(['git', 'pull', 'origin', 'GeneralizationDeepFakeDetection'], cwd=REPO_ROOT, check=False)

DATA_ROOT = Path('/content/dataset/DatesetNoVideosNoGAN')
UNSEEN_ROOT = Path('/content/unseenDataset/val to be deleted 2')
for root in (DATA_ROOT / 'train', DATA_ROOT / 'val', UNSEEN_ROOT):
    for label in ('real', 'fake'):
        assert (root / label).is_dir(), f'Missing image folder: {root / label}'

OUTPUT_BASE = Path('/content/drive/MyDrive/deepfake_checkpoints_Gen_Third_Try')
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
RUN_NAME = 'rgb_conservative_160k'
RUN_ID = 'seed42_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC') + '_' + uuid4().hex[:8]
RUN_ROOT = OUTPUT_BASE / f'{RUN_NAME}_{RUN_ID}'
assert not RUN_ROOT.exists(), f'Run already exists: {RUN_ROOT}'
# The training engine must create RUN_ROOT itself.
(OUTPUT_BASE / 'launch_logs').mkdir(exist_ok=True)
values = dict(REPO_ROOT=REPO_ROOT, DATA_ROOT=DATA_ROOT, UNSEEN_ROOT=UNSEEN_ROOT,
              OUTPUT_BASE=OUTPUT_BASE, RUN_NAME=RUN_NAME, RUN_ID=RUN_ID,
              RUN_ROOT=RUN_ROOT, CHECKPOINT=RUN_ROOT / 'checkpoints/best.pth',
              EVAL_ROOT=RUN_ROOT / 'external_eval')
os.environ.update({k: str(v) for k, v in values.items()})

import torch
assert torch.cuda.is_available(), 'Select a GPU runtime before training.'
print('GPU:', torch.cuda.get_device_name(0))
print('Repository:', REPO_ROOT)
print('New run:', RUN_ROOT)
print('Training split:', DATA_ROOT / 'train')
print('Validation split:', DATA_ROOT / 'val')
print('External evaluation:', UNSEEN_ROOT)

help_text = subprocess.check_output([sys.executable, 'train.py', '--help'], cwd=REPO_ROOT, text=True)
for flag in ('--fine_tune_policy', '--bn_policy', '--backbone_lr_mult', '--run_id'):
    assert flag in help_text, f'Checkout is missing required option: {flag}'
revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True, capture_output=True)
record = {k: str(v) for k, v in values.items()}
record['git_commit'] = revision.stdout.strip() if revision.returncode == 0 else 'unavailable'
(OUTPUT_BASE / 'launch_logs' / f'{RUN_NAME}_{RUN_ID}.setup.json').write_text(json.dumps(record, indent=2))
```

If imports are missing in a fresh runtime, install the repository dependencies before proceeding. In the existing working runtime no reinstall is needed. Do not rerun setup during a training run: it deliberately creates a new run identifier.

## 2. Train once in a new cell

FP32 is intentional to match the previous command, which did not enable AMP. Do not add resume options. This starts from ImageNet weights, not the failed trained checkpoint. Two workers match the CPU limit reported by the user's Colab runtime; worker count is a throughput setting, not a generalization fix.

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
PYTHONUNBUFFERED=1 python -u train.py \
  --arch Wang2020_128 --pretrained \
  --dataroot "$DATA_ROOT/train" --val_root "$DATA_ROOT/val" \
  --allow_folder_training \
  --name "$RUN_NAME" --run_id "$RUN_ID" --checkpoints_dir "$OUTPUT_BASE" \
  --rgb_head_type 128d --rgb_dropout 0.5 \
  --fine_tune_policy layer4_and_head --bn_policy frozen --backbone_lr_mult 0.1 \
  --gpu_ids 0 --batch_size 64 --grad_accum_steps 1 --val_batch_size 64 \
  --num_workers 2 --prefetch_factor 2 --persistent_workers --pin_memory \
  --epochs 1 --epochs_decay 0 --no-early_stopping \
  --optim adam --beta1 0.9 --weight_decay 0 --lr 0.0001 --lr_policy none \
  --aug_recipe rgb_v1 --val_precision fp32 --seed 42 \
  --monitor_metric auc --val_epoch_freq 1 --save_epoch_freq 1 --log_freq 50 \
  2>&1 | tee "$OUTPUT_BASE/launch_logs/${RUN_NAME}_${RUN_ID}.log"
test -f "$CHECKPOINT"
echo "Completed checkpoint: $CHECKPOINT"
```

If you encounter GPU out-of-memory, stop and use a fresh run ID with batch size 32, accumulation 2, and validation batch size 32. Record that change; do not silently overwrite an existing run.

## 3. Evaluate once in a new cell

This uses the fixed 0.5 threshold, exactly as in the preceding external report. It does not fit a threshold to external labels. Plotting and profiling are disabled to save time. The external dataset has already been inspected repeatedly and is external development evidence, not an untouched final test set.

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
test -f "$CHECKPOINT"
test ! -e "$EVAL_ROOT"
PYTHONUNBUFFERED=1 python -u evaluate.py \
  --checkpoint "$CHECKPOINT" --arch Wang2020_128 \
  --val_root "$UNSEEN_ROOT" --split external_dev \
  --output_dir "$EVAL_ROOT" --device cuda:0 \
  --batch_size 32 --num_workers 0 --eval_precision fp32 --bootstrap 0 \
  --no_plots --no_tsne --no_gradcam --no_profiling \
  2>&1 | tee "$OUTPUT_BASE/launch_logs/${RUN_NAME}_${RUN_ID}.evaluation.log"
```

## 4. Print the result in a Python cell

```python
import os, json
from pathlib import Path
p = Path(os.environ['EVAL_ROOT']) / 'Wang2020_128/generalization_report.json'
r = json.loads(p.read_text())
m = r['overall']
for key in ('samples', 'real', 'fake', 'threshold', 'accuracy', 'balanced_accuracy',
            'roc_auc', 'real_recall', 'fake_recall', 'confusion_matrix', 'score_quantiles'):
    print(f'{key}: {m.get(key)}')
print('Previous full-fine-tuning result: accuracy=0.5003, AUC=0.6094145, real recall=0.0006')
print('Report:', p)
print('Saved predictions:', p.parent / 'predictions.csv')
```

Send back this output and the epoch validation line. Assess external AUC together with both class recalls at the unchanged threshold. Strong internal validation alone does not count as success. A single epoch and seed are a bounded diagnostic; a gain would need confirmation before claiming a generalization improvement. This guide does not repair the incomplete duplicate audit or the real-source confound in the current dataset. Those remain required for a defensible final experiment.
