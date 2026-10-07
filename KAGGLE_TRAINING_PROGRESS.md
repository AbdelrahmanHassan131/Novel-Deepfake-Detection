# Visible training startup and a persistent console log

Twenty minutes without output does not establish that training has started or that it has hung. Previously, worker output could be buffered; imports ran before the first banner, and training's own manifest checks did not request progress output.

Your recovered selection has 98,590 training and 209,871 development images. Both distributed workers currently check metadata and image paths before starting batches. Loader construction reads the manifests again. This is distinct from the earlier image-content hash audit, which is not requested by your training command. Reading many file paths can still be slow. These changes expose progress; they do not claim to eliminate that work or establish the cause of your current delay.

## Update and restart the training launch safely

1. Interrupt the currently running training cell before launching another copy. Keep the dataset, prepared folder, and training outputs.
2. Update your Kaggle clone with these repository changes. Do not replace source files underneath workers you leave running.
3. Keep the existing environment variables, especially the recovered `MANIFEST`. No inventory or recovery rerun is needed.
4. If `rgb128_<RUN_ID>` already exists under `RUN_ROOT`, inspect its checkpoints. Use the documented resume command if `last.pth` exists and you want to continue it. Otherwise choose a fresh training-output directory using setup with `PREPARED_MANIFEST` set to the recovered CSV. Do not delete existing results or bypass run-collision protection.
5. Run the cell below. It requests GPU status, launches unbuffered workers, prints every ten training batches, and appends all output to `rgb128_console.log`. A launch failure still fails the cell because `pipefail` is enabled.

```bash
%%bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT"

echo "Launching RGB-128 training; manifest: $MANIFEST"
echo "Console log: $RUN_ROOT/rgb128_console.log"
nvidia-smi

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
  --log_freq 10 \
  "$INIT_FLAG" 2>&1 | tee -a "$RUN_ROOT/rgb128_console.log"
```

The initial `echo` lines should appear before Python training starts. If even those do not appear, verify the notebook cell is actually running rather than queued; the problem is earlier than model training. The script writes to the log regardless of notebook rendering. A separate terminal/file browser can inspect the log during execution; another cell in the same busy notebook kernel may simply queue.

## Interpret the last visible stage

| Last message | What is happening or waiting |
|---|---|
| `Loading PyTorch...` | Worker interpreter started and is importing PyTorch |
| `PyTorch loaded; loading project modules...` | Project imports |
| `Parsing options and loading model registry...` | Argument parsing and model-class imports |
| `Checking training/development metadata and image paths...` | Manifest/path checks; no training batches yet |
| `Manifest progress` / `Audit progress` | Rows checked so far; counts should advance |
| `Initializing ... process group` | Distributed synchronization; look for both ranks |
| `Building dataloaders` | Dataset construction, with additional manifest progress |
| `Constructing model` / `Pretrained backbone requested` | Model initialization; ImageNet weights may need downloading if absent |
| `waiting for first batch` | Loader workers reading and transforming images |
| `First batch received` | Forward/backward computation has started |
| `First batch completed` and periodic loss | Training is executing batches |
| `Validation` | Validation across the large development set; local batch counts printed |

If a stage stops advancing, share the last 30–50 log lines and any errors. This allows diagnosis of the actual stage rather than assuming a GPU problem. Do not change the pretrained initialization policy merely to hide a possible download delay; choose cached/local weights or a deliberate scratch experiment if needed.

Validation here: Python/command syntax checked only. No imports of training modules, model construction, GPU runs, training, inference, or dataset scans were executed locally.
