# Google Colab Single-GPU Workflow: Controlled RGB Data-Mixture Pilot

This document provides a complete, copy-pasteable, cell-by-cell workflow for running the controlled RGB data-mixture pilot (Arm A vs. Arm B) on a single Colab GPU (T4, V100, A100, or L4).

**Updated 2026-10-10: fresh Git clone workflow; no code ZIP upload required.** The pilot manifests are tracked in the repository. Cell 3 clones `GeneralizationDeepFakeDetection` into local Colab storage and checks that the required manifests arrived. Start with a fresh runtime or an empty/nonexistent `LOCAL_REPO`; keep your dataset, Drive preparation files and completed model outputs. Do not delete those when replacing the old code checkout.

Run Cells 1–6 once, then Cell 7 with `ARM="arm_a"`, Cell 8 to save it, Cell 7 with `ARM="arm_b"`, Cell 8 again, and finally Cell 9. No manifest regeneration or full-dataset hashing is needed for unchanged images. If the dataset is already at `/content/dataset/Preapred Dataset/train`, Cell 4 reuses it; otherwise put all split dataset archive volumes together in the configured Drive folder. Cloning code does not download the image archive.

## Should we train on 1M images for two epochs instead?

**Recommendation: run the two small controlled arms first.** More diverse, correctly labeled images can help; simply adding more images from the current biased collection is not established as a fix. Two epochs are not inherently too few or sufficient: update count, unique content, learning schedule and domain coverage all matter. We cannot assign a reliable probability of reaching 85% from dataset size alone.

The archive has 1,016,855 images in total, but only **805,421 are in its train directory**; the other 211,434 belong to archive val. The saved cleanup left 779,408 training rows in its older split. Under the new connected-family partition and eight-per-group/class cap, the candidate training pool contains **299,297 images**, before arm quotas. These are different preparation stages, not interchangeable training sizes. Never merge val/dev into training merely to reach 1M.

One million images times two epochs means about two million image presentations. Both proposed 20K arms times five epochs together mean 200,000 presentations: roughly ten times fewer. This is an exposure comparison, not a runtime estimate. The smaller experiment tests whether the mixture change helps before committing that compute. The current guide intentionally trains 20K per arm for five epochs; it does not silently label this as full-data training.

After training, compare both checkpoints on the same external-development dataset using the same evaluation procedure. Check real recall, fake recall, balanced accuracy at 0.5, AUC and separately reported development-calibrated results. Internal development improvement alone is insufficient. If the new mixture helps externally, evaluate a larger *group-respecting, available* cohort as the next experiment; if it does not, prioritize different source coverage or representations rather than scaling the identical recipe blindly. Equal four-stratum quotas are limited by the smallest available stratum (11,270 video-pattern real candidates here), so 100K equal-stratum unique images cannot be promised under these exact caps without changing the protocol.

Related primary evidence: [B-Free](https://openaccess.thecvf.com/content/CVPR2025/html/Guillaro_A_Bias-Free_Training_Paradigm_for_More_General_AI-generated_Image_Detection_CVPR_2025_paper.html) studies content/format/resolution bias; [Community Forensics](https://openaccess.thecvf.com/content/CVPR2025/html/Park_Community_Forensics_Using_Thousands_of_Generators_to_Train_Fake_Image_CVPR_2025_paper.html) studies generator diversity. Neither predicts the improvement for this particular collection.

---

## Protocol & Engineering Summary

- **Hypothesis**: The baseline R2 100K sampling mixture contained 2,753 numeric fake and 11,430 video-pattern real images. Arm A approximates those proportions at 20K (551 numeric fake and 2,286 video real). Arm B has 5,000 across each of the four strata. This tests whether equal representation improves robustness; it is an engineering data-representation pilot.
- **Model**: `Wang2020_128` (ResNet-50 backbone with 128-d projection head, ImageNet pretrained via `--pretrained`).
- **Fine-tuning & Regularization**: `layer4_and_head` fine-tuning, `frozen` batch normalization (eval mode), backbone learning rate multiplier 0.1, head dropout 0.5.
- **Augmentation Recipe**: `rgb_v1` (Gaussian blur prob 0.5, JPEG prob 0.5, qualities 50–95) identical to R2.
- **Exposure Budget**: Exactly 5 epochs without early stopping (`--no-early_stopping`) to guarantee matched sample exposure between arms.
- **Optimizer**: Adam (`lr=0.0001`, `beta1=0.9`, `weight_decay=0.0`, `lr_policy=cosine`).
- **Batching**: Batch size 32, gradient accumulation 2 (effective batch size 64).
- **Evaluation**: Both arms are evaluated on the **exact same 2,000 shared development samples** (500 in each stratum), with zero sample, group, token, or hash overlap with either training set.

---

## Step 1: Connect to GPU and Mount Google Drive

In Colab, select **Runtime > Change runtime type > T4 GPU** (or any GPU).

```python
# [Cell 1: Environment & Drive Mount]
import os
import sys
import json
import subprocess
from pathlib import Path

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Verify GPU availability
try:
    gpu_name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True
    ).strip()
    print(f"Active GPU: {gpu_name}")
except Exception as e:
    print(f"GPU check: {e}")
```

---

## Step 2: Global Configuration and Directory Layout

Set all shared paths once in Python so they persist cleanly across all subsequent cells via `os.environ`.

```python
# [Cell 2: Global Configuration]
import os
from pathlib import Path

# Repository and configurable Drive dataset archive location
REPO_URL = "https://github.com/AbdelrahmanHassan131/Novel-Deepfake-Detection.git"
REPO_BRANCH = "GeneralizationDeepFakeDetection"
DRIVE_ARCHIVE_DIR = "/content/drive/MyDrive/PreparedDataset"
DRIVE_BACKUP_DIR = "/content/drive/MyDrive/Generalization_First_Try_RGB/controlled_mixture_experiments"

# Local SSD paths
LOCAL_REPO = "/content/Novel_Deepfake_Detection"
DATA_ROOT = "/content/dataset/Preapred Dataset/train"
MANIFEST_DIR = f"{LOCAL_REPO}/output/controlled_mixture/manifests_v2"
SHARED_DEV = f"{MANIFEST_DIR}/shared_dev/selected_manifest.csv"
OUTPUT_BASE = "/content/experiments"
REGISTRY_FILE = "/content/run_registry.json"

# Export into environment for all child processes and bash cells
os.environ["REPO_URL"] = REPO_URL
os.environ["REPO_BRANCH"] = REPO_BRANCH
os.environ["DRIVE_ARCHIVE_DIR"] = DRIVE_ARCHIVE_DIR
os.environ["DRIVE_BACKUP_DIR"] = DRIVE_BACKUP_DIR
os.environ["LOCAL_REPO"] = LOCAL_REPO
os.environ["DATA_ROOT"] = DATA_ROOT
os.environ["MANIFEST_DIR"] = MANIFEST_DIR
os.environ["SHARED_DEV"] = SHARED_DEV
os.environ["OUTPUT_BASE"] = OUTPUT_BASE
os.environ["REGISTRY_FILE"] = REGISTRY_FILE

print("Environment configured:")
print(f"  LOCAL_REPO   : {LOCAL_REPO}")
print(f"  MANIFEST_DIR : {MANIFEST_DIR}")
print(f"  OUTPUT_BASE  : {OUTPUT_BASE}")
```

---

## Step 3: Fresh Clone of the Pushed Branch

Clone code and the tracked `manifests_v2` files to local SSD. This does not change your local PC or Drive repository. If the branch or required files are absent remotely, push them first; the cell fails instead of using old code. If you already ran this cell successfully, skip it when switching arms—do not clone again between Arm A and Arm B.

```bash
%%bash
# [Cell 3: Fresh Git Clone and Dependencies]
set -euo pipefail

mkdir -p "$OUTPUT_BASE"

# 1. Clone to a fresh destination; never delete old work automatically.
python - <<'PY'
import os, subprocess, json, hashlib
from pathlib import Path
target = Path(os.environ['LOCAL_REPO']).resolve()
assert target.parent == Path('/content'), 'LOCAL_REPO must be a direct child of /content'
assert not target.exists() or not any(target.iterdir()), f'Choose a fresh LOCAL_REPO in Cell 2: {target}'
subprocess.run(['git', 'clone', '--depth', '1', '--single-branch', '--branch',
                os.environ['REPO_BRANCH'], os.environ['REPO_URL'], str(target)], check=True)
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=target, text=True).strip()
base = target / 'output/controlled_mixture/manifests_v2'
for arm in ('arm_a', 'arm_b', 'shared_dev'):
    manifest = base / arm / 'selected_manifest.csv'
    gate_path = manifest.with_suffix('.verified.json')
    assert manifest.is_file() and gate_path.is_file(), f'Push the manifest and gate for {arm} to the selected branch'
    gate = json.loads(gate_path.read_text(encoding='utf-8'))
    raw = manifest.read_bytes()
    expected = gate['manifest_sha256']
    valid_digests = {
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest(),
        hashlib.sha256(raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest(),
    }
    assert expected in valid_digests, f'Manifest digest failed: {arm}'
for name in ('union_members_list.txt', 'union_archive_members.txt', 'mixture_manifest_report.json'):
    assert (base / name).is_file(), f'Missing tracked preparation file: {name}'
(Path(os.environ['OUTPUT_BASE']) / 'clone_commit.txt').write_text(commit + '\n', encoding='utf-8')
print('Cloned branch:', os.environ['REPO_BRANCH'], '\nCommit:', commit, '\nCode and manifests:', target)
PY

# 2. Install required Python packages (excluding Torch/CUDA reinstall)
pip install --quiet timm scikit-learn PyWavelets ptwt scipy opencv-python-headless tqdm matplotlib seaborn pandas
cd "$LOCAL_REPO"
python train.py --help > /content/pilot_train_help.txt

# 3. Install OS 7-Zip package
sudo apt-get update -qq && sudo apt-get install -y -qq p7zip-full

# 4. Verify 7-Zip executable
if ! command -v 7z &> /dev/null; then
    echo "ERROR: 7z executable not found in PATH." >&2
    exit 1
fi

echo "Repository and system utilities ready at $LOCAL_REPO."
```

---

## Step 4: Selective Dataset Extraction & Member Verification

The total union across Arm A, Arm B, and Shared Dev is **29,163 images** (~1.0 GiB uncompressed). You do not need to extract the entire 33.8 GiB archive.

The cell checks whether all 29,163 required files already exist on local disk. If any are missing, it selectively extracts only those members from the multi-volume split-ZIP archive.

```python
# [Cell 4: Dataset Restoration and Full Path Verification]
import os
import sys
import subprocess
from pathlib import Path

local_repo = Path(os.environ["LOCAL_REPO"])
data_root = Path(os.environ["DATA_ROOT"])
archive_dir = Path(os.environ["DRIVE_ARCHIVE_DIR"])
member_list_file = local_repo / "output/controlled_mixture/manifests_v2/union_members_list.txt"
archive_members_file = local_repo / "output/controlled_mixture/manifests_v2/union_archive_members.txt"

if not member_list_file.is_file():
    raise FileNotFoundError(f"Missing member list: {member_list_file}")

with open(member_list_file, "r", encoding="utf-8") as f:
    required_paths = [Path(line.strip()) for line in f if line.strip()]

total_required = len(required_paths)
print(f"Total union cohort members required: {total_required:,}")

# Check which required images are already present on disk
missing_paths = [p for p in required_paths if not p.is_file()]

if len(missing_paths) == 0:
    print(f"PASS: All {total_required:,} required images are already present on local SSD. Skipping extraction.")
else:
    print(f"Found {len(missing_paths):,} missing images out of {total_required:,}. Initiating selective extraction...")

    # Validate split archive volumes on Drive
    zip_header = archive_dir / "Preapred Dataset.zip"
    if not zip_header.is_file():
        raise FileNotFoundError(
            f"Archive header not found: {zip_header}. "
            f"Please verify DRIVE_ARCHIVE_DIR in Cell 2."
        )

    # Check for split volumes .z01 .. .z16
    for i in range(1, 17):
        vol = archive_dir / f"Preapred Dataset.z{i:02d}"
        if not vol.is_file():
            raise FileNotFoundError(f"Missing split archive volume: {vol}")

    # Check available local disk space (require at least 8 GiB free)
    stat = os.statvfs("/content")
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
    print(f"Available disk space on /content: {free_gb:.1f} GiB")
    if free_gb < 8.0:
        raise RuntimeError(f"Insufficient disk space on /content ({free_gb:.1f} GiB free). At least 8 GiB required.")

    # Extract exact missing members only; do not overwrite already restored images.
    extract_target = "/content/dataset"
    os.makedirs(extract_target, exist_ok=True)
    from pathlib import PurePosixPath
    allowed = set(archive_members_file.read_text(encoding='utf-8').splitlines())
    missing_members = []
    for path in missing_paths:
        member = path.relative_to(extract_target).as_posix()
        parts = PurePosixPath(member).parts
        assert member in allowed and parts[:2] == ('Preapred Dataset', 'train')
        assert '..' not in parts and not PurePosixPath(member).is_absolute()
        missing_members.append(member)
    exact_list = Path('/content/pilot_missing_members.txt')
    exact_list.write_text('\n'.join(missing_members) + '\n', encoding='utf-8')
    cmd = [
        "7z", "x", str(zip_header),
        f"-o{extract_target}",
        f"-i@{exact_list}", "-spd",
        "-y"
    ]
    print(f"Extracting required members with 7-Zip...")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        print("7-Zip Output:\n", res.stdout[-2000:])
        raise RuntimeError(f"7-Zip extraction failed with exit code {res.returncode}")

    # Re-verify all paths after extraction
    still_missing = [p for p in required_paths if not p.is_file()]
    if still_missing:
        raise RuntimeError(
            f"Extraction incomplete: {len(still_missing)} images still missing! "
            f"First missing: {still_missing[0]}"
        )
    print(f"PASS: Successfully extracted and verified all {total_required:,} required images.")
```

---

## Step 5: Cryptographic Manifest Gate Preflight Check

Validate that all manifests match their audited cryptographic SHA-256 digests and inherit verified clean-parent evidence.

```python
# [Cell 5: Cryptographic Gate Check]
import sys
from pathlib import Path

local_repo = Path(os.environ["LOCAL_REPO"])
if str(local_repo) not in sys.path:
    sys.path.insert(0, str(local_repo))

from data.manifest import verify_manifest_gate, read_manifest

manifest_dir = Path(os.environ["MANIFEST_DIR"])
cohorts = ["arm_a", "arm_b", "shared_dev"]

print("=" * 65)
print("CRYPTOGRAPHIC MANIFEST GATE VERIFICATION")
print("=" * 65)

for cohort in cohorts:
    csv_file = manifest_dir / cohort / "selected_manifest.csv"
    verify_manifest_gate(str(csv_file), require_hashes=True)
    rows = read_manifest(str(csv_file))
    print(f"PASS: {cohort:12s} | {len(rows):6d} rows | Gate verified and intact.")

print("=" * 65)
print("All gates verified. Ready for execution.")
```

---

## Step 6: Bounded Throughput Benchmark (Optional Pre-Flight)

Run a 10-warmup and 50-measured microbatch throughput check to measure GPU utilization, DataLoader wait time, and compute speed.

```bash
%%bash
# [Cell 6: Bounded Throughput Check]
set -euo pipefail
cd "$LOCAL_REPO"

python tools/benchmark_training_throughput.py \
  --arch Wang2020_128 \
  --manifest "$MANIFEST_DIR/arm_a/selected_manifest.csv" \
  --val_manifest "$SHARED_DEV" \
  --dataroot "$DATA_ROOT" \
  --batch_size 32 \
  --grad_accum_steps 2 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --persistent_workers \
  --pin_memory \
  --aug_recipe rgb_v1 \
  --fine_tune_policy layer4_and_head \
  --bn_policy frozen \
  --backbone_lr_mult 0.1 \
  --use_amp \
  --amp_dtype fp16 \
  --warmup_microbatches 10 \
  --measure_microbatches 50 \
  --val_samples 100 \
  --output "/content/throughput_benchmark.json"

cat /content/throughput_benchmark.json
```

---

## Step 7: Train Arm A (or Arm B) with Unbuffered Logging

Choose `ARM="arm_a"` or `ARM="arm_b"`. Both arms use the exact same architecture, ImageNet initialization, hyperparameters, and fixed 5-epoch budget.

`ExperimentManager` creates `$OUTPUT_BASE/${ARM}_${RUN_ID}/checkpoints/`. The cell records the exact run directory into `/content/run_registry.json`.

```python
# [Cell 7: Launch Pilot Training]
import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import shutil

# SELECT ARM: "arm_a" or "arm_b"
ARM = "arm_a"
assert ARM in ("arm_a", "arm_b")

local_repo = os.environ["LOCAL_REPO"]
manifest_dir = os.environ["MANIFEST_DIR"]
data_root = os.environ["DATA_ROOT"]
output_base = os.environ["OUTPUT_BASE"]
registry_file = Path(os.environ["REGISTRY_FILE"])

run_id = f"seed42_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
run_dir_name = f"{ARM}_{run_id}"
run_dir = Path(output_base) / run_dir_name
assert not run_dir.exists(), f'Run already exists: {run_dir}'

manifest_path = f"{manifest_dir}/{ARM}/selected_manifest.csv"
# ExperimentManager must create run_dir itself. Keep the live log outside it.
log_dir = Path(output_base) / "launch_logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"{run_dir_name}.log"

print("=" * 65)
print(f"STARTING PILOT TRAINING: {ARM.upper()}")
print(f"Run ID    : {run_id}")
print(f"Run Dir   : {run_dir}")
print(f"Manifest  : {manifest_path}")
print("=" * 65)

cmd = [
    sys.executable, "-u", "train.py",
    "--arch", "Wang2020_128",
    "--name", ARM,
    "--run_id", run_id,
    "--checkpoints_dir", output_base,
    "--dataroot", data_root,
    "--manifest", manifest_path,
    "--manifest_split", "train",
    "--val_manifest", manifest_path,
    "--val_manifest_split", "dev",
    "--rgb_head_type", "128d",
    "--rgb_dropout", "0.5",
    "--fine_tune_policy", "layer4_and_head",
    "--bn_policy", "frozen",
    "--backbone_lr_mult", "0.1",
    "--aug_recipe", "rgb_v1",
    "--gpu_ids", "0",
    "--batch_size", "32",
    "--grad_accum_steps", "2",
    "--val_batch_size", "32",
    "--num_workers", "4",
    "--prefetch_factor", "2",
    "--persistent_workers",
    "--pin_memory",
    "--epochs", "5",
    "--epochs_decay", "0",
    "--optim", "adam",
    "--lr", "0.0001",
    "--beta1", "0.9",
    "--weight_decay", "0.0",
    "--lr_policy", "cosine",
    "--use_amp",
    "--amp_dtype", "fp16",
    "--val_precision", "fp32",
    "--seed", "42",
    "--pretrained",
    "--no-early_stopping",
    "--save_epoch_freq", "1",
    "--allow_aggregate_sources",
    "--allow_source_overlap",
    "--monitor_metric", "auc",
]

env = dict(os.environ, PYTHONUNBUFFERED="1")

with open(log_file, "w", encoding="utf-8") as lf:
    proc = subprocess.Popen(
        cmd,
        cwd=local_repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        lf.write(line)
    proc.wait()

if proc.returncode != 0:
    raise RuntimeError(f"Training failed with exit code {proc.returncode}. See {log_file}")
shutil.copy2(log_file, run_dir / "train.log")

# Verify expected output artifacts exist
ckpt_dir = run_dir / "checkpoints"
best_ckpt = ckpt_dir / "best.pth"
dev_preds = ckpt_dir / "best_dev_predictions.csv"

if not best_ckpt.is_file():
    raise FileNotFoundError(f"Expected best checkpoint not found: {best_ckpt}")
if not dev_preds.is_file():
    raise FileNotFoundError(f"Expected development predictions not found: {dev_preds}")

# Update persistent run registry
registry = {}
if registry_file.is_file():
    registry = json.loads(registry_file.read_text(encoding="utf-8"))

registry[ARM] = {
    "run_id": run_id,
    "run_dir": str(run_dir),
    "checkpoints_dir": str(ckpt_dir),
    "best_checkpoint": str(best_ckpt),
    "best_dev_predictions": str(dev_preds),
    "completed_at": datetime.now().isoformat(),
}
registry_file.write_text(json.dumps(registry, indent=2), encoding="utf-8")

print(f"\nSUCCESS: Training complete for {ARM}. Run recorded in {registry_file}.")
```

*(To run the second arm, change `ARM = "arm_b"` in Cell 7 and execute it again).*

---

## Step 8: Persist Completed Arm to Google Drive

Immediately persist checkpoints, configuration, logs, and development predictions to Drive after each arm completes.

```python
# [Cell 8: Sync Run to Google Drive]
import os
import shutil
import json
from pathlib import Path
import hashlib
from uuid import uuid4

registry_file = Path(os.environ["REGISTRY_FILE"])
if not registry_file.is_file():
    raise FileNotFoundError(f"No run registry found: {registry_file}")

registry = json.loads(registry_file.read_text(encoding="utf-8"))
backup_dir = Path(os.environ["DRIVE_BACKUP_DIR"])
backup_dir.mkdir(parents=True, exist_ok=True)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def tree_inventory(root):
    return {p.relative_to(root).as_posix(): digest(p)
            for p in root.rglob('*') if p.is_file() and p.name != 'BACKUP_COMPLETE.json'}

for arm_name, entry in registry.items():
    src_dir = Path(entry["run_dir"])
    dest_dir = backup_dir / src_dir.name
    print(f"Syncing {arm_name} ({src_dir.name}) to Google Drive...")
    expected = tree_inventory(src_dir)
    if dest_dir.exists():
        if tree_inventory(dest_dir) != expected or not (dest_dir / 'BACKUP_COMPLETE.json').is_file():
            raise RuntimeError(f'Existing backup differs or is incomplete; preserved for inspection: {dest_dir}')
        print(f'  Already saved and verified: {dest_dir}')
        continue
    staging = backup_dir / f'.{src_dir.name}.partial_{uuid4().hex[:8]}'
    shutil.copytree(src_dir, staging)
    if tree_inventory(staging) != expected:
        raise RuntimeError(f'Backup verification failed; partial copy retained: {staging}')
    (staging / 'BACKUP_COMPLETE.json').write_text(json.dumps(expected, indent=2), encoding='utf-8')
    staging.rename(dest_dir)

    # Verify backup completeness
    if not (dest_dir / "checkpoints/best.pth").is_file():
        raise RuntimeError(f"Backup incomplete for {arm_name}: best.pth missing on Drive.")
    if not (dest_dir / "checkpoints/best_dev_predictions.csv").is_file():
        raise RuntimeError(f"Backup incomplete for {arm_name}: best_dev_predictions.csv missing on Drive.")
    print(f"  Verified on Drive: {dest_dir}")

shutil.copy2(registry_file, backup_dir / "run_registry.json")
print("Drive sync complete.")
```

---

## Step 9: Per-Stratum Development Evaluation & Paired Comparison

Once both Arm A and Arm B have completed, run `tools/evaluate_mixture_predictions.py` using the exact paths recorded in `run_registry.json`.

```python
# [Cell 9: Comparative Evaluation]
import os
import sys
import json
import subprocess
from pathlib import Path
from uuid import uuid4

local_repo = os.environ["LOCAL_REPO"]
registry_file = Path(os.environ["REGISTRY_FILE"])
shared_dev = os.environ["SHARED_DEV"]
eval_out = Path(os.environ["DRIVE_BACKUP_DIR"]) / f"evaluation_{uuid4().hex[:10]}"

if not registry_file.is_file():
    raise FileNotFoundError(f"Run registry not found: {registry_file}")

registry = json.loads(registry_file.read_text(encoding="utf-8"))

if "arm_a" not in registry or "arm_b" not in registry:
    raise RuntimeError(
        f"Registry must contain both 'arm_a' and 'arm_b' for comparison. "
        f"Currently present: {list(registry.keys())}"
    )

pred_a = registry["arm_a"]["best_dev_predictions"]
pred_b = registry["arm_b"]["best_dev_predictions"]

print("=" * 65)
print("RUNNING PER-STRATUM DEVELOPMENT EVALUATION & PAIRED COMPARISON")
print(f"Arm A Predictions: {pred_a}")
print(f"Arm B Predictions: {pred_b}")
print(f"Shared Dev       : {shared_dev}")
print(f"Output Directory : {eval_out}")
print("=" * 65)

cmd = [
    sys.executable,
    "tools/evaluate_mixture_predictions.py",
    "--pred_a", pred_a,
    "--pred_b", pred_b,
    "--shared_dev_manifest", shared_dev,
    "--output_dir", str(eval_out),
]

res = subprocess.run(cmd, cwd=local_repo, text=True)
if res.returncode != 0:
    raise RuntimeError(f"Evaluation tool failed with exit code {res.returncode}")

report_md = eval_out / "mixture_evaluation_report.md"
if report_md.is_file():
    print("\n" + report_md.read_text(encoding="utf-8"))
```

---

## Step 10: Decision Criteria & Scientific Interpretation

1. **Inspect `min_stratum_recall`**: Compare the weakest stratum recall between Arm A and Arm B.
2. **Inspect Paired Shifts**: Review whether Arm B reduced errors on video real and numeric fake without regressing overall accuracy.
3. **Caveat**: Metrics on the shared development set measure within-distribution stratum representation, not generalizability to unseen external distributions. Retain holdouts untouched. If both arms fail on external validation, data provenance and genuine source diversity must take precedence over scaling up sample count to 100K/500K.
