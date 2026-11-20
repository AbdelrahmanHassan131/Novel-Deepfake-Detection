# Universal Benchmark Evaluation: Evaluate All `best.pth` Checkpoints

This notebook evaluates your trained models sequentially on any new unseen dataset (e.g., DFDC, GenImage, or a new real/fake collection).

- **Execution:** Pure Python cells (no buffered bash). Tqdm progress bars and evaluation metrics are visible in real-time as each model runs.
- **GPU Recommendation:** **NVIDIA L4** (strongly recommended over T4). L4 evaluates ~3–4× faster (FP16/FP32 Tensor Cores, 24 GB VRAM), taking ~3–6 minutes per model compared to 15–25 minutes on T4.

---

## Cell 1: Mount Drive, Clone Repo, Install Dependencies

```python
from google.colab import drive
drive.mount('/content/drive')
import os, sys, json, csv, shutil, subprocess, runpy, gc
from pathlib import Path
import torch

REPO_ROOT = Path('/content/Novel-Deepfake-Detection')
if not REPO_ROOT.exists():
    subprocess.run(['git', 'clone', '--branch', 'GeneralizationDeepFakeDetection',
        'https://github.com/AbdelrahmanHassan131/Novel-Deepfake-Detection.git', str(REPO_ROOT)], check=True)
else:
    subprocess.run(['git', 'pull', '--ff-only', 'origin', 'GeneralizationDeepFakeDetection'], cwd=REPO_ROOT, check=True)

os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Install evaluation dependencies
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'timm', 'PyWavelets', 'pytorch-wavelets', 'scikit-learn', 'scipy',
    'pillow', 'opencv-python-headless', 'matplotlib', 'pyyaml', 'tqdm',
    'git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33'], check=True)

print("=" * 60)
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU DETECTED'}")
print(f"PyTorch Version: {torch.__version__}")
print(f"Repository Commit: {subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}")
print("=" * 60)

def run_python(script, args):
    """Direct Python execution inside Colab so stdout and tqdm progress bars stream live."""
    previous_argv = sys.argv[:]
    sys.argv = [str(REPO_ROOT / script), *map(str, args)]
    try:
        runpy.run_path(str(REPO_ROOT / script), run_name='__main__')
    finally:
        sys.argv = previous_argv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

---

## Cell 2: Define and Validate Your Unseen Dataset (e.g., DFDC)

Set the path to your extracted dataset. It must contain `real/` and `fake/` subdirectories.

```python
# -------------------------------------------------------------
# SPECIFY YOUR DATASET PATH HERE
# -------------------------------------------------------------
DATASET_NAME = "DFDC_Unseen_PNG"
DATASET_ROOT = Path('/content/dfdc')  # <-- Change this to your dataset directory if different

# If your dataset is in a ZIP in Drive, you can extract it here:
# ZIP_PATH = Path('/content/drive/MyDrive/dfdc_unseen.zip')
# if not DATASET_ROOT.is_dir() and ZIP_PATH.is_file():
#     print(f"Extracting {ZIP_PATH}...")
#     subprocess.run(['7z', 'x', str(ZIP_PATH), f'-o{DATASET_ROOT.parent}', '-y'], check=True)

assert (DATASET_ROOT / 'real').is_dir(), f"Missing real folder: {DATASET_ROOT / 'real'}"
assert (DATASET_ROOT / 'fake').is_dir(), f"Missing fake folder: {DATASET_ROOT / 'fake'}"

real_files = list((DATASET_ROOT / 'real').glob('*.*'))
fake_files = list((DATASET_ROOT / 'fake').glob('*.*'))
total_images = len(real_files) + len(fake_files)

print("=" * 60)
print(f"  Target Evaluation Dataset: {DATASET_NAME}")
print(f"  Location:                  {DATASET_ROOT}")
print(f"  Real Images:               {len(real_files):,} ({(len(real_files)/total_images)*100:.1f}%)")
print(f"  Fake Images:               {len(fake_files):,} ({(len(fake_files)/total_images)*100:.1f}%)")
print(f"  Total Samples:             {total_images:,}")
print("=" * 60)
```

---

## Cell 3: Register Your Trained `best.pth` Checkpoints

This cell discovers and registers your trained checkpoints across Drive.

```python
MODELS_TO_EVALUATE = []

# 1. ConvNeXt-128 (Latest Run)
convnext_runs = list(Path('/content/drive/MyDrive/deepfake_convnext_128/runs').glob('*/checkpoints/best.pth'))
if convnext_runs:
    latest_convnext = sorted(convnext_runs, key=lambda p: p.stat().st_mtime)[-1]
    MODELS_TO_EVALUATE.append({
        'name': 'ConvNeXt-128',
        'arch': 'ConvNeXt_128',
        'checkpoint': latest_convnext,
        'threshold_file': latest_convnext.parent / 'threshold_logit.json'
    })

# 2. DRCT-128 (CLIP-based Model)
drct_runs = list(Path('/content/drive/MyDrive/deepfake_drct_128/runs').glob('*/checkpoints/best.pth'))
if drct_runs:
    latest_drct = sorted(drct_runs, key=lambda p: p.stat().st_mtime)[-1]
    MODELS_TO_EVALUATE.append({
        'name': 'DRCT-128',
        'arch': 'DRCT_128',
        'checkpoint': latest_drct,
        'threshold_file': latest_drct.parent / 'threshold_logit.json'
    })

# 3. Wang2020 / ResNet-50 Baselines
wang_runs = list(Path('/content/drive/MyDrive/deepfake_checkpoints_Gen_Third_Try').glob('**/checkpoints/best.pth'))
for ckpt in wang_runs:
    parent_name = ckpt.parent.parent.name
    MODELS_TO_EVALUATE.append({
        'name': f'Wang2020 ({parent_name[:20]})',
        'arch': 'Wang2020_128',
        'checkpoint': ckpt,
        'threshold_file': ckpt.parent / 'threshold_logit.json'
    })

# You can also manually add any exact checkpoint path here:
# MODELS_TO_EVALUATE.append({
#     'name': 'Custom-Model',
#     'arch': 'ConvNeXt_128',
#     'checkpoint': Path('/content/drive/MyDrive/path/to/best.pth'),
#     'threshold_file': None
# })

print(f"Registered {len(MODELS_TO_EVALUATE)} models for sequential evaluation:")
for i, m in enumerate(MODELS_TO_EVALUATE, 1):
    print(f"  [{i}] {m['name']} (Arch: {m['arch']})")
    print(f"      Checkpoint: {m['checkpoint']}")
    print(f"      Exists: {m['checkpoint'].is_file()}")
```

---

## Cell 4: Run Sequential Evaluation with Live Progress & Metrics Cards

Each model runs individually. You will see its live progress bar, followed by its complete metrics card and an updated comparison leaderboard.

```python
EVAL_BASE = Path('/content/drive/MyDrive/benchmark_evaluations') / DATASET_NAME
EVAL_BASE.mkdir(parents=True, exist_ok=True)

leaderboard = []

for idx, model_spec in enumerate(MODELS_TO_EVALUATE, 1):
    name = model_spec['name']
    arch = model_spec['arch']
    ckpt = model_spec['checkpoint']
    thresh_file = model_spec.get('threshold_file')

    if not ckpt.is_file():
        print(f"\n[SKIP] Checkpoint not found: {ckpt}")
        continue

    print("\n" + "=" * 70)
    print(f"  [{idx}/{len(MODELS_TO_EVALUATE)}] RUNNING EVALUATION: {name}")
    print(f"  Arch: {arch} | Checkpoint: {ckpt.name}")
    print("=" * 70)

    out_dir = EVAL_BASE / f"eval_{name.replace(' ', '_').replace('/', '_')}"
    
    args = [
        '--arch', arch,
        '--checkpoint', str(ckpt),
        '--val_root', str(DATASET_ROOT),
        '--split', 'external_benchmark',
        '--device', 'cuda:0',
        '--batch_size', '16',
        '--num_workers', '0',
        '--eval_precision', 'fp32',
        '--no_plots', '--no_tsne', '--no_gradcam', '--no_profiling',
        '--output_dir', str(out_dir),
    ]

    if thresh_file and thresh_file.is_file():
        args.extend(['--threshold_file', str(thresh_file)])

    # Execute with live output
    run_python('evaluate.py', args)

    # Load results
    report_file = out_dir / arch / 'generalization_report.json'
    eval_report_file = out_dir / arch / 'evaluation_report.json'
    
    rep = {}
    if report_file.is_file():
        rep = json.loads(report_file.read_text()).get('overall', {})
    elif eval_report_file.is_file():
        raw_rep = json.loads(eval_report_file.read_text())
        rep = raw_rep.get('classification_metrics', {})

    bal_acc = rep.get('balanced_accuracy', rep.get('accuracy', 0.0)) * 100
    auc = rep.get('roc_auc', rep.get('auc', 0.0))
    ap = rep.get('average_precision', rep.get('pr_auc', rep.get('ap', 0.0)))
    eer = rep.get('eer', 0.0) * 100
    real_rec = rep.get('real_recall', rep.get('specificity', 0.0)) * 100
    fake_rec = rep.get('fake_recall', rep.get('sensitivity', 0.0)) * 100

    # Display individual model metrics card
    print("\n" + "-" * 55)
    print(f"       METRICS CARD: {name}")
    print("-" * 55)
    print(f"  Balanced Accuracy:           {bal_acc:.2f}%")
    print(f"  ROC AUC:                     {auc:.4f}")
    print(f"  Average Precision (PR AUC):  {ap:.4f}")
    print(f"  Equal Error Rate (EER):      {eer:.2f}%")
    print(f"  Real Recall (Specificity):   {real_rec:.2f}%")
    print(f"  Fake Recall (Sensitivity):   {fake_rec:.2f}%")
    print("-" * 55)

    leaderboard.append({
        'Model': name,
        'Arch': arch,
        'ROC AUC': f"{auc:.4f}",
        'PR AUC': f"{ap:.4f}",
        'EER': f"{eer:.2f}%",
        'Balanced Acc': f"{bal_acc:.2f}%",
        'Real Recall': f"{real_rec:.2f}%",
        'Fake Recall': f"{fake_rec:.2f}%"
    })

# -------------------------------------------------------------
# FINAL BENCHMARK SUMMARY LEADERBOARD
# -------------------------------------------------------------
print("\n\n" + "=" * 80)
print(f"        FINAL BENCHMARK LEADERBOARD ON {DATASET_NAME}")
print("=" * 80)

header = f"{'Model':<25} | {'ROC AUC':<9} | {'PR AUC':<8} | {'EER':<8} | {'Bal Acc':<9} | {'Real Rec':<9} | {'Fake Rec':<9}"
print(header)
print("-" * len(header))
for r in leaderboard:
    print(f"{r['Model']:<25} | {r['ROC AUC']:<9} | {r['PR AUC']:<8} | {r['EER']:<8} | {r['Balanced Acc']:<9} | {r['Real Recall']:<9} | {r['Fake Recall']:<9}")
print("=" * 80)

# Save leaderboard to Drive
import pandas as pd
df = pd.DataFrame(leaderboard)
summary_csv = EVAL_BASE / 'benchmark_summary.csv'
df.to_csv(summary_csv, index=False)
print(f"\nLeaderboard CSV saved to: {summary_csv}")
```
