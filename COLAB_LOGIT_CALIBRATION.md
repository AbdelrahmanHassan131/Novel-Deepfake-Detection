# Colab: calibrate the existing conservative checkpoint in logit space

No retraining or repository update is needed. Cells 1–2 run on CPU using saved predictions. Cell 3 needs a GPU and a genuinely new image collection. Do not run Cell 3 against the calibration images again and call it an independent test.

We are explicitly using all 60,000 previously inspected external images as external calibration data. This adapts the decision threshold to that collection. The resulting approximately 84.64% is calibration fit, not validated test accuracy or evidence of zero-shot threshold transfer. No images are added to model training. A new source-disjoint test is needed for a further cross-dataset claim.

## 1. Mount Drive and define checked logit-metric helpers (Python cell)

```python
from google.colab import drive
drive.mount('/content/drive')

import csv, hashlib, json, os, sys, subprocess
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score

RUN_ROOT = Path(
    '/content/drive/MyDrive/deepfake_checkpoints_Gen_Third_Try/'
    'rgb_conservative_160k_seed42_20261011_202438_UTC_d1ba30fd'
)
CHECKPOINT = RUN_ROOT / 'checkpoints/best.pth'
CAL_PREDICTIONS = RUN_ROOT / 'external_eval/Wang2020_128/predictions.csv'

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def load_scores(path, checkpoint_hash):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Prediction file is empty')
    if {r['checkpoint_sha256'] for r in rows} != {checkpoint_hash}:
        raise ValueError('Predictions belong to a different checkpoint')
    if {r.get('eval_precision') for r in rows} != {'fp32'}:
        raise ValueError('Expected saved FP32 evaluation predictions')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate prediction sample IDs')
    paths = {r['path'].replace('\\', '/') for r in rows}
    if len(paths) != len(rows):
        raise ValueError('Duplicate prediction paths')
    y = np.array([int(r['label']) for r in rows])
    z = np.array([float(r['logit']) for r in rows], dtype=np.float64)
    if set(y.tolist()) != {0, 1} or not np.isfinite(z).all():
        raise ValueError('Need both binary classes and finite original logits')
    return y, z, paths

def choose_threshold(y, z):
    fpr, tpr, thresholds = roc_curve(y, z, drop_intermediate=False)
    finite = np.flatnonzero(np.isfinite(thresholds))
    i = finite[np.argmax((tpr - fpr)[finite])]
    return float(thresholds[i])

def logit_metrics(y, z, threshold):
    if not np.isfinite(threshold):
        raise ValueError('Threshold must be finite')
    pred = (z >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y, z, drop_intermediate=False)
    gap = fpr - (1 - tpr)
    j = int(np.searchsorted(gap, 0))
    if j == 0:
        eer = float(fpr[0])
    else:
        w = -gap[j - 1] / (gap[j] - gap[j - 1])
        eer = float(fpr[j - 1] + w * (fpr[j] - fpr[j - 1]))
    return dict(
        samples=len(y), real=int(tn + fp), fake=int(tp + fn),
        logit_threshold=float(threshold),
        accuracy=float((tn + tp) / len(y)),
        balanced_accuracy=float((tn / (tn + fp) + tp / (tp + fn)) / 2),
        roc_auc_logit=float(roc_auc_score(y, z)),
        interpolated_eer_logit=eer,
        real_recall=float(tn / (tn + fp)), fake_recall=float(tp / (tp + fn)),
        macro_f1=float(f1_score(y, pred, average='macro', zero_division=0)),
        confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]]
    )

assert CHECKPOINT.is_file(), f'Missing checkpoint: {CHECKPOINT}'
assert CAL_PREDICTIONS.is_file(), f'Missing predictions: {CAL_PREDICTIONS}'
CHECKPOINT_HASH = file_hash(CHECKPOINT)
cal_y, cal_z, cal_paths = load_scores(CAL_PREDICTIONS, CHECKPOINT_HASH)
print(f'Checked {len(cal_y):,} saved predictions against the checkpoint. No inference run.')
```

## 2. Fit and save the threshold once (Python cell)

The decision rule is `fake if logit >= logit_threshold`. Do not convert this threshold to a float32 probability: that would reintroduce saturation. This artifact is deliberately named `logit_threshold.json` and uses a different schema from the repository's probability threshold files. **Do not pass it to the old `evaluate.py --threshold_file` option.**

```python
CAL_DIR = RUN_ROOT / 'external_logit_calibration_v1'
CAL_DIR.mkdir(exist_ok=True)
THRESHOLD_FILE = CAL_DIR / 'logit_threshold.json'

identity = dict(
    checkpoint_sha256=CHECKPOINT_HASH,
    calibration_prediction_sha256=file_hash(CAL_PREDICTIONS),
    score_space='logit',
    criterion='external calibration Youden J; first maximum on descending ROC thresholds',
    label_mapping={'real': 0, 'fake': 1},
    decision_rule='fake if logit >= logit_threshold',
    calibration_status='previously_inspected_external_data_not_independent_test',
)
if THRESHOLD_FILE.exists():
    artifact = json.loads(THRESHOLD_FILE.read_text())
    for key, value in identity.items():
        assert artifact.get(key) == value, f'Saved calibration mismatch: {key}'
    threshold = float(artifact['logit_threshold'])
    assert np.isfinite(threshold)
    print('Reusing saved threshold; not fitting again.')
else:
    threshold = choose_threshold(cal_y, cal_z)
    artifact = dict(identity, logit_threshold=threshold,
                    created_utc=datetime.now(timezone.utc).isoformat())
    with THRESHOLD_FILE.open('x', encoding='utf-8') as f:
        json.dump(artifact, f, indent=2, allow_nan=False)

fit = logit_metrics(cal_y, cal_z, threshold)
fit['status'] = 'calibration_fit_only_not_independent_test'
(CAL_DIR / 'calibration_fit.json').write_text(json.dumps(fit, indent=2, allow_nan=False))
print('CALIBRATION FIT ONLY:')
print(json.dumps(fit, indent=2))
print('Frozen threshold:', THRESHOLD_FILE)
print('Keep this file together with best.pth.')
```

Stop here if no untouched test dataset is available. Send back the printed threshold and calibration report. The point of this step is to save an exact, checkpoint-linked decision rule, not to claim a new independent result.

## 3. Only when a new untouched dataset is available: run inference once

The new folder must contain `real/` and `fake/`, with images not used for training, development, or calibration. For cross-dataset evidence, use new sources as well. Source provenance and content/group overlap need verification separately; the path checks below cannot prove independence. Do not obtain a supposedly new test simply by copying or renaming the calibration images.

Set the two paths below. This example requires a GPU runtime and a working checkout with its dependencies installed. If returning in a fresh runtime, rerun Cells 1–2 to restore and verify the saved threshold; restore the checkpoint and calibration CSV from Drive as needed.

```python
# Fill these in only when you have a genuinely new test collection.
candidates = [
    Path('/content/code/Novel-Deepfake-Detection'),
    Path('/content/Novel_Deepfake_Detection'),
    Path('/content/Novel-Deepfake-Detection'),
    Path('/content/drive/MyDrive/Generalization_First_Try_RGB/Novel-Deepfake-Detection'),
]
REPO_ROOT = next((p for p in candidates if (p / 'evaluate.py').is_file()), None)
assert REPO_ROOT is not None and (REPO_ROOT / 'evaluate.py').is_file(), 'Could not find repository with evaluate.py'
NEW_TEST_FOLDER = ''  # e.g. /content/new_test_dataset, containing real/ and fake/
assert NEW_TEST_FOLDER, 'Provide a new untouched dataset; do not reuse the calibration collection.'
NEW_TEST_ROOT = Path(NEW_TEST_FOLDER).resolve()
for label in ('real', 'fake'):
    assert (NEW_TEST_ROOT / label).is_dir(), f'Missing: {NEW_TEST_ROOT / label}'
assert NEW_TEST_ROOT != Path('/content/unseenDataset/val to be deleted 2').resolve()
assert NEW_TEST_ROOT not in {
    Path('/content/dataset/DatesetNoVideosNoGAN/train').resolve(),
    Path('/content/dataset/DatesetNoVideosNoGAN/val').resolve(),
}
assert file_hash(CHECKPOINT) == artifact['checkpoint_sha256']

TEST_OUTPUT = RUN_ROOT / ('independent_candidate_' + uuid4().hex[:10])
subprocess.run([
    sys.executable, '-u', str(REPO_ROOT / 'evaluate.py'),
    '--checkpoint', str(CHECKPOINT), '--arch', 'Wang2020_128',
    '--val_root', str(NEW_TEST_ROOT), '--split', 'external_test',
    '--output_dir', str(TEST_OUTPUT), '--device', 'cuda:0',
    '--batch_size', '32', '--num_workers', '0', '--eval_precision', 'fp32',
    '--bootstrap', '0', '--no_plots', '--no_tsne', '--no_gradcam', '--no_profiling'
], cwd=REPO_ROOT, check=True)
print('Inference completed. Run Cell 4 for the frozen-logit-threshold report.')
```

The existing evaluator still prints its old probability-based report at threshold 0.5. Those numbers are not the calibrated result. Cell 4 computes the intended result directly from saved logits and does not choose a new threshold.

## 4. Apply the saved threshold without fitting on the new labels

```python
test_csv = TEST_OUTPUT / 'Wang2020_128/predictions.csv'
assert file_hash(test_csv) != artifact['calibration_prediction_sha256'], 'Calibration file reused as test'
test_y, test_z, test_paths = load_scores(test_csv, artifact['checkpoint_sha256'])
assert not (test_paths & cal_paths), 'Calibration and test paths overlap'

# Load the frozen value from Drive. Never call choose_threshold on test labels.
frozen = json.loads(THRESHOLD_FILE.read_text())
assert frozen['checkpoint_sha256'] == CHECKPOINT_HASH
assert frozen['calibration_prediction_sha256'] == file_hash(CAL_PREDICTIONS)
assert frozen['score_space'] == 'logit'
test_report = logit_metrics(test_y, test_z, float(frozen['logit_threshold']))
test_report.update(
    checkpoint_sha256=CHECKPOINT_HASH,
    threshold_artifact_sha256=file_hash(THRESHOLD_FILE),
    test_prediction_sha256=file_hash(test_csv),
    dataset=str(NEW_TEST_ROOT),
    threshold_fitted_on_test=False,
    independence_status='requires_separate_source_and_content_overlap_verification',
)
out = TEST_OUTPUT / 'frozen_logit_test_report.json'
out.write_text(json.dumps(test_report, indent=2, allow_nan=False))
print(json.dumps(test_report, indent=2))
print('Send back:', out)
```

Do not adjust the threshold after inspecting this test result and continue calling that dataset an untouched test. If calibration fails to transfer, that is a useful outcome to report, not a reason to tune against the test answers.
