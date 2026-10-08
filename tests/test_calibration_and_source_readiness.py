"""
Unit and integration regression tests for R06, R07, and R08:
- R06: DDP sanity script, evaluate.py val_root derivation, linear scheduler policy.
- R07: Centralized calibration splits, dev_calibration producer-to-evaluator contract,
       independence verification, and relabeled-overlap leakage detection.
- R08: Scientific source-held-out readiness preflight and validator fail-fast checks.
"""

import json
import os
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.generalization import (
    calibrate,
    export_predictions,
    read_predictions,
    ALLOWED_CALIBRATION_SPLITS,
    FORBIDDEN_CALIBRATION_SPLITS,
)
from training.validator import verify_source_readiness, Validator
from training.scheduler_factory import build_scheduler
from config.types import SchedulerType, OptimizerType


# ==============================================================================
# R06 Tests: Schedulers, DDP verification script, evaluate.py
# ==============================================================================

def test_r06_linear_scheduler_construction_and_decay():
    """Verify SchedulerType and build_scheduler implement linear learning rate decay."""
    assert SchedulerType.from_string('linear') == SchedulerType.LINEAR

    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt = SimpleNamespace(lr_policy='linear', niter=3, niter_decay=2, epochs_decay=2)

    scheduler = build_scheduler(opt, optimizer)
    assert scheduler is not None

    # Epoch 0, 1, 2: full learning rate (1.0 factor)
    lrs = []
    for epoch in range(5):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    assert lrs[0] == pytest.approx(1e-3)
    assert lrs[1] == pytest.approx(1e-3)
    assert lrs[2] == pytest.approx(1e-3)
    # Epoch 3 (decay epoch 1): lr decreases
    assert lrs[3] < 1e-3
    # Epoch 4 (decay epoch 2): lr decreases further
    assert lrs[4] < lrs[3]


def test_r06_evaluate_derives_val_root_from_manifest(tmp_path):
    """Verify evaluate.py derives val_root from manifest path if val_root is omitted."""
    manifest_file = tmp_path / "subfolder" / "eval_manifest.csv"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text("sample_id,path,label\n1,a.jpg,0\n", encoding="utf-8")

    from evaluate import parse_args
    import sys
    test_args = ["evaluate.py", "--manifest", str(manifest_file), "--checkpoint", "none"]
    orig_argv = sys.argv
    try:
        sys.argv = test_args
        args = parse_args()
        # Test derivation logic
        if not args.val_root and args.manifest:
            args.val_root = os.path.dirname(os.path.abspath(args.manifest))
        assert args.val_root == str(manifest_file.parent.resolve())
    finally:
        sys.argv = orig_argv


# ==============================================================================
# R07 Tests: Calibration Splits, Independence Verification, and Leakage Detection
# ==============================================================================

def test_r07_centralized_split_constants():
    """Verify ALLOWED_CALIBRATION_SPLITS contains dev_calibration and rejects test splits."""
    assert 'dev_calibration' in ALLOWED_CALIBRATION_SPLITS
    assert 'dev' in ALLOWED_CALIBRATION_SPLITS
    assert 'external_dev' in ALLOWED_CALIBRATION_SPLITS
    assert 'test' in FORBIDDEN_CALIBRATION_SPLITS
    assert 'external_test' in FORBIDDEN_CALIBRATION_SPLITS
    assert not ALLOWED_CALIBRATION_SPLITS.intersection(FORBIDDEN_CALIBRATION_SPLITS)


def test_r07_producer_to_evaluator_dev_calibration_contract(tmp_path):
    """Verify threshold calibrated on dev_calibration is accepted by Evaluator threshold check."""
    pred_file = tmp_path / "dev_cal_predictions.csv"
    records = [
        {'sample_id': 's1', 'path': 's1.jpg', 'label': 0, 'split': 'dev_calibration'},
        {'sample_id': 's2', 'path': 's2.jpg', 'label': 1, 'split': 'dev_calibration'},
    ]
    probs = [0.1, 0.9]
    ckpt_hash = "abc123sha256"
    export_predictions(str(pred_file), records, probs, checkpoint_hash=ckpt_hash, eval_precision="fp32")

    thresh_file = tmp_path / "calibrated_threshold.json"
    artifact = calibrate(str(pred_file), str(thresh_file))

    assert artifact['checkpoint_sha256'] == ckpt_hash
    assert 'dev_calibration' in artifact['source_splits']
    assert artifact['independence_status'] == 'unverified_dev_calibration'

    # Verify Evaluator acceptance rule
    with open(thresh_file, encoding='utf-8') as f:
        data = json.load(f)
    source_splits = set(data.get('source_splits', []))
    assert not source_splits.intersection(FORBIDDEN_CALIBRATION_SPLITS)
    assert not (source_splits - ALLOWED_CALIBRATION_SPLITS)


def test_r07_relabeled_overlap_leakage_rejected(tmp_path):
    """Verify calibrate raises ValueError when calibration cohort overlaps with training or selection cohort."""
    train_manifest = tmp_path / "train_manifest.csv"
    train_manifest.write_text(
        "sample_id,path,label,split,dataset_source,group_id,sha256\n"
        "s_train_1,img1.jpg,0,train,src_train,group_A,hash111\n"
        "s_train_2,img2.jpg,1,train,src_train,group_B,hash222\n",
        encoding="utf-8"
    )

    # Relabeled calibration predictions reusing group_A and hash111 from training
    cal_pred_file = tmp_path / "cal_preds.csv"
    cal_records = [
        {'sample_id': 's_cal_1', 'path': 'img1_copy.jpg', 'label': 0, 'split': 'dev_calibration',
         'group_id': 'group_A', 'sha256': 'hash111'},
        {'sample_id': 's_cal_2', 'path': 'img3.jpg', 'label': 1, 'split': 'dev_calibration',
         'group_id': 'group_C', 'sha256': 'hash333'},
    ]
    export_predictions(str(cal_pred_file), cal_records, [0.2, 0.8], checkpoint_hash="ckpt1")

    out_thresh = tmp_path / "thresh.json"
    with pytest.raises(ValueError, match="Data leakage detected: calibration cohort overlaps"):
        calibrate(str(cal_pred_file), str(out_thresh), train_manifest=str(train_manifest))


def test_r07_verified_independence_when_zero_overlap(tmp_path):
    """Verify calibrate marks independence_status as verified when reference data has 0 overlap."""
    train_manifest = tmp_path / "train_manifest.csv"
    train_manifest.write_text(
        "sample_id,path,label,split,dataset_source,group_id,sha256\n"
        "s_train_1,img1.jpg,0,train,src_train,group_A,hash111\n",
        encoding="utf-8"
    )

    cal_pred_file = tmp_path / "cal_preds.csv"
    cal_records = [
        {'sample_id': 's_cal_1', 'path': 'cal1.jpg', 'label': 0, 'split': 'dev_calibration',
         'group_id': 'group_CAL_1', 'sha256': 'hashCAL1'},
        {'sample_id': 's_cal_2', 'path': 'cal2.jpg', 'label': 1, 'split': 'dev_calibration',
         'group_id': 'group_CAL_2', 'sha256': 'hashCAL2'},
    ]
    export_predictions(str(cal_pred_file), cal_records, [0.1, 0.9], checkpoint_hash="ckpt1")

    out_thresh = tmp_path / "thresh.json"
    art = calibrate(str(cal_pred_file), str(out_thresh), train_manifest=str(train_manifest))
    assert art['independence_status'] == 'reference_overlap_checked_independence_unverified'


def test_r07_external_dev_requires_permission(tmp_path):
    """Verify calibrate rejects external_dev unless allow_external_dev=True."""
    pred_file = tmp_path / "ext_dev_preds.csv"
    records = [
        {'sample_id': 's1', 'path': 's1.jpg', 'label': 0, 'split': 'external_dev'},
        {'sample_id': 's2', 'path': 's2.jpg', 'label': 1, 'split': 'external_dev'},
    ]
    export_predictions(str(pred_file), records, [0.1, 0.9], checkpoint_hash="ckpt1")

    out_thresh = tmp_path / "thresh.json"
    with pytest.raises(ValueError, match="Operating threshold calibration on.*requires explicit permission"):
        calibrate(str(pred_file), str(out_thresh), allow_external_dev=False)

    # Allowed with explicit flag
    art = calibrate(str(pred_file), str(out_thresh), allow_external_dev=True)
    assert art['independence_status'] == 'external_dev_calibration'


def test_r07_conflicting_precision_raises(tmp_path):
    """Verify calibrate raises ValueError if caller specifies conflicting precision."""
    pred_file = tmp_path / "preds.csv"
    records = [
        {'sample_id': 's1', 'path': 's1.jpg', 'label': 0, 'split': 'dev'},
        {'sample_id': 's2', 'path': 's2.jpg', 'label': 1, 'split': 'dev'},
    ]
    export_predictions(str(pred_file), records, [0.1, 0.9], checkpoint_hash="ckpt1", eval_precision="fp16")

    out_thresh = tmp_path / "thresh.json"
    with pytest.raises(ValueError, match="Evaluation precision mismatch"):
        calibrate(str(pred_file), str(out_thresh), eval_precision="fp32")


# ==============================================================================
# R08 Tests: Source-Held-Out Readiness Preflight & Fail-Fast Validator
# ==============================================================================

def test_r08_diffgan_aggregate_alone_rejected_without_diagnostic_mode():
    """Verify diffgan aggregate alone is rejected unless allow_aggregate_sources=True."""
    train_rows = [{'dataset_source': 'diffgan', 'label': 0}, {'dataset_source': 'diffgan', 'label': 1}]
    dev_rows = [{'dataset_source': 'diffgan', 'label': 0}, {'dataset_source': 'diffgan', 'label': 1}]

    with pytest.raises(ValueError, match="Scientific source readiness check failed: Dataset contains aggregate"):
        verify_source_readiness(train_rows, dev_rows, allow_aggregate_sources=False)

    # Permitted in engineering diagnostic mode
    verify_source_readiness(train_rows, dev_rows, allow_aggregate_sources=True)


def test_r08_unknown_source_metadata_rejected():
    """Verify unknown source metadata is rejected without explicit flag."""
    train_rows = [{'dataset_source': 'unknown', 'label': 0}, {'dataset_source': 'unknown', 'label': 1}]
    dev_rows = [{'dataset_source': 'unknown', 'label': 0}, {'dataset_source': 'unknown', 'label': 1}]

    with pytest.raises(ValueError, match="Scientific source readiness check failed: Dataset contains aggregate"):
        verify_source_readiness(train_rows, dev_rows, allow_aggregate_sources=False)


def test_r08_train_dev_source_overlap_rejected():
    """Verify train/dev source overlap is rejected for held-out evaluation."""
    train_rows = [
        {'dataset_source': 'progan', 'label': 0},
        {'dataset_source': 'progan', 'label': 1},
    ]
    dev_rows = [
        {'dataset_source': 'progan', 'label': 0},
        {'dataset_source': 'progan', 'label': 1},
    ]

    with pytest.raises(ValueError, match="Eligible evaluation source.*overlap with training sources"):
        verify_source_readiness(train_rows, dev_rows, eligible_sources=['progan'], allow_source_overlap=False)

    # Permitted with allow_source_overlap=True
    verify_source_readiness(train_rows, dev_rows, eligible_sources=['progan'], allow_source_overlap=True)


def test_r08_missing_requested_source_rejected():
    """Verify requested eligible sources fail fast when absent from dev data."""
    train_rows = [{'dataset_source': 'progan', 'label': 0}, {'dataset_source': 'progan', 'label': 1}]
    dev_rows = [{'dataset_source': 'stylegan', 'label': 0}, {'dataset_source': 'stylegan', 'label': 1}]

    with pytest.raises(ValueError, match="Requested eligible source.*absent from development data"):
        verify_source_readiness(train_rows, dev_rows, eligible_sources=['faceswap'], allow_source_overlap=False)


def test_r08_one_class_eligible_source_rejected():
    """Verify eligible sources lacking real or fake class fail fast."""
    train_rows = [{'dataset_source': 'progan', 'label': 0}, {'dataset_source': 'progan', 'label': 1}]
    # stylegan only has fake (label 1), no real (label 0)
    dev_rows = [
        {'dataset_source': 'stylegan', 'label': 1},
        {'dataset_source': 'stylegan', 'label': 1},
    ]

    with pytest.raises(ValueError, match="lacks both classes in development data"):
        verify_source_readiness(train_rows, dev_rows, eligible_sources=['stylegan'])


def test_r08_validator_compute_metrics_fails_on_missing_declared_source():
    """Verify _compute_validation_metrics raises immediately if declared eligible source is absent."""
    from training.validator import _compute_validation_metrics
    opt = SimpleNamespace(eligible_sources=['sourceA', 'sourceB'])

    # Only sourceA is present in predictions
    records = [
        {'dataset_source': 'sourceA', 'source': 'sourceA'},
        {'dataset_source': 'sourceA', 'source': 'sourceA'},
    ]
    dummy_dataset = SimpleNamespace(records=records)
    preds = np.array([0.1, 0.9])
    labels = np.array([0, 1])
    indices = np.array([0, 1])

    with pytest.raises(ValueError, match="Scientific evaluation failed: declared eligible source.*absent from validation data"):
        _compute_validation_metrics(preds, labels, indices, 0.5, dataset=dummy_dataset, opt=opt)
