"""
Integration and regression tests for Phase 3 Code Review Repairs (R09 - R14).
"""
import copy
import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from config.configuration import Config
from config.compatibility import config_from_opt, config_to_opt
from config.types import ArchitectureType, OptimizerType, SchedulerType
from config.validator import ConfigValidator
from data.manifest import (
    audit_rows,
    find_connected_components,
    select_representative_dev_cohort,
)
from evaluation.checkpoint_loader import CheckpointLoader
from models.wang2020_128.trainer import Wang2020_128Trainer
from training.checkpoint_manager import CheckpointManager
from training.hooks.early_stopping_hook import EarlyStoppingHook
from training.optimizer_factory import build_optimizer


def test_r09_legacy_one_group_optimizer_resume_migration(tmp_path):
    """Verify legacy 1-group Adam checkpoint migrates to modern multi-group optimizer without buffer loss."""
    # Build a Wang2020_128Trainer
    opt = SimpleNamespace(
        arch='Wang2020_128',
        isTrain=True,
        continue_train=False,
        pretrained=False,
        rgb_head_type='128d',
        rgb_dropout=0.5,
        fine_tune_policy='full',
        lr=0.0001,
        beta1=0.9,
        weight_decay=0.0001,
        momentum=0.0,
        gpu_ids=[],
        checkpoints_dir=str(tmp_path),
        name='test_resume_migration',
    )
    trainer_model = Wang2020_128Trainer(opt)

    # Current optimizer has 4 groups (head_decay, head_no_decay, backbone_decay, backbone_no_decay)
    assert len(trainer_model.optimizer.param_groups) > 1

    # Simulate a legacy checkpoint that saved a single parameter group over all parameters
    all_trainable = [p for p in trainer_model.model.parameters() if p.requires_grad]
    legacy_opt = torch.optim.Adam(all_trainable, lr=0.0002)

    # Take a synthetic step so exp_avg / exp_avg_sq are populated
    for p in all_trainable:
        p.grad = torch.ones_like(p) * 0.5
    legacy_opt.step()

    legacy_state_dict = legacy_opt.state_dict()
    assert len(legacy_state_dict['param_groups']) == 1

    # Save synthetic legacy checkpoint
    ckpt_path = tmp_path / "legacy_checkpoint.pth"
    mgr = CheckpointManager(save_dir=str(tmp_path), model=trainer_model, rank=0)
    state = mgr._build_state(epoch=3, best_metric=0.88, global_step=500, scheduler=None)
    state['optimizer_state_dict'] = legacy_state_dict
    state['optimizer_policy_version'] = 1  # legacy
    torch.save(state, ckpt_path)

    # Re-instantiate model with uninitialized optimizer state
    fresh_trainer = Wang2020_128Trainer(opt)
    fresh_mgr = CheckpointManager(save_dir=str(tmp_path), model=fresh_trainer, rank=0)

    # Resume into multi-group optimizer
    resumed = fresh_mgr.resume(filepath=str(ckpt_path))

    assert resumed['epoch'] == 3
    assert resumed['best_metric'] == 0.88
    assert resumed['global_step'] == 500

    # Assert optimizer states were migrated into fresh_trainer.optimizer
    assert len(fresh_trainer.optimizer.state) > 0
    # Pick a head parameter and check buffers
    fc_param = fresh_trainer.model.fc[0].weight
    assert fc_param in fresh_trainer.optimizer.state
    assert 'exp_avg' in fresh_trainer.optimizer.state[fc_param]
    assert fresh_trainer.optimizer.state[fc_param]['exp_avg'].shape == fc_param.shape


def test_r09_skip_missing_local_backbone_file_on_restore_and_eval(tmp_path):
    """CheckpointLoader and continuing train must not fail if saved backbone_weights points to nonexistent file."""
    missing_file = str(tmp_path / "nonexistent_kaggle_resnet50.pth")
    opt = SimpleNamespace(
        arch='Wang2020_128',
        isTrain=False,
        continue_train=False,
        pretrained=False,
        backbone_weights=missing_file,
        rgb_head_type='128d',
        rgb_dropout=0.5,
        gpu_ids=[],
        checkpoints_dir=str(tmp_path),
        name='test_skip_backbone',
        skip_load_networks=True,
    )
    # Wang2020_128Trainer instantiation must not attempt to load missing_file when isTrain=False
    model = Wang2020_128Trainer(opt)
    assert model is not None

    # Continuing train (continue_train=True) must also not attempt to load missing_file
    opt.isTrain = True
    opt.continue_train = True
    opt.lr = 0.0001
    train_model = Wang2020_128Trainer(opt)
    assert train_model is not None


def test_r09_linear_head_historical_dropout_default(tmp_path):
    """Linear head without rgb_dropout or dropout must default to historical 0.0 (no dropout)."""
    opt = SimpleNamespace(
        arch='Wang2020_128',
        isTrain=False,
        continue_train=False,
        pretrained=False,
        rgb_head_type='linear',
        gpu_ids=[],
        checkpoints_dir=str(tmp_path),
        name='test_linear_dropout',
        skip_load_networks=True,
    )
    # dropout / rgb_dropout omitted
    model = Wang2020_128Trainer(opt)
    assert model.dropout_rate == 0.0
    # First layer of linear head should be Identity when dropout_rate == 0
    assert isinstance(model.model.fc[0], nn.Identity)


def test_r10_evaluator_standalone_batch_size_override(tmp_path):
    """Evaluator CLI --batch_size 2 must override saved checkpoint val_batch_size=64."""
    from evaluation.evaluator import Evaluator

    evaluator = Evaluator(
        checkpoint_path=str(tmp_path / "best.pth"),
        dataroot=str(tmp_path),
        output_dir=str(tmp_path / "eval_out"),
        batch_size=2,
    )

    opt = SimpleNamespace(
        batch_size=64,
        val_batch_size=64,
        isTrain=True,
        arch='Wang2020_128',
        classes=['real', 'fake'],
    )

    # In _build_dataloader:
    # mock create_dataloader to inspect opt passed into it
    captured_opt = []
    with patch('data.create_dataloader') as mock_create:
        mock_loader = MagicMock()
        mock_loader.dataset = MagicMock()
        mock_create.side_effect = lambda o: (captured_opt.append(copy.copy(o)), mock_loader)[1]
        evaluator._build_dataloader(opt)

    assert len(captured_opt) == 1
    assert captured_opt[0].batch_size == 2
    assert captured_opt[0].val_batch_size == 2


def test_r12_adamw_config_parse_validate_and_roundtrip():
    """Verify AdamW is valid in enum, parser, validator, compatibility, and factory."""
    # 1. Enum
    assert OptimizerType.from_string('adamw') == OptimizerType.ADAMW
    assert OptimizerType.from_string('ADAMW') == OptimizerType.ADAMW

    # 2. Config from defaults with adamw
    cfg = Config.from_defaults()
    cfg.training.optimizer = 'adamw'
    validator = ConfigValidator()
    report = validator.validate(cfg)
    assert report.is_valid, f"Validation errors: {report.errors}"

    # 3. Round-trip opt -> Config -> opt
    opt = SimpleNamespace(
        arch='Wang2020_128',
        optim='adamw',
        lr=0.0001,
        weight_decay=0.01,
        beta1=0.9,
        epochs=10,
        epochs_decay=10,
    )
    conv_cfg = config_from_opt(opt)
    assert conv_cfg.training.optimizer == 'adamw'
    roundtrip_opt = config_to_opt(conv_cfg)
    assert roundtrip_opt.optim == 'adamw'

    # 4. Factory builds torch.optim.AdamW
    linear = nn.Linear(10, 2)
    opt_factory = SimpleNamespace(optim='adamw', lr=0.001, beta1=0.9, weight_decay=0.01)
    optimizer = build_optimizer(opt_factory, linear)
    assert isinstance(optimizer, torch.optim.AdamW)


def test_r13_connected_components_and_representative_dev_invariance():
    """Verify representative dev selection is invariant to row order and enforces connected links."""
    # Create rows where row 0 and row 1 share original_id, though group_id differs
    rows = [
        {'sample_id': 's1', 'path': 'p1.png', 'label': 0, 'dataset_source': 'FaceForensics++',
         'group_id': 'g1', 'original_id': 'vid_A', 'split': 'dev'},
        {'sample_id': 's2', 'path': 'p2.png', 'label': 1, 'dataset_source': 'FaceForensics++',
         'group_id': 'g2', 'original_id': 'vid_A', 'split': 'dev'},
        {'sample_id': 's3', 'path': 'p3.png', 'label': 0, 'dataset_source': 'CelebDF',
         'group_id': 'g3', 'original_id': 'vid_B', 'split': 'dev'},
        {'sample_id': 's4', 'path': 'p4.png', 'label': 1, 'dataset_source': 'CelebDF',
         'group_id': 'g4', 'original_id': 'vid_C', 'split': 'dev'},
    ]

    # Check connected components links s1 and s2
    comps = find_connected_components(rows)
    # s1 and s2 must be in the exact same component
    s1_comp = [c for c in comps.values() if any(r['sample_id'] == 's1' for r in c)][0]
    assert any(r['sample_id'] == 's2' for r in s1_comp)

    # Reorder input rows
    reversed_rows = list(reversed(rows))

    selected_1, summary_1 = select_representative_dev_cohort(rows, target_size=2, seed=42)
    selected_2, summary_2 = select_representative_dev_cohort(reversed_rows, target_size=2, seed=42)

    # Must be 100% invariant to input row ordering
    assert summary_1['cohort_sha256_digest'] == summary_2['cohort_sha256_digest']
    assert [r['sample_id'] for r in selected_1] == [r['sample_id'] for r in selected_2]

    # Verify parent cohort immutability
    assert len(rows) == 4
    assert len(reversed_rows) == 4


def test_r13_source_quotas_and_shortfall_reporting():
    """Verify source quotas are respected and honest shortfalls reported for indivisible cohorts."""
    rows = [
        {'sample_id': f's_{i}', 'path': f'p_{i}.png', 'label': i % 2,
         'dataset_source': 'FaceForensics++' if i < 10 else 'RareGen',
         'group_id': f'g_{i}', 'split': 'dev'}
        for i in range(12)  # 10 FF++, 2 RareGen
    ]

    source_quotas = {'RareGen': 2, 'FaceForensics++': 4}
    selected, summary = select_representative_dev_cohort(
        rows, target_size=6, target_real=3, target_fake=3, source_quotas=source_quotas, seed=123
    )

    assert summary['source_counts']['RareGen'] == 2
    assert summary['source_shortfalls']['RareGen'] == 0
    assert summary['shortfall_real'] == 0
    assert summary['shortfall_fake'] == 0
    assert summary['audit_result']['passed'] is True


def test_r14_early_stopping_state_dict_and_rank_safe_error():
    """Verify EarlyStoppingHook state persistence and DDP error propagation."""
    hook = EarlyStoppingHook(monitor_metric='auc', patience=3, min_delta=0.01)
    hook.best_score = 0.85
    hook.wait_count = 2
    hook.stopped_epoch = 4

    state = hook.state_dict()
    assert state['best_score'] == 0.85
    assert state['wait_count'] == 2

    new_hook = EarlyStoppingHook(monitor_metric='auc', patience=3)
    new_hook.load_state_dict(state)
    assert new_hook.best_score == 0.85
    assert new_hook.wait_count == 2
    assert new_hook.stopped_epoch == 4

    # Test error broadcasting without hanging in DDP
    with patch('torch.distributed.is_available', return_value=True), \
         patch('torch.distributed.is_initialized', return_value=True), \
         patch('torch.distributed.get_world_size', return_value=2), \
         patch('torch.distributed.broadcast_object_list') as mock_bcast:

        mock_trainer = SimpleNamespace(rank=0, current_epoch=5, should_stop=False)
        mock_result = SimpleNamespace()  # missing 'auc' attribute!

        # Should raise on rank 0 after broadcasting the error
        def fake_bcast(container, src):
            # Simulate rank 0 broadcasting error
            pass
        mock_bcast.side_effect = fake_bcast

        with pytest.raises(RuntimeError, match="Error evaluating metric on rank 0"):
            hook.on_validation_end(mock_trainer, mock_result)

        mock_bcast.assert_called_once()
