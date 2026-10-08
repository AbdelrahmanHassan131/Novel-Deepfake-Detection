# STATUS: NOT RUN
"""
Unit tests for Task 8: Source-aware model selection and correct training lifecycle.

Verifies:
1. Source-level metrics calculation (counts, AUC, balanced accuracy, single-class defined recall, quantiles).
2. Single-class sources report defined class recall without inventing AUC or balanced accuracy.
3. Worst-source recall at fixed 0.5 threshold identification.
4. Source-macro AUC calculation and declared eligible sources filtering.
5. Rejection of source_macro_auc monitor when source metadata is missing.
6. Development calibration split role, independence labeling, and rejection of external dev/test.
7. Plateau LR scheduler stepping ONLY when fresh validation occurred in that epoch.
8. Early stopping patience, min_epochs, and rank stop flag propagation.
9. Total epochs / decay semantics (niter + niter_decay).
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from training.validator import Validator, ValidationResult, _compute_validation_metrics
from training.hooks.checkpoint_hook import CheckpointHook
from training.hooks.scheduler_hook import SchedulerHook
from training.hooks.early_stopping_hook import EarlyStoppingHook
from evaluation.generalization import calibrate, export_predictions
from config import load_config, config_to_opt


class DummyPlateauScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """Mock ReduceLROnPlateau scheduler for testing step semantics."""
    def __init__(self, mode='max'):
        self.mode = mode
        self.step_calls = []

    def step(self, metric):
        self.step_calls.append(metric)


class DummyEpochScheduler:
    """Mock standard epoch scheduler."""
    def __init__(self):
        self.step_count = 0

    def step(self):
        self.step_count += 1


class MockDataset:
    """Dataset providing records with dataset_source annotations."""
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)


class MockCheckpointManager:
    """Mock CheckpointManager recording save calls."""
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.rank = 0
        self.saved_best = []
        self.saved_last = []
        self.saved_epoch = []

    def save_best(self, epoch, best_metric, global_step, scheduler, amp_state):
        step_val = getattr(scheduler, 'step_count', 0) if scheduler else 0
        self.saved_best.append({
            'epoch': epoch,
            'best_metric': best_metric,
            'global_step': global_step,
            'scheduler': scheduler,
        })
        ckpt_path = os.path.join(self.save_dir, 'best.pth')
        Path(ckpt_path).write_text(f'best_weights_content_step_{step_val}', encoding='utf-8')

    def save_last(self, epoch, best_metric, global_step, scheduler, amp_state):
        self.saved_last.append({'epoch': epoch, 'best_metric': best_metric})
        ckpt_path = os.path.join(self.save_dir, 'last.pth')
        Path(ckpt_path).write_text('last_weights_content', encoding='utf-8')

    def save_epoch(self, epoch, best_metric, global_step, scheduler, amp_state):
        self.saved_epoch.append({'epoch': epoch, 'best_metric': best_metric})
        self.saved_last.append({'epoch': epoch, 'best_metric': best_metric})
        ckpt_path = os.path.join(self.save_dir, 'last.pth')
        Path(ckpt_path).write_text('last_weights_content', encoding='utf-8')


class TestSourceAwareLifecycle(unittest.TestCase):

    def test_source_metrics_computation(self):
        """Verify per-source counts, AUC, BA, defined recall, and quantiles."""
        # 10 samples across 3 sources:
        # Source A: 2 real, 2 fake -> 2 classes
        # Source B: 3 fake -> 1 class (fake only)
        # Source C: 3 real -> 1 class (real only)
        records = [
            {'sample_id': '0', 'label': 0, 'dataset_source': 'srcA', 'split': 'dev'},
            {'sample_id': '1', 'label': 0, 'dataset_source': 'srcA', 'split': 'dev'},
            {'sample_id': '2', 'label': 1, 'dataset_source': 'srcA', 'split': 'dev'},
            {'sample_id': '3', 'label': 1, 'dataset_source': 'srcA', 'split': 'dev'},
            {'sample_id': '4', 'label': 1, 'dataset_source': 'srcB', 'split': 'dev'},
            {'sample_id': '5', 'label': 1, 'dataset_source': 'srcB', 'split': 'dev'},
            {'sample_id': '6', 'label': 1, 'dataset_source': 'srcB', 'split': 'dev'},
            {'sample_id': '7', 'label': 0, 'dataset_source': 'srcC', 'split': 'dev'},
            {'sample_id': '8', 'label': 0, 'dataset_source': 'srcC', 'split': 'dev'},
            {'sample_id': '9', 'label': 0, 'dataset_source': 'srcC', 'split': 'dev'},
        ]
        dataset = MockDataset(records)
        all_labels = np.array([r['label'] for r in records], dtype=np.float32)
        all_preds = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.85, 0.6, 0.15, 0.4, 0.25], dtype=np.float32)
        all_indices = np.arange(len(records), dtype=np.int64)

        metrics = _compute_validation_metrics(all_preds, all_labels, all_indices, avg_loss=0.35, dataset=dataset)

        self.assertIsNotNone(metrics['source_metrics'])
        sm = metrics['source_metrics']

        # Check Source A (2 classes)
        self.assertEqual(sm['srcA']['num_samples'], 4)
        self.assertEqual(sm['srcA']['real_count'], 2)
        self.assertEqual(sm['srcA']['fake_count'], 2)
        self.assertTrue(sm['srcA']['has_both_classes'])
        self.assertEqual(sm['srcA']['auc'], 1.0)
        self.assertEqual(sm['srcA']['balanced_accuracy'], 1.0)
        self.assertIn('p50', sm['srcA']['score_quantiles'])

        # Check Source B (Fake only) -> AUC & BA must be None (never 0 or invented)
        self.assertEqual(sm['srcB']['num_samples'], 3)
        self.assertEqual(sm['srcB']['fake_count'], 3)
        self.assertEqual(sm['srcB']['real_count'], 0)
        self.assertFalse(sm['srcB']['has_both_classes'])
        self.assertIsNone(sm['srcB']['auc'])
        self.assertIsNone(sm['srcB']['balanced_accuracy'])
        self.assertEqual(sm['srcB']['fake_recall'], 1.0)
        self.assertEqual(sm['srcB']['defined_class_recall'], 1.0)

        # Check Source C (Real only) -> AUC & BA must be None
        self.assertEqual(sm['srcC']['num_samples'], 3)
        self.assertEqual(sm['srcC']['real_count'], 3)
        self.assertEqual(sm['srcC']['fake_count'], 0)
        self.assertFalse(sm['srcC']['has_both_classes'])
        self.assertIsNone(sm['srcC']['auc'])
        self.assertIsNone(sm['srcC']['balanced_accuracy'])
        self.assertEqual(sm['srcC']['real_recall'], 1.0)

        # Worst source recall at 0.5
        self.assertIsNotNone(metrics['worst_source_recall_05'])

    def test_source_macro_auc_and_declared_eligibility(self):
        """Verify source-macro AUC averages only eligible two-class sources."""
        records = [
            {'sample_id': '0', 'label': 0, 'dataset_source': 'srcA', 'split': 'dev'},
            {'sample_id': '1', 'label': 1, 'dataset_source': 'srcA', 'split': 'dev'},
            {'sample_id': '2', 'label': 0, 'dataset_source': 'srcB', 'split': 'dev'},
            {'sample_id': '3', 'label': 1, 'dataset_source': 'srcB', 'split': 'dev'},
            {'sample_id': '4', 'label': 1, 'dataset_source': 'srcC', 'split': 'dev'}, # single class
        ]
        dataset = MockDataset(records)
        all_labels = np.array([r['label'] for r in records], dtype=np.float32)
        # srcA: perfect separation (AUC = 1.0)
        # srcB: inverted separation (AUC = 0.0)
        all_preds = np.array([0.1, 0.9, 0.8, 0.2, 0.9], dtype=np.float32)
        all_indices = np.arange(len(records), dtype=np.int64)

        # 1. No declared eligibility filter -> averages srcA and srcB -> (1.0 + 0.0)/2 = 0.5
        res_all = _compute_validation_metrics(all_preds, all_labels, all_indices, 0.2, dataset=dataset)
        self.assertAlmostEqual(res_all['source_macro_auc'], 0.5)

        # 2. Declared eligible sources: only 'srcA'
        opt = SimpleNamespace(eligible_sources=['srcA'])
        res_eligible = _compute_validation_metrics(all_preds, all_labels, all_indices, 0.2, dataset=dataset, opt=opt)
        self.assertEqual(res_eligible['source_macro_auc'], 1.0)

    def test_missing_source_metadata_raises_for_source_macro_monitor(self):
        """Ensure CheckpointHook raises ValueError if source_macro_auc is monitored without metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = MockCheckpointManager(tmpdir)
            hook = CheckpointHook(mgr, monitor_metric='source_macro_auc')
            trainer = SimpleNamespace(best_metric=None, current_epoch=1, global_step=10, scheduler=None)
            res_without_source = ValidationResult(accuracy=0.85, auc=0.90, source_macro_auc=None)

            with self.assertRaises(ValueError) as ctx:
                hook.on_validation_end(trainer, res_without_source)
            self.assertIn("Cannot monitor 'source_macro_auc'", str(ctx.exception))

    def test_dev_calibration_split_and_external_rejection(self):
        """Verify calibration split rules: allows dev/dev_calibration, rejects external folders and test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_hash = 'a' * 64
            out_file = Path(tmpdir) / "threshold.json"

            # 1. Independent dev calibration
            pred_file_indep = Path(tmpdir) / "preds_cal.csv"
            rows_cal = [
                {'sample_id': '1', 'path': 'p1.png', 'label': 0, 'group_id': 'g1', 'split': 'dev_calibration', 'eval_precision': 'fp16'},
                {'sample_id': '2', 'path': 'p2.png', 'label': 1, 'group_id': 'g2', 'split': 'dev_calibration', 'eval_precision': 'fp16'},
            ]
            export_predictions(str(pred_file_indep), rows_cal, [0.2, 0.8], checkpoint_hash=ckpt_hash)
            artifact_indep = calibrate(str(pred_file_indep), str(out_file))
            self.assertEqual(artifact_indep['independence_status'], 'unverified_dev_calibration')
            self.assertEqual(artifact_indep['eval_precision'], 'fp16')

            # 2. Same-dev reused selection and calibration
            pred_file_dev = Path(tmpdir) / "preds_dev.csv"
            rows_dev = [
                {'sample_id': '1', 'path': 'p1.png', 'label': 0, 'group_id': 'g1', 'split': 'dev'},
                {'sample_id': '2', 'path': 'p2.png', 'label': 1, 'group_id': 'g2', 'split': 'dev'},
            ]
            export_predictions(str(pred_file_dev), rows_dev, [0.3, 0.7], checkpoint_hash=ckpt_hash)
            artifact_dev = calibrate(str(pred_file_dev), str(out_file), eval_precision='fp32')
            self.assertEqual(artifact_dev['independence_status'], 'same_dev_reused_selection_and_calibration')
            self.assertEqual(artifact_dev['eval_precision'], 'fp32')

            # 3. Reject external development folder
            pred_file_ext = Path(tmpdir) / "preds_ext.csv"
            rows_ext = [
                {'sample_id': '1', 'path': 'p1.png', 'label': 0, 'group_id': 'g1', 'split': 'external_dev'},
                {'sample_id': '2', 'path': 'p2.png', 'label': 1, 'group_id': 'g2', 'split': 'external_dev'},
            ]
            export_predictions(str(pred_file_ext), rows_ext, [0.3, 0.7], checkpoint_hash=ckpt_hash)
            with self.assertRaises(ValueError) as ctx:
                calibrate(str(pred_file_ext), str(out_file))
            self.assertIn("external development", str(ctx.exception))

            # 4. Reject final test split
            pred_file_test = Path(tmpdir) / "preds_test.csv"
            rows_test = [
                {'sample_id': '1', 'path': 'p1.png', 'label': 0, 'group_id': 'g1', 'split': 'external_test'},
                {'sample_id': '2', 'path': 'p2.png', 'label': 1, 'group_id': 'g2', 'split': 'external_test'},
            ]
            export_predictions(str(pred_file_test), rows_test, [0.3, 0.7], checkpoint_hash=ckpt_hash)
            with self.assertRaises(ValueError) as ctx:
                calibrate(str(pred_file_test), str(out_file))
            self.assertIn("Forbidden", str(ctx.exception))

    def test_plateau_scheduler_steps_only_on_fresh_validation(self):
        """Verify ReduceLROnPlateau consumes fresh metric and only steps when validation ran."""
        scheduler = DummyPlateauScheduler(mode='max')
        hook = SchedulerHook(scheduler, monitor_metric='auc')
        trainer = SimpleNamespace(opt=SimpleNamespace(monitor_metric='auc'))

        # Epoch 1: No validation occurred
        hook.on_epoch_start(trainer)
        hook.on_epoch_end(trainer)
        self.assertEqual(len(scheduler.step_calls), 0, "Plateau scheduler stepped without validation!")

        # Epoch 2: Validation occurred with AUC = 0.94
        hook.on_epoch_start(trainer)
        result = ValidationResult(auc=0.94, accuracy=0.88)
        hook.on_validation_end(trainer, result)
        hook.on_epoch_end(trainer)
        self.assertEqual(len(scheduler.step_calls), 1)
        self.assertEqual(scheduler.step_calls[0], 0.94)

        # Epoch 3: No validation occurred again -> must not step on stale 0.94!
        hook.on_epoch_start(trainer)
        hook.on_epoch_end(trainer)
        self.assertEqual(len(scheduler.step_calls), 1, "Plateau scheduler stepped on stale metric from previous epoch!")

    def test_early_stopping_hook_logic_and_broadcast(self):
        """Verify early stopping patience, min_epochs, and should_stop signaling."""
        hook = EarlyStoppingHook(monitor_metric='auc', patience=2, min_epochs=1)
        trainer = SimpleNamespace(current_epoch=1, should_stop=False, rank=0)

        # Epoch 1: score = 0.80 -> best = 0.80, wait = 0
        hook.on_validation_end(trainer, ValidationResult(auc=0.80))
        self.assertFalse(trainer.should_stop)
        self.assertEqual(hook.best_score, 0.80)

        # Epoch 2: score = 0.75 -> no improvement, wait = 1
        trainer.current_epoch = 2
        hook.on_validation_end(trainer, ValidationResult(auc=0.75))
        self.assertFalse(trainer.should_stop)
        self.assertEqual(hook.wait_count, 1)

        # Epoch 3: score = 0.70 -> no improvement, wait = 2 -> trigger stop!
        trainer.current_epoch = 3
        hook.on_validation_end(trainer, ValidationResult(auc=0.70))
        self.assertTrue(trainer.should_stop)
        self.assertEqual(hook.stopped_epoch, 3)

    def test_checkpoint_hook_epoch_end_scheduler_consistency(self):
        """Verify best.pth resynchronizes with stepped scheduler at epoch end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = MockCheckpointManager(tmpdir)
            hook = CheckpointHook(mgr, monitor_metric='auc')

            scheduler = DummyEpochScheduler()
            records = [{'sample_id': f's{i}', 'path': f'p{i}', 'label': i % 2, 'split': 'dev', 'group_id': f'g{i}'} for i in range(4)]
            mock_val_dataset = MockDataset(records)
            mock_val_loader = SimpleNamespace(dataset=mock_val_dataset)
            trainer = SimpleNamespace(
                best_metric=None,
                current_epoch=1,
                global_step=10,
                scheduler=scheduler,
                val_loader=mock_val_loader,
                opt=SimpleNamespace(name='test_run', seed=42)
            )

            # 1. Validation fires midway through epoch: CheckpointHook stages result without premature saving
            res = ValidationResult(
                auc=0.91,
                accuracy=0.85,
                predictions=np.array([0.1, 0.9, 0.2, 0.8]),
                labels=np.array([0, 1, 0, 1]),
                indices=list(range(4)),
            )
            hook.on_epoch_start(trainer)
            hook.on_validation_end(trainer, res)

            # Not saved yet to prevent pre-scheduler hash discrepancy
            self.assertEqual(len(mgr.saved_best), 0)
            self.assertEqual(scheduler.step_count, 0)

            # 2. Scheduler steps
            scheduler.step()
            self.assertEqual(scheduler.step_count, 1)

            # 3. Epoch end fires: CheckpointHook finalizes best.pth capturing stepped scheduler and exports predictions tied to final hash
            hook.on_epoch_end(trainer)
            self.assertEqual(len(mgr.saved_best), 1)
            self.assertEqual(mgr.saved_best[0]['scheduler'].step_count, 1)

            from data.manifest import sha256
            final_ckpt_hash = sha256(os.path.join(tmpdir, 'best.pth'))
            meta_path = os.path.join(tmpdir, 'best_selection_metadata.json')
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            self.assertEqual(meta['best_checkpoint_sha256'], final_ckpt_hash)

            # 4. Calibrate saved predictions: verify threshold binding matches final best.pth hash
            pred_path = os.path.join(tmpdir, 'best_dev_predictions.csv')
            thresh_out = os.path.join(tmpdir, 'threshold.json')
            calib_res = calibrate(pred_path, thresh_out)
            self.assertEqual(calib_res['checkpoint_sha256'], final_ckpt_hash)

    def test_total_epochs_decay_semantics(self):
        """Verify train.py and BaseTrainer correctly handle niter + niter_decay total epochs."""
        opt = SimpleNamespace(
            dataroot='./fake_dir',
            classes='fake,real',
            gpu_ids=[0],
            batch_size=32,
            lr=0.0001,
            arch='Wang2020_128',
            niter=5,
            niter_decay=3,
            epochs=5,
            epochs_decay=3,
            monitor_metric='source_macro_auc',
            early_stopping=True,
            early_stopping_patience=4,
        )
        cfg = load_config(opt, validate=False, freeze=True)
        opt_clean = config_to_opt(cfg)

        total_epochs = opt_clean.niter + getattr(opt_clean, 'niter_decay', 0)
        self.assertEqual(total_epochs, 8)
        self.assertTrue(opt_clean.early_stopping)
        self.assertEqual(opt_clean.early_stopping_patience, 4)
        self.assertEqual(opt_clean.monitor_metric, 'source_macro_auc')


if __name__ == '__main__':
    unittest.main()
