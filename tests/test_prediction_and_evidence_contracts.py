# STATUS: NOT RUN
"""
Deferred tests for prediction export, threshold calibration, and evidence contracts (Prompt F).
Verifies:
1. CheckpointHook exports dev_predictions.csv linked to best.pth and its exact sha256 checksum upon save_best.
2. Paired difference reports actual aligned full-sample difference rather than CI midpoint, correctly handling asymmetric CIs and honest significance labeling without claiming p < 0.05.
3. Paired comparison and multi-seed summarization support independent model thresholds without cross-contamination.
4. Multi-seed evaluation strictly rejects different epochs from the same training run/seed and requires identical evaluation cohorts.
5. Evidence tables validate producer/consumer schemas and reject empty/invalid split data.
"""
import os
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np


class TestPredictionAndEvidenceContracts(unittest.TestCase):

    def test_checkpoint_hook_exports_dev_predictions_with_sha256(self):
        from training.hooks.checkpoint_hook import CheckpointHook
        from training.validator import ValidationResult

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            ckpt_dir.mkdir(parents=True)

            # Create dummy best.pth
            best_pth = ckpt_dir / "best.pth"
            best_pth.write_bytes(b"mock_checkpoint_binary_content_12345")

            class MockManager:
                save_dir = str(ckpt_dir)
                rank = 0
                def save_best(self, epoch, best_metric, global_step, scheduler=None, amp_state=None):
                    return str(best_pth)

            class MockDataset:
                samples = [
                    ("/path/to/img0.png", 0),
                    ("/path/to/img1.png", 1),
                    ("/path/to/img2.png", 1),
                ]

            class MockLoader:
                dataset = MockDataset()

            class MockTrainer:
                current_epoch = 3
                global_step = 300
                best_metric = 0.80
                scheduler = None
                val_loader = MockLoader()
                def amp_state_dict(self):
                    return None

            result = ValidationResult(
                accuracy=0.92,
                auc=0.95,
                balanced_accuracy=0.91,
                loss=0.25,
                best_threshold=0.58,
                predictions=np.array([0.15, 0.85, 0.72]),
                labels=np.array([0, 1, 1]),
            )

            hook = CheckpointHook(MockManager(), monitor_metric='auc')
            hook.on_validation_end(MockTrainer(), result)

            # Check metadata
            meta_file = ckpt_dir / "best_selection_metadata.json"
            self.assertTrue(meta_file.exists())
            meta = json.loads(meta_file.read_text(encoding='utf-8'))
            self.assertEqual(meta['monitored_metric'], 'auc')
            self.assertEqual(meta['metric_value'], 0.95)
            self.assertIn('best_checkpoint_sha256', meta)

            # Check dev_predictions.csv
            pred_file = ckpt_dir / "dev_predictions.csv"
            self.assertTrue(pred_file.exists())
            from evaluation.generalization import read_predictions
            rows = read_predictions(str(pred_file))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]['checkpoint_sha256'], meta['best_checkpoint_sha256'])
            self.assertEqual(rows[0]['label'], 0)
            self.assertAlmostEqual(rows[0]['probability'], 0.15, places=4)

    def test_paired_comparison_asymmetric_interval_and_observed_difference(self):
        from evaluation.generalization import group_intervals
        from evaluation.reports.evidence_tables import format_paired_comparison_table

        # Construct labels, predictions, and groups
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        p1 = [0.1, 0.2, 0.1, 0.3, 0.8, 0.9, 0.7, 0.85]
        p2 = [0.2, 0.3, 0.2, 0.4, 0.7, 0.8, 0.6, 0.75]
        groups = [f"g{i}" for i in range(8)]

        diffs = group_intervals(labels, p1, groups, threshold=0.5, repeats=100, seed=42,
                                other=p2, other_threshold=0.5)
        self.assertEqual(diffs['status'], 'computed')
        self.assertIn('observed_difference', diffs)
        self.assertIn('accuracy', diffs['observed_difference'])

        acc_ci = diffs['intervals_95']['accuracy']
        self.assertIn('observed', acc_ci)
        self.assertIn('ci_excludes_zero', acc_ci)

        # Check that table format uses actual observed difference rather than CI midpoint
        report = {
            'models': [
                {'overall': {'accuracy': 1.0, 'balanced_accuracy': 1.0, 'roc_auc': 1.0, 'eer': 0.0, 'fpr': 0.0, 'fnr': 0.0}},
                {'overall': {'accuracy': 0.875, 'balanced_accuracy': 0.875, 'roc_auc': 0.9, 'eer': 0.1, 'fpr': 0.1, 'fnr': 0.1}},
            ],
            'first_minus_second': {
                'status': 'computed',
                'observed_difference': {'accuracy': 0.05},
                'intervals_95': {
                    'accuracy': {
                        'observed': 0.05,
                        'low': 0.01,
                        'high': 0.12,  # Asymmetric CI: midpoint is (0.01 + 0.12)/2 = 0.065
                        'ci_excludes_zero': True,
                    }
                }
            }
        }
        table = format_paired_comparison_table(report)
        self.assertIn("+5.00%", table, "Table must display actual observed difference (+5.00%), not CI midpoint (+6.50%)")
        self.assertNotIn("+6.50%", table)
        self.assertIn("Yes (95% CI excludes 0)", table)
        self.assertNotIn("p < 0.05", table, "Table must not claim an uncalculated p-value")

    def test_independent_model_thresholds_in_paired_comparison(self):
        from evaluation.generalization import group_intervals, binary_metrics

        labels = [0, 0, 1, 1]
        p1 = [0.45, 0.55, 0.65, 0.75]
        p2 = [0.35, 0.45, 0.55, 0.65]
        groups = ['g1', 'g2', 'g3', 'g4']

        # Model 1 uses threshold 0.60; Model 2 uses threshold 0.40
        diff = group_intervals(labels, p1, groups, threshold=0.60, repeats=20, seed=42,
                               other=p2, other_threshold=0.40)
        m1_metrics = binary_metrics(labels, p1, threshold=0.60)
        m2_metrics = binary_metrics(labels, p2, threshold=0.40)

        # Expected difference for accuracy
        expected_diff = m1_metrics['accuracy'] - m2_metrics['accuracy']
        self.assertAlmostEqual(diff['observed_difference']['accuracy'], expected_diff, places=4)

    def test_summarize_seeds_rejects_duplicate_epochs_from_same_seed(self):
        from evaluation.generalization import export_predictions, summarize_seeds

        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "run1_seed42_ep5.csv"
            p2 = Path(tmpdir) / "run1_seed42_ep10.csv"
            p3 = Path(tmpdir) / "run2_seed43_ep10.csv"

            # 4 samples
            rows = [
                {'sample_id': f's{i}', 'path': f'p{i}.png', 'label': i % 2, 'group_id': f'g{i}', 'split': 'dev', 'seed': 42}
                for i in range(4)
            ]
            rows_seed43 = [
                {'sample_id': f's{i}', 'path': f'p{i}.png', 'label': i % 2, 'group_id': f'g{i}', 'split': 'dev', 'seed': 43}
                for i in range(4)
            ]

            export_predictions(str(p1), rows, [0.2, 0.8, 0.3, 0.7], checkpoint_hash='ckpt_hash_a')
            export_predictions(str(p2), rows, [0.15, 0.85, 0.25, 0.75], checkpoint_hash='ckpt_hash_b')
            export_predictions(str(p3), rows_seed43, [0.18, 0.82, 0.28, 0.72], checkpoint_hash='ckpt_hash_c')

            # Check that duplicate seed/run_id is rejected even with different checkpoint hashes
            with self.assertRaises(ValueError) as ctx:
                summarize_seeds([str(p1), str(p2), str(p3)])
            self.assertIn("same training run/seed", str(ctx.exception))

    def test_summarize_seeds_rejects_cohort_mismatch(self):
        from evaluation.generalization import export_predictions, summarize_seeds

        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "p1.csv"
            p2 = Path(tmpdir) / "p2.csv"
            p3 = Path(tmpdir) / "p3.csv"

            rows_base = [
                {'sample_id': f's{i}', 'path': f'p{i}.png', 'label': i % 2, 'group_id': f'g{i}', 'split': 'dev', 'seed': 41}
                for i in range(4)
            ]
            rows_mismatch = [
                {'sample_id': f's{i}', 'path': f'p{i}.png', 'label': (i + 1) % 2, 'group_id': f'g{i}', 'split': 'dev', 'seed': 43}
                for i in range(4)
            ]
            rows_s42 = [
                {'sample_id': f's{i}', 'path': f'p{i}.png', 'label': i % 2, 'group_id': f'g{i}', 'split': 'dev', 'seed': 42}
                for i in range(4)
            ]

            export_predictions(str(p1), rows_base, [0.2, 0.8, 0.3, 0.7], checkpoint_hash='h1')
            export_predictions(str(p2), rows_s42, [0.2, 0.8, 0.3, 0.7], checkpoint_hash='h2')
            export_predictions(str(p3), rows_mismatch, [0.2, 0.8, 0.3, 0.7], checkpoint_hash='h3')

            with self.assertRaises(ValueError) as ctx:
                summarize_seeds([str(p1), str(p2), str(p3)])
            self.assertIn("cohort metadata mismatch", str(ctx.exception).lower())

    def test_format_dataset_counts_table_schema_validation(self):
        from evaluation.reports.evidence_tables import format_dataset_counts_table

        # Empty summary without splits raises ValueError
        with self.assertRaises(ValueError) as ctx:
            format_dataset_counts_table({})
        self.assertIn("splits", str(ctx.exception).lower())

        # Valid pilot summary produces full table
        summary = {
            'splits': {
                'train': {'real_count': 50000, 'fake_count': 50000, 'total_count': 100000, 'group_count': 1200, 'frame_cap': 32},
                'dev': {'real_count': 5000, 'fake_count': 5000, 'total_count': 10000, 'group_count': 150, 'frame_cap': 32},
            }
        }
        table = format_dataset_counts_table(summary)
        self.assertIn("**train**", table)
        self.assertIn("50,000", table)
        self.assertIn("1.00:1", table)


if __name__ == '__main__':
    unittest.main()
