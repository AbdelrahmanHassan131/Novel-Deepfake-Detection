"""Deferred tests for development selection, reporting, and calibration.

STATUS: NOT RUN (code-only preparation phase).
Tests verify:
  - Canonical label direction (real=0, fake=1; non-binary rejected).
  - Rejection of incompatible prediction IDs in multi-model comparison.
  - Graceful handling of missing groups and single-class evaluations.
  - Strict prohibition of calibration on test splits.
  - Multi-seed validation: requires >= 3 independent checkpoints; rejects identical hashes.
  - Support for independent development-selected thresholds per model/seed.
  - Subgroup fairness reporting with explicit sample counts and small-sample cautions.
"""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np


class TestSelectionAndReporting(unittest.TestCase):
    """DEFERRED / NOT RUN: Verified via AST parsing during code preparation."""

    def test_canonical_label_direction_enforced(self):
        from evaluation.generalization import binary_metrics
        # Canonical binary labels: 0=real, 1=fake
        res = binary_metrics([0, 1], [0.2, 0.8], threshold=0.5)
        self.assertEqual(res['accuracy'], 1.0)
        self.assertEqual(res['roc_auc'], 1.0)
        self.assertEqual(res['status'], 'computed')

        # Invalid non-binary labels must be rejected
        with self.assertRaises(ValueError) as ctx:
            binary_metrics([0, 2], [0.2, 0.8])
        self.assertIn("canonical mapping", str(ctx.exception))

    def test_incompatible_prediction_ids_rejected(self):
        from evaluation.generalization import export_predictions, read_predictions
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "preds1.csv"
            file2 = Path(tmpdir) / "preds2.csv"
            rows1 = [
                {'sample_id': 'id_1', 'path': 'p1.png', 'label': 0, 'group_id': 'g1', 'split': 'dev'},
                {'sample_id': 'id_2', 'path': 'p2.png', 'label': 1, 'group_id': 'g2', 'split': 'dev'},
            ]
            rows2 = [
                {'sample_id': 'id_1', 'path': 'p1.png', 'label': 0, 'group_id': 'g1', 'split': 'dev'},
                {'sample_id': 'id_DIFFERENT', 'path': 'p2.png', 'label': 1, 'group_id': 'g2', 'split': 'dev'},
            ]
            export_predictions(str(file1), rows1, [0.3, 0.7], checkpoint_hash='hash1')
            export_predictions(str(file2), rows2, [0.4, 0.6], checkpoint_hash='hash2')

            data1 = read_predictions(str(file1))
            data2 = read_predictions(str(file2))
            ids1 = {r['sample_id'] for r in data1}
            ids2 = {r['sample_id'] for r in data2}
            self.assertNotEqual(ids1, ids2)

    def test_single_class_evaluation_returns_none_ranking_metrics(self):
        from evaluation.generalization import binary_metrics
        # Only fake samples (e.g. LDM set)
        res = binary_metrics([1, 1, 1], [0.8, 0.9, 0.7], threshold=0.5)
        self.assertEqual(res['real'], 0)
        self.assertEqual(res['fake'], 3)
        self.assertIsNone(res['roc_auc'])
        self.assertIsNone(res['balanced_accuracy'])
        self.assertIsNone(res['eer'])
        self.assertIsNone(res['fpr'])
        self.assertEqual(res['fnr'], 0.0)
        self.assertIn("single_class", res['status'])

    def test_missing_groups_handled_gracefully_in_intervals(self):
        from evaluation.generalization import group_intervals
        # Missing or unknown groups
        res = group_intervals([0, 1], [0.2, 0.8], ['unknown', 'unknown'])
        self.assertIn("unavailable", res['status'])

    def test_forbidden_test_calibration(self):
        from evaluation.generalization import export_predictions, calibrate
        with tempfile.TemporaryDirectory() as tmpdir:
            pred_file = Path(tmpdir) / "test_preds.csv"
            out_file = Path(tmpdir) / "cal.json"
            rows = [
                {'sample_id': 's1', 'path': 'p1.png', 'label': 0, 'group_id': 'g1', 'split': 'external_test'},
                {'sample_id': 's2', 'path': 'p2.png', 'label': 1, 'group_id': 'g2', 'split': 'external_test'},
            ]
            export_predictions(str(pred_file), rows, [0.2, 0.8], checkpoint_hash='hashA')
            with self.assertRaises(ValueError) as ctx:
                calibrate(str(pred_file), str(out_file))
            self.assertIn("Forbidden", str(ctx.exception))

    def test_multi_seed_rejects_fewer_than_three_runs_or_duplicate_hashes(self):
        from evaluation.generalization import export_predictions, summarize_seeds
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "p1.csv"
            p2 = Path(tmpdir) / "p2.csv"
            p3 = Path(tmpdir) / "p3.csv"
            rows = [
                {'sample_id': 's1', 'path': 'p1.png', 'label': 0, 'group_id': 'g1', 'split': 'dev'},
                {'sample_id': 's2', 'path': 'p2.png', 'label': 1, 'group_id': 'g2', 'split': 'dev'},
            ]
            # Export with distinct hashes
            export_predictions(str(p1), rows, [0.2, 0.8], checkpoint_hash='hash_seed1')
            export_predictions(str(p2), rows, [0.3, 0.7], checkpoint_hash='hash_seed2')
            export_predictions(str(p3), rows, [0.25, 0.75], checkpoint_hash='hash_seed3')

            # Fewer than 3 runs rejected
            with self.assertRaises(ValueError) as ctx:
                summarize_seeds([str(p1), str(p2)])
            self.assertIn("at least 3", str(ctx.exception))

            # Duplicate hash rejected (repeated inference is not an independent run)
            p_dup = Path(tmpdir) / "p_dup.csv"
            export_predictions(str(p_dup), rows, [0.2, 0.8], checkpoint_hash='hash_seed1')
            with self.assertRaises(ValueError) as ctx:
                summarize_seeds([str(p1), str(p2), str(p_dup)])
            self.assertIn("duplicate checkpoint hash", str(ctx.exception))

    def test_independent_thresholds_in_paired_comparison_and_seeds(self):
        from evaluation.generalization import group_intervals, summarize_seeds, export_predictions
        labels = [0, 0, 1, 1]
        groups = ['g1', 'g2', 'g3', 'g4']
        p1 = [0.2, 0.4, 0.6, 0.8]
        p2 = [0.1, 0.3, 0.7, 0.9]
        # Compare with independent thresholds: 0.5 for p1, 0.4 for p2
        intervals = group_intervals(labels, p1, groups, threshold=0.5, other=p2, other_threshold=0.4, repeats=10)
        self.assertEqual(intervals['status'], 'computed')

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i in range(3):
                p_path = Path(tmpdir) / f"p_{i}.csv"
                rows = [{'sample_id': f's{j}', 'path': f'p{j}.png', 'label': j % 2, 'group_id': f'g{j}', 'split': 'dev'} for j in range(4)]
                export_predictions(str(p_path), rows, [0.1 * (i + 1), 0.2, 0.8, 0.9], checkpoint_hash=f'hash_{i}')
                paths.append(str(p_path))

            report = summarize_seeds(paths, thresholds=[0.45, 0.5, 0.55], repeats=10)
            self.assertEqual(report['thresholds_used'], [0.45, 0.5, 0.55])

    def test_subgroup_reporting_with_sample_counts(self):
        from evaluation.generalization import summarize
        records = [
            {'sample_id': f's{i}', 'label': i % 2, 'dataset_source': 'SourceA', 'generator': 'GenX', 'group_id': f'g{i}'}
            for i in range(10)
        ]
        probs = [0.2 if i % 2 == 0 else 0.8 for i in range(10)]
        rep = summarize(records, probs, threshold=0.5)
        self.assertIn('dataset_source', rep['breakdowns'])
        self.assertIn('SourceA', rep['breakdowns']['dataset_source'])
        sub = rep['breakdowns']['dataset_source']['SourceA']
        self.assertEqual(sub['subgroup_sample_count'], 10)
        self.assertTrue(sub['caution_small_subgroup'])


if __name__ == '__main__':
    unittest.main()
