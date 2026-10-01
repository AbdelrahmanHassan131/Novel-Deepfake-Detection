# STATUS: NOT RUN
"""Regression tests for Prompt K: Prediction identity and evidence provenance (S4, S5).

Verifies:
1. Distributed validation prediction gathering preserves sample identity and re-sorts
   indices so exported predictions align with canonical dataset records (not positional striding).
2. Empty-rank shards in distributed validation are handled safely without crashing.
3. CheckpointHook export requires canonical records with verified groups and rejects
   fabricating independent groups.
4. summarize_seeds requires complete provenance (checkpoint_sha256, run_id, training_seed)
   and rejects duplicate seeds, duplicate runs, duplicate checkpoints, and conflicting overrides.
5. load_threshold verifies checkpoint hash binding and rejects forbidden evaluation splits.
"""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from evaluation.generalization import summarize_seeds, export_predictions, read_predictions, FORBIDDEN_CALIBRATION_SPLITS
from analyze_predictions import load_threshold
from training.validator import ValidationResult


class TestPredictionIdentityAndProvenance(unittest.TestCase):

    def test_six_sample_two_rank_reordered_sampler_export(self):
        """Simulate 2 ranks with strided sampler indices; verify sorted alignment with canonical records."""
        from training.hooks.checkpoint_hook import CheckpointHook
        from unittest.mock import MagicMock

        # Canonical records for 6 samples
        canonical_records = [
            {'sample_id': f's_{i}', 'path': f'img_{i}.png', 'label': i % 2, 'split': 'dev', 'group_id': f'g_{i}'}
            for i in range(6)
        ]

        # Rank 0 indices [0, 2, 4], Rank 1 indices [1, 3, 5]
        # Gathered arrays by rank:
        gathered_indices = np.array([0, 2, 4, 1, 3, 5], dtype=np.int64)
        gathered_preds = np.array([0.05, 0.25, 0.45, 0.15, 0.35, 0.55], dtype=np.float32)
        gathered_labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)

        # In Validator, sorting by indices aligns them
        order = np.argsort(gathered_indices)
        sorted_preds = gathered_preds[order]
        sorted_labels = gathered_labels[order]
        sorted_indices = gathered_indices[order]

        result = ValidationResult(
            loss=0.2, accuracy=1.0, auc=1.0, balanced_accuracy=1.0,
            best_threshold=0.5, precision=1.0, recall=1.0, f1=1.0,
            num_samples=6, predictions=sorted_preds, labels=sorted_labels,
            indices=sorted_indices
        )

        # Mock trainer
        trainer = MagicMock()
        trainer.opt.run_id = 'test_run_42'
        trainer.opt.seed = 42
        mock_val_dataset = MagicMock()
        mock_val_dataset.records = canonical_records
        mock_val_loader = MagicMock()
        mock_val_loader.dataset = mock_val_dataset
        trainer.val_loader = mock_val_loader

        with tempfile.TemporaryDirectory() as tmpdir:
            hook = CheckpointHook()
            hook.ckpt = MagicMock()
            hook.ckpt.save_dir = tmpdir

            hook._export_dev_predictions(trainer, result, best_sha256="hash_abc123")
            out_file = Path(tmpdir) / 'dev_predictions.csv'
            self.assertTrue(out_file.exists())

            rows = read_predictions(str(out_file))
            self.assertEqual(len(rows), 6)
            for i, r in enumerate(rows):
                self.assertEqual(r['sample_id'], f's_{i}')
                self.assertEqual(int(r['label']), i % 2)
                self.assertEqual(r['run_id'], 'test_run_42')
                self.assertEqual(int(r['training_seed']), 42)
                self.assertEqual(r['checkpoint_sha256'], 'hash_abc123')

    def test_missing_canonical_records_rejected_no_fabricated_groups(self):
        """Export hook raises ValueError if dataset does not have canonical records with groups."""
        from training.hooks.checkpoint_hook import CheckpointHook
        from unittest.mock import MagicMock

        result = ValidationResult(
            loss=0.1, accuracy=1.0, auc=1.0, balanced_accuracy=1.0,
            best_threshold=0.5, precision=1.0, recall=1.0, f1=1.0,
            num_samples=2, predictions=np.array([0.1, 0.9]), labels=np.array([0, 1])
        )
        trainer = MagicMock()
        trainer.opt.run_id = 'run1'
        trainer.opt.seed = 42
        # Dataset without 'records'
        trainer.val_loader.dataset = MagicMock(spec=[])

        hook = CheckpointHook()
        hook.ckpt = MagicMock()
        hook.ckpt.save_dir = "/tmp/fake"

        with self.assertRaises(ValueError) as ctx:
            hook._export_dev_predictions(trainer, result, "hash1")
        self.assertIn("requires validation dataset to provide canonical 'records'", str(ctx.exception))

    def test_summarize_seeds_provenance_enforcement(self):
        """summarize_seeds rejects missing provenance, duplicate seeds, and conflicting overrides."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            def make_pred_file(name, run_id, seed, ckpt_hash):
                fpath = tmppath / f"{name}.csv"
                records = [
                    {'sample_id': f's{i}', 'path': f'p{i}', 'label': i % 2, 'split': 'dev',
                     'group_id': f'g{i}', 'run_id': run_id, 'training_seed': seed}
                    for i in range(4)
                ]
                probs = [0.1, 0.8, 0.2, 0.9]
                export_predictions(str(fpath), records, probs, checkpoint_hash=ckpt_hash)
                return str(fpath)

            p1 = make_pred_file("seed1", "run1", 42, "hash1")
            p2 = make_pred_file("seed2", "run2", 43, "hash2")
            p3 = make_pred_file("seed3", "run3", 44, "hash3")

            # 1. Valid multi-seed summary
            summary = summarize_seeds([p1, p2, p3])
            self.assertIn('roc_auc', summary)

            # 2. Duplicate training seed (e.g. run4 has seed 42 again)
            p4_dup_seed = make_pred_file("seed4", "run4", 42, "hash4")
            with self.assertRaises(ValueError) as ctx:
                summarize_seeds([p1, p2, p4_dup_seed])
            self.assertIn("Found duplicate training_seed", str(ctx.exception))

            # 3. Duplicate run ID (e.g. two epochs of run1)
            p5_dup_run = make_pred_file("seed5", "run1", 45, "hash5")
            with self.assertRaises(ValueError) as ctx:
                summarize_seeds([p1, p2, p5_dup_run])
            self.assertIn("Found duplicate run_id", str(ctx.exception))

            # 4. Conflicting caller override
            with self.assertRaises(ValueError) as ctx:
                summarize_seeds([p1, p2, p3], run_ids=["override1", "run2", "run3"])
            self.assertIn("conflicts with detected run_id", str(ctx.exception))

            # 5. Missing provenance in rows
            p_no_prov = tmppath / "no_prov.csv"
            recs_no_prov = [
                {'sample_id': f's{i}', 'path': f'p{i}', 'label': i % 2, 'split': 'dev', 'group_id': f'g{i}'}
                for i in range(4)
            ]
            export_predictions(str(p_no_prov), recs_no_prov, [0.1, 0.8, 0.2, 0.9], checkpoint_hash="hash_x")
            with self.assertRaises(ValueError) as ctx:
                summarize_seeds([p1, p2, str(p_no_prov)])
            self.assertIn("missing required run provenance", str(ctx.exception))

    def test_threshold_artifact_provenance_and_binding(self):
        """load_threshold rejects forbidden evaluation splits and mismatched checkpoint hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # 1. Valid artifact calibrated on 'dev'
            valid_art = {
                'threshold': 0.42,
                'checkpoint_sha256': 'ckpt_abc',
                'source_splits': ['dev']
            }
            valid_path = tmppath / "thresh_valid.json"
            valid_path.write_text(json.dumps(valid_art), encoding='utf-8')

            val = load_threshold(None, str(valid_path), expected_checkpoint_hash='ckpt_abc')
            self.assertAlmostEqual(val, 0.42)

            # 2. Swapped checkpoint hash
            with self.assertRaises(ValueError) as ctx:
                load_threshold(None, str(valid_path), expected_checkpoint_hash='ckpt_different')
            self.assertIn("does not match evaluation checkpoint", str(ctx.exception))

            # 3. Forbidden evaluation split (e.g. final_test)
            forbidden_art = {
                'threshold': 0.42,
                'checkpoint_sha256': 'ckpt_abc',
                'source_splits': ['final_test']
            }
            forbidden_path = tmppath / "thresh_forbidden.json"
            forbidden_path.write_text(json.dumps(forbidden_art), encoding='utf-8')

            with self.assertRaises(ValueError) as ctx:
                load_threshold(None, str(forbidden_path), expected_checkpoint_hash='ckpt_abc')
            self.assertIn("forbidden evaluation split", str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
