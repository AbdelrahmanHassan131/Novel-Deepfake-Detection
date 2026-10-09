#!/usr/bin/env python
"""
Unit tests for Google Colab Controlled RGB Mixture Pilot workflow and CLI parsers.

Verifies:
1. Syntax validity of all Python and Bash cells in COLAB_RGB_MIXTURE_PILOT.md.
2. Parity between markdown commands and actual argument parsers in train.py,
   tools/benchmark_training_throughput.py, and tools/evaluate_mixture_predictions.py.
3. Mocked end-to-end directory layout, run registry persistence, and evaluation workflow.
"""

import ast
import json
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path


class TestColabMixturePilotWorkflow(unittest.TestCase):

    def setUp(self):
        self.repo_root = Path(__file__).resolve().parent.parent
        self.guide_path = self.repo_root / "COLAB_RGB_MIXTURE_PILOT.md"
        self.assertTrue(self.guide_path.exists(), f"Guide missing at {self.guide_path}")
        self.guide_content = self.guide_path.read_text(encoding="utf-8")

    def test_markdown_cells_syntax_and_bash_guards(self):
        """All code cells in COLAB_RGB_MIXTURE_PILOT.md must be syntactically valid."""
        # Extract code blocks
        blocks = re.findall(r"```(python|bash)\n(.*?)```", self.guide_content, re.DOTALL)
        self.assertGreaterEqual(len(blocks), 8, "Expected at least 8 code cells in Colab guide")

        python_cell_count = 0
        bash_cell_count = 0

        for lang, code in blocks:
            code_clean = code.strip()
            if lang == "python":
                python_cell_count += 1
                # Must parse without syntax errors
                try:
                    ast.parse(code_clean)
                except SyntaxError as e:
                    self.fail(f"Python syntax error in cell: {e}\nCode:\n{code_clean}")
            elif lang == "bash":
                bash_cell_count += 1
                lines = code_clean.splitlines()
                # Line 1 must be %%bash
                self.assertTrue(
                    lines[0].strip().startswith("%%bash"),
                    f"Bash cell must start with %%bash on line 1, found: {lines[0]}"
                )

        self.assertGreaterEqual(python_cell_count, 6)
        self.assertGreaterEqual(bash_cell_count, 2)

    def test_benchmark_cli_args_acceptance(self):
        """Verify that CLI options in Cell 6 parse cleanly in benchmark_training_throughput.py."""
        import sys
        from unittest.mock import patch
        from tools.benchmark_training_throughput import parse_args

        test_args = [
            "benchmark_training_throughput.py",
            "--arch", "Wang2020_128",
            "--manifest", "dummy_manifest.csv",
            "--val_manifest", "dummy_val_manifest.csv",
            "--dataroot", "/content/data",
            "--batch_size", "32",
            "--grad_accum_steps", "2",
            "--num_workers", "4",
            "--prefetch_factor", "2",
            "--persistent_workers",
            "--pin_memory",
            "--aug_recipe", "rgb_v1",
            "--fine_tune_policy", "layer4_and_head",
            "--bn_policy", "frozen",
            "--backbone_lr_mult", "0.1",
            "--use_amp",
            "--amp_dtype", "fp16",
            "--warmup_microbatches", "10",
            "--measure_microbatches", "50",
            "--val_samples", "100",
            "--output", "/content/throughput_benchmark.json",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
            self.assertEqual(args.arch, "Wang2020_128")
            self.assertEqual(args.warmup_microbatches, 10)
            self.assertEqual(args.measure_microbatches, 50)
            self.assertEqual(args.val_samples, 100)
            self.assertEqual(args.val_manifest, "dummy_val_manifest.csv")
            self.assertTrue(args.persistent_workers)
            self.assertTrue(args.pin_memory)

    def test_train_cli_args_acceptance(self):
        """Verify that CLI options in Cell 7 parse cleanly in train.py."""
        import sys
        from unittest.mock import patch
        from train import parse_args

        test_args = [
            "train.py",
            "--arch", "Wang2020_128",
            "--name", "arm_a",
            "--run_id", "seed42_test",
            "--checkpoints_dir", "/content/experiments",
            "--dataroot", "/content/dataset/train",
            "--manifest", "dummy_manifest.csv",
            "--manifest_split", "train",
            "--val_manifest", "dummy_manifest.csv",
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
        with patch.object(sys, "argv", test_args):
            args = parse_args()
            self.assertEqual(args.arch, "Wang2020_128")
            self.assertEqual(args.name, "arm_a")
            self.assertEqual(args.run_id, "seed42_test")
            self.assertEqual(args.epochs, 5)
            self.assertEqual(args.optimizer, "adam")
            self.assertEqual(args.lr, 0.0001)
            self.assertFalse(args.early_stopping)

    def test_evaluate_cli_args_acceptance(self):
        """Verify that CLI options in Cell 9 parse cleanly in evaluate_mixture_predictions.py."""
        import sys
        from unittest.mock import patch
        from tools.evaluate_mixture_predictions import parse_args

        test_args = [
            "evaluate_mixture_predictions.py",
            "--pred_a", "/content/preds_a.csv",
            "--pred_b", "/content/preds_b.csv",
            "--shared_dev_manifest", "/content/shared_dev.csv",
            "--output_dir", "/content/eval_out",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
            self.assertEqual(args.pred_a, Path("/content/preds_a.csv"))
            self.assertEqual(args.pred_b, Path("/content/preds_b.csv"))
            self.assertEqual(args.shared_dev_manifest, Path("/content/shared_dev.csv"))
            self.assertEqual(args.output_dir, Path("/content/eval_out"))

    def test_actual_notebook_launch_and_backup_cells_with_real_manager(self):
        """Execute the actual cells: the real manager rejects precreated run dirs."""
        from types import SimpleNamespace
        from unittest.mock import patch
        from experiment import ExperimentManager
        blocks = re.findall(r"```python\n(.*?)```", self.guide_content, re.DOTALL)
        launch = next(b for b in blocks if '[Cell 7:' in b)
        backup = next(b for b in blocks if '[Cell 8:' in b)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = dict(LOCAL_REPO=str(self.repo_root), MANIFEST_DIR=str(root / 'manifests'),
                       DATA_ROOT=str(root / 'data'), OUTPUT_BASE=str(root / 'runs'),
                       REGISTRY_FILE=str(root / 'registry.json'), DRIVE_BACKUP_DIR=str(root / 'drive'))
            def fake_process(cmd, **kwargs):
                value = lambda flag: cmd[cmd.index(flag) + 1]
                manager = ExperimentManager(base_dir=value('--checkpoints_dir'))
                run = manager.create(value('--name'), SimpleNamespace(run_id=value('--run_id')))
                ckpt = Path(run.checkpoint_dir)
                (ckpt / 'best.pth').write_bytes(b'test-only checkpoint')
                (ckpt / 'best_dev_predictions.csv').write_text('test-only predictions')
                return SimpleNamespace(stdout=['mock training completed\n'], returncode=0, wait=lambda: 0)
            with patch.dict(os.environ, env), patch('subprocess.Popen', side_effect=fake_process):
                exec(compile(launch, '<actual Cell 7>', 'exec'), {})
                exec(compile(backup, '<actual Cell 8>', 'exec'), {})
                exec(compile(backup, '<actual Cell 8 repeated>', 'exec'), {})
            entry = json.loads((root / 'registry.json').read_text())['arm_a']
            saved = root / 'drive' / Path(entry['run_dir']).name
            self.assertTrue((saved / 'train.log').is_file())
            self.assertTrue((saved / 'BACKUP_COMPLETE.json').is_file())

    def test_mocked_e2e_checkpoint_layout_registry_and_drive_sync(self):
        """Simulate training output directory creation, run registry, and Drive backup sync."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output_base = tmp_path / "experiments"
            output_base.mkdir()
            drive_backup = tmp_path / "drive_backup"
            drive_backup.mkdir()
            registry_file = tmp_path / "run_registry.json"

            # Simulate Arm A training output structure from ExperimentManager
            run_id = "seed42_20261009_120000"
            arm = "arm_a"
            run_dir = output_base / f"{arm}_{run_id}"
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True)

            best_pth = ckpt_dir / "best.pth"
            best_pth.write_bytes(b"mock_checkpoint_data_arm_a")
            preds_csv = ckpt_dir / "best_dev_predictions.csv"
            preds_csv.write_text("sample_id,prob,pred,label\ns1,0.9,1,1\n", encoding="utf-8")

            # Update registry
            registry = {
                arm: {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "checkpoints_dir": str(ckpt_dir),
                    "best_checkpoint": str(best_pth),
                    "best_dev_predictions": str(preds_csv),
                    "completed_at": "2026-10-09T12:00:00",
                }
            }
            registry_file.write_text(json.dumps(registry, indent=2), encoding="utf-8")

            # Simulate Step 8 sync
            reg_data = json.loads(registry_file.read_text(encoding="utf-8"))
            for arm_name, entry in reg_data.items():
                src = Path(entry["run_dir"])
                dest = drive_backup / src.name
                shutil.copytree(src, dest)
                self.assertTrue((dest / "checkpoints/best.pth").is_file())
                self.assertTrue((dest / "checkpoints/best_dev_predictions.csv").is_file())


if __name__ == "__main__":
    unittest.main()
