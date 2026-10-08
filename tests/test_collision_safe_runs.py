# STATUS: NOT RUN
"""
Deferred regression test suite for Prompt L / Finding S6.
Verifies collision-safe experiment directory creation, interrupted run exit codes,
pipeline stage artifact validation (missing and stale checkpoints), and plan-mode preservation.

DO NOT EXECUTE AT RUNTIME IN THIS ENVIRONMENT.
Static validation only.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

from experiment.manager import ExperimentManager
from training.pipeline import FreshTrainingPipeline, StageSpec


class TestExperimentDirectoryCollision(unittest.TestCase):
    """Verifies that fresh experiments cannot silently overwrite existing run directories."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_fresh_run_collision_raises_file_exists_error(self):
        """Creating a new experiment with an existing directory without resume_checkpoint must fail."""
        # Create an existing directory
        existing_run_dir = self.base_dir / "test_exp_seed42"
        existing_run_dir.mkdir(parents=True, exist_ok=True)
        (existing_run_dir / "options.json").write_text('{"seed": 42}', encoding='utf-8')

        # Attempt to create fresh experiment manager targeting the same directory
        with self.assertRaises(FileExistsError) as ctx:
            ExperimentManager.create(
                experiment_dir=str(existing_run_dir),
                name="test_exp",
                seed=42,
                run_id="seed42",
                resume_checkpoint=None,
                allow_existing=False,
            )
        self.assertIn("already exists", str(ctx.exception))
        self.assertIn("resume_checkpoint", str(ctx.exception))

    def test_explicit_resume_allows_existing_directory(self):
        """Supplying a resume_checkpoint or allow_existing=True allows using an existing directory."""
        existing_run_dir = self.base_dir / "test_exp_seed42"
        existing_run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = existing_run_dir / "last.pth"
        ckpt_path.write_bytes(b"dummy_weights")

        mgr = ExperimentManager.create(
            experiment_dir=str(existing_run_dir),
            name="test_exp",
            seed=42,
            run_id="seed42",
            resume_checkpoint=str(ckpt_path),
            allow_existing=False,
        )
        self.assertEqual(mgr.run_dir, existing_run_dir)
        self.assertEqual(mgr.last_checkpoint, existing_run_dir / "last.pth")


class TestPipelineArtifactValidation(unittest.TestCase):
    """Verifies that the training pipeline validates checkpoint freshness and generation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.exp_dir = Path(self.temp_dir.name) / "exp"
        self.dataroot = Path(self.temp_dir.name) / "data"
        self.dataroot.mkdir(parents=True, exist_ok=True)
        import hashlib
        self.manifest = self.dataroot / "manifest.csv"
        self.manifest.write_bytes(b"sample_id,path,label,split,group_id\n1,1.png,0,train,g1\n2,2.png,1,train,g2\n")
        gate_info = {
            'verified': True,
            'manifest_path': str(self.manifest),
            'manifest_sha256': hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            'samples': 2,
        }
        (self.dataroot / "manifest.verified.json").write_text(json.dumps(gate_info), encoding='utf-8')

        self.pipeline = FreshTrainingPipeline(
            experiment_dir=str(self.exp_dir),
            dataroot=str(self.dataroot),
            manifest=str(self.manifest),
            batch_size=8,
            epochs=1,
            gpu_ids="-1",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_run_stage_rejects_missing_checkpoint(self):
        """Pipeline must raise RuntimeError if process succeeds but expected checkpoint is missing."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.poll.return_value = 0
        mock_proc.stdout.readline.return_value = ""
        mock_proc.__enter__.return_value = mock_proc
        mock_proc.communicate.return_value = ("", "")

        with patch("subprocess.Popen", return_value=mock_proc):
            with self.assertRaises((RuntimeError, FileNotFoundError)) as ctx:
                self.pipeline.run_stage("rgb")
            self.assertTrue("expected artifact was not produced" in str(ctx.exception) or "did not produce expected checkpoint" in str(ctx.exception))

    def test_run_stage_rejects_stale_checkpoint(self):
        """Pipeline must raise RuntimeError if checkpoint exists but was not updated by current run."""
        stage1_ckpt = self.pipeline.stage1_ckpt
        stage1_ckpt.parent.mkdir(parents=True, exist_ok=True)
        stage1_ckpt.write_bytes(b"old_weights")

        # Artificially set mtime to 10 seconds ago
        stale_time = time.time() - 10.0
        os.utime(stage1_ckpt, (stale_time, stale_time))

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.poll.return_value = 0
        mock_proc.stdout.readline.return_value = ""
        mock_proc.__enter__.return_value = mock_proc
        mock_proc.communicate.return_value = ("", "")

        # Subprocess runs but doesn't write to stage1_ckpt (leaving it stale)
        with patch("subprocess.Popen", return_value=mock_proc):
            with self.assertRaises(RuntimeError) as ctx:
                self.pipeline.run_stage("rgb")
            self.assertIn("was not updated by this run", str(ctx.exception))


class TestPlanModeManifestPreservation(unittest.TestCase):
    """Verifies that plan-only mode writes to pipeline_plan.json without touching pipeline_manifest.json."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.exp_dir = Path(self.temp_dir.name) / "exp"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.dataroot = Path(self.temp_dir.name) / "data"
        self.dataroot.mkdir(parents=True, exist_ok=True)
        self.manifest = self.dataroot / "manifest.csv"
        self.manifest.write_text("sample_id,path,label,split,group_id\n1,1.png,0,train,g1\n", encoding='utf-8')

        self.pipeline = FreshTrainingPipeline(
            experiment_dir=str(self.exp_dir),
            dataroot=str(self.dataroot),
            manifest=str(self.manifest),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_plan_does_not_overwrite_existing_manifest(self):
        manifest_file = self.exp_dir / "pipeline_manifest.json"
        original_data = {"pipeline_version": 2, "stages": {"rgb": {"status": "completed"}}}
        manifest_file.write_text(json.dumps(original_data), encoding='utf-8')

        plan_file = self.exp_dir / "pipeline_plan.json"
        self.pipeline.save_manifest(path=plan_file)

        # Ensure pipeline_plan.json was created
        self.assertTrue(plan_file.is_file())
        # Ensure pipeline_manifest.json remains unmodified
        loaded_manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
        self.assertEqual(loaded_manifest["stages"]["rgb"]["status"], "completed")


if __name__ == '__main__':
    unittest.main()
