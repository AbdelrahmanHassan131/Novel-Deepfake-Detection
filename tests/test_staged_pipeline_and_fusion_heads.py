# STATUS: NOT RUN
# Execution restriction: Static AST verification only. Do not execute test suite now.
"""
Tests for Prompt D: Staged Fresh-Training Pipeline, Artifact Registration, and Fusion Head Resolution.

Verifies:
1. Plan mode is side-effect-free (no model instantiation, no checkpoint files required).
2. Immutable run identifiers reconcile ExperimentManager output paths with pipeline Stage 3 expert inputs.
3. Architecture & fusion head resolution: gated, concat, and token attention map to distinct classes.
4. Unsupported combinations (e.g. MHA_128 with concat, Fusion_128 with token_attention) are strictly rejected.
5. Expert checkpoint validation contract: architecture, label direction, and preprocessing protocol checks.
6. Mocked subprocess launcher verifies stage execution, exit-code propagation, and artifact sha256 registration.
7. Expert checkpoint paths are shared across all fusion heads for the same seed.
"""
import unittest
from unittest.mock import MagicMock
import tempfile
import json
from pathlib import Path

from training.pipeline import (
    FreshTrainingPipeline,
    FUSION_TYPE_TO_ARCH,
    VALID_FUSION_COMBINATIONS,
    file_sha256,
)
from models.fusion.trainer import ControlledFusionTrainer, ConcatenationFusionTrainer
from models.mha.trainer import MHAFusionTrainer
from models.fusion.fusion_classifier import ConcatenationFusionClassifier
from models.mha.token_fusion import TokenAttentionFusion, GatedFusion
from models.shared.expert_loading import validate_expert_checkpoint


class TestStagedPipelineAndFusionHeads(unittest.TestCase):
    """Static and structural test suite for fresh training pipeline and fusion heads."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.exp_dir = Path(self.temp_dir.name) / "exp"
        self.dataroot = Path(self.temp_dir.name) / "data"
        self.manifest = Path(self.temp_dir.name) / "manifest.csv"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.dataroot.mkdir(parents=True, exist_ok=True)
        self.manifest.write_text("filepath,label,source_group,split\na.png,0,g1,train\n", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_plan_mode_is_side_effect_free(self):
        """Plan mode must generate valid manifests and commands without requiring checkpoints on disk."""
        pipeline = FreshTrainingPipeline(
            experiment_dir=str(self.exp_dir),
            dataroot=str(self.dataroot),
            manifest=str(self.manifest),
            seed=42,
            fusion_type="token_attention",
        )
        manifest_path = pipeline.save_manifest()
        self.assertTrue(manifest_path.is_file())

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["stages"]["rgb"]["status"], "pending")
        self.assertEqual(data["stages"]["wavelet"]["status"], "pending")
        self.assertEqual(data["stages"]["fusion"]["status"], "pending")

        # Verify Stage 3 points to exact Stage 1 and Stage 2 outputs
        stage1_ckpt = data["stages"]["rgb"]["checkpoint_path"]
        stage2_ckpt = data["stages"]["wavelet"]["checkpoint_path"]
        cmd3 = data["stages"]["fusion"]["command"]
        self.assertIn("--rgb_model_path", cmd3)
        self.assertIn(stage1_ckpt, cmd3)
        self.assertIn("--wavelet_model_path", cmd3)
        self.assertIn(stage2_ckpt, cmd3)

    def test_immutable_run_id_path_reconciliation(self):
        """Pipeline must pass --run_id to train.py matching the exact directory structure created."""
        run_id = "test_pilot_seed42"
        pipeline = FreshTrainingPipeline(
            experiment_dir=str(self.exp_dir),
            dataroot=str(self.dataroot),
            manifest=str(self.manifest),
            seed=42,
            run_id=run_id,
            fusion_type="gated",
        )
        # Verify run_id is in stage commands
        for stage_name, spec in pipeline.stages.items():
            self.assertIn("--run_id", spec.command)
            run_id_idx = spec.command.index("--run_id")
            self.assertEqual(spec.command[run_id_idx + 1], run_id)

        # Verify checkpoint paths include run_id and checkpoints/best.pth
        stage1_spec = pipeline.stages["rgb"]
        self.assertIn(f"stage1_rgb_{run_id}", stage1_spec.checkpoint_path)
        self.assertTrue(stage1_spec.checkpoint_path.endswith("best.pth"))

        stage2_spec = pipeline.stages["wavelet"]
        self.assertIn(f"stage2_wavelet_{run_id}", stage2_spec.checkpoint_path)
        self.assertTrue(stage2_spec.checkpoint_path.endswith("best.pth"))

    def test_fusion_architecture_and_head_class_resolution(self):
        """Gated, concat, and token attention must resolve to distinct, correct classes."""
        self.assertEqual(VALID_FUSION_COMBINATIONS[("Fusion_128", "concat")], "ConcatenationFusionClassifier")
        self.assertEqual(VALID_FUSION_COMBINATIONS[("Fusion_128", "gated")], "GatedFusion")
        self.assertEqual(VALID_FUSION_COMBINATIONS[("MHA_128", "token_attention")], "TokenAttentionFusion")

        # Verify distinct head classes
        self.assertIsNot(ConcatenationFusionClassifier, GatedFusion)
        self.assertIsNot(ConcatenationFusionClassifier, TokenAttentionFusion)
        self.assertIsNot(GatedFusion, TokenAttentionFusion)

        # Test ControlledFusionTrainer head construction
        trainer = ControlledFusionTrainer.__new__(ControlledFusionTrainer)
        trainer.embed_dim = 128

        class DummyOpt:
            def __init__(self, fusion_type, dropout=0.1):
                self.fusion_type = fusion_type
                self.dropout = dropout

        concat_head = trainer.make_head(DummyOpt("concat"))
        self.assertIsInstance(concat_head, ConcatenationFusionClassifier)

        gated_head = trainer.make_head(DummyOpt("gated"))
        self.assertIsInstance(gated_head, GatedFusion)

        with self.assertRaises(ValueError):
            trainer.make_head(DummyOpt("token_attention"))

        # Test MHAFusionTrainer head construction
        mha_trainer = MHAFusionTrainer.__new__(MHAFusionTrainer)
        mha_trainer.embed_dim = 128

        token_head = mha_trainer.make_head(DummyOpt("token_attention"))
        self.assertIsInstance(token_head, TokenAttentionFusion)

        with self.assertRaises(ValueError):
            mha_trainer.make_head(DummyOpt("concat"))

    def test_rejection_of_unsupported_fusion_combinations(self):
        """FreshTrainingPipeline must reject unsupported architecture/strategy pairings."""
        with self.assertRaises(ValueError):
            FreshTrainingPipeline(
                experiment_dir=str(self.exp_dir),
                dataroot=str(self.dataroot),
                manifest=str(self.manifest),
                fusion_arch="MHA_128",
                fusion_type="concat",
            )

        with self.assertRaises(ValueError):
            FreshTrainingPipeline(
                experiment_dir=str(self.exp_dir),
                dataroot=str(self.dataroot),
                manifest=str(self.manifest),
                fusion_arch="Fusion_128",
                fusion_type="token_attention",
            )

    def test_expert_checkpoint_validation_contract(self):
        """validate_expert_checkpoint must enforce architecture, labels, and preprocessing keys."""
        import torch

        # Valid RGB state dict
        valid_rgb_ckpt = {
            "model_state_dict": {"fc.0.weight": torch.zeros(128, 2048)},
            "protocol": {
                "label_mapping": {"real": 0, "fake": 1},
                "options": {"arch": "Wang2020_128", "cropSize": 224},
            },
        }
        # Should pass
        validate_expert_checkpoint(valid_rgb_ckpt, expected_type="rgb", expected_embed_dim=128)

        # 1. Dim mismatch
        bad_dim_ckpt = {
            "model_state_dict": {"fc.0.weight": torch.zeros(64, 2048)},
            "protocol": {"label_mapping": {"real": 0, "fake": 1}},
        }
        with self.assertRaises(ValueError):
            validate_expert_checkpoint(bad_dim_ckpt, expected_type="rgb", expected_embed_dim=128)

        # 2. Architecture mismatch
        bad_arch_ckpt = {
            "model_state_dict": {"fc.0.weight": torch.zeros(128, 2048)},
            "protocol": {
                "label_mapping": {"real": 0, "fake": 1},
                "options": {"arch": "WolterWavelet2021_128"},
            },
        }
        with self.assertRaises(ValueError):
            validate_expert_checkpoint(bad_arch_ckpt, expected_type="rgb", expected_embed_dim=128)

        # 3. Inverted label mapping
        bad_labels_ckpt = {
            "model_state_dict": {"fc.0.weight": torch.zeros(128, 2048)},
            "protocol": {"label_mapping": {"real": 1, "fake": 0}},
        }
        with self.assertRaises(ValueError):
            validate_expert_checkpoint(bad_labels_ckpt, expected_type="rgb", expected_embed_dim=128)

        # 4. Preprocessing option mismatch
        class Opt:
            cropSize = 256

        with self.assertRaises(ValueError):
            validate_expert_checkpoint(valid_rgb_ckpt, expected_type="rgb", expected_embed_dim=128, opt=Opt())

    def test_mocked_subprocess_pipeline_execution_and_artifact_registration(self):
        """Pipeline execution must register artifacts with sha256 checksums and propagate failure exit codes."""
        mock_launcher = MagicMock()

        def side_effect(cmd):
            # Create the checkpoint file expected by Stage 1
            ckpt_path = Path(pipeline.stages["rgb"].checkpoint_path)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt_path.write_bytes(b"mock_checkpoint_data_bytes")
            res = MagicMock()
            res.returncode = 0
            return res

        mock_launcher.side_effect = side_effect

        pipeline = FreshTrainingPipeline(
            experiment_dir=str(self.exp_dir),
            dataroot=str(self.dataroot),
            manifest=str(self.manifest),
            seed=42,
            launcher=mock_launcher,
        )

        rc = pipeline.run_stage("rgb")
        self.assertEqual(rc, 0)
        self.assertEqual(pipeline.stages["rgb"].status, "completed")
        self.assertIsNotNone(pipeline.stages["rgb"].checkpoint_sha256)
        expected_sha = file_sha256(Path(pipeline.stages["rgb"].checkpoint_path))
        self.assertEqual(pipeline.stages["rgb"].checkpoint_sha256, expected_sha)

        # Test failure propagation
        fail_launcher = MagicMock()
        fail_res = MagicMock()
        fail_res.returncode = 2
        fail_launcher.return_value = fail_res

        pipeline_fail = FreshTrainingPipeline(
            experiment_dir=str(self.exp_dir),
            dataroot=str(self.dataroot),
            manifest=str(self.manifest),
            seed=42,
            launcher=fail_launcher,
        )

        with self.assertRaises(RuntimeError) as ctx:
            pipeline_fail.run_stage("rgb")
        self.assertIn("failed with exit code 2", str(ctx.exception))
        self.assertEqual(pipeline_fail.stages["rgb"].status, "failed")

    def test_resume_dispatch(self):
        """build_resume_command must inject --continue_train and --resume_checkpoint."""
        pipeline = FreshTrainingPipeline(
            experiment_dir=str(self.exp_dir),
            dataroot=str(self.dataroot),
            manifest=str(self.manifest),
            seed=42,
        )
        fake_ckpt = str(Path(self.exp_dir) / "checkpoints" / "last.pth")
        cmd = pipeline.build_resume_command("rgb", fake_ckpt)
        self.assertIn("--continue_train", cmd)
        self.assertIn("--resume_checkpoint", cmd)
        idx = cmd.index("--resume_checkpoint")
        self.assertEqual(cmd[idx + 1], str(Path(fake_ckpt).resolve()))

    def test_shared_experts_among_heads_per_seed(self):
        """All fusion controls for a given seed must point to the identical expert checkpoint paths."""
        seed = 42
        p_token = FreshTrainingPipeline(
            experiment_dir=str(self.exp_dir),
            dataroot=str(self.dataroot),
            manifest=str(self.manifest),
            seed=seed,
            fusion_type="token_attention",
        )
        p_gated = FreshTrainingPipeline(
            experiment_dir=str(self.exp_dir),
            dataroot=str(self.dataroot),
            manifest=str(self.manifest),
            seed=seed,
            fusion_type="gated",
        )
        p_concat = FreshTrainingPipeline(
            experiment_dir=str(self.exp_dir),
            dataroot=str(self.dataroot),
            manifest=str(self.manifest),
            seed=seed,
            fusion_type="concat",
        )

        # Stage 1 and Stage 2 checkpoint paths must be completely identical across all 3 heads
        self.assertEqual(p_token.stages["rgb"].checkpoint_path, p_gated.stages["rgb"].checkpoint_path)
        self.assertEqual(p_token.stages["rgb"].checkpoint_path, p_concat.stages["rgb"].checkpoint_path)
        self.assertEqual(p_token.stages["wavelet"].checkpoint_path, p_gated.stages["wavelet"].checkpoint_path)
        self.assertEqual(p_token.stages["wavelet"].checkpoint_path, p_concat.stages["wavelet"].checkpoint_path)


if __name__ == "__main__":
    unittest.main()
