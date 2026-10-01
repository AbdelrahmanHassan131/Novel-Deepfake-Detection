# STATUS: NOT RUN
"""
Deferred regression test suite for Prompt N / Finding S10 & Round 3 Prompt Q / Finding T4.
Verifies resume protocol validation using real producer metadata schemas (config.protocol:checkpoint_metadata / _build_state),
manifest content change rejection, relocation acceptance, strict expert metadata enforcement,
and sample-weighted gradient accumulation mathematics.

DO NOT EXECUTE AT RUNTIME IN THIS ENVIRONMENT.
Static validation only.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
from pathlib import Path
import tempfile
import hashlib
import torch
import numpy as np

from training.checkpoint_manager import CheckpointManager
from config.protocol import checkpoint_metadata
from models.shared.expert_loading import validate_expert_checkpoint


class TestResumeProtocolEnforcement(unittest.TestCase):
    """Verifies that CheckpointManager._validate_protocol strictly validates real producer metadata schemas on resume."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmppath = Path(self.temp_dir.name)

        # Create real physical manifest files for producer-consumer integrity
        self.train_manifest = self.tmppath / "train_manifest.csv"
        self.val_manifest = self.tmppath / "val_manifest.csv"

        self.train_manifest.write_text("sample_id,path,label,split,dataset_source,group_id\ns1,img1.png,0,train,src,g1\n", encoding='utf-8')
        self.val_manifest.write_text("sample_id,path,label,split,dataset_source,group_id\ns2,img2.png,1,dev,src,g2\n", encoding='utf-8')

        self.mock_model = MagicMock()
        self.mock_model.name.return_value = 'Wang2020_128'
        self.mock_model.device = torch.device('cpu')
        self.mock_model.opt = SimpleNamespace(
            arch='Wang2020_128',
            embed_dim=128,
            fusion_type='token_attention',
            manifest=str(self.train_manifest),
            val_manifest=str(self.val_manifest),
            manifest_split='train',
            val_manifest_split='dev',
            seed=42,
            monitor_metric='auc',
            cropSize=224,
            loadSize=224,
            no_crop=False,
            no_resize=False,
            rz_interp='bilinear',
            wavelet_type='haar',
            wavelet_level=3,
            wavelet_mode='symmetric',
            use_log_packets=False,
            wavelet_log_mode='signed_log1p',
        )
        self.manager = CheckpointManager(str(self.tmppath / 'exp'), self.mock_model, rank=0)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _base_valid_checkpoint(self):
        """Construct checkpoint using the ACTUAL production schema from config.protocol:checkpoint_metadata."""
        prod_meta = checkpoint_metadata(self.mock_model)
        return {
            'model_name': 'Wang2020_128',
            'protocol': prod_meta,
        }

    def test_valid_real_protocol_passes(self):
        """Verify checkpoint produced by actual metadata writer passes validation when manifests are unchanged."""
        ckpt = self._base_valid_checkpoint()
        self.manager._validate_protocol(ckpt)

    def test_changed_train_manifest_contents_rejected(self):
        """Verify modifying actual training manifest file contents on disk causes resume rejection."""
        ckpt = self._base_valid_checkpoint()
        # Mutate the training manifest content on disk
        self.train_manifest.write_text("sample_id,path,label,split,dataset_source,group_id\ns1,img1_MODIFIED.png,0,train,src,g1\n", encoding='utf-8')

        with self.assertRaises(ValueError) as ctx:
            self.manager._validate_protocol(ckpt)
        self.assertIn("Resume dataset manifest mismatch", str(ctx.exception))

    def test_changed_val_manifest_contents_rejected(self):
        """Verify modifying development manifest file contents on disk causes resume rejection."""
        ckpt = self._base_valid_checkpoint()
        # Mutate the validation manifest content on disk
        self.val_manifest.write_text("sample_id,path,label,split,dataset_source,group_id\ns2,img2_MODIFIED.png,1,dev,src,g2\n", encoding='utf-8')

        with self.assertRaises(ValueError) as ctx:
            self.manager._validate_protocol(ckpt)
        self.assertIn("Resume validation manifest mismatch", str(ctx.exception))

    def test_relocated_manifest_path_accepted_when_digest_matches(self):
        """Verify that relocating manifest to a new path is accepted if content digest matches (preserving data identity)."""
        ckpt = self._base_valid_checkpoint()
        new_manifest_path = self.tmppath / "relocated_train_manifest.csv"
        new_manifest_path.write_text(self.train_manifest.read_text(encoding='utf-8'), encoding='utf-8')

        # Update opt to point to relocated file
        self.mock_model.opt.manifest = str(new_manifest_path)
        # Should pass without error because SHA256 digest is identical
        self.manager._validate_protocol(ckpt)

    def test_missing_manifest_rejected_fail_closed(self):
        """Verify that if manifest file is missing, resume fails closed with FileNotFoundError instead of skipping check."""
        ckpt = self._base_valid_checkpoint()
        self.train_manifest.unlink()

        with self.assertRaises(FileNotFoundError) as ctx:
            self.manager._validate_protocol(ckpt)
        self.assertIn("Cannot verify manifest identity", str(ctx.exception))

    def test_changed_split_rejected(self):
        ckpt = self._base_valid_checkpoint()
        ckpt['protocol']['options']['manifest_split'] = 'final_test'

        with self.assertRaises(ValueError) as ctx:
            self.manager._validate_protocol(ckpt)
        self.assertIn("Resume split mismatch for manifest_split", str(ctx.exception))

    def test_changed_seed_rejected(self):
        ckpt = self._base_valid_checkpoint()
        ckpt['protocol']['options']['seed'] = 43

        with self.assertRaises(ValueError) as ctx:
            self.manager._validate_protocol(ckpt)
        self.assertIn("Resume random seed mismatch", str(ctx.exception))

    def test_changed_monitor_metric_rejected(self):
        ckpt = self._base_valid_checkpoint()
        ckpt['protocol']['options']['monitor_metric'] = 'accuracy'

        with self.assertRaises(ValueError) as ctx:
            self.manager._validate_protocol(ckpt)
        self.assertIn("Resume monitor metric mismatch", str(ctx.exception))


class TestExpertMetadataEnforcement(unittest.TestCase):
    """Verifies that fresh expert validation rejects missing metadata unless allow_legacy=True."""

    def setUp(self):
        # Create dummy weights matching 128-D RGB expert
        self.dummy_rgb_weights = {
            'fc.0.weight': torch.randn(128, 2048),
            'fc.0.bias': torch.randn(128),
        }

    def test_fresh_expert_missing_protocol_raises(self):
        ckpt = {'model_state_dict': self.dummy_rgb_weights}
        with self.assertRaises(ValueError) as ctx:
            validate_expert_checkpoint(ckpt, expected_type='rgb', expected_embed_dim=128, allow_legacy=False)
        self.assertIn("Fresh training protocol requires verified metadata", str(ctx.exception))
        self.assertIn("missing 'protocol'", str(ctx.exception))

    def test_legacy_expert_without_protocol_allowed_with_flag(self):
        ckpt = {'model_state_dict': self.dummy_rgb_weights}
        validate_expert_checkpoint(ckpt, expected_type='rgb', expected_embed_dim=128, allow_legacy=True)


class TestSampleWeightedGradientAccumulationMath(unittest.TestCase):
    """Verifies that uneven microbatch sample weighting mathematically reproduces the true combined sample mean."""

    def test_uneven_microbatch_sample_weighting_matches_true_mean(self):
        np.random.seed(42)
        losses_batch1 = np.random.uniform(0.1, 1.5, size=32)
        losses_batch2 = np.random.uniform(0.1, 1.5, size=8)

        l1_mean = float(np.mean(losses_batch1))
        l2_mean = float(np.mean(losses_batch2))

        all_losses = np.concatenate([losses_batch1, losses_batch2])
        true_combined_sample_mean = float(np.mean(all_losses))

        # Old naive accumulation: equal weighting (loss / accum_steps)
        naive_accumulated_loss = (l1_mean / 2.0) + (l2_mean / 2.0)

        # Correct sample-proportional weighting: 32/40 and 8/40
        w1 = 32.0 / 40.0
        w2 = 8.0 / 40.0
        sample_weighted_accumulated_loss = (l1_mean * w1) + (l2_mean * w2)

        self.assertAlmostEqual(sample_weighted_accumulated_loss, true_combined_sample_mean, places=6)
        discrepancy = abs(naive_accumulated_loss - true_combined_sample_mean)
        self.assertGreater(discrepancy, 1e-4)


if __name__ == '__main__':
    unittest.main()
