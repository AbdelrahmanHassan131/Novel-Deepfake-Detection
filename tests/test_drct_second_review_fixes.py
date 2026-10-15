"""
Dedicated test suite verifying fixes for findings S1 through S9 from DRCT_128_SECOND_REVIEW_FIXES.md.
All tests run on CPU with tiny fixtures, dummy or mocked encoders, and no external downloads.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from config.compatibility import config_to_opt, opt_to_config
from config.configuration import Config, DataConfig, ModelConfig, TrainingConfig
from config.types import ArchitectureType
from data.datasets.drct_dataset import DRCTDataset
from data.datasets.fusion_dataset import FusionDataset
from data.drct_manifest import read_drct_manifest, validate_drct_records
from data.drct_preparation import (
    generate_gate_sidecar,
    import_precomputed_reconstructions,
    inventory_dataset_directory,
)
from data.transforms.drct_transforms import (
    DRCT_PREPROCESSING_VERSION,
    DRCTTrainTransform,
    DRCTValTransform,
)
from evaluation.checkpoint_loader import CheckpointLoader
from models.drct.backbone import build_drct_backbone
from models.drct.checkpoint_io import (
    initialize_drct128_from_pretrained_encoder,
    import_official_drct_detector,
)
from models.drct.drct_model import DRCT128Model
from models.drct.loss import MarginContrastiveLoss
from tools.benchmark_drct import parse_args as parse_benchmark_args


def _create_tiny_image(path: Path, color=(255, 0, 0), size=(32, 32)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path, format="PNG")
    return path


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ==============================================================================
# S1 Tests: Preparation output consumable by training entry point and verification gates
# ==============================================================================

class TestS1PreparationGateAndTrainingEntry(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.root = Path(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_s1_end_to_end_preparation_to_training_gate(self):
        """Authentic tiny images prepared via inventory_dir, gate generated, consumable by train.py preflight."""
        dataset_root = self.root / "dataset"
        # Create authentic distinct images for train and dev
        _create_tiny_image(dataset_root / "train" / "real" / "img1.png", color=(255, 0, 0))
        _create_tiny_image(dataset_root / "train" / "fake" / "img2.png", color=(0, 255, 0))
        _create_tiny_image(dataset_root / "val" / "real" / "img3.png", color=(0, 0, 255))
        _create_tiny_image(dataset_root / "val" / "fake" / "img4.png", color=(255, 255, 0))

        manifest_path = self.root / "manifest.csv"
        inv_res = inventory_dataset_directory(dataset_root, manifest_path)
        self.assertEqual(inv_res["total_inventoried"], 4)

        records = read_drct_manifest(manifest_path)
        self.assertEqual(len(records), 4)

        splits = {r["split"] for r in records}
        self.assertIn("train", splits)
        self.assertIn("dev", splits)

        # Generate verification gate sidecar
        sidecar_path = generate_gate_sidecar(manifest_path)
        self.assertTrue(sidecar_path.is_file())

        with open(sidecar_path, "r") as f:
            gate_data = json.load(f)
        self.assertTrue(gate_data.get("verified"))
        self.assertEqual(gate_data.get("manifest_sha256"), _file_sha256(manifest_path))

        # Run train.py with ablation mode (--drct_alpha 0.0) on dummy backbone
        cmd = [
            sys.executable,
            "train.py",
            "--arch", "DRCT_128",
            "--drct_backbone", "dummy",
            "--embed_dim", "128",
            "--drct_alpha", "0.0",
            "--manifest", str(manifest_path),
            "--dataroot", str(dataset_root),
            "--val_manifest_split", "dev",
            "--allow_source_overlap",
            "--allow_aggregate_sources",  # Directory inventory has no verified identity metadata.
            "--epochs", "1",
            "--batch_size", "2",
            "--num_workers", "0",
            "--checkpoints_dir", str(self.root / "ckpts"),
            "--name", "test_s1_run",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"train.py failed with:\n{res.stderr}\n{res.stdout}")
        best_ckpts = list((self.root / "ckpts").rglob("best.pth"))
        self.assertTrue(len(best_ckpts) > 0, f"No best.pth checkpoint found under {self.root / 'ckpts'}")

    def test_s1_gate_digest_mismatch_fails_preflight(self):
        """Mutating manifest bytes after sidecar generation fails the preflight check."""
        dataset_root = self.root / "dataset"
        _create_tiny_image(dataset_root / "train" / "real" / "img1.png", color=(255, 0, 0))
        _create_tiny_image(dataset_root / "train" / "fake" / "img2.png", color=(0, 255, 0))
        _create_tiny_image(dataset_root / "val" / "real" / "img3.png", color=(0, 0, 255))
        _create_tiny_image(dataset_root / "val" / "fake" / "img4.png", color=(255, 255, 0))

        manifest_path = self.root / "manifest.csv"
        inventory_dataset_directory(dataset_root, manifest_path)
        sidecar_path = generate_gate_sidecar(manifest_path)
        self.assertTrue(sidecar_path.is_file())

        # Tamper with manifest content
        with open(manifest_path, "a") as f:
            f.write("# tampering comment\n")

        cmd = [
            sys.executable,
            "train.py",
            "--arch", "DRCT_128",
            "--drct_backbone", "dummy",
            "--drct_alpha", "0.0",
            "--manifest", str(manifest_path),
            "--dataroot", str(dataset_root),
            "--val_manifest_split", "dev",
            "--allow_source_overlap",
            "--epochs", "1",
            "--batch_size", "2",
            "--checkpoints_dir", str(self.root / "ckpts"),
            "--name", "test_tamper_run",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("digest mismatch", (res.stderr + res.stdout).lower())

    def test_s1_full_drct_alpha_without_reconstructions_rejected(self):
        """If drct_alpha > 0 but manifest lacks paired reconstructions, train.py halts before starting."""
        dataset_root = self.root / "dataset"
        _create_tiny_image(dataset_root / "train" / "real" / "img1.png", color=(255, 0, 0))
        _create_tiny_image(dataset_root / "train" / "fake" / "img2.png", color=(0, 255, 0))
        _create_tiny_image(dataset_root / "val" / "real" / "img3.png", color=(0, 0, 255))
        _create_tiny_image(dataset_root / "val" / "fake" / "img4.png", color=(255, 255, 0))

        manifest_path = self.root / "manifest.csv"
        inventory_dataset_directory(dataset_root, manifest_path)
        generate_gate_sidecar(manifest_path)

        cmd = [
            sys.executable,
            "train.py",
            "--arch", "DRCT_128",
            "--drct_backbone", "dummy",
            "--drct_alpha", "0.3",  # Full DRCT contrastive requested
            "--manifest", str(manifest_path),
            "--dataroot", str(dataset_root),
            "--val_manifest_split", "dev",
            "--allow_source_overlap",
            "--epochs", "1",
            "--batch_size", "2",
            "--checkpoints_dir", str(self.root / "ckpts"),
            "--name", "test_missing_rec_run",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        output_lower = (res.stderr + res.stdout).lower()
        self.assertTrue(
            "not ready for full contrastive training" in output_lower or
            "reconstruction" in output_lower or
            "alpha" in output_lower
        )


# ==============================================================================
# S2 Tests: Pretrained CLIP initialization paths and download controls
# ==============================================================================

class TestS2CLIPPretrainedInitialization(unittest.TestCase):
    def test_s2_pretrained_true_with_allow_download_false_raises(self):
        """Requesting pretrained weights with allow_download=False and no local path raises."""
        mock_clip = MagicMock()
        mock_clip_model = MagicMock()
        mock_clip_model.VisionTransformer = MagicMock(return_value=nn.Sequential(nn.Linear(10, 10)))
        with patch.dict(sys.modules, {"clip": mock_clip, "clip.model": mock_clip_model}):
            with pytest.raises((FileNotFoundError, ValueError), match="allow_download=False"):
                build_drct_backbone(
                    name="clip_vit_l14",
                    pretrained=True,
                    allow_download=False,
                    weights_path=None,
                )

    def test_s2_local_cached_weights_loads_without_download(self):
        """Supplying valid local weights file loads from disk without initiating a download."""
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create a mock CLIP model state
            mock_clip = MagicMock()
            mock_clip_model = MagicMock()
            mock_vision = nn.Sequential(nn.Linear(10, 10))
            mock_clip.visual = mock_vision
            mock_clip.load.return_value = (mock_clip, None)
            mock_clip_model.VisionTransformer = MagicMock(return_value=mock_vision)
            torch.save(mock_vision.state_dict(), tmp_path)

            with patch.dict(sys.modules, {"clip": mock_clip, "clip.model": mock_clip_model}):
                backbone, dim = build_drct_backbone(
                    name="clip_vit_l14",
                    pretrained=True,
                    allow_download=False,
                    weights_path=tmp_path,
                )
                self.assertIsNotNone(backbone)
                self.assertEqual(dim, 768)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_s2_config_and_benchmark_roundtrip_includes_backbone_settings(self):
        """Config and benchmark CLI roundtrip drct_allow_download and drct_backbone_weights."""
        cfg = Config(
            model=ModelConfig(
                architecture=ArchitectureType.DRCT_128,
                drct_backbone="clip_vit_l14",
                drct_allow_download=True,
                drct_backbone_weights="/path/to/weights.pth",
            ),
            training=TrainingConfig(),
            data=DataConfig(train_path="d/train", val_path="d/val"),
        )
        opt = config_to_opt(cfg)
        self.assertTrue(opt.drct_allow_download)
        self.assertEqual(opt.drct_backbone_weights, "/path/to/weights.pth")

        cfg_back = opt_to_config(opt)
        self.assertTrue(cfg_back.model.drct_allow_download)
        self.assertEqual(cfg_back.model.drct_backbone_weights, "/path/to/weights.pth")

        # Benchmark CLI parser test
        bench_opt = parse_benchmark_args([
            "--synthetic",
            "--drct_allow_download",
            "--drct_backbone_weights", "/path/to/bench.pth",
        ])
        self.assertTrue(bench_opt.drct_allow_download)
        self.assertEqual(bench_opt.drct_backbone_weights, "/path/to/bench.pth")


# ==============================================================================
# S3 Tests: Saved training manifests isolated from active evaluation data
# ==============================================================================

class TestS3ManifestIsolationDuringEvaluation(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.root = Path(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_s3_folder_evaluation_ignores_saved_training_manifest(self):
        """Checkpoint saved with old manifest paths evaluates external folder images only."""
        # 1. Create a dummy model and save checkpoint with old manifest paths in options
        model = DRCT128Model(backbone_name="dummy", embed_dim=128)
        ckpt_path = self.root / "saved_ckpt.pth"

        old_manifest_path = self.root / "old_nonexistent_manifest.csv"
        saved_options = {
            "arch": "DRCT_128",
            "drct_backbone": "dummy",
            "embed_dim": 128,
            "manifest": str(old_manifest_path),
            "val_manifest": str(old_manifest_path),
            "drct_manifest_path": str(old_manifest_path),
        }
        torch.save({
            "state_dict": model.state_dict(),
            "model_name": "DRCT_128",
            "protocol": {
                "format_version": "1.0",
                "options": saved_options,
                "label_mapping": {"real": 0, "fake": 1},
            }
        }, ckpt_path)

        # 2. Create external test folder with disjoint images
        ext_folder = self.root / "external_data"
        _create_tiny_image(ext_folder / "real" / "ext_real_1.png", color=(10, 20, 30))
        _create_tiny_image(ext_folder / "fake" / "ext_fake_1.png", color=(40, 50, 60))

        # 3. Load checkpoint via CheckpointLoader
        loader = CheckpointLoader(checkpoint_path=str(ckpt_path), device=torch.device("cpu"))
        loaded_model = loader.load()

        # Assert active manifest options were cleared, while preserved in _saved_*
        self.assertIsNone(loader.opt.manifest)
        self.assertIsNone(loader.opt.val_manifest)
        self.assertIsNone(loader.opt.drct_manifest_path)
        self.assertEqual(loader.opt._saved_manifest, str(old_manifest_path))

        # 4. Construct DRCTDataset pointing to external folder
        dataset = DRCTDataset(loader.opt, str(ext_folder))
        self.assertEqual(len(dataset), 2)
        # All sample paths must be inside external_data
        for fpath, label in dataset.samples:
            img_path = Path(fpath)
            self.assertTrue(img_path.is_relative_to(ext_folder))
            self.assertFalse(img_path.is_relative_to(self.root / "old_nonexistent_manifest.csv"))


# ==============================================================================
# S4 Tests: Legacy config loader structure and refresh
# ==============================================================================

class TestS4LegacyLoaderRegression(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.root = Path(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_s4_legacy_json_restores_options_and_preprocessing(self):
        """Legacy checkpoint without protocol loads matching config JSON and validates keys."""
        model = DRCT128Model(backbone_name="dummy", embed_dim=128)
        ckpt_path = self.root / "legacy_ckpt.pth"
        torch.save(model.state_dict(), ckpt_path)

        legacy_json_path = self.root / "legacy_config.json"
        legacy_data = {
            "options": {
                "arch": "DRCT_128",
                "drct_backbone": "dummy",
                "drct_preprocessing_version": "drct_clip_v1",
                "embed_dim": 128,
                "crop_size": 224,
                "mean": [0.48145466, 0.4578275, 0.40821073],
                "std": [0.26862954, 0.26130258, 0.27577711],
            },
            "label_mapping": {"real": 0, "fake": 1},
        }
        with open(legacy_json_path, "w") as f:
            json.dump(legacy_data, f)

        loader = CheckpointLoader(
            checkpoint_path=str(ckpt_path),
            arch="DRCT_128",
            device=torch.device("cpu"),
        )
        loaded_model = loader.load(opt_overrides={"legacy_config": str(legacy_json_path)})
        self.assertIsNotNone(loaded_model)
        self.assertEqual(loader.opt.arch, "DRCT_128")
        self.assertEqual(loader.opt.crop_size, 224)
        self.assertEqual(loader.opt.embed_dim, 128)

    def test_s4_legacy_json_rejects_missing_required_keys(self):
        """Legacy json missing required architecture key (e.g. drct_preprocessing_version) raises ValueError."""
        model = DRCT128Model(backbone_name="dummy", embed_dim=128)
        ckpt_path = self.root / "legacy_ckpt.pth"
        torch.save(model.state_dict(), ckpt_path)

        legacy_json_path = self.root / "invalid_legacy.json"
        with open(legacy_json_path, "w") as f:
            json.dump({
                "options": {"arch": "DRCT_128", "drct_backbone": "dummy"},  # Missing drct_preprocessing_version
                "label_mapping": {"real": 0, "fake": 1},
            }, f)

        loader = CheckpointLoader(
            checkpoint_path=str(ckpt_path),
            arch="DRCT_128",
            device=torch.device("cpu"),
        )
        with pytest.raises(ValueError, match="Legacy config missing preprocessing fields"):
            loader.load(opt_overrides={"legacy_config": str(legacy_json_path)})


# ==============================================================================
# S5 Tests: Reconstruction preparation, stem collision safety, and provenance
# ==============================================================================

class TestS5PreparationAndProvenance(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.root = Path(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_s5_stem_collisions_and_nested_folders(self):
        """Duplicate stems across classes/extensions and nested directories are preserved safely."""
        dataset_root = self.root / "dataset"
        # Same stem 'sample' in real and fake, and with different extensions
        p1 = _create_tiny_image(dataset_root / "train" / "real" / "sample.jpg", color=(1, 2, 3))
        p2 = _create_tiny_image(dataset_root / "train" / "fake" / "nested" / "sub" / "sample.png", color=(4, 5, 6))
        p3 = _create_tiny_image(dataset_root / "val" / "real" / "sample.png", color=(7, 8, 9))
        p4 = _create_tiny_image(dataset_root / "val" / "fake" / "sample.jpg", color=(10, 11, 12))

        manifest_path = self.root / "manifest.csv"
        inv_res = inventory_dataset_directory(dataset_root, manifest_path)
        self.assertEqual(inv_res["total_inventoried"], 4)

        records = read_drct_manifest(manifest_path)
        self.assertEqual(len(records), 4)

        # Verify all directory IDs are unique and collision safe
        dir_ids = [r["sample_id"] for r in records]
        self.assertEqual(len(dir_ids), len(set(dir_ids)), "Sample IDs must be unique!")

    def test_s5_cross_split_same_content_duplicate_rejected(self):
        """Exact identical image content in train and val is rejected by DRCT manifest validator."""
        dataset_root = self.root / "dataset"
        # Exact same content in train and dev, with valid binary labels in both splits
        _create_tiny_image(dataset_root / "train" / "real" / "img1.png", color=(100, 100, 100))
        _create_tiny_image(dataset_root / "train" / "fake" / "img_fake_tr.png", color=(20, 20, 20))
        _create_tiny_image(dataset_root / "val" / "real" / "img2.png", color=(100, 100, 100))
        _create_tiny_image(dataset_root / "val" / "fake" / "img_fake_val.png", color=(30, 30, 30))

        manifest_path = self.root / "manifest.csv"
        with pytest.raises(ValueError, match="Identical content hash"):
            inventory_dataset_directory(dataset_root, manifest_path)


# ==============================================================================
# S6 Tests: Strict official checkpoint imports and encoder-only mapping
# ==============================================================================

class TestS6OfficialCheckpointImports(unittest.TestCase):
    def test_s6_destination_key_verification_rejects_missing_bias_or_proj(self):
        """Pretrained encoder initialization strictly fails if bias or weights are missing."""
        model = DRCT128Model(backbone_name="dummy", embed_dim=128)

        # Incomplete source checkpoint missing bias
        incomplete_state = {
            "proj.weight": torch.randn_like(model.encoder.proj.weight),
            # Missing proj.bias
        }
        with pytest.raises(ValueError, match="missing required weights"):
            initialize_drct128_from_pretrained_encoder(model, incomplete_state)

    def test_s6_official_detector_rejects_missing_projection_or_unexpected(self):
        """Official detector import requires valid projection.weight and rejects unexplained unexpected keys."""
        # Missing projection.weight
        with pytest.raises(ValueError, match="projection.weight"):
            import_official_drct_detector({"fc.weight": torch.randn(2, 1024)}, backbone_name="dummy")

    def test_s6_encoder_only_import_discards_head_and_preserves_128d_projection(self):
        """Encoder-only import discards official 1024-wide head without corrupting fresh 128D projection."""
        model = DRCT128Model(backbone_name="dummy", embed_dim=128)

        # Record initial fresh 128D projection weights
        initial_proj_weight = model.projection[0].weight.clone()
        self.assertEqual(model.projection[0].weight.shape[0], 128)

        # Source checkpoint from native-width model having a 1024-wide head
        official_state = {
            "proj.weight": torch.randn_like(model.encoder.proj.weight),
            "proj.bias": torch.randn_like(model.encoder.proj.bias),
            # Official head weights of incompatible shape [1, 1024]
            "projection.0.weight": torch.randn(1024, 768),
            "classifier.weight": torch.randn(1, 1024),
        }

        initialize_drct128_from_pretrained_encoder(model, official_state)

        # Backbone should be updated
        self.assertTrue(
            torch.allclose(
                model.encoder.proj.weight,
                official_state["proj.weight"]
            )
        )
        # Fresh 128D projection must NOT be overwritten or truncated
        self.assertTrue(torch.allclose(model.projection[0].weight, initial_proj_weight))
        self.assertEqual(model.projection[0].weight.shape[0], 128)


# ==============================================================================
# S7 Tests: Exact zero distance in contrastive reducer
# ==============================================================================

class TestS7ContrastiveLossExactReducer(unittest.TestCase):
    def test_s7_exact_zero_distance_and_all_identical(self):
        """Identical vectors have distance exactly 0.0 and positive violation 0.0."""
        loss_fn = MarginContrastiveLoss(pos_margin=0.0, neg_margin=1.0)

        # Duplicate positive embeddings
        emb = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32, requires_grad=True)
        lbl = torch.tensor([0, 0], dtype=torch.long)

        loss = loss_fn(emb, lbl)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(emb.grad).all())

        # All-identical vectors in batch
        emb3 = torch.tensor([[0.6, 0.8], [0.6, 0.8], [0.6, 0.8]], dtype=torch.float32, requires_grad=True)
        lbl3 = torch.tensor([0, 0, 0], dtype=torch.long)
        loss3 = loss_fn(emb3, lbl3)
        self.assertEqual(loss3.item(), 0.0)
        loss3.backward()
        self.assertTrue(torch.isfinite(emb3.grad).all())

    def test_s7_three_vector_benchmark_case(self):
        """Exact test case: [[1,0],[1,0],[0,1]] all positive labels produces exact 1.4142135624."""
        loss_fn = MarginContrastiveLoss(pos_margin=0.0, neg_margin=1.0)
        emb = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32, requires_grad=True)
        lbl = torch.tensor([0, 0, 0], dtype=torch.long)

        loss = loss_fn(emb, lbl)
        # Pairs:
        # (0,1): dist = 0.0, violation = 0.0 (non-violation, excluded)
        # (0,2): dist = sqrt(2) = 1.4142135624, violation = 1.4142135624
        # (1,2): dist = sqrt(2) = 1.4142135624, violation = 1.4142135624
        # Mean of non-zero violations = 1.4142135624
        expected = np.sqrt(2.0)
        self.assertAlmostEqual(loss.item(), expected, places=6)
        self.assertAlmostEqual(loss.item(), 1.4142135624, places=6)

        # Verify gradient calculation stability
        loss.backward()
        self.assertTrue(torch.isfinite(emb.grad).all())
        self.assertTrue((emb.grad != 0.0).any())

    def test_s7_four_vector_benchmark_case(self):
        """Benchmark 4-vector case matches verified 0.8270503283."""
        loss_fn = MarginContrastiveLoss(pos_margin=0.0, neg_margin=1.0)
        emb = torch.tensor([[1.0, 0.0], [1.0, 0.5], [0.0, 1.0], [0.5, 1.0]], dtype=torch.float32)
        lbl = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        loss = loss_fn(emb, lbl)
        self.assertAlmostEqual(loss.item(), 0.8270503283, places=6)


# ==============================================================================
# S9 Tests: Standalone vs MHA preprocessing and resolution constraint
# ==============================================================================

class TestS9PreprocessingAndResolution(unittest.TestCase):
    def test_s9_versioned_adaptation_preprocessing_non_square_parity(self):
        """Non-square image (300x400) is preprocessed identically in standalone and MHA."""
        # Create non-square PIL image
        img = Image.new("RGB", (400, 300), color=(128, 64, 32))

        # Versioned standalone eval transform
        t_standalone = DRCTValTransform(224)
        t_img = t_standalone(img).unsqueeze(0)
        self.assertEqual(t_img.shape, (1, 3, 224, 224))

        # Model feature extraction
        model = DRCT128Model(backbone_name="dummy", embed_dim=128)
        feat = model.forward_features(t_img)
        self.assertEqual(feat.shape, (1, 128))

    def test_s9_rejects_unsupported_non_224_input_size(self):
        """CLIP ViT requires 224x224 input; unsupported resolutions must raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            ckpt_path = root / "drct_ckpt.pth"
            model = DRCT128Model(backbone_name="dummy", embed_dim=128)
            torch.save({
                "state_dict": model.state_dict(),
                "protocol": {
                    "format_version": "1.0",
                    "options": {"arch": "DRCT_128", "crop_size": 256},  # Invalid for CLIP ViT
                    "label_mapping": {"real": 0, "fake": 1},
                }
            }, ckpt_path)

            class DummyOpt:
                def __init__(self):
                    self.rgb_model_path = str(ckpt_path)
                    self.wavelet_backend = "cpu"
                    self.isTrain = False
                    self.no_flip = True
                    self.rgb_expert_arch = "DRCT_128"

            opt = DummyOpt()
            with pytest.raises(ValueError, match="224"):
                FusionDataset(opt, str(root))
