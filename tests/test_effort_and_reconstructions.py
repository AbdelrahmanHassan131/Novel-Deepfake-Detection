"""Unit tests for EFFORT_128, SVDResidualLinear, forensic CLIP transforms, reconstruction generator, and LOGO split."""
import argparse
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from config import load_config, config_to_opt
from config.protocol import checkpoint_metadata
from data.drct_preparation import inventory_dataset_directory
from data.datasets.drct_dataset import DRCTDataset
from data.datasets.effort_dataset import EFFORTDataset
from data.transforms.clip_forensic_transforms import (
    CLIPForensicTrainTransform,
    CLIPForensicValTransform,
    CLIP_MEAN,
    CLIP_STD,
)
from evaluation.checkpoint_loader import CheckpointLoader
from models.effort.svd_linear import SVDResidualLinear, collect_orthogonality_loss
from models.effort.trainer import EFFORT_128Trainer
from models.registry import build_model
from models.shared.expert_loading import load_expert
from tools.build_logo_manifest import build_logo_manifest
from tools.generate_reconstructions import generate_reconstructions


def _make_effort_opt(svd_rank=4, multi_layer=True, drct_alpha=0.0):
    opt = argparse.Namespace()
    opt.arch = "EFFORT_128"
    opt.isTrain = True
    opt.continue_train = False
    opt.pretrained = False
    opt.gpu_ids = []
    opt.drct_backbone = "dummy"
    opt.embed_dim = 128
    opt.drct_alpha = drct_alpha
    opt.drct_pos_margin = 0.0
    opt.drct_neg_margin = 1.0
    opt.effort_svd_rank = svd_rank
    opt.effort_svd_blocks = 2
    opt.effort_multi_layer = multi_layer
    opt.effort_num_layers = 4
    opt.effort_layer_indices = None
    opt.effort_head_hidden = 64
    opt.effort_dropout = 0.2
    opt.effort_ortho_weight = 0.1
    opt.effort_label_smoothing = 0.05
    opt.lr = 1e-3
    opt.backbone_lr_mult = 0.2
    opt.weight_decay = 1e-2
    opt.optim = "adamw"
    opt.beta1 = 0.9
    opt.name = "test_effort"
    opt.checkpoints_dir = "./tmp_effort_ckpt"
    return opt


def _write_dummy_rgb(path: Path, color: tuple, size=(256, 256), fmt="JPEG"):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(sum(color) + len(path.name))
    arr = np.clip(
        np.full((size[1], size[0], 3), color, dtype=np.int16)
        + rng.randint(-15, 16, (size[1], size[0], 3)),
        0,
        255,
    ).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path, format=fmt)


class TestEffortAndReconstructions(unittest.TestCase):
    def test_svd_residual_linear_exact_init_and_orthogonality(self):
        """Verify W_main + U_res @ diag(S_res) @ V_res == W at init and only residual params train."""
        torch.manual_seed(0)
        lin = nn.Linear(32, 24, bias=True)
        x = torch.randn(4, 32)
        y_orig = lin(x)

        svd_lin = SVDResidualLinear.from_linear(lin, rank=4)
        y_svd = svd_lin(x)
        torch.testing.assert_close(y_orig, y_svd, rtol=1e-4, atol=1e-4)

        ortho0 = svd_lin.orthogonality_loss()
        self.assertLess(ortho0.item(), 1e-5)

        with torch.no_grad():
            svd_lin.U_res.add_(0.2)
        ortho1 = svd_lin.orthogonality_loss()
        self.assertGreater(ortho1.item(), 1e-3)

        loss = svd_lin(x).pow(2).mean() + ortho1
        loss.backward()
        self.assertIsNotNone(svd_lin.U_res.grad)
        self.assertIsNotNone(svd_lin.S_res.grad)
        self.assertIsNotNone(svd_lin.V_res.grad)
        self.assertFalse(svd_lin.W_main.requires_grad)
        self.assertFalse(svd_lin.bias.requires_grad)

        # Verify MultiheadAttention direct .weight property compatibility
        attn = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=False)
        attn.out_proj = SVDResidualLinear.from_linear(attn.out_proj, rank=2)
        seq = torch.randn(5, 2, 16)
        attn_out, _ = attn(seq, seq, seq)
        self.assertEqual(tuple(attn_out.shape), (5, 2, 16))
        attn_out.sum().backward()
        self.assertIsNotNone(attn.out_proj.U_res.grad)

    def test_effort_128_shapes_parity_and_gradient_isolation(self):
        """Verify [B, 1] logits, [B, 128] features, exact fc parity, and frozen semantic weights."""
        for batch_size in (1, 4):
            for svd_rank in (0, 4):
                with self.subTest(batch_size=batch_size, svd_rank=svd_rank):
                    opt = _make_effort_opt(
                        svd_rank=svd_rank,
                        multi_layer=True,
                        drct_alpha=0.3 if batch_size > 1 else 0.0,
                    )
                    trainer = build_model(opt)
                    self.assertIsInstance(trainer, EFFORT_128Trainer)
                    self.assertEqual(trainer.name(), "EFFORT_128")

                    trainer.eval()
                    x = torch.randn(batch_size, 3, 224, 224)
                    logits, feat = trainer.model(x, return_feature=True)
                    self.assertEqual(tuple(logits.shape), (batch_size, 1))
                    self.assertEqual(tuple(feat.shape), (batch_size, 128))
                    torch.testing.assert_close(logits, trainer.model.fc(feat))
                    torch.testing.assert_close(feat, trainer.forward_features(x))

                    imp = trainer.model.get_layer_importance()
                    self.assertEqual(len(imp), 4)
                    self.assertAlmostEqual(sum(imp.values()), 1.0, places=5)

                    trainer.train(True)
                    conv_before = trainer.model.encoder.conv1.weight.clone()
                    fc_before = trainer.model.fc.weight.clone()
                    labels = torch.tensor([0.0 if i % 2 == 0 else 1.0 for i in range(batch_size)])
                    trainer.set_input((x, labels))
                    loss = trainer.optimize_parameters()
                    self.assertTrue(torch.isfinite(loss).item())

                    torch.testing.assert_close(conv_before, trainer.model.encoder.conv1.weight)
                    self.assertFalse(torch.equal(fc_before, trainer.model.fc.weight))

                    if svd_rank > 0:
                        self.assertGreater(len(trainer.model.adapted_layers), 0)
                        self.assertGreaterEqual(collect_orthogonality_loss(trainer.model.encoder).item(), 0.0)
                    else:
                        self.assertEqual(len(trainer.model.adapted_layers), 0)

    def test_effort_checkpoint_roundtrip_and_mha_expert_loading(self):
        """Verify EFFORT_128 saves, reloads via CheckpointLoader, and plugs into load_expert for MHA_128."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            opt = _make_effort_opt(svd_rank=4, multi_layer=True)
            config = load_config(opt, validate=True, freeze=True)
            opt_clean = config_to_opt(config)
            trainer = build_model(opt_clean)
            trainer.eval()

            x = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                orig_logits, orig_feat = trainer.model(x, return_feature=True)

            ckpt_path = tmp_path / "effort_best.pth"
            torch.save({
                "model_name": "EFFORT_128",
                "epoch": 1,
                "model_state_dict": trainer.model.state_dict(),
                "protocol": checkpoint_metadata(trainer),
            }, ckpt_path)

            loader = CheckpointLoader(str(ckpt_path), device="cpu")
            restored = loader.load()
            with torch.no_grad():
                reloaded_logits, reloaded_feat = restored.model(x, return_feature=True)
            torch.testing.assert_close(orig_logits, reloaded_logits, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(orig_feat, reloaded_feat, rtol=1e-5, atol=1e-5)

            fusion_opt = argparse.Namespace(
                embed_dim=128,
                rgb_model_path=str(ckpt_path),
                drct_backbone="dummy",
                effort_preprocessing_version="clip_forensic_v1",
            )
            expert = load_expert(fusion_opt, "rgb", torch.device("cpu"))
            with torch.no_grad():
                expert_feat = expert.forward_features(x)
            torch.testing.assert_close(orig_feat, expert_feat, rtol=1e-5, atol=1e-5)

    def test_clip_forensic_transforms_and_drct_forensic_mode(self):
        """Verify native CLIP norm, symmetric degradations, and opt-in DRCT forensic transform mode."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            img = Image.fromarray(np.random.randint(0, 255, (280, 260, 3), dtype=np.uint8), mode="RGB")
            train_tf = CLIPForensicTrainTransform(
                size=224, jpg_prob=1.0, blur_prob=1.0, downscale_prob=1.0, color_jitter_prob=1.0
            )
            val_tf = CLIPForensicValTransform(size=224)

            t_train = train_tf(img)
            t_val1 = val_tf(img)
            t_val2 = val_tf(img)

            self.assertEqual(tuple(t_train.shape), (3, 224, 224))
            self.assertEqual(tuple(t_val1.shape), (3, 224, 224))
            torch.testing.assert_close(t_val1, t_val2)
            self.assertEqual(val_tf.mean, CLIP_MEAN)
            self.assertEqual(val_tf.std, CLIP_STD)

            for cls_name, color in [("real", (100, 120, 140)), ("fake", (160, 90, 80))]:
                _write_dummy_rgb(tmp_path / cls_name / "img1.jpg", color)

            drct_opt = argparse.Namespace(
                arch="DRCT_128", drct_backbone="dummy", cropSize=224, isTrain=True,
                drct_aug_mode="forensic", no_flip=False, manifest=None, classes=["real", "fake"], mode="binary",
            )
            ds_drct = DRCTDataset(drct_opt, str(tmp_path))
            sample_x, sample_y = ds_drct[0]
            self.assertEqual(tuple(sample_x.shape), (3, 224, 224))
            self.assertIn(sample_y, (0, 1))

            eff_opt = argparse.Namespace(
                arch="EFFORT_128", drct_backbone="dummy", cropSize=224, isTrain=True,
                no_flip=False, manifest=None, classes=["real", "fake"], mode="binary",
            )
            ds_eff = EFFORTDataset(eff_opt, str(tmp_path))
            ex, ey = ds_eff[0]
            self.assertEqual(tuple(ex.shape), (3, 224, 224))
            self.assertIn(ey, (0, 1))

    def test_paired_reconstruction_pipeline_end_to_end(self):
        """Test base inventory -> generate_reconstructions (dummy) -> DRCT import -> full readiness."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ds_root = tmp_path / "dataset"
            for split in ("train", "val"):
                for i in range(3):
                    _write_dummy_rgb(ds_root / split / "real" / f"r_{split}_{i}.jpg", (80 + i * 10, 100, 120))
                    _write_dummy_rgb(ds_root / split / "fake" / f"f_{split}_{i}.jpg", (180 - i * 10, 90, 70))

            base_csv = tmp_path / "base_inventory.csv"
            inventory_dataset_directory(ds_root, base_csv, duplicate_policy="error")

            rec_dir = tmp_path / "reconstructions"
            mapping_csv = tmp_path / "reconstruction_mapping.csv"
            drct_manifest = tmp_path / "drct_ready_manifest.csv"

            summary = generate_reconstructions(
                inventory_path=base_csv,
                output_dir=rec_dir,
                mapping_csv=mapping_csv,
                backend="dummy",
                target_split="train",
                reconstruct_fakes=True,
                batch_size=2,
                seed=42,
                import_manifest=drct_manifest,
            )
            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["generated"], 6)
            self.assertTrue(summary["readiness"]["is_ready_for_full_drct"])
            self.assertTrue(drct_manifest.with_suffix(".verified.json").is_file())

            # Resumability check
            summary2 = generate_reconstructions(
                inventory_path=base_csv,
                output_dir=rec_dir,
                mapping_csv=mapping_csv,
                backend="dummy",
                target_split="train",
                reconstruct_fakes=True,
                batch_size=2,
                seed=42,
            )
            self.assertEqual(summary2["resumed"], 6)
            self.assertEqual(summary2["generated"], 0)

    def test_logo_manifest_builder_strict_generator_isolation(self):
        """Verify Leave-One-Generator-Out builder strictly isolates held-out dev & test generators."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            real_dir = tmp_path / "real"
            fake_dir = tmp_path / "fake"
            for i in range(12):
                _write_dummy_rgb(real_dir / "CelebA" / f"real_{i:03d}.jpg", (50 + i * 5, 100, 150))
            for gen, base_c in [("ADM", 20), ("DDPM", 60), ("DDIM", 120), ("LDM", 180)]:
                for i in range(4):
                    _write_dummy_rgb(fake_dir / gen / f"{gen}_{i:03d}.png", (base_c, 80 + i * 10, 200 - i * 10), fmt="PNG")

            out_csv = tmp_path / "logo_manifest.csv"
            report = build_logo_manifest(
                real_dir=real_dir,
                fake_dir=fake_dir,
                output_manifest=out_csv,
                dev_generators=["DDIM"],
                test_generators=["LDM"],
                seed=42,
            )
            self.assertEqual(report["status"], "success")
            self.assertTrue(out_csv.with_suffix(".verified.json").is_file())
            self.assertTrue(any("FORMAT SHORTCUT RISK" in w for w in report["shortcut_audit"]["warnings"]))

            with open(out_csv, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            train_gens = {r["generator"] for r in rows if r["split"] == "train" and int(r["label"]) == 1}
            dev_gens = {r["generator"] for r in rows if r["split"] == "dev" and int(r["label"]) == 1}
            test_gens = {r["generator"] for r in rows if r["split"] == "external_test" and int(r["label"]) == 1}

            self.assertEqual(train_gens, {"ADM", "DDPM"})
            self.assertEqual(dev_gens, {"DDIM"})
            self.assertEqual(test_gens, {"LDM"})
            self.assertTrue(train_gens.isdisjoint(dev_gens))
            self.assertTrue(train_gens.isdisjoint(test_gens))
            self.assertTrue(dev_gens.isdisjoint(test_gens))


if __name__ == "__main__":
    unittest.main()
