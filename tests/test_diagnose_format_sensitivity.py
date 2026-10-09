"""Unit tests for bounded paired-preprocessing format sensitivity diagnostic.

Tests:
1. PNG RGB equality control (lossless roundtrip gives exact pixel and tensor match).
2. Paired JPEG interventions introduce nonzero distortion.
3. Original disk images strictly unmodified (SHA256 verified before and after).
4. Manifest contract enforcement:
   - Rejection of >128 samples
   - Rejection of empty manifest
   - Rejection of invalid labels (must be 0 or 1)
   - Rejection of duplicate sample IDs
   - Rejection of missing image files
   - Rejection of hash mismatches
5. End-to-end execution with deterministic dummy model:
   - Output directory creation and writability
   - Exactly 4 condition records per sample ID
   - Output CSV and JSON report validity
6. Non-default preprocessing handling (custom interpolation, no_crop, no_resize).
7. Canonical score sign handling (score_sign = -1.0 reverses logits).
8. Production transform tensor parity.
9. Workers restriction enforcement (workers > 0 rejected).
"""

import csv
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

from tools.diagnose_format_sensitivity import (
    apply_interventions,
    build_production_eval_transform,
    load_and_validate_manifest,
    run_format_diagnostic,
    MAX_SAMPLES_CEILING,
    compute_file_sha256,
)


class DummyMockModel(nn.Module):
    """Deterministic mock model mimicking Wang2020 interface."""
    def __init__(self, score_sign=1.0, crop_size=224, load_size=256, rz_interp=None, no_crop=False, no_resize=False):
        super().__init__()
        self.arch = "DummyMockModel"
        self.score_sign = score_sign
        self.opt = SimpleNamespace(
            cropSize=crop_size,
            loadSize=load_size,
            rz_interp=rz_interp or ["bilinear"],
            no_crop=no_crop,
            no_resize=no_resize,
            isTrain=False,
            no_flip=True,
            data_aug=False,
        )
        torch.manual_seed(42)
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        h = self.pool(torch.relu(self.conv(x)))
        out = self.fc(h.view(x.size(0), -1))
        return out


class TestFormatSensitivityDiagnostic(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Create seeded synthetic images
        np.random.seed(42)
        self.img_png = self.root / "synth_0.png"
        self.img_jpg = self.root / "synth_1.jpg"

        arr1 = np.random.randint(0, 256, (120, 140, 3), dtype=np.uint8)
        arr2 = np.random.randint(0, 256, (150, 130, 3), dtype=np.uint8)

        Image.fromarray(arr1).save(self.img_png, format="PNG")
        Image.fromarray(arr2).save(self.img_jpg, format="JPEG", quality=95)

        self.sha_png = compute_file_sha256(self.img_png)
        self.sha_jpg = compute_file_sha256(self.img_jpg)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_png_rgb_equality_control(self):
        """Verify PNG round-trip produces exact byte-for-byte RGB equality."""
        with Image.open(self.img_png) as img:
            interventions = apply_interventions(img)
            orig_img, orig_diff = interventions["original"]
            png_img, png_diff = interventions["png_control"]
            self.assertEqual(png_diff, 0.0, "PNG round-trip must not alter decoded RGB pixels")
            orig_arr = np.array(orig_img)
            png_arr = np.array(png_img)
            self.assertTrue(np.array_equal(orig_arr, png_arr), "Pixel arrays must match identically")

    def test_02_intervention_ordering_and_distortion(self):
        """Verify JPEG interventions introduce nonzero pixel differences."""
        with Image.open(self.img_png) as img:
            interventions = apply_interventions(img)
            jpg95_img, jpg95_diff = interventions["jpeg95"]
            jpg75_img, jpg75_diff = interventions["jpeg75"]
            self.assertGreater(jpg95_diff, 0.0, "JPEG 95 must produce nonzero pixel difference")
            self.assertGreater(jpg75_diff, 0.0, "JPEG 75 must produce nonzero pixel difference")

    def test_03_original_files_on_disk_unmodified(self):
        """Verify original files on disk are strictly preserved and never mutated."""
        for path in (self.img_png, self.img_jpg):
            with Image.open(path) as img:
                _ = apply_interventions(img)

        self.assertEqual(compute_file_sha256(self.img_png), self.sha_png, "PNG file on disk was modified")
        self.assertEqual(compute_file_sha256(self.img_jpg), self.sha_jpg, "JPEG file on disk was modified")

    def test_04_manifest_contract_rejections(self):
        """Verify manifest contract rejections: bounds, empty, invalid labels, duplicate IDs, missing files."""
        # A. Bounds check: > 128 rows
        p_oversized = self.root / "oversized.csv"
        with open(p_oversized, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "path", "label", "split"])
            writer.writeheader()
            for i in range(MAX_SAMPLES_CEILING + 1):
                writer.writerow({"sample_id": f"s_{i}", "path": str(self.img_png), "label": "0", "split": "dev"})
        with self.assertRaises(ValueError) as ctx:
            load_and_validate_manifest(str(p_oversized))
        self.assertIn("exceeds bounded maximum", str(ctx.exception))

        # B. Empty manifest
        p_empty = self.root / "empty.csv"
        with open(p_empty, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "path", "label"])
            writer.writeheader()
        with self.assertRaises(ValueError) as ctx:
            load_and_validate_manifest(str(p_empty))
        self.assertIn("contains 0 rows", str(ctx.exception))

        # C. Invalid label
        p_bad_label = self.root / "bad_label.csv"
        with open(p_bad_label, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "path", "label", "cohort"])
            writer.writeheader()
            writer.writerow({"sample_id": "s_0", "path": str(self.img_png), "label": "fake", "cohort": "training_diagnostic"})
        with self.assertRaises(ValueError) as ctx:
            load_and_validate_manifest(str(p_bad_label))
        self.assertIn("label must be strictly '0' or '1'", str(ctx.exception))

        # D. Duplicate sample IDs
        p_dup = self.root / "dup.csv"
        with open(p_dup, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "path", "label", "cohort"])
            writer.writeheader()
            writer.writerow({"sample_id": "same_id", "path": str(self.img_png), "label": "0", "cohort": "training_diagnostic"})
            writer.writerow({"sample_id": "same_id", "path": str(self.img_jpg), "label": "1", "cohort": "training_diagnostic"})
        with self.assertRaises(ValueError) as ctx:
            load_and_validate_manifest(str(p_dup))
        self.assertIn("duplicate sample_id", str(ctx.exception))

        # E. Missing file
        p_missing_img = self.root / "missing_img.csv"
        with open(p_missing_img, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "path", "label", "cohort"])
            writer.writeheader()
            writer.writerow({"sample_id": "s_0", "path": str(self.root / "nonexistent.png"), "label": "0", "cohort": "training_diagnostic"})
        with self.assertRaises(FileNotFoundError):
            load_and_validate_manifest(str(p_missing_img))

        # F. SHA256 mismatch
        p_bad_sha = self.root / "bad_sha.csv"
        with open(p_bad_sha, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "path", "label", "sha256", "cohort"])
            writer.writeheader()
            writer.writerow({"sample_id": "s_0", "path": str(self.img_png), "label": "0", "sha256": "wrong_sha256_hash", "cohort": "training_diagnostic"})
        with self.assertRaises(ValueError) as ctx:
            load_and_validate_manifest(str(p_bad_sha))
        self.assertIn("SHA256 mismatch", str(ctx.exception))

        # G. Missing cohort metadata
        p_no_cohort = self.root / "no_cohort.csv"
        with open(p_no_cohort, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "path", "label", "split"])
            writer.writeheader()
            writer.writerow({"sample_id": "s_0", "path": str(self.img_png), "label": "0", "split": "dev"})
        with self.assertRaises(ValueError) as ctx:
            load_and_validate_manifest(str(p_no_cohort))
        self.assertIn("explicit 'cohort' metadata is required", str(ctx.exception))

    def test_05_end_to_end_diagnostic_execution(self):
        """Execute complete diagnostic on CPU with deterministic DummyMockModel."""
        # Create valid 2-sample manifest
        m_path = self.root / "valid_manifest.csv"
        with open(m_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "path", "label", "split", "cohort", "subcollection", "sha256"])
            writer.writeheader()
            writer.writerow({
                "sample_id": "s_0", "path": str(self.img_png), "label": "0", "split": "dev",
                "cohort": "external_development", "subcollection": "external_real", "sha256": self.sha_png
            })
            writer.writerow({
                "sample_id": "s_1", "path": str(self.img_jpg), "label": "1", "split": "train",
                "cohort": "training_diagnostic", "subcollection": "numeric_fake", "sha256": self.sha_jpg
            })

        out_dir = self.root / "diag_out"
        model = DummyMockModel(score_sign=1.0)

        report = run_format_diagnostic(
            checkpoint_path="dummy.pth",
            manifest_path=str(m_path),
            output_dir=str(out_dir),
            device="cpu",
            batch_size=2,
            num_workers=0,
            model_override=model,
        )

        # Verify output directory and files exist
        self.assertTrue((out_dir / "format_sensitivity_samples.csv").is_file())
        self.assertTrue((out_dir / "format_sensitivity_report.json").is_file())

        # Verify exactly 4 condition records per sample ID
        with open(out_dir / "format_sensitivity_samples.csv", "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 8, "Expected 2 samples * 4 conditions = 8 rows")

        for sid in ("s_0", "s_1"):
            sample_conds = [r["condition"] for r in rows if r["sample_id"] == sid]
            self.assertEqual(sorted(sample_conds), ["jpeg75", "jpeg95", "original", "png_control"])

        # Verify PNG control passed
        self.assertIn("PASS", report["png_rgb_equality_control"]["status"])
        self.assertEqual(report["png_rgb_equality_control"]["max_pixel_abs_diff"], 0.0)
        self.assertEqual(report["png_rgb_equality_control"]["max_tensor_linf_diff"], 0.0)

    def test_06_nondefault_preprocessing(self):
        """Verify non-default options (bicubic, no_crop, no_resize) execute cleanly."""
        opt = SimpleNamespace(
            cropSize=200,
            loadSize=250,
            rz_interp=["bicubic"],
            no_crop=True,
            no_resize=True,
            isTrain=False,
            no_flip=True,
            data_aug=False,
        )
        tfm = build_production_eval_transform(opt)
        with Image.open(self.img_png) as img:
            t = tfm(img)
            self.assertEqual(t.dim(), 3)
            self.assertEqual(t.shape[0], 3)
            # Since no_crop and no_resize are true, spatial size matches original image
            self.assertEqual(t.shape[1], 120)
            self.assertEqual(t.shape[2], 140)

    def test_07_reversed_score_sign(self):
        """Verify model score_sign inversion reverses logits."""
        m_path = self.root / "sign_manifest.csv"
        with open(m_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "path", "label", "split", "cohort"])
            writer.writeheader()
            writer.writerow({"sample_id": "s_0", "path": str(self.img_png), "label": "0", "split": "dev", "cohort": "external_development"})

        model_pos = DummyMockModel(score_sign=1.0)
        model_neg = DummyMockModel(score_sign=-1.0)

        out_pos = self.root / "out_pos"
        out_neg = self.root / "out_neg"

        run_format_diagnostic("dummy.pth", str(m_path), str(out_pos), "cpu", 1, 0, model_override=model_pos)
        run_format_diagnostic("dummy.pth", str(m_path), str(out_neg), "cpu", 1, 0, model_override=model_neg)

        with open(out_pos / "format_sensitivity_samples.csv") as f:
            rows_pos = list(csv.DictReader(f))
        with open(out_neg / "format_sensitivity_samples.csv") as f:
            rows_neg = list(csv.DictReader(f))

        orig_pos = next(r for r in rows_pos if r["condition"] == "original")
        orig_neg = next(r for r in rows_neg if r["condition"] == "original")

        log_pos = float(orig_pos["canonical_logit"])
        log_neg = float(orig_neg["canonical_logit"])
        self.assertAlmostEqual(log_pos, -log_neg, places=5, msg="score_sign=-1.0 must negate canonical logit")

    def test_08_workers_and_batch_restrictions(self):
        """Verify workers > 0 and batch_size <= 0 raise ValueError."""
        m_path = self.root / "workers_manifest.csv"
        with open(m_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "path", "label", "split", "cohort"])
            writer.writeheader()
            writer.writerow({"sample_id": "s_0", "path": str(self.img_png), "label": "0", "split": "dev", "cohort": "training_diagnostic"})

        with self.assertRaises(ValueError) as ctx:
            run_format_diagnostic("dummy.pth", str(m_path), str(self.root / "out_w"), "cpu", 1, num_workers=2)
        self.assertIn("Only --workers 0 is supported", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            run_format_diagnostic("dummy.pth", str(m_path), str(self.root / "out_b"), "cpu", batch_size=0, num_workers=0)
        self.assertIn("batch_size must be positive", str(ctx.exception))

    def test_09_nonempty_output_dir_rejected(self):
        """Verify run_format_diagnostic rejects non-empty destination directory."""
        m_path = self.root / "nonempty_manifest.csv"
        with open(m_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "path", "label", "split", "cohort"])
            writer.writeheader()
            writer.writerow({"sample_id": "s_0", "path": str(self.img_png), "label": "0", "split": "dev", "cohort": "training_diagnostic"})

        nonempty_dir = self.root / "nonempty_dir"
        nonempty_dir.mkdir(parents=True, exist_ok=True)
        (nonempty_dir / "existing_file.txt").write_text("data")

        model = DummyMockModel()
        with self.assertRaises(ValueError) as ctx:
            run_format_diagnostic("dummy.pth", str(m_path), str(nonempty_dir), "cpu", 1, 0, model_override=model)
        self.assertIn("already exists and is not empty", str(ctx.exception))

    def test_10_production_transform_tensor_parity(self):
        """Verify production evaluation transform produces bit-identical tensor to reference pipeline."""
        opt = SimpleNamespace(
            cropSize=224,
            loadSize=256,
            rz_interp=["bilinear"],
            no_crop=False,
            no_resize=False,
            isTrain=False,
            no_flip=True,
            data_aug=False,
        )
        tfm = build_production_eval_transform(opt)
        
        # Reference manual pipeline
        ref_tfm = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        with Image.open(self.img_png) as img:
            rgb_img = img.convert("RGB")
            t_prod = tfm(rgb_img)
            t_ref = ref_tfm(rgb_img)
            self.assertTrue(torch.allclose(t_prod, t_ref, atol=1e-6), "Production eval transform must match reference tensor")


if __name__ == "__main__":
    unittest.main()

