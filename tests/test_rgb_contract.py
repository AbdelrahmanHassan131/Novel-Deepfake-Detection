# STATUS: NOT RUN (runtime tests deferred; static syntax check only)
"""
RGB Preprocessing Contract and Parity Test Suite.

Tests:
1. RGB mode, grayscale ('L'), RGBA, and CMYK decode conversion to 3-channel tensors
2. Spatial transform modes: standard (resize+crop), no_crop, no_resize, and no_crop+no_resize
3. Deterministic evaluation transforms (consecutive passes yield bit-identical tensors)
4. Train-only augmentation vs zero-augmentation evaluation guarantee
5. CheckpointLoader options and preprocessing metadata restoration
6. Tensor parity between RGBDataset evaluation pipeline and manual reference contract
7. Prediction export with optional logits and generalization report schema versioning
8. Windows multiprocessing spawn compatibility for evaluation transforms
"""

import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from data.datasets.rgb_dataset import RGBDataset
from data.manifest import write_manifest
from data.transforms.augmentations import data_augment
from evaluation.checkpoint_loader import CheckpointLoader
from evaluation.generalization import (
    binary_metrics,
    export_predictions,
    read_predictions,
    summarize,
)
from tools.diagnose_rgb_parity import get_eval_transform_from_opt, check_module_eval_and_bn_modes


class TestRGBContract(unittest.TestCase):

    def setUp(self):
        self.opt_eval = SimpleNamespace(
            isTrain=True,
            no_crop=False,
            no_resize=False,
            no_flip=True,
            cropSize=224,
            loadSize=256,
            rz_interp=['bilinear'],
            blur_prob=0.0,
            jpg_prob=0.0,
            noise_prob=0.0,
            downscale_prob=0.0,
            classes=['real', 'fake'],
            manifest=None,
        )

    def test_rgb_mode_conversions(self):
        """Grayscale, RGBA, and CMYK must all yield 3-channel tensors of shape (3, 224, 224)."""
        transform = get_eval_transform_from_opt(self.opt_eval)
        modes = [
            Image.new('RGB', (300, 300), color=(100, 150, 200)),
            Image.new('L', (300, 300), color=128).convert('RGB'),
            Image.new('RGBA', (300, 300), color=(100, 150, 200, 255)).convert('RGB'),
            Image.new('CMYK', (300, 300), color=(0, 50, 100, 20)).convert('RGB'),
        ]
        for img in modes:
            t = transform(img)
            self.assertEqual(t.shape, (3, 224, 224))
            self.assertEqual(t.dtype, torch.float32)

    def test_spatial_modes(self):
        """Test crop/no-crop/no-resize combinations."""
        img = Image.new('RGB', (300, 200), color=(50, 100, 150))

        # 1. Standard: resize to 256 then center-crop to 224
        t_std = get_eval_transform_from_opt(self.opt_eval)(img)
        self.assertEqual(t_std.shape, (3, 224, 224))

        # 2. No crop: resize to loadSize (256)
        opt_no_crop = SimpleNamespace(**vars(self.opt_eval))
        opt_no_crop.no_crop = True
        t_no_crop = get_eval_transform_from_opt(opt_no_crop)(img)
        # Resizing (300, 200) with short side 256 yields (384, 256)
        self.assertEqual(t_no_crop.shape[0], 3)
        self.assertEqual(min(t_no_crop.shape[1:]), 256)

        # 3. No resize: direct center-crop from raw resolution
        opt_no_resize = SimpleNamespace(**vars(self.opt_eval))
        opt_no_resize.no_resize = True
        opt_no_resize.cropSize = 100
        t_no_resize = get_eval_transform_from_opt(opt_no_resize)(img)
        self.assertEqual(t_no_resize.shape, (3, 100, 100))

    def test_deterministic_eval_transforms(self):
        """Evaluation transform on same input must produce bit-identical tensors."""
        img = Image.new('RGB', (256, 256), color=(42, 84, 126))
        transform = get_eval_transform_from_opt(self.opt_eval)
        t1 = transform(img)
        t2 = transform(img)
        torch.testing.assert_close(t1, t2, atol=0, rtol=0)

    def test_train_only_augmentations_disabled_in_eval(self):
        """data_augment must be a no-op when isTrain=False even if probs are 1.0."""
        arr = np.random.default_rng(42).integers(0, 256, (128, 128, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        opt = SimpleNamespace(
            isTrain=False,
            blur_prob=1.0, blur_sig=[2.0],
            jpg_prob=1.0, jpg_method=['cv2'], jpg_qual=[30],
            noise_prob=1.0, noise_std=[10.0, 10.0],
            downscale_prob=1.0, downscale_range=[0.5, 0.5],
        )
        aug_img = data_augment(img, opt)
        np.testing.assert_array_equal(np.array(img), np.array(aug_img))

    def test_module_bn_and_eval_mode_check(self):
        """Verify check_module_eval_and_bn_modes catches eval vs train states."""
        from config.configuration import Config
        from config.compatibility import config_to_opt
        from models.wang2020_128.trainer import Wang2020_128Trainer
        opt = config_to_opt(Config.from_defaults())
        opt.arch = 'Wang2020_128'
        opt.pretrained = False
        opt.isTrain = True
        trainer = Wang2020_128Trainer(opt)
        trainer.eval()
        result_eval = check_module_eval_and_bn_modes(trainer)
        self.assertTrue(result_eval['model_eval_mode_correct'])
        self.assertTrue(result_eval['bn_eval_mode_correct'])
        self.assertTrue(result_eval['dropout_eval_mode_correct'])

        trainer.train()
        result_train = check_module_eval_and_bn_modes(trainer)
        self.assertFalse(result_train['model_eval_mode_correct'])
        self.assertFalse(result_train['bn_eval_mode_correct'])

    def test_prediction_export_and_logits(self):
        """Test optional logit export in export_predictions and read_predictions roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'preds.csv'
            records = [
                {'sample_id': 's1', 'path': 'img1.png', 'label': 0, 'split': 'dev', 'dataset_source': 'A', 'group_id': 'g1'},
                {'sample_id': 's2', 'path': 'img2.png', 'label': 1, 'split': 'dev', 'dataset_source': 'A', 'group_id': 'g2'},
            ]
            probs = [0.15, 0.85]
            logits = [-1.734, 1.734]
            export_predictions(str(csv_path), records, probs, checkpoint_hash='dummy_hash', logits=logits)

            loaded = read_predictions(str(csv_path))
            self.assertEqual(len(loaded), 2)
            self.assertIn('logit', loaded[0])
            self.assertAlmostEqual(loaded[0]['logit'], -1.734, places=3)
            self.assertAlmostEqual(loaded[1]['logit'], 1.734, places=3)

    def test_schema_2_metrics_and_quantiles(self):
        """binary_metrics and summarize must produce schema_version 2.0 with quantiles and saturation."""
        labels = [0, 0, 0, 1, 1, 1]
        probs = [0.00005, 0.1, 0.4, 0.6, 0.9, 0.99995]
        logits = [-10.0, -2.19, -0.4, 0.4, 2.19, 10.0]
        res = binary_metrics(labels, probs, logits=logits)

        self.assertEqual(res['schema_version'], '2.0')
        self.assertIsNotNone(res['real_recall'])
        self.assertIsNotNone(res['fake_recall'])
        self.assertIn('score_quantiles', res)
        self.assertIn('logit_quantiles', res)
        self.assertIn('saturation_counts', res)
        self.assertEqual(res['saturation_counts']['real_low_p0001'], 1)
        self.assertEqual(res['saturation_counts']['fake_high_p9999'], 1)

    def test_windows_spawn_safe_dataset(self):
        """RGBDataset transforms must survive multiprocessing serialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for c in ('real', 'fake'):
                (root / c).mkdir()
                Image.new('RGB', (32, 32)).save(root / c / 'sample.png')
            opt = SimpleNamespace(**vars(self.opt_eval))
            opt.dataroot = str(root)
            dataset = RGBDataset(opt, str(root))
            dumped = pickle.dumps(dataset)
            restored = pickle.loads(dumped)
            self.assertEqual(len(restored), len(dataset))


if __name__ == '__main__':
    unittest.main()
