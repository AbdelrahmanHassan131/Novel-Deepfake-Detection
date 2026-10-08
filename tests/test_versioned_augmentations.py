"""
Unit tests for versioned augmentation recipes, class independence,
crop policies, interpolation options, and parameter validation.
STATUS: NOT RUN (Static code phase test suite)
"""

import argparse
import unittest
from types import SimpleNamespace
import numpy as np
from PIL import Image

from config.configuration import Config
from config.compatibility import config_from_opt, config_to_opt
from config.validator import ConfigValidator
from data.transforms.augmentations import (
    AUGMENTATION_RECIPES,
    resolve_augmentation_recipe,
    validate_augmentation_params,
    data_augment,
)
from data.transforms.resize import custom_resize


class TestVersionedAugmentationRecipes(unittest.TestCase):
    def test_rgb_v1_preset_parameters(self):
        opt = SimpleNamespace(aug_recipe='rgb_v1')
        resolved = resolve_augmentation_recipe(opt)

        self.assertEqual(resolved.blur_prob, 0.5)
        self.assertEqual(resolved.blur_sig, [0.0, 3.0])
        self.assertEqual(resolved.jpg_prob, 0.5)
        self.assertEqual(resolved.jpg_qual, [50, 60, 70, 80, 90, 95])
        self.assertEqual(resolved.jpg_method, ['cv2'])
        self.assertEqual(resolved.noise_prob, 0.0)
        self.assertEqual(resolved.downscale_prob, 0.0)

    def test_legacy_recipe_is_unaugmented_by_default(self):
        opt = SimpleNamespace(aug_recipe='legacy')
        resolved = resolve_augmentation_recipe(opt)
        self.assertEqual(resolved.blur_prob, 0.0)
        self.assertEqual(resolved.jpg_prob, 0.0)

    def test_custom_recipe_preserves_user_settings(self):
        opt = SimpleNamespace(
            aug_recipe='custom',
            blur_prob=0.8,
            jpg_prob=0.3,
            blur_sig=[1.5],
            jpg_qual=[85],
        )
        resolved = resolve_augmentation_recipe(opt)
        self.assertEqual(resolved.blur_prob, 0.8)
        self.assertEqual(resolved.jpg_prob, 0.3)

    def test_parameter_validation_catches_invalid_values(self):
        # Invalid blur_prob > 1.0
        opt_bad_prob = SimpleNamespace(blur_prob=1.5, jpg_prob=0.5, blur_sig=[0.5], jpg_qual=[75])
        with self.assertRaises(ValueError):
            validate_augmentation_params(opt_bad_prob)

        # Invalid negative blur_sig
        opt_bad_sig = SimpleNamespace(blur_prob=0.5, jpg_prob=0.5, blur_sig=[-1.0], jpg_qual=[75])
        with self.assertRaises(ValueError):
            validate_augmentation_params(opt_bad_sig)

        # Invalid jpg_qual > 100
        opt_bad_qual = SimpleNamespace(blur_prob=0.5, jpg_prob=0.5, blur_sig=[0.5], jpg_qual=[120])
        with self.assertRaises(ValueError):
            validate_augmentation_params(opt_bad_qual)


class TestClassIndependenceAndEvaluationBypass(unittest.TestCase):
    def test_evaluation_bypass_preserves_raw_image(self):
        arr = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)

        # In evaluation mode, even with 100% blur and 100% JPEG probability, image must be identical
        opt_eval = SimpleNamespace(
            isTrain=False,
            blur_prob=1.0, blur_sig=[3.0],
            jpg_prob=1.0, jpg_qual=[30], jpg_method=['cv2'],
            noise_prob=1.0, noise_std=[10.0],
            downscale_prob=1.0, downscale_range=[0.5],
        )
        result = data_augment(img, opt_eval)
        np.testing.assert_array_equal(np.array(img), np.array(result))

    def test_class_independence(self):
        # Augmentation function must not depend on sample label
        opt = SimpleNamespace(
            isTrain=True,
            blur_prob=0.0,
            jpg_prob=0.0,
            noise_prob=0.0,
            downscale_prob=0.0,
        )
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(arr)

        res_real = data_augment(img, opt)
        res_fake = data_augment(img, opt)
        np.testing.assert_array_equal(np.array(res_real), np.array(res_fake))


class TestCropPolicyAndResizeInterpolation(unittest.TestCase):
    def test_resize_interpolation_parsing(self):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        opt = SimpleNamespace(
            isTrain=False,
            loadSize=64,
            image_size=64,
            rz_interp='bicubic',
        )
        resized = custom_resize(img, opt)
        self.assertEqual(resized.size, (64, 64))

    def test_config_roundtrip_and_validation(self):
        opt = argparse.Namespace(
            dataroot='./dummy',
            crop_policy='scale_and_crop',
            aug_recipe='rgb_v1',
            rz_interp='bilinear',
        )
        cfg = config_from_opt(opt)
        self.assertEqual(cfg.data.crop_policy, 'scale_and_crop')
        self.assertEqual(cfg.augmentation.aug_recipe, 'rgb_v1')

        opt_back = config_to_opt(cfg)
        self.assertEqual(opt_back.crop_policy, 'scale_and_crop')
        self.assertEqual(opt_back.aug_recipe, 'rgb_v1')

        # Invalid crop_policy
        cfg.data.crop_policy = 'invalid_crop'
        report = ConfigValidator.validate(cfg)
        self.assertFalse(report.is_valid)
        self.assertTrue(any("crop_policy" in e for e in report.errors))


if __name__ == '__main__':
    unittest.main()
