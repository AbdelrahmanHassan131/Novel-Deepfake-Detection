import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image

from config import Config, config_to_opt, load_config
from data.manifest import read_manifest, write_manifest, audit_rows
from data.builders.dataset_factory import get_dataset
from data.transforms.augmentations import data_augment
from models.mha.token_fusion import TokenAttentionFusion, GatedFusion
from evaluation.generalization import binary_metrics, group_intervals, calibrate, export_predictions
from evaluation.pixel_pipeline import projected_attack

torch.set_num_threads(2)


def options(root):
    opt = config_to_opt(Config.from_defaults())
    opt.arch = 'Wang2020_128'
    opt.dataroot = str(root)
    opt.compute_wavelets = False
    opt.isTrain = False
    opt.cropSize = opt.loadSize = 32
    opt.gpu_ids = []
    opt.pretrained = False
    opt.continue_train = False
    opt.batch_size = 2
    return opt


class DataTests(unittest.TestCase):
    def test_nested_labels_do_not_become_generator_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('fake/ADM', 'fake/DDIM', 'real/CelebA'):
                (root / name).mkdir(parents=True)
                Image.new('RGB', (32, 32)).save(root / name / 'test.png')
            dataset = get_dataset(options(root))
            self.assertEqual([1, 1, 0], dataset.targets)
            for path, label in dataset.samples:
                self.assertEqual(label, int('fake' in Path(path).parts))

    def test_manifest_loading_and_group_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (32, 32)).save(root / 'x.png')
            rows = [dict(sample_id='a', path='x.png', label=0, split='train', dataset_source='A', group_id='same'),
                    dict(sample_id='b', path='x.png', label=1, split='dev', dataset_source='A', group_id='same')]
            write_manifest(root / 'manifest.csv', rows)
            self.assertFalse(audit_rows(read_manifest(root / 'manifest.csv'))['passed'])
            opt = options(root)
            opt.manifest, opt.manifest_split = str(root / 'manifest.csv'), 'dev'
            self.assertEqual(get_dataset(opt).targets, [1])

    def test_validation_never_augments(self):
        image = Image.fromarray(np.random.default_rng(5).integers(0, 256, (32, 32, 3), dtype=np.uint8))
        opt = options('.')
        opt.blur_prob = opt.jpg_prob = opt.noise_prob = opt.downscale_prob = 1.0
        self.assertTrue(np.array_equal(np.array(image), np.array(data_augment(image, opt))))

    def test_config_roundtrip(self):
        opt = options('.')
        opt.manifest = 'manifest.csv'
        opt.val_manifest_split = 'external_dev'
        opt.noise_prob = .3
        restored = config_to_opt(load_config(opt, validate=False))
        self.assertEqual(restored.manifest, opt.manifest)
        self.assertEqual(restored.noise_prob, .3)
        self.assertEqual(restored.val_manifest_split, 'external_dev')


class ModelTests(unittest.TestCase):
    def test_attention_has_multiple_keys_and_varies(self):
        torch.manual_seed(7)
        model = TokenAttentionFusion(16, 4, 0).eval()
        rgb, wavelet = torch.randn(3, 16, requires_grad=True), torch.randn(3, 16, requires_grad=True)
        output = model(rgb, wavelet)
        weights = model.last_attention.clone()
        self.assertEqual(weights.shape, (3, 4, 3, 3))
        model(rgb, wavelet.flip(-1))
        self.assertFalse(torch.allclose(weights[:, :, 0], model.last_attention[:, :, 0]))
        output.sum().backward()
        self.assertGreater(float(rgb.grad.abs().sum()), 0)
        self.assertGreater(float(wavelet.grad.abs().sum()), 0)

    def test_gate_is_normalized(self):
        model = GatedFusion(16, 0)
        self.assertEqual(model(torch.randn(2, 16), torch.randn(2, 16)).shape, (2, 1))
        torch.testing.assert_close(model.last_gate.sum(-1), torch.ones(2))

    def test_missing_expert_fails(self):
        from models.mha.trainer import MHAFusionTrainer
        opt = options('.')
        opt.isTrain = True
        with self.assertRaises(FileNotFoundError):
            MHAFusionTrainer(opt)

    def test_level_four_and_embedding_ablation(self):
        from models.wolter2021.wavelet_cnn import WaveletPacketCNN128
        model = WaveletPacketCNN128(768, embed_dim=64).eval()
        self.assertEqual(model(torch.randn(2, 768, 14, 14)).shape, (2, 1))

    def test_cpu_gpu_wavelet_parity_and_gradient(self):
        from data.wavelets.backends.cpu_backend import CPUWaveletBackend
        from data.wavelets.backends.gpu_backend import GPUWaveletBackend
        image = np.random.default_rng(9).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        for log in (False, True):
            cpu = CPUWaveletBackend(wavelet='haar', level=2, mode='reflect', log_scale=log, log_mode='signed_log1p')
            gpu = GPUWaveletBackend(wavelet='haar', level=2, mode='reflect', log_scale=log, device=torch.device('cpu'), log_mode='signed_log1p')
            inputs = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32, requires_grad=True)
            actual = gpu(inputs)
            expected = cpu(Image.fromarray(image))
            torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
            self.assertTrue(torch.isfinite(actual).all())
            actual.sum().backward()
            self.assertIsNotNone(inputs.grad)


class EvaluationTests(unittest.TestCase):
    def test_metrics_reproduce_counts(self):
        result = binary_metrics([0, 0, 1, 1], [.1, .7, .8, .9])
        self.assertEqual(result['confusion_matrix'], [[1, 1], [0, 2]])
        self.assertEqual(result['accuracy'], .75)
        self.assertEqual(result['balanced_accuracy'], .75)
        self.assertIsNone(binary_metrics([1, 1], [.8, .9])['roc_auc'])

    def test_paired_intervals_identical_predictions(self):
        p = [.1, .6, .8, .9]
        ci = group_intervals([0, 0, 1, 1], p, ['a', 'b', 'c', 'd'], repeats=20, other=p)
        self.assertEqual(ci['intervals_95']['accuracy']['low'], 0)
        self.assertEqual(ci['intervals_95']['accuracy']['high'], 0)

    def test_cannot_calibrate_on_test(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            rows = [dict(sample_id=str(i), label=i, split='external_test') for i in range(2)]
            export_predictions(path / 'predictions.csv', rows, [.1, .9], 'hash')
            with self.assertRaises(ValueError):
                calibrate(path / 'predictions.csv', path / 'threshold.json')

    def test_pgd_pixel_budget_and_loss(self):
        class Toy(torch.nn.Module):
            def forward(self, x):
                return x.flatten(1).mean(1) * 10 - 5
        x = torch.rand(2, 3, 8, 8)
        y = torch.tensor([0., 1.])
        model = Toy()
        adv = projected_attack(model, x, y, eps=.02, step_size=.01, steps=3)
        self.assertLessEqual(float((adv - x).abs().max()), .020001)
        self.assertGreaterEqual(float(adv.min()), 0)
        self.assertLessEqual(float(adv.max()), 1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits
        self.assertGreaterEqual(float(loss(model(adv), y)), float(loss(model(x), y)))


if __name__ == '__main__':
    unittest.main()
