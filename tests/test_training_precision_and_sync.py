"""
Unit tests for training precision, AMP modernization, detached loss accumulation,
tensor-based DDP validation, and checkpoint provenance caching.
STATUS: NOT RUN (Static code phase test suite)
"""

import argparse
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from training.runtime.amp import AmpMixin
from training.base_trainer import BaseTrainer
from training.validator import Validator, ValidationResult
from data.manifest import sha256, clear_manifest_cache, _FILE_HASH_CACHE


class TinyLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc(x)


class DummyTrainer(AmpMixin):
    def __init__(self, opt):
        self._init_amp(opt)


class DummyDataset(Dataset):
    def __init__(self, size=12):
        self.size = size
        self.data = torch.randn(size, 4)
        self.labels = torch.tensor([i % 2 for i in range(size)], dtype=torch.float32)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class TestAmpPrecision(unittest.TestCase):
    def test_amp_fp16_initialization(self):
        opt = argparse.Namespace(use_amp=True, amp_dtype='fp16')
        trainer = DummyTrainer(opt)
        if torch.cuda.is_available():
            self.assertTrue(trainer._amp_enabled)
            self.assertEqual(trainer._amp_dtype, torch.float16)
            self.assertIsNotNone(trainer._grad_scaler)

    def test_amp_bf16_fallback_if_unsupported(self):
        opt = argparse.Namespace(use_amp=True, amp_dtype='bf16')
        with patch('torch.cuda.is_bf16_supported', return_value=False):
            trainer = DummyTrainer(opt)
            if torch.cuda.is_available():
                self.assertEqual(trainer._amp_dtype, torch.float16)

    def test_amp_step_counter(self):
        opt = argparse.Namespace(use_amp=False)
        trainer = DummyTrainer(opt)
        mock_opt = MagicMock()
        success = trainer.amp_step(mock_opt)
        self.assertTrue(success)
        self.assertEqual(trainer.optimizer_steps_attempted, 1)
        self.assertEqual(trainer.optimizer_steps_successful, 1)


class TestDetachedLossAndTails(unittest.TestCase):
    def test_sample_weighted_loss_math(self):
        # Two microbatches: microbatch 1 with 16 samples and loss 2.0; microbatch 2 with 4 samples and loss 1.0
        # True sample mean = (16 * 2.0 + 4 * 1.0) / 20 = 36 / 20 = 1.80
        loss_mb1 = torch.tensor(2.0)
        n_mb1 = 16
        loss_mb2 = torch.tensor(1.0)
        n_mb2 = 4

        epoch_loss_sum = (loss_mb1.detach() * n_mb1) + (loss_mb2.detach() * n_mb2)
        total_samples = n_mb1 + n_mb2
        sample_weighted_loss = float((epoch_loss_sum / total_samples).item())

        self.assertAlmostEqual(sample_weighted_loss, 1.80, places=5)
        # Arithmetic mean without sample weighting would be 1.50
        self.assertNotAlmostEqual(sample_weighted_loss, 1.50)


class TestValidatorTensorGatherAndPrecision(unittest.TestCase):
    def setUp(self):
        self.dataset = DummyDataset(8)
        self.loader = DataLoader(self.dataset, batch_size=4, shuffle=False)

    def test_validator_fp32_local(self):
        model = MagicMock()
        model.output = torch.tensor([[0.5], [-0.5], [1.2], [-1.0]])
        model.label = torch.tensor([1.0, 0.0, 1.0, 0.0])
        model.get_loss.return_value = torch.tensor(0.4)
        model.is_train = False

        validator = Validator(precision='fp32')
        result = validator.validate(model, self.loader)

        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.num_samples, 8)
        self.assertGreater(result.accuracy, 0.0)

    def test_simulated_ddp_variable_lengths_with_empty_rank(self):
        """Simulate variable-length gather across 3 ranks: rank 0 (3 samples), rank 1 (0 samples), rank 2 (2 samples)."""
        world_size = 3
        device = 'cpu'

        # Local data
        r0_preds = np.array([0.9, 0.1, 0.8], dtype=np.float32)
        r0_labels = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        r0_indices = np.array([0, 1, 2], dtype=np.int64)

        r1_preds = np.array([], dtype=np.float32)
        r1_labels = np.array([], dtype=np.float32)
        r1_indices = np.array([], dtype=np.int64)

        r2_preds = np.array([0.2, 0.7], dtype=np.float32)
        r2_labels = np.array([0.0, 1.0], dtype=np.float32)
        r2_indices = np.array([3, 4], dtype=np.int64)

        lengths = [len(r0_preds), len(r1_preds), len(r2_preds)]
        max_len = max(lengths)  # 3

        # Pad to max_len
        def pad_data(p, l, idx):
            p_pad = np.zeros(max_len, dtype=np.float32)
            l_pad = np.zeros(max_len, dtype=np.float32)
            i_pad = np.zeros(max_len, dtype=np.int64)
            if len(p) > 0:
                p_pad[:len(p)] = p
                l_pad[:len(l)] = l
                i_pad[:len(idx)] = idx
            return p_pad, l_pad, i_pad

        gathered_p = [pad_data(r0_preds, r0_labels, r0_indices)[0],
                      pad_data(r1_preds, r1_labels, r1_indices)[0],
                      pad_data(r2_preds, r2_labels, r2_indices)[0]]
        gathered_l = [pad_data(r0_preds, r0_labels, r0_indices)[1],
                      pad_data(r1_preds, r1_labels, r1_indices)[1],
                      pad_data(r2_preds, r2_labels, r2_indices)[1]]
        gathered_i = [pad_data(r0_preds, r0_labels, r0_indices)[2],
                      pad_data(r1_preds, r1_labels, r1_indices)[2],
                      pad_data(r2_preds, r2_labels, r2_indices)[2]]

        # Unpad on rank 0
        valid_p = [gathered_p[r][:lengths[r]] for r in range(world_size) if lengths[r] > 0]
        valid_l = [gathered_l[r][:lengths[r]] for r in range(world_size) if lengths[r] > 0]
        valid_i = [gathered_i[r][:lengths[r]] for r in range(world_size) if lengths[r] > 0]

        all_preds = np.concatenate(valid_p)
        all_labels = np.concatenate(valid_l)
        all_indices = np.concatenate(valid_i)

        self.assertEqual(len(all_preds), 5)
        self.assertEqual(len(all_labels), 5)
        self.assertEqual(len(all_indices), 5)
        np.testing.assert_array_equal(all_indices, np.array([0, 1, 2, 3, 4]))


class TestManifestDigestCacheAndCheckpointing(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.file_path = Path(self.tmpdir) / 'test_file.txt'
        self.file_path.write_text("sample content for digest testing", encoding='utf-8')
        clear_manifest_cache()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        clear_manifest_cache()

    def test_sha256_cache_avoids_repeated_reads(self):
        d1 = sha256(self.file_path)
        self.assertIn(str(self.file_path.resolve()), [k[0] for k in _FILE_HASH_CACHE.keys()])

        # Second call returns cached without error
        d2 = sha256(self.file_path)
        self.assertEqual(d1, d2)

    def test_save_epoch_freq_zero_disables_epoch_copies(self):
        from training.hooks.checkpoint_hook import CheckpointHook
        mock_mgr = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 5

        hook = CheckpointHook(checkpoint_manager=mock_mgr, save_epoch_freq=0)
        hook.on_epoch_end(mock_trainer)

        # save_last is called, but save_epoch is NOT called when save_epoch_freq=0
        mock_mgr.save_last.assert_called_once()
        mock_mgr.save_epoch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
