# STATUS: NOT RUN
"""
Deferred test suite for Code Review Round 3 - Prompt P.
Validates:
  - T2: Gradient accumulation weights computed from actual local micro-batch samples and rank-local samplers under DDP,
        avoiding global dataset length estimation errors (e.g. 16/18 and 2/18 instead of 16/68 and 52/68).
  - T3: Race-free distributed experiment directory initialization: atomic rank 0 decision, broadcast, barrier synchronization,
        and coherent FileExistsError propagation across all ranks on pre-existing runs.

DO NOT EXECUTE AT RUNTIME IN THIS ENVIRONMENT.
Static AST / syntax validation only.
"""

import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
import tempfile
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from training.base_trainer import BaseTrainer
from experiment.manager import ExperimentManager


class DummyToyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            'image': torch.zeros((3, 32, 32)),
            'label': torch.tensor(idx % 2, dtype=torch.long),
            'index': idx,
        }


class DummyRankSampler(Sampler):
    """Simulates a per-rank distributed sampler with rank_samples elements."""
    def __init__(self, rank_samples=50):
        self.rank_samples = rank_samples

    def __len__(self):
        return self.rank_samples

    def __iter__(self):
        return iter(range(self.rank_samples))


class TestRound3DistributedCorrectness(unittest.TestCase):
    """Static deferred unit test contract for Prompt P distributed training correctness."""

    def test_t2_ddp_accumulation_weight_calculation(self):
        """
        Verify that for N=100, 2 ranks (50 samples per rank), batch_size=16, grad_accum=2:
        The last accumulation window is weighted 16/18 and 2/18 (NOT 16/68 and 52/68).
        """
        dataset = DummyToyDataset(size=100)
        # Rank-local sampler with 50 samples -> batches of 16, 16, 16, 2
        rank_sampler = DummyRankSampler(rank_samples=50)
        loader = DataLoader(dataset, batch_size=16, sampler=rank_sampler, drop_last=False)

        # Mock model and runtime simulating rank 0 in a 2-rank DDP job
        mock_opt = SimpleNamespace(batch_size=16, grad_accum_steps=2, use_amp=False)
        mock_runtime = SimpleNamespace(is_distributed=True, world_size=2, rank=0, is_main=True)

        mock_model = MagicMock()
        mock_model.train.return_value = None
        mock_model.name.return_value = 'TestModel'
        mock_model.opt = mock_opt
        mock_model.optimizer = MagicMock()

        trainer = BaseTrainer(mock_opt, mock_model, mock_runtime)
        trainer.train_loader = loader
        trainer.grad_accum_steps = 2

        recorded_weights = []

        def mock_train_step(batch, is_accum_end=True, accum_steps=1, loss_weight=None):
            recorded_weights.append((loss_weight, is_accum_end))
            return 0.1

        trainer.train_step = mock_train_step
        trainer.train_epoch()

        # 4 batches total: [16, 16, 16, 2]
        self.assertEqual(len(recorded_weights), 4)

        # Window 0: batch 0 (size 16), batch 1 (size 16) -> equal weights 0.5
        w0, end0 = recorded_weights[0]
        w1, end1 = recorded_weights[1]
        self.assertAlmostEqual(w0, 0.5, places=5)
        self.assertFalse(end0)
        self.assertAlmostEqual(w1, 0.5, places=5)
        self.assertTrue(end1)

        # Window 1: batch 2 (size 16), batch 3 (size 2) -> window total 18
        # Batch 2 weight must be 16/18, batch 3 weight must be 2/18!
        w2, end2 = recorded_weights[2]
        w3, end3 = recorded_weights[3]
        expected_w2 = 16.0 / 18.0
        expected_w3 = 2.0 / 18.0
        self.assertAlmostEqual(w2, expected_w2, places=5)
        self.assertFalse(end2)
        self.assertAlmostEqual(w3, expected_w3, places=5)
        self.assertTrue(end3)

        # Sum of weights in window 1 must be exactly 1.0
        self.assertAlmostEqual(w2 + w3, 1.0, places=5)

    def test_t2_drop_last_accumulation_weights(self):
        """Verify that when drop_last=True, incomplete tail batches are dropped and weights remain nominal."""
        dataset = DummyToyDataset(size=100)
        rank_sampler = DummyRankSampler(rank_samples=50)
        loader = DataLoader(dataset, batch_size=16, sampler=rank_sampler, drop_last=True)

        mock_opt = SimpleNamespace(batch_size=16, grad_accum_steps=2, use_amp=False)
        mock_runtime = SimpleNamespace(is_distributed=True, world_size=2, rank=0, is_main=True)

        mock_model = MagicMock()
        mock_model.train.return_value = None
        mock_model.name.return_value = 'TestModel'
        mock_model.opt = mock_opt
        mock_model.optimizer = MagicMock()

        trainer = BaseTrainer(mock_opt, mock_model, mock_runtime)
        trainer.train_loader = loader
        trainer.grad_accum_steps = 2

        recorded_weights = []

        def mock_train_step(batch, is_accum_end=True, accum_steps=1, loss_weight=None):
            recorded_weights.append((loss_weight, is_accum_end))
            return 0.1

        trainer.train_step = mock_train_step
        trainer.train_epoch()

        # 3 batches of 16 (48 samples total), tail 2 samples dropped
        self.assertEqual(len(recorded_weights), 3)
        # Window 0: 2 batches of 16 -> 0.5 each
        self.assertAlmostEqual(recorded_weights[0][0], 0.5, places=5)
        self.assertAlmostEqual(recorded_weights[1][0], 0.5, places=5)
        # Window 1: 1 batch of 16 (single batch window) -> 1.0
        self.assertAlmostEqual(recorded_weights[2][0], 1.0, places=5)

    def test_t3_experiment_collision_coordination_and_race_prevention(self):
        """Verify rank 0 atomic decision and coherent broadcast across simulated distributed ranks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExperimentManager(base_dir=tmpdir)
            opt = SimpleNamespace(continue_train=False, resume_checkpoint=None)

            # Scenario A: Directory already exists before run launch
            existing_dir = os.path.join(tmpdir, "exp_run01")
            os.makedirs(existing_dir)

            # Mock distributed environment with 2 ranks
            with patch('torch.distributed.is_available', return_value=True), \
                 patch('torch.distributed.is_initialized', return_value=True), \
                 patch('torch.distributed.get_rank', return_value=0), \
                 patch('torch.distributed.barrier', return_value=None):

                def mock_broadcast(obj_list, src=0):
                    pass

                with patch('torch.distributed.broadcast_object_list', side_effect=mock_broadcast):
                    # Rank 0 must raise FileExistsError
                    with self.assertRaises(FileExistsError):
                        mgr.create_experiment("exp", opt, experiment_id="run01")

            # Scenario B: Fresh directory created by rank 0 and received by rank 1 without race collision
            with patch('torch.distributed.is_available', return_value=True), \
                 patch('torch.distributed.is_initialized', return_value=True), \
                 patch('torch.distributed.barrier', return_value=None):

                # Simulate Rank 0 creating
                with patch('torch.distributed.get_rank', return_value=0):
                    shared_decision = []
                    def rank0_broadcast(obj_list, src=0):
                        shared_decision.clear()
                        shared_decision.extend(obj_list)

                    with patch('torch.distributed.broadcast_object_list', side_effect=rank0_broadcast):
                        exp0 = mgr.create_experiment("exp", opt, experiment_id="fresh_run02")
                        self.assertTrue(os.path.isdir(exp0.root_dir))

                # Simulate Rank 1 receiving decision
                with patch('torch.distributed.get_rank', return_value=1):
                    def rank1_broadcast(obj_list, src=0):
                        obj_list[0] = shared_decision[0]

                    with patch('torch.distributed.broadcast_object_list', side_effect=rank1_broadcast):
                        exp1 = mgr.create_experiment("exp", opt, experiment_id="fresh_run02")
                        # Rank 1 successfully attaches to the created experiment directory without raising FileExistsError
                        self.assertEqual(exp1.root_dir, exp0.root_dir)


if __name__ == '__main__':
    unittest.main()
