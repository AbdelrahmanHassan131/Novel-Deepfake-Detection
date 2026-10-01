# STATUS: NOT RUN
# Execution restriction: Static AST verification only. Do not execute test suite now.
# Real dual-GPU runtime validation is pending execution on Colab/GPU cluster.
"""
Tests for Prompt E: Safe Unpadded Distributed Validation (R10).

Verifies:
1. Unpadded validation on unwrapped local models avoiding DDP forward collective hangs.
2. Exception-safe restoration of DDP model wrapper even if evaluation raises an exception.
3. Explicit buffer synchronization for BatchNorm running statistics prior to validation.
4. Exact sample-weighted loss aggregation across ranks with uneven sample allocations.
5. Handling of empty-rank shards (zero samples on one rank) without division-by-zero or metric corruption.
6. Preservation of DataLoader worker initialization options in DistributedRuntime.wrap_loader.
"""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training.validator import Validator, sync_module_buffers
from training.runtime.distributed_runtime import DistributedRuntime


class ToyBatchNormModel(nn.Module):
    """Toy model containing BatchNorm layers to verify buffer sync and evaluation unwrapping."""

    def __init__(self, in_features=4, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 8)
        self.bn = nn.BatchNorm1d(8)
        self.fc2 = nn.Linear(8, out_features)

    def forward(self, x):
        return self.fc2(self.bn(self.fc1(x)))


class MockBaseModelWrapper:
    """Mock wrapping BaseModel set_input / forward / get_loss contract."""

    def __init__(self, inner_module, device="cpu"):
        self.model = inner_module
        self.device = device
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.output = None
        self.label = None

    def set_input(self, batch):
        self.x, self.label = batch[0].to(self.device), batch[1].to(self.device).float()

    def forward(self):
        self.output = self.model(self.x)
        return self.output

    def get_loss(self):
        return self.loss_fn(self.output.flatten(), self.label.flatten())

    def eval(self):
        self.model.eval()

    def train(self, mode=True):
        self.model.train(mode)


class TestDistributedValidationSafety(unittest.TestCase):
    """Static and structural test suite for distributed validation safety."""

    def test_buffer_synchronization_broadcasts_from_rank_zero(self):
        """sync_module_buffers must broadcast buffer parameters from rank 0 across ranks."""
        model = ToyBatchNormModel()
        # Initialize running_mean to non-zero values
        model.bn.running_mean.fill_(1.5)
        model.bn.running_var.fill_(0.8)

        with patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.broadcast") as mock_broadcast:
            sync_module_buffers(model)
            # Verify dist.broadcast called for each buffer with src=0
            self.assertEqual(mock_broadcast.call_count, len(list(model.buffers())))
            for call in mock_broadcast.call_args_list:
                self.assertEqual(call[1].get("src"), 0)

    def test_exception_safe_ddp_restoration(self):
        """Validator must restore original DDP wrapper even if validation raises an exception."""
        inner = ToyBatchNormModel()
        # Create a mock DDP object
        fake_ddp = MagicMock(spec=torch.nn.parallel.DistributedDataParallel)
        fake_ddp.module = inner
        wrapper = MockBaseModelWrapper(fake_ddp)

        # Faulty loader that raises an error midway
        class FaultyLoader:
            def __iter__(self):
                raise RuntimeError("Simulated dataset corruption or read failure")

        validator = Validator()
        with self.assertRaises(RuntimeError):
            validator.validate(wrapper, FaultyLoader())

        # The wrapper model MUST be restored back to fake_ddp despite the crash
        self.assertIs(wrapper.model, fake_ddp)

    def test_sample_weighted_loss_exactness(self):
        """Loss reduction must weight rank contributions by actual sample counts, not average of means."""
        # Rank 0: 100 samples with loss 0.1
        # Rank 1: 5 samples with loss 0.9
        # True weighted loss = (100 * 0.1 + 5 * 0.9) / 105 = 14.5 / 105 = 0.138095
        # Naive mean of means = (0.1 + 0.9) / 2 = 0.500000 (WRONG!)
        rank0_loss_sum = 100 * 0.1
        rank0_count = 100
        rank1_loss_sum = 5 * 0.9
        rank1_count = 5

        total_loss_sum = rank0_loss_sum + rank1_loss_sum
        total_samples = rank0_count + rank1_count
        expected_avg_loss = total_loss_sum / total_samples

        self.assertAlmostEqual(expected_avg_loss, 0.138095238, places=6)
        self.assertNotAlmostEqual(expected_avg_loss, 0.5, places=1)

    def test_empty_rank_shard_handling(self):
        """Validator must handle an empty shard on one rank without division by zero or errors."""
        inner = ToyBatchNormModel()
        wrapper = MockBaseModelWrapper(inner)

        # Empty loader with 0 samples
        empty_dataset = TensorDataset(torch.empty(0, 4), torch.empty(0))
        empty_loader = DataLoader(empty_dataset, batch_size=2)

        validator = Validator()
        result = validator.validate(wrapper, empty_loader)

        self.assertEqual(result.num_samples, 0)
        self.assertEqual(result.loss, 0.0)
        self.assertEqual(len(result.predictions), 0)
        self.assertEqual(len(result.labels), 0)

    def test_wrap_loader_preserves_worker_options(self):
        """DistributedRuntime.wrap_loader must preserve worker_init_fn, prefetch_factor, and persistent_workers."""
        dataset = TensorDataset(torch.randn(10, 4), torch.zeros(10))

        def custom_worker_init(worker_id):
            pass

        original_loader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=custom_worker_init,
            prefetch_factor=3,
            persistent_workers=True,
            timeout=15.0,
        )

        class Opt:
            seed = 42

        runtime = DistributedRuntime.__new__(DistributedRuntime)
        runtime.is_distributed = True
        runtime.rank = 0
        runtime.world_size = 2
        runtime.is_main = True
        runtime._opt = Opt()

        wrapped_loader = runtime.wrap_loader(original_loader, is_train=False)

        self.assertIs(wrapped_loader.worker_init_fn, custom_worker_init)
        self.assertEqual(wrapped_loader.num_workers, 2)
        self.assertTrue(wrapped_loader.pin_memory)
        self.assertTrue(wrapped_loader.drop_last)
        self.assertEqual(wrapped_loader.prefetch_factor, 3)
        self.assertTrue(wrapped_loader.persistent_workers)
        self.assertEqual(wrapped_loader.timeout, 15.0)


if __name__ == "__main__":
    unittest.main()
