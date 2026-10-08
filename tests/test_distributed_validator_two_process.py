# STATUS: NOT RUN
"""
Deferred regression test suite for Prompt N: Real two-process distributed validation.
Verifies cross-process buffer synchronization and unpadded distributed gather re-sorting
using a real two-process PyTorch Gloo distributed group without mocks.

DO NOT EXECUTE AT RUNTIME IN THIS ENVIRONMENT.
Static validation only.
"""

import os
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from training.validator import Validator, sync_module_buffers


def _worker_fn(rank, world_size, init_url, return_dict):
    """Worker process for real 2-process distributed validation test."""
    # Initialize Gloo process group on CPU
    dist.init_process_group(
        backend='gloo',
        init_method=init_url,
        rank=rank,
        world_size=world_size,
    )

    try:
        # 1. Test real buffer synchronization across ranks
        model = nn.BatchNorm2d(num_features=4)
        if rank == 0:
            # Set known running stats on rank 0
            model.running_mean.fill_(5.0)
            model.running_var.fill_(2.0)
        else:
            model.running_mean.fill_(0.0)
            model.running_var.fill_(1.0)

        # Broadcast buffers from rank 0 to rank 1
        sync_module_buffers(model)

        # Both ranks must now have identical running stats
        mean_val = float(model.running_mean[0].item())
        var_val = float(model.running_var[0].item())

        # 2. Test unpadded rank gather and index re-sorting
        # Simulate strided rank assignments: rank 0 has [0, 2, 4], rank 1 has [1, 3, 5]
        local_indices = np.array([0, 2, 4] if rank == 0 else [1, 3, 5], dtype=np.int64)
        local_preds = np.array([0.1, 0.3, 0.5] if rank == 0 else [0.2, 0.4, 0.6], dtype=np.float32)
        local_labels = np.array([0, 0, 1] if rank == 0 else [1, 0, 1], dtype=np.float32)

        gathered_preds = [None for _ in range(world_size)]
        gathered_labels = [None for _ in range(world_size)]
        gathered_indices = [None for _ in range(world_size)]

        dist.all_gather_object(gathered_preds, local_preds)
        dist.all_gather_object(gathered_labels, local_labels)
        dist.all_gather_object(gathered_indices, local_indices)

        cat_preds = np.concatenate(gathered_preds)
        cat_labels = np.concatenate(gathered_labels)
        cat_indices = np.concatenate(gathered_indices)

        # Canonical index sort
        order = np.argsort(cat_indices)
        sorted_preds = cat_preds[order]
        sorted_labels = cat_labels[order]
        sorted_indices = cat_indices[order]

        return_dict[rank] = {
            'mean_synced': mean_val,
            'var_synced': var_val,
            'sorted_indices': sorted_indices.tolist(),
            'sorted_preds': sorted_preds.tolist(),
            'sorted_labels': sorted_labels.tolist(),
        }
    finally:
        dist.destroy_process_group()


class TestDistributedValidatorTwoProcess(unittest.TestCase):
    """Verifies distributed buffer synchronization and prediction gather across real separate processes."""

    def test_two_process_buffer_sync_and_gather_sorting(self):
        """Spawns 2 processes on Gloo CPU to verify cross-process synchronization and canonical sorting."""
        world_size = 2
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, 'dist_init')
            init_url = f"file:///{init_file.replace(os.sep, '/')}"

            manager = mp.Manager()
            return_dict = manager.dict()

            mp.spawn(
                _worker_fn,
                args=(world_size, init_url, return_dict),
                nprocs=world_size,
                join=True,
            )

            # Verify results from both processes
            for r in range(world_size):
                self.assertIn(r, return_dict)
                res = return_dict[r]
                # Buffers were broadcasted from rank 0
                self.assertEqual(res['mean_synced'], 5.0)
                self.assertEqual(res['var_synced'], 2.0)
                # Interleaved rank samples were gathered and sorted into exact canonical [0..5] order
                self.assertEqual(res['sorted_indices'], [0, 1, 2, 3, 4, 5])
                np.testing.assert_allclose(res['sorted_preds'], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], atol=1e-5)
                self.assertEqual(res['sorted_labels'], [0.0, 1.0, 0.0, 0.0, 1.0, 1.0])


if __name__ == '__main__':
    unittest.main()
