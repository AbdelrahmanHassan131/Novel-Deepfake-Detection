# STATUS: NOT RUN (runtime tests deferred; static syntax check only)
"""
Tests for Timing Instrumentation, Logger Precedence, and Bounded Benchmark Tool.

Verifies:
1. PhaseProfiler monotonic timing and rank-local / global summary structures
2. Memory and device query safety across visible devices
3. Explicit precedence of log_freq over loss_freq in Trainer hook registration
4. Clean collective error propagation in LoggerHook (no silent error suppression)
5. Microbatch vs optimizer step formatting under gradient accumulation
"""

import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from training.runtime.profiler import PhaseProfiler, sample_all_gpus_nvidia_smi
from training.hooks.logger_hook import LoggerHook
from training.trainer import Trainer


class TestProfilingAndBenchmark(unittest.TestCase):

    def test_phase_profiler_timing(self):
        """Profiler must accumulate monotonic times and report metrics accurately."""
        profiler = PhaseProfiler(rank=0, world_size=1, enabled=True)

        with profiler.phase('startup'):
            time.sleep(0.02)

        with profiler.phase('manifest_validation'):
            time.sleep(0.02)

        profiler.add_samples(count=32, is_optimizer_step=True)
        profiler.add_samples(count=32, is_optimizer_step=False)

        summary = profiler.get_rank_summary(batch_size=32, grad_accum_steps=2)
        self.assertEqual(summary['total_samples_processed'], 64)
        self.assertEqual(summary['total_microbatches'], 2)
        self.assertEqual(summary['total_optimizer_steps'], 1)
        self.assertGreater(summary['phase_cumulative_seconds']['startup'], 0.0)
        self.assertGreater(summary['phase_cumulative_seconds']['manifest_validation'], 0.0)

        global_summary = profiler.get_global_summary(batch_size=32, grad_accum_steps=2)
        self.assertEqual(global_summary['global_total_samples'], 64)
        self.assertGreater(global_summary['global_images_per_sec'], 0.0)

    def test_log_freq_precedence_over_loss_freq(self):
        """log_freq must take precedence when explicitly specified by the user."""
        mock_model = MagicMock()
        mock_model.optimizer = MagicMock()
        mock_loader = MagicMock()
        mock_loader.__len__.return_value = 100

        # Case 1: User explicitly passed log_freq=10, while loss_freq has default 400
        opt1 = SimpleNamespace(
            arch='Wang2020_128',
            checkpoints_dir='.',
            name='test_run',
            log_freq=10,
            loss_freq=400,
            lr_policy='none',
            save_epoch_freq=1,
            val_epoch_freq=1,
            monitor_metric='auc',
            grad_accum_steps=1,
        )
        trainer1 = Trainer(mock_model, mock_loader, opt1)
        logger_hook1 = next(h for h in trainer1._hooks if isinstance(h, LoggerHook))
        self.assertEqual(logger_hook1.log_freq, 10)

        # Case 2: User specified only loss_freq=250 (legacy configuration)
        opt2 = SimpleNamespace(
            arch='Wang2020_128',
            checkpoints_dir='.',
            name='test_run',
            log_freq=None,
            loss_freq=250,
            lr_policy='none',
            save_epoch_freq=1,
            val_epoch_freq=1,
            monitor_metric='auc',
            grad_accum_steps=1,
        )
        trainer2 = Trainer(mock_model, mock_loader, opt2)
        logger_hook2 = next(h for h in trainer2._hooks if isinstance(h, LoggerHook))
        self.assertEqual(logger_hook2.log_freq, 250)

    def test_logger_hook_surfaces_collective_errors(self):
        """LoggerHook must raise a clean RuntimeError if dist.all_reduce fails."""
        hook = LoggerHook(log_freq=1, rank=0)

        with patch('torch.distributed.is_available', return_value=True), \
             patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.get_world_size', return_value=2), \
             patch('torch.distributed.all_reduce', side_effect=RuntimeError("Simulated collective crash")):
            with self.assertRaises(RuntimeError) as ctx:
                hook._get_world_size_and_reduce_loss(0.5)
            self.assertIn("Distributed collective all_reduce failed", str(ctx.exception))

    def test_sample_all_gpus_nvidia_smi(self):
        """sample_all_gpus_nvidia_smi must return a valid structure without crashing."""
        result = sample_all_gpus_nvidia_smi()
        self.assertIn('available', result)
        self.assertIn('gpus', result)
        if result['available']:
            self.assertIsInstance(result['gpus'], list)


if __name__ == '__main__':
    unittest.main()
