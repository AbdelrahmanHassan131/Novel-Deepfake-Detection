"""
PhaseProfiler: Comprehensive, rank-safe performance and timing instrumentation.

Instruments:
    - Host/startup phases: startup, manifest_validation, loader_construction,
      pretrained_initialization, first_batch_latency, steady_data_wait,
      validation, metric_aggregation, checkpoint_export.
    - GPU execution phases: host_to_device_transfer, forward_backward, optimizer_update.

Design principles:
    - Uses CPU monotonic timing (time.monotonic()) for all host phases.
    - Uses sampled CUDA events for GPU work when CUDA is available.
    - Strictly avoids calling torch.cuda.synchronize() every step; bounds synchronization
      to profiling window or epoch boundaries.
    - Collects rank-local and cross-rank global metrics (images/sec = total_samples / slowest_rank_wall_time).
    - Queries nvidia-smi across ALL available GPUs, not just GPU 0.
"""

import os
import sys
import time
import subprocess
from contextlib import contextmanager
from typing import Dict, Any, List, Optional
from collections import defaultdict

import torch
import torch.distributed as dist


class PhaseProfiler:
    """Instruments training and validation phases with monotonic time and sampled CUDA events."""

    def __init__(self, rank: int = 0, world_size: int = 1, enabled: bool = True, device=None):
        self.rank = rank
        self.world_size = world_size
        self.enabled = enabled
        self.device = torch.device(device or 'cpu')
        self.use_cuda = self.device.type == 'cuda'

        # Cumulative elapsed times (in seconds)
        self.phase_times: Dict[str, float] = defaultdict(float)
        self.phase_counts: Dict[str, int] = defaultdict(int)
        self._active_starts: Dict[str, float] = {}

        # Sampled CUDA events for GPU operations
        self._cuda_events: List[Dict[str, Any]] = []
        self._cuda_active: Dict[str, torch.cuda.Event] = {}

        # Global counters
        self.total_samples_processed = 0
        self.total_microbatches = 0
        self.total_optimizer_steps = 0
        self.profiler_start_time = time.monotonic()

    @contextmanager
    def phase(self, name: str, sample_cuda: bool = False):
        """Context manager to measure host execution time of a phase."""
        if not self.enabled:
            yield
            return

        t0 = time.monotonic()
        cuda_start = None
        if sample_cuda and self.use_cuda and len(self._cuda_events) < 64:
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_start.record()

        try:
            yield
        finally:
            t1 = time.monotonic()
            elapsed = t1 - t0
            self.phase_times[name] += elapsed
            self.phase_counts[name] += 1

            if sample_cuda and self.use_cuda and cuda_start is not None:
                cuda_end = torch.cuda.Event(enable_timing=True)
                cuda_end.record()
                # Store events for bounded batch evaluation (max 100 stored to avoid memory leaks)
                if len(self._cuda_events) < 64:
                    self._cuda_events.append({'name': name, 'start': cuda_start, 'end': cuda_end})

    def start_phase(self, name: str):
        """Start a manual timing phase."""
        if self.enabled:
            self._active_starts[name] = time.monotonic()

    def end_phase(self, name: str) -> float:
        """End a manual timing phase and accumulate elapsed time."""
        if not self.enabled or name not in self._active_starts:
            return 0.0
        elapsed = time.monotonic() - self._active_starts.pop(name)
        self.phase_times[name] += elapsed
        self.phase_counts[name] += 1
        return elapsed

    def add_samples(self, count: int, is_optimizer_step: bool = False):
        """Record processed microbatch samples and step counter."""
        if self.enabled:
            self.total_samples_processed += count
            self.total_microbatches += 1
            if is_optimizer_step:
                self.total_optimizer_steps += 1

    def sync_cuda_events(self) -> Dict[str, float]:
        """Synchronize stored CUDA events at a bounded profiling boundary."""
        cuda_metrics = defaultdict(float)
        cuda_counts = defaultdict(int)
        if self.use_cuda and self._cuda_events:
            torch.cuda.synchronize()
            for item in self._cuda_events:
                ms = item['start'].elapsed_time(item['end'])
                sec = ms / 1000.0
                cuda_metrics[item['name']] += sec
                cuda_counts[item['name']] += 1
            self._cuda_events.clear()

        return {k: round(cuda_metrics[k] / max(1, cuda_counts[k]), 6) for k in cuda_metrics}

    def get_rank_summary(self, batch_size: int = 32, grad_accum_steps: int = 1) -> Dict[str, Any]:
        """Compile complete rank-local performance summary."""
        wall_time = time.monotonic() - self.profiler_start_time
        effective_global_batch = batch_size * self.world_size * grad_accum_steps

        # Check peak GPU memory
        peak_gpu_mem_mb = {}
        if self.use_cuda:
            peak_gpu_mem_mb[str(self.device)] = round(torch.cuda.max_memory_allocated(self.device) / (1024 * 1024), 2)

        avg_data_wait = (
            self.phase_times['steady_data_wait'] / max(1, self.phase_counts['steady_data_wait'])
            if 'steady_data_wait' in self.phase_times else 0.0
        )
        avg_fwd_bwd = (
            self.phase_times['forward_backward'] / max(1, self.phase_counts['forward_backward'])
            if 'forward_backward' in self.phase_times else 0.0
        )

        return {
            'rank': self.rank,
            'world_size': self.world_size,
            'cpu_count': os.cpu_count(),
            'per_rank_batch_size': batch_size,
            'grad_accum_steps': grad_accum_steps,
            'effective_global_batch': effective_global_batch,
            'total_samples_processed': self.total_samples_processed,
            'total_microbatches': self.total_microbatches,
            'total_optimizer_steps': self.total_optimizer_steps,
            'rank_wall_time_sec': round(wall_time, 3),
            'images_per_sec_rank': round(self.total_samples_processed / max(0.001, wall_time), 2),
            'phase_cumulative_seconds': {k: round(v, 4) for k, v in self.phase_times.items()},
            'phase_counts': dict(self.phase_counts),
            'average_steady_data_wait_sec': round(avg_data_wait, 5),
            'average_forward_backward_sec': round(avg_fwd_bwd, 5),
            'peak_gpu_memory_mb': peak_gpu_mem_mb,
        }

    def get_global_summary(self, batch_size: int = 32, grad_accum_steps: int = 1) -> Dict[str, Any]:
        """Aggregate timing across all ranks via clean DDP collectives."""
        local = self.get_rank_summary(batch_size, grad_accum_steps)

        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            device = self.device if dist.get_backend() == 'nccl' else 'cpu'

            # Gather slowest rank wall time and sum of processed samples
            stats_tensor = torch.tensor(
                [local['rank_wall_time_sec'], float(self.total_samples_processed)],
                dtype=torch.float64,
                device=device
            )
            # Find max wall time
            max_wall_tensor = stats_tensor[0].clone()
            dist.all_reduce(max_wall_tensor, op=dist.ReduceOp.MAX)
            slowest_wall_time = float(max_wall_tensor.item())

            # Sum total global samples
            total_samples_tensor = stats_tensor[1].clone()
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
            global_samples = int(total_samples_tensor.item())

            global_throughput = round(global_samples / max(0.001, slowest_wall_time), 2)
        else:
            slowest_wall_time = local['rank_wall_time_sec']
            global_samples = local['total_samples_processed']
            global_throughput = local['images_per_sec_rank']

        return {
            'world_size': self.world_size,
            'slowest_rank_wall_time_sec': round(slowest_wall_time, 3),
            'global_total_samples': global_samples,
            'global_images_per_sec': global_throughput,
            'effective_global_batch': local['effective_global_batch'],
            'per_rank_summary': local,
        }


def sample_all_gpus_nvidia_smi() -> Dict[str, Any]:
    """Query nvidia-smi across ALL available GPUs, not just GPU 0."""
    try:
        cmd = [
            'nvidia-smi',
            '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.used',
            '--format=csv,noheader,nounits'
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
        lines = [line.strip() for line in proc.stdout.strip().split('\n') if line.strip()]

        gpu_records = []
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 6:
                gpu_records.append({
                    'gpu_index': int(parts[0]),
                    'name': parts[1],
                    'gpu_utilization_pct': float(parts[2]),
                    'memory_utilization_pct': float(parts[3]),
                    'memory_total_mb': float(parts[4]),
                    'memory_used_mb': float(parts[5]),
                })
        return {
            'available': True,
            'gpu_count': len(gpu_records),
            'gpus': gpu_records,
        }
    except Exception as e:
        return {
            'available': False,
            'error': str(e),
            'gpus': [],
        }
