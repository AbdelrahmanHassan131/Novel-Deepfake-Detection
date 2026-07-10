"""
Performance Metrics.

Measures inference speed, FLOPs, and memory footprint.

Uses CUDA synchronization for accurate GPU timing and a standard
PyTorch FLOPs library (``thop`` or ``fvcore``) for FLOPs counting.

Usage::

    from evaluation.metrics import PerformanceProfiler

    profiler = PerformanceProfiler(model, device)
    perf = profiler.profile(dataloader, input_shape=(1, 3, 224, 224))
"""

import os
import time
import torch
import numpy as np


class PerformanceProfiler:
    """
    Measures inference speed, FLOPs, and memory usage.

    Args:
        model: A ``BaseModel`` subclass instance.
        device: The torch device.
    """

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or model.device

    def profile(self, dataloader=None, input_shape=None,
                warmup_batches=3, max_batches=50):
        """
        Run all performance measurements.

        Args:
            dataloader: DataLoader for latency measurement.
            input_shape: Tuple for FLOPs estimation (e.g.
                ``(1, 3, 224, 224)``).
            warmup_batches: Number of warmup batches before timing.
            max_batches: Maximum batches to time.

        Returns:
            dict with all performance metrics.
        """
        result = {}

        # Inference timing
        if dataloader is not None:
            result.update(
                self._measure_latency(dataloader, warmup_batches,
                                      max_batches)
            )

        # FLOPs
        if input_shape is not None:
            result.update(self._estimate_flops(input_shape))

        # Memory footprint
        result.update(self._measure_memory())

        return result

    # ------------------------------------------------------------------
    # Inference timing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _measure_latency(self, dataloader, warmup_batches, max_batches):
        """Measure average latency per image and per batch."""
        self.model.eval()
        is_cuda = str(self.device).startswith('cuda')

        # Warmup
        for i, batch in enumerate(dataloader):
            if i >= warmup_batches:
                break
            self.model.set_input(batch)
            self.model.forward()
            if is_cuda:
                torch.cuda.synchronize(self.device)

        # Timed runs
        batch_times = []
        batch_sizes = []
        total_samples = 0

        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            self.model.set_input(batch)

            if is_cuda:
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()

            self.model.forward()

            if is_cuda:
                torch.cuda.synchronize(self.device)
            end = time.perf_counter()

            elapsed = end - start
            batch_times.append(elapsed)

            # Determine batch size from labels
            bs = len(self.model.label)
            batch_sizes.append(bs)
            total_samples += bs

        if not batch_times:
            return {
                'avg_latency_per_image_ms': 0.0,
                'avg_latency_per_batch_ms': 0.0,
                'throughput_images_per_sec': 0.0,
            }

        total_time = sum(batch_times)
        avg_batch_ms = float(np.mean(batch_times)) * 1000
        avg_image_ms = (total_time / total_samples) * 1000
        throughput = total_samples / total_time

        return {
            'avg_latency_per_image_ms': round(avg_image_ms, 4),
            'avg_latency_per_batch_ms': round(avg_batch_ms, 4),
            'throughput_images_per_sec': round(throughput, 2),
            'total_batches_timed': len(batch_times),
            'total_samples_timed': total_samples,
        }

    # ------------------------------------------------------------------
    # FLOPs estimation
    # ------------------------------------------------------------------

    def _estimate_flops(self, input_shape=None):
        """
        Estimate FLOPs using ``thop`` (preferred) or ``fvcore``.

        Supports both single-input models and multi-input fusion/MHA models.
        Falls back gracefully if neither library is available or supported.
        """
        raw_model = getattr(self.model, 'model', self.model)
        if hasattr(raw_model, 'module'):
            raw_model = raw_model.module

        # Determine dummy inputs based on input_shape or model architecture
        model_name = getattr(self.model, 'name', lambda: '')() if callable(getattr(self.model, 'name', None)) else getattr(self.model, 'arch', '')
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 and isinstance(input_shape[0], (list, tuple)):
            dummy_inputs = tuple(torch.randn(*shape).to(self.device) for shape in input_shape)
        elif model_name in ('Fusion_128', 'MHA_128'):
            dummy_inputs = (torch.randn(1, 128).to(self.device), torch.randn(1, 128).to(self.device))
        else:
            if input_shape is None:
                input_shape = (1, 3, 256, 256)
            dummy_inputs = (torch.randn(*input_shape).to(self.device),)

        # Try thop
        try:
            from thop import profile as thop_profile  # type: ignore
            from thop import clearing  # type: ignore
            flops, params = thop_profile(raw_model, inputs=dummy_inputs,
                                         verbose=False)
            clearing(raw_model)
            return {
                'flops': int(flops),
                'macs': int(flops / 2),
                'flops_source': 'thop',
            }
        except ImportError:
            pass
        except Exception as e:
            try:
                from thop import clearing  # type: ignore
                clearing(raw_model)
            except Exception:
                pass

        # Try fvcore
        try:
            from fvcore.nn import FlopCountAnalysis  # type: ignore
            fca = FlopCountAnalysis(raw_model, dummy_inputs)
            total_flops = fca.total()
            return {
                'flops': int(total_flops),
                'macs': int(total_flops / 2),
                'flops_source': 'fvcore',
            }
        except ImportError:
            pass
        except Exception:
            pass

        # Try native PyTorch FlopCounterMode (PyTorch 2.1+)
        try:
            from torch.utils.flop_counter import FlopCounterMode
            with FlopCounterMode(display=False) as flop_counter:
                raw_model(*dummy_inputs)
            total_flops = flop_counter.get_total_flops()
            return {
                'flops': int(total_flops),
                'macs': int(total_flops / 2),
                'flops_source': 'torch.utils.flop_counter',
            }
        except Exception:
            pass

        return {
            'flops': None,
            'macs': None,
            'flops_source': 'unavailable (install thop or fvcore)',
        }

    # ------------------------------------------------------------------
    # Memory footprint
    # ------------------------------------------------------------------

    def _measure_memory(self):
        """Measure model parameter count and GPU memory."""
        raw_model = self.model.model
        if hasattr(raw_model, 'module'):
            raw_model = raw_model.module

        # Parameter counts
        total_params = sum(p.numel() for p in raw_model.parameters())
        trainable_params = sum(
            p.numel() for p in raw_model.parameters() if p.requires_grad
        )

        result = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': sum(
                p.numel() * p.element_size()
                for p in raw_model.parameters()
            ) / (1024 * 1024),
            'checkpoint_size_mb': (
                os.path.getsize(self.model.opt.checkpoints_dir)
                if hasattr(self.model, 'opt')
                and hasattr(self.model.opt, 'checkpoints_dir')
                and os.path.isfile(self.model.opt.checkpoints_dir)
                else None
            ),
        }

        # GPU memory (only if on CUDA)
        if str(self.device).startswith('cuda'):
            device_idx = (int(str(self.device).split(':')[1])
                          if ':' in str(self.device) else 0)
            try:
                result['gpu_allocated_mb'] = round(
                    torch.cuda.memory_allocated(device_idx) / (1024 * 1024), 2
                )
                result['gpu_reserved_mb'] = round(
                    torch.cuda.memory_reserved(device_idx) / (1024 * 1024), 2
                )
                result['gpu_peak_mb'] = round(
                    torch.cuda.max_memory_allocated(device_idx) / (1024 * 1024), 2
                )
            except Exception:
                pass

        return result
