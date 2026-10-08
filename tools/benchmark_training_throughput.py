#!/usr/bin/env python
"""
Bounded Training Throughput Benchmark Tool.

Measures realistic training and data feeding throughput on 1 or 2 GPUs:
    - Warmup: exactly ~20 microbatches (excluded from steady-state measurements)
    - Measured: exactly ~100 microbatches
    - Bounded validation: optional ~500 images
    - Measures data loading wait and model compute time separately
    - Records images/sec, rank-local times, slowest rank wall time
    - Reports memory and utilization across ALL GPUs via multi-GPU nvidia-smi helper
    - Exits cleanly on all ranks after the requested bound

SAFETY CONSTRAINTS:
    - Explicit opt-in tool; never modifies production checkpoints
    - Does NOT count as scientific training; restores no production state
    - Bounded microbatches prevent runaway execution
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.runtime.profiler import PhaseProfiler, sample_all_gpus_nvidia_smi
from training.runtime.distributed_runtime import DistributedRuntime
from models import build_model
from data.loaders.dataloader_factory import create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run bounded training throughput benchmark without affecting production checkpoints."
    )
    parser.add_argument('--arch', type=str, default='Wang2020_128',
                        help="Model architecture to benchmark.")
    parser.add_argument('--manifest', type=str, required=True,
                        help="Path to training manifest CSV.")
    parser.add_argument('--val_manifest', type=str, default=None,
                        help="Optional validation manifest CSV for bounded validation timing.")
    parser.add_argument('--dataroot', type=str, default=None,
                        help="Root directory for images.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size per GPU (candidate: 16, 32, 64).")
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help="Gradient accumulation steps (e.g. 1 or 2).")
    parser.add_argument('--num_workers', type=int, default=2,
                        help="Number of dataloader worker processes per rank.")
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help="DataLoader prefetch factor.")
    parser.add_argument('--persistent_workers', action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to retain DataLoader worker processes between epochs.")
    parser.add_argument('--pin_memory', action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to pin CPU memory for faster host-to-device transfer.")
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=True,
                        help="Enable Automatic Mixed Precision (AMP).")
    parser.add_argument('--channels_last', action='store_true', default=False,
                        help="Opt-in: convert inputs and model weights to memory_format=torch.channels_last.")
    parser.add_argument('--warmup_microbatches', type=int, default=20,
                        help="Warmup microbatches to discard from timing (default: 20).")
    parser.add_argument('--measure_microbatches', type=int, default=100,
                        help="Steady-state microbatches to measure (default: 100).")
    parser.add_argument('--val_samples', type=int, default=500,
                        help="Number of validation images to test (default: 500; 0 to skip).")
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help="Comma-separated GPU IDs or torchrun environment.")
    parser.add_argument('--output', type=str, default='benchmark_report.json',
                        help="Path to save machine-readable JSON report.")
    parser.add_argument('--sample_nvidia_smi', action=argparse.BooleanOptionalAction, default=True,
                        help="Query nvidia-smi across all visible GPUs during benchmark.")
    parser.add_argument('--aug_recipe', choices=['legacy', 'rgb_v1', 'custom'], default='rgb_v1')
    parser.add_argument('--fine_tune_policy', choices=['full', 'head_only', 'layer4_and_head'], default='layer4_and_head')
    parser.add_argument('--bn_policy', choices=['train', 'frozen'], default='frozen')
    parser.add_argument('--rgb_head_type', choices=['128d', 'linear'], default='128d')
    parser.add_argument('--rgb_dropout', type=float, default=0.5)
    parser.add_argument('--backbone_lr_mult', type=float, default=0.1)
    parser.add_argument('--crop_policy', choices=['scale_and_crop', 'random_resized_crop', 'patch_crop'], default='scale_and_crop')
    parser.add_argument('--amp_dtype', choices=['fp16', 'bf16'], default='fp16')
    return parser.parse_args()


def run_benchmark(args):
    """Use the production train step, including AMP/DDP and sample weighting."""
    import copy
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    from data.manifest import read_manifest
    from data.transforms.augmentations import resolve_augmentation_recipe
    from training.base_trainer import BaseTrainer
    from training.validator import Validator

    if min(args.batch_size, args.grad_accum_steps, args.measure_microbatches) < 1 or args.warmup_microbatches < 0:
        raise ValueError('Batch/accumulation/measurement counts must be positive; warmup may be zero')
    if args.val_samples < 0:
        raise ValueError('val_samples must be nonnegative')
    opt = SimpleNamespace(
        arch=args.arch, dataroot=args.dataroot or str(Path(args.manifest).parent),
        manifest=args.manifest, manifest_split='train', batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps, num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor, persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory, use_amp=args.use_amp, amp_dtype=args.amp_dtype,
        channels_last=args.channels_last, isTrain=True, serial_batches=False,
        class_bal=False, no_crop=False, no_resize=False, no_flip=False,
        cropSize=224, loadSize=256, rz_interp=['bilinear'],
        aug_recipe=args.aug_recipe, crop_policy=args.crop_policy,
        fine_tune_policy=args.fine_tune_policy, bn_policy=args.bn_policy,
        rgb_head_type=args.rgb_head_type, rgb_dropout=args.rgb_dropout,
        backbone_lr_mult=args.backbone_lr_mult, val_precision='fp32',
        classes=['real', 'fake'], mode='binary',
        gpu_ids=[int(x) for x in args.gpu_ids.split(',') if x.strip() and x.strip() != '-1'],
        pretrained=False, init_gain=0.02, optim='adam', lr=0.0001, beta1=0.9,
        weight_decay=0.0, embed_dim=128, score_sign=1.0, continue_train=False,
        seed=42, eligible_sources=None, name='',
    )
    opt = resolve_augmentation_recipe(opt)
    runtime = DistributedRuntime(opt)
    device = runtime.device
    try:
        with tempfile.TemporaryDirectory(prefix='rgb_benchmark_') as tmp:
            opt.checkpoints_dir = tmp
            loader = create_dataloader(opt)
            model = build_model(opt)
            trainer = BaseTrainer(model, loader, opt, runtime=runtime)
            loader = trainer.train_loader
            if not len(loader):
                raise ValueError('Training loader is empty')
            model.train()
            iterator = iter(loader)

            def next_batch():
                nonlocal iterator
                try:
                    return next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    return next(iterator)

            for _ in range(args.warmup_microbatches):
                trainer.train_step(next_batch(), is_accum_end=True, loss_weight=1.0)
            model.optimizer.zero_grad(set_to_none=True)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            runtime.barrier()
            trainer.profiler = PhaseProfiler(rank=runtime.rank, world_size=runtime.world_size, device=device)
            measured_samples = 0
            smi_future = None
            with ThreadPoolExecutor(max_workers=1) as sampler_pool:
                t0 = time.monotonic()
                step = 0
                while step < args.measure_microbatches:
                    count = min(args.grad_accum_steps, args.measure_microbatches - step)
                    # Buffer only one accumulation window to compute exact sample weights,
                    # including small dataset tails. This buffering is disclosed in report.
                    with trainer.profiler.phase('steady_data_wait'):
                        window = [next_batch() for _ in range(count)]
                    sizes = [int(batch[-1].shape[0]) for batch in window]
                    for j, (batch, size) in enumerate(zip(window, sizes)):
                        trainer.train_step(batch, is_accum_end=j == count - 1,
                                           accum_steps=count, loss_weight=size / sum(sizes))
                        measured_samples += size
                        step += 1
                        if step == max(1, args.measure_microbatches // 2) and runtime.is_main and args.sample_nvidia_smi:
                            smi_future = sampler_pool.submit(sample_all_gpus_nvidia_smi)
                    del window
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                elapsed = time.monotonic() - t0
                cuda_times = trainer.profiler.sync_cuda_events()
                local_report = trainer.profiler.get_rank_summary(args.batch_size, args.grad_accum_steps)
                local_report['sampled_cuda_seconds_per_call'] = cuda_times
                local_report['measured_samples'] = measured_samples
                local_report['measured_wall_seconds'] = elapsed
                reports = [local_report]
                if runtime.is_distributed:
                    reports = [None] * runtime.world_size
                    dist.all_gather_object(reports, local_report)
                global_count = sum(r['measured_samples'] for r in reports)
                slowest = max(r['measured_wall_seconds'] for r in reports)
                smi = smi_future.result() if smi_future else {'available': False, 'reason': 'not sampled on this rank'}

            val_report = None
            if args.val_samples and args.val_manifest:
                val_opt = copy.copy(opt)
                val_opt.manifest = args.val_manifest
                val_opt.manifest_split = 'dev'
                val_opt.isTrain = False
                val_opt.serial_batches = True
                # Bound paths/images before distributed sampling (exact global cap).
                rows = read_manifest(args.val_manifest, opt.dataroot, 'dev', check_files=False)
                val_opt._manifest_records = rows[:args.val_samples]
                val_loader = runtime.wrap_loader(create_dataloader(val_opt), is_train=False)
                del val_opt._manifest_records
                start = time.monotonic()
                result = Validator().validate(model, val_loader)
                elapsed_val = time.monotonic() - start
                if runtime.is_distributed:
                    duration = torch.tensor(elapsed_val, device=device)
                    dist.all_reduce(duration, op=dist.ReduceOp.MAX)
                    elapsed_val = duration.item()
                val_report = dict(samples_evaluated=result.num_samples, val_wall_time_sec=elapsed_val,
                                  val_images_per_sec=result.num_samples / max(elapsed_val, 1e-9))
            report = dict(status='benchmark_completed', world_size=runtime.world_size,
                          recipe={k: v for k, v in vars(opt).items() if not k.startswith('_')},
                          global_total_samples=global_count, slowest_rank_wall_time_sec=slowest,
                          global_images_per_sec=global_count / max(slowest, 1e-9),
                          effective_global_batch=args.batch_size * runtime.world_size * args.grad_accum_steps,
                          per_rank=reports, nvidia_smi=smi, bounded_validation=val_report,
                          limitations='Random initialization; one accumulation window buffered; CUDA timing '
                          'samples first 64 phase calls after warmup. One utilization snapshot is not duty cycle.')
            if runtime.is_main:
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(report, indent=2), encoding='utf-8')
                print(f"Benchmark: {report['global_images_per_sec']:.1f} images/s; report: {output}", flush=True)
            return report
    finally:
        runtime.cleanup()


if __name__ == '__main__':
    run_benchmark(parse_args())
