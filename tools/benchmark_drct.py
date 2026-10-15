#!/usr/bin/env python
"""
DRCT-128 Throughput Benchmark and Colab Observability Tool.

Measures realistic training throughput and resource utilization for DRCT_128:
    - Warmup microbatches (discarded from steady-state measurements)
    - Steady-state measured microbatches
    - Separate measurement of data wait time and compute step time
    - Peak GPU memory allocation (torch.cuda.max_memory_allocated)
    - nvidia-smi GPU query integration across visible GPUs
    - Sweep mode: tests feasible combinations of batch sizes and worker counts
    - Contrastive pair pool accounting: distinguishes rank-local batch size from
      accumulated gradient batch size.

SAFETY & CONSTRAINTS:
    - Bounded microbatches prevent runaway execution.
    - Gracefully catches CUDA OOM in sweep mode without crashing the process.
    - Supports --synthetic mode for offline CPU/CI testing without real dataset files.
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.runtime.profiler import sample_all_gpus_nvidia_smi
from models.drct.trainer import DRCT_128Trainer
from models.drct.backbone import DummyBackbone


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run bounded DRCT-128 throughput and observability benchmark."
    )
    parser.add_argument('--arch', type=str, default='DRCT_128',
                        help="Model architecture (default: DRCT_128).")
    parser.add_argument('--drct_backbone', type=str, default='clip_vit_b16',
                        help="Backbone name: clip_vit_b16 or dummy (default: clip_vit_b16).")
    parser.add_argument('--manifest', type=str, default=None,
                        help="Path to training manifest CSV or JSON.")
    parser.add_argument('--dataroot', type=str, default=None,
                        help="Root directory for dataset.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size per GPU (default: 32).")
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help="Gradient accumulation steps (default: 1).")
    parser.add_argument('--num_workers', type=int, default=2,
                        help="Number of DataLoader workers per rank (default: 2).")
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help="DataLoader prefetch factor (default: 2).")
    parser.add_argument('--persistent_workers', action=argparse.BooleanOptionalAction, default=True,
                        help="Retain DataLoader workers between epochs (default: True).")
    parser.add_argument('--pin_memory', action=argparse.BooleanOptionalAction, default=True,
                        help="Pin CPU memory for faster GPU transfer (default: True).")
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=True,
                        help="Enable Automatic Mixed Precision (default: True).")
    parser.add_argument('--amp_dtype', choices=['fp16', 'bf16'], default='fp16',
                        help="AMP precision type (default: fp16).")
    parser.add_argument('--warmup_steps', type=int, default=5,
                        help="Warmup steps to discard from steady-state timing (default: 5).")
    parser.add_argument('--measure_steps', type=int, default=20,
                        help="Steady-state steps to measure (default: 20).")
    parser.add_argument('--drct_alpha', type=float, default=0.3,
                        help="Weight for contrastive loss (default: 0.3).")
    parser.add_argument('--drct_pos_margin', type=float, default=0.0)
    parser.add_argument('--drct_neg_margin', type=float, default=1.0)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=False,
                        help="Load pretrained backbone weights (default: False).")
    parser.add_argument('--drct_allow_download', '--allow_download', dest='drct_allow_download',
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Allow downloading pretrained CLIP weights.")
    parser.add_argument('--drct_backbone_weights', '--backbone_weights', dest='drct_backbone_weights',
                        type=str, default=None,
                        help="Path to local cached pretrained CLIP weights.")
    parser.add_argument('--output', type=str, default='drct_benchmark_report.json',
                        help="Output path for JSON benchmark report.")
    parser.add_argument('--sweep', action='store_true', default=False,
                        help="Run short sweep over batch sizes and worker counts.")
    parser.add_argument('--sweep_batch_sizes', type=str, default='16,32,64',
                        help="Comma-separated batch sizes for sweep (default: 16,32,64).")
    parser.add_argument('--sweep_workers', type=str, default='0,2,4',
                        help="Comma-separated worker counts for sweep (default: 0,2,4).")
    parser.add_argument('--synthetic', action='store_true', default=False,
                        help="Use synthetic in-memory data instead of loading dataset files.")
    parser.add_argument('--device', type=str, default=None,
                        help="Target device (cuda or cpu; defaults to cuda if available).")
    return parser.parse_args(args)


class SyntheticDataLoader:
    """In-memory synthetic batch generator for DRCT benchmarking."""

    def __init__(self, batch_size: int, steps: int = 100, image_size: int = 224):
        self.batch_size = batch_size
        self.steps = steps
        self.image_size = image_size

    def __iter__(self):
        for _ in range(self.steps):
            # Equal mix of real (0) and fake (1)
            half = self.batch_size // 2
            other = self.batch_size - half
            labels = torch.cat([torch.zeros(half), torch.ones(other)], dim=0).float()
            # Random shuffle
            perm = torch.randperm(self.batch_size)
            labels = labels[perm]
            images = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
            yield (images, labels)

    def __len__(self):
        return self.steps


def run_single_benchmark(
    opt: SimpleNamespace,
    device: torch.device,
    warmup_steps: int = 5,
    measure_steps: int = 20,
    synthetic: bool = False,
) -> Dict[str, Any]:
    """Execute a single bounded benchmark pass."""
    batch_size = opt.batch_size
    num_workers = opt.num_workers
    use_cuda = device.type == 'cuda'

    # 1. Model Setup
    gpu_ids = [device.index] if (use_cuda and device.index is not None) else ([0] if use_cuda else [])
    model_opt = SimpleNamespace(
        arch='DRCT_128',
        drct_backbone=opt.drct_backbone,
        embed_dim=opt.embed_dim,
        drct_alpha=opt.drct_alpha,
        drct_pos_margin=opt.drct_pos_margin,
        drct_neg_margin=opt.drct_neg_margin,
        drct_projection_activation=getattr(opt, 'drct_projection_activation', None),
        drct_projection_dropout=getattr(opt, 'drct_projection_dropout', 0.0),
        pretrained=getattr(opt, 'pretrained', False),
        drct_allow_download=getattr(opt, 'drct_allow_download', False),
        drct_backbone_weights=getattr(opt, 'drct_backbone_weights', None),
        lr=getattr(opt, 'lr', 1e-4),
        optim='adam',
        beta1=0.9,
        weight_decay=0.0,
        isTrain=True,
        gpu_ids=gpu_ids,
    )

    if opt.drct_backbone == 'dummy':
        model_opt._injected_encoder = DummyBackbone(in_channels=3, out_features=768)
        model_opt._injected_backbone_dim = 768
    model = DRCT_128Trainer(model_opt)

    model.train(True)

    # 2. Loader Setup
    if synthetic:
        loader = SyntheticDataLoader(batch_size=batch_size, steps=warmup_steps + measure_steps + 10)
    else:
        if not getattr(opt, 'manifest', None):
            raise ValueError("Manifest path must be provided when synthetic=False.")
        from data.loaders.dataloader_factory import create_dataloader
        loader_opt = SimpleNamespace(
            arch='DRCT_128',
            mode='binary',
            manifest=opt.manifest,
            drct_manifest_path=opt.manifest,
            dataroot=opt.dataroot or str(Path(opt.manifest).parent),
            manifest_split='train',
            batch_size=batch_size,
            isTrain=True,
            serial_batches=False,
            class_bal=False,
            num_workers=num_workers,
            pin_memory=opt.pin_memory and use_cuda,
            persistent_workers=opt.persistent_workers and num_workers > 0,
            prefetch_factor=opt.prefetch_factor if num_workers > 0 else None,
            cropSize=224,
            no_flip=False,
        )
        loader = create_dataloader(loader_opt)

    accum_steps = max(1, getattr(opt, 'grad_accum_steps', 1))
    scaler = torch.amp.GradScaler('cuda') if (use_cuda and opt.use_amp) else None
    amp_dtype = torch.float16 if opt.amp_dtype == 'fp16' else torch.bfloat16

    iterator = iter(loader)

    def get_batch():
        nonlocal iterator
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(loader)
            return next(iterator)

    # 3. Warmup
    model.optimizer.zero_grad(set_to_none=True)
    for step in range(warmup_steps):
        batch = get_batch()
        model.set_input(batch)
        if scaler is not None:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                model.forward()
                loss = model.get_loss() / accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == warmup_steps:
                scaler.step(model.optimizer)
                scaler.update()
                model.optimizer.zero_grad(set_to_none=True)
        else:
            model.forward()
            loss = model.get_loss() / accum_steps
            loss.backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == warmup_steps:
                model.optimizer.step()
                model.optimizer.zero_grad(set_to_none=True)

    if use_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    # 4. Measured Steps
    total_data_wait = 0.0
    total_step_time = 0.0
    measured_samples = 0
    optimizer_steps = 0
    model.optimizer.zero_grad(set_to_none=True)

    t_start = time.monotonic()
    for step in range(measure_steps):
        t_data_start = time.monotonic()
        batch = get_batch()
        t_data_end = time.monotonic()
        total_data_wait += (t_data_end - t_data_start)

        t_step_start = time.monotonic()
        model.set_input(batch)
        if scaler is not None:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                model.forward()
                loss = model.get_loss() / accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == measure_steps:
                scaler.step(model.optimizer)
                scaler.update()
                model.optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
        else:
            model.forward()
            loss = model.get_loss() / accum_steps
            loss.backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == measure_steps:
                model.optimizer.step()
                model.optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

        if use_cuda:
            torch.cuda.synchronize(device)
        t_step_end = time.monotonic()
        total_step_time += (t_step_end - t_step_start)
        measured_samples += batch[0].shape[0]

    t_wall = time.monotonic() - t_start

    peak_memory_mb = 0.0
    if use_cuda:
        peak_memory_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)

    # Accounting for contrastive pair pool:
    # Contrastive pairs are formed within rank-local microbatch B.
    # Gradient accumulation matches effective batch size for weights,
    # but does NOT enlarge the contrastive pair pool.
    contrastive_pairs_per_step = (batch_size * (batch_size - 1)) // 2

    return {
        'status': 'completed',
        'data_source': 'synthetic' if synthetic else 'manifest',
        'encoder': opt.drct_backbone,
        'pretrained': getattr(opt, 'pretrained', False),
        'allow_download': getattr(opt, 'drct_allow_download', False),
        'weights_path': getattr(opt, 'drct_backbone_weights', None),
        'batch_size': batch_size,
        'grad_accum_steps': accum_steps,
        'effective_batch_size': batch_size * accum_steps,
        'optimizer_steps': optimizer_steps,
        'num_workers': num_workers,
        'use_amp': opt.use_amp,
        'amp_dtype': opt.amp_dtype if opt.use_amp else None,
        'measured_steps': measure_steps,
        'measured_samples': measured_samples,
        'total_wall_sec': round(t_wall, 4),
        'images_per_sec': round(measured_samples / max(t_wall, 1e-6), 2),
        'avg_data_wait_ms': round((total_data_wait / measure_steps) * 1000, 2),
        'avg_step_time_ms': round((total_step_time / measure_steps) * 1000, 2),
        'data_wait_pct': round((total_data_wait / max(t_wall, 1e-6)) * 100, 1),
        'peak_gpu_memory_mb': peak_memory_mb,
        'contrastive_pool_size': batch_size,
        'contrastive_pairs_per_step': contrastive_pairs_per_step,
        'contrastive_pair_note': (
            "Contrastive pairs are formed strictly within the rank-local batch size (pool=B). "
            "Gradient accumulation matches effective optimizer batch size for parameter updates, "
            "but does NOT enlarge the contrastive pair pool."
        ),
    }


def main():
    args = parse_args()
    if not args.synthetic and not args.manifest:
        sys.exit("Error: Must provide --manifest or explicitly specify --synthetic for synthetic benchmark mode.")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    system_info = {
        'device': str(device),
        'cpu_count': os.cpu_count(),
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None',
        'gpu_capability': torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        'total_gpu_memory_mb': round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 2) if torch.cuda.is_available() else 0,
    }

    smi_report = sample_all_gpus_nvidia_smi()

    opt = SimpleNamespace(
        drct_backbone=args.drct_backbone,
        manifest=args.manifest,
        dataroot=args.dataroot,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        drct_alpha=args.drct_alpha,
        drct_pos_margin=args.drct_pos_margin,
        drct_neg_margin=args.drct_neg_margin,
        embed_dim=args.embed_dim,
    )

    results = []

    if args.sweep:
        if args.synthetic:
            print("[INFO] Benchmark sweep is running in synthetic mode. DataLoader worker counts measure generator throughput, not disk I/O bottlenecks.")
        batch_sizes = [int(x.strip()) for x in args.sweep_batch_sizes.split(',') if x.strip()]
        cpu_max = os.cpu_count() or 4
        worker_counts = [min(int(x.strip()), cpu_max) for x in args.sweep_workers.split(',') if x.strip()]
        # Unique preserve order
        worker_counts = list(dict.fromkeys(worker_counts))

        print(f"=== Starting DRCT-128 Sweep on {device} (CPUs: {cpu_max}) ===")
        print(f"Batch sizes: {batch_sizes} | Workers: {worker_counts}")

        for bs in batch_sizes:
            for nw in worker_counts:
                sweep_opt = SimpleNamespace(**vars(opt))
                sweep_opt.batch_size = bs
                sweep_opt.num_workers = nw
                print(f"\n[Benchmarking] Batch size={bs}, Workers={nw}...", flush=True)
                try:
                    res = run_single_benchmark(
                        opt=sweep_opt,
                        device=device,
                        warmup_steps=args.warmup_steps,
                        measure_steps=args.measure_steps,
                        synthetic=args.synthetic,
                    )
                    print(f"  -> {res['images_per_sec']:.1f} img/s | Data wait: {res['data_wait_pct']:.1f}% | VRAM: {res['peak_gpu_memory_mb']} MB")
                    results.append(res)
                except torch.cuda.OutOfMemoryError:
                    print(f"  -> CUDA OutOfMemoryError at batch_size={bs}. Skipping.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    results.append({
                        'status': 'oom',
                        'batch_size': bs,
                        'num_workers': nw,
                        'images_per_sec': 0.0,
                        'error': 'CUDA OutOfMemoryError',
                    })
                    break  # Larger batch sizes with same or more workers will also OOM
                except Exception as e:
                    print(f"  -> Error: {e}")
                    results.append({
                        'status': 'error',
                        'batch_size': bs,
                        'num_workers': nw,
                        'images_per_sec': 0.0,
                        'error': str(e),
                    })
    else:
        print(f"=== Running Single DRCT-128 Benchmark on {device} ===")
        print(f"Batch size: {args.batch_size} | Workers: {args.num_workers} | AMP: {args.use_amp}")
        res = run_single_benchmark(
            opt=opt,
            device=device,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            synthetic=args.synthetic,
        )
        print(f"Result: {res['images_per_sec']:.1f} img/s | Step: {res['avg_step_time_ms']:.1f}ms | VRAM: {res['peak_gpu_memory_mb']} MB")
        results.append(res)

    report = {
        'benchmark': 'drct_128_throughput',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'system_info': system_info,
        'nvidia_smi': smi_report,
        'results': results,
        'colab_recommendations': {
            'T4 (16GB)': {'batch_size': 32, 'num_workers': 2, 'use_amp': True, 'pin_memory': True},
            'L4 (24GB)': {'batch_size': 64, 'num_workers': 4, 'use_amp': True, 'pin_memory': True},
            'A100 (40GB/80GB)': {'batch_size': 64, 'num_workers': 4, 'use_amp': True, 'pin_memory': True},
            'notes': [
                "Always run training from local /content storage, not directly from mounted Google Drive.",
                "Persist completed checkpoints and training manifests to Google Drive.",
                "Frozen backbone feature caching is disabled by default to preserve stochastic data augmentation.",
                "Contrastive pairs are computed on the rank-local batch size (B*(B-1)/2).",
                "Gradient accumulation simulates larger effective batch sizes for SGD updates, but does NOT enlarge the contrastive pair pool.",
            ],
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"\nSaved benchmark report to: {out_path}")


if __name__ == '__main__':
    main()
