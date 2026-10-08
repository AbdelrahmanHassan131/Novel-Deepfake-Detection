#!/usr/bin/env python
"""
Standalone DDP Environment Sanity Check Script.

Verifies PyTorch DistributedDataParallel (DDP) initialization, NCCL/Gloo communication,
and tensor all-reduce across all available GPU ranks before launching training.

Usage:
    torchrun --standalone --nproc_per_node=2 tools/verify_ddp_environment.py
"""

import os
import sys
import torch
import torch.distributed as dist


def main():
    print(f"[Sanity] Process PID {os.getpid()} starting...", flush=True)

    if not torch.cuda.is_available():
        print("[Sanity] CUDA is not available. Checking CPU Gloo communication...", flush=True)
        backend = "gloo"
        device = torch.device("cpu")
    else:
        # Default to nccl on CUDA; fallback to gloo on Windows if nccl unavailable
        backend = "nccl" if dist.is_nccl_available() and os.name != 'nt' else "gloo"
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print("[Sanity] Not running under torchrun/torch.distributed. Running single-process sanity check.", flush=True)
        print(f"[Sanity] PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[Sanity] GPU 0: {torch.cuda.get_device_name(0)} (Capability: {torch.cuda.get_device_capability(0)})")
        return 0

    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(local_rank)
        vram_gb = torch.cuda.get_device_properties(local_rank).total_memory / (1024**3)
    else:
        device_name = "CPU"
        vram_gb = 0.0

    print(
        f"[Rank {rank}/{world_size}] Local Rank: {local_rank} on {device_name} "
        f"({vram_gb:.2f} GB VRAM) | Backend: {backend}",
        flush=True,
    )

    # Perform an all-reduce test
    tensor_val = float(rank + 1)
    t = torch.tensor([tensor_val], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)

    # Expected sum of 1 + 2 + ... + world_size
    expected_sum = float(world_size * (world_size + 1) / 2)
    actual_sum = t.item()

    if rank == 0:
        print(f"[Sanity] All-Reduce Sum: {actual_sum:.1f} (Expected: {expected_sum:.1f})", flush=True)

    if abs(actual_sum - expected_sum) > 1e-4:
        raise RuntimeError(
            f"DDP all-reduce verification failed on rank {rank}! "
            f"Expected {expected_sum}, got {actual_sum}"
        )

    dist.barrier()
    if rank == 0:
        print("[Sanity] DDP Environment sanity verification PASSED successfully.", flush=True)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
