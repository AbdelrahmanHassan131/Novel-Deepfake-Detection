"""
Distributed seed handling.

Provides :func:`seed_everything` which sets deterministic seeds for
PyTorch, NumPy, and Python ``random`` in a DDP-aware manner.

Design:
    Each worker receives a deterministic but **unique** seed computed
    as ``base_seed + rank``.  This ensures different workers process
    different random augmentations while still being reproducible.

When deterministic mode is enabled (``opt.deterministic == True``),
additional PyTorch flags are set for full reproducibility at the cost
of potential performance regression.

Usage::

    from Refactored.training.runtime import seed_everything

    seed_everything(42, rank=0, deterministic=True)

Interaction with the Training Engine:
    Called by the ``Trainer`` during ``__init__``, before any data
    loading or model initialization that depends on randomness.

Inputs:
    - ``base_seed`` (int): The base random seed.
    - ``rank`` (int): Process rank (0 in non-distributed mode).
    - ``deterministic`` (bool): Whether to enable full determinism.

Outputs:
    - All random number generators are seeded.
    - If deterministic, CUDNN is configured for reproducibility.
"""

import os
import random

import numpy as np
import torch


def seed_everything(base_seed, rank=0, deterministic=False):
    """
    Seed all random number generators for reproducibility.

    Each distributed worker gets seed ``base_seed + rank`` so that
    workers produce different augmentations but the experiment is
    fully reproducible when re-launched with the same seed and
    world size.

    Args:
        base_seed (int): Base seed shared across all workers.
        rank (int): Global rank of the current process.  Defaults
            to 0 (single-process).
        deterministic (bool): If ``True``, enable full PyTorch
            determinism (``torch.use_deterministic_algorithms``,
            ``cudnn.deterministic``, disable ``cudnn.benchmark``).
            This may reduce performance.

    Example::

        # Single-GPU
        seed_everything(42)

        # DDP rank 2 of 4
        seed_everything(42, rank=2)

        # Full determinism
        seed_everything(42, rank=0, deterministic=True)
    """
    seed = base_seed + rank

    # Python stdlib
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic mode
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # CUBLAS workspace config MUST be set before enabling
        # deterministic algorithms so the CUDA runtime sees it.
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        # PyTorch >= 1.8
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                # Some operations do not have deterministic implementations.
                # Fall back silently.
                pass

    if rank == 0:
        print(
            f'[Seed] base_seed={base_seed}, effective_seed={seed}, '
            f'deterministic={deterministic}'
        )
