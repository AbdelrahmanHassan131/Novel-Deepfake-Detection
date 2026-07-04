"""
Wavelet Backend Factory.

Provides a single entry-point :func:`create_wavelet_backend` that reads
``opt.wavelet_backend`` and returns the corresponding backend instance.

Backend selection logic is **centralised here** — no other module in the
project should contain ``if backend == …`` branching.

Responsibility:
    Map the ``opt.wavelet_backend`` string to a concrete
    :class:`WaveletBackend` subclass, passing through the relevant
    wavelet parameters from ``opt``.

Expected inputs:
    An ``opt`` namespace with at least:

    - ``wavelet_backend`` (str): ``'cpu'`` | ``'gpu'`` | ``'precomputed'``.
      Defaults to ``'cpu'`` if absent.
    - ``wavelet_type`` (str): Wavelet family.  Defaults to ``'haar'``.
    - ``wavelet_level`` (int): Decomposition level.  Defaults to 3.
    - ``wavelet_mode`` (str): Extension mode.  Defaults to ``'reflect'``.
    - ``use_log_packets`` (bool): Log-scale flag.  Defaults to ``True``.

    For the GPU backend, optionally:

    - ``gpu_ids`` (list[int]): Used to infer the CUDA device when DDP
      environment variables are not set.

Expected outputs:
    A :class:`WaveletBackend` instance ready to be called inside a
    dataset ``__getitem__``.

Interaction with the data pipeline:
    Called once per dataset instantiation (in ``WaveletDataset.__init__``
    and ``FusionDataset.__init__``).  The returned backend is stored on
    the dataset and used on every sample.

Usage::

    from data.wavelets.backends import create_wavelet_backend

    backend = create_wavelet_backend(opt)
    wavelet_tensor = backend(image)   # CPU or GPU tensor
"""

import os
import torch

from .cpu_backend import CPUWaveletBackend
from .gpu_backend import GPUWaveletBackend
from .precomputed_backend import PrecomputedWaveletBackend


def create_wavelet_backend(opt, device=None):
    """
    Create a wavelet backend from an options namespace.

    Args:
        opt: Project options.  Must expose ``wavelet_backend`` (or
            defaults to ``'cpu'``).
        device: Optional explicit ``torch.device`` for the GPU backend.
            When ``None``, the device is inferred from the environment
            (``LOCAL_RANK`` for DDP, ``gpu_ids[0]`` for single-GPU,
            or ``'cuda'`` as a fallback).

    Returns:
        WaveletBackend: A ready-to-use backend instance.

    Raises:
        ValueError: If ``opt.wavelet_backend`` is not one of
            ``'cpu'``, ``'gpu'``, ``'precomputed'``.
    """
    backend_name = getattr(opt, 'wavelet_backend', 'cpu')
    wavelet = getattr(opt, 'wavelet_type', 'haar')
    level = getattr(opt, 'wavelet_level', 3)
    mode = getattr(opt, 'wavelet_mode', 'reflect')
    log_scale = getattr(opt, 'use_log_packets', True)

    if backend_name == 'cpu':
        return CPUWaveletBackend(
            wavelet=wavelet, level=level, mode=mode, log_scale=log_scale,
        )

    if backend_name == 'gpu':
        if device is None:
            device = _resolve_device(opt)
        return GPUWaveletBackend(
            wavelet=wavelet, level=level, mode=mode,
            log_scale=log_scale, device=device,
        )

    if backend_name == 'precomputed':
        return PrecomputedWaveletBackend(
            wavelet=wavelet, level=level, mode=mode, log_scale=log_scale,
        )

    raise ValueError(
        f"Unknown wavelet_backend={backend_name!r}.  "
        f"Choose from: 'cpu', 'gpu', 'precomputed'."
    )


def _resolve_device(opt):
    """
    Infer the CUDA device for the GPU wavelet backend.

    Priority:
        1. ``LOCAL_RANK`` environment variable (DDP via ``torchrun``).
        2. ``opt.gpu_ids[0]`` (single-GPU launch).
        3. ``'cuda'`` (default CUDA device).

    Returns:
        torch.device
    """
    local_rank = os.environ.get('LOCAL_RANK')
    if local_rank is not None:
        return torch.device(f'cuda:{local_rank}')

    gpu_ids = getattr(opt, 'gpu_ids', [])
    if gpu_ids and torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_ids[0]}')

    return torch.device('cuda')
