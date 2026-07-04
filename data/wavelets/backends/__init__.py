"""
Wavelet Backend Abstraction Layer.

Provides interchangeable backends for wavelet computation:

    - **CPUWaveletBackend** — on-the-fly computation using PyWavelets (pywt).
    - **GPUWaveletBackend** — on-the-fly computation using pytorch_wavelets
      (DWTForward) on CUDA tensors.
    - **PrecomputedWaveletBackend** — loads pre-computed ``.npy`` files from disk.

All backends produce tensors with identical shapes so downstream models
are completely unaware of which backend generated their input.

Selection is controlled by ``opt.wavelet_backend`` (values: ``'cpu'``,
``'gpu'``, ``'precomputed'``) and routed through a single factory
function :func:`create_wavelet_backend`.

Usage::

    from Refactored.data.wavelets.backends import create_wavelet_backend

    backend = create_wavelet_backend(opt)
    wavelet_tensor = backend(image_tensor)
"""

from .base import WaveletBackend
from .cpu_backend import CPUWaveletBackend
from .gpu_backend import GPUWaveletBackend
from .precomputed_backend import PrecomputedWaveletBackend
from .factory import create_wavelet_backend

__all__ = [
    'WaveletBackend',
    'CPUWaveletBackend',
    'GPUWaveletBackend',
    'PrecomputedWaveletBackend',
    'create_wavelet_backend',
]
