"""
CPUWaveletBackend — wavelet packet computation using PyWavelets (pywt).

This is the original on-the-fly wavelet computation path.  It wraps the
existing :func:`compute_wavelet_packet_coeffs` and
:func:`log_scale_packets` from ``data.wavelets.packet_transform``
so that their behaviour is preserved exactly.

Responsibility:
    Compute wavelet packet coefficients on the CPU using PyWavelets.

Expected inputs:
    A PIL Image, numpy array of shape ``(H, W, 3)``, or torch Tensor
    of shape ``(3, H, W)`` — any format accepted by the existing
    :func:`compute_wavelet_packet_coeffs`.

Expected outputs:
    ``torch.Tensor`` of shape ``(C, H', W')`` where
    ``C = 3 * 4^level`` (e.g. 192 for level 3).

Interaction with the data pipeline:
    Used by ``WaveletDataset`` and ``FusionDataset`` when
    ``opt.wavelet_backend == 'cpu'`` (the default).  The dataset calls
    ``backend(augmented_image)`` inside ``__getitem__``.

Data flow::

    Image → CPU augmentations → numpy array → pywt WaveletPacket2D
          → log-scale (optional) → torch.Tensor
"""

import numpy as np
import torch

from .base import WaveletBackend
from ..packet_transform import compute_wavelet_packet_coeffs, log_scale_packets


class CPUWaveletBackend(WaveletBackend):
    """
    CPU-based wavelet packet backend using PyWavelets.

    Preserves the exact behaviour of the original on-the-fly wavelet
    computation.  No behavioural changes compared to the pre-refactor
    implementation.

    Args:
        wavelet: Wavelet family (default ``'haar'``).
        level: Decomposition level (default 3).
        mode: Signal extension mode (default ``'reflect'``).
        log_scale: Apply log-scaling to coefficients (default ``True``).
    """

    def __call__(self, data):
        """
        Compute wavelet packets from an image on CPU.

        Args:
            data: PIL Image, numpy array ``(H, W, 3)`` or ``(3, H, W)``,
                or torch Tensor ``(3, H, W)``.

        Returns:
            torch.Tensor: Wavelet coefficients ``(C, H', W')``.
        """
        # Convert PIL to numpy if needed
        if hasattr(data, 'mode'):  # PIL Image
            data = np.array(data)

        wavelet_coeffs = compute_wavelet_packet_coeffs(
            data,
            wavelet=self.wavelet,
            level=self.level,
            mode=self.mode,
        )

        if self.log_scale:
            wavelet_coeffs = log_scale_packets(wavelet_coeffs)

        return torch.from_numpy(wavelet_coeffs).float()

    @property
    def name(self):
        """Return ``'cpu'``."""
        return 'cpu'
