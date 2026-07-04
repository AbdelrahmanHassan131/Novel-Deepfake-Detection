"""
PrecomputedWaveletBackend — loads wavelet coefficients from ``.npy`` files.

This backend does **not** compute wavelets.  It loads pre-computed
coefficient arrays that were generated offline (e.g. by the
``Refactored/data/tools/precompute_wavelets.py`` script).

Responsibility:
    Load a ``.npy`` file and return its contents as a ``torch.Tensor``.

Expected inputs:
    An absolute file path (``str`` or ``os.PathLike``) pointing to a
    ``.npy`` file that contains a float32 numpy array of shape
    ``(C, H', W')`` where ``C = 3 * 4^level``.

Expected outputs:
    ``torch.Tensor`` of shape ``(C, H', W')``, dtype ``float32``.

Interaction with the data pipeline:
    Used by ``WaveletDataset`` when ``opt.wavelet_backend == 'precomputed'``.
    The dataset maps each image path to its ``.npy`` counterpart and
    passes that path to ``backend(npy_path)``.

    Note that the wavelet/level/mode/log_scale parameters stored on the
    base class are informational only for this backend — the actual
    parameters were baked in at pre-computation time.

Data flow::

    .npy file path → np.load → torch.from_numpy → Wavelet Tensor
"""

import numpy as np
import torch

from .base import WaveletBackend


class PrecomputedWaveletBackend(WaveletBackend):
    """
    Backend that loads pre-computed wavelet coefficients from ``.npy`` files.

    This backend preserves the existing precomputed-wavelet behaviour
    with zero behavioural changes.

    Args:
        wavelet: Wavelet family (informational; default ``'haar'``).
        level: Decomposition level (informational; default 3).
        mode: Signal extension mode (informational; default ``'reflect'``).
        log_scale: Whether log-scaling was applied during pre-computation
            (informational; default ``True``).
    """

    def __call__(self, data):
        """
        Load wavelet coefficients from a ``.npy`` file.

        Args:
            data: ``str`` — absolute path to the ``.npy`` file.

        Returns:
            torch.Tensor: Wavelet coefficients ``(C, H', W')``.
        """
        wavelet_coeffs = np.load(data)
        return torch.from_numpy(wavelet_coeffs).float()

    @property
    def name(self):
        """Return ``'precomputed'``."""
        return 'precomputed'
