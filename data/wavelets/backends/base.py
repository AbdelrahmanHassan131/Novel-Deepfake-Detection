"""
WaveletBackend — abstract base class for all wavelet backends.

Every concrete backend must implement :meth:`__call__` so that it can be
used as a simple callable inside dataset ``__getitem__`` methods:

    wavelet_tensor = backend(image_or_path, ...)

Responsibility:
    Define the contract that all wavelet backends must satisfy.

Expected inputs:
    Determined by each concrete subclass.  CPU and GPU backends accept
    image data (PIL Image, numpy array, or torch Tensor).  The
    precomputed backend accepts a file path.

Expected outputs:
    A ``torch.Tensor`` of shape ``(C, H', W')`` where
    ``C = 3 * 4^level`` (e.g. 192 for level 3).

Interaction with the data pipeline:
    The dataset creates a backend once via :func:`create_wavelet_backend`
    and calls it on every sample in ``__getitem__``.  The model never
    sees the backend.
"""

from abc import ABC, abstractmethod
import torch


class WaveletBackend(ABC):
    """
    Abstract base class for wavelet computation backends.

    Subclasses must implement:

    - :meth:`__call__` — produce a wavelet tensor from input data.
    - :attr:`name` — a human-readable identifier for logging.

    Attributes:
        wavelet (str): Wavelet family name (e.g. ``'haar'``).
        level (int): Decomposition level (default 3).
        mode (str): Signal extension mode (default ``'reflect'``).
        log_scale (bool): Whether to apply log-scaling to coefficients.
    """

    def __init__(self, wavelet='haar', level=3, mode='reflect',
                 log_scale=True):
        """
        Args:
            wavelet: Wavelet family (default ``'haar'``).
            level: Decomposition level (default 3).
            mode: Signal extension mode (default ``'reflect'``).
            log_scale: Apply log-scaling to coefficients (default ``True``).
        """
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.log_scale = log_scale

    @abstractmethod
    def __call__(self, data):
        """
        Compute or load wavelet packet coefficients.

        Args:
            data: Backend-specific input.  For CPU/GPU backends this is
                an image (PIL Image, numpy array, or torch Tensor).
                For the precomputed backend this is a file path.

        Returns:
            torch.Tensor: Wavelet coefficients of shape
                ``(C, H', W')`` where ``C = 3 * 4^level``.
        """
        ...

    @property
    @abstractmethod
    def name(self):
        """Human-readable backend name (e.g. ``'cpu'``, ``'gpu'``)."""
        ...

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'wavelet={self.wavelet!r}, level={self.level}, '
            f'mode={self.mode!r}, log_scale={self.log_scale})'
        )
