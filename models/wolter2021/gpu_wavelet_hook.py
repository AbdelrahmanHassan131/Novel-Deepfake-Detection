"""
GPU Wavelet Extension Point.

This module provides an interface/extension point that would later allow
replacing CPU-based wavelet computation (pywt) with GPU-based wavelet
computation (e.g., pytorch_wavelets.DWTForward).

IMPORTANT:
    - This does NOT replace the current wavelet implementation.
    - This does NOT change datasets or data loading.
    - This is architecture preparation ONLY.
    - No behavioral changes.

Future usage example:
    from models.wolter2021.gpu_wavelet_hook import get_wavelet_transform
    wavelet_fn = get_wavelet_transform(backend='cpu')  # current behavior
    # wavelet_fn = get_wavelet_transform(backend='gpu')  # future GPU path
"""
import torch
from abc import ABC, abstractmethod


class WaveletTransformBase(ABC):
    """
    Abstract interface for wavelet packet transforms.

    Subclass this to provide alternative wavelet backends
    (e.g., GPU-based pytorch_wavelets).
    """

    @abstractmethod
    def __call__(self, img, wavelet='haar', level=3, mode='reflect'):
        """
        Compute wavelet packet coefficients for an image.

        Args:
            img: Input image tensor or numpy array.
            wavelet: Wavelet type (e.g., 'haar', 'db2').
            level: Decomposition level.
            mode: Signal extension mode.

        Returns:
            torch.Tensor: Wavelet packet coefficients.
        """
        pass

    @property
    @abstractmethod
    def backend(self):
        """Return the backend name ('cpu' or 'gpu')."""
        pass


class CPUWaveletTransform(WaveletTransformBase):
    """
    CPU-based wavelet transform using pywt.
    This is the current default implementation.
    Delegates to the existing compute_wavelet_packet_coeffs function.
    """

    def __call__(self, img, wavelet='haar', level=3, mode='reflect'):
        from .wavelet_utils import compute_wavelet_packet_coeffs
        return compute_wavelet_packet_coeffs(img, wavelet=wavelet,
                                              level=level, mode=mode)

    @property
    def backend(self):
        return 'cpu'


class GPUWaveletTransform(WaveletTransformBase):
    """
    GPU-based wavelet transform using pytorch_wavelets via GPUWaveletBackend.
    """

    def __init__(self, device=None):
        self._device = device
        self._backend = None
        self._last_params = None

    def __call__(self, img, wavelet='haar', level=3, mode='reflect'):
        device = self._device or ('cuda' if torch.cuda.is_available() else 'cpu')
        params = (wavelet, level, mode, device)
        if self._backend is None or self._last_params != params:
            from data.wavelets.backends.gpu_backend import GPUWaveletBackend
            self._backend = GPUWaveletBackend(wavelet=wavelet, level=level, mode=mode, device=device)
            self._last_params = params

        if not isinstance(img, torch.Tensor):
            import torchvision.transforms.functional as TF
            img = TF.to_tensor(img)

        return self._backend(img)

    @property
    def backend(self):
        return 'gpu'


def get_wavelet_transform(backend='cpu'):
    """
    Factory function to get the appropriate wavelet transform.

    Args:
        backend: 'cpu' (default, uses pywt) or 'gpu' (future, uses pytorch_wavelets).

    Returns:
        WaveletTransformBase instance.
    """
    if backend == 'cpu':
        return CPUWaveletTransform()
    elif backend == 'gpu':
        return GPUWaveletTransform()
    else:
        raise ValueError(f"Unknown wavelet backend: {backend}. Use 'cpu' or 'gpu'.")
