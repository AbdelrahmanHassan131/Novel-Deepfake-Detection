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
    from Refactored.models.wolter2021.gpu_wavelet_hook import get_wavelet_transform
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
    Placeholder for future GPU-based wavelet transform.

    When pytorch_wavelets or equivalent is ready for integration:
        1. Implement this class
        2. Use pytorch_wavelets.DWTForward for GPU computation
        3. Ensure output shape matches CPUWaveletTransform

    Currently raises NotImplementedError.
    """

    def __call__(self, img, wavelet='haar', level=3, mode='reflect'):
        raise NotImplementedError(
            "GPU wavelet transform is not yet implemented. "
            "This is a placeholder for future integration with "
            "pytorch_wavelets.DWTForward or equivalent."
        )

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
