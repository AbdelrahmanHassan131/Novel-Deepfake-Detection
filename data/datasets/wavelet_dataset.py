import os
import torch
import numpy as np
from torchvision import transforms
from .base_dataset import BaseDataset
from ..transforms.augmentations import data_augment
from ..transforms.resize import custom_resize
from ..wavelets.backends import create_wavelet_backend


class WaveletDataset(BaseDataset):
    """
    Wavelet Dataset that produces wavelet packet tensors.
    Returns: (wavelet_tensor, label)
    Represents behavior used by: WolterWavelet2021Raw, WolterWavelet2021_128.

    Supports three backends controlled by ``opt.wavelet_backend``:

        ``'cpu'`` (default):
            Images are loaded, augmented, and wavelet packets are computed
            on CPU using PyWavelets every time ``__getitem__`` is called.

        ``'gpu'``:
            Images are loaded and augmented on CPU, converted to a tensor,
            then wavelet packets are computed on GPU using pytorch_wavelets.
            The returned tensor is already on the CUDA device.

        ``'precomputed'``:
            ``opt.precomputed_dir`` must point to a directory of ``.npy``
            files whose folder structure mirrors the image dataset.
            Wavelet tensors are loaded directly from disk.

    Backward compatibility:
        When ``opt.wavelet_backend`` is not set, the dataset defaults to
        ``'cpu'``, preserving the original on-the-fly behaviour.

        The legacy ``opt.precomputed_dir`` mechanism is still honoured:
        if ``opt.precomputed_dir`` is set **and** ``opt.wavelet_backend``
        is not explicitly set, the backend is automatically switched to
        ``'precomputed'``.
    """
    def __init__(self, opt, root):
        # --- Resolve backend ---
        # Legacy compatibility: if precomputed_dir is set but
        # wavelet_backend is not explicitly configured, auto-select
        # the precomputed backend.
        precomputed_dir = getattr(opt, 'precomputed_dir', None)
        backend_name = getattr(opt, 'wavelet_backend', None)

        if backend_name is None:
            if precomputed_dir:
                backend_name = 'precomputed'
            else:
                backend_name = 'cpu'
            # Temporarily inject for factory
            opt.wavelet_backend = backend_name

        self._backend = create_wavelet_backend(opt)

        # --- Precomputed root (only for precomputed backend) ---
        self._precomputed_root = None
        if self._backend.name == 'precomputed':
            if not precomputed_dir:
                raise ValueError(
                    "opt.precomputed_dir must be set when using the "
                    "'precomputed' wavelet backend."
                )
            dataroot = os.path.normpath(opt.dataroot)
            root_norm = os.path.normpath(root)
            rel = os.path.relpath(root_norm, dataroot)
            self._precomputed_root = os.path.normpath(
                os.path.join(precomputed_dir, rel)
            )

        # --- Image transforms (not needed for precomputed) ---
        if self._backend.name != 'precomputed':
            if opt.isTrain:
                crop_func = transforms.RandomCrop(opt.cropSize)
            elif opt.no_crop:
                crop_func = transforms.Lambda(lambda img: img)
            else:
                crop_func = transforms.CenterCrop(opt.cropSize)

            if opt.isTrain and not opt.no_flip:
                flip_func = transforms.RandomHorizontalFlip()
            else:
                flip_func = transforms.Lambda(lambda img: img)

            if not opt.isTrain and opt.no_resize:
                rz_func = transforms.Lambda(lambda img: img)
            else:
                rz_func = transforms.Lambda(
                    lambda img: custom_resize(img, opt))

            self.image_transform = transforms.Compose([
                rz_func,
                transforms.Lambda(lambda img: data_augment(img, opt)),
                crop_func,
                flip_func,
            ])
        else:
            self.image_transform = None

        # --- GPU backend needs ToTensor before wavelet computation ---
        self._needs_to_tensor = (self._backend.name == 'gpu')

        super().__init__(opt, root, transform=None)

    # ------------------------------------------------------------------
    # Precomputed helpers
    # ------------------------------------------------------------------

    def _get_npy_path(self, image_path):
        """Map an image file path to its precomputed ``.npy`` counterpart."""
        rel = os.path.relpath(image_path, self.root)
        npy_rel = os.path.splitext(rel)[0] + '.npy'
        return os.path.join(self._precomputed_root, npy_rel)

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, index):
        path, target = self.samples[index]

        # ----- Precomputed backend: load .npy directly -----
        if self._backend.name == 'precomputed':
            npy_path = self._get_npy_path(path)
            sample = self._backend(npy_path)
            return sample, target

        # ----- CPU / GPU backend: load image and compute -----
        img = self.loader(path)

        if self.image_transform is not None:
            img = self.image_transform(img)

        if self._needs_to_tensor:
            # GPU backend expects a torch Tensor.
            # Convert PIL → numpy → tensor (3, H, W) without normalisation.
            img_array = np.array(img, dtype=np.float32)
            if img_array.ndim == 3 and img_array.shape[2] == 3:
                img_tensor = torch.from_numpy(
                    img_array.transpose(2, 0, 1))  # (3, H, W)
            else:
                img_tensor = torch.from_numpy(img_array)
            sample = self._backend(img_tensor)
        else:
            # CPU backend accepts PIL / numpy directly.
            sample = self._backend(img)

        return sample, target
