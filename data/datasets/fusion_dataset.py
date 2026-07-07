import os
import torch
import numpy as np
from torchvision import transforms
from .base_dataset import BaseDataset
from ..transforms.augmentations import data_augment
from ..transforms.resize import custom_resize
from ..wavelets.backends import create_wavelet_backend


class FusionDataset(BaseDataset):
    """
    Dataset that returns both RGB images and Wavelet packets.
    Returns: (rgb_tensor, wavelet_tensor, label)
    Represents behavior used by: MHA_128, Fusion_128.

    Supports three backends controlled by ``opt.wavelet_backend``:

        ``'cpu'`` (default):
            RGB images are loaded, augmented, and wavelet packets are
            computed from the same augmented image each time.

        ``'gpu'``:
            RGB images are loaded and augmented on CPU.  The augmented
            image is converted to a tensor, then wavelet packets are
            computed on GPU using pytorch_wavelets.

        ``'precomputed'``:
            ``opt.precomputed_dir`` must point to a directory of ``.npy``
            files whose folder structure mirrors the image dataset.
            RGB images are still loaded and augmented normally.
            Wavelet tensors are loaded from ``.npy`` files.

    Backward compatibility:
        When ``opt.wavelet_backend`` is not set, the dataset defaults to
        ``'cpu'``.  The legacy ``opt.precomputed_dir`` auto-detection is
        also honoured (see ``WaveletDataset`` for details).
    """
    def __init__(self, opt, root):
        # --- Resolve backend ---
        precomputed_dir = getattr(opt, 'precomputed_dir', None)
        backend_name = getattr(opt, 'wavelet_backend', None)

        if backend_name is None:
            if precomputed_dir:
                backend_name = 'precomputed'
            else:
                backend_name = 'cpu'
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

        # --- Image transforms (always needed for RGB branch) ---
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
            rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))

        self.image_transform = transforms.Compose([
            rz_func,
            transforms.Lambda(lambda img: data_augment(img, opt)),
            crop_func,
            flip_func,
        ])

        self.rgb_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

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
        img = self.loader(path)

        if self.image_transform is not None:
            img = self.image_transform(img)

        # RGB branch — always processed normally
        rgb_tensor = self.rgb_normalize(img)

        # Wavelet branch — routed through the backend
        if self._backend.name == 'precomputed':
            # ----- Precomputed: load .npy -----
            npy_path = self._get_npy_path(path)
            wavelet_tensor = self._backend(npy_path)
        elif self._needs_to_tensor:
            # ----- GPU backend: return raw CPU float32 tensor for GPU batched computation -----
            img_array = np.array(img, dtype=np.float32)
            if img_array.ndim == 3 and img_array.shape[2] == 3:
                img_tensor = torch.from_numpy(
                    img_array.transpose(2, 0, 1))  # (3, H, W)
            else:
                img_tensor = torch.from_numpy(img_array)
            wavelet_tensor = img_tensor
        else:
            # ----- CPU backend: pass image directly -----
            wavelet_tensor = self._backend(img)

        return rgb_tensor, wavelet_tensor, target
