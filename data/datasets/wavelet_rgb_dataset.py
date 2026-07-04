"""
WaveletRGBDataset — lightweight RGB loader for GPU wavelet pipeline.

Returns unnormalized (image_tensor, label) pairs.  The wavelet packet
decomposition is deferred to the model's ``set_input`` method which
runs batched ``DWTForward`` directly on the GPU.

This allows DataLoader workers to focus exclusively on fast disk I/O
and image augmentation while keeping the GPU fully utilised for both
wavelet computation and model training.
"""

import numpy as np
import torch
from torchvision import transforms
from .base_dataset import BaseDataset
from ..transforms.augmentations import data_augment
from ..transforms.resize import custom_resize


class WaveletRGBDataset(BaseDataset):
    """
    Lightweight RGB dataset for GPU-batched wavelet computation.

    Returns ``(image_tensor, label)`` where ``image_tensor`` is a
    ``float32`` CPU tensor of shape ``(3, H, W)`` with pixel values
    in ``[0, 255]`` (no ImageNet normalisation — wavelets need raw
    pixel intensities).

    The model trainer is responsible for computing wavelet packets
    on GPU inside its ``set_input`` method.
    """

    def __init__(self, opt, root):
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

        # No Normalize — wavelets need raw pixel values
        self.image_transform = transforms.Compose([
            rz_func,
            transforms.Lambda(lambda img: data_augment(img, opt)),
            crop_func,
            flip_func,
        ])

        super().__init__(opt, root, transform=None)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)

        if self.image_transform is not None:
            img = self.image_transform(img)

        # Convert PIL -> float32 tensor (3, H, W) with values [0, 255]
        img_array = np.array(img, dtype=np.float32)
        if img_array.ndim == 3 and img_array.shape[2] == 3:
            img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
        else:
            img_tensor = torch.from_numpy(img_array)

        return img_tensor, target
