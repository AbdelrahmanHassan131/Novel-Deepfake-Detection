import os
import torch
from torchvision import datasets
from ..datasets.rgb_dataset import RGBDataset
from ..datasets.wavelet_dataset import WaveletDataset
from ..datasets.fusion_dataset import FusionDataset

class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path

def has_subfolders(directory):
    """Check if directory contains subfolders (training structure) or direct images (validation structure)"""
    if not os.path.exists(directory):
        return False
    items = os.listdir(directory)
    # Check if any item is a directory
    for item in items:
        if os.path.isdir(os.path.join(directory, item)):
            return True
    return False

def dataset_folder(opt, root):
    if opt.mode == 'binary':
        # Route to the appropriate dataset class based on configuration
        arch = getattr(opt, 'arch', '') or getattr(opt, 'architecture', '')
        backend = getattr(opt, 'wavelet_backend', 'cpu')

        if 'Wolter' in arch and backend == 'gpu':
            # GPU wavelet mode: load RGB images with fast parallel workers.
            # The model computes wavelets on GPU in batch inside set_input.
            from ..datasets.wavelet_rgb_dataset import WaveletRGBDataset
            return WaveletRGBDataset(opt, root)
        elif getattr(opt, 'compute_wavelets', False) or 'Wolter' in arch:
            return WaveletDataset(opt, root)
        elif 'Fusion' in arch or 'MHA' in arch:
            return FusionDataset(opt, root)
        else:
            return RGBDataset(opt, root)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')

def get_dataset(opt):
    """Read the binary root once; nested source folders retain outer labels."""
    return dataset_folder(opt, opt.dataroot)


def get_mha_dataset(opt):
    return FusionDataset(opt, opt.dataroot)
