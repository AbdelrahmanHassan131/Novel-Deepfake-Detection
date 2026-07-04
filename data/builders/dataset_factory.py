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
        if getattr(opt, 'compute_wavelets', False):
            return WaveletDataset(opt, root)
        else:
            return RGBDataset(opt, root)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')

def get_dataset(opt):
    """Smart dataloader - handles both training structure (subfolders) and validation structure (direct images)"""
    first_class_path = opt.dataroot + '/' + opt.classes[0]
    
    if has_subfolders(first_class_path):
        # Training structure: fake/method1/, real/dataset1/, etc.
        print("Detected training structure (subfolders)")
        dset_lst = []
        for cls in opt.classes:
            root = opt.dataroot + '/' + cls
            dset = dataset_folder(opt, root)
            dset_lst.append(dset)
        return torch.utils.data.ConcatDataset(dset_lst)
    else:
        # Validation structure: fake/img.jpg, real/img.jpg
        print("Detected validation structure (direct images)")
        # Use dataroot directly - ImageFolder will find fake and real folders
        dset = dataset_folder(opt, opt.dataroot)
        return dset

def get_mha_dataset(opt):
    """
    Create dataset that returns both RGB and Wavelet inputs.
    Smart detection for training vs validation folder structure.
    """
    first_class_path = opt.dataroot + '/' + opt.classes[0]
    
    if has_subfolders(first_class_path):
        # Training structure: fake/method1/, real/dataset1/, etc.
        print("Detected training structure (subfolders) for MHA")
        dset_lst = []
        for cls in opt.classes:
            root = opt.dataroot + '/' + cls
            dset = FusionDataset(opt, root)
            dset_lst.append(dset)
        return torch.utils.data.ConcatDataset(dset_lst)
    else:
        # Validation structure: fake/img.jpg, real/img.jpg
        print("Detected validation structure (direct images) for MHA")
        # Use dataroot directly - FusionDataset will find fake and real folders
        dset = FusionDataset(opt, opt.dataroot)
        return dset
