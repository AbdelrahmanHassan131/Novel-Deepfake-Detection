import torch
from ..builders.dataset_factory import get_dataset, get_mha_dataset
from ..samplers.balanced_sampler import get_bal_sampler

def create_dataloader(opt):
    """Standard dataloader for single-input models (RGB or Wavelet)"""
    shuffle = not opt.serial_batches if (
        opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,  # Set to 0 to avoid pickle errors on Windows
        pin_memory=True
    )
    return data_loader


def create_mha_dataloader(opt):
    """
    DataLoader for MHA Fusion model.
    Returns: (rgb_images, wavelet_packets, labels)
    """
    shuffle = not opt.serial_batches if (
        opt.isTrain and not opt.class_bal) else False
    dataset = get_mha_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(opt.num_threads),
        pin_memory=True
    )
    return data_loader
