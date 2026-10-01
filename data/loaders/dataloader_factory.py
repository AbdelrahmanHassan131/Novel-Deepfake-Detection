import random
import numpy as np
import torch
from ..builders.dataset_factory import get_dataset, get_mha_dataset
from ..samplers.balanced_sampler import get_bal_sampler

def _worker_init_fn(worker_id):
    """Deterministic worker seeding without worker-side CUDA initialization."""
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloader(opt):
    """Standard dataloader for single-input models (RGB or Wavelet)"""
    shuffle = not opt.serial_batches if (
        opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal and opt.isTrain else None

    num_workers = getattr(opt, 'num_workers',
                          getattr(opt, 'num_threads', 4))
    use_pin = True

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn,
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
    sampler = get_bal_sampler(dataset) if opt.class_bal and opt.isTrain else None

    num_workers = getattr(opt, 'num_workers',
                          getattr(opt, 'num_threads', 4))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn,
    )
    return data_loader

