import random
import numpy as np
import torch
from ..builders.dataset_factory import get_dataset, get_mha_dataset
from ..samplers.balanced_sampler import get_bal_sampler

def _worker_init_fn(worker_id):
    """Deterministic worker seeding and thread limiting to avoid oversubscription."""
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

def create_dataloader(opt):
    """Standard dataloader for single-input models (RGB or Wavelet)"""
    is_train = getattr(opt, 'isTrain', True)
    shuffle = not opt.serial_batches if (
        is_train and not opt.class_bal) else False
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if (opt.class_bal and is_train) else None

    if is_train:
        batch_size = opt.batch_size
        num_workers = getattr(opt, 'num_workers', getattr(opt, 'num_threads', 4))
    else:
        batch_size = getattr(opt, 'val_batch_size', None)
        if batch_size is None:
            batch_size = opt.batch_size
        num_workers = getattr(opt, 'val_num_workers', None)
        if num_workers is None:
            num_workers = getattr(opt, 'num_workers', getattr(opt, 'num_threads', 4))

    pin_memory = getattr(opt, 'pin_memory', True)
    persistent_workers = getattr(opt, 'persistent_workers', True) if num_workers > 0 else False
    prefetch_factor = getattr(opt, 'prefetch_factor', 2) if num_workers > 0 else None

    loader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'sampler': sampler,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers,
        'worker_init_fn': _worker_init_fn,
    }
    if prefetch_factor is not None:
        loader_kwargs['prefetch_factor'] = prefetch_factor

    return torch.utils.data.DataLoader(**loader_kwargs)


def create_mha_dataloader(opt):
    """
    DataLoader for MHA Fusion model.
    Returns: (rgb_images, wavelet_packets, labels)
    """
    is_train = getattr(opt, 'isTrain', True)
    shuffle = not opt.serial_batches if (
        is_train and not opt.class_bal) else False
    dataset = get_mha_dataset(opt)
    sampler = get_bal_sampler(dataset) if (opt.class_bal and is_train) else None

    if is_train:
        batch_size = opt.batch_size
        num_workers = getattr(opt, 'num_workers', getattr(opt, 'num_threads', 4))
    else:
        batch_size = getattr(opt, 'val_batch_size', None)
        if batch_size is None:
            batch_size = opt.batch_size
        num_workers = getattr(opt, 'val_num_workers', None)
        if num_workers is None:
            num_workers = getattr(opt, 'num_workers', getattr(opt, 'num_threads', 4))

    pin_memory = getattr(opt, 'pin_memory', True)
    persistent_workers = getattr(opt, 'persistent_workers', True) if num_workers > 0 else False
    prefetch_factor = getattr(opt, 'prefetch_factor', 2) if num_workers > 0 else None

    loader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'sampler': sampler,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers,
        'worker_init_fn': _worker_init_fn,
    }
    if prefetch_factor is not None:
        loader_kwargs['prefetch_factor'] = prefetch_factor

    return torch.utils.data.DataLoader(**loader_kwargs)
