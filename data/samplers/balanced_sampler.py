import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

def get_bal_sampler(dataset):
    """Balanced sampler - handles both ConcatDataset and single ImageFolder"""
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        # Training structure - ConcatDataset
        targets = []
        for d in dataset.datasets:
            targets.extend(d.targets)
    else:
        # Validation structure - single ImageFolder
        targets = dataset.targets

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights)
    )
    return sampler
