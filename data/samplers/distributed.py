"""Distributed sampling that preserves weighting and does not pad evaluation."""
import math
import torch
from torch.utils.data import Sampler


class EvaluationSampler(Sampler):
    def __init__(self, dataset, rank, world_size):
        self.indices = range(rank, len(dataset), world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class DistributedWeightedSampler(Sampler):
    def __init__(self, weights, rank, world_size, seed=42):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.rank, self.world_size, self.seed, self.epoch = rank, world_size, seed, 0
        self.count = math.ceil(len(weights) / world_size)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.count

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        draws = torch.multinomial(self.weights, self.count * self.world_size, replacement=True, generator=generator)
        return iter(draws[self.rank::self.world_size].tolist())
