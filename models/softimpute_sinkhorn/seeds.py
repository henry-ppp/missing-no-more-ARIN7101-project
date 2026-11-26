"""Seeding: the random-forest package's single-seed streams, plus a PyTorch generator built the same way.

Every random step gets its own stream derived from the benchmark seed (0) and a fixed label, so a rerun gives exactly
the same numbers and the random forests, SoftImpute and Sinkhorn all follow one seeding rule.
"""
from models.random_forest.seeds import derive_seed, stream

__all__ = ['derive_seed', 'stream', 'torch_generator']


def torch_generator(seed: int, *labels):
    import torch
    return torch.Generator().manual_seed(derive_seed(seed, *labels))
