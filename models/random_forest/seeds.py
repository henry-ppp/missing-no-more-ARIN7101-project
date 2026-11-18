"""One shared seed for every random step.

The experiment uses a single seed: the benchmark's own seed (0), the same one that chose the rows and the hidden
cells of the prepared data. Each random step (the tuning sample, the cells hidden for tuning, every forest) gets
its own stream derived from that seed and a fixed label naming the step, so streams never collide and a rerun gives
exactly the same numbers. Labels are hashed with CRC-32, which is stable across
Python runs (unlike hash()).
"""
from __future__ import annotations

import zlib

import numpy as np


def _label_ids(labels) -> list[int]:
    return [label if isinstance(label, int) else zlib.crc32(str(label).encode()) for label in labels]


def stream(seed: int, *labels) -> np.random.Generator:
    """A random generator for one step, e.g. stream(0, 'tuning', 'rows')."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), *_label_ids(labels)]))


def derive_seed(seed: int, *labels) -> int:
    """One integer for code that needs a number (scikit-learn's random_state), e.g. derive_seed(0, 'forest', 3, 7)."""
    return int(np.random.SeedSequence([int(seed), *_label_ids(labels)]).generate_state(1)[0] % (2**31 - 1))
