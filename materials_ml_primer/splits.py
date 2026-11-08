"""Train/validation/test splits.

Only the random split is here so far; chemistry- and structure-aware splits
are added with the ``data_splits`` experiment.
"""

from __future__ import annotations

import numpy as np


def random_split(n, seed, fractions=(0.8, 0.1, 0.1)):
    """Disjoint index arrays (train, val, test) covering range(n), in random order.

    The validation and test sizes are floor(f * n); the training set gets the rest.
    """
    f = np.asarray(fractions, dtype=float)
    if f.shape != (3,) or (f < 0).any() or not np.isclose(f.sum(), 1.0):
        raise ValueError(f"fractions must be three non-negative numbers summing to 1: {fractions}")
    perm = np.random.default_rng(seed).permutation(n)
    n_val, n_test = int(f[1] * n), int(f[2] * n)
    n_train = n - n_val - n_test
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]
