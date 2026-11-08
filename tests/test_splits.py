"""Tests for materials_ml_primer.splits."""

import numpy as np
import pytest

from materials_ml_primer.splits import random_split


@pytest.mark.parametrize("n", [10, 55712, 101])
def test_random_split_partitions_indices(n):
    tr, va, te = random_split(n, seed=0)
    allidx = np.concatenate([tr, va, te])
    assert np.array_equal(np.sort(allidx), np.arange(n))
    assert len(va) == int(0.1 * n) and len(te) == int(0.1 * n)
    assert len(tr) == n - len(va) - len(te)


def test_random_split_is_reproducible_and_seed_dependent():
    a = random_split(1000, seed=3)
    b = random_split(1000, seed=3)
    c = random_split(1000, seed=4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert not np.array_equal(a[2], c[2])


@pytest.mark.parametrize("bad", [(0.8, 0.1), (0.8, 0.3, 0.1), (1.1, -0.1, 0.0)])
def test_random_split_rejects_bad_fractions(bad):
    with pytest.raises(ValueError):
        random_split(10, seed=0, fractions=bad)
