"""Train/validation/test splits, the group labels they use, and the distance of a material to a training set.

A random split puts close relatives of most test materials into the training set. The group splits here keep
every member of a group (all polymorphs of a formula, all compounds of a chemical system, all materials of a
structure prototype) on one side, and :func:`distance_to_train` measures how far each test material is from
the training set in feature space.
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from functools import lru_cache, reduce
from importlib import resources
from itertools import combinations
from math import gcd

import numpy as np


def _check_fractions(fractions):
    f = np.asarray(fractions, dtype=float)
    if f.shape != (3,) or (f < 0).any() or not np.isclose(f.sum(), 1.0):
        raise ValueError(f"fractions must be three non-negative numbers summing to 1: {fractions}")
    return f


def random_split(n, seed, fractions=(0.8, 0.1, 0.1)):
    """Disjoint index arrays (train, val, test) covering range(n), in random order.

    The validation and test sizes are floor(f * n); the training set gets the rest.
    """
    f = _check_fractions(fractions)
    perm = np.random.default_rng(seed).permutation(n)
    n_val, n_test = int(f[1] * n), int(f[2] * n)
    n_train = n - n_val - n_test
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


def group_split(groups, seed, fractions=(0.8, 0.1, 0.1)):
    """Disjoint sorted index arrays (train, val, test) such that no group is in two of them.

    ``groups`` holds one hashable label per sample. The groups are shuffled and taken in turn for the test
    set until it holds at least ``fractions[2] * n`` samples, then for the validation set until it holds at
    least ``fractions[1] * n``; the rest is the training set. Each part therefore overshoots its target by
    less than the size of one group.
    """
    f = _check_fractions(fractions)
    index = {}
    codes = np.array([index.setdefault(g, len(index)) for g in groups], dtype=np.int64)
    n = len(codes)
    sizes = np.bincount(codes, minlength=len(index))
    order = np.random.default_rng(seed).permutation(len(index))
    cum = np.cumsum(sizes[order])

    def groups_needed(target):
        return 0 if target <= 0 else min(int(np.searchsorted(cum, target)) + 1, len(order))

    k_test = groups_needed(f[2] * n)
    k_val = max(groups_needed(f[2] * n + f[1] * n), k_test + (f[1] > 0))
    part = np.zeros(len(index), dtype=np.int8)       # 0 train, 1 val, 2 test
    part[order[:k_test]] = 2
    part[order[k_test:k_val]] = 1
    where = part[codes]
    return np.flatnonzero(where == 0), np.flatnonzero(where == 1), np.flatnonzero(where == 2)


def random_folds(n, n_folds, seed):
    """Fold index (0 .. n_folds - 1) of each of n samples: a random partition into folds of equal size (±1)."""
    folds = np.empty(n, dtype=np.int64)
    folds[np.random.default_rng(seed).permutation(n)] = np.arange(n) % n_folds
    return folds


def group_folds(groups, n_folds, seed):
    """Fold index of each sample such that every group lies in one fold and the folds have nearly equal size.

    The groups are taken from the largest to the smallest (equal sizes in random order) and each is put into
    the fold that holds the fewest samples so far, as in scikit-learn's ``GroupKFold``. The folds then differ
    in size by at most the size of the largest group, and usually by much less.
    """
    index = {}
    codes = np.array([index.setdefault(g, len(index)) for g in groups], dtype=np.int64)
    sizes = np.bincount(codes, minlength=len(index))
    order = np.random.default_rng(seed).permutation(len(index))
    order = order[np.argsort(-sizes[order], kind="stable")]
    fill = np.zeros(n_folds, dtype=np.int64)
    fold_of_group = np.empty(len(index), dtype=np.int64)
    for g in order:
        k = int(np.argmin(fill))
        fold_of_group[g] = k
        fill[k] += sizes[g]
    return fold_of_group[codes]


def fold_split(folds, k):
    """(train, val, test) for cross-validation: fold k is the test set, fold k + 1 the validation set."""
    n_folds = int(folds.max()) + 1
    test = np.flatnonzero(folds == k)
    val = np.flatnonzero(folds == (k + 1) % n_folds)
    train = np.flatnonzero((folds != k) & (folds != (k + 1) % n_folds))
    return train, val, test


def reduced_formula(species):
    """Reduced formula of a list of species as a sorted tuple ((element, count), ...): Fe4O6 -> Fe2O3."""
    counts = Counter(species)
    g = reduce(gcd, counts.values())
    return tuple(sorted((e, k // g) for e, k in counts.items()))


def chemical_system(species):
    """Sorted tuple of the distinct elements: Fe2O3 -> ('Fe', 'O')."""
    return tuple(sorted(set(species)))


WYCKOFF_ORDER = "abcdefghijklmnopqrstuvwxyzA"   # spglib and the International Tables use 'A' after 'z'


@lru_cache(maxsize=1)
def wyckoff_sets():
    """{space-group number: list of dicts letter -> letter}, one per operation of the normalizer.

    The same crystal can be described with different Wyckoff letters, depending on the choice of origin
    and axes (zincblende: Zn on 4a and S on 4c, or 4a and 4d, 4b and 4c, 4b and 4d). Two descriptions are
    equivalent exactly when an operation of the normalizer of the space group maps one onto the other;
    this table lists how each such operation permutes the letters. Data: the Wyckoff-set table of PyXtal
    (MIT License, see THIRD_PARTY_NOTICES.md), compacted by ``_build/make_wyckoff_sets.py``.
    """
    raw = json.loads(resources.files(__package__).joinpath("wyckoff_sets.json").read_text(encoding="utf-8"))
    return {int(sg): [dict(zip(WYCKOFF_ORDER, image, strict=False)) for image in images]
            for sg, images in raw.items()}


def prototype_label(lattice, frac, species, symprec=0.1):
    """Structure-type label: space group and, per species, its reduced count and Wyckoff letters.

    The species themselves are left out (NaCl and MgO get the same label), so the label is
    (space-group number, sorted tuple of (reduced count, sorted Wyckoff letters) per species), similar to the
    AFLOW prototype label without the Pearson symbol. Of the equivalent descriptions related by the normalizer
    of the space group (see :func:`wyckoff_sets`), the smallest is returned, so that the label does not depend
    on the choice of origin or axes. Symmetry is found with spglib at the tolerance ``symprec`` (angstrom).
    Returns None if spglib finds no symmetry dataset.
    """
    import spglib

    kinds = sorted(set(species))
    numbers = [kinds.index(s) + 1 for s in species]
    with warnings.catch_warnings():
        # spglib >= 2.5 warns on every call unless its global error handling is switched; failure still
        # returns None, which is handled below
        warnings.simplefilter("ignore", DeprecationWarning)
        ds = spglib.get_symmetry_dataset((np.asarray(lattice, float), np.asarray(frac, float), numbers),
                                         symprec=symprec)
    if ds is None:
        return None
    letters = {}
    # one representative per orbit of the space group; crystallographic_orbits uses the symmetry of the
    # primitive cell (equivalent_atoms would split orbits of a supercell whose shape breaks the point group)
    for atom in set(ds.crystallographic_orbits.tolist()):
        letters.setdefault(species[atom], []).append(ds.wyckoffs[atom])
    counts = dict(reduced_formula(species))
    sg = int(ds.number)
    sites = min(tuple(sorted((counts[s], tuple(sorted(perm[x] for x in w))) for s, w in letters.items()))
                for perm in wyckoff_sets()[sg])
    return sg, sites


def system_relations(train_systems, test_systems):
    """For each test chemical system, how it is related to the chemical systems of the training set.

    Returns a dict of boolean arrays over the test systems:

    - ``same``: the same system occurs in the training set;
    - ``larger``: a training system strictly contains it (Li-O in test, Li-Mn-O in train);
    - ``smaller``: a training system with at least two elements is strictly contained in it
      (Li-Mn-O in test, Li-O in train);
    - ``none``: none of the three.
    """
    train = {tuple(sorted(s)) for s in train_systems}
    contained = set()
    for s in train:
        for k in range(1, len(s)):
            contained.update(combinations(s, k))
    same, larger, smaller = [], [], []
    for s in test_systems:
        s = tuple(sorted(s))
        same.append(s in train)
        larger.append(s in contained)
        smaller.append(any(c in train for k in range(2, len(s)) for c in combinations(s, k)))
    out = {"same": np.array(same, bool), "larger": np.array(larger, bool), "smaller": np.array(smaller, bool)}
    out["none"] = ~(out["same"] | out["larger"] | out["smaller"])
    return out


def distance_to_train(X_train, X_test, k=5, chunk=1024):
    """Mean Euclidean distance from each test row to its k nearest training rows, in standardized units.

    Each feature is centered and scaled with the mean and standard deviation of the training rows (features
    that are constant in the training set are only centered); missing values are replaced by the training
    median first. The distance is divided by the square root of the number of features, so it is a
    root-mean-square difference per feature.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    median = np.nanmedian(X_train, axis=0)
    A = np.where(np.isnan(X_train), median, X_train)
    B = np.where(np.isnan(X_test), median, X_test)
    mean, std = A.mean(axis=0), A.std(axis=0)
    std[std == 0] = 1.0
    A = (A - mean) / std
    B = (B - mean) / std
    a2 = np.einsum("ij,ij->i", A, A)
    out = np.empty(len(B))
    for start in range(0, len(B), chunk):
        b = B[start:start + chunk]
        d2 = a2[None, :] - 2.0 * b @ A.T + np.einsum("ij,ij->i", b, b)[:, None]
        nearest = np.partition(np.maximum(d2, 0.0), k - 1, axis=1)[:, :k]
        out[start:start + chunk] = np.sqrt(nearest).mean(axis=1)
    return out / np.sqrt(A.shape[1])
