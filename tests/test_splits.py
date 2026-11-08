"""Tests for materials_ml_primer.splits."""

from itertools import combinations, permutations

import numpy as np
import pytest

from materials_ml_primer.splits import (
    WYCKOFF_ORDER,
    chemical_system,
    distance_to_train,
    fold_split,
    group_folds,
    group_split,
    prototype_label,
    random_folds,
    random_split,
    reduced_formula,
    system_relations,
    wyckoff_sets,
)


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
def test_splits_reject_bad_fractions(bad):
    with pytest.raises(ValueError):
        random_split(10, seed=0, fractions=bad)
    with pytest.raises(ValueError):
        group_split(list(range(10)), seed=0, fractions=bad)


# ---------------------------------------------------------------- group split

def random_groups(n, n_groups, seed=0):
    rng = np.random.default_rng(seed)
    sizes = rng.zipf(1.8, n_groups).clip(max=n // 20)       # heavy-tailed, like chemical systems
    labels = np.repeat(np.arange(n_groups), sizes)
    return [f"g{k}" for k in rng.permutation(labels)]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_group_split_keeps_groups_together(seed):
    groups = random_groups(5000, 1500, seed)
    n = len(groups)
    tr, va, te = group_split(groups, seed=seed)
    assert np.array_equal(np.sort(np.concatenate([tr, va, te])), np.arange(n))
    sides = [{groups[i] for i in part} for part in (tr, va, te)]
    assert not (sides[0] & sides[1]) and not (sides[0] & sides[2]) and not (sides[1] & sides[2])
    largest = max(np.unique(groups, return_counts=True)[1])
    assert 0.1 * n <= len(te) < 0.1 * n + largest
    assert 0.2 * n <= len(te) + len(va) < 0.2 * n + 2 * largest


def test_group_split_with_singleton_groups_has_exact_sizes():
    tr, va, te = group_split(list(range(1000)), seed=0)
    assert (len(tr), len(va), len(te)) == (800, 100, 100)


def test_group_split_is_reproducible_and_accepts_tuple_labels():
    groups = [("Fe", "O"), ("Na", "Cl"), ("Fe", "O"), ("Li", "Mn", "O")] * 50
    a = group_split(groups, seed=5)
    b = group_split(groups, seed=5)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_group_split_without_test_set():
    groups = random_groups(2000, 600)
    tr, va, te = group_split(groups, seed=0, fractions=(0.9, 0.1, 0.0))
    assert len(te) == 0 and len(va) >= 200
    assert not ({groups[i] for i in tr} & {groups[i] for i in va})


# ---------------------------------------------------------------- cross-validation folds

def test_random_folds_are_balanced_and_reproducible():
    f = random_folds(1003, 10, seed=0)
    assert sorted(np.bincount(f)) == [100] * 7 + [101] * 3
    assert np.array_equal(f, random_folds(1003, 10, seed=0))
    assert not np.array_equal(f, random_folds(1003, 10, seed=1))


@pytest.mark.parametrize("seed", [0, 1])
def test_group_folds_keep_groups_together_and_balance_sizes(seed):
    groups = random_groups(20000, 6000, seed)
    f = group_folds(groups, 10, seed)
    fold_of = {}
    for g, k in zip(groups, f, strict=True):
        assert fold_of.setdefault(g, k) == k
    counts = np.bincount(f, minlength=10)
    assert counts.sum() == len(groups) and (counts > 0).all()
    assert counts.max() - counts.min() <= 1          # many small groups fill the gaps exactly


def test_group_folds_balance_like_scikit_learn():
    from sklearn.model_selection import GroupKFold

    groups = random_groups(5000, 400, 3)
    ours = np.bincount(group_folds(groups, 10, 0), minlength=10)
    theirs = np.array([len(te) for _, te in GroupKFold(10).split(np.zeros(len(groups)), groups=groups)])
    assert ours.max() - ours.min() <= theirs.max() - theirs.min()


def test_one_large_group_gets_a_fold_of_its_own():
    groups = ["big"] * 300 + [f"s{i}" for i in range(700)]
    f = group_folds(groups, 10, seed=0)
    big = f[0]
    assert (f[:300] == big).all() and np.bincount(f)[big] == 300
    others = np.delete(np.bincount(f), big)
    assert others.max() - others.min() <= 1


def test_fold_split_partitions_and_rotates():
    f = random_folds(500, 10, seed=0)
    seen = []
    for k in range(10):
        tr, va, te = fold_split(f, k)
        assert np.array_equal(np.sort(np.concatenate([tr, va, te])), np.arange(500))
        assert (f[te] == k).all() and (f[va] == (k + 1) % 10).all()
        seen.append(te)
    assert np.array_equal(np.sort(np.concatenate(seen)), np.arange(500))       # every sample tested once


# ---------------------------------------------------------------- labels

def test_reduced_formula_and_chemical_system():
    assert reduced_formula(["Fe"] * 4 + ["O"] * 6) == (("Fe", 2), ("O", 3))
    assert reduced_formula(["O", "Fe", "O", "Fe", "O"]) == (("Fe", 2), ("O", 3))
    assert reduced_formula(["Cu"] * 4) == (("Cu", 1),)
    assert chemical_system(["O", "Li", "Mn", "O"]) == ("Li", "Mn", "O")


def test_system_relations():
    train = [("Li", "O"), ("Li", "Mn", "O"), ("Fe",)]
    test = [("Li", "O"), ("Mn", "O"), ("Li", "Mn", "O", "Co"), ("Fe", "O"), ("Na", "Cl")]
    r = system_relations(train, test)
    assert r["same"].tolist() == [True, False, False, False, False]
    assert r["larger"].tolist() == [True, True, False, False, False]      # Li-Mn-O contains Li-O and Mn-O
    assert r["smaller"].tolist() == [False, False, True, False, False]    # unary Fe does not count
    assert r["none"].tolist() == [False, False, False, True, True]


FCC = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float) / 2
CUBIC = np.eye(3)


def hexagonal(a, c):
    return np.array([[a, 0, 0], [-a / 2, a * 3 ** 0.5 / 2, 0], [0, 0, c]])


def rocksalt(a, A, B):
    return a * FCC, [[0, 0, 0], [0.5, 0.5, 0.5]], [A, B]


def zincblende(a, A, B):
    return a * FCC, [[0, 0, 0], [0.25, 0.25, 0.25]], [A, B]


def wurtzite(a, c, A, B, u=0.375):
    return (hexagonal(a, c), [[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0.5], [1 / 3, 2 / 3, u], [2 / 3, 1 / 3, 0.5 + u]],
            [A, A, B, B])


def perovskite(a, A, B, X):
    return a * CUBIC, [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], [A, B, X, X, X]


def full_heusler(a, X, Y, Z):
    return a * FCC, [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]], [Y, Z, X, X]


def half_heusler(a, X, Y, Z):
    return a * FCC, [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]], [Y, Z, X]


STRUCTURES = {
    "NaCl": rocksalt(5.64, "Na", "Cl"), "MgO": rocksalt(4.21, "Mg", "O"), "KCl": rocksalt(6.29, "K", "Cl"),
    "CsCl": (4.12 * CUBIC, [[0, 0, 0], [0.5, 0.5, 0.5]], ["Cs", "Cl"]),
    "GaAs zincblende": zincblende(5.65, "Ga", "As"), "ZnS zincblende": zincblende(5.41, "Zn", "S"),
    "ZnO wurtzite": wurtzite(3.25, 5.21, "Zn", "O"), "GaN wurtzite": wurtzite(3.19, 5.19, "Ga", "N", u=0.377),
    "Si diamond": zincblende(5.43, "Si", "Si"), "C diamond": zincblende(3.57, "C", "C"),
    "Cu fcc": (3.61 * FCC, [[0, 0, 0]], ["Cu"]), "Al fcc": (4.05 * FCC, [[0, 0, 0]], ["Al"]),
    "Fe bcc": (2.87 * CUBIC, [[0, 0, 0], [0.5, 0.5, 0.5]], ["Fe", "Fe"]),
    "SrTiO3": perovskite(3.905, "Sr", "Ti", "O"), "BaZrO3": perovskite(4.19, "Ba", "Zr", "O"),
    "Cu3Au L1_2": (3.75 * CUBIC, [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], ["Au", "Cu", "Cu", "Cu"]),
    "ReO3": (3.75 * CUBIC, [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]], ["Re", "O", "O", "O"]),
    "Cu2MnAl": full_heusler(5.95, "Cu", "Mn", "Al"), "Co2MnSi": full_heusler(5.65, "Co", "Mn", "Si"),
    "MgAgAs half-Heusler": half_heusler(6.24, "Ag", "Mg", "As"),
    "TiNiSn half-Heusler": half_heusler(5.93, "Ni", "Ti", "Sn"),
}
SAME_TYPE = [("NaCl", "MgO", "KCl"), ("GaAs zincblende", "ZnS zincblende"), ("ZnO wurtzite", "GaN wurtzite"),
             ("Si diamond", "C diamond"), ("Cu fcc", "Al fcc"), ("SrTiO3", "BaZrO3"), ("Cu2MnAl", "Co2MnSi"),
             ("MgAgAs half-Heusler", "TiNiSn half-Heusler")]


def label(name):
    lat, frac, sp = STRUCTURES[name]
    return prototype_label(lat, np.array(frac, float), sp)


@pytest.mark.parametrize("names", SAME_TYPE)
def test_same_structure_type_gets_the_same_label(names):
    assert len({label(n) for n in names}) == 1


def test_different_structure_types_get_different_labels():
    representatives = [names[0] for names in SAME_TYPE] + ["CsCl", "Fe bcc", "Cu3Au L1_2", "ReO3"]
    labels = [label(n) for n in representatives]
    assert len(set(labels)) == len(labels)


@pytest.mark.parametrize("name, expected", [
    ("NaCl", (225, ((1, ("a",)), (1, ("b",))))),
    ("GaAs zincblende", (216, ((1, ("a",)), (1, ("c",))))),
    ("Si diamond", (227, ((1, ("a",)),))),
    ("Cu fcc", (225, ((1, ("a",)),))),
    ("Fe bcc", (229, ((1, ("a",)),))),
    ("CsCl", (221, ((1, ("a",)), (1, ("b",))))),
    ("SrTiO3", (221, ((1, ("a",)), (1, ("b",)), (3, ("c",))))),
    ("Cu3Au L1_2", (221, ((1, ("a",)), (3, ("c",))))),
    ("ReO3", (221, ((1, ("a",)), (3, ("d",))))),
    ("Cu2MnAl", (225, ((1, ("a",)), (1, ("b",)), (2, ("c",))))),
    ("ZnO wurtzite", (186, ((1, ("b",)), (1, ("b",))))),
])
def test_labels_match_the_international_tables(name, expected):
    # Wyckoff positions from the International Tables, Vol. A (smallest equivalent letters)
    assert label(name) == expected


@pytest.mark.parametrize("name", sorted(STRUCTURES))
def test_label_is_invariant_to_description(name):
    lat, frac, sp = STRUCTURES[name]
    frac = np.array(frac, float)
    ref = label(name)
    rng = np.random.default_rng(0)
    rot = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    for trial in range(12):
        p = rng.permutation(len(sp))
        f = (frac[p] + rng.random(3)) % 1.0                    # origin shift and atom order
        if trial % 2:
            f = (-f) % 1.0                                      # inversion through the origin
        assert prototype_label(lat @ rot.T, f, [sp[i] for i in p]) == ref
    super_frac = np.vstack([(frac + [i, 0, 0]) / [2, 1, 1] for i in range(2)])
    super_lat = lat * np.array([[2], [1], [1]])
    assert prototype_label(super_lat, super_frac, sp + sp) == ref
    renamed = {s: t for s, t in zip(sorted(set(sp)), ["Xe", "Kr", "Ar"], strict=False)}
    assert prototype_label(lat, frac, [renamed[s] for s in sp]) == ref


def test_label_tolerates_small_displacements():
    lat, frac, sp = STRUCTURES["SrTiO3"]
    noise = np.random.default_rng(1).normal(scale=0.005 / 3.905, size=(len(sp), 3))
    assert prototype_label(lat, np.array(frac) + noise, sp) == label("SrTiO3")


def test_wyckoff_sets_are_groups_of_permutations():
    table = wyckoff_sets()
    assert sorted(table) == list(range(1, 231))
    for sg, perms in table.items():
        letters = sorted(perms[0], key=WYCKOFF_ORDER.index)
        assert letters == list(WYCKOFF_ORDER[:len(letters)]), sg
        as_tuples = {tuple(p[x] for x in letters) for p in perms}
        assert tuple(letters) in as_tuples, sg                                 # identity
        assert all(sorted(t) == sorted(letters) for t in as_tuples), sg        # bijections
        for p, q in permutations(perms, 2):                                    # closed under composition
            assert tuple(p[q[x]] for x in letters) in as_tuples, sg


# ---------------------------------------------------------------- distance to the training set

def test_distance_matches_brute_force():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(300, 7)) * [1, 2, 3, 4, 5, 6, 7] + 10
    B = rng.normal(size=(40, 7)) * [1, 2, 3, 4, 5, 6, 7] + 10
    A[5, 2] = np.nan
    B[3, 4] = np.nan
    med = np.nanmedian(A, axis=0)
    Af, Bf = np.where(np.isnan(A), med, A), np.where(np.isnan(B), med, B)
    mu, sd = Af.mean(0), Af.std(0)
    Az, Bz = (Af - mu) / sd, (Bf - mu) / sd
    d = np.sqrt(((Bz[:, None, :] - Az[None, :, :]) ** 2).sum(-1))
    expected = np.sort(d, axis=1)[:, :5].mean(1) / np.sqrt(7)
    assert np.allclose(distance_to_train(A, B, k=5, chunk=16), expected)


def test_distance_properties():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(200, 4))
    A[:, 3] = 2.0                                                    # a constant feature is harmless
    assert np.allclose(distance_to_train(A, A[:10], k=1), 0, atol=1e-6)
    B = rng.normal(size=(20, 4))
    scale = np.array([1e3, 1e-3, 5, 1])
    assert np.allclose(distance_to_train(A * scale, B * scale), distance_to_train(A, B))
    assert np.allclose(distance_to_train(A[rng.permutation(200)], B), distance_to_train(A, B))
    far = distance_to_train(A, A[:5] + 10)
    near = distance_to_train(A, A[:5] + 0.01)
    assert (far > near).all()


def test_combinations_helper_sanity():
    # the subsets counted by system_relations for a quaternary: 6 binaries and 4 ternaries
    s = ("Co", "Li", "Mn", "O")
    assert sum(1 for k in range(2, len(s)) for _ in combinations(s, k)) == 10
