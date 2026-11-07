"""Tests for materials_ml_primer.graphs: closed-form shells, pymatgen, invariances, features."""

import numpy as np
import pytest

from materials_ml_primer.graphs import (
    build_graph,
    edge_vectors,
    gaussian_centers,
    gaussian_expansion,
    image_bounds,
    neighbor_list,
    plane_spacings,
    primitive_cell,
    reference_shells,
    shells,
)

KINDS = ["sc", "bcc", "fcc", "hcp", "diamond", "rocksalt"]
N_SHELLS = 8

# The simple cubic lattice described by a strongly skewed cell (unimodular rows).
SKEW = np.array([[1, 0, 0], [5, 1, 0], [3, 4, 1]], dtype=float)


def random_cell(seed, n_atoms=5):
    """A random, reasonably shaped triclinic cell with coordinates outside [0, 1)."""
    rng = np.random.default_rng(seed)
    lattice = 3.0 * np.eye(3) + rng.uniform(-1.0, 1.0, (3, 3))
    frac = rng.uniform(-0.5, 1.5, (n_atoms, 3))
    return lattice, frac


def edge_set(centers, neighbors, images):
    return {(int(i), int(j), *map(int, n)) for i, j, n in zip(centers, neighbors, images)}


def random_rotation(seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


# --------------------------------------------------------------------------
# Lattice geometry
# --------------------------------------------------------------------------

def test_plane_spacings_cubic():
    np.testing.assert_allclose(plane_spacings(3.7 * np.eye(3)), [3.7] * 3, rtol=1e-15)


def test_plane_spacings_hexagonal():
    a, c = 3.2, 5.1
    lat = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    np.testing.assert_allclose(plane_spacings(lat), [a * np.sqrt(3) / 2, a * np.sqrt(3) / 2, c],
                               rtol=1e-14)


@pytest.mark.parametrize("seed", range(3))
def test_plane_spacings_equal_volume_over_area(seed):
    lat, _ = random_cell(seed)
    vol = abs(np.linalg.det(lat))
    areas = [np.linalg.norm(np.cross(lat[1], lat[2])), np.linalg.norm(np.cross(lat[2], lat[0])),
             np.linalg.norm(np.cross(lat[0], lat[1]))]
    np.testing.assert_allclose(plane_spacings(lat), vol / np.array(areas), rtol=1e-13)


@pytest.mark.parametrize("seed", range(4))
def test_image_bounds_are_enough(seed):
    """Searching one more image layer in every direction finds nothing new."""
    lat, frac = random_cell(seed)
    cutoff = 7.0
    base = neighbor_list(lat, frac, cutoff)
    wider = neighbor_list(lat, frac, cutoff, bounds=image_bounds(lat, cutoff) + 1)
    assert edge_set(*base[:3]) == edge_set(*wider[:3])


# --------------------------------------------------------------------------
# Closed-form coordination shells
# --------------------------------------------------------------------------

TEXTBOOK = {
    # (radii in units of a, counts): standard tables for the first shells
    "sc": ([1, 2 ** .5, 3 ** .5, 2, 5 ** .5, 6 ** .5, 8 ** .5, 3], [6, 12, 8, 6, 24, 24, 12, 30]),
    "fcc": ([.5 ** .5, 1, 1.5 ** .5, 2 ** .5, 2.5 ** .5, 3 ** .5, 3.5 ** .5, 2],
            [12, 6, 24, 12, 24, 8, 48, 6]),
    "bcc": ([3 ** .5 / 2, 1, 2 ** .5, 11 ** .5 / 2, 3 ** .5, 2, 19 ** .5 / 2, 5 ** .5],
            [8, 6, 12, 24, 8, 6, 24, 24]),
    "hcp": ([1, 2 ** .5, (8 / 3) ** .5, 3 ** .5], [12, 6, 2, 18]),
    "diamond": ([3 ** .5 / 4, 8 ** .5 / 4, 11 ** .5 / 4, 1], [4, 12, 12, 6]),
    "rocksalt": ([.5, 2 ** .5 / 2, 3 ** .5 / 2, 1, 5 ** .5 / 2, 6 ** .5 / 2], [6, 12, 8, 6, 24, 24]),
}


@pytest.mark.parametrize("kind", KINDS)
def test_reference_shells_match_textbook(kind):
    radii_tb, counts_tb = TEXTBOOK[kind]
    radii, counts, _ = reference_shells(kind, len(counts_tb))
    np.testing.assert_allclose(radii, radii_tb, rtol=1e-14)
    np.testing.assert_array_equal(counts, counts_tb)


def test_reference_rocksalt_species_alternate():
    _, counts, other = reference_shells("rocksalt", 6)
    np.testing.assert_array_equal(other, [6, 0, 8, 0, 24, 0])


@pytest.mark.parametrize("kind", KINDS)
def test_neighbor_shells_match_lattice_sums(kind):
    """Search in the primitive cell = integer lattice sums, for every atom."""
    radii, counts, other = reference_shells(kind, N_SHELLS + 1)
    a = 3.6
    cutoff = a * 0.5 * (radii[-2] + radii[-1])           # between shells 8 and 9
    lat, frac, _ = primitive_cell(kind, a=a)
    centers, neighbors, _, dist = neighbor_list(lat, frac, cutoff)
    for i in range(len(frac)):
        r, k = shells(dist[centers == i])
        np.testing.assert_allclose(r, a * radii[:N_SHELLS], rtol=1e-12)
        np.testing.assert_array_equal(k, counts[:N_SHELLS])
        # neighbors on the other sublattice (other species for rock salt)
        mask = centers == i
        d_i, j_i = dist[mask], neighbors[mask]
        other_counts = [np.sum((np.abs(d_i - a * rr) < 1e-9) & (j_i != i)) for rr in radii[:N_SHELLS]]
        np.testing.assert_array_equal(other_counts, other[:N_SHELLS])


def test_skewed_cell_gives_simple_cubic_shells():
    radii, counts, _ = reference_shells("sc", N_SHELLS + 1)
    cutoff = 0.5 * (radii[-2] + radii[-1])
    centers, _, _, dist = neighbor_list(SKEW, [[0.3, 0.6, 0.1]], cutoff)
    r, k = shells(dist)
    np.testing.assert_allclose(r, radii[:N_SHELLS], rtol=1e-12)
    np.testing.assert_array_equal(k, counts[:N_SHELLS])


def test_skewed_cell_needs_more_than_one_image():
    cutoff = 2.5
    assert image_bounds(SKEW, cutoff).max() > 1
    full = neighbor_list(SKEW, [[0.3, 0.6, 0.1]], cutoff)
    naive = neighbor_list(SKEW, [[0.3, 0.6, 0.1]], cutoff, bounds=[1, 1, 1])
    assert len(naive[0]) < len(full[0])
    assert edge_set(*naive[:3]) < edge_set(*full[:3])


# --------------------------------------------------------------------------
# Agreement with pymatgen
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(6))
def test_matches_pymatgen_edge_for_edge(seed):
    from pymatgen.core import Lattice, Structure

    lat, frac = random_cell(seed, n_atoms=4 + seed)
    cutoff = 6.0
    s = Structure(Lattice(lat), ["Si"] * len(frac), frac)
    pc, pj, pn, pd = s.get_neighbor_list(cutoff)
    c, j, n, d = neighbor_list(lat, frac, cutoff)
    assert edge_set(c, j, n) == edge_set(pc, pj, pn.astype(int))
    mine = {(int(a), int(b), *map(int, m)): x for a, b, m, x in zip(c, j, n, d)}
    theirs = {(int(a), int(b), *map(int, m)): x for a, b, m, x in zip(pc, pj, pn.astype(int), pd)}
    np.testing.assert_allclose([mine[e] for e in theirs], list(theirs.values()), rtol=1e-12)


# --------------------------------------------------------------------------
# Invariances
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("cell", ["random", "diamond"])
def test_rotation_invariance(seed, cell):
    """Same edges and lengths; compared as sets, since equal distances may be reordered."""
    if cell == "random":
        lat, frac = random_cell(seed)
    else:
        lat, frac, _ = primitive_cell("diamond", a=3.6)
    rot = random_rotation(seed + 10)
    c0, j0, n0, d0 = neighbor_list(lat, frac, 6.0)
    c1, j1, n1, d1 = neighbor_list(lat @ rot.T, frac, 6.0)
    e0 = {(int(a), int(b), *map(int, m)): x for a, b, m, x in zip(c0, j0, n0, d0)}
    e1 = {(int(a), int(b), *map(int, m)): x for a, b, m, x in zip(c1, j1, n1, d1)}
    assert e0.keys() == e1.keys()
    np.testing.assert_allclose([e1[k] for k in e0], list(e0.values()), rtol=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_translation_invariance(seed):
    lat, frac = random_cell(seed)
    shift = np.random.default_rng(seed).uniform(-2, 2, 3)
    c0, j0, _, d0 = neighbor_list(lat, frac, 6.0)
    c1, j1, _, d1 = neighbor_list(lat, frac + shift, 6.0)
    np.testing.assert_array_equal(c0, c1)
    np.testing.assert_allclose(d1, d0, rtol=1e-12)
    for i in range(len(frac)):
        assert sorted(j0[c0 == i]) == sorted(j1[c1 == i])


@pytest.mark.parametrize("seed", range(3))
def test_permutation_invariance(seed):
    lat, frac = random_cell(seed, n_atoms=6)
    perm = np.random.default_rng(seed).permutation(len(frac))   # new atom k = old atom perm[k]
    c0, j0, n0, _ = neighbor_list(lat, frac, 6.0)
    c1, j1, n1, _ = neighbor_list(lat, frac[perm], 6.0)
    assert edge_set(perm[c1], perm[j1], n1) == edge_set(c0, j0, n0)


@pytest.mark.parametrize("seed", range(2))
def test_supercell_invariance(seed):
    """Every atom of a 2 x 1 x 3 supercell has the neighbor distances of its parent."""
    lat, frac = random_cell(seed, n_atoms=3)
    reps = np.array([2, 1, 3])
    super_lat = lat * reps[:, None]
    shifts = np.array([[a, b, c] for a in range(2) for b in range(1) for c in range(3)])
    super_frac = ((frac[None, :, :] + shifts[:, None, :]) / reps).reshape(-1, 3)
    parent = np.tile(np.arange(len(frac)), len(shifts))
    c0, _, _, d0 = neighbor_list(lat, frac, 6.0)
    c1, _, _, d1 = neighbor_list(super_lat, super_frac, 6.0)
    for k in range(len(super_frac)):
        np.testing.assert_allclose(np.sort(d1[c1 == k]), np.sort(d0[c0 == parent[k]]), rtol=1e-12)


# --------------------------------------------------------------------------
# Graph assembly
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(3))
def test_full_graph_is_symmetric(seed):
    """Without truncation, (i, j, n) is an edge exactly when (j, i, -n) is."""
    lat, frac = random_cell(seed)
    g = build_graph(lat, frac, 6.0)
    src, dst = g["edge_index"]
    forward = edge_set(dst, src, g["images"])
    backward = edge_set(src, dst, -g["images"])
    assert forward == backward


@pytest.mark.parametrize("seed", range(3))
def test_edge_vectors_have_edge_lengths(seed):
    lat, frac = random_cell(seed)
    g = build_graph(lat, frac, 6.0)
    np.testing.assert_allclose(np.linalg.norm(g["vectors"], axis=1), g["distances"], rtol=1e-12)
    src, dst = g["edge_index"]
    np.testing.assert_allclose(edge_vectors(lat, frac, dst, src, g["images"]), g["vectors"])


def test_max_neighbors_bcc_tie():
    """bcc within 1.2 a: 8 + 6 neighbors; keeping 12 takes 4 of the 6 equidistant ones."""
    lat, frac, _ = primitive_cell("bcc", a=2.87)
    g = build_graph(lat, frac, 1.2 * 2.87, max_neighbors=12)
    d = np.sort(g["distances"])
    np.testing.assert_allclose(d[:8], 2.87 * np.sqrt(3) / 2, rtol=1e-12)
    np.testing.assert_allclose(d[8:], 2.87, rtol=1e-12)
    assert len(d) == 12


def test_max_neighbors_keeps_nearest():
    lat, frac = random_cell(0, n_atoms=6)
    full = build_graph(lat, frac, 6.0)
    trunc = build_graph(lat, frac, 6.0, max_neighbors=5)
    for i in range(len(frac)):
        d_full = np.sort(full["distances"][full["edge_index"][1] == i])
        d_trunc = np.sort(trunc["distances"][trunc["edge_index"][1] == i])
        np.testing.assert_allclose(d_trunc, d_full[:5])


# --------------------------------------------------------------------------
# Distance features
# --------------------------------------------------------------------------

def test_cgcnn_gaussian_centers():
    mu = gaussian_centers()
    assert len(mu) == 41
    np.testing.assert_allclose(mu, np.arange(0, 8.0 + 0.2, 0.2), atol=1e-12)


def test_gaussian_expansion_values():
    mu = gaussian_centers()
    f = gaussian_expansion([2.0, 3.1], mu, 0.2)
    assert f.shape == (2, 41)
    assert f[0, 10] == pytest.approx(1.0)                   # d on a center
    assert gaussian_expansion([2.2], [2.0], 0.2)[0, 0] == pytest.approx(np.exp(-1.0), rel=1e-12)
    assert np.argmax(f[1]) in (15, 16)


def test_shells_grouping():
    r, k = shells([1.0, 1.0 + 1e-9, 2.0, 2.0, 2.0, 3.5])
    np.testing.assert_allclose(r, [1.0, 2.0, 3.5], atol=1e-8)
    np.testing.assert_array_equal(k, [2, 3, 1])
