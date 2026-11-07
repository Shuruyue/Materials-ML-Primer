"""Crystal structures as graphs: periodic neighbor lists and distance features.

A crystal is described by a lattice matrix ``L`` whose rows are the lattice
vectors a1, a2, a3 (in angstrom), fractional coordinates ``F`` (one row per
atom) and a list of species. Cartesian positions are ``X = F @ L``.

Atom j is a neighbor of atom i if one of its periodic images is within the
cutoff radius:

    |X_j + n @ L - X_i| <= r_c,   n in Z^3,   (i, j, n) != (i, i, 0).

Each such triple (i, j, n) becomes a directed edge j -> i of the graph.
"""

from fractions import Fraction
from itertools import product

import numpy as np


# --------------------------------------------------------------------------
# Lattice geometry
# --------------------------------------------------------------------------

def reciprocal_rows(lattice):
    """Rows b1, b2, b3 with a_i . b_j = delta_ij (no factor 2 pi)."""
    return np.linalg.inv(np.asarray(lattice, dtype=float)).T


def plane_spacings(lattice):
    """Spacing d_i = 1 / |b_i| of the lattice planes spanned by the other two vectors.

    Equivalently d_1 = V / |a2 x a3|, and cyclically.
    """
    return 1.0 / np.linalg.norm(reciprocal_rows(lattice), axis=1)


def image_bounds(lattice, cutoff):
    """Smallest integers N_i such that every neighbor has |n_i| <= N_i.

    For wrapped fractional coordinates (in [0, 1)) the difference
    Delta f_i lies in (-1, 1). The component of a displacement along the unit
    normal b_i/|b_i| is (Delta f_i + n_i) d_i, which must not exceed the
    cutoff, so |n_i| < r_c/d_i + 1, i.e. |n_i| <= ceil(r_c/d_i).
    """
    return np.ceil(cutoff / plane_spacings(lattice)).astype(int)


# --------------------------------------------------------------------------
# Neighbor search
# --------------------------------------------------------------------------

def neighbor_list(lattice, frac_coords, cutoff, bounds=None):
    """All neighbor triples (i, j, n) within the cutoff, by direct search.

    Parameters
    ----------
    lattice : (3, 3) array, rows are lattice vectors (angstrom).
    frac_coords : (N, 3) array of fractional coordinates (any range).
    cutoff : float, neighbor radius r_c (angstrom); pairs with d <= r_c count.
    bounds : optional (3,) ints, image range |n_i| <= bounds[i] to search.
        Defaults to ``image_bounds(lattice, cutoff)``, which is always enough;
        smaller values are only useful to show what a too small search misses.

    Returns
    -------
    centers, neighbors : (E,) int arrays (i and j).
    images : (E, 3) int array n, relative to the coordinates as given.
    distances : (E,) float array.
    Edges are sorted by center, then distance, then neighbor, then image.
    """
    L = np.asarray(lattice, dtype=float)
    F = np.atleast_2d(np.asarray(frac_coords, dtype=float))
    shift = np.floor(F)
    X = (F - shift) @ L                                   # wrapped positions
    if bounds is None:
        bounds = image_bounds(L, cutoff)
    ranges = [np.arange(-b, b + 1) for b in np.asarray(bounds, dtype=int)]
    images = np.array(list(product(*ranges)), dtype=int)  # (M, 3)

    n_atoms = len(X)
    diff = X[None, :, :] - X[:, None, :]                  # diff[i, j] = X_j - X_i
    chunk = max(1, 2_000_000 // max(1, n_atoms * n_atoms))
    found = []
    for start in range(0, len(images), chunk):
        imgs = images[start:start + chunk]
        disp = diff[None, :, :, :] + (imgs @ L)[:, None, None, :]
        dist = np.sqrt(np.einsum("mijk,mijk->mij", disp, disp))
        m, i, j = np.nonzero(dist <= cutoff)
        self_image = (i == j) & np.all(imgs[m] == 0, axis=1)
        keep = ~self_image
        found.append((i[keep], j[keep], imgs[m[keep]], dist[m[keep], i[keep], j[keep]]))

    centers = np.concatenate([f[0] for f in found])
    neighbors = np.concatenate([f[1] for f in found])
    imgs = np.concatenate([f[2] for f in found])
    distances = np.concatenate([f[3] for f in found])
    # Back to the caller's coordinates: X_j + n L - X_i is unchanged when
    # n -> n - floor(F_j) + floor(F_i).
    imgs = imgs - shift[neighbors].astype(int) + shift[centers].astype(int)

    order = np.lexsort((imgs[:, 2], imgs[:, 1], imgs[:, 0], neighbors, distances, centers))
    return centers[order], neighbors[order], imgs[order], distances[order]


def edge_vectors(lattice, frac_coords, centers, neighbors, images):
    """Cartesian vectors X_j + n L - X_i of each edge (angstrom)."""
    L = np.asarray(lattice, dtype=float)
    X = np.asarray(frac_coords, dtype=float) @ L
    return X[neighbors] + images @ L - X[centers]


def shells(distances, tol=1e-6):
    """Group distances into coordination shells.

    Returns the shell radii (mean distance in each group) and the number of
    neighbors in each shell. Distances closer than ``tol`` belong together.
    """
    d = np.sort(np.asarray(distances, dtype=float))
    if d.size == 0:
        return np.array([]), np.array([], dtype=int)
    breaks = np.nonzero(np.diff(d) > tol)[0] + 1
    groups = np.split(d, breaks)
    return np.array([g.mean() for g in groups]), np.array([len(g) for g in groups])


# --------------------------------------------------------------------------
# Graph assembly and distance features
# --------------------------------------------------------------------------

def build_graph(lattice, frac_coords, cutoff, max_neighbors=None):
    """Directed graph with an edge j -> i for every neighbor j of center i.

    With ``max_neighbors`` only the nearest ``max_neighbors`` neighbors of each
    center are kept, as in CGCNN. When several neighbors are equally far at the
    truncation point, the choice among them follows the sort order of
    ``neighbor_list`` (distance, then neighbor index, then image): the kept
    distances are well defined, the kept atoms are not.

    Returns a dict with ``edge_index`` (2, E) [source j, target i], ``images``,
    ``distances`` and ``vectors`` (X_j + n L - X_i).
    """
    centers, neighbors, images, distances = neighbor_list(lattice, frac_coords, cutoff)
    if max_neighbors is not None:
        rank = np.zeros(len(centers), dtype=int)
        starts = np.r_[0, np.nonzero(np.diff(centers))[0] + 1]
        counts = np.diff(np.r_[starts, len(centers)])
        for s, c in zip(starts, counts):
            rank[s:s + c] = np.arange(c)
        keep = rank < max_neighbors
        centers, neighbors = centers[keep], neighbors[keep]
        images, distances = images[keep], distances[keep]
    return {
        "edge_index": np.vstack([neighbors, centers]),
        "images": images,
        "distances": distances,
        "vectors": edge_vectors(lattice, frac_coords, centers, neighbors, images),
    }


def gaussian_centers(dmin=0.0, dmax=8.0, step=0.2):
    """Equally spaced centers dmin, dmin + step, ..., dmax (CGCNN defaults: 41 centers)."""
    n = int(round((dmax - dmin) / step)) + 1
    return np.linspace(dmin, dmax, n)


def gaussian_expansion(distances, centers, width):
    """Edge features exp(-((d - mu_k) / width)^2), one column per center mu_k.

    CGCNN uses width = step = 0.2 angstrom.
    """
    d = np.asarray(distances, dtype=float)[..., None]
    return np.exp(-((d - np.asarray(centers, dtype=float)) / width) ** 2)


# --------------------------------------------------------------------------
# Structures used for verification
# --------------------------------------------------------------------------

def primitive_cell(kind, a=1.0, c_over_a=None):
    """Lattice, fractional coordinates and species of simple structures.

    kind: 'sc', 'bcc', 'fcc', 'diamond', 'rocksalt' (cubic lattice constant a)
    or 'hcp' (in-plane constant a, default ideal c/a = sqrt(8/3)).
    """
    if kind == "sc":
        return a * np.eye(3), np.zeros((1, 3)), ["X"]
    fcc = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    if kind == "fcc":
        return fcc, np.zeros((1, 3)), ["X"]
    if kind == "bcc":
        bcc = 0.5 * a * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=float)
        return bcc, np.zeros((1, 3)), ["X"]
    if kind == "diamond":
        return fcc, np.array([[0, 0, 0], [0.25, 0.25, 0.25]]), ["X", "X"]
    if kind == "rocksalt":
        return fcc, np.array([[0, 0, 0], [0.5, 0.5, 0.5]]), ["A", "B"]
    if kind == "hcp":
        q = np.sqrt(8.0 / 3.0) if c_over_a is None else c_over_a
        lat = a * np.array([[1.0, 0.0, 0.0], [-0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, q]])
        return lat, np.array([[1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.75]]), ["X", "X"]
    raise ValueError(f"unknown structure kind: {kind}")


def reference_shells(kind, n_shells):
    """Exact coordination shells from integer lattice sums (no neighbor search).

    Counts the lattice vectors of each squared length directly, in a frame
    where they have integer (or, for hcp, rational) coordinates:

    - sc:       all integer vectors, unit a.
    - fcc:      integer vectors with even coordinate sum, unit a/2.
    - bcc:      integer vectors whose coordinates are all even or all odd, unit a/2.
    - rocksalt: all integer vectors, unit a/2; odd coordinate sum = other species.
    - diamond:  unit a/4; around an atom of sublattice A, sites of A are
                vectors with all coordinates even and sum = 0 (mod 4), sites of
                B are A + (1, 1, 1).
    - hcp:      ideal c/a = sqrt(8/3); |n1 a1 + n2 a2 + n3 c|^2 =
                a^2 (n1^2 - n1 n2 + n2^2) + c^2 n3^2 with the B site shifted by
                (1/3, 2/3, 1/2) in (n1, n2, n3), in exact fractions of a^2.

    Returns (radii in units of a, total counts, counts of the other species or
    sublattice) for the first ``n_shells`` shells.
    """
    m = 4
    while True:
        total, other, complete_below = _count_lattice_vectors(kind, m)
        keys = sorted(total)[:n_shells]
        # Every vector outside the enumerated range is longer than
        # complete_below, so shells below it are counted completely.
        if len(keys) == n_shells and keys[-1] < complete_below:
            break
        m *= 2
    radii = np.array([np.sqrt(float(k)) for k in keys])
    return radii, np.array([total[k] for k in keys]), np.array([other[k] for k in keys])


def _count_lattice_vectors(kind, m):
    """Squared lengths (exact, units of a^2) of lattice vectors with coordinates in [-m, m]."""
    other = {}
    total = {}

    def add(key, is_other):
        total[key] = total.get(key, 0) + 1
        other[key] = other.get(key, 0) + int(is_other)

    rng = range(-m, m + 1)
    if kind in ("sc", "fcc", "bcc", "rocksalt"):
        unit2 = Fraction(1) if kind == "sc" else Fraction(1, 4)
        for v in product(rng, rng, rng):
            if v == (0, 0, 0):
                continue
            s = sum(v)
            if kind == "fcc" and s % 2:
                continue
            if kind == "bcc" and len({x % 2 for x in v}) > 1:
                continue
            add(unit2 * sum(x * x for x in v), kind == "rocksalt" and s % 2 == 1)
        complete_below = unit2 * (m + 1) ** 2
    elif kind == "diamond":
        unit2 = Fraction(1, 16)
        for v in product(rng, rng, rng):
            on_a = all(x % 2 == 0 for x in v) and sum(v) % 4 == 0
            on_b = all(x % 2 == 1 for x in v) and (sum(v) - 3) % 4 == 0
            if (on_a and v != (0, 0, 0)) or on_b:
                add(unit2 * sum(x * x for x in v), on_b)
        complete_below = unit2 * (m + 1) ** 2
    elif kind == "hcp":
        c2 = Fraction(8, 3)
        for n1, n2, n3 in product(rng, rng, rng):
            if (n1, n2, n3) != (0, 0, 0):
                add(Fraction(n1 * n1 - n1 * n2 + n2 * n2) + c2 * n3 * n3, False)
            p, q, r = n1 + Fraction(1, 3), n2 + Fraction(2, 3), n3 + Fraction(1, 2)
            add(p * p - p * q + q * q + c2 * r * r, True)
        # p^2 - p q + q^2 >= (3/4) max(|p|, |q|)^2, and |p|, |q| > m - 1 outside.
        complete_below = min(Fraction(3, 4) * (m - 1) ** 2, c2 * (m - Fraction(1, 2)) ** 2)
    else:
        raise ValueError(f"unknown structure kind: {kind}")
    return total, other, complete_below


def from_structure(structure):
    """Lattice matrix, fractional coordinates and species symbols of a pymatgen Structure."""
    return (structure.lattice.matrix.copy(), structure.frac_coords.copy(),
            [site.specie.symbol for site in structure])
