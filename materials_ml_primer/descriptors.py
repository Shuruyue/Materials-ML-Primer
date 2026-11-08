"""Composition (Magpie-style) and simple structure descriptors.

Composition features follow Ward et al., npj Comput. Mater. 2, 16028 (2016):
for each elemental property p and atomic fractions x_i of a compound,

    mean    = sum_i x_i p_i
    avg_dev = sum_i x_i |p_i - mean|
    min, max, range = max - min      (over the elements present)
    mode    = p of the element with the largest fraction (ties: smallest p)

plus stoichiometric attributes (number of elements and the L_p norms of the
fractions) and the fractions of valence electrons in s, p, d and f shells.

The elemental values are the Magpie table shipped with jarvis-tools. Two
corrections are applied: the table codes missing values as -9999 (the
electronegativity of He, Ne and Ar, the melting point of He), which are
turned into NaN, and the ground-state space-group number of the element is
left out, because a space-group number is a label, not a quantity, and its
mean or spread has no meaning. NValence, the sum of the four valence counts,
is computed. A compound containing an element with a missing value gets NaN
for all statistics of that property.

Structure features are the number of sites, volume per atom, density, the
crystal system, and nearest-neighbor distances and coordination numbers from
the neighbor search of ``materials_ml_primer.graphs``.
"""

from __future__ import annotations

import functools

import numpy as np

from materials_ml_primer import graphs

MAGPIE_PROPERTIES = (
    "Number", "MendeleevNumber", "AtomicWeight", "MeltingT", "Column", "Row",
    "CovalentRadius", "Electronegativity",
    "NsValence", "NpValence", "NdValence", "NfValence", "NValence",
    "NsUnfilled", "NpUnfilled", "NdUnfilled", "NfUnfilled", "NUnfilled",
    "GSvolume_pa", "GSbandgap", "GSmagmom",
)
STATISTICS = ("mean", "avg_dev", "min", "max", "range", "mode")
LP_NORMS = (2, 3, 5, 7, 10)
SHELLS = ("s", "p", "d", "f")
CRYSTAL_SYSTEMS = ("triclinic", "monoclinic", "orthorhombic", "tetragonal",
                   "trigonal", "hexagonal", "cubic")
_SENTINEL = -999.0
_AMU_G_PER_A3_TO_G_PER_CM3 = 1.66053906660  # 1 amu / 1 angstrom^3 in g/cm^3


@functools.lru_cache(maxsize=1)
def magpie_table():
    """Elemental property table: ({symbol: row}, (n_elements, n_properties) array)."""
    from jarvis.core.specie import chem_data_magpie

    symbols = sorted(chem_data_magpie)
    index = {s: k for k, s in enumerate(symbols)}
    table = np.full((len(symbols), len(MAGPIE_PROPERTIES)), np.nan)
    for s in symbols:
        row = chem_data_magpie[s]
        for j, name in enumerate(MAGPIE_PROPERTIES):
            if name == "NValence":
                value = sum(row[f"N{sh}Valence"] for sh in SHELLS)
            else:
                value = row[name]
            if value is not None and value > _SENTINEL:
                table[index[s], j] = float(value)
    return index, table


def composition_feature_names():
    names = ["n_elements"] + [f"frac_L{p}" for p in LP_NORMS]
    names += [f"{prop}_{stat}" for prop in MAGPIE_PROPERTIES for stat in STATISTICS]
    names += [f"frac_{sh}_valence" for sh in SHELLS]
    return names


def _weighted_stats(x, values):
    """The six statistics of one property; x are fractions summing to 1."""
    if np.isnan(values).any():
        return [np.nan] * len(STATISTICS)
    mean = float(np.dot(x, values))
    avg_dev = float(np.dot(x, np.abs(values - mean)))
    vmin, vmax = float(values.min()), float(values.max())
    mode = float(values[x == x.max()].min())
    return [mean, avg_dev, vmin, vmax, vmax - vmin, mode]


def composition_features(comp):
    """Feature vector of a composition {symbol: amount} (any positive amounts)."""
    index, table = magpie_table()
    symbols = sorted(comp)
    amounts = np.array([float(comp[s]) for s in symbols])
    if (amounts <= 0).any():
        raise ValueError("amounts must be positive")
    x = amounts / amounts.sum()
    values = table[[index[s] for s in symbols]]

    out = [float(len(symbols))]
    out += [float(np.sum(x ** p) ** (1.0 / p)) for p in LP_NORMS]
    for j in range(len(MAGPIE_PROPERTIES)):
        out += _weighted_stats(x, values[:, j])
    shells = [MAGPIE_PROPERTIES.index(f"N{sh}Valence") for sh in SHELLS]
    mean_shell = x @ values[:, shells]
    total = mean_shell.sum()
    out += list(mean_shell / total) if total > 0 else [np.nan] * len(SHELLS)
    return np.array(out)


def crystal_system(spg_number):
    """Index into CRYSTAL_SYSTEMS of a space-group number 1..230."""
    n = int(spg_number)
    if not 1 <= n <= 230:
        raise ValueError(f"space-group number out of range: {spg_number!r}")
    for k, upper in enumerate((2, 15, 74, 142, 167, 194, 230)):
        if n <= upper:
            return k
    raise AssertionError("unreachable")


def structure_feature_names():
    names = ["n_sites", "volume_per_atom", "density",
             "nn_distance_mean", "nn_distance_min", "nn_distance_max", "coordination_mean"]
    return names + [f"crystal_system_{c}" for c in CRYSTAL_SYSTEMS]


def nearest_neighbors(lattice, frac_coords, tol=0.1, start_cutoff=4.0):
    """Nearest-neighbor distance d1 of every atom and its coordination number.

    The coordination number counts the neighbors within (1 + tol) * d1. The
    search radius starts at ``start_cutoff`` and grows until every atom has a
    neighbor and the radius covers (1 + tol) * d1 for all atoms.
    """
    n = len(frac_coords)
    cutoff = start_cutoff
    while True:
        centers, _, _, d = graphs.neighbor_list(lattice, frac_coords, cutoff)
        if np.unique(centers).size == n:
            d1 = np.full(n, np.inf)
            np.minimum.at(d1, centers, d)
            needed = (1.0 + tol) * d1.max()
            if needed <= cutoff:
                close = d <= (1.0 + tol) * d1[centers]
                return d1, np.bincount(centers[close], minlength=n)
            cutoff = needed
        else:
            cutoff *= 1.5


def structure_features(lattice, frac_coords, species, spg_number, tol=0.1):
    """Feature vector of a crystal structure (lattice rows and lengths in angstrom)."""
    index, table = magpie_table()
    n = len(species)
    volume = abs(float(np.linalg.det(np.asarray(lattice, dtype=float))))
    mass = sum(table[index[s], MAGPIE_PROPERTIES.index("AtomicWeight")] for s in species)
    d1, coordination = nearest_neighbors(lattice, frac_coords, tol=tol)
    one_hot = np.zeros(len(CRYSTAL_SYSTEMS))
    one_hot[crystal_system(spg_number)] = 1.0
    head = [float(n), volume / n, mass * _AMU_G_PER_A3_TO_G_PER_CM3 / volume,
            float(d1.mean()), float(d1.min()), float(d1.max()), float(coordination.mean())]
    return np.concatenate([head, one_hot])


def jarvis_feature_matrices(records):
    """Composition and structure feature matrices of JARVIS records (one row per record)."""
    from materials_ml_primer import data

    comp = np.array([composition_features(data.composition(r["atoms"]["elements"]))
                     for r in records])
    struct = np.array([structure_features(*data.structure_arrays(r["atoms"]), r["spg_number"])
                       for r in records])
    return comp, struct
