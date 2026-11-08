"""Tests for materials_ml_primer.descriptors: hand calculations, element data, geometry."""

import re

import numpy as np
import pytest

from materials_ml_primer import graphs
from materials_ml_primer.descriptors import (
    CRYSTAL_SYSTEMS,
    MAGPIE_PROPERTIES,
    STATISTICS,
    composition_feature_names,
    composition_features,
    crystal_system,
    magpie_table,
    nearest_neighbors,
    structure_feature_names,
    structure_features,
)

NAMES = composition_feature_names()


def feats(comp):
    return dict(zip(NAMES, composition_features(comp), strict=True))


def stats(comp, prop):
    f = feats(comp)
    return [f[f"{prop}_{s}"] for s in STATISTICS]


# ---------------------------------------------------------------------------
# Element table
# ---------------------------------------------------------------------------


def value(symbol, prop):
    index, table = magpie_table()
    return table[index[symbol], MAGPIE_PROPERTIES.index(prop)]


def test_sentinels_become_nan():
    for s in ("He", "Ne", "Ar"):
        assert np.isnan(value(s, "Electronegativity"))
    assert np.isnan(value("He", "MeltingT"))
    _, table = magpie_table()
    finite = table[np.isfinite(table)]
    assert (finite > -999).all()


def test_space_group_number_is_not_used():
    assert "SpaceGroupNumber" not in MAGPIE_PROPERTIES


def test_nvalence_is_sum_of_shells():
    for s in ("H", "O", "Fe", "Pb", "Ce", "U"):
        total = sum(value(s, f"N{sh}Valence") for sh in "spdf")
        assert value(s, "NValence") == total


def _valence_from_configuration(configuration):
    out = dict.fromkeys("spdf", 0)
    for orb in configuration.split("."):
        m = re.fullmatch(r"(\d)([spdf])(\d+)", orb)
        if m:
            out[m.group(2)] += int(m.group(3))
    return out


# Elements without stable isotopes have no standard atomic weight; tables give the
# mass number of one isotope. Magpie uses the longest-lived isotope (the IUPAC
# convention), pymatgen differs for these two (210Po, 220Rn).
LONGEST_LIVED_ISOTOPE = {84: 209.0, 86: 222.0}


@pytest.mark.parametrize("z", range(1, 95))
def test_table_agrees_with_pymatgen(z):
    from pymatgen.core import Element

    el = Element.from_Z(z)
    s = el.symbol
    assert value(s, "Number") == z
    if z in LONGEST_LIVED_ISOTOPE:
        assert value(s, "AtomicWeight") == LONGEST_LIVED_ISOTOPE[z]
    else:
        assert value(s, "AtomicWeight") == pytest.approx(float(el.atomic_mass), rel=1e-3)
    assert value(s, "Row") == el.row
    assert value(s, "Column") == el.group
    shells = _valence_from_configuration(el.electronic_structure)
    for sh in "spdf":
        assert value(s, f"N{sh}Valence") == shells[sh], (s, sh)


@pytest.mark.parametrize("symbol, expected", [
    ("H", 2.20), ("O", 3.44), ("F", 3.98), ("Na", 0.93), ("Cl", 3.16), ("Fe", 1.83), ("Cs", 0.79),
])
def test_pauling_electronegativity_reference_values(symbol, expected):
    assert value(symbol, "Electronegativity") == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Composition features: hand calculations and invariances
# ---------------------------------------------------------------------------


def test_feature_count():
    assert len(NAMES) == 1 + 5 + 6 * len(MAGPIE_PROPERTIES) + 4
    assert len(composition_features({"Fe": 2, "O": 3})) == len(NAMES)


def test_nacl_by_hand():
    # Na: Z = 11, chi = 0.93, s1;  Cl: Z = 17, chi = 3.16, s2 p5;  x = 1/2 each
    assert stats({"Na": 1, "Cl": 1}, "Number") == pytest.approx([14, 3, 11, 17, 6, 11])
    assert stats({"Na": 1, "Cl": 1}, "Electronegativity") == pytest.approx(
        [2.045, 1.115, 0.93, 3.16, 2.23, 0.93])
    f = feats({"Na": 1, "Cl": 1})
    assert f["n_elements"] == 2
    assert f["frac_L2"] == pytest.approx(np.sqrt(0.5))
    assert f["frac_L3"] == pytest.approx(0.25 ** (1 / 3))
    assert f["frac_L10"] == pytest.approx((2 * 0.5 ** 10) ** 0.1)
    # mean valence electrons: s = 1.5, p = 2.5, d = f = 0
    assert [f[f"frac_{sh}_valence"] for sh in "spdf"] == pytest.approx([0.375, 0.625, 0, 0])


def test_fe2o3_by_hand():
    # x_Fe = 0.4, x_O = 0.6: mean Z = 15.2, avg_dev = 0.4 * 10.8 + 0.6 * 7.2 = 8.64
    assert stats({"Fe": 2, "O": 3}, "Number") == pytest.approx([15.2, 8.64, 8, 26, 18, 8])


def test_single_element():
    f = feats({"Fe": 1})
    assert f["n_elements"] == 1
    assert all(f[f"frac_L{p}"] == pytest.approx(1.0) for p in (2, 3, 5, 7, 10))
    for prop in MAGPIE_PROPERTIES:
        mean, avg_dev, vmin, vmax, vrange, mode = stats({"Fe": 1}, prop)
        assert mean == vmin == vmax == mode == pytest.approx(value("Fe", prop))
        assert avg_dev == vrange == 0
    # Fe: s2 d6
    assert [f[f"frac_{sh}_valence"] for sh in "spdf"] == pytest.approx([0.25, 0, 0.75, 0])


def test_invariant_to_scaling_and_order():
    a = composition_features({"Fe": 2, "O": 3})
    b = composition_features({"O": 6, "Fe": 4})
    c = composition_features({"O": 1.5, "Fe": 1.0})
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a, c, rtol=0, atol=1e-12)


def test_mode_ties_take_smallest_value():
    # equal fractions: the mode is the smaller of the two values
    assert stats({"Na": 1, "Cl": 1}, "Number")[5] == 11
    assert stats({"Cl": 1, "Na": 1}, "Number")[5] == 11


def test_missing_value_gives_nan_only_for_that_property():
    f = feats({"Ar": 1, "F": 2})
    assert all(np.isnan(f[f"Electronegativity_{s}"]) for s in STATISTICS)
    others = [v for k, v in f.items() if not k.startswith("Electronegativity_")]
    assert np.isfinite(others).all()


def test_rejects_non_positive_amounts():
    with pytest.raises(ValueError):
        composition_features({"Fe": 1, "O": 0})


# ---------------------------------------------------------------------------
# Structure features
# ---------------------------------------------------------------------------

A = 3.615  # angstrom


@pytest.mark.parametrize("kind, d1, coordination", [
    ("sc", A, 6),
    ("bcc", A * np.sqrt(3) / 2, 8),
    ("fcc", A / np.sqrt(2), 12),
    ("diamond", A * np.sqrt(3) / 4, 4),
    ("rocksalt", A / 2, 6),
    ("hcp", A, 12),
])
def test_nearest_neighbors_closed_form(kind, d1, coordination):
    lattice, frac, _ = graphs.primitive_cell(kind, a=A)
    dist, coord = nearest_neighbors(lattice, frac)
    np.testing.assert_allclose(dist, d1, rtol=1e-12)
    assert (coord == coordination).all()


def test_search_radius_grows_past_start_cutoff():
    # simple cubic with a = 6 angstrom: no neighbor within the 4 angstrom start radius
    dist, coord = nearest_neighbors(6.0 * np.eye(3), np.zeros((1, 3)))
    assert dist[0] == pytest.approx(6.0)
    assert coord[0] == 6


def test_density_of_fcc_copper_closed_form():
    lattice, frac, _ = graphs.primitive_cell("fcc", a=A)
    f = dict(zip(structure_feature_names(), structure_features(lattice, frac, ["Cu"], 225), strict=True))
    volume = A ** 3 / 4
    assert f["volume_per_atom"] == pytest.approx(volume)
    assert f["density"] == pytest.approx(value("Cu", "AtomicWeight") * 1.66053906660 / volume)
    assert f["coordination_mean"] == 12
    assert f["crystal_system_cubic"] == 1 and sum(f[f"crystal_system_{c}"] for c in CRYSTAL_SYSTEMS) == 1


def test_supercell_changes_only_the_number_of_sites():
    lattice, frac, _ = graphs.primitive_cell("rocksalt", a=5.64)
    species = ["Na", "Cl"]
    base = structure_features(lattice, frac, species, 225)
    shifts = np.array([[0, 0, 0], [1, 0, 0]])
    big_frac = np.concatenate([(frac + s) / [2, 1, 1] for s in shifts])
    big_lattice = lattice * np.array([[2], [1], [1]])
    big = structure_features(big_lattice, big_frac, species * 2, 225)
    names = structure_feature_names()
    for name, a, b in zip(names, base, big, strict=True):
        if name == "n_sites":
            assert b == 2 * a
        else:
            assert b == pytest.approx(a, rel=1e-12, abs=1e-12), name


@pytest.mark.parametrize("spg, system", [
    (1, "triclinic"), (2, "triclinic"), (3, "monoclinic"), (15, "monoclinic"),
    (16, "orthorhombic"), (74, "orthorhombic"), (75, "tetragonal"), (142, "tetragonal"),
    (143, "trigonal"), (167, "trigonal"), (168, "hexagonal"), (194, "hexagonal"),
    (195, "cubic"), (230, "cubic"),
])
def test_crystal_system_boundaries(spg, system):
    assert CRYSTAL_SYSTEMS[crystal_system(spg)] == system
    assert CRYSTAL_SYSTEMS[crystal_system(str(spg))] == system


@pytest.mark.parametrize("bad", [0, 231])
def test_crystal_system_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        crystal_system(bad)
