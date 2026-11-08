"""Tests for materials_ml_primer.data (no download needed)."""

import math

import numpy as np

from materials_ml_primer.data import as_float, composition, deduplicate, structure_arrays


def test_cartesian_coordinates_become_fractional():
    rng = np.random.default_rng(0)
    lattice = rng.normal(size=(3, 3)) + 4 * np.eye(3)
    frac = rng.random((5, 3))
    atoms = {"lattice_mat": lattice.tolist(), "coords": (frac @ lattice).tolist(),
             "elements": ["Si"] * 5, "cartesian": True}
    lat, f, species = structure_arrays(atoms)
    np.testing.assert_allclose(lat, lattice)
    np.testing.assert_allclose(f, frac, atol=1e-12)
    assert species == ["Si"] * 5


def test_fractional_coordinates_pass_through():
    atoms = {"lattice_mat": np.eye(3).tolist(), "coords": [[0.1, 0.2, 0.3]],
             "elements": ["Cu"], "cartesian": False}
    _, f, _ = structure_arrays(atoms)
    np.testing.assert_allclose(f, [[0.1, 0.2, 0.3]])


def test_deduplicate_keeps_first_copy():
    records = [{"jid": "a", "v": 1}, {"jid": "b", "v": 2}, {"jid": "a", "v": 3}, {"jid": "a", "v": 4}]
    kept, dups = deduplicate(records)
    assert [r["v"] for r in kept] == [1, 2]
    assert dups == {"a": 3}


def test_as_float():
    assert as_float(1) == 1.0
    assert as_float("2.5") == 2.5
    assert math.isnan(as_float("na"))
    assert math.isnan(as_float(None))


def test_composition_counts():
    assert composition(["Fe", "O", "Fe", "O", "O"]) == {"Fe": 2, "O": 3}
