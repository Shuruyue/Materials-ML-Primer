"""Checks of the pretrained feature extractors. The models' weights are downloaded on first use (tens of MB each,
cached by each package); a test is skipped if its package is not installed. The tests that load a model are marked
``needs_weights``: ``pytest -m "not needs_weights"`` runs the others (CI does)."""
import importlib.util

import numpy as np
import pytest
import torch

from materials_ml_primer import pretrained

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PACKAGE = {"mace-mp-0b3": "mace", "chgnet-0.3.0": "chgnet", "orb-v3-omat": "orb_models"}
_loaded = {}
needs_weights = pytest.mark.needs_weights


def extractor(name):
    if importlib.util.find_spec(PACKAGE[name]) is None:
        pytest.skip(f"{PACKAGE[name]} is not installed")
    if name not in _loaded:
        _loaded[name] = pretrained.load(name, DEVICE)
    return _loaded[name]


def random_cell(seed, n_atoms=5):
    rng = np.random.default_rng(seed)
    lattice = np.diag(rng.uniform(4.0, 6.0, 3)) + rng.uniform(-0.5, 0.5, (3, 3))
    frac = rng.random((n_atoms, 3))
    return lattice, frac, list(rng.choice(["Li", "O", "Fe", "F", "Si"], n_atoms))


def features(name, structures):
    atoms = [pretrained.to_atoms(*s) for s in structures]
    with torch.no_grad():
        f, e = extractor(name)(atoms)
    return pretrained.summarize(f), f, e


def test_summary_is_mean_and_population_std():
    a = np.array([[1.0, 2.0], [3.0, 6.0]])
    assert np.allclose(pretrained.summarize([a]), [[2.0, 4.0, 1.0, 2.0]])


@needs_weights
@pytest.mark.parametrize("name", list(pretrained.MODELS))
def test_feature_dimension_is_the_documented_one(name):
    s, f, _ = features(name, [random_cell(0)])
    assert f[0].shape == (5, pretrained.MODELS[name]["dim"])
    assert s.shape == (1, 2 * pretrained.MODELS[name]["dim"]) and np.isfinite(s).all()


@needs_weights
@pytest.mark.parametrize("name", list(pretrained.MODELS))
def test_a_structure_gets_the_same_features_in_a_batch_and_alone(name):
    cells = [random_cell(k, n_atoms=3 + k) for k in range(4)]
    together, _, _ = features(name, cells)
    alone = np.vstack([features(name, [c])[0] for c in cells])
    assert np.allclose(together, alone, rtol=0, atol=1e-4)


@needs_weights
@pytest.mark.parametrize("name", list(pretrained.MODELS))
def test_features_ignore_atom_order_and_choice_of_cell(name):
    lattice, frac, species = random_cell(7)
    ref, _, _ = features(name, [(lattice, frac, species)])
    perm = [3, 0, 4, 1, 2]
    permuted, _, _ = features(name, [(lattice, frac[perm], [species[p] for p in perm])])
    sup_frac = np.vstack([frac * [0.5, 1, 1], (frac + [1, 0, 0]) * [0.5, 1, 1]])
    supercell, _, _ = features(name, [(lattice * [[2], [1], [1]], sup_frac, species * 2)])
    scale = np.abs(ref).max()
    assert np.abs(permuted - ref).max() < 1e-5 * scale
    assert np.abs(supercell - ref).max() < 1e-4 * scale


def orthogonal(seed, proper):
    """A random rotation (proper) or rotation combined with a reflection."""
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if (np.linalg.det(q) > 0) != proper:
        q[:, 0] *= -1
    return q


@needs_weights
@pytest.mark.parametrize("proper", [True, False], ids=["rotation", "reflection"])
@pytest.mark.parametrize("name", ["mace-mp-0b3", "chgnet-0.3.0"])
def test_features_of_invariant_models_ignore_rotations_and_reflections(name, proper):
    lattice, frac, species = random_cell(8)
    q = orthogonal(3, proper)
    assert np.isclose(np.linalg.det(q), 1.0 if proper else -1.0)
    ref, _, _ = features(name, [(lattice, frac, species)])
    moved, _, _ = features(name, [(lattice @ q.T, frac, species)])
    assert np.abs(moved - ref).max() < 1e-4 * np.abs(ref).max()


@needs_weights
def test_orb_energy_is_nearly_rotation_invariant():
    # Orb is not equivariant by construction; it learns the symmetry from data. Its energy changes little under a
    # rotation, but some of its latent features change a lot (measured in the notebook), so only the energy is
    # tested here.
    ex = extractor("orb-v3-omat")
    lattice, frac, species = random_cell(8)
    q = orthogonal(3, True)
    e = ex.energy([pretrained.to_atoms(lattice, frac, species), pretrained.to_atoms(lattice @ q.T, frac, species)])
    assert abs(e[1] - e[0]) < 5e-3


@needs_weights
def test_mace_features_are_the_calculators_descriptors_and_energy():
    ex = extractor("mace-mp-0b3")
    cell = random_cell(9)
    _, f, e = features("mace-mp-0b3", [cell])
    atoms = pretrained.to_atoms(*cell)
    assert np.allclose(f[0], ex.calc.get_descriptors(atoms), rtol=0, atol=1e-5)
    atoms.calc = ex.calc
    assert e[0] == pytest.approx(atoms.get_potential_energy() / len(atoms), abs=1e-5)


@needs_weights
def test_chgnet_features_sum_to_its_crystal_feature_and_give_its_energy():
    from pymatgen.io.ase import AseAtomsAdaptor

    ex = extractor("chgnet-0.3.0")
    cell = random_cell(10)
    _, f, e = features("chgnet-0.3.0", [cell])
    p = ex.model.predict_structure(AseAtomsAdaptor.get_structure(pretrained.to_atoms(*cell)), task="e",
                                   return_crystal_feas=True)
    assert np.allclose(f[0].sum(axis=0), p["crystal_fea"], rtol=0, atol=1e-4)    # CHGNet pools by summing
    assert e[0] == pytest.approx(float(p["e"]), abs=1e-5)


@needs_weights
def test_chgnet_handles_an_atom_without_neighbors():
    # JARVIS has structures in which no atom has a neighbor within CHGNet's 6 A; such an atom keeps its embedding
    ex = extractor("chgnet-0.3.0")
    before = ex.isolated_atoms
    lone = (np.eye(3) * 20.0, [[0.1, 0.2, 0.3]], ["Fe"])
    s, f, e = features("chgnet-0.3.0", [lone, random_cell(13)])
    assert f[0].shape == (1, 64) and np.isfinite(s).all()
    assert ex.isolated_atoms - before == 1


@needs_weights
def test_orb_batched_energy_is_the_calculators():
    from orb_models.forcefield.inference.calculator import ORBCalculator

    ex = extractor("orb-v3-omat")
    cells = [random_cell(k) for k in (11, 12)]
    atoms = [pretrained.to_atoms(*c) for c in cells]
    calc = ORBCalculator(ex.model, ex.adapter, device=DEVICE)
    ref = []
    for a in atoms:
        b = a.copy()
        b.calc = calc
        ref.append(b.get_potential_energy() / len(b))
    assert np.allclose(ex.energy(atoms), ref, rtol=0, atol=1e-5)
