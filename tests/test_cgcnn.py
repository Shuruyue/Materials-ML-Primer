import numpy as np
import pytest
import torch

from materials_ml_primer import cgcnn


def random_structure(seed, n_atoms=4):
    rng = np.random.default_rng(seed)
    lattice = np.diag(rng.uniform(3.5, 5.5, 3)) + rng.uniform(-0.6, 0.6, (3, 3))
    frac = rng.random((n_atoms, 3))
    species = list(rng.choice(["Na", "Cl", "O", "Fe"], n_atoms))
    return lattice, frac, species


def original_construction(lattice, frac, species, radius=8.0, max_num_nbr=12):
    """The lines of the original CIFData.__getitem__ that build the neighbor lists, on a pymatgen Structure."""
    from pymatgen.core import Lattice, Structure

    crystal = Structure(Lattice(lattice), species, frac)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in crystal.get_all_neighbors(radius, include_index=True)]
    idx, dist = [], []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
            idx.append([x[2] for x in nbr] + [0] * (max_num_nbr - len(nbr)))
            dist.append([x[1] for x in nbr] + [radius + 1.0] * (max_num_nbr - len(nbr)))
        else:
            idx.append([x[2] for x in nbr[:max_num_nbr]])
            dist.append([x[1] for x in nbr[:max_num_nbr]])
    return np.array(idx), np.array(dist)


def test_gaussian_centers_and_atom_table_have_the_original_sizes():
    assert np.array_equal(cgcnn.gaussian_centers(), np.arange(0, 8.0 + 0.2, 0.2))
    assert len(cgcnn.gaussian_centers()) == 41
    table = cgcnn.atom_feature_table()
    assert table.shape == (101, 92)
    assert not table[0].any() and set(np.unique(table[1:])) == {0.0, 1.0}


@pytest.mark.parametrize("structure", [
    (np.eye(3) * 5.64, [[0, 0, 0], [0.5, 0.5, 0.5]], ["Cs", "Cl"]),                 # CsCl: 8 + 6, cut at 12
    (np.array([[0, 2.82, 2.82], [2.82, 0, 2.82], [2.82, 2.82, 0]]), [[0, 0, 0], [0.5, 0.5, 0.5]], ["Na", "Cl"]),
    random_structure(0), random_structure(1, n_atoms=7),
])
def test_neighbors_are_those_of_the_original(structure):
    lattice, frac, species = structure
    numbers, idx, dist, padded = cgcnn.crystal_graph(lattice, frac, species)
    ref_idx, ref_dist = original_construction(np.asarray(lattice, float), frac, species)
    assert np.array_equal(idx, ref_idx)
    assert np.allclose(dist, ref_dist, rtol=0, atol=1e-12)
    assert padded == 0
    assert np.all(np.diff(dist, axis=1) >= 0)


def test_an_isolated_atom_gets_the_original_padding():
    numbers, idx, dist, padded = cgcnn.crystal_graph(np.eye(3) * 20.0, [[0.1, 0.2, 0.3]], ["Fe"])
    assert numbers.tolist() == [26]
    assert padded == 12 and not idx.any() and np.all(dist == 9.0)


def test_a_batch_is_the_concatenation_of_its_crystals():
    graphs = [cgcnn.crystal_graph(*random_structure(s, n_atoms=n)) for s, n in ((2, 3), (3, 5), (4, 2))]
    gs = cgcnn.GraphSet.from_graphs(graphs)
    numbers, dist, nbrs, owner = gs.batch([2, 0])
    order = [2, 0]
    offset = np.cumsum([0] + [len(graphs[c][0]) for c in order])[:-1]
    assert np.array_equal(numbers.numpy(), np.concatenate([graphs[c][0] for c in order]))
    assert np.allclose(dist.numpy(), np.concatenate([graphs[c][2] for c in order]))
    assert np.array_equal(nbrs.numpy(), np.concatenate([graphs[c][1] + o for c, o in zip(order, offset, strict=True)]))
    assert owner.tolist() == [0, 0, 1, 1, 1]
    rebuilt = cgcnn.GraphSet(**gs.arrays())
    assert torch.equal(rebuilt.batch([1])[2], gs.batch([1])[2])


def randomize_batchnorm(module, gen):
    for bn in (m for m in module.modules() if isinstance(m, torch.nn.BatchNorm1d)):
        bn.running_mean.copy_(torch.randn(bn.num_features, generator=gen))
        bn.running_var.copy_(torch.rand(bn.num_features, generator=gen) + 0.5)
        bn.weight.data.copy_(torch.randn(bn.num_features, generator=gen))
        bn.bias.data.copy_(torch.randn(bn.num_features, generator=gen))


def test_convolution_equals_the_formula_written_out_atom_by_atom():
    gen = torch.Generator().manual_seed(0)
    n, m, a, b = 5, 3, 4, 6
    layer = cgcnn.ConvLayer(a, b).eval()
    randomize_batchnorm(layer, gen)
    v = torch.randn(n, a, generator=gen, dtype=torch.float64)
    u = torch.randn(n, m, b, generator=gen, dtype=torch.float64)
    idx = torch.randint(0, n, (n, m), generator=gen)
    layer = layer.double()
    out = layer(v, u, idx).detach()

    def bn(x, norm):
        return (x - norm.running_mean) / torch.sqrt(norm.running_var + norm.eps) * norm.weight + norm.bias

    def softplus(x):
        return torch.log1p(torch.exp(x))

    w, bias = layer.fc_full.weight.detach(), layer.fc_full.bias.detach()
    for i in range(n):
        total = torch.zeros(a, dtype=torch.float64)
        for k in range(m):
            z = torch.cat([v[i], v[idx[i, k]], u[i, k]])
            g = bn(w @ z + bias, layer.bn1).detach()
            total += torch.sigmoid(g[:a]) * softplus(g[a:])
        expected = softplus(v[i] + bn(total, layer.bn2).detach())
        assert torch.allclose(out[i], expected, atol=1e-12)


def predict_one(model, graph, featurizer):
    gs = cgcnn.GraphSet.from_graphs([graph])
    with torch.no_grad():
        return float(model(*featurizer(gs, [0])))


def test_prediction_ignores_atom_order_rotation_and_choice_of_cell():
    torch.manual_seed(0)
    model = cgcnn.CrystalGraphConvNet().eval()
    randomize_batchnorm(model, torch.Generator().manual_seed(1))
    featurizer = cgcnn.Featurizer("cpu")
    lattice, frac, species = random_structure(5, n_atoms=3)
    ref = predict_one(model, cgcnn.crystal_graph(lattice, frac, species), featurizer)
    perm = [2, 0, 1]
    permuted = predict_one(model, cgcnn.crystal_graph(lattice, frac[perm], [species[p] for p in perm]), featurizer)
    q, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(3, 3)))
    rotated = predict_one(model, cgcnn.crystal_graph(lattice @ q.T, frac, species), featurizer)
    super_frac = np.vstack([frac * [0.5, 1, 1], (frac + [1, 0, 0]) * [0.5, 1, 1]])
    supercell = predict_one(model, cgcnn.crystal_graph(lattice * [[2], [1], [1]], super_frac, species * 2), featurizer)
    for other in (permuted, rotated, supercell):
        assert other == pytest.approx(ref, abs=1e-5)


def test_training_overfits_one_batch():
    # one batch of all crystals, so that the batch-norm statistics of training and evaluation agree
    graphs = [cgcnn.crystal_graph(*random_structure(s)) for s in range(12)]
    y = np.linspace(-2.0, 1.0, 12)
    gs = cgcnn.GraphSet.from_graphs(graphs)
    model, info, featurizer = cgcnn.fit(gs, y, np.arange(12), np.arange(12), epochs=300, batch_size=12, lr=0.01)
    assert info["train_loss"][-1] < 0.1 * info["train_loss"][0]
    pred = cgcnn.predict(model, gs, np.arange(12), featurizer, info["mean"], info["std"])
    assert np.mean(np.abs(pred - y)) < 0.15 * np.std(y)
    assert info["val_mae"][info["best_epoch"]] == info["val_mae"].min()
