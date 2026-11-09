"""CGCNN, the crystal graph convolutional neural network of Xie & Grossman, Phys. Rev. Lett. 120, 145301 (2018).

The graph, the network and the training defaults follow the authors' released code (github.com/txie-93/cgcnn,
commit f42ab23), the code that produced the published benchmark numbers. It differs from Eq. 5 of the paper,
v_i <- v_i + sum_j sigmoid(z_ij W_f + b_f) * g(z_ij W_s + b_s) with z_ij = v_i (+) v_j (+) u_ij, in two places:
the gate and core inputs z W + b pass through batch normalization (one over both halves), and so does their sum
over neighbors, and the residual sum passes through a softplus:

    v_i <- softplus(v_i + BN(sum_j sigmoid(f_ij) * softplus(c_ij))),   (f_ij, c_ij) = BN(z_ij W + b).

Graph: every atom is joined to its 12 nearest neighbors within 8 angstrom (periodic images included), ordered by
distance; an atom with fewer neighbors gets padding entries that point at the first atom of its crystal at
distance 9 angstrom (the radius plus 1), as in the original. Edge features are 41 Gaussians
exp(-((d - mu_k) / 0.2)^2), mu_k = 0, 0.2, ..., 8 angstrom. Atom features are the 92-dimensional one-hot vectors
of the original's atom_init.json, vendored as cgcnn_atom_init.json (MIT license, see THIRD_PARTY_NOTICES.md).
After the convolutions the atom vectors are averaged over the crystal, passed through softplus, a linear layer to
128 features and softplus again, and a linear output layer.

Training follows the original's defaults with the settings of the JARVIS leaderboard run (300 epochs): mean
squared error on targets standardized with the mean and standard deviation of 500 random training materials, SGD
with learning rate 0.01 and momentum 0.9, batches of 256 crystals, the learning rate divided by 10 after epoch
100, and the weights of the epoch with the lowest validation MAE kept.

The parameter names are those of the original (``embedding``, ``convs.<t>.fc_full``, ``convs.<t>.bn1``, ...), so a
state dict of one loads into the other.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

RADIUS = 8.0
MAX_NEIGHBORS = 12
GAUSSIAN_STEP = 0.2
ATOM_INIT = Path(__file__).with_name("cgcnn_atom_init.json")


# --------------------------------------------------------------------------
# Graphs
# --------------------------------------------------------------------------

def atom_feature_table(path=ATOM_INIT):
    """(Z_max + 1, 92) float32 array whose row Z is the CGCNN feature vector of element Z (row 0 is zeros)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    width = len(next(iter(raw.values())))
    table = np.zeros((max(int(z) for z in raw) + 1, width), dtype=np.float32)
    for z, vector in raw.items():
        table[int(z)] = vector
    return table


def gaussian_centers(radius=RADIUS, step=GAUSSIAN_STEP):
    """The original's np.arange(0, radius + step, step): 0, 0.2, ..., 8 angstrom (41 centers)."""
    return np.arange(0.0, radius + step, step)


def crystal_graph(lattice, frac_coords, species, radius=RADIUS, max_neighbors=MAX_NEIGHBORS):
    """Atomic numbers and the nearest neighbors of every atom, as the original builds them.

    The original sorts the output of pymatgen's ``Structure.get_all_neighbors(radius)`` by distance with Python's
    stable sort; here the same pairs come from ``Structure.get_neighbor_list(radius)`` (which
    ``get_all_neighbors`` calls) and are sorted by distance with a stable sort within each atom, which keeps the
    same order, including the choice among equally distant neighbors at the twelfth place.

    Returns ``numbers`` (n,), ``neighbors`` (n, max_neighbors) indices into the crystal's atoms, ``distances``
    (n, max_neighbors) in angstrom, and the number of padding entries.
    """
    from pymatgen.core import Lattice, Structure

    structure = Structure(Lattice(np.asarray(lattice, dtype=float)), list(species), np.asarray(frac_coords, float))
    centers, nbrs, _, dist = structure.get_neighbor_list(radius)
    n = len(structure)
    order = np.argsort(dist, kind="stable")
    order = order[np.argsort(centers[order], kind="stable")]
    centers, nbrs, dist = centers[order], nbrs[order], dist[order]
    starts = np.searchsorted(centers, np.arange(n))
    rank = np.arange(len(centers)) - starts[centers]
    keep = rank < max_neighbors
    neighbors = np.zeros((n, max_neighbors), dtype=np.int64)
    distances = np.full((n, max_neighbors), radius + 1.0)
    neighbors[centers[keep], rank[keep]] = nbrs[keep]
    distances[centers[keep], rank[keep]] = dist[keep]
    numbers = np.array([site.specie.Z for site in structure], dtype=np.int64)
    padded = int(n * max_neighbors - keep.sum())
    return numbers, neighbors, distances, padded


class GraphSet:
    """The crystal graphs of many structures in flat arrays, from which batches are cut on one device."""

    def __init__(self, numbers, neighbors, distances, n_atoms, device="cpu"):
        self.device = torch.device(device)
        self.numbers = torch.as_tensor(numbers, dtype=torch.long, device=self.device)
        self.neighbors = torch.as_tensor(neighbors, dtype=torch.long, device=self.device)   # within each crystal
        self.distances = torch.as_tensor(distances, dtype=torch.float32, device=self.device)
        self.n_atoms = torch.as_tensor(n_atoms, dtype=torch.long, device=self.device)
        self.start = torch.cumsum(self.n_atoms, 0) - self.n_atoms

    @classmethod
    def from_graphs(cls, graphs, device="cpu"):
        """From a list of ``crystal_graph`` results."""
        return cls(np.concatenate([g[0] for g in graphs]), np.concatenate([g[1] for g in graphs]),
                   np.concatenate([g[2] for g in graphs]), np.array([len(g[0]) for g in graphs]), device)

    def arrays(self):
        """numpy arrays from which ``GraphSet(**arrays)`` rebuilds the set (for caching)."""
        return {k: getattr(self, k).cpu().numpy() for k in ("numbers", "neighbors", "distances", "n_atoms")}

    def to(self, device):
        return GraphSet(**self.arrays(), device=device)

    def __len__(self):
        return len(self.n_atoms)

    def batch(self, crystals):
        """Atomic numbers (N,), neighbor distances (N, M), neighbor indices into the batch (N, M) and the position
        in ``crystals`` of each atom's crystal (N,), for the crystals with the given indices."""
        crystals = torch.as_tensor(crystals, dtype=torch.long, device=self.device)
        counts = self.n_atoms[crystals]
        owner = torch.repeat_interleave(torch.arange(len(crystals), device=self.device), counts)
        first = torch.cumsum(counts, 0) - counts                       # position of each crystal's first atom
        atoms = self.start[crystals][owner] + torch.arange(len(owner), device=self.device) - first[owner]
        neighbors = self.neighbors[atoms] + first[owner][:, None]
        return self.numbers[atoms], self.distances[atoms], neighbors, owner


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------

class ConvLayer(nn.Module):
    """One convolution of the released code (see the module docstring)."""

    def __init__(self, atom_fea_len, nbr_fea_len):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * atom_fea_len + nbr_fea_len, 2 * atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * atom_fea_len)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        n, m = nbr_fea_idx.shape
        total = torch.cat([atom_in_fea.unsqueeze(1).expand(n, m, self.atom_fea_len),
                           atom_in_fea[nbr_fea_idx, :], nbr_fea], dim=2)
        gated = self.bn1(self.fc_full(total).view(-1, 2 * self.atom_fea_len)).view(n, m, 2 * self.atom_fea_len)
        nbr_filter, nbr_core = gated.chunk(2, dim=2)
        summed = torch.sum(self.sigmoid(nbr_filter) * self.softplus1(nbr_core), dim=1)
        return self.softplus2(atom_in_fea + self.bn2(summed))


class CrystalGraphConvNet(nn.Module):
    """CGCNN for one regression target, with the original's layer sizes as defaults.

    ``forward`` takes atom features (N, orig_atom_fea_len), edge features (N, M, nbr_fea_len), neighbor indices
    (N, M) and the crystal of every atom (N,) (values 0 .. n_crystals - 1); it returns (n_crystals, 1).
    """

    def __init__(self, orig_atom_fea_len=92, nbr_fea_len=41, atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1):
        super().__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len, nbr_fea_len) for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])
        self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_of_atom, n_crystals=None):
        atom_fea = self.embedding(atom_fea)
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_of_atom, n_crystals)
        crys_fea = self.conv_to_fc_softplus(self.conv_to_fc(self.conv_to_fc_softplus(crys_fea)))
        for fc, softplus in zip(getattr(self, "fcs", []), getattr(self, "softpluses", []), strict=True):
            crys_fea = softplus(fc(crys_fea))
        return self.fc_out(crys_fea)

    @staticmethod
    def pooling(atom_fea, crystal_of_atom, n_crystals=None):
        """Mean of the atom vectors of each crystal."""
        if n_crystals is None:
            n_crystals = int(crystal_of_atom.max()) + 1
        sums = torch.zeros(n_crystals, atom_fea.shape[1], dtype=atom_fea.dtype, device=atom_fea.device)
        sums.index_add_(0, crystal_of_atom, atom_fea)
        counts = torch.bincount(crystal_of_atom, minlength=n_crystals).clamp(min=1)
        return sums / counts[:, None].to(atom_fea.dtype)


class Featurizer:
    """Turns a ``GraphSet`` batch into the network's inputs: atom feature vectors and Gaussian edge features."""

    def __init__(self, device, table=None, centers=None, step=GAUSSIAN_STEP):
        self.table = torch.as_tensor(atom_feature_table() if table is None else table, dtype=torch.float32,
                                     device=device)
        self.centers = torch.as_tensor(gaussian_centers() if centers is None else centers, dtype=torch.float32,
                                       device=device)
        self.step = step

    def __call__(self, graphs, crystals):
        numbers, distances, neighbors, owner = graphs.batch(crystals)
        edges = torch.exp(-((distances[..., None] - self.centers) / self.step) ** 2)
        return self.table[numbers], edges, neighbors, owner, len(crystals)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def predict(model, graphs, crystals, featurizer, mean, std, batch_size=1024):
    """Predictions (denormalized) for the given crystals, in evaluation mode."""
    model.eval()
    crystals = np.asarray(crystals)
    out = []
    with torch.no_grad():
        for b in range(0, len(crystals), batch_size):
            out.append(model(*featurizer(graphs, crystals[b:b + batch_size])).view(-1))
    return (torch.cat(out) * std + mean).cpu().numpy().astype(float)


def fit(graphs, y, train, val, seed=0, epochs=300, batch_size=256, lr=0.01, momentum=0.9, weight_decay=0.0,
        milestones=(100,), n_normalizer=500, model_kw=None, track=None, log=None, table=None):
    """Train one CGCNN on ``graphs`` (a GraphSet on the training device) with the original's procedure.

    Returns the model with the weights of the epoch with the lowest validation MAE, a dict with the normalizer
    (``mean``, ``std``), ``best_epoch``, ``seconds`` and per-epoch ``train_loss`` (normalized MSE, training mode),
    ``train_mae`` and ``val_mae`` (eV/atom), and the ``Featurizer``. ``table`` replaces the atom feature table
    (default: the original's ``atom_init.json``).

    ``track`` = (graphs, crystals, targets): every time the validation MAE reaches a new minimum, the MAE on these
    crystals is recorded (``track_epoch``, ``track_mae``), so that the result of stopping after fewer epochs can be
    read off one run: it is the tracked MAE at the last new minimum before that epoch.
    """
    device = graphs.device
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    train, val = np.asarray(train), np.asarray(val)
    sample = y[rng.choice(train, size=min(n_normalizer, len(train)), replace=False)]
    mean, std = float(np.mean(sample)), float(np.std(sample, ddof=1))    # torch.std is the n - 1 estimate
    model = CrystalGraphConvNet(**(model_kw or {})).to(device)
    featurizer = Featurizer(device, table=table)
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(milestones), gamma=0.1)
    target = torch.as_tensor(y, dtype=torch.float32, device=device)
    history = {"train_loss": [], "train_mae": [], "val_mae": [], "track_epoch": [], "track_mae": []}
    best, best_state, best_epoch = np.inf, None, -1
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = train[rng.permutation(len(train))]
        loss_sum = torch.zeros((), device=device)
        abs_sum = torch.zeros((), device=device)
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            out = model(*featurizer(graphs, idx)).view(-1)
            t = target[idx]
            loss = nn.functional.mse_loss(out, (t - mean) / std)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(idx)
            abs_sum += torch.sum(torch.abs(out.detach() * std + mean - t))
        val_mae = float(np.mean(np.abs(predict(model, graphs, val, featurizer, mean, std) - y[val])))
        scheduler.step()
        history["train_loss"].append(float(loss_sum) / len(train))
        history["train_mae"].append(float(abs_sum) / len(train))
        history["val_mae"].append(val_mae)
        if val_mae < best:
            best, best_epoch = val_mae, epoch
            best_state = copy.deepcopy(model.state_dict())
            if track is not None:
                t_graphs, t_crystals, t_y = track
                history["track_epoch"].append(epoch)
                history["track_mae"].append(float(np.mean(np.abs(
                    predict(model, t_graphs, t_crystals, featurizer, mean, std) - t_y))))
        if log is not None:
            log(epoch, history)
    model.load_state_dict(best_state)
    info = {"mean": mean, "std": std, "best_epoch": best_epoch, "seconds": time.perf_counter() - t0,
            **{k: np.array(v) for k, v in history.items()}}
    return model, info, featurizer
