"""Pretrained universal interatomic potentials used as frozen feature extractors.

Three models, chosen for what they were trained on:

========================  ===================================================  ===========================
name                      model (package, weights)                             training data
========================  ===================================================  ===========================
``mace-mp-0b3``           MACE-MP-0b3 medium (mace-torch; MIT)                 MPtrj: Materials Project
                                                                               relaxations, PBE/PBE+U
``chgnet-0.3.0``          CHGNet 0.3.0 (chgnet; BSD)                           MPtrj
``orb-v3-omat``           Orb v3 conservative, unlimited neighbors, OMat24     OMat24 (AIMD subset): structures
                          (orb-models; Apache-2.0)                             derived from Alexandria
========================  ===================================================  ===========================

Each model maps a structure to one feature vector per atom; the vector used here is the one the model's
energy readout reads:

- MACE: the invariant (scalar) channels of the node features after each of the two interactions, 2 x 128;
  this is what ``MACECalculator.get_descriptors`` returns.
- CHGNet: the atom features after the last convolution and the layer norm that follows it, 64; the input of
  the readout network (``mlp_first``: CHGNet sums per-atom energies).
- Orb: the node features after the last message-passing step, 256; the input of the energy head.

A structure is summarized by the mean and the standard deviation of its atoms' vectors (the first two cumulants
of their distribution; the mean alone, the usual choice, discards how the atomic environments differ,
Chorna et al., arXiv:2512.05717). Both are unchanged by the order of the atoms, a rotation and the choice of
the unit cell, which the tests check.

Orb's neighbor search (nvalchemiops) is wrapped in ``torch.compile``, which needs Triton, and Triton is not
available on Windows; loading Orb therefore disables TorchDynamo (``torch._dynamo.config.disable``) so that
the same code runs eagerly.
"""

from __future__ import annotations

import time

import numpy as np
import torch

MODELS = {
    "mace-mp-0b3": {"label": "MACE-MP-0b3", "data": "MPtrj", "dim": 256, "license": "MIT",
                    "package": "mace-torch 0.3.17", "weights": "mace-mp-0b3-medium.model"},
    "chgnet-0.3.0": {"label": "CHGNet 0.3.0", "data": "MPtrj", "dim": 64, "license": "BSD-3-Clause",
                     "package": "chgnet 0.4.2", "weights": "bundled with the package"},
    "orb-v3-omat": {"label": "Orb v3 (OMat24)", "data": "OMat24", "dim": 256, "license": "Apache-2.0",
                    "package": "orb-models 0.7.0", "weights": "orb-v3-conservative-inf-omat-20250404.ckpt"},
}


def to_atoms(lattice, frac_coords, species):
    """An ase.Atoms with periodic boundaries in all three directions."""
    import ase

    return ase.Atoms(list(species), scaled_positions=np.asarray(frac_coords, float), cell=np.asarray(lattice, float),
                     pbc=True)


def summarize(atom_features):
    """Mean and standard deviation (over atoms, ddof 0) of each structure's atom features: (n, 2 d)."""
    return np.stack([np.concatenate([f.mean(axis=0), f.std(axis=0)]) for f in atom_features])


class _Mace:
    def __init__(self, device):
        from mace.calculators import mace_mp

        self.calc = mace_mp(model="medium-0b3", device=device, default_dtype="float32")
        self.model = self.calc.models[0]
        self.device = device

    def _graph(self, atoms):
        from mace import data as mace_data
        from mace.tools import torch_tools

        calc = self.calc
        calc.arrays_keys.update({"charges": calc.charges_key})       # as MACECalculator._atoms_to_batch
        keyspec = mace_data.KeySpecification(info_keys=calc.info_keys, arrays_keys=calc.arrays_keys)
        with torch_tools.default_dtype(calc.default_dtype):
            config = mace_data.config_from_atoms(atoms, key_specification=keyspec, head_name=calc.head)
            return mace_data.AtomicData.from_config(config, z_table=calc.z_table, cutoff=calc.r_max,
                                                    heads=calc.available_heads)

    def __call__(self, atoms_list):
        from e3nn import o3
        from mace.modules.utils import extract_invariant
        from mace.tools import torch_geometric

        batch = torch_geometric.Batch.from_data_list([self._graph(a) for a in atoms_list]).to(self.device)
        out = self.model(batch.to_dict(), compute_force=False, compute_stress=False, training=False)
        irreps = o3.Irreps(str(self.model.products[0].linear.irreps_out))
        n_inv = irreps.dim // (irreps.lmax + 1) ** 2
        n_layers = int(self.model.num_interactions)
        feats = extract_invariant(out["node_feats"], num_layers=n_layers, num_features=n_inv, l_max=irreps.lmax)
        feats = feats[:, :n_inv * n_layers].detach().cpu().numpy()
        counts = np.diff(batch.ptr.cpu().numpy())
        energy = out["energy"].detach().cpu().numpy() / counts
        return np.split(feats, np.cumsum(counts)[:-1]), energy


class _Chgnet:
    def __init__(self, device):
        from chgnet.model import CHGNet

        self.model = CHGNet.load(model_name="0.3.0", use_device=device, verbose=False)
        self.device = device
        if not self.model.mlp_first:
            raise RuntimeError("expected a CHGNet that sums per-atom energies (mlp_first)")
        # an atom without a neighbor within 6 A keeps only its embedding; count such atoms instead of failing
        self.model.graph_converter.on_isolated_atoms = "ignore"
        self.isolated_atoms = 0
        self._captured = []
        self.model.mlp.register_forward_pre_hook(lambda module, args: self._captured.append(args[0].detach()))

    def __call__(self, atoms_list):
        from pymatgen.io.ase import AseAtomsAdaptor

        graphs = [self.model.graph_converter(AseAtomsAdaptor.get_structure(a), graph_id=str(i))
                  for i, a in enumerate(atoms_list)]
        for g, a in zip(graphs, atoms_list, strict=True):
            self.isolated_atoms += len(a) - len(torch.unique(g.atom_graph.reshape(-1, 2)[:, 0]))
        graphs = [g.to(self.device) for g in graphs]
        self._captured.clear()
        out = self.model(graphs, task="e")
        feats = self._captured[0].cpu().numpy()
        counts = np.array([len(a) for a in atoms_list])
        if feats.shape[0] != counts.sum():
            raise RuntimeError("captured CHGNet features do not match the atoms of the batch")
        return np.split(feats, np.cumsum(counts)[:-1]), out["e"].detach().cpu().numpy().reshape(-1)


class _Orb:
    def __init__(self, device):
        torch._dynamo.config.disable = True       # nvalchemiops is compiled with Triton, not available on Windows
        from orb_models.forcefield import pretrained

        # "float32-highest": plain float32 matrix products; the default "float32-high" allows TF32, which makes a
        # structure's features depend on the other structures of its batch at the 1e-3 level (and is global state)
        self.model, self.adapter = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-highest",
                                                                            compile=False)
        self.device = device

    def __call__(self, atoms_list):
        batch = self.adapter.from_ase_atoms_list(list(atoms_list), device=self.device)
        out = self.model.model(batch)
        feats = out["node_features"].detach().cpu().numpy()
        counts = batch.n_node.cpu().numpy()
        return np.split(feats, np.cumsum(counts)[:-1]), np.full(len(atoms_list), np.nan)

    def energy(self, atoms_list):
        """Predicted energy per atom (with forces, as the model computes it); for checks only."""
        batch = self.adapter.from_ase_atoms_list(list(atoms_list), device=self.device)
        e = self.model.predict(batch)["energy"].detach().cpu().numpy().reshape(-1)
        return e / batch.n_node.cpu().numpy()


def load(name, device="cuda"):
    """A feature extractor: called with a list of ase.Atoms, it returns (per-structure atom features, energy per
    atom); the energy is NaN for Orb, whose energy needs a gradient pass (see ``_Orb.energy``)."""
    return {"mace-mp-0b3": _Mace, "chgnet-0.3.0": _Chgnet, "orb-v3-omat": _Orb}[name](device)


def featurize(extractor, atoms_list, max_atoms=2000, log=None):
    """Mean and standard deviation of the atom features of every structure, in batches of at most ``max_atoms``
    atoms; returns (features (n, 2 d), energy per atom (n,), seconds)."""
    t0 = time.perf_counter()
    feats, energy = [], []
    start = 0
    with torch.no_grad():
        while start < len(atoms_list):
            stop, total = start, 0
            while stop < len(atoms_list) and (stop == start or total + len(atoms_list[stop]) <= max_atoms):
                total += len(atoms_list[stop])
                stop += 1
            f, e = extractor(atoms_list[start:stop])
            feats.append(summarize(f))
            energy.append(e)
            if log and (len(feats) % 50 == 0 or stop == len(atoms_list)):
                log(f"{stop} / {len(atoms_list)} structures, {time.perf_counter() - t0:.0f} s")
            start = stop
    return np.concatenate(feats).astype(np.float32), np.concatenate(energy), time.perf_counter() - t0
