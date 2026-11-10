"""Trainings of the cgcnn experiment, one network at a time, cached so that a full run can be split and resumed.

Families of tasks:

- "leaderboard": the formation-energy split of the JARVIS leaderboard (dft_3d_2021; 44 569 training, 5 572
  validation and 5 572 test materials, file dft_3d_formation_energy_peratom.json.zip), five networks with seeds
  0-4. The leaderboard's CGCNN entry, trained with the original code for 300 epochs, has a test MAE of
  0.0625 eV/atom.
- "shift": the tasks of the uncertainty experiment with the same training, validation, calibration and test
  materials (random, chemical-system and prototype 10-fold cross-validation, the prospective test, and leave one
  element out for the elements chosen by ``chosen_elements``), five networks per task, trained for 150 epochs
  (see SETTINGS).
- "budget": one network of each of three shift tasks trained for 300 epochs, to check the shorter training.
- "blanked": the leave-one-element-out tasks whose held-out element has a one-hot column in the atom feature table
  that no training material has (its embedding weights are never trained and stay random); five networks per task
  with those columns set to zero for that element, which the table already does for unknown values.

Each network is stored as data/cgcnn_cache/<task>/member-<seed>.npz (predictions for the validation, calibration
and test materials, training history, settings); a cached file with other settings is computed again. To use the
GPU fully, run several workers at once from this folder (each takes every n-th network):

    python runs.py leaderboard 0 3
    python runs.py leaderboard 1 3
    python runs.py leaderboard 2 3
    python runs.py budget,shift 0 3     # and 1 3, 2 3
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from pymatgen.core import Element

from materials_ml_primer import cgcnn, data

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_DIR = ROOT / "data"
CACHE = DATA_DIR / "cgcnn_cache"
MEMBERS = 5
LEADERBOARD_COMMIT = "4f1356e3304d64d28217161b870af68668653911"
LEADERBOARD_URL = (f"https://raw.githubusercontent.com/atomgptlab/jarvis_leaderboard/{LEADERBOARD_COMMIT}/"
                   "jarvis_leaderboard/")
LEADERBOARD_FILES = {
    "split": "benchmarks/AI/SinglePropertyPrediction/dft_3d_formation_energy_peratom.json.zip",
    "cgcnn": "contributions/cgcnn_model/AI-SinglePropertyPrediction-formation_energy_peratom-dft_3d-test-mae.csv.zip",
}
N_HARDEST, N_SPREAD = 6, 6
# The shift networks stop after 150 epochs instead of 300: on the leaderboard split this raised the validation MAE
# by 2.6 % (mean of five seeds) and halved the training time. "budget" trains one network of three shift tasks
# for 300 epochs, to check that stopping early costs as little under distribution shift.
SETTINGS = {
    "leaderboard": {"version": 1, "epochs": 300, "milestones": [100], "lr": 0.01, "batch_size": 256},
    "shift": {"version": 1, "epochs": 150, "milestones": [100], "lr": 0.01, "batch_size": 256},
    "budget": {"version": 1, "epochs": 300, "milestones": [100], "lr": 0.01, "batch_size": 256},
    "blanked": {"version": 1, "epochs": 150, "milestones": [100], "lr": 0.01, "batch_size": 256,
                "table": "columns of the held-out element that no training material has set to zero"},
}
BUDGET_TASKS = ["chemical-0", "prototype-0", "element-F"]
MEMBERS_OF = {"leaderboard": MEMBERS, "shift": MEMBERS, "budget": 1, "blanked": MEMBERS}


def _uncertainty_runs():
    """The task definitions of the uncertainty experiment (experiments/uncertainty/runs.py)."""
    name = "uncertainty_runs"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE.parent / "uncertainty" / "runs.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module          # dataclasses look their module up here
        spec.loader.exec_module(module)
    return sys.modules[name]


def leaderboard_file(name):
    """A file of the jarvis_leaderboard repository (pinned commit), downloaded into data/ on first use."""
    path = DATA_DIR / "jarvis_leaderboard" / Path(LEADERBOARD_FILES[name]).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(LEADERBOARD_URL + LEADERBOARD_FILES[name], tmp)
        tmp.replace(path)
    with zipfile.ZipFile(path) as z:
        return z.read(z.namelist()[0])


def graphs_of(records, name):
    """CGCNN graphs of the records, cached in data/cgcnn_cache/graphs_<name>.npz."""
    path = CACHE / f"graphs_{name}.npz"
    jids = [rec["jid"] for rec in records]
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if z["jids"].tolist() == jids:
                return cgcnn.GraphSet(**{k: z[k] for k in ("numbers", "neighbors", "distances", "n_atoms")}), \
                    int(z["padded"]), z["padded_per_crystal"]
    graphs = []
    for rec in records:
        lat, frac, sp = data.structure_arrays(rec["atoms"])
        graphs.append(cgcnn.crystal_graph(lat, frac, sp))
    gs = cgcnn.GraphSet.from_graphs(graphs)
    padded = np.array([g[3] for g in graphs])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, jids=np.array(jids), padded=np.array(padded.sum()), padded_per_crystal=padded,
                        **gs.arrays())
    return gs, int(padded.sum()), padded


def load():
    """Records, targets, group labels and graphs of the 2021.8.18 snapshot and of the materials added later."""
    uq = _uncertainty_runs()
    records, _ = data.deduplicate(data.download(data.DFT_3D_2021, DATA_DIR))
    latest, _ = data.deduplicate(data.download(data.DFT_3D_LATEST, DATA_DIR))
    if len(latest) != data.DFT_3D_LATEST_SIZE:
        raise RuntimeError(f"jarvis-tools serves another 'dft_3d' snapshot ({len(latest)} materials)")
    old_ids = {rec["jid"] for rec in records}
    added = [rec for rec in latest if rec["jid"] not in old_ids]
    system, proto = uq.labels(records)
    d = {"records": records, "added": added, "system": system, "proto": proto,
         "y": np.array([data.as_float(rec["formation_energy_peratom"]) for rec in records]),
         "y_add": np.array([data.as_float(rec["formation_energy_peratom"]) for rec in added])}
    uq.check_targets(d["y"], records, "2021.8.18")
    uq.check_targets(d["y_add"], added, "added")
    d["graphs"], d["padded"], d["padded_per_crystal"] = graphs_of(records, "2021")
    d["graphs_add"], d["padded_add"], _ = graphs_of(added, "added")
    return d


def leaderboard_split(d):
    """Indices of the leaderboard's training, validation and test materials in the 2021 records.

    Two materials (JVASP-113961 and JVASP-116461, whose records are repeated in dft_3d_2021) are in both the
    leaderboard's training and test sets; they are kept in the test set, so that the test set is the
    leaderboard's, and removed from the training set (``dropped``).
    """
    split = json.loads(leaderboard_file("split"))          # {part: {jid: target}}
    pos = {rec["jid"]: i for i, rec in enumerate(d["records"])}
    out = {part: np.array(sorted({pos[j] for j in split[part]})) for part in ("train", "val", "test")}
    out["dropped"] = np.intersect1d(out["train"], np.union1d(out["val"], out["test"]))
    out["train"] = np.setdiff1d(out["train"], out["dropped"])
    if np.intersect1d(out["val"], out["test"]).size:
        raise RuntimeError("leaderboard validation and test sets overlap")
    return out


def chosen_elements():
    """Elements for leave one element out: the N_HARDEST with the lowest test R^2 of gradient boosting in the
    data_splits experiment, and N_SPREAD more at the middle of N_SPREAD equal parts of the others ordered by R^2."""
    results = json.loads((HERE.parent / "data_splits" / "data_splits_results.json").read_text(encoding="utf-8"))
    ranked = sorted(results["loeo"], key=lambda e: results["loeo"][e]["r2"])
    rest = ranked[N_HARDEST:]
    spread = [rest[int((k + 0.5) * len(rest) / N_SPREAD)] for k in range(N_SPREAD)]
    return ranked[:N_HARDEST], spread


def unseen_columns(d, train, element):
    """Columns of the atom feature table that are 1 for ``element`` and 0 for every element of the training
    materials: the embedding weights of these columns get no gradient in training and keep their initial values."""
    table = cgcnn.atom_feature_table()
    seen = sorted({Element(e).Z for i in train for e in d["records"][i]["atoms"]["elements"]})
    return np.flatnonzero((table[Element(element).Z] > 0) & ~table[seen].any(axis=0))


def blanked_table(d, task):
    """Atom feature table of a "blanked" task and the columns set to zero for its held-out element."""
    element = task[0].rsplit("-", 1)[1]
    cols = unseen_columns(d, task[1], element)
    table = cgcnn.atom_feature_table().copy()
    table[Element(element).Z, cols] = 0.0
    return table, cols


def tasks(d, family):
    """(key, train, val, cal, test, test on the added materials) of every task of a family, in a fixed order."""
    if family == "leaderboard":
        s = leaderboard_split(d)
        return [("leaderboard", s["train"], s["val"], np.array([], dtype=int), s["test"], False)]
    if family == "budget":
        return [(f"budget-{t[0]}",) + t[1:] for t in tasks(d, "shift") if t[0] in BUDGET_TASKS]
    if family == "blanked":
        return [(f"blanked-{t[0]}",) + t[1:] for t in tasks(d, "shift")
                if t[0].startswith("element-") and len(unseen_columns(d, t[1], t[0].split("-", 1)[1]))]
    hardest, spread = chosen_elements()
    keep = {f"element-{e}" for e in hardest + spread}
    out = []
    for t in _uncertainty_runs().tasks({**d, "y_add": d["y_add"]}):
        if t.key.startswith("element-") and t.key not in keep:
            continue
        out.append((t.key, t.train, t.val, t.cal, t.test, t.added))
    return out


def jobs(d, family):
    return [(family, task, seed) for task in tasks(d, family) for seed in range(MEMBERS_OF[family])]


def result(d, family, task, seed, compute=True, device="cuda"):
    """Cached result of one network: (arrays, 'cached' | 'computed'); None if missing and not computed."""
    key, train, val, cal, test, added = task
    path = CACHE / key / f"member-{seed}.npz"
    settings = json.dumps(SETTINGS[family], sort_keys=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if str(z["settings"]) == settings and np.array_equal(z["test_index"], test) \
                    and np.array_equal(z["cal_index"], cal):
                return {k: z[k] for k in z.files}, "cached"
    if not compute:
        return None
    s = SETTINGS[family]
    graphs = d["graphs"] if d["graphs"].device.type == device else d["graphs"].to(device)
    test_graphs = d["graphs_add"] if added else graphs
    if added and test_graphs.device.type != device:
        test_graphs = test_graphs.to(device)
    y_test = d["y_add"][test] if added else d["y"][test]
    table, blanked = blanked_table(d, task) if family == "blanked" else (None, np.array([], dtype=int))
    model, info, featurizer = cgcnn.fit(graphs, d["y"], train, val, seed=seed, epochs=s["epochs"],
                                        batch_size=s["batch_size"], lr=s["lr"], milestones=s["milestones"],
                                        track=(test_graphs, test, y_test), table=table)
    out = {f"{k}": np.asarray(v) for k, v in info.items()}
    out["blanked_columns"] = blanked
    out["val_pred"] = cgcnn.predict(model, graphs, val, featurizer, info["mean"], info["std"])
    out["cal_pred"] = cgcnn.predict(model, graphs, cal, featurizer, info["mean"], info["std"]) if len(cal) \
        else np.array([])
    out["test_pred"] = cgcnn.predict(model, test_graphs, test, featurizer, info["mean"], info["std"])
    out.update(settings=np.array(settings), val_index=val, cal_index=cal, test_index=test)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(path)                       # a run stopped mid-write leaves no half-written result
    return out, "computed"


def main(families, worker, n_workers):
    import torch

    torch.set_num_threads(2)
    d = load()
    d["graphs"] = d["graphs"].to("cuda")
    d["graphs_add"] = d["graphs_add"].to("cuda")
    todo = [job for family in families for job in jobs(d, family)][worker::n_workers]
    t0 = time.perf_counter()
    for i, (family, task, seed) in enumerate(todo, 1):
        _, how = result(d, family, task, seed)
        print(f"[{worker}/{n_workers}] {i}/{len(todo)} {task[0]} member {seed}: {how} "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) == 4 else []
    if not names or any(n not in SETTINGS for n in names):
        sys.exit("usage: python runs.py FAMILY[,FAMILY...] WORKER N_WORKERS   (families: " + ", ".join(SETTINGS) + ")")
    main(names, int(sys.argv[2]), int(sys.argv[3]))
