"""Per-task computations of the uncertainty experiment, cached so that a full run can be split and resumed.

A task is one test set with its training, validation and calibration sets:

- the 10 folds of the random, chemical-system and prototype splits (fold k test, k + 1 validation for early
  stopping, k + 2 calibration, the other seven training: 70/10/10/10), with the folds of the data_splits
  experiment;
- leave one element out for every element in at least 200 materials (the other materials split by chemical
  system into training, validation and calibration, 80/10/10);
- the prospective test (the 2021.8.18 snapshot split by chemical system 80/10/10; test: the materials added up
  to 2025.9.24).

Each task has two parts. "cpu": a bootstrap ensemble of gradient-boosting models, gradient-boosting quantile
regression, a random forest and the distance to the training set. "gpu": a sparse variational Gaussian process
and a deep ensemble. The predictions for the calibration and test sets of each part are stored in
data/uncertainty_cache/<part>/<task>.npz with the settings that produced them; a cached file with other settings
is computed again. The notebook loads what is cached and computes what is missing; to fill the cache faster,
run both parts at once from this folder:

    python runs.py cpu
    python runs.py gpu
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from materials_ml_primer import data, descriptors, splits
from materials_ml_primer import uncertainty as uq

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CACHE = DATA_DIR / "uncertainty_cache"
N_FOLDS = 10
MIN_MATERIALS = 200
ALPHA = 0.1
SETTINGS = {
    "cpu": {"version": 1, "ensemble_members": 5, "learning_rate": 0.1, "max_iter": 2500, "quantile_alpha": ALPHA,
            "forest_trees": 300, "neighbors": 5},
    "gpu": {"version": 2, "inducing_points": 1024, "gp_epochs": 60, "gp_jitter": 1e-4, "nn_members": 5,
            "nn_hidden": 256, "nn_max_epochs": 200, "nn_patience": 20},
}


@dataclass
class Task:
    key: str
    split: str
    train: np.ndarray
    val: np.ndarray
    cal: np.ndarray
    test: np.ndarray
    added: bool = False          # the test set is the materials added after 2021.8.18


def load():
    """Records, features, targets and group labels of the 2021.8.18 snapshot and of the materials added later."""
    records, _ = data.deduplicate(data.download(data.DFT_3D_2021, DATA_DIR))
    latest, _ = data.deduplicate(data.download(data.DFT_3D_LATEST, DATA_DIR))
    if len(latest) != data.DFT_3D_LATEST_SIZE:
        raise RuntimeError(f"jarvis-tools serves another 'dft_3d' snapshot ({len(latest)} materials)")
    old_ids = {rec["jid"] for rec in records}
    added = [rec for rec in latest if rec["jid"] not in old_ids]
    y = np.array([data.as_float(rec["formation_energy_peratom"]) for rec in records])
    y_add = np.array([data.as_float(rec["formation_energy_peratom"]) for rec in added])
    X = np.hstack(descriptors.jarvis_feature_matrices(records))
    X_add = np.hstack(descriptors.jarvis_feature_matrices(added))
    system, proto = labels(records)
    return {"records": records, "added": added, "y": y, "y_add": y_add, "X": X, "X_add": X_add,
            "system": system, "proto": proto}


def labels(records):
    """Chemical system and prototype label of every record, as strings (computed once and cached)."""
    path = CACHE / "labels_2021.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached["jids"] == [rec["jid"] for rec in records]:
            return cached["system"], cached["proto"]
    system, proto = [], []
    for rec in records:
        lat, frac, sp = data.structure_arrays(rec["atoms"])
        system.append("-".join(splits.chemical_system(sp)))
        label = splits.prototype_label(lat, frac, sp)
        proto.append(repr(label) if label is not None else f"no symmetry found {rec['jid']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"jids": [rec["jid"] for rec in records], "system": system, "proto": proto}),
                    encoding="utf-8")
    return system, proto


def elements_tested(d):
    counts = Counter(e for s in d["system"] for e in s.split("-"))
    from pymatgen.core import Element

    return sorted((e for e, c in counts.items() if c >= MIN_MATERIALS), key=lambda e: Element(e).Z)


def tasks(d):
    """All tasks of the experiment, in a fixed order."""
    n = len(d["y"])
    out = []
    for split, folds in (("random", splits.random_folds(n, N_FOLDS, seed=0)),
                         ("chemical-system groups", splits.group_folds(d["system"], N_FOLDS, seed=0)),
                         ("prototype groups", splits.group_folds(d["proto"], N_FOLDS, seed=0))):
        slug = split.split()[0].split("-")[0]
        for k in range(N_FOLDS):
            test, val, cal = (np.flatnonzero(folds == (k + i) % N_FOLDS) for i in range(3))
            train = np.flatnonzero(~np.isin(folds, [k, (k + 1) % N_FOLDS, (k + 2) % N_FOLDS]))
            out.append(Task(f"{slug}-{k}", split, train, val, cal, test))
    members = [set(s.split("-")) for s in d["system"]]
    for e in elements_tested(d):
        test = np.array([i for i, m in enumerate(members) if e in m])
        rest = np.setdiff1d(np.arange(n), test)
        tr, va, ca = splits.group_split([d["system"][i] for i in rest], seed=0, fractions=(0.8, 0.1, 0.1))
        out.append(Task(f"element-{e}", "leave one element out", rest[tr], rest[va], rest[ca], test))
    tr, va, ca = splits.group_split(d["system"], seed=0, fractions=(0.8, 0.1, 0.1))
    out.append(Task("prospective", "prospective", tr, va, ca, np.arange(len(d["y_add"])), added=True))
    return out


def _sets(d, task):
    X_test = d["X_add"] if task.added else d["X"]
    return {"cal": d["X"][task.cal], "test": X_test[task.test]}


def run_cpu(d, task):
    s = SETTINGS["cpu"]
    X, y = d["X"], d["y"]
    tr, va = task.train, task.val
    hgb_kw = {"learning_rate": s["learning_rate"], "max_iter": s["max_iter"]}
    out = {}
    models = uq.fit_bootstrap_ensemble(X[tr], y[tr], X[va], y[va], n_members=s["ensemble_members"], seed=0, **hgb_kw)
    lo_m, hi_m = uq.fit_quantile_pair(X[tr], y[tr], X[va], y[va], alpha=s["quantile_alpha"], **hgb_kw)
    forest = uq.fit_random_forest(X[tr], y[tr], n_estimators=s["forest_trees"])
    for name, Xs in _sets(d, task).items():
        out[f"{name}_mu_ens"], out[f"{name}_sd_ens"] = uq.ensemble_predict(models, Xs)
        out[f"{name}_q_lo"], out[f"{name}_q_hi"] = lo_m.predict(Xs), hi_m.predict(Xs)
        out[f"{name}_mu_rf"], out[f"{name}_sd_rf"] = uq.forest_predict(forest, Xs)
        out[f"{name}_dist"] = splits.distance_to_train(X[tr], Xs, k=s["neighbors"])
    out["iterations"] = np.array([m.n_iter_ for m in models] + [lo_m.n_iter_, hi_m.n_iter_])
    out["repeated_fits"] = np.array(len(uq.REPEATED_FITS))
    uq.REPEATED_FITS.clear()
    return out


def run_gpu(d, task):
    s = SETTINGS["gpu"]
    X, y = d["X"], d["y"]
    transform = uq.standardizer(X[task.train])
    Ztr, Zva = transform(X[task.train]), transform(X[task.val])
    gp = uq.fit_svgp(Ztr, y[task.train], Zva, y[task.val], n_inducing=s["inducing_points"], epochs=s["gp_epochs"],
                     jitter=s["gp_jitter"])
    nn = uq.fit_deep_ensemble(Ztr, y[task.train], Zva, y[task.val], n_members=s["nn_members"], hidden=s["nn_hidden"],
                              max_epochs=s["nn_max_epochs"], patience=s["nn_patience"])
    out = {}
    for name, Xs in _sets(d, task).items():
        Zs = transform(Xs)
        out[f"{name}_mu_gp"], out[f"{name}_sd_gp"] = gp(Zs)
        out[f"{name}_mu_nn"], out[f"{name}_sd_nn"] = nn(Zs)
    return out


def result(d, task, part, compute=True):
    """Cached results of one part of a task: (arrays, 'cached' | 'computed'); None if missing and not computed."""
    path = CACHE / part / f"{task.key}.npz"
    settings = json.dumps(SETTINGS[part], sort_keys=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if str(z["settings"]) == settings and np.array_equal(z["cal_index"], task.cal) \
                    and np.array_equal(z["test_index"], task.test):
                return {k: z[k] for k in z.files}, "cached"
    if not compute:
        return None
    t0 = time.perf_counter()
    out = (run_cpu if part == "cpu" else run_gpu)(d, task)
    out.update(settings=np.array(settings), cal_index=task.cal, test_index=task.test,
               seconds=np.array(time.perf_counter() - t0))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(path)                       # a run stopped mid-write leaves no half-written result
    return out, "computed"


def main(part):
    if part == "gpu":
        import torch

        torch.set_num_threads(2)             # leave the processor to the cpu part running at the same time
    d = load()
    todo = tasks(d)
    t0 = time.perf_counter()
    for i, task in enumerate(todo, 1):
        _, how = result(d, task, part)
        print(f"[{part}] {i}/{len(todo)} {task.key}: {how} ({time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in SETTINGS:
        sys.exit("usage: python runs.py cpu|gpu")
    main(sys.argv[1])
