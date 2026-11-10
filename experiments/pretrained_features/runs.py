"""Computations of the pretrained_features experiment, cached so that a full run can be split and resumed.

Two steps:

- "features": the features of one pretrained model (see ``materials_ml_primer.pretrained``) for every structure of
  the JARVIS-DFT 2021.8.18 snapshot and of the materials added up to 2025.9.24, stored in
  data/pretrained_cache/features/<model>_<2021|added>.npz. Each model runs in its own process (Orb switches off
  TorchDynamo for the whole process). The variant "orb-v3-omat-rotated" turns every structure by a random rotation
  first (Orb is not rotation invariant; see the notebook); it is computed for the 2021 records only, since none of
  its tasks is prospective.
- "fit": for every task of the uncertainty experiment (the same training, validation, calibration and test
  materials), the bootstrap ensemble of five gradient-boosting models of that experiment, trained on the
  features instead of the 150 descriptors; predictions for the calibration and test sets are stored in
  data/pretrained_cache/fits/<model>/<task>.npz. The rotated variant is fitted only on ROTATION_TASKS.
  A stored fit is reused only if it was made with the same fit settings, on the same features (a SHA-256 digest
  of the feature arrays it used) and on the same training, validation, calibration and test materials.

From this folder:

    python runs.py features mace-mp-0b3       # and chgnet-0.3.0, orb-v3-omat, orb-v3-omat-rotated
    python runs.py fit mace-mp-0b3            # and the others

The fits run on the CPU. Gradient boosting on ~40 000 materials does not use 16 threads well, so several
processes with a few threads each finish sooner; each takes every n-th job of a fixed shuffled list:

    OMP_NUM_THREADS=4 python runs.py fit chgnet-0.3.0,mace-mp-0b3,orb-v3-omat,orb-v3-omat-rotated 0 4
    (and with 1 4, 2 4, 3 4 in three more terminals)

The workers read the task definitions from data/pretrained_cache/tasks.npz (written from the uncertainty
experiment's definitions on first use) instead of loading the JARVIS records.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

from materials_ml_primer import data, pretrained
from materials_ml_primer import uncertainty as uq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_DIR = ROOT / "data"
CACHE = DATA_DIR / "pretrained_cache"
MODELS = list(pretrained.MODELS)
VARIANTS = MODELS + ["orb-v3-omat-rotated"]
ROTATION_TASKS = ["random-0", "chemical-0", "prototype-0", "element-F", "element-O", "element-Cu"]
FEATURE_SETTINGS = {"version": 1, "summary": "mean and std over atoms", "max_atoms": 2000, "dtype": "float32"}
FIT_SETTINGS = {"version": 1, "ensemble_members": 5, "learning_rate": 0.1, "max_iter": 2500}   # as uncertainty "cpu"
# Fits stored before the fit cache recorded its features hold no feature digest; they were all made on features of
# these settings, and are reused only while the feature files still have them.
LEGACY_FEATURE_SETTINGS = {"version": 1, "summary": "mean and std over atoms", "max_atoms": 2000, "dtype": "float32"}


def parts(variant):
    """The feature files of a variant: the 2021 records and the added materials, or only the 2021 records for the
    rotated variant (its tasks, ROTATION_TASKS, test on 2021 materials)."""
    return ("2021",) if variant.endswith("-rotated") else ("2021", "added")


def digest(X):
    """SHA-256 of a feature array (its dtype, shape and values)."""
    X = np.ascontiguousarray(X)
    h = hashlib.sha256(f"{X.dtype.str} {X.shape}".encode())
    h.update(X.tobytes())
    return h.hexdigest()


def _uncertainty_runs():
    """The task definitions of the uncertainty experiment (experiments/uncertainty/runs.py)."""
    name = "uncertainty_runs"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE.parent / "uncertainty" / "runs.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module          # dataclasses look their module up here
        spec.loader.exec_module(module)
    return sys.modules[name]


def load():
    """The data of the uncertainty experiment (records, descriptors, targets, group labels)."""
    return _uncertainty_runs().load()


def rotation(i):
    """The random proper rotation applied to structure i in the rotated variant."""
    q, r = np.linalg.qr(np.random.default_rng([12345, i]).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _atoms(records, rotate):
    out = []
    for i, rec in enumerate(records):
        lat, frac, sp = data.structure_arrays(rec["atoms"])
        out.append(pretrained.to_atoms(lat @ rotation(i).T if rotate else lat, frac, sp))
    return out


def task_file(d=None):
    """The tasks of the uncertainty experiment with the targets and material ids, from data/pretrained_cache/
    tasks.npz (written on first use): a dict with "tasks" (uncertainty Task objects), "y", "y_add", "jids_2021",
    "jids_added"."""
    uqr = _uncertainty_runs()
    path = CACHE / "tasks.npz"
    if not path.exists():
        d = d if d is not None else load()
        arrays = {"y": d["y"], "y_add": d["y_add"], "jids_2021": np.array([r["jid"] for r in d["records"]]),
                  "jids_added": np.array([r["jid"] for r in d["added"]])}
        every = uqr.tasks(d)
        arrays.update(keys=np.array([t.key for t in every]), splits=np.array([t.split for t in every]),
                      added=np.array([t.added for t in every]))
        for t in every:
            for part in ("train", "val", "cal", "test"):
                arrays[f"{t.key}__{part}"] = getattr(t, part)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in ("y", "y_add")}
        out["jids_2021"], out["jids_added"] = z["jids_2021"].tolist(), z["jids_added"].tolist()
        out["tasks"] = [uqr.Task(str(k), str(s), *(z[f"{k}__{p}"] for p in ("train", "val", "cal", "test")),
                                 added=bool(a)) for k, s, a in zip(z["keys"], z["splits"], z["added"], strict=True)]
    return out


def _jids(d, part):
    if "jids_2021" in d:
        return d["jids_2021"] if part == "2021" else d["jids_added"]
    return [rec["jid"] for rec in (d["records"] if part == "2021" else d["added"])]


def features(d, variant, part, compute=True, device="cuda", log=print):
    """Features of the 2021 records (part "2021") or of the added materials ("added"); None if not cached and
    not computed. ``d`` is the data of ``load`` or the dict of ``task_file`` (enough to read the cache).
    The returned dict holds the arrays of the cache file and "digest", the digest of "X" (not stored)."""
    path = CACHE / "features" / f"{variant}_{part}.npz"
    jids = _jids(d, part)
    settings = json.dumps(FEATURE_SETTINGS, sort_keys=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if str(z["settings"]) == settings and z["jids"].tolist() == jids:
                out = {k: z[k] for k in z.files}
                return out | {"digest": digest(out["X"])}
    if not compute:
        return None
    records = d["records"] if part == "2021" else d["added"]
    model = variant.removesuffix("-rotated")
    extractor = pretrained.load(model, device)
    atoms = _atoms(records, rotate=variant.endswith("-rotated"))
    X, energy, seconds = pretrained.featurize(extractor, atoms, max_atoms=FEATURE_SETTINGS["max_atoms"],
                                              log=lambda m: log(f"{variant} {part}: {m}"))
    out = {"X": X, "energy": energy, "seconds": np.array(seconds), "jids": np.array(jids),
           "isolated_atoms": np.array(getattr(extractor, "isolated_atoms", -1)), "settings": np.array(settings)}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(path)
    return out | {"digest": digest(out["X"])}


def tasks(d, variant):
    every = _uncertainty_runs().tasks(d)
    return [t for t in every if t.key in ROTATION_TASKS] if variant.endswith("-rotated") else every


def _is_current(z, settings, feature_settings, feature_digest, task):
    """Whether the stored fit ``z`` was made with these fit settings, on these features and on the materials of
    ``task``. Fits stored before the fit cache recorded its features and its training and validation materials
    are checked on what they hold: the fit settings, the calibration and test materials, and that the features
    still have the settings they were all made with (LEGACY_FEATURE_SETTINGS)."""
    if str(z["settings"]) != settings:
        return False
    for part in ("train", "val", "cal", "test"):
        key = f"{part}_index"
        if key in z.files and not np.array_equal(z[key], getattr(task, part)):
            return False
        if key not in z.files and part in ("cal", "test"):
            return False
    if "feature_digest" in z.files:
        return str(z["feature_digest"]) == feature_digest
    return feature_settings == json.dumps(LEGACY_FEATURE_SETTINGS, sort_keys=True)


def fit(d, variant, task, feats, compute=True):
    """Cached bootstrap ensemble of gradient-boosting models on the features of one variant, for one task.

    ``feats``: the features of the variant, the dicts of ``features`` for its ``parts`` (the 2021 records, and the
    added materials, which only a prospective task uses)."""
    path = CACHE / "fits" / variant / f"{task.key}.npz"
    settings = json.dumps(FIT_SETTINGS, sort_keys=True)
    used = feats[:2] if task.added else feats[:1]
    feature_settings = str(feats[0]["settings"])
    feature_digest = " ".join(f["digest"] for f in used)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if _is_current(z, settings, feature_settings, feature_digest, task):
                return {k: z[k] for k in z.files}, "cached"
    if not compute:
        return None
    s, y = FIT_SETTINGS, d["y"]
    X = feats[0]["X"]
    t0 = time.perf_counter()
    models = uq.fit_bootstrap_ensemble(X[task.train], y[task.train], X[task.val], y[task.val],
                                       n_members=s["ensemble_members"], seed=0, learning_rate=s["learning_rate"],
                                       max_iter=s["max_iter"])
    out = {}
    for name, Xs in (("cal", X[task.cal]), ("test", (feats[1]["X"] if task.added else X)[task.test])):
        out[f"{name}_mu"], out[f"{name}_sd"] = uq.ensemble_predict(models, Xs)
    out.update(iterations=np.array([m.n_iter_ for m in models]), seconds=np.array(time.perf_counter() - t0),
               settings=np.array(settings), feature_settings=np.array(feature_settings),
               feature_digest=np.array(feature_digest), train_index=task.train, val_index=task.val,
               cal_index=task.cal, test_index=task.test)
    uq.REPEATED_FITS.clear()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(path)
    return out, "computed"


def fit_jobs(task_list, variants):
    """(variant, task) of every fit, in a fixed shuffled order (so that workers taking every n-th job get a
    similar mix of cheap 128-feature and expensive 512-feature fits)."""
    jobs = [(v, t) for t in task_list for v in variants if not v.endswith("-rotated") or t.key in ROTATION_TASKS]
    return [jobs[i] for i in np.random.default_rng(0).permutation(len(jobs))]


def main(step, variants, worker=0, n_workers=1):
    if step == "features":
        d = load()
        task_file(d)
        for variant in variants:
            for part in parts(variant):
                r = features(d, variant, part)
                print(f"{variant} {part}: {r['X'].shape}, {float(r['seconds']):.0f} s", flush=True)
        return
    T = task_file()
    feats = {}
    for v in variants:
        feats[v] = [features(T, v, part, compute=False) for part in parts(v)]
        if any(f is None for f in feats[v]):
            sys.exit(f"features of {v} are not computed yet: python runs.py features {v}")
    todo = fit_jobs(T["tasks"], variants)[worker::n_workers]
    t0 = time.perf_counter()
    for i, (v, task) in enumerate(todo, 1):
        _, how = fit(T, v, task, feats[v])
        print(f"[{worker}/{n_workers}] {i}/{len(todo)} {v} {task.key}: {how} ({time.perf_counter() - t0:.0f} s)",
              flush=True)


if __name__ == "__main__":
    names = sys.argv[2].split(",") if len(sys.argv) in (3, 5) else []
    if sys.argv[1:2] not in (["features"], ["fit"]) or not names or any(n not in VARIANTS for n in names):
        sys.exit("usage: python runs.py features|fit VARIANT[,VARIANT...] [WORKER N_WORKERS]   (variants: "
                 + ", ".join(VARIANTS) + ")")
    main(sys.argv[1], names, *(int(a) for a in sys.argv[3:5]))
