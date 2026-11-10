"""The fit cache of experiments/pretrained_features/runs.py: a stored fit is reused only if it was made on the same
features and the same materials. Small synthetic features; no JARVIS data, model weights or GPU needed."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")        # runs.py imports materials_ml_primer.pretrained
pytest.importorskip("sklearn")

RUNS = Path(__file__).resolve().parents[1] / "experiments" / "pretrained_features" / "runs.py"
N, N_ADDED = 90, 30


def load_runs():
    spec = importlib.util.spec_from_file_location("pretrained_features_runs", RUNS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runs(tmp_path, monkeypatch):
    module = load_runs()
    monkeypatch.setattr(module, "CACHE", tmp_path)
    monkeypatch.setattr(module, "FIT_SETTINGS", {**module.FIT_SETTINGS, "ensemble_members": 2, "max_iter": 20})
    return module


def feature_file(runs, X, settings=None):
    """The dict that runs.features returns, for features X."""
    settings = runs.FEATURE_SETTINGS if settings is None else settings
    return {"X": X, "settings": np.array(json.dumps(settings, sort_keys=True)), "digest": runs.digest(X)}


def setup(runs, added=False):
    rng = np.random.default_rng(0)
    X, X_add = rng.normal(size=(N, 4)).astype(np.float32), rng.normal(size=(N_ADDED, 4)).astype(np.float32)
    d = {"y": 2.0 * X[:, 0] + rng.normal(scale=0.1, size=N)}
    task = SimpleNamespace(key="task", train=np.arange(0, 60), val=np.arange(60, 70), cal=np.arange(70, 80),
                           test=np.arange(0, N_ADDED) if added else np.arange(80, 90), added=added)
    return d, task, [feature_file(runs, X), feature_file(runs, X_add)]


def changed(X):
    X = X.copy()
    X[2, 1] += 1e-3
    return X


def test_a_fit_is_reused_on_the_same_features_and_materials(runs):
    d, task, feats = setup(runs)
    first, how = runs.fit(d, "m", task, feats)
    assert how == "computed"
    again, how = runs.fit(d, "m", task, feats)
    assert how == "cached"
    assert np.array_equal(first["test_mu"], again["test_mu"])
    assert np.array_equal(again["train_index"], task.train) and np.array_equal(again["val_index"], task.val)


def test_a_fit_on_other_features_is_not_reused(runs):
    d, task, feats = setup(runs)
    runs.fit(d, "m", task, feats)
    other = [feature_file(runs, changed(feats[0]["X"])), feats[1]]
    assert runs.fit(d, "m", task, other, compute=False) is None
    assert runs.fit(d, "m", task, other)[1] == "computed"


@pytest.mark.parametrize("part", ["train", "val", "cal", "test"])
def test_a_fit_on_other_materials_is_not_reused(runs, part):
    d, task, feats = setup(runs)
    runs.fit(d, "m", task, feats)
    moved = SimpleNamespace(**{**vars(task), part: getattr(task, part)[1:]})
    assert runs.fit(d, "m", moved, feats, compute=False) is None


def test_only_a_prospective_fit_depends_on_the_added_materials_features(runs):
    for added in (False, True):
        d, task, feats = setup(runs, added=added)
        runs.fit(d, f"m{added}", task, feats)
        other = [feats[0], feature_file(runs, changed(feats[1]["X"]))]
        result = runs.fit(d, f"m{added}", task, other, compute=False)
        assert (result is None) == added


def test_the_rotated_variant_needs_no_added_materials(runs):
    assert runs.parts("orb-v3-omat-rotated") == ("2021",)
    assert runs.parts("orb-v3-omat") == ("2021", "added")
    d, task, feats = setup(runs)
    assert runs.fit(d, "orb-v3-omat-rotated", task, feats[:1])[1] == "computed"


def legacy(path):
    """Rewrite a stored fit as the fit cache wrote it before it recorded the features and the training and
    validation materials."""
    with np.load(path, allow_pickle=False) as z:
        old = {k: z[k] for k in z.files if k not in ("feature_settings", "feature_digest", "train_index", "val_index")}
    np.savez_compressed(path, **old)


def test_fits_stored_before_the_digest_are_reused_while_the_features_have_version_1_settings(runs):
    assert runs.FEATURE_SETTINGS == runs.LEGACY_FEATURE_SETTINGS
    d, task, feats = setup(runs)
    runs.fit(d, "m", task, feats)
    path = runs.CACHE / "fits" / "m" / "task.npz"
    legacy(path)
    assert runs.fit(d, "m", task, feats)[1] == "cached"
    # features recomputed with other settings: the old fit is not reused, and the new one records its features
    version_2 = [feature_file(runs, f["X"], {**runs.LEGACY_FEATURE_SETTINGS, "version": 2}) for f in feats]
    assert runs.fit(d, "m", task, version_2, compute=False) is None
    out, how = runs.fit(d, "m", task, version_2)
    assert how == "computed" and str(out["feature_digest"]) == feats[0]["digest"]


def test_fits_stored_before_the_digest_still_check_the_calibration_and_test_materials(runs):
    d, task, feats = setup(runs)
    runs.fit(d, "m", task, feats)
    legacy(runs.CACHE / "fits" / "m" / "task.npz")
    moved = SimpleNamespace(**{**vars(task), "test": task.test[1:]})
    assert runs.fit(d, "m", moved, feats, compute=False) is None


def test_digest_depends_on_values_dtype_and_shape():
    module = load_runs()
    X = np.arange(12, dtype=np.float32).reshape(3, 4)
    digests = {module.digest(a) for a in (X, changed(X), X.astype(np.float64), X.reshape(4, 3), X.T)}
    assert len(digests) == 5
    assert module.digest(X) == module.digest(X.copy(order="F"))
