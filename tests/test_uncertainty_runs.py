"""The target check of experiments/uncertainty/runs.py (also used by experiments/cgcnn/runs.py)."""
import importlib.util
import math
import sys
from pathlib import Path

import pytest

RUNS = Path(__file__).resolve().parents[1] / "experiments" / "uncertainty" / "runs.py"


@pytest.fixture(scope="module")
def runs():
    name = "uncertainty_runs_under_test"
    spec = importlib.util.spec_from_file_location(name, RUNS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # dataclasses look their module up here
    spec.loader.exec_module(module)
    yield module
    del sys.modules[name]


def test_finite_targets_pass(runs):
    runs.check_targets([-1.2, 0.0, 0.3], [{"jid": "a"}, {"jid": "b"}, {"jid": "c"}], "2021.8.18")


def test_a_missing_target_names_the_material(runs):
    with pytest.raises(ValueError, match=r"1 materials of the added records .*\['b'\]"):
        runs.check_targets([-1.2, math.nan, 0.3], [{"jid": "a"}, {"jid": "b"}, {"jid": "c"}], "added")


def test_targets_and_records_must_match_in_length(runs):
    with pytest.raises(ValueError):
        runs.check_targets([-1.2, 0.3], [{"jid": "a"}, {"jid": "b"}, {"jid": "c"}], "added")
