"""JARVIS-DFT 3D data: download, de-duplication and conversion to arrays.

The default snapshot is JARVIS-DFT 2021.8.18 (``dft_3d_2021`` in jarvis-tools),
the one used for the CFID, CGCNN and ALIGNN results in Choudhary & DeCost,
npj Comput. Mater. 7, 185 (2021), so that results on it can be compared with
theirs. Newer snapshots exist (``dft_3d_2022``, ``dft_3d``); their numbers are
not comparable with that paper.
"""

from __future__ import annotations

import contextlib
import io
import math
from collections import Counter
from pathlib import Path

import numpy as np

DFT_3D_2021 = "dft_3d_2021"
CFID_3D_2021 = "cfid_3d"


def download(name, data_dir):
    """Records of a JARVIS dataset, downloaded into ``data_dir`` on first use.

    The messages jarvis-tools prints while loading are suppressed.
    """
    from jarvis.db.figshare import data

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        return data(name, store_dir=str(data_dir))


def deduplicate(records, key="jid"):
    """Keep the first record of every key.

    Returns the kept records and a dict {key: number of copies} for every key
    that occurred more than once.
    """
    counts = Counter(r[key] for r in records)
    seen = set()
    kept = []
    for r in records:
        if r[key] in seen:
            continue
        seen.add(r[key])
        kept.append(r)
    return kept, {k: n for k, n in counts.items() if n > 1}


def as_float(value):
    """JARVIS stores missing values as the string 'na'; return NaN for those."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out


def structure_arrays(atoms):
    """Lattice (rows, angstrom), fractional coordinates and species of a JARVIS atoms dict."""
    lattice = np.asarray(atoms["lattice_mat"], dtype=float)
    coords = np.asarray(atoms["coords"], dtype=float)
    if atoms["cartesian"]:
        frac = np.linalg.solve(lattice.T, coords.T).T
    else:
        frac = coords
    return lattice, frac, list(atoms["elements"])


def composition(species):
    """Element counts {symbol: count} of a list of species."""
    return dict(Counter(species))
