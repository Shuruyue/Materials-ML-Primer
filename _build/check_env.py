"""Print package versions and run quick checks of NumPy, PyTorch (CUDA) and pymatgen.

Run inside the environment: python _build/check_env.py
"""
import importlib.metadata as md
import sys

import numpy as np
import torch
import torch_geometric  # noqa: F401  (only checks that it imports)
from pymatgen.core import Lattice, Structure

PACKAGES = ["numpy", "scipy", "matplotlib", "pandas", "pymatgen", "scikit-learn",
            "torch", "torch-geometric", "jarvis-tools", "jupyterlab", "pytest"]


def main():
    print("python", sys.version.split()[0])
    for name in PACKAGES:
        try:
            print(f"{name:16s} {md.version(name)}")
        except md.PackageNotFoundError:
            print(f"{name:16s} MISSING")

    print("numpy.linalg ok:", np.allclose(np.linalg.inv(np.eye(3) * 2), np.eye(3) / 2))

    print("cuda available:", torch.cuda.is_available(), "| torch cuda:", torch.version.cuda)
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        x = torch.randn(2048, 2048, device="cuda")
        print("matmul on GPU ok:", bool(torch.isfinite(x @ x).all()))

    s = Structure(Lattice.cubic(3.615), ["Cu"] * 4,
                  [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    print("fcc Cu neighbors within 2.6 A (expect 12 each):", [len(n) for n in s.get_all_neighbors(2.6)])


if __name__ == "__main__":
    main()
