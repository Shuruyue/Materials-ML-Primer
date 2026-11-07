# Materials ML Primer

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Small, tested building blocks for machine learning on crystal structures, and experiments that ask how far each one can be trusted. Every method is first shown to be correct against something independent (**verification**: closed forms, symmetry invariants, reference software), then applied to real data and compared with published results (**validation**). The recurring question is the one every materials ML result has to answer: how well does the model do on materials unlike the ones it was trained on, and does it know when it is unreliable?

Every experiment is sized to run in hours, not days, on a single 12 GB consumer GPU.

## Reading order

| Order | Experiment | Question | Code | Status |
|---|---|---|---|---|
| 1 | [crystal_graphs](experiments/crystal_graphs/) | How is a periodic crystal turned into a graph, and how do we know the neighbor list is right? | `graphs` | done: shells of six structures equal exact lattice sums; 19 770 edges identical to pymatgen; 55 tests |
| 2 | baselines | How far do composition and structure descriptors with random forests and gradient boosting get? | `baselines` | planned |
| 3 | data_splits | How much does a random train/test split overstate accuracy compared with chemistry- and structure-aware splits? | `splits` | planned |
| 4 | uncertainty | Can a model tell when its prediction is unreliable? Ensembles, Gaussian processes and Bayesian methods | `uncertainty` | planned |
| 5 | cgcnn | A faithful re-implementation of CGCNN (Xie & Grossman, 2018), trained in stages | `cgcnn` | planned |
| 6 | pretrained_features | Pretrained universal models used as feature extractors | `pretrained` | planned |

Later experiments reuse earlier code (the CGCNN experiment uses the graphs of the first), so the reusable code lives in one package and the experiments import it.

## Repository layout

```text
Materials-ML-Primer/
├── pyproject.toml
├── environment.yml           # conda environment "mmp"
├── materials_ml_primer/      # reusable code, one module per topic
├── tests/                    # pytest suite for the package
└── experiments/
    └── <topic>/              # one folder per question
        ├── README.md         # goal, key results, validation table, figures
        ├── <topic>.ipynb     # executed notebook: every number and figure
        ├── <topic>.html      # the same notebook as a web page
        └── figures/
```

Downloaded datasets and trained model weights are not committed; the notebooks download or regenerate them.

## Quick start

```bash
conda env create -f environment.yml   # once; installs PyTorch with CUDA 13.2 wheels
conda activate mmp
pip install -e .                      # makes materials_ml_primer importable everywhere
pytest -q                             # automated tests
cd experiments/crystal_graphs
jupyter lab crystal_graphs.ipynb      # Run All reproduces every number and figure
```

## License

[MIT](LICENSE)
