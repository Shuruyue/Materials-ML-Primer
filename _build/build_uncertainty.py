"""Build experiments/uncertainty/uncertainty.ipynb with nbformat. Run from that folder.

    python ../../_build/build_uncertainty.py uncertainty.ipynb [--quick]

The models are fitted by experiments/uncertainty/runs.py, which caches every task in data/uncertainty_cache;
the notebook computes what is not cached. --quick writes a notebook that uses only the cached tasks and computes
nothing, to test the analysis while the cache is being filled.
"""
import sys

import nbformat as nbf

out = sys.argv[1]
QUICK = "--quick" in sys.argv[2:]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md(r"""
# Uncertainty: can a model tell when its prediction is unreliable?

The `data_splits` experiment showed that the error of a formation-energy model grows when the test materials are unlike the training set (new chemical systems, new structure types, unseen elements, materials computed later), and that the distance to the training set flags risky regions but not risky materials. Here every prediction comes with an uncertainty, from six methods, and the question is whether that uncertainty

1. **ranks** the predictions: are the most uncertain ones the most wrong (rank correlation, retention curves)?
2. is **calibrated**: does a stated standard deviation or 90 % interval mean what it says?
3. stays useful under **distribution shift**, the situations in which it matters most.

| | Method | Uncertainty | Reference |
|---|---|---|---|
| A | distance to the training set | mean distance to the 5 nearest training materials (no scale) | `data_splits` |
| B | bootstrap ensemble | standard deviation of 5 gradient-boosting models trained on bootstrap samples | Efron & Tibshirani 1993 |
| C | quantile regression | width of the 5 %–95 % band of two gradient-boosting quantile models | Koenker & Bassett 1978 |
| D | random forest | standard deviation of the 300 trees | Breiman 2001 |
| E | conformal prediction | turns any of the above into intervals with guaranteed coverage under exchangeability | Vovk et al. 2005; Romano et al. 2019 |
| F | Gaussian process | posterior standard deviation of a sparse variational GP (1024 inducing points) | Hensman et al. 2013 |
| G | deep ensemble | mixture of 5 neural networks that each predict a mean and a variance | Lakshminarayanan et al. 2017 |

Everything uses the 150 composition and structure features of the `baselines` experiment and the JARVIS-DFT formation energy.
""")

code(r"""
import json
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
from scipy import stats

import runs
from materials_ml_primer import uncertainty as uq

warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.* is deprecated", category=FutureWarning)  # from GPyTorch
QUICK = @QUICK@       # True only in a test run of this notebook: use the cached tasks, compute nothing
ALPHA = runs.ALPHA    # 90 % intervals
FRACTIONS = np.linspace(0.05, 1, 96)     # retention curves: fraction of the predictions kept
SPLITS = ["random", "chemical-system groups", "prototype groups", "leave one element out", "prospective"]
# uncertainty scores: (prediction they belong to, score, is it a Gaussian standard deviation?)
METHODS = {
    "distance": ("mu_ens", "dist", False),
    "bootstrap ensemble": ("mu_ens", "sd_ens", True),
    "quantile regression": ("mu_ens", "q_width", False),
    "random forest": ("mu_rf", "sd_rf", True),
    "Gaussian process": ("mu_gp", "sd_gp", True),
    "deep ensemble": ("mu_nn", "sd_nn", True),
}
# conformal intervals: (prediction, scale of the scores; None: plain residuals, "cqr": quantile band)
CONFORMAL = {
    "residual (constant width)": ("mu_ens", None),
    "distance": ("mu_ens", "dist"),
    "bootstrap ensemble": ("mu_ens", "sd_ens"),
    "quantile regression (CQR)": (None, "cqr"),
    "random forest": ("mu_rf", "sd_rf"),
    "Gaussian process": ("mu_gp", "sd_gp"),
    "deep ensemble": ("mu_nn", "sd_nn"),
}
COLORS = {m: f"C{i}" for i, m in enumerate(METHODS)}
CONF_COLORS = {"residual (constant width)": "0.3", "quantile regression (CQR)": COLORS["quantile regression"],
               **{n: COLORS[n] for n in CONFORMAL if n in COLORS}}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "font.size": 10})
summary = []          # rows of the summary table: (quantity, this run, check, check from)
""".replace("@QUICK@", str(QUICK)))

code(r"""
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "--color=no", "-p", "no:cacheprovider",
                    "../../tests/test_uncertainty.py"], capture_output=True, text=True)
print(r.stdout[-1500:])
""")

md(r"""
## 1. Verification: does the conformal coverage behave as the theory says?

Split conformal prediction takes the ⌈(n + 1)(1 − α)⌉-th smallest of n calibration scores as the threshold. With exchangeable calibration and test points, the coverage is then at least 1 − α on average; given one calibration set it is a random number with the distribution Beta(n + 1 − l, l), l = ⌊(n + 1)α⌋ (Vovk 2012; Angelopoulos & Bates, arXiv:2107.07511). The pytest suite checks this against exact results; here the same check is drawn: 4000 calibration sets of n = 200 points with standard normal residuals, for which the coverage of each threshold q is exactly 2Φ(q) − 1.
""")

code(r"""
rng = np.random.default_rng(0)
n_demo = 200
cover = np.array([2 * stats.norm.cdf(uq.conformal_quantile(np.abs(rng.normal(size=n_demo)), ALPHA)) - 1
                  for _ in range(4000)])
a, b = uq.coverage_beta(n_demo, ALPHA)
ks = stats.kstest(cover, stats.beta(a, b).cdf)
print(f"mean coverage {cover.mean():.4f}; theory: at least {1 - ALPHA}, at most {1 - ALPHA + 1 / (n_demo + 1):.4f}, "
      f"Beta mean {a / (a + b):.4f}; Kolmogorov-Smirnov p = {ks.pvalue:.2f}")
fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.hist(cover, bins=40, density=True, color="0.75", label="simulated, 4000 calibration sets")
xs = np.linspace(cover.min(), cover.max(), 300)
ax.plot(xs, stats.beta(a, b).pdf(xs), color="C3", label=f"Beta({a}, {b})")
ax.axvline(1 - ALPHA, color="k", lw=0.8, ls="--")
ax.set_xlabel("coverage given the calibration set")
ax.set_ylabel("density")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
plt.show()
summary.append(("conformal coverage of 4000 simulated calibration sets (n = 200)", f"KS p = {ks.pvalue:.2f}",
                "Beta(n + 1 - l, l)", "Vovk 2012; Angelopoulos & Bates"))
""")

md(r"""
## 2. Data and the 109 test sets

Every test set ("task") has its own training, validation and calibration sets. The validation set is used only for early stopping; the calibration set is used only by conformal prediction, so that its scores stay exchangeable with those of the test set when the test set is drawn like it.

- **Cross-validation** (random, chemical-system and prototype groups, the folds of `data_splits`): fold k is the test set, k + 1 the validation set, k + 2 the calibration set and the other seven folds the training set (70/10/10/10).
- **Leave one element out** (78 elements) and the **prospective** test: the remaining materials are split by chemical system into training, validation and calibration sets (80/10/10).

Why these proportions: Sesia & Candès (Stat 2020, arXiv:1909.05433) varied the share of the data used for training against calibration on twelve data sets and found 70–90 % for training the best compromise; here training gets 7/9 = 78 % of the non-test data in cross-validation and 80 % otherwise. The calibration sets hold more than 5000 materials; the next cell computes, from the Beta distribution above, how close to 90 % the coverage of each split is then guaranteed to be.

A material that contains several of the 78 elements is in the test set of each of them, so the pooled leave-one-element-out numbers below count it once per element.
""")

code(r"""
t0 = time.perf_counter()
d = runs.load()
TASKS = runs.tasks(d)
print(f"data loaded in {time.perf_counter() - t0:.0f} s: {len(d['y'])} materials of 2021.8.18, "
      f"{len(d['y_add'])} added up to 2025.9.24; {d['X'].shape[1]} features")
print(f"{'split':24s} {'tasks':>5s} {'train':>7s} {'val':>6s} {'cal':>6s} {'test (total)':>13s} {'P(|coverage - 0.9| < 0.01)':>27s}")
for split in SPLITS:
    ts = [t for t in TASKS if t.split == split]
    n_cal = int(np.median([len(t.cal) for t in ts]))
    beta = stats.beta(*uq.coverage_beta(n_cal, ALPHA))
    p = beta.cdf(0.91) - beta.cdf(0.89)
    print(f"{split:24s} {len(ts):5d} {int(np.median([len(t.train) for t in ts])):7d} {int(np.median([len(t.val) for t in ts])):6d} "
          f"{n_cal:6d} {sum(len(t.test) for t in ts):13d} {p:27.4f}")
""")

md(r"""
## 3. Fitting all models

`runs.py` fits the models of every task and caches the predictions for the calibration and test sets (the "cpu" part: bootstrap ensemble, quantile regression, random forest, distance; the "gpu" part: Gaussian process and deep ensemble). This cell loads the cached tasks and computes the missing ones.
""")

code(r"""
counts, seconds, missing, results = Counter(), defaultdict(float), [], []
for task in TASKS:
    parts = {}
    for part in ("cpu", "gpu"):
        res = runs.result(d, task, part, compute=not QUICK)
        if res is None:
            break
        parts[part], how = res
        counts[(part, how)] += 1
        seconds[part] += float(parts[part]["seconds"])
    if len(parts) < 2:
        missing.append(task.key)
        continue
    r = {**parts["cpu"], **parts["gpu"]}
    for s in ("cal", "test"):
        r[f"{s}_q_width"] = r[f"{s}_q_hi"] - r[f"{s}_q_lo"]
    r["y_cal"] = d["y"][task.cal]
    r["y_test"] = (d["y_add"] if task.added else d["y"])[task.test]
    r["task"] = task
    results.append(r)
print({f"{p} {h}": c for (p, h), c in sorted(counts.items())})
print(f"fitting time stored with the results: cpu {seconds['cpu'] / 3600:.1f} h, gpu {seconds['gpu'] / 3600:.1f} h; "
      f"tasks used: {len(results)} of {len(TASKS)}" + (f"; missing: {', '.join(missing[:8])} ..." if missing else ""))
repeated = sum(int(r["repeated_fits"]) for r in results if "repeated_fits" in r)
print(f"fits repeated after a thread-pool error on Windows (the results do not change): {repeated}")
iters = np.concatenate([r["iterations"][:5] for r in results])
print(f"ensemble members that stopped at the iteration limit ({runs.SETTINGS['cpu']['max_iter']}): "
      f"{np.mean(iters == runs.SETTINGS['cpu']['max_iter']):.0%}")
""")

md(r"""
### Is the Gaussian process trained long enough?

The sparse Gaussian process is the most expensive model per unit of accuracy, so its settings were chosen with a convergence check on the first random fold: training longer and with more inducing points, does accuracy or ranking change? The experiment uses 1024 inducing points and 60 epochs; 2048 inducing points would take more than four times as long for every task.
""")

code(r"""
if not QUICK:
    t = TASKS[0]
    transform = uq.standardizer(d["X"][t.train])
    Ztr, Zva, Zte = (transform(d["X"][idx]) for idx in (t.train, t.val, t.test))
    print(f"{'inducing points':>16s} {'epochs':>7s} {'test MAE':>9s} {'Spearman':>9s} {'validation NLL':>15s} {'sd range':>12s} {'seconds':>8s}")
    for n_ind, epochs in ((1024, 20), (1024, 60), (2048, 60)):
        t0 = time.perf_counter()
        gp = uq.fit_svgp(Ztr, d["y"][t.train], Zva, d["y"][t.val], n_inducing=n_ind, epochs=epochs)
        mu, sd = gp(Zte)
        vmu, vsd = gp(Zva)
        err = np.abs(mu - d["y"][t.test])
        print(f"{n_ind:16d} {epochs:7d} {err.mean():9.4f} {uq.spearman(sd, err):9.3f} "
              f"{uq.gaussian_nll(d['y'][t.val], vmu, vsd):15.3f} {sd.min():5.3f}-{sd.max():5.3f} {time.perf_counter() - t0:8.0f}")
""")

md(r"""
## 4. Conformal intervals for every task

For each task and each conformal variant, the scores of its calibration set give the threshold and the test set gets 90 % intervals: plain residuals |y − μ| (constant width), residuals divided by an uncertainty score (width proportional to it), or the conformalized quantile band (CQR).
""")

code(r"""
def conformal(r, name):
    mu_key, scale = CONFORMAL[name]
    if scale == "cqr":
        lo, hi = uq.cqr_interval(r["test_q_lo"], r["test_q_hi"], uq.cqr_scores(r["y_cal"], r["cal_q_lo"], r["cal_q_hi"]), ALPHA)
    else:
        s_cal = None if scale is None else r[f"cal_{scale}"]
        s_test = None if scale is None else r[f"test_{scale}"]
        lo, hi = uq.conformal_interval(r[f"test_{mu_key}"], uq.residual_scores(r["y_cal"], r[f"cal_{mu_key}"], s_cal),
                                       ALPHA, scale=s_test)
    return (r["y_test"] >= lo) & (r["y_test"] <= hi), hi - lo


for r in results:
    r["covered"], r["width"] = {}, {}
    for name in CONFORMAL:
        r["covered"][name], r["width"][name] = conformal(r, name)


def pooled(split, key):
    return np.concatenate([r[key] for r in results if r["task"].split == split])


print(f"{'split':24s}" + "".join(f"{n.split(' (')[0][:14]:>15s}" for n in CONFORMAL))
cov_table = {}
for split in SPLITS:
    rs = [r for r in results if r["task"].split == split]
    if not rs:
        continue
    cov_table[split] = {n: float(np.mean(np.concatenate([r["covered"][n] for r in rs]))) for n in CONFORMAL}
    print(f"{split:24s}" + "".join(f"{cov_table[split][n]:15.3f}" for n in CONFORMAL))
""")

md(r"""
### Check on the random split

In the random split, calibration and test folds are exchangeable, so the coverage of every fold must lie in the range the theory allows: given n calibration points, the fraction covered among m test points is beta-binomial, BetaBin(m, n + 1 − l, l) (Angelopoulos & Bates, Appendix C). In the group splits the calibration and test folds hold different groups, and the materials of a group are alike, so the coverage varies more from fold to fold than this range for independent materials allows; their counts are shown for comparison only.
""")

code(r"""
for split in ("random", "chemical-system groups", "prototype groups"):
    inside, total = 0, 0
    for r in (r for r in results if r["task"].split == split):
        n, m = len(r["y_cal"]), len(r["y_test"])
        a, b = uq.coverage_beta(n, ALPHA)
        lo_ok, hi_ok = stats.betabinom(m, a, b).ppf([0.0005, 0.9995]) / m
        for name in CONFORMAL:
            inside += lo_ok <= r["covered"][name].mean() <= hi_ok
            total += 1
    if total:
        print(f"{split:24s} {inside:3d} of {total} fold-and-method coverages inside the 99.9 % beta-binomial range "
              f"({lo_ok:.3f}-{hi_ok:.3f} for n = {n}, m = {m})")
    if split == "random" and total:
        summary.append(("random split: fold coverages inside the 99.9 % beta-binomial range", f"{inside}/{total}",
                        f"{total}/{total}", "exchangeability (theory)"))
""")

md(r"""
## 5. Ranking: are the most uncertain predictions the most wrong?

For every split and method: the MAE of the prediction the uncertainty belongs to, the Spearman correlation between uncertainty and absolute error (over all test sets of the split pooled, and the median over the test sets, which leaves out differences between test sets, such as between elements), the share of the 10 % largest errors among the 10 % most uncertain predictions (10 % by chance), and the retention gain: how much of the possible improvement the method achieves when the most uncertain predictions are rejected (1: as good as rejecting by the true error, 0: no better than chance).
""")

code(r"""
rank_table = defaultdict(dict)
print(f"{'split':24s} {'method':20s} {'MAE':>7s} {'Spearman':>9s} {'(median per test set)':>22s} {'top-10 %':>9s} "
      f"{'retention gain':>15s}")
for split in SPLITS:
    if split not in cov_table:
        continue
    y = pooled(split, "y_test")
    for name, (mu_key, score, _) in METHODS.items():
        err = np.abs(pooled(split, f"test_{mu_key}") - y)
        u = pooled(split, f"test_{score}")
        per_task = [uq.spearman(r[f"test_{score}"], np.abs(r[f"test_{mu_key}"] - r["y_test"]))
                    for r in results if r["task"].split == split]
        row = {"mae": float(err.mean()), "spearman": uq.spearman(u, err), "spearman_task": float(np.median(per_task)),
               "top10": uq.top_overlap(u, err), "gain": uq.retention_gain(u, err, FRACTIONS),
               "retention": uq.retention_curve(u, err, FRACTIONS)}
        rank_table[split][name] = row
        print(f"{split:24s} {name:20s} {row['mae']:7.4f} {row['spearman']:9.3f} {row['spearman_task']:22.3f} "
              f"{row['top10']:9.0%} {row['gain']:15.3f}")
print("\nThe random forest ranks its errors best, but its predictions are less accurate than the ensemble's. "
      "Fraction of the predictions kept\nabove which the kept predictions of the forest have a larger MAE than those of the bootstrap ensemble:")
for split, t in rank_table.items():
    worse = t["random forest"]["retention"] > t["bootstrap ensemble"]["retention"]
    print(f"  {split:24s} {FRACTIONS[np.argmax(worse)]:.2f}" if worse.any() else f"  {split:24s} never")
""")

code(r"""
fig, axes = plt.subplots(1, len(rank_table), figsize=(3.3 * len(rank_table), 3.4), sharey=False)
for ax, split in zip(np.atleast_1d(axes), rank_table, strict=True):
    for name in METHODS:
        ax.plot(FRACTIONS, rank_table[split][name]["retention"], color=COLORS[name], lw=1.4, label=name)
    y = pooled(split, "y_test")
    err = np.abs(pooled(split, "test_mu_ens") - y)
    ax.plot(FRACTIONS, uq.retention_curve(err, err, FRACTIONS), color="k", lw=0.9, ls=":", label="oracle (ensemble)")
    ax.set_title(split, fontsize=9)
    ax.set_xlabel("fraction of predictions kept")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
np.atleast_1d(axes)[0].set_ylabel("MAE of the kept predictions (eV/atom)")
fig.legend(*np.atleast_1d(axes)[0].get_legend_handles_labels(), loc="lower center", ncol=7, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("figures/retention.png", bbox_inches="tight")
plt.show()
""")

md(r"""
### A confidently wrong structure type

A prototype split holds out whole structure types. The cell below takes the structure type with the largest error of the ensemble among those with at least 500 materials, and asks where each method ranks the uncertainty of its materials among all predictions of the prototype split.
""")

code(r"""
proto_rs = [r for r in results if r["task"].split == "prototype groups"]
if proto_rs:
    idx = np.concatenate([r["task"].test for r in proto_rs])
    y = np.concatenate([r["y_test"] for r in proto_rs])
    labels = np.array(d["proto"])[idx]
    err_ens = np.abs(np.concatenate([r["test_mu_ens"] for r in proto_rs]) - y)
    big = [g for g, c in Counter(labels).items() if c >= 500]
    worst = max(big, key=lambda g: err_ens[labels == g].mean())
    m = labels == worst
    formulas = Counter(d["records"][i]["formula"] for i in idx[m])
    icsd = np.mean([str(d["records"][i].get("icsd", "")).strip() not in ("", "na", "None") for i in idx[m]])
    print(f"structure type {worst}: {m.sum()} materials, all in one test fold: "
          f"{len({r['task'].key for r in proto_rs if np.isin(r['task'].test, idx[m]).any()}) == 1}")
    print(f"  formation energy {y[m].mean():+.3f} eV/atom on average (all materials {d['y'].mean():+.3f}); "
          f"with an ICSD id: {icsd:.1%}; e.g. {', '.join(f for f, _ in formulas.most_common(4))}")
    print(f"  {'method':20s} {'MAE on this type':>17s} {'median percentile of its uncertainty':>37s} {'share in the top 10 %':>22s}")
    for name, (mu_key, score, _) in METHODS.items():
        u = np.concatenate([r[f"test_{score}"] for r in proto_rs])
        pct = stats.rankdata(u)[m] / len(u)
        mae_g = np.abs(np.concatenate([r[f"test_{mu_key}"] for r in proto_rs]) - y)[m].mean()
        print(f"  {name:20s} {mae_g:17.3f} {np.median(pct):37.2f} {np.mean(pct > 0.9):22.0%}")
    print(f"  MAE of the ensemble on all other materials of the prototype split: {err_ens[~m].mean():.3f}")
""")

md(r"""
## 6. Calibration: does a stated standard deviation mean what it says?

For the four methods that give a Gaussian standard deviation: the fraction of test materials inside central Gaussian intervals of each probability level (a perfect method follows the diagonal), the miscalibration area between that curve and the diagonal (Tran et al. 2020), and the Gaussian negative log-likelihood.

The same numbers are then given after "std scaling" (Levi et al. 2022): every standard deviation of a test set is multiplied by the one factor that fits its calibration set best (maximum likelihood). This corrects a wrong overall scale, but not a wrong ranking. Palmer et al. (2022, npj Comput. Mater.) found that the bootstrap standard deviations of materials models need such a recalibration.
""")

code(r"""
gauss = [m for m, (_, _, g) in METHODS.items() if g]
levels = np.linspace(0, 1, 101)
cal_table = defaultdict(dict)
print(f"{'':45s} {'miscalibration area':>20s} {'NLL':>13s} {'inside the 90 % interval':>25s}  {'scale factor':>12s}")
print(f"{'split':24s} {'method':20s} {'raw':>9s} {'scaled':>10s} {'raw':>6s} {'scaled':>6s} {'raw':>12s} {'scaled':>12s}  "
      f"{'(median)':>12s}")
fig, axes = plt.subplots(1, len(rank_table), figsize=(3.3 * len(rank_table), 3.3))
for ax, split in zip(np.atleast_1d(axes), rank_table, strict=True):
    y = pooled(split, "y_test")
    rs = [r for r in results if r["task"].split == split]
    for name in gauss:
        mu_key, score, _ = METHODS[name]
        mu, sd = pooled(split, f"test_{mu_key}"), pooled(split, f"test_{score}")
        factors = [uq.std_scale(r["y_cal"], r[f"cal_{mu_key}"], r[f"cal_{score}"]) for r in rs]
        sd_s = np.concatenate([f * r[f"test_{score}"] for f, r in zip(factors, rs, strict=True)])
        curve = uq.calibration_curve(y, mu, sd, levels)
        row = {"area": uq.miscalibration_area(y, mu, sd), "nll": uq.gaussian_nll(y, mu, sd),
               "inside90": float(uq.calibration_curve(y, mu, sd, [0.9])[0]), "curve": curve,
               "area_scaled": uq.miscalibration_area(y, mu, sd_s), "nll_scaled": uq.gaussian_nll(y, mu, sd_s),
               "inside90_scaled": float(uq.calibration_curve(y, mu, sd_s, [0.9])[0]), "factor": float(np.median(factors))}
        cal_table[split][name] = row
        print(f"{split:24s} {name:20s} {row['area']:9.3f} {row['area_scaled']:10.3f} {row['nll']:6.2f} {row['nll_scaled']:6.2f} "
              f"{row['inside90']:12.1%} {row['inside90_scaled']:12.1%}  {row['factor']:12.2f}")
        ax.plot(levels, curve, color=COLORS[name], lw=1.4, label=name)
    ax.plot([0, 1], [0, 1], color="k", lw=0.8, ls="--")
    ax.set_title(split, fontsize=9)
    ax.set_xlabel("expected fraction inside")
np.atleast_1d(axes)[0].set_ylabel("observed fraction inside")
fig.legend(*np.atleast_1d(axes)[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("figures/calibration.png", bbox_inches="tight")
plt.show()
""")

md(r"""
### Error against stated uncertainty

The same question binned (Levi et al. 2022): the test materials are sorted by their stated standard deviation into ten equal bins, and the RMSE of each bin is compared with its root mean variance. On the diagonal, a stated σ is the error one gets; below it, the method is too cautious, above it too confident.
""")

code(r"""
fig, axes = plt.subplots(1, len(rank_table), figsize=(3.3 * len(rank_table), 3.3))
for ax, split in zip(np.atleast_1d(axes), rank_table, strict=True):
    y = pooled(split, "y_test")
    top = 0
    for name in gauss:
        mu_key, score, _ = METHODS[name]
        rmv, rmse = uq.rmse_vs_rmv(pooled(split, f"test_{mu_key}") - y, pooled(split, f"test_{score}"))
        ax.plot(rmv, rmse, "o-", ms=3, color=COLORS[name], lw=1.2, label=name)
        top = max(top, rmv.max(), rmse.max())
    ax.plot([0, top], [0, top], color="k", lw=0.8, ls="--")
    ax.set_title(split, fontsize=9)
    ax.set_xlabel("root mean variance in the bin (eV/atom)")
np.atleast_1d(axes)[0].set_ylabel("RMSE in the bin (eV/atom)")
fig.legend(*np.atleast_1d(axes)[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("figures/rmse_vs_rmv.png", bbox_inches="tight")
plt.show()
""")

md(r"""
## 7. Conformal intervals under distribution shift

The guarantee needs calibration and test materials to be exchangeable. That holds for the random split; it does not hold when the test set is a new structure type, an unseen element or a later batch of materials, and that is where an interval is most needed.
""")

code(r"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
ypos = np.arange(len(cov_table))[::-1]
markers = "osD^vP*"
width_table = defaultdict(dict)
for j, name in enumerate(CONFORMAL):
    covs = [cov_table[s][name] for s in cov_table]
    widths = []
    for s in cov_table:
        w = np.concatenate([r["width"][name] for r in results if r["task"].split == s])
        width_table[s][name] = float(np.median(w))
        widths.append(width_table[s][name])
    off = (j - 3) * 0.09
    ax1.plot(covs, ypos + off, markers[j], color=CONF_COLORS[name], ms=5, label=name)
    ax2.plot(widths, ypos + off, markers[j], color=CONF_COLORS[name], ms=5)
ax1.axvline(1 - ALPHA, color="k", lw=0.8, ls="--")
for ax in (ax1, ax2):
    ax.set_yticks(ypos, list(cov_table))
ax1.set_xlabel("coverage of the 90 % intervals")
ax2.set_xlabel("median interval width (eV/atom)")
ax2.set_yticklabels([])
fig.legend(*ax1.get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig("figures/conformal_coverage.png", bbox_inches="tight")
plt.show()
for s in cov_table:
    print(f"{s:24s} median width: " + ", ".join(f"{n.split(' (')[0]} {width_table[s][n]:.3f}" for n in CONFORMAL))
""")

md(r"""
### Per element

Leave one element out gives one test set per element. For each, the coverage of two conformal variants, against the MAE of the ensemble on that element.
""")

code(r"""
loeo = [r for r in results if r["task"].split == "leave one element out"]
if loeo:
    elem = [r["task"].key.split("-", 1)[1] for r in loeo]
    mae_e = np.array([np.mean(np.abs(r["test_mu_ens"] - r["y_test"])) for r in loeo])
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for name, mk in (("residual (constant width)", "o"), ("bootstrap ensemble", "s"), ("deep ensemble", "^")):
        c = np.array([r["covered"][name].mean() for r in loeo])
        ax.scatter(mae_e, c, s=16, marker=mk, color=CONF_COLORS[name], label=name)
        worst = np.argsort(c)[:3]
        for i in worst:
            ax.annotate(elem[i], (mae_e[i], c[i]), xytext=(3, 2), textcoords="offset points", fontsize=7)
    ax.axhline(1 - ALPHA, color="k", lw=0.8, ls="--")
    ax.set_xscale("log")
    ticks = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("MAE of the ensemble on the element's compounds (eV/atom)")
    ax.set_ylabel("coverage of the 90 % intervals")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/coverage_per_element.png", bbox_inches="tight")
    plt.show()
    for name in ("residual (constant width)", "bootstrap ensemble", "deep ensemble"):
        c = np.array([r["covered"][name].mean() for r in loeo])
        print(f"{name:26s} elements below 80 % coverage: {np.sum(c < 0.8):2d} of {len(c)}; "
              f"lowest: " + ", ".join(f"{elem[i]} {c[i]:.2f}" for i in np.argsort(c)[:6]))
""")

md(r"""
## 8. Comparison with other studies

No study has used these models on these data, so the numbers below are context, not checks: they show whether the results are of the usual size.
""")

code(r"""
lit = [
    ("Spearman, uncertainty vs error, formation energy (MP, random split)", "MEGNet ensemble, evidential, conformal",
     "0.12-0.27 (conformal)", "Varivoda et al. 2023, arXiv:2211.02235"),
    ("miscalibration area, ensemble, formation energy (MP, random split)", "MEGNet bootstrap ensemble", "0.078",
     "Varivoda et al. 2023"),
    ("Spearman, uncertainty vs error, formation energy, OOD splits (perovskites)", "12 GNNs, evidential + dropout",
     "0.12-0.22", "MatUQ, arXiv:2511.11697"),
]
if "random" in rank_table:
    rho = ", ".join(f"{m} {row['spearman']:.2f}" for m, row in rank_table["random"].items())
    area = ", ".join(f"{m} {row['area']:.3f}" for m, row in cal_table["random"].items())
    print(f"this run, random split: Spearman {rho}")
    print(f"this run, random split: miscalibration area {area}")
for row in lit:
    print(" | ".join(row))
""")

md(r"""
## 9. Summary and data for the interactive page
""")

code(r"""
export = {
    "about": {"alpha": ALPHA, "tasks": len(results), "quick_test_run": QUICK, "fractions": FRACTIONS.round(4).tolist()},
    "coverage": cov_table,
    "width": width_table,
    "ranking": {s: {m: {k: (round(v, 4) if isinstance(v, float) else np.round(v, 5).tolist()) for k, v in row.items()}
                    for m, row in t.items()} for s, t in rank_table.items()},
    "calibration": {s: {m: {**{k: round(row[k], 4) for k in ("area", "nll", "inside90", "area_scaled", "nll_scaled",
                                                             "inside90_scaled", "factor")},
                            "curve": np.round(row["curve"], 4).tolist()} for m, row in t.items()} for s, t in cal_table.items()},
    "elements": [{"element": r["task"].key.split("-", 1)[1], "n": len(r["y_test"]),
                  "mae": round(float(np.mean(np.abs(r["test_mu_ens"] - r["y_test"]))), 4),
                  "coverage": {n: round(float(r["covered"][n].mean()), 4) for n in CONFORMAL}} for r in loeo],
    "checks": [list(row) for row in summary],
}
with open("uncertainty_results.json", "w", encoding="utf-8") as f:
    json.dump(export, f, ensure_ascii=False, separators=(",", ":"))
w = [max(len(r[j]) for r in summary) for j in range(3)]
print(f"{'quantity':{w[0]}s}  {'this run':>{w[1]}s}  {'check':>{w[2]}s}  check from")
for q_, a_, b_, src in summary:
    print(f"{q_:{w[0]}s}  {a_:>{w[1]}s}  {b_:>{w[2]}s}  {src}")
""")

md(r"""
## Discussion

**Conformal intervals keep their promise where the theory says they should, and only there.** On the random split every one of the seven variants covers 89.9–90.2 % of the test materials, and all 70 fold coverages lie in the range the beta-binomial distribution allows. With chemical systems held out the coverage is the same (89.7–90.2 %), because calibration and test folds are drawn the same way. Holding out structure types lowers it slightly (88.7–89.8 %) and makes it vary from fold to fold far more than for independent materials (21 of 70 fold coverages inside that range): a structure type acts as one large, correlated unit. On the materials added to JARVIS after 2021 the coverage falls to 85.6–89.0 %, and with an element held out to 71–81 %, down to 14 % for the fluorine compounds with intervals of constant width. The guarantee rests on exchangeability, not on the model: a calibration set drawn like the training data says nothing about how wrong the model is on chemistry it has not seen.

**Scaling the intervals by an uncertainty helps twice, but does not close the gap.** At the same coverage the intervals become narrower where the model is reliable: on the random split the median width falls from 0.41 eV/atom (constant width) to 0.31–0.32 with the bootstrap or the deep ensemble, on the prototype split from 0.78 to 0.59. The Gaussian process gives the widest intervals (0.63–0.91), because its predictions are the least accurate. Under shift, scaled intervals also lose less coverage: with an element held out, intervals scaled by the spread of the random forest cover 81 % against 72 % for constant width, and on the prospective set quantile regression (CQR) and the Gaussian process reach 89 %.

**Ranking survives a shift of the test set, but not into an unseen element.** The spread of the random forest ranks the errors best in every split (Spearman 0.59–0.64, retention gain 0.73–0.78), followed by the deep ensemble (0.47–0.57) and the bootstrap ensemble (0.45–0.54); the distance to the training set (0.22–0.30) and the Gaussian process (0.11–0.21) are far behind. From the random split to the chemical-system, prototype and prospective splits these correlations change by at most 0.1. Leave one element out looks similar when all elements are pooled (0.21–0.61), but that mostly reflects differences between elements: within the compounds of one element, the median Spearman correlation is 0.43 for the forest, 0.40 for the deep ensemble, 0.27 for the bootstrap ensemble and 0.08 for the distance. The models can tell that fluorine compounds are hard; they can hardly tell which fluorine compounds.

**The best ranking does not come with the best prediction.** The forest's own predictions have a 20 % larger error than those of the gradient-boosting ensemble (0.106 against 0.088 eV/atom on the random split), and once more than 42–80 % of the predictions are kept (depending on the split), the forest's kept predictions are worse than the ensemble's. Whether the forest's spread can rank the ensemble's predictions was not tested.

**Stated standard deviations are rarely right as they come.** The bootstrap ensemble is overconfident: only 68 % of the test materials lie inside its 90 % Gaussian interval on the random split and 46 % with an element held out, and the factor that fits its calibration set best is 2.5 to 3.8, in line with Palmer et al. (2022), who found raw bootstrap standard deviations of materials models inaccurate. The forest is too cautious (98.5 % inside on the random split; factors 0.55–0.82). The Gaussian process looks well calibrated on average (miscalibration area 0.03 with an element held out), but its standard deviation hardly varies, and the binned RMSE rises steeply over that narrow range: it is dominated by the noise term and says little about the individual material, the limitation of average calibration that Levi et al. (2022) point out. The deep ensemble is the best calibrated without any correction (miscalibration area 0.013–0.065 on four of the five splits; a MEGNet ensemble on Materials Project: 0.078, Varivoda et al. 2023). After std scaling, the 90 % Gaussian intervals of all four methods contain 89–93 % of the test materials on every split except leave one element out, where they contain 75–84 %: a scale factor corrects a wrong scale, not a shift.

**Confidently wrong.** In the prototype split one structure type, 1402 ABC₄ compounds of space group F-43m on the Wyckoff positions of the MgCu₄Sn type (2.5 % with an ICSD entry; mean formation energy +0.58 eV/atom against −0.83 for the whole snapshot), is predicted with errors of 0.48–0.65 eV/atom by every model, 4.5 times the error of the ensemble on the rest. The forest ranks its uncertainty high (83rd percentile), the distance and the bootstrap ensemble above the median, but the deep ensemble and quantile regression put it at the 22nd and 25th percentile: more confident than for a typical material. An uncertainty learned from the training data cannot know that an unseen structure type behaves differently.

**In practice** (for these features and models): to rank predictions by reliability, the spread of a random forest or of a deep ensemble works under every shift tested here, though much less within an unseen element; for intervals with a stated coverage, conformal prediction with scores scaled by an ensemble's standard deviation gives the narrowest valid intervals, but only for test data drawn like the calibration data. For new chemistry the per-element results show where intervals fail (F, O, N, H). Calibrating on held-out chemistry, for example on held-out elements, might address that; it was not tested here.
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
