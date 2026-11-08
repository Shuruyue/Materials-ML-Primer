"""Uncertainty of regression predictions: models that report it, conformal intervals and ways to evaluate it.

Three parts:

- **Conformal prediction** (split conformal): turn any uncertainty score into intervals with a finite-sample
  coverage guarantee under exchangeability (Vovk et al. 2005; Angelopoulos & Bates, arXiv:2107.07511), including
  conformalized quantile regression (Romano et al. 2019).
- **Metrics**: rank correlation and top-fraction overlap between uncertainty and error, retention curves, the
  miscalibration area of Gaussian uncertainties (Tran et al. 2020), the Gaussian negative log-likelihood, binned
  RMSE against root mean variance and the scale factor of "std scaling" (Levi et al. 2022), and interval
  coverage.
- **Models** that return a prediction and an uncertainty: a bootstrap ensemble of gradient-boosting models,
  gradient-boosting quantile regression, the spread of the trees of a random forest, a sparse variational
  Gaussian process (GPyTorch) and a deep ensemble of mean-variance networks (Lakshminarayanan et al. 2017).
  PyTorch and GPyTorch are imported only by the functions that need them.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

# ------------------------------------------------------------------------------------------------ conformal


def conformal_quantile(scores, alpha):
    """The ceil((n + 1)(1 - alpha))-th smallest of n calibration scores (inf if that rank exceeds n).

    With exchangeable calibration and test points, a test score falls at or below it with probability at least
    1 - alpha, and at most 1 - alpha + 1/(n + 1) for continuous scores (Angelopoulos & Bates, Theorem D.1-D.2).
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        raise ValueError("no calibration scores")
    k = math.ceil((n + 1) * (1 - alpha))
    return math.inf if k > n else float(s[k - 1])


def coverage_beta(n, alpha):
    """Parameters (a, b) of the Beta distribution of the coverage conditional on n calibration points.

    The coverage of a split-conformal interval, given the calibration set and evaluated on infinitely many test
    points, is Beta(n + 1 - l, l) distributed with l = floor((n + 1) alpha) (Vovk 2012).
    """
    ell = math.floor((n + 1) * alpha)
    return n + 1 - ell, ell


def residual_scores(y, pred, scale=None):
    """Nonconformity scores |y - pred| / scale (scale = 1: plain residuals; scale = sigma: normalized)."""
    r = np.abs(np.asarray(y, float) - np.asarray(pred, float))
    return r if scale is None else r / np.asarray(scale, float)


def conformal_interval(pred, scores_cal, alpha, scale=None):
    """Symmetric interval pred +- q scale, q the conformal quantile of the calibration scores."""
    q = conformal_quantile(scores_cal, alpha)
    half = q if scale is None else q * np.asarray(scale, float)
    return np.asarray(pred, float) - half, np.asarray(pred, float) + half


def cqr_scores(y, lo, hi):
    """Scores of conformalized quantile regression: max(lo - y, y - hi) (negative inside the band)."""
    y = np.asarray(y, float)
    return np.maximum(np.asarray(lo, float) - y, y - np.asarray(hi, float))


def cqr_interval(lo, hi, scores_cal, alpha):
    """The quantile band widened (or narrowed) by the conformal quantile of the CQR scores."""
    q = conformal_quantile(scores_cal, alpha)
    return np.asarray(lo, float) - q, np.asarray(hi, float) + q


# ------------------------------------------------------------------------------------------------ metrics


def interval_coverage(y, lo, hi):
    """Fraction of y inside [lo, hi]."""
    y = np.asarray(y, float)
    return float(np.mean((y >= lo) & (y <= hi)))


def spearman(u, err):
    """Spearman rank correlation between an uncertainty score and the absolute error."""
    return float(stats.spearmanr(u, err).statistic)


def top_overlap(u, err, q=0.1):
    """Share of the q largest errors that are among the q most uncertain predictions (q by chance)."""
    u, err = np.asarray(u), np.asarray(err)
    k = max(1, int(q * len(err)))
    return len(set(np.argsort(-u, kind="stable")[:k]) & set(np.argsort(-err, kind="stable")[:k])) / k


def retention_curve(u, err, fractions):
    """MAE of the predictions kept when only the given fraction with the smallest uncertainty is retained."""
    u, err = np.asarray(u, float), np.abs(np.asarray(err, float))
    order = np.argsort(u, kind="stable")
    cum = np.cumsum(err[order])
    out = []
    for f in fractions:
        k = max(1, int(round(f * len(err))))
        out.append(cum[k - 1] / k)
    return np.array(out)


def retention_gain(u, err, fractions):
    """How much of the possible improvement the uncertainty achieves when rejecting predictions.

    The area between the retention curve of a random ordering (constant MAE) and that of u, divided by the same
    area for the oracle ordering (by the error itself): 1 for a perfect ranking, 0 for a random one.
    """
    err = np.abs(np.asarray(err, float))
    curve = retention_curve(u, err, fractions)
    oracle = retention_curve(err, err, fractions)
    base = err.mean()
    best = np.trapezoid(base - oracle, fractions)
    return float(np.trapezoid(base - curve, fractions) / best) if best > 0 else math.nan


def calibration_curve(y, mu, sigma, levels):
    """Observed fraction of |y - mu| <= z sigma for central Gaussian intervals of the given probability levels."""
    z = stats.norm.ppf(0.5 + np.asarray(levels, float) / 2)
    r = np.abs(np.asarray(y, float) - np.asarray(mu, float)) / np.asarray(sigma, float)
    return np.array([np.mean(r <= zi) for zi in z])


def miscalibration_area(y, mu, sigma, n_levels=101):
    """Area between the observed and the expected fraction of points inside central Gaussian intervals.

    0 for a perfectly calibrated Gaussian uncertainty, up to 0.5 (Tran et al. 2020; Varivoda et al. 2023).
    """
    levels = np.linspace(0, 1, n_levels)
    observed = calibration_curve(y, mu, sigma, levels)
    return float(np.trapezoid(np.abs(observed - levels), levels))


def gaussian_nll(y, mu, sigma):
    """Mean negative log-likelihood of y under N(mu, sigma^2)."""
    y, mu, sigma = (np.asarray(a, float) for a in (y, mu, sigma))
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma**2) + (y - mu) ** 2 / (2 * sigma**2)))


def rmse_vs_rmv(err, sigma, n_bins=10):
    """Root mean variance and RMSE in bins of equal size sorted by sigma (Levi et al. 2022)."""
    err, sigma = np.asarray(err, float), np.asarray(sigma, float)
    bins = np.array_split(np.argsort(sigma, kind="stable"), n_bins)
    rmv = np.array([np.sqrt(np.mean(sigma[b] ** 2)) for b in bins])
    rmse = np.array([np.sqrt(np.mean(err[b] ** 2)) for b in bins])
    return rmv, rmse


def std_scale(y, mu, sigma):
    """The factor s that minimizes the Gaussian NLL of y under N(mu, (s sigma)^2): sqrt(mean(((y - mu)/sigma)^2)).

    Fitted on a calibration set, it corrects the overall scale of a standard deviation ("std scaling", Levi et al.
    2022) but not its ranking.
    """
    z = (np.asarray(y, float) - np.asarray(mu, float)) / np.asarray(sigma, float)
    return float(np.sqrt(np.mean(z**2)))


def mixture_moments(means, variances):
    """Mean and variance of an equal-weight mixture of Gaussians (members along axis 0)."""
    means, variances = np.asarray(means, float), np.asarray(variances, float)
    mu = means.mean(axis=0)
    return mu, (variances + means**2).mean(axis=0) - mu**2


# ------------------------------------------------------------------------------------------------ models


REPEATED_FITS = []     # fits repeated after the thread-pool error below (the results do not change)


def fit_repeating(model, *args, attempts=3, **kwargs):
    """model.fit(*args, **kwargs), repeated if joblib's thread pool fails on Windows.

    scikit-learn's gradient boosting (feature binning) and random forest run joblib thread pools; on Windows,
    shutting such a pool down occasionally raises "concurrent send_bytes() calls are not supported", a race in
    Python's multiprocessing. These fits are deterministic, so repeating one gives the same model.
    """
    for attempt in range(attempts):
        try:
            return model.fit(*args, **kwargs)
        except ValueError as err:
            if "concurrent send_bytes" not in str(err) or attempt == attempts - 1:
                raise
            REPEATED_FITS.append(type(model).__name__)
    raise AssertionError("unreachable")


def hgb(learning_rate=0.1, max_iter=2500, seed=0, **kwargs):
    """Gradient boosting with the tree settings of the baselines experiment and early stopping."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(learning_rate=learning_rate, max_iter=max_iter, max_leaf_nodes=63,
                                         early_stopping=True, n_iter_no_change=100, random_state=seed, **kwargs)


def fit_bootstrap_ensemble(X, y, X_val, y_val, n_members=5, seed=0, **kwargs):
    """Gradient-boosting models, each trained on a bootstrap resample of (X, y) and early-stopped on X_val."""
    rng = np.random.default_rng(seed)
    models = []
    for m in range(n_members):
        idx = rng.integers(0, len(y), len(y))
        models.append(fit_repeating(hgb(seed=seed + m, **kwargs), X[idx], y[idx], X_val=X_val, y_val=y_val))
    return models


def ensemble_predict(models, X):
    """Mean and standard deviation (ddof = 1) of the members' predictions."""
    preds = np.array([m.predict(X) for m in models])
    return preds.mean(axis=0), preds.std(axis=0, ddof=1)


def fit_quantile_pair(X, y, X_val, y_val, alpha=0.1, **kwargs):
    """Gradient-boosting regressors of the alpha/2 and 1 - alpha/2 quantiles (pinball loss)."""
    return tuple(fit_repeating(hgb(loss="quantile", quantile=q, **kwargs), X, y, X_val=X_val, y_val=y_val)
                 for q in (alpha / 2, 1 - alpha / 2))


def fit_random_forest(X, y, n_estimators=300, seed=0):
    """The random forest of the baselines experiment (one third of the features per split)."""
    from sklearn.ensemble import RandomForestRegressor

    forest = RandomForestRegressor(n_estimators=n_estimators, max_features=1 / 3, n_jobs=-1, random_state=seed)
    return fit_repeating(forest, X, y)


def forest_predict(forest, X):
    """Mean and standard deviation of the predictions of the individual trees."""
    preds = np.array([t.predict(X) for t in forest.estimators_])
    return preds.mean(axis=0), preds.std(axis=0)


def standardizer(X_train):
    """A function that imputes the training median and standardizes with the training mean and std (float32)."""
    med = np.nanmedian(X_train, axis=0)
    filled = np.where(np.isnan(X_train), med, X_train)
    mean, std = filled.mean(axis=0), filled.std(axis=0)
    std = np.where(std > 0, std, 1.0)

    def transform(X):
        return ((np.where(np.isnan(X), med, X) - mean) / std).astype(np.float32)

    return transform


def _torch_device(device=None):
    import torch

    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def fit_svgp(Z, y, Z_val, y_val, n_inducing=1024, epochs=20, batch=1024, lr=0.01, seed=0, device=None,
             jitter=1e-4):
    """Sparse variational GP (Hensman et al. 2013) with an ARD RBF kernel; returns predict(Z) -> (mean, std).

    Inputs are standardized features; the target is standardized internally. The inducing points start at
    distinct training feature vectors (two identical ones make the kernel matrix singular), and `jitter` is
    added to its diagonal before the Cholesky factorization (float32). The epoch with the lowest validation
    negative log-likelihood is kept. The predictive std includes the Gaussian noise.
    """
    import gpytorch
    import torch

    torch.manual_seed(seed)
    dev = _torch_device(device)
    y_mean, y_std = float(np.mean(y)), float(np.std(y))
    Zt = torch.tensor(Z, device=dev)
    yt = torch.tensor((y - y_mean) / y_std, dtype=torch.float32, device=dev)
    Zv = torch.tensor(Z_val, device=dev)
    yv = torch.tensor((y_val - y_mean) / y_std, dtype=torch.float32, device=dev)

    class SVGP(gpytorch.models.ApproximateGP):
        def __init__(self, inducing):
            dist = gpytorch.variational.CholeskyVariationalDistribution(inducing.size(0))
            strategy = gpytorch.variational.VariationalStrategy(self, inducing, dist, learn_inducing_locations=True)
            super().__init__(strategy)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=Z.shape[1]))

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

    g = torch.Generator(device="cpu").manual_seed(seed)
    distinct = np.sort(np.unique(Z, axis=0, return_index=True)[1])
    pick = np.random.default_rng(seed).choice(distinct, size=min(n_inducing, len(distinct)), replace=False)
    model = SVGP(torch.tensor(Z[pick], device=dev)).to(dev)
    lik = gpytorch.likelihoods.GaussianLikelihood().to(dev)
    elbo = gpytorch.mlls.VariationalELBO(lik, model, num_data=len(Zt))
    opt = torch.optim.Adam([*model.parameters(), *lik.parameters()], lr=lr)

    def predict_std(Zq):
        model.eval()
        lik.eval()
        means, stds = [], []
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(
                float_value=jitter, double_value=1e-8), gpytorch.settings.cholesky_max_tries(6):
            for i in range(0, len(Zq), 4096):
                d = lik(model(Zq[i:i + 4096]))
                means.append(d.mean)
                stds.append(d.variance.sqrt())
        return torch.cat(means), torch.cat(stds)

    stable = (gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=1e-8),
              gpytorch.settings.cholesky_max_tries(6))
    best, best_state = math.inf, None
    for _ in range(epochs):
        model.train()
        lik.train()
        perm = torch.randperm(len(Zt), generator=g).to(dev)
        for i in range(0, len(Zt), batch):
            idx = perm[i:i + batch]
            with stable[0], stable[1]:
                loss = -elbo(model(Zt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        m, s = predict_std(Zv)
        nll = float(torch.mean(0.5 * torch.log(2 * math.pi * s**2) + (yv - m) ** 2 / (2 * s**2)))
        if nll < best:
            best = nll
            best_state = ({k: v.clone() for k, v in model.state_dict().items()},
                          {k: v.clone() for k, v in lik.state_dict().items()})
    model.load_state_dict(best_state[0])
    lik.load_state_dict(best_state[1])

    def predict(Zq):
        m, s = predict_std(torch.tensor(Zq, device=dev))
        return m.cpu().numpy() * y_std + y_mean, s.cpu().numpy() * y_std

    return predict


def fit_deep_ensemble(Z, y, Z_val, y_val, n_members=5, hidden=256, max_epochs=200, patience=20, batch=512,
                      lr=1e-3, seed=0, device=None):
    """Mean-variance networks trained with the Gaussian negative log-likelihood; returns predict(Z) -> (mean, std).

    Each member has two hidden layers of `hidden` SiLU units, its own random initialization and data order, and
    early stopping on the validation negative log-likelihood. The ensemble is the equal-weight Gaussian mixture.
    """
    import torch

    dev = _torch_device(device)
    y_mean, y_std = float(np.mean(y)), float(np.std(y))
    Zt = torch.tensor(Z, device=dev)
    yt = torch.tensor((y - y_mean) / y_std, dtype=torch.float32, device=dev)
    Zv = torch.tensor(Z_val, device=dev)
    yv = torch.tensor((y_val - y_mean) / y_std, dtype=torch.float32, device=dev)
    nll = torch.nn.functional.gaussian_nll_loss
    members = []
    for m in range(n_members):
        torch.manual_seed(seed + m)
        g = torch.Generator(device="cpu").manual_seed(seed + m)
        net = torch.nn.Sequential(torch.nn.Linear(Z.shape[1], hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden),
                                  torch.nn.SiLU(), torch.nn.Linear(hidden, 2)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=lr)

        def heads(out):
            return out[:, 0], torch.nn.functional.softplus(out[:, 1]) + 1e-6

        best, best_state, wait = math.inf, None, 0
        for _ in range(max_epochs):
            net.train()
            perm = torch.randperm(len(Zt), generator=g).to(dev)
            for i in range(0, len(Zt), batch):
                idx = perm[i:i + batch]
                mean, var = heads(net(Zt[idx]))
                loss = nll(mean, yt[idx], var)
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                mean, var = heads(net(Zv))
                val = float(nll(mean, yv, var))
            if val < best:
                best, wait = val, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
        net.load_state_dict(best_state)
        members.append((net, heads))

    def predict(Zq):
        Zq = torch.tensor(Zq, device=dev)
        means, variances = [], []
        with torch.no_grad():
            for net, heads in members:
                net.eval()
                mean, var = heads(net(Zq))
                means.append(mean.cpu().numpy())
                variances.append(var.cpu().numpy())
        mu, var = mixture_moments(np.array(means, dtype=float), np.array(variances, dtype=float))
        return mu * y_std + y_mean, np.sqrt(var) * y_std

    return predict
