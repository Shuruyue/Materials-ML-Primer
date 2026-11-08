"""Tests of materials_ml_primer.uncertainty against closed forms and exact statistical results."""

import math

import numpy as np
import pytest
from scipy import integrate, stats

from materials_ml_primer import uncertainty as uq

# ------------------------------------------------------------------------------------------------ conformal


def test_conformal_quantile_rank():
    s = np.arange(1, 10)                                   # n = 9
    assert uq.conformal_quantile(s, 0.1) == 9               # ceil(10 * 0.9) = 9th smallest
    assert uq.conformal_quantile(s, 0.5) == 5               # ceil(10 * 0.5) = 5
    assert uq.conformal_quantile(s, 0.05) == math.inf       # ceil(10 * 0.95) = 10 > n
    assert uq.conformal_quantile(s[::-1], 0.2) == 8         # order of the input does not matter


def test_coverage_beta_parameters():
    assert uq.coverage_beta(1000, 0.1) == (901, 100)
    a, b = uq.coverage_beta(1000, 0.1)
    assert a / (a + b) == pytest.approx(0.9 * 1001 / 1001, abs=1e-3)


@pytest.mark.parametrize("n, alpha", [(50, 0.1), (200, 0.2)])
def test_conditional_coverage_is_beta_distributed(n, alpha):
    # residuals |e| with e ~ N(0, 1): given the calibration set, the coverage is exactly 2 Phi(q) - 1
    rng = np.random.default_rng(1)
    cover = np.array([2 * stats.norm.cdf(uq.conformal_quantile(np.abs(rng.normal(size=n)), alpha)) - 1
                      for _ in range(4000)])
    a, b = uq.coverage_beta(n, alpha)
    assert stats.kstest(cover, stats.beta(a, b).cdf).pvalue > 0.01
    assert 1 - alpha - 3e-3 <= cover.mean() <= 1 - alpha + 1 / (n + 1) + 3e-3


def test_normalized_and_cqr_intervals_cover_heteroscedastic_data():
    rng = np.random.default_rng(2)

    def sample(m):
        x = rng.uniform(-2, 2, m)
        s = 0.2 + np.abs(x)
        return x, x + s * rng.normal(size=m), s

    x_cal, y_cal, s_cal = sample(2000)
    x_te, y_te, s_te = sample(200000)
    # given the calibration set the coverage is Beta distributed (sd 0.0067 here); allow its 0.1-99.9 % range
    beta = stats.beta(*uq.coverage_beta(2000, 0.1))
    lo_ok, hi_ok = beta.ppf(0.001) - 0.003, beta.ppf(0.999) + 0.003
    # normalized residuals with the true scale, and with a scale that is wrong by a constant factor
    for scale in (1.0, 3.0):
        lo, hi = uq.conformal_interval(x_te, uq.residual_scores(y_cal, x_cal, scale * s_cal), 0.1, scale=scale * s_te)
        assert lo_ok < uq.interval_coverage(y_te, lo, hi) < hi_ok
    # CQR with deliberately too narrow quantile bands: the conformal step restores the coverage
    lo_c, hi_c = x_cal - 0.5 * s_cal, x_cal + 0.5 * s_cal
    lo, hi = uq.cqr_interval(x_te - 0.5 * s_te, x_te + 0.5 * s_te, uq.cqr_scores(y_cal, lo_c, hi_c), 0.1)
    assert lo_ok < uq.interval_coverage(y_te, lo, hi) < hi_ok
    # plain residuals: valid marginally, but much wider where the noise is small
    lo, hi = uq.conformal_interval(x_te, uq.residual_scores(y_cal, x_cal), 0.1)
    assert lo_ok < uq.interval_coverage(y_te, lo, hi) < hi_ok


# ------------------------------------------------------------------------------------------------ metrics


def test_miscalibration_area_closed_form():
    rng = np.random.default_rng(3)
    e = rng.normal(size=400000)
    assert uq.miscalibration_area(e, 0, np.ones_like(e)) < 0.003
    # sigma underestimated by a factor k: observed fraction 2 Phi(z_p k) - 1 at level p, z_p = Phi^-1(1/2 + p/2)
    for k in (2.0, 0.5):
        exact = integrate.quad(lambda p, k=k: abs(2 * stats.norm.cdf(k * stats.norm.ppf(0.5 + p / 2)) - 1 - p),
                               0, 1, limit=200)[0]
        assert uq.miscalibration_area(e, 0, np.ones_like(e) / k, n_levels=2001) == pytest.approx(exact, abs=3e-3)


def test_gaussian_nll_matches_scipy():
    rng = np.random.default_rng(4)
    y, mu, s = rng.normal(size=50), rng.normal(size=50), rng.uniform(0.1, 2, 50)
    assert uq.gaussian_nll(y, mu, s) == pytest.approx(-np.mean(stats.norm.logpdf(y, mu, s)), rel=1e-12)


def test_rmse_equals_rmv_for_calibrated_gaussians():
    rng = np.random.default_rng(5)
    s = rng.uniform(0.1, 2.0, 200000)
    rmv, rmse = uq.rmse_vs_rmv(s * rng.normal(size=s.size), s, n_bins=10)
    assert np.allclose(rmse / rmv, 1, atol=0.02)
    assert np.all(np.diff(rmv) > 0)


def test_std_scale_recovers_a_wrong_scale_and_minimizes_the_nll():
    rng = np.random.default_rng(11)
    s = rng.uniform(0.1, 2.0, 100000)
    e = s * rng.normal(size=s.size)
    assert uq.std_scale(e, 0, s) == pytest.approx(1, abs=0.01)
    for k in (0.3, 4.0):                       # stated sigma wrong by a factor k
        f = uq.std_scale(e, 0, k * s)
        assert f * k == pytest.approx(1, abs=0.01)
        nll = uq.gaussian_nll(e, 0, f * k * s)
        assert nll <= min(uq.gaussian_nll(e, 0, g * f * k * s) for g in (0.95, 1.05))
        assert uq.miscalibration_area(e, 0, f * k * s) < 0.01


def test_mixture_moments():
    mu, var = uq.mixture_moments([[-1.0], [1.0]], [[1.0], [1.0]])
    assert mu[0] == 0 and var[0] == pytest.approx(2.0)
    rng = np.random.default_rng(6)
    means, variances = rng.normal(size=(5, 3)), rng.uniform(0.1, 1, (5, 3))
    draws = rng.normal(means[:, :, None], np.sqrt(variances)[:, :, None], (5, 3, 200000))
    draws = draws.transpose(1, 0, 2).reshape(3, -1)
    mu, var = uq.mixture_moments(means, variances)
    assert np.allclose(draws.mean(axis=1), mu, atol=0.01)
    assert np.allclose(draws.var(axis=1), var, rtol=0.01)


def test_rankings_and_retention():
    rng = np.random.default_rng(7)
    err = rng.exponential(size=5000)
    fr = np.linspace(0.05, 1, 20)
    assert uq.top_overlap(err, err) == 1 and uq.top_overlap(-err, err) == 0
    assert uq.spearman(err, err) == pytest.approx(1)
    oracle = uq.retention_curve(err, err, fr)
    assert np.all(np.diff(oracle) >= 0) and oracle[-1] == pytest.approx(err.mean())
    assert uq.retention_gain(err, err, fr) == pytest.approx(1)
    assert uq.retention_gain(-err, err, fr) < -0.5
    gains = [uq.retention_gain(rng.normal(size=err.size), err, fr) for _ in range(20)]
    assert abs(np.mean(gains)) < 0.02
    assert uq.retention_curve(np.zeros(4), [1, 2, 3, 4], [0.5])[0] == 1.5   # ties keep the input order


def test_interval_coverage_and_scores():
    assert uq.interval_coverage([0, 1, 2, 3], [0, 0, 0, 0], [1, 1, 1, 1]) == 0.5
    assert np.allclose(uq.cqr_scores([0, 2, 5], [1, 1, 1], [3, 3, 3]), [1, -1, 2])
    assert np.allclose(uq.residual_scores([1, 4], [0, 0], [1, 2]), [1, 2])


# ------------------------------------------------------------------------------------------------ models


def test_bootstrap_ensemble_quantiles_and_forest():
    rng = np.random.default_rng(8)
    X = rng.uniform(-2, 2, (1500, 3))
    y = X[:, 0] + (0.1 + np.abs(X[:, 0])) * rng.normal(size=1500)
    tr, va, te = slice(0, 1000), slice(1000, 1250), slice(1250, None)
    models = uq.fit_bootstrap_ensemble(X[tr], y[tr], X[va], y[va], n_members=3, max_iter=200)
    mu, sd = uq.ensemble_predict(models, X[te])
    assert sd.min() > 0 and np.mean(np.abs(mu - X[te, 0])) < 0.5
    lo_m, hi_m = uq.fit_quantile_pair(X[tr], y[tr], X[va], y[va], max_iter=200)
    lo, hi = lo_m.predict(X[te]), hi_m.predict(X[te])
    assert np.mean(lo < hi) > 0.95
    assert uq.spearman(hi - lo, np.abs(X[te, 0])) > 0.5          # wider where the noise is larger
    forest = uq.fit_random_forest(X[tr], y[tr], n_estimators=50)
    mu, sd = uq.forest_predict(forest, X[te])
    assert np.allclose(mu, forest.predict(X[te])) and sd.min() >= 0


def test_standardizer():
    X = np.array([[1.0, np.nan], [3.0, 2.0], [5.0, 4.0]])
    Z = uq.standardizer(X)(X)
    assert Z.dtype == np.float32 and not np.isnan(Z).any()
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-6) and np.allclose(Z[:, 0], [-1.2247449, 0, 1.2247449])


def _hetero(rng, m):
    x = rng.uniform(-3, 3, (m, 1)).astype(np.float32)
    s = 0.05 + 0.3 * np.abs(x[:, 0])
    return x, np.sin(x[:, 0]) + s * rng.normal(size=m), s


def test_svgp_is_accurate_and_roughly_calibrated_on_a_smooth_function():
    pytest.importorskip("gpytorch")
    rng = np.random.default_rng(9)
    Z, y, _ = _hetero(rng, 800)
    Zv, yv, _ = _hetero(rng, 200)
    Zt, yt, _ = _hetero(rng, 4000)
    predict = uq.fit_svgp(Z, y, Zv, yv, n_inducing=64, epochs=60, batch=256, lr=0.05, device="cpu")
    mu, sd = predict(Zt)
    assert np.mean(np.abs(mu - np.sin(Zt[:, 0]))) < 0.15
    assert 0.8 < uq.interval_coverage(yt, mu - 1.645 * sd, mu + 1.645 * sd) < 0.97   # homoscedastic GP noise


def test_deep_ensemble_learns_the_noise_level():
    pytest.importorskip("torch")
    rng = np.random.default_rng(10)
    Z, y, _ = _hetero(rng, 3000)
    Zv, yv, _ = _hetero(rng, 500)
    Zt, yt, s = _hetero(rng, 3000)
    predict = uq.fit_deep_ensemble(Z, y, Zv, yv, n_members=2, hidden=64, max_epochs=150, patience=15, device="cpu")
    mu, sd = predict(Zt)
    assert uq.spearman(sd, s) > 0.8
    assert uq.interval_coverage(yt, mu - 1.645 * sd, mu + 1.645 * sd) == pytest.approx(0.9, abs=0.05)
