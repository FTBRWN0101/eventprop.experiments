"""The Harvey-Leybourne-Newbold small-sample correction to the DM test.

Hand-computed values, so a typo in the formula fails here rather than in a thesis table.
HLN (1997) eq. (9): factor = [(n + 1 - 2h + h(h-1)/n) / n]^(1/2), h = lag + 1.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from core import metrics


#(n, lag, factor) computed independently at 30 decimal places from HLN eq. (9)
HAND_COMPUTED = [
    #n=302, h=1: (302 + 1 - 2 + 0)/302 = 301/302 = 0.996688741721854...
    (302, 0, 0.998342998033168),
    #n=302, h=5: (302 + 1 - 10 + 20/302)/302 = 0.970417964124381...
    (302, 4, 0.985097946462371),
    #n=72, h=21: (72 + 1 - 42 + 420/72)/72 = 0.511574074074074... a 28.5% shrink,
    #which is why the monthly horizon needed this and the weekly barely notices
    (72, 20, 0.715244066087985),
    #the daily (stride 1) view, for contrast
    (1508, 4, 0.997015859987173),
]


@pytest.mark.parametrize("n,lag,expected", HAND_COMPUTED)
def test_hln_factor_matches_hand_computed_values(n, lag, expected):
    assert metrics.hln_factor(n, lag) == pytest.approx(expected, abs=1e-12)


def test_hln_factor_matches_the_formula_written_out_longhand():
    #the same expression built from scratch, so a refactor of hln_factor is checked
    #against the paper rather than against itself
    for n, lag in ((302, 0), (302, 4), (72, 20), (1508, 4)):
        h = lag + 1
        expected = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        assert metrics.hln_factor(n, lag) == pytest.approx(expected, abs=1e-14)


def test_hln_factor_is_below_one_for_every_sensible_horizon():
    #with h < n the factor shrinks the statistic, i.e. the correction is conservative
    for n in (10, 50, 302, 1508):
        for lag in range(0, min(n - 2, 40)):
            assert 0.0 < metrics.hln_factor(n, lag) < 1.0, (n, lag)


def test_hln_factor_shrinks_harder_as_the_lag_grows():
    factors = [metrics.hln_factor(72, lag) for lag in range(0, 21)]
    assert all(b < a for a, b in zip(factors, factors[1:]))


def test_hln_factor_is_nan_when_the_horizon_reaches_the_sample_size():
    #at h == n the bracket is exactly zero for every n: n + 1 - 2n + n(n-1)/n = 0
    for n in (4, 10, 72):
        assert np.isnan(metrics.hln_factor(n, n - 1))


def _series(values):
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values)))


def test_dm_statistic_is_the_uncorrected_one_times_the_factor():
    rng = np.random.default_rng(7)
    a = _series(rng.normal(size=120) ** 2 + 0.4)
    b = _series(rng.normal(size=120) ** 2)
    for lag in (0, 4):
        plain, _ = metrics.diebold_mariano(a, b, lag=lag, hln=False)
        corrected, _ = metrics.diebold_mariano(a, b, lag=lag, hln=True)
        assert corrected == pytest.approx(plain * metrics.hln_factor(120, lag))


def test_hln_p_value_uses_the_t_distribution_not_the_normal():
    rng = np.random.default_rng(11)
    a = _series(rng.normal(size=40) ** 2 + 0.5)
    b = _series(rng.normal(size=40) ** 2)
    stat, p = metrics.diebold_mariano(a, b, lag=0, hln=True)
    assert p == pytest.approx(2.0 * stats.t.cdf(-abs(stat), df=39))
    #and it is strictly more conservative than the normal at the same statistic
    assert p > 2.0 * (1.0 - stats.norm.cdf(abs(stat)))


def test_hln_never_rejects_where_the_normal_convention_would_not():
    #both effects go the same way: the statistic shrinks and the tails fatten
    rng = np.random.default_rng(3)
    a = _series(rng.normal(size=72) ** 2 + 0.3)
    b = _series(rng.normal(size=72) ** 2)
    _, p_old = metrics.diebold_mariano(a, b, lag=20, hln=False)
    _, p_new = metrics.diebold_mariano(a, b, lag=20, hln=True)
    assert p_new > p_old


def test_hln_default_is_on():
    rng = np.random.default_rng(5)
    a = _series(rng.normal(size=60) ** 2 + 0.2)
    b = _series(rng.normal(size=60) ** 2)
    assert metrics.diebold_mariano(a, b, lag=4) == metrics.diebold_mariano(
        a, b, lag=4, hln=True)


def test_dm_still_degrades_gracefully_on_identical_errors():
    a = _series(np.ones(10))
    stat, p = metrics.diebold_mariano(a, a.copy(), lag=0)
    assert np.isnan(stat) and np.isnan(p)


def test_dm_returns_nan_when_the_factor_is_undefined():
    #n=4 with lag 3 makes h == n, where HLN eq. (9) collapses to zero
    rng = np.random.default_rng(1)
    a = _series(rng.normal(size=4) ** 2 + 1.0)
    b = _series(rng.normal(size=4) ** 2)
    stat, p = metrics.diebold_mariano(a, b, lag=3, hln=True)
    assert np.isnan(stat) and np.isnan(p)
    #the uncorrected convention still produces a number there
    assert np.isfinite(metrics.diebold_mariano(a, b, lag=3, hln=False)[0])
