import numpy as np
import pandas as pd

from core import metrics


def test_dm_aligns_on_common_dates():
    idx_a = pd.date_range("2020-01-01", periods=10)
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=10) ** 2, index=idx_a)
    b = pd.Series(rng.normal(size=8) ** 2, index=idx_a[2:])
    stat, p = metrics.diebold_mariano(a, b, lag=0)
    assert np.isfinite(stat) and 0.0 <= p <= 1.0


def test_dm_sign_favours_smaller_loss():
    idx = pd.date_range("2020-01-01", periods=50)
    base = np.random.default_rng(0).normal(size=50) ** 2
    a = pd.Series(base + 1.0, index=idx)
    b = pd.Series(base, index=idx)
    stat, _ = metrics.diebold_mariano(a, b, lag=0)
    assert stat > 0


def test_dm_identical_errors_returns_nan():
    idx = pd.date_range("2020-01-01", periods=10)
    a = pd.Series(np.ones(10), index=idx)
    stat, p = metrics.diebold_mariano(a, a.copy(), lag=0)
    assert np.isnan(stat) and np.isnan(p)
