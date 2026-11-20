import numpy as np
import pandas as pd
import pytest

from core import metrics


def test_qlike_refuses_an_entire_run_with_an_invalid_forecast():
    """Invalid forecasts must not vanish from a model's reported QLIKE."""
    assert np.isnan(metrics.qlike(np.array([1.0, 2.0]), np.array([1.0, 0.0])))
    assert np.isnan(metrics.qlike(np.array([1.0, 2.0]), np.array([1.0, -0.5])))
    assert np.isnan(metrics.qlike(np.array([1.0, 2.0]), np.array([1.0, np.inf])))


def test_evaluate_refuses_an_incomplete_prediction():
    from core.dataset import Split

    index = pd.date_range("2020-01-01", periods=2)
    split = Split("test", pd.DataFrame(index=index), pd.Series([1., 2.], index=index),
                  pd.Series([1., 2.], index=index), pd.Series([1., 1.], index=index),
                  pd.Series([1., 1.], index=index), "rv_fwd", 1)
    with pytest.raises(ValueError, match="missing"):
        metrics.evaluate(split, pd.Series([1.], index=index[:1]))


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
