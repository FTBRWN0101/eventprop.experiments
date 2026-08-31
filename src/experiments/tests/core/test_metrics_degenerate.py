"""Metrics that go degenerate on particular targets, and the rv-space error series."""

import numpy as np
import pandas as pd

from core import metrics
from core.config import ExperimentConfig
from core.dataset import VrpDataset


def test_dir_acc_is_nan_when_the_target_never_changes_sign():
    #rv_fwd is strictly positive: any positive forecast would score a perfect 1.0
    real = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.5, 2.5, 0.5, 9.0])
    assert np.isnan(metrics.directional_accuracy(real, pred))


def test_dir_acc_still_scores_a_two_signed_target():
    real = np.array([1.0, -2.0, 3.0, -4.0])
    pred = np.array([1.0, -1.0, -1.0, -1.0])
    assert metrics.directional_accuracy(real, pred) == 0.75


def test_dir_acc_ignores_zeros_before_judging_sign_variation():
    real = np.array([0.0, 1.0, 2.0])
    pred = np.array([1.0, 1.0, 1.0])
    assert np.isnan(metrics.directional_accuracy(real, pred))


def test_squared_errors_rv_is_the_error_after_inversion():
    data = VrpDataset(ExperimentConfig.load())
    split = data.split("test")
    #a forecast that is right in rv space must have zero rv-space error, whatever the
    #target space does to it
    pred = split.to_target(split.rv_fwd)
    errors = metrics.squared_errors_rv(split, pred)
    assert np.allclose(errors.to_numpy(), 0.0, atol=1e-9)


def test_squared_errors_rv_differs_from_target_space_on_vvrp():
    #vvrp squares the map, so the two error series are not proportional
    data = VrpDataset(ExperimentConfig.load(target="vvrp"))
    split = data.split("test")
    pred = split.to_target(split.rv_fwd * 1.1)
    target_space = metrics.squared_errors(split, pred).to_numpy()
    rv_space = metrics.squared_errors_rv(split, pred).to_numpy()
    ratio = target_space / rv_space
    assert ratio.std() > 1e-6, "a constant ratio would mean the spaces agree"


def test_squared_errors_rv_equals_target_space_on_rv_fwd():
    data = VrpDataset(ExperimentConfig.load(target="rv_fwd"))
    split = data.split("test")
    pred = split.target * 1.05
    assert np.allclose(metrics.squared_errors(split, pred).to_numpy(),
                       metrics.squared_errors_rv(split, pred).to_numpy())


def test_dm_lag_changes_the_statistic_on_autocorrelated_errors():
    index = pd.date_range("2020-01-01", periods=200, freq="D")
    rng = np.random.default_rng(0)
    shared = pd.Series(rng.normal(size=200), index=index).rolling(5).mean().bfill()
    a = shared + 1.0
    b = shared
    plain, _ = metrics.diebold_mariano(a, b, lag=0)
    hac, _ = metrics.diebold_mariano(a, b, lag=4)
    assert not np.isclose(plain, hac), "the HAC window must do something here"


def test_evaluate_reports_nan_dir_acc_on_the_rv_fwd_arm():
    data = VrpDataset(ExperimentConfig.load(target="rv_fwd"))
    split = data.split("test")
    train_mean = float(data.split("train").target.mean())
    out = metrics.evaluate(split, split.target * 1.05, train_mean, train_mean)
    assert np.isnan(out["dir_acc"])
    #gate and r2_rv are the same quantity on this arm, by construction
    assert np.isclose(out["gate"], out["r2_rv"])
