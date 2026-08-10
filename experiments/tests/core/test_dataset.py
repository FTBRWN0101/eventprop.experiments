"""Windowing, alignment and split discipline against the real processed panels."""

import numpy as np
import pandas as pd
import pytest

from core.config import ExperimentConfig
from core.dataset import VrpDataset


@pytest.fixture(scope="module")
def default_data():
    return VrpDataset(ExperimentConfig.load())


def test_sequences_dates_match_split(default_data):
    _, _, dates = default_data.sequences("test")
    assert list(dates) == list(default_data.split("test").target.index)


def test_long_window_keeps_early_test_dates():
    #windows need history from before the split boundary
    data = VrpDataset(ExperimentConfig.load(input_window=60))
    X, _, dates = data.sequences("test")
    split_dates = data.split("test").target.index
    assert X.shape[1] == 60
    assert dates[0] == split_dates[0]
    assert set(dates) <= set(split_dates)
    assert len(dates) >= 0.95 * len(split_dates)


def test_window_ends_on_evaluation_date(default_data):
    X, _, dates = default_data.sequences("test")
    frame = default_data.daily("full")
    cols = default_data.feature_columns(frame)
    mean, std = default_data._fit_scaler(cols)
    row = (frame.loc[dates[0], cols].to_numpy() - mean) / std
    np.testing.assert_allclose(X[0, -1], row, rtol=1e-10)


def test_target_round_trip(default_data):
    split = default_data.split("test")
    rt = split.to_rv_fwd(split.to_target(split.rv_fwd))
    pd.testing.assert_series_equal(rt, split.rv_fwd, check_names=False)


def test_holdout_val_bounds():
    data = VrpDataset(ExperimentConfig.load(holdout_val=True))
    assert data.daily("train").index.max() < pd.Timestamp("2017-01-01")
    val = data.daily("val")
    assert val.index.min() >= pd.Timestamp("2017-01-01")
    assert val.index.max() < pd.Timestamp("2020-01-01")


def test_scaler_excludes_val_under_holdout():
    held = VrpDataset(ExperimentConfig.load(holdout_val=True))
    full = VrpDataset(ExperimentConfig.load())
    cols = held.feature_columns(held.daily("train"))
    held_mean, _ = held._fit_scaler(cols)
    full_mean, _ = full._fit_scaler(cols)
    assert not np.allclose(held_mean, full_mean)

    manual = pd.read_csv(held.config.processed_path("train"), parse_dates=["date"])
    manual = manual.set_index("date").sort_index()
    manual = manual.loc[manual.index < "2017-01-01", cols].dropna().to_numpy().mean(axis=0)
    np.testing.assert_allclose(held_mean, manual)


def test_sample_start_trims_all_splits():
    data = VrpDataset(ExperimentConfig.load(sample_start="2011-01-04"))
    assert data.daily("train").index.min() >= pd.Timestamp("2011-01-04")
    assert data.daily("full").index.min() >= pd.Timestamp("2011-01-04")
