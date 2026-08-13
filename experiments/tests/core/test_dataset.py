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
    X, _, dates = default_data.sequences("test", space="zscore")
    frame = default_data.daily("full")
    cols = default_data.feature_columns(frame)
    row, _ = default_data._apply_transform(
        frame.loc[[dates[0]], cols].to_numpy(), cols, "zscore")
    np.testing.assert_allclose(X[0, -1], row[0], rtol=1e-10)


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
    held_means = np.array([m for m, _ in held._fit_transform(cols, "zscore")])
    full_means = np.array([m for m, _ in full._fit_transform(cols, "zscore")])
    assert not np.allclose(held_means, full_means)

    manual = pd.read_csv(held.config.processed_path("train"), parse_dates=["date"])
    manual = manual.set_index("date").sort_index()
    manual = manual.loc[manual.index < "2017-01-01"]
    #per column on its own non-NaN rows
    expected = np.array([manual[c].dropna().mean() for c in cols])
    np.testing.assert_allclose(held_means, expected)


def test_transform_basis_is_invariant_to_feature_set():
    """A shared feature must be scaled identically in both ablation arms."""
    options = VrpDataset(ExperimentConfig.load(feature_set="options+price"))
    price = VrpDataset(ExperimentConfig.load(feature_set="price-only"))
    shared = "rv_1"
    o_cols = options.feature_columns(options.daily("full"))
    p_cols = price.feature_columns(price.daily("full"))
    o_fit = options._fit_transform(o_cols, "zscore")[o_cols.index(shared)]
    p_fit = price._fit_transform(p_cols, "zscore")[p_cols.index(shared)]
    np.testing.assert_allclose(o_fit, p_fit)


def test_unit_space_is_bounded_and_monotone(default_data):
    frame = default_data.daily("full")
    cols = default_data.feature_columns(frame)
    values = frame[cols].to_numpy()
    out, pinned = default_data._apply_transform(values, cols, "unit")
    finite = np.isfinite(out)
    assert out[finite].min() >= 0.0 and out[finite].max() <= 1.0
    #sorting by raw value must leave the transform non-decreasing
    column = values[:, 0]
    keep = np.isfinite(column)
    order = np.argsort(column[keep])
    assert np.all(np.diff(out[keep, 0][order]) >= -1e-12)
    #saturation must be recorded, not silent
    assert set(pinned) == set(cols)


def test_excluded_features_are_absent(default_data):
    from core.config import EXCLUDED_FEATURES

    cols = default_data.feature_columns(default_data.daily("full"))
    assert not set(cols) & set(EXCLUDED_FEATURES)


def test_sample_start_trims_all_splits():
    data = VrpDataset(ExperimentConfig.load(sample_start="2011-01-04"))
    assert data.daily("train").index.min() >= pd.Timestamp("2011-01-04")
    assert data.daily("full").index.min() >= pd.Timestamp("2011-01-04")
