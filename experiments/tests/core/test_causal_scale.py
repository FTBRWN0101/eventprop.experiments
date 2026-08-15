"""The adaptive delta threshold must be causal and correctly aligned."""

import numpy as np
import pandas as pd

from core.config import ExperimentConfig
from core.dataset import SCALE_MIN_PERIODS, SCALE_WINDOW, VrpDataset


def test_scale_uses_no_future_data():
    #truncating the series after row k must not change the scale at any row <= k
    rng = np.random.default_rng(1)
    feats = np.cumsum(rng.normal(size=(200, 2)), axis=0)
    full = VrpDataset._causal_scale(feats)
    for k in (30, 90, 150):
        assert np.allclose(full[:k + 1], VrpDataset._causal_scale(feats[:k + 1]),
                           equal_nan=True)


def test_scale_matches_the_rolling_std_of_increments():
    rng = np.random.default_rng(2)
    feats = rng.normal(size=(60, 3))
    expected = (pd.DataFrame(feats).diff()
                .rolling(SCALE_WINDOW, min_periods=SCALE_MIN_PERIODS).std().to_numpy())
    assert np.allclose(VrpDataset._causal_scale(feats), expected, equal_nan=True)


def test_short_history_leaves_no_threshold():
    feats = np.arange(20.0).reshape(-1, 1)
    scale = VrpDataset._causal_scale(feats)
    #diff eats one row, so the first usable window ends at SCALE_MIN_PERIODS
    assert np.isnan(scale[:SCALE_MIN_PERIODS]).all()
    assert np.isfinite(scale[SCALE_MIN_PERIODS:]).all()


def test_sequences_publishes_an_aligned_scale():
    data = VrpDataset(ExperimentConfig.load(input_window=20))
    X, _, _ = data.sequences("test", space="raw")
    assert data.window_scale.shape == X.shape
