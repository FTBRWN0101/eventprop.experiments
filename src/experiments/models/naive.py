"""Naive baselines: constant and persistence.

Persistence lags by the horizon, not one day, because the target at t-1 is
not known at t.
"""

from __future__ import annotations

import pandas as pd

from core.dataset import VrpDataset
from models.base import Forecaster


class ConstantForecaster(Forecaster):
    """Predicts the training-split mean of the target at every date."""

    name = "constant"
    description = "Train-split mean of the target (the bar every model must clear)"

    def fit(self, data: VrpDataset) -> None:
        train = data.split("train")
        self.fitted_range = (str(train.target.index.min().date()),
                             str(train.target.index.max().date()))
        self._mean = float(train.target.mean())

    def predict(self, data: VrpDataset, split: str) -> pd.Series:
        index = data.split(split).target.index
        return pd.Series(self._mean, index=index)


class PersistenceForecaster(Forecaster):
    """Predicts the target as last observed ``horizon_days`` ago.

    Lagged by the horizon, not a day: the target at t-1 is not known at t.
    """

    name = "persistence"
    description = "Target lagged by the forecast horizon (last observable value)"

    def fit(self, data: VrpDataset) -> None:
        train = data.split("train")
        self.fitted_range = (str(train.target.index.min().date()),
                             str(train.target.index.max().date()))
        #fallback for dates with no history, e.g. the first rows of a split
        self._fallback = float(train.target.mean())

    def predict(self, data: VrpDataset, split: str) -> pd.Series:
        lag = self.config.horizon_days
        #lag on the daily frame so the offset is exact
        daily = data.daily("full")[self.config.target_column].sort_index()
        lagged = daily.shift(lag)
        index = data.split(split).target.index
        predictions = lagged.reindex(index)
        if predictions.isna().any():
            predictions = predictions.fillna(self._fallback)
        return predictions.astype(float)
