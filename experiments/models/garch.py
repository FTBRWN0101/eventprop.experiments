"""GARCH(1,1) baseline on daily S&P 500 log returns.

Returns are scaled by 100 for optimiser stability; the 100s cancel in the
annualised-vol formula.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model

from core.dataset import VrpDataset
from models.base import Forecaster


class GarchForecaster(Forecaster):
    """GARCH(1,1) with constant mean and normal innovations."""

    name = "garch"
    description = "GARCH(1,1) on daily returns -> forward-vol forecast -> VRP"

    def fit(self, data: VrpDataset) -> None:
        #last_obs fits on train only, still filters test returns
        returns = (data.full()["spx_logret"] * 100.0).dropna()
        self._first_test = data.daily("test").index.min()
        fit_returns = returns.loc[returns.index < self._first_test]
        self.fitted_range = (str(fit_returns.index.min().date()),
                             str(fit_returns.index.max().date()))
        model = arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
        self._result = model.fit(last_obs=self._first_test, disp="off")

    def predict(self, data: VrpDataset, split: str) -> pd.Series:
        view = data.split(split)
        horizon = view.horizon_days
        forecast = self._result.forecast(
            horizon=horizon, start=self._first_test, reindex=False)
        #sum h daily variance forecasts, then annualise
        forward_var = forecast.variance.sum(axis=1)
        rv_fwd_hat = np.sqrt(forward_var * (data.config.annualisation / horizon))
        rv_fwd_hat = rv_fwd_hat.reindex(view.features.index)
        return view.to_target(rv_fwd_hat)
