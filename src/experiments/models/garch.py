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

    #D27/D38: clip to the shared fit start so every model sees one sample
    restrict_sample: bool = True

    def fit(self, data: VrpDataset) -> None:
        #last_obs fits on train only, still filters test returns
        returns = (data.full()["spx_logret"] * 100.0).dropna()
        self._first_test = data.daily("test").index.min()
        #D27: clip the lower bound to the shared fit start. The upper end stays open
        #because arch needs the test-period returns present to forecast over them --
        #last_obs is what keeps them out of the fit.
        if self.restrict_sample:
            returns = returns.loc[returns.index >= data.fit_start()]
        #P2-2: the fit is bounded by the end of the TRAIN split, not by the first test
        #date. Under --holdout-val the train split stops in 2016 while the test split
        #still begins in 2020, so keying on the test date fed garch the three
        #validation years that har_rv and the sequence models lose. Without
        #--holdout-val no dates sit between the two bounds, so this changes nothing.
        train_end = data.daily("train").index.max()
        later = returns.index[returns.index > train_end]
        self._last_obs = later.min() if len(later) else self._first_test
        fit_returns = returns.loc[returns.index <= train_end]
        self.fitted_range = (str(fit_returns.index.min().date()),
                             str(fit_returns.index.max().date()))
        model = arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
        self._result = model.fit(last_obs=self._last_obs, disp="off")

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


class GarchFullForecaster(GarchForecaster):
    """GARCH(1,1) on the full return history, ignoring the D27 shared fit start.

    Sibling of ``har_rv_full``: the restricted arm fits ~9 years of daily returns
    rather than ~30, so the benchmark is weaker than it needs to be. Not
    sample-matched to the sequence models.
    """

    name = "garch_full"
    description = "GARCH(1,1) fitted on the full return history (1990+), sample-unmatched"

    restrict_sample = False
