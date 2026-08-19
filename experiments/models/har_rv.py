"""HAR-RV baseline (Corsi, 2009), the standard econometric benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.dataset import VrpDataset
from models.base import Forecaster


class HarRvForecaster(Forecaster):
    """Heterogeneous Autoregressive model of Realised Volatility."""

    name = "har_rv"
    description = "HAR-RV: OLS of forward RV on daily/weekly/monthly RV components"

    COMPONENTS: tuple[str, ...] = ("rv_1", "rv_5", "rv_21")
    #D27/D38: clip to the shared fit start so every model sees one sample
    restrict_sample: bool = True
    #explicit fit-sample floor, overriding restrict_sample; tools/har_sample_curve.py
    #sets it to price the D27 restriction. Never set on a scored run.
    sample_floor: pd.Timestamp | None = None

    def fit(self, data: VrpDataset) -> None:
        train = data.daily("train")[[*self.COMPONENTS, "rv_fwd"]].dropna()
        #D27: the sequence models start where the target column does, so clip to the
        #same date rather than fitting HAR on 20 extra years the SNN never sees
        floor = self.sample_floor
        if floor is None and self.restrict_sample:
            floor = data.fit_start()
        if floor is not None:
            train = train.loc[train.index >= floor]
        self.fitted_range = (str(train.index.min().date()),
                             str(train.index.max().date()))
        design = self._design(train[list(self.COMPONENTS)].to_numpy())
        self._coef, *_ = np.linalg.lstsq(design, train["rv_fwd"].to_numpy(), rcond=None)

    def predict(self, data: VrpDataset, split: str) -> pd.Series:
        view = data.split(split)
        design = self._design(view.features[list(self.COMPONENTS)].to_numpy())
        rv_fwd_hat = pd.Series(design @ self._coef, index=view.features.index)
        return view.to_target(rv_fwd_hat)

    @staticmethod
    def _design(components: np.ndarray) -> np.ndarray:
        """Prepend an intercept column to the HAR components."""
        return np.column_stack([np.ones(len(components)), components])


class HarRvFullForecaster(HarRvForecaster):
    """HAR-RV on the full daily history, ignoring the D27 shared fit start.

    D27/D38 restricting HAR to ~2011+ is the right call for a like-for-like DM, and it
    also weakens the benchmark, which flatters everything compared against it. This arm
    keeps the ~20 extra years so the cost of that restriction is reportable rather than
    asserted. It is NOT sample-matched to the sequence models -- do not quote a DM
    against it as if it were.
    """

    name = "har_rv_full"
    description = "HAR-RV fitted on the full daily history (1990+), sample-unmatched"

    restrict_sample = False
