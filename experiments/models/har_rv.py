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

    def fit(self, data: VrpDataset) -> None:
        train = data.daily("train")[[*self.COMPONENTS, "rv_fwd"]].dropna()
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
