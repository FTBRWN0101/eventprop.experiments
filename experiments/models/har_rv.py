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
    #O8/D61: regress the configured target itself instead of forward vol, skipping the
    #to_target map. False keeps the standard construction, see HarRvDirectForecaster.
    direct: bool = False

    def _regressand(self, data: VrpDataset) -> str:
        """Column the OLS is fitted against."""
        return data.config.target_column if self.direct else "rv_fwd"

    def fit(self, data: VrpDataset) -> None:
        column = self._regressand(data)
        train = data.daily("train")[[*self.COMPONENTS, column]].dropna()
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
        self._coef, *_ = np.linalg.lstsq(design, train[column].to_numpy(), rcond=None)

    def predict(self, data: VrpDataset, split: str) -> pd.Series:
        view = data.split(split)
        design = self._design(view.features[list(self.COMPONENTS)].to_numpy())
        fitted = pd.Series(design @ self._coef, index=view.features.index)
        return fitted if self.direct else view.to_target(fitted)

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


class HarRvDirectForecaster(HarRvForecaster):
    """HAR-RV regressed on the target itself rather than on forward vol (O8, D61).

    The standard arm forecasts ``rv_fwd`` and lets ``Split.to_target`` convert, which
    hands it ``iv_t`` exactly, whatever the feature set. ``snn`` and ``lstm`` predict
    the target column directly and have to rebuild that relationship from features, so
    the two families are not solving the same problem on a VRP-family target. This arm
    makes the baseline solve the sequence models' problem, so the gap is measured
    rather than declared.

    D61 measured the direct route as scoring *better* on ``vrp`` (MSE 75.667 against
    95.544) and identified why: the near-white VRP target rewards collapsing toward the
    constant rather than forecasting. Read the pair together, never this arm alone.

    Identical to its parent on ``rv_fwd``, where ``to_target`` is the identity and the
    target column is ``rv_fwd``. A test pins that.
    """

    name = "har_rv_direct"
    description = "HAR-RV regressed straight onto the configured target, no to_target map"

    direct = True
