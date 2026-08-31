"""Trailing realised-volatility features from S&P 500 daily returns."""

from __future__ import annotations

import pandas as pd

from core.base import FeatureBuilder
from core.loaders import RawData
from core.volatility import trailing_realised_vol


class RealisedVolFeature(FeatureBuilder):
    """Annualised trailing realised volatility over each configured window."""

    name = "realised_vol"
    description = "Trailing realised volatility (annualised) over HAR-style windows"
    requires = ("spx",)
    optional = False

    def build(self, raw: RawData) -> pd.DataFrame:
        prices = raw["spx"]
        columns = {
            f"rv_{window}": trailing_realised_vol(
                prices, window, self.config.annualisation)
            for window in self.config.trailing_windows
        }
        return pd.DataFrame(columns)
