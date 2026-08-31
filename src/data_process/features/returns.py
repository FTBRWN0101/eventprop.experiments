"""S&P 500 daily log-return feature."""

from __future__ import annotations

import pandas as pd

from core.base import FeatureBuilder
from core.loaders import RawData
from core.volatility import log_returns


class ReturnsFeature(FeatureBuilder):
    """Daily log return of the S&P 500 (signed)."""

    name = "returns"
    description = "S&P 500 daily log return (signed)"
    requires = ("spx",)
    optional = False

    def build(self, raw: RawData) -> pd.DataFrame:
        return pd.DataFrame({"spx_logret": log_returns(raw["spx"])})
