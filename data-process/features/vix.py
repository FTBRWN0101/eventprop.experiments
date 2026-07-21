"""VIX level and its daily log change. Required: it anchors the usable sample."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.base import FeatureBuilder
from core.loaders import RawData


class VixLevelFeature(FeatureBuilder):
    """The VIX close and its one-day log change."""

    name = "vix_level"
    description = "VIX close level and daily log change"
    requires = ("vix",)
    optional = False

    def build(self, raw: RawData) -> pd.DataFrame:
        vix = raw["vix"].dropna()
        return pd.DataFrame({
            "vix": vix,
            "vix_logret": np.log(vix / vix.shift(1)),
        })
