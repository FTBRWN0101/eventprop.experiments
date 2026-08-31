"""VVIX and SKEW: vol-of-vol and tail risk. Both optional."""

from __future__ import annotations

import pandas as pd

from core.base import FeatureBuilder
from core.loaders import RawData


class VolIndicesFeature(FeatureBuilder):
    """VVIX and SKEW levels, included when present in the raw panel."""

    name = "vol_indices"
    description = "VVIX (vol-of-vol) and SKEW (tail-risk) index levels"
    requires = ()
    optional = True

    #output column -> raw series
    LEVELS: dict[str, str] = {"vvix": "vvix", "skew": "skew"}

    def build(self, raw: RawData) -> pd.DataFrame:
        columns = {
            name: raw[series]
            for name, series in self.LEVELS.items()
            if series in raw.columns
        }
        return pd.DataFrame(columns)
