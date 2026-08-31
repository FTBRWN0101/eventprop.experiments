"""VIX term-structure features: ratios of term points to the 30-day VIX.

Optional: the term points start later than the VIX, so they carry NaN before
their history begins.
"""

from __future__ import annotations

import pandas as pd

from core.base import FeatureBuilder
from core.loaders import RawData


class TermStructureFeature(FeatureBuilder):
    """Term-structure slope ratios relative to the 30-day VIX."""

    name = "term_structure"
    description = "VIX term-structure ratios (VIX9D/3M/6M relative to VIX)"
    requires = ("vix",)
    optional = True

    #output column -> raw term point
    RATIOS: dict[str, str] = {
        "ts_9d": "vix9d",
        "ts_3m": "vix3m",
        "ts_6m": "vix6m",
    }

    def build(self, raw: RawData) -> pd.DataFrame:
        vix = raw["vix"]
        columns = {
            name: raw[point] / vix
            for name, point in self.RATIOS.items()
            if point in raw.columns
        }
        #3M minus 9D slope
        if "vix3m" in raw.columns and "vix9d" in raw.columns:
            columns["ts_slope"] = raw["vix3m"] - raw["vix9d"]
        return pd.DataFrame(columns)
