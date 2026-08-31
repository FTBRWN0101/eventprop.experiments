"""OptionMetrics vol surface -> daily columns.

One block per date: 11 tenors x 34 deltas, puts negative and calls positive.
"""

from __future__ import annotations

import pandas as pd

from core.base import FeatureBuilder
from core.loaders import RawData

_ATM = 50.0
_WING = 25.0


class SurfaceFeature(FeatureBuilder):
    """ATM level, smile asymmetry and term slope from the SPX vol surface."""

    name = "om_surface"
    description = "OptionMetrics SPX vol-surface slices (ATM, skew, term structure)"
    requires = ()
    optional = True

    def _load(self) -> pd.DataFrame:
        paths = sorted(self.config.source_dir("optionmetrics").glob("spx_vsurf_*.csv"))
        if not paths:
            self.logger.warning("no spx_vsurf_*.csv found; skipping surface features")
            return pd.DataFrame()
        frame = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        frame["date"] = pd.to_datetime(frame["date"])
        #IV ships as a decimal; x100 keeps it commensurate with the VIX-scale columns
        frame["iv"] = pd.to_numeric(frame["impl_volatility"], errors="coerce") * 100.0
        return frame

    def _slice(self, frame: pd.DataFrame, days: float, delta: float) -> pd.Series:
        rows = frame[(frame["days"] == days) & (frame["delta"] == delta)]
        series = rows.set_index("date")["iv"].sort_index()
        return series[~series.index.duplicated(keep="last")]

    def build(self, raw: RawData) -> pd.DataFrame:
        frame = self._load()
        if frame.empty:
            return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))

        atm30 = (self._slice(frame, 30.0, -_ATM) + self._slice(frame, 30.0, _ATM)) / 2.0
        atm91 = (self._slice(frame, 91.0, -_ATM) + self._slice(frame, 91.0, _ATM)) / 2.0
        put25 = self._slice(frame, 30.0, -_WING)
        call25 = self._slice(frame, 30.0, _WING)

        out = pd.DataFrame({
            "om_atm30": atm30,
            #put wing over call wing: how much more the market pays for crash protection
            "om_skew30": put25 - call25,
            #positive = contango; inverts under stress
            "om_term": atm91 - atm30,
        })
        out.index.name = "date"
        return out
