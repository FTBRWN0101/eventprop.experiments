"""Declarative loading of the raw daily CSVs into one normalised panel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from core.config import ProcessConfig


@dataclass(frozen=True)
class RawSeries:
    """One column extracted from one raw CSV, mapped to a canonical name."""

    name: str
    source: str
    filename: str
    column: str
    date_column: str = "DATE"
    date_format: str | None = None
    required: bool = True


#CBOE ships MM/DD/YYYY with OHLC, or one named column
_CBOE_DATE_FMT = "%m/%d/%Y"

#the raw inputs; add a series here
RAW_SERIES: tuple[RawSeries, ...] = (
    RawSeries("vix", "cboe", "vix_daily.csv", "CLOSE", date_format=_CBOE_DATE_FMT),
    RawSeries("vix9d", "cboe", "vix9d_daily.csv", "CLOSE",
              date_format=_CBOE_DATE_FMT, required=False),
    RawSeries("vix3m", "cboe", "vix3m_daily.csv", "CLOSE",
              date_format=_CBOE_DATE_FMT, required=False),
    RawSeries("vix6m", "cboe", "vix6m_daily.csv", "CLOSE",
              date_format=_CBOE_DATE_FMT, required=False),
    RawSeries("vvix", "cboe", "vvix_daily.csv", "VVIX",
              date_format=_CBOE_DATE_FMT, required=False),
    RawSeries("skew", "cboe", "skew_daily.csv", "SKEW",
              date_format=_CBOE_DATE_FMT, required=False),
    RawSeries("spx", "index_returns", "spx_daily.csv", "adj_close",
              date_column="date", date_format="%Y-%m-%d"),
)


@dataclass
class RawData:
    """A normalised daily panel of every successfully loaded raw series."""

    frame: pd.DataFrame
    series: tuple[RawSeries, ...] = field(default_factory=tuple)

    @property
    def columns(self) -> list[str]:
        """Canonical names actually present in the panel."""
        return list(self.frame.columns)

    def __getitem__(self, name: str) -> pd.Series:
        return self.frame[name]

    @classmethod
    def load(cls, config: "ProcessConfig",
             series: tuple[RawSeries, ...] = RAW_SERIES) -> "RawData":
        """Read every *series* from ``data-save`` and outer-join into one daily panel."""
        loaded: dict[str, pd.Series] = {}
        used: list[RawSeries] = []
        for spec in series:
            value = cls._read_series(config, spec)
            if value is None:
                continue
            loaded[spec.name] = value
            used.append(spec)
        if not loaded:
            raise FileNotFoundError("no raw series could be loaded from data-save")
        frame = pd.concat(loaded, axis=1).sort_index()
        frame.index.name = "date"
        return cls(frame=frame, series=tuple(used))

    @staticmethod
    def _read_series(config: "ProcessConfig", spec: RawSeries) -> pd.Series | None:
        path = config.source_dir(spec.source) / spec.filename
        if not path.is_file():
            if spec.required:
                raise FileNotFoundError(f"required raw series missing: {path}")
            return None
        raw = pd.read_csv(path, usecols=[spec.date_column, spec.column])
        dates = pd.to_datetime(raw[spec.date_column], format=spec.date_format)
        out = pd.Series(
            pd.to_numeric(raw[spec.column], errors="coerce").to_numpy(),
            index=pd.DatetimeIndex(dates), name=spec.name,
        )
        #collapse duplicate dates to the last value
        return out[~out.index.duplicated(keep="last")].sort_index()
