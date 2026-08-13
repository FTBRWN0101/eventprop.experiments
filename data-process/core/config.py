"""Runtime configuration for the VRP dataset builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import pandas as pd

from core.base import Horizon

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]   #<repo>/data-process
_REPO_ROOT = _PACKAGE_ROOT.parent                      #<repo>

DATA_SAVE_DIRNAME = "data-save"
PROCESSED_DIRNAME = "processed"

#horizon name -> trading days
DEFAULT_HORIZONS: tuple[Horizon, ...] = (
    Horizon("weekly", 5),
    Horizon("monthly", 21),
)


@dataclass(frozen=True)
class ProcessConfig:
    """Immutable view of paths and numerical conventions for one build."""

    repo_root: Path = _REPO_ROOT
    horizons: tuple[Horizon, ...] = DEFAULT_HORIZONS

    #trading days per year, for annualising
    annualisation: int = 252
    #HAR daily, weekly and monthly components
    trailing_windows: tuple[int, ...] = (1, 5, 21)
    #vix9d only exists from 2011
    iv_legs: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: {"weekly": ("vix9d", "vix"), "monthly": ("vix",)})
    #rolling-mean denominator for the relative VRP
    rvrp_smooth_window: int = 5
    #first out-of-sample date
    split_date: str = "2020-01-01"
    #last test date, None keeps everything after
    test_end: str | None = "2025-12-31"
    #bounds fitted on train only, no look-ahead
    winsor_quantiles: tuple[float, float] = (0.01, 0.99)
    #carry short CBOE gaps forward, longer ones stay NaN
    max_fill_gap: int = 5

    def legs_for(self, horizon: Horizon) -> tuple[str, ...]:
        """Implied-vol legs to build for *horizon* (defaults to the 30-day VIX)."""
        return tuple(self.iv_legs.get(horizon.name, ("vix",)))

    @classmethod
    def load(cls, **overrides: object) -> "ProcessConfig":
        """Build a config, applying any keyword *overrides* over the defaults."""
        return cls(**overrides)  # type: ignore[arg-type]

    @property
    def data_save_dir(self) -> Path:
        """Root the raw sources were downloaded to (``<repo>/data-save``)."""
        return self.repo_root / DATA_SAVE_DIRNAME

    @property
    def processed_dir(self) -> Path:
        """Root all processed panels are written under (``data-save/processed``)."""
        return self.data_save_dir / PROCESSED_DIRNAME

    def source_dir(self, source: str) -> Path:
        """Return ``data-save/<source>/`` (where a downloader wrote its CSVs)."""
        return self.data_save_dir / source

    def horizon_dir(self, horizon: Horizon) -> Path:
        """Return ``data-save/processed/<horizon>/``, creating it on first use."""
        path = self.processed_dir / horizon.name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def split_ts(self) -> pd.Timestamp:
        """The train/test boundary as a timestamp."""
        return pd.Timestamp(self.split_date)

    @property
    def test_end_ts(self) -> pd.Timestamp | None:
        """The inclusive end of the test window, or ``None`` if unbounded."""
        return pd.Timestamp(self.test_end) if self.test_end is not None else None
