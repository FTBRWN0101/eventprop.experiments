"""Realised-volatility primitives shared by features and targets."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns of a positive price series, NaNs from gaps dropped."""
    clean = prices.dropna()
    return np.log(clean / clean.shift(1)).dropna()


def _annualise(squared_sum: pd.Series, window: int, annualisation: int) -> pd.Series:
    """Annualise a window's summed squared returns into volatility points (x100)."""
    return np.sqrt(squared_sum * (annualisation / window)) * 100.0


def trailing_realised_vol(prices: pd.Series, window: int,
                          annualisation: int = 252) -> pd.Series:
    """Annualised realised vol over the trailing *window*. Known at t, so usable as a feature."""
    r2 = log_returns(prices) ** 2
    return _annualise(r2.rolling(window).sum(), window, annualisation)


def forward_realised_vol(prices: pd.Series, window: int,
                         annualisation: int = 252) -> pd.Series:
    """Annualised realised vol over the forward *window*. Label only, never a feature."""
    r2 = log_returns(prices) ** 2
    #shift back by window so the forward sum lands on row t
    forward_sum = r2.rolling(window).sum().shift(-window)
    return _annualise(forward_sum, window, annualisation)
