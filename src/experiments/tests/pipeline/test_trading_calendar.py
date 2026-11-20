"""The panel calendar is the equity index's, not the union of every vendor's.

CBOE publishes a VIX row on days the US market was shut. Outer-joined onto the SPX
series those dates arrive as trading days carrying no price, and ``log_returns``
rightly refuses to bridge a missing observation, so the return on the next real
trading day, and every realised-vol window spanning the holiday, would be voided.
The 2022-2025 holidays alone cost 563 of the 1508 weekly test rows this way.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest


def _write_sources(root, spx: pd.Series, vix_dates: pd.DatetimeIndex) -> None:
    """Lay out a minimal ``data-save`` tree with just the two required series."""
    save = root / "data-save"
    (save / "index_returns").mkdir(parents=True, exist_ok=True)
    (save / "cboe").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": spx.index.strftime("%Y-%m-%d"), "adj_close": spx.to_numpy()}
                 ).to_csv(save / "index_returns" / "spx_daily.csv", index=False)
    pd.DataFrame({"DATE": vix_dates.strftime("%m/%d/%Y"), "CLOSE": [20.0] * len(vix_dates)}
                 ).to_csv(save / "cboe" / "vix_daily.csv", index=False)


def test_a_vendor_only_holiday_never_becomes_a_panel_row(tmp_path, data_process_import):
    config_module = data_process_import("core.config")
    loaders = data_process_import("core.loaders")

    #Juneteenth 2022: CBOE published, the equity market was shut
    published = pd.bdate_range("2022-06-15", periods=6)
    holiday = pd.Timestamp("2022-06-20")
    traded = published[published != holiday]
    _write_sources(tmp_path, pd.Series(range(100, 100 + len(traded)), index=traded,
                                       dtype=float), published)

    raw = loaders.RawData.load(config_module.ProcessConfig.load(repo_root=tmp_path))

    assert holiday not in raw.frame.index
    assert raw.frame.index.equals(pd.DatetimeIndex(traded))
    #the day after the holiday still has a predecessor, so its return survives
    volatility = data_process_import("core.volatility")
    returns = volatility.log_returns(raw["spx"])
    prices = raw["spx"]
    expected = math.log(prices.loc[traded[3]] / prices.loc[traded[2]])
    assert returns.loc[traded[3]] == pytest.approx(expected)
    assert returns.loc[traded[1:]].notna().all()


def test_a_blank_price_inside_the_index_file_is_kept_as_a_gap(tmp_path, data_process_import):
    """Restricting to the calendar must not swallow a genuine missing observation."""
    config_module = data_process_import("core.config")
    loaders = data_process_import("core.loaders")

    traded = pd.bdate_range("2022-06-15", periods=5)
    prices = pd.Series([100.0, 101.0, float("nan"), 103.0, 104.0], index=traded)
    _write_sources(tmp_path, prices, traded)

    raw = loaders.RawData.load(config_module.ProcessConfig.load(repo_root=tmp_path))
    volatility = data_process_import("core.volatility")

    #the date is still a panel row, so the no-bridging guard can see the gap
    assert traded[2] in raw.frame.index
    assert pd.isna(raw["spx"].loc[traded[2]])
    returns = volatility.log_returns(raw["spx"])
    assert pd.isna(returns.loc[traded[2]])
    assert pd.isna(returns.loc[traded[3]])
