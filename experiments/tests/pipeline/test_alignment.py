"""The forward target must look only forward, and the features only backward.

This is the check the proposal commits to running before any model is trained. It
covers the primitive in ``data-process/core/volatility.py`` and the panels actually
written to ``data-save/processed``, so a regression in either is caught.
"""

import numpy as np
import pandas as pd
import pytest

HORIZON_DAYS = {"weekly": 5, "monthly": 21}
ANNUALISATION = 252


def _flat_prices(n: int = 200, start: float = 100.0) -> pd.Series:
    dates = pd.bdate_range("2000-01-03", periods=n)
    return pd.Series(start, index=dates, dtype=float)


def test_forward_at_t_equals_trailing_at_t_plus_window(volatility):
    #both annualise the same w squared returns, so the two must coincide exactly
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2000-01-03", periods=300)
    prices = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=dates)

    for window in HORIZON_DAYS.values():
        forward = volatility.forward_realised_vol(prices, window, ANNUALISATION)
        trailing = volatility.trailing_realised_vol(prices, window, ANNUALISATION)
        shifted = trailing.shift(-window)
        pair = pd.concat([forward, shifted], axis=1).dropna()
        assert len(pair) > 200
        assert np.allclose(pair.iloc[:, 0], pair.iloc[:, 1])


def test_forward_covers_exactly_the_next_window_returns(volatility):
    """One jump, and only the rows whose forward window contains it may move."""
    window = 5
    prices = _flat_prices()
    jump_position = 100  #index into the price series
    prices.iloc[jump_position:] = 110.0

    forward = volatility.forward_realised_vol(prices, window, ANNUALISATION)
    #log_returns drops the first date, so the single non-zero return sits here
    returns = volatility.log_returns(prices)
    jump_index = returns.index.get_loc(prices.index[jump_position])

    nonzero = forward[forward > 0].dropna()
    positions = [returns.index.get_loc(d) for d in nonzero.index]
    #the return at position j is included by rows j-window .. j-1
    assert positions == list(range(jump_index - window, jump_index))


def test_forward_never_includes_the_current_row_return(volatility):
    window = 5
    prices = _flat_prices()
    prices.iloc[100:] = 110.0
    forward = volatility.forward_realised_vol(prices, window, ANNUALISATION)
    returns = volatility.log_returns(prices)
    jump_date = prices.index[100]

    #the jump return is realised *on* jump_date, so that row's forward vol must not see it
    assert forward.loc[jump_date] == 0.0
    #and the row one step before it must
    previous = returns.index[returns.index.get_loc(jump_date) - 1]
    assert forward.loc[previous] > 0.0


def test_trailing_is_causal_under_truncation(volatility):
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2000-01-03", periods=250)
    prices = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 250))), index=dates)

    full = volatility.trailing_realised_vol(prices, 21, ANNUALISATION)
    for cut in (60, 130, 200):
        truncated = volatility.trailing_realised_vol(prices.iloc[:cut], 21, ANNUALISATION)
        pair = pd.concat([full, truncated], axis=1).dropna()
        assert len(pair) > 0
        assert np.allclose(pair.iloc[:, 0], pair.iloc[:, 1])


def test_forward_is_stable_once_its_window_has_passed(volatility):
    """Truncating the series must not change any row whose forward window is complete."""
    rng = np.random.default_rng(13)
    dates = pd.bdate_range("2000-01-03", periods=250)
    prices = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 250))), index=dates)
    window = 21

    full = volatility.forward_realised_vol(prices, window, ANNUALISATION)
    for cut in (80, 150, 220):
        truncated = volatility.forward_realised_vol(prices.iloc[:cut], window, ANNUALISATION)
        pair = pd.concat([full, truncated], axis=1).dropna()
        assert len(pair) > 0
        assert np.allclose(pair.iloc[:, 0], pair.iloc[:, 1])


@pytest.mark.parametrize("horizon", ["weekly", "monthly"])
def test_panel_rv_fwd_matches_an_independent_recomputation(
        volatility, spx_prices, panels, horizon):
    """Recompute the target from the raw price file and compare to the written panel."""
    key = (horizon, "full")
    if key not in panels:
        pytest.skip(f"no full panel for {horizon}")
    panel = panels[key]
    expected = volatility.forward_realised_vol(
        spx_prices, HORIZON_DAYS[horizon], ANNUALISATION)

    #join="inner": the two indexes differ at the edges and concat will not sort a union
    pair = pd.concat([panel["rv_fwd"].rename("panel"),
                      expected.rename("recomputed")], axis=1, join="inner").dropna()
    assert len(pair) > 1000, f"only {len(pair)} comparable rows"
    assert np.allclose(pair["panel"], pair["recomputed"]), (
        f"{horizon}: panel rv_fwd differs from an independent recomputation, "
        f"max abs diff {np.abs(pair['panel'] - pair['recomputed']).max()}")


@pytest.mark.parametrize("horizon", ["weekly", "monthly"])
def test_train_ends_before_test_begins(panels, horizon):
    train, test = panels.get((horizon, "train")), panels.get((horizon, "test"))
    if train is None or test is None:
        pytest.skip(f"no split panels for {horizon}")
    assert train.index.max() < test.index.min()
    assert test.index.min().year == 2020


@pytest.mark.parametrize("horizon", ["weekly", "monthly"])
def test_vrp_columns_reconstruct_from_their_legs(panels, horizon):
    """``vrp_<leg>`` must be exactly ``iv_<leg> - rv_fwd`` on every row."""
    panel = panels.get((horizon, "full"))
    if panel is None:
        pytest.skip(f"no full panel for {horizon}")
    legs = [c[len("iv_"):] for c in panel.columns if c.startswith("iv_")]
    assert legs, "panel carries no implied-vol leg"
    for leg in legs:
        frame = panel[[f"iv_{leg}", f"vrp_{leg}", "rv_fwd"]].dropna()
        assert len(frame) > 1000
        assert np.allclose(frame[f"vrp_{leg}"], frame[f"iv_{leg}"] - frame["rv_fwd"])
