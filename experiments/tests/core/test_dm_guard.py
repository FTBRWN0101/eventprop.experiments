"""The DM sample guard: which comparisons it admits, and which it refuses (D81).

The four cases below are the ones that exist in the results root, and they are why
the guard compares realised fit ranges rather than permitted spans. Case 4 is the
decisive one: run 20260811-195156 fitted garch to 2019-12-31 and har_rv to
2016-12-30 under --holdout-val, and a guard built on permitted spans would have
admitted it because both models were *allowed* the same dates.
"""

import pandas as pd
import pytest

from run_experiment import _samples_comparable, _scored_dates_match, _trading_gap

#a stand-in trading calendar: weekdays over the span every case below uses
CALENDAR = pd.bdate_range("1990-01-01", "2020-12-31")


def diag(**models):
    """Diagnostics dict from ``name=(start, end, tolerance)`` triples."""
    return {name: {"fitted_range": [start, end], "fit_tolerance_days": tol}
            for name, (start, end, tol) in models.items()}


def test_identical_ranges_are_comparable():
    d = diag(a=("2011-01-04", "2019-12-31", 0), b=("2011-01-04", "2019-12-31", 0))
    assert _samples_comparable(d, CALENDAR, "a", "b")[0]


def test_sequence_model_truncation_is_admitted():
    #the SNN at L=5: same start, end 24 trading days earlier from the dropped batch
    d = diag(har=("2011-01-04", "2019-12-31", 0), snn=("2011-01-04", "2019-11-26", 37))
    comparable, reason = _samples_comparable(d, CALENDAR, "har", "snn")
    assert comparable, reason


def test_long_window_truncation_is_admitted():
    #the SNN at L=45: start 44 trading days later, allowance 45 + 32
    d = diag(har=("2011-01-04", "2019-12-31", 0), snn=("2011-03-09", "2019-12-13", 77))
    comparable, reason = _samples_comparable(d, CALENDAR, "har", "snn")
    assert comparable, reason


def test_superset_is_refused_however_large_the_allowance():
    #har_rv_full reaches back to 1990: a genuine advantage, refused at any tolerance
    d = diag(har=("2011-01-04", "2019-12-31", 0), full=("1990-01-03", "2019-12-31", 9999))
    comparable, reason = _samples_comparable(d, CALENDAR, "har", "full")
    assert not comparable
    assert "1990-01-03" in reason


def test_holdout_val_asymmetry_is_refused():
    #the decisive case: garch keeps 2017-2019 while har_rv loses it (P2-2)
    d = diag(har=("1990-01-03", "2016-12-30", 0), garch=("1990-01-03", "2019-12-31", 0))
    comparable, reason = _samples_comparable(d, CALENDAR, "har", "garch")
    assert not comparable
    assert "allowance" in reason


def test_a_generous_allowance_does_not_admit_the_holdout_gap():
    #even an SNN-sized allowance is nowhere near three years, so the guard still bites
    d = diag(har=("1990-01-03", "2016-12-30", 77), garch=("1990-01-03", "2019-12-31", 77))
    assert not _samples_comparable(d, CALENDAR, "har", "garch")[0]


def test_straddling_ranges_are_refused():
    #neither inside the other: two different samples, not a truncation
    d = diag(a=("2011-01-04", "2018-12-31", 999), b=("2012-01-03", "2019-12-31", 999))
    comparable, reason = _samples_comparable(d, CALENDAR, "a", "b")
    assert not comparable
    assert "neither inside the other" in reason


def test_absent_diagnostics_count_as_matching():
    d = diag(a=("2011-01-04", "2019-12-31", 0))
    assert _samples_comparable(d, CALENDAR, "a", "nobody")[0]
    assert _samples_comparable({}, CALENDAR, "a", "b")[0]


def test_gap_is_counted_in_trading_days_not_calendar_days():
    #44 trading days is about 63 calendar days; the guard must use the former
    gap = _trading_gap(CALENDAR, "2011-01-04", "2011-03-09")
    assert 40 <= gap <= 48, gap


@pytest.mark.parametrize("tolerance,expected", [(20, False), (40, True)])
def test_allowance_is_the_deciding_quantity(tolerance, expected):
    d = diag(har=("2011-01-04", "2019-12-31", 0),
             snn=("2011-01-04", "2019-11-26", tolerance))
    assert _samples_comparable(d, CALENDAR, "har", "snn")[0] is expected


def test_scored_dates_must_match():
    dates = pd.bdate_range("2020-01-01", periods=10)
    a = pd.Series(1.0, index=dates)
    assert _scored_dates_match({"a": a, "b": a.copy()}, "a", "b")
    assert not _scored_dates_match({"a": a, "b": a.iloc[:-1]}, "a", "b")
    assert not _scored_dates_match({"a": a, "b": a.shift(1, freq="D")}, "a", "b")


def test_scored_check_tolerates_a_missing_series():
    a = pd.Series(1.0, index=pd.bdate_range("2020-01-01", periods=5))
    assert _scored_dates_match({"a": a}, "a", "absent")
