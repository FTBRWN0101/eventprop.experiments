"""The HAR fit-sample curve: does it hold everything but the fit sample fixed?"""

import pandas as pd
import pytest

from core.config import ExperimentConfig
from core.dataset import VrpDataset
from tools import har_sample_curve as tool


@pytest.fixture(scope="module")
def frame():
    return tool.curve(ExperimentConfig.load(), ("1990-01-01", "2005-01-01"))


def test_the_shared_start_is_always_included(frame):
    #whatever starts are asked for, the D27 row must be there to compare against
    assert frame["is_d27"].sum() == 1
    d27 = frame[frame["is_d27"]].iloc[0]
    expected = VrpDataset(ExperimentConfig.load()).fit_start()
    assert pd.Timestamp(d27["fit_from"]) >= expected


def test_a_later_start_fits_on_strictly_fewer_rows(frame):
    rows = frame.sort_values("fit_from")["fit_rows"].to_numpy()
    assert all(b < a for a, b in zip(rows, rows[1:]))


def test_the_fit_range_upper_bound_never_moves(frame):
    assert frame["fit_to"].nunique() == 1


def test_the_coefficients_actually_change_with_the_sample(frame):
    assert frame["b_rv21"].nunique() == len(frame)


def test_the_d27_restriction_improves_the_score_here(frame):
    #the direction HARNESS-REVIEW R2 assumed was the other way round; pin it
    full = frame[frame["fit_from"].str.startswith("1990")].iloc[0]
    d27 = frame[frame["is_d27"]].iloc[0]
    assert d27["r2_rv"] > full["r2_rv"]
    assert d27["mse_target"] < full["mse_target"]


def test_the_d27_restriction_costs_qlike_while_it_buys_r2_rv(frame):
    """The two co-primary metrics disagree about D27, so the report has to price both.

    Measured on the vix9d weekly cell: fitting from 2011 instead of 1990 takes r2_rv
    from 0.4784 to 0.4998 and QLIKE the wrong way, from 0.4696 to 0.4854. "D27 improves
    the benchmark" is therefore true of r2_rv and false of QLIKE.
    """
    full = frame[frame["fit_from"].str.startswith("1990")].iloc[0]
    d27 = frame[frame["is_d27"]].iloc[0]
    assert d27["qlike"] > full["qlike"], "QLIKE no longer disagrees with r2_rv here"


def test_the_tool_runs_end_to_end(capsys):
    tool.main(["--starts", "1990-01-01,2005-01-01"])
    out = capsys.readouterr().out
    assert "<- D27" in out and "cost of D27" in out
    #the co-primary metric is priced too, and the split between the two is stated
    assert "cost on QLIKE" in out
    assert "headline metrics DISAGREE" in out
