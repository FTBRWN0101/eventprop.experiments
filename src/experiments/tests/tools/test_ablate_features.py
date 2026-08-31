"""The feature ablation: which withheld signal is costliest, on each co-primary metric.

The table used to rank the withheld signals by ``d_r2_rv`` alone, which makes "the most
important signal" a claim about one metric dressed up as a claim about the model. QLIKE
is reported beside it, and the two are allowed to disagree out loud.

Nothing here fits a model. ``run_ablation`` refits an LSTM once per served column, which
is minutes of work and is covered by running the tool; these cover the reporting, which
is the part that was wrong.
"""

import pandas as pd
import pytest

from tools import ablate_features as tool


def _frame(*rows):
    """An ablation table of ``(dropped, d_r2_rv, d_qlike)`` rows, plus its baseline."""
    baseline = {"model": "lstm", "dropped": "(none)", "d_r2_rv": float("nan"),
                "d_qlike": float("nan")}
    return pd.DataFrame([baseline] + [
        {"model": "lstm", "dropped": name, "d_r2_rv": r2, "d_qlike": q}
        for name, r2, q in rows])


def test_the_costliest_signal_is_named_on_both_metrics(caplog):
    #vix9d hurts most on both: the largest r2_rv lost and the largest QLIKE gained
    frame = _frame(("vix9d", -0.30, 0.40), ("rv_5", -0.02, 0.01))
    with caplog.at_level("INFO"):
        tool._report_most_damaging(frame)
    assert "vix9d is the costliest signal on both metrics" in caplog.text
    assert "d_r2_rv -0.3000" in caplog.text and "d_qlike +0.4000" in caplog.text


def test_a_split_between_the_metrics_is_stated_outright(caplog):
    #r2_rv says vix9d, QLIKE says rv_21: the ranking depends on the loss
    frame = _frame(("vix9d", -0.30, 0.05), ("rv_21", -0.05, 0.60))
    with caplog.at_level("INFO"):
        tool._report_most_damaging(frame)
    assert "DISAGREE about the costliest signal" in caplog.text
    assert "r2_rv says vix9d (-0.3000)" in caplog.text
    assert "QLIKE says rv_21 (+0.6000)" in caplog.text


def test_the_baseline_row_is_not_a_candidate(caplog):
    """"(none)" is the arm nothing was withheld from, so it cannot be the costliest."""
    frame = _frame(("rv_5", -0.02, 0.01))
    with caplog.at_level("INFO"):
        tool._report_most_damaging(frame)
    assert "(none)" not in caplog.text and "rv_5" in caplog.text


def test_an_unrankable_delta_is_dropped_rather_than_winning(caplog):
    """A QLIKE that diverged gives an infinite delta, which would win any max()."""
    frame = _frame(("vix9d", -0.30, 0.40), ("rv_5", -0.02, float("inf")))
    with caplog.at_level("INFO"):
        tool._report_most_damaging(frame)
    assert "vix9d is the costliest signal on both metrics" in caplog.text


def test_no_finite_delta_at_all_says_so_rather_than_raising(caplog):
    frame = _frame(("vix9d", float("nan"), float("nan")))
    with caplog.at_level("INFO"):
        tool._report_most_damaging(frame)
    assert "not enough finite deltas to rank the signals" in caplog.text


def test_each_model_is_ranked_on_its_own_deltas(caplog):
    """Two models in one table are two ablations, not one pooled ranking."""
    frame = pd.concat([_frame(("vix9d", -0.30, 0.40), ("rv_5", -0.02, 0.01)),
                       _frame(("vix9d", -0.01, 0.01),
                              ("rv_5", -0.50, 0.90)).assign(model="snn")])
    with caplog.at_level("INFO"):
        tool._report_most_damaging(frame)
    assert "lstm: vix9d is the costliest" in caplog.text
    assert "snn: rv_5 is the costliest" in caplog.text


def test_a_model_that_cannot_be_ablated_is_still_refused():
    """Unrelated to the reporting change, but the guard this tool rests on."""
    with pytest.raises(ValueError, match="har_rv"):
        tool.check_ablatable(["lstm", "har_rv"])
