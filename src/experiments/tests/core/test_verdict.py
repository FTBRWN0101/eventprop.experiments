"""The verdict block: two co-primary metrics, and what it says when they split.

``r2_rv`` has been the headline throughout, but QLIKE is the loss the volatility
forecasting literature scores on and ``core/metrics.py`` has written it to every
``metrics.csv`` since the harness was built. The two really do split on runs already on
disk: 20260819-200058 puts ``snn`` first on ``r2_rv`` at 0.6155 and ``har_rv_direct``
first on QLIKE at 0.3479, and on the monthly ``rv_fwd`` cell the best SNN takes
``r2_rv`` 0.166 against persistence's 0.126 and then loses QLIKE 0.93 to 0.525. So the
cases here are: both winners named, a split stated as a split, and the values QLIKE can
take that cannot be ranked at all.

QLIKE's unrankable values are not hypothetical. ``qlike()`` returns ``+inf`` when a
forecast puts vol far enough above the truth that the log term wins, and ``NaN`` when
the squared forecast underflows to zero or no scored row has positive vol on both
sides. ``metrics.py`` is deliberately left computing exactly that, so the refusal has to
happen in the reporting.
"""

import numpy as np
import pytest

from core.config import ExperimentConfig
from run_experiment import _best_on, _headline_note, _print_verdict, _unrankable

CONFIG = ExperimentConfig.load(target="vrp")


def scores(**models):
    """A results dict from ``name=(r2_rv, qlike)`` pairs, plus the columns the block reads.

    ``gate`` is positive so every model clears it; the gate line is not what these are
    about, but ``_print_verdict`` reads it and a missing key would be a different test.
    """
    return {name: {"n": 302.0, "r2_rv": r2, "qlike": q, "gate": 0.1}
            for name, (r2, q) in models.items()}


def verdict_lines(results, caplog, config=CONFIG):
    """The verdict block as the lines it emitted, without caplog's own record prefix."""
    caplog.clear()
    with caplog.at_level("INFO"):
        _print_verdict(config, results, None)
    return list(caplog.messages)


def verdict_text(results, caplog, config=CONFIG):
    """The verdict block as one string, for asserting on what it actually said."""
    return "\n".join(verdict_lines(results, caplog, config))


def flat(text):
    """*text* with every run of whitespace collapsed, so a wrapped sentence still matches."""
    return " ".join(text.split())


def test_qlike_is_ranked_the_other_way_round_from_r2_rv():
    #the whole point of carrying the direction with the metric: lower QLIKE wins
    results = scores(a=(0.1, 0.9), b=(0.2, 0.5))
    assert _best_on(results, "r2_rv", True) == ("b", 0.2)
    assert _best_on(results, "qlike", False) == ("b", 0.5)
    assert _best_on(scores(a=(0.9, 0.9), b=(0.1, 0.5)), "qlike", False) == ("b", 0.5)


def test_both_headline_metrics_are_reported_with_their_winners(caplog):
    #the real monthly rv_fwd cell: garch takes both, snn is nowhere on either
    text = verdict_text(scores(snn=(0.136, 1.0567), har_rv=(0.4462, 0.3577),
                               garch=(0.5563, 0.2675), constant=(0.0, 1.3261),
                               persistence=(0.1260, 0.5249)), caplog)
    assert "best r2_rv" in text and "best qlike" in text
    assert "garch 0.5563" in text and "garch 0.2675" in text
    assert "higher is better" in text and "lower is better" in text


def test_agreement_is_stated_rather_than_left_to_be_inferred(caplog):
    text = verdict_text(scores(garch=(0.5563, 0.2675), snn=(0.136, 1.0567)), caplog)
    assert "agree: garch is first on both" in text
    assert "DISAGREE" not in text


def test_a_disagreement_is_stated_plainly(caplog):
    #run 20260819-200058: snn wins r2_rv, har_rv_direct wins QLIKE, by a hair
    results = scores(snn=(0.6155, 0.3505), har_rv=(0.4998, 0.4854),
                     har_rv_direct=(0.5798, 0.3479), constant=(0.5953, 0.3571))
    text = flat(verdict_text(results, caplog))
    assert "DISAGREE" in text
    #both winners are named, and each is given the other's number so the size of the
    #split is on the page rather than left for the reader to look up
    assert "r2_rv ranks snn first at 0.615" in text
    assert "QLIKE ranks har_rv_direct first at 0.3479" in text
    assert "snn scores QLIKE 0.3505 against har_rv_direct's 0.3479" in text
    assert "Neither metric is the tie-break" in text


def test_the_headline_case_that_started_this_reverses_between_metrics(caplog):
    #monthly rv_fwd, best SNN cell against naive persistence and nothing else: r2_rv
    #says the SNN wins, QLIKE says it loses, and the block has to say so
    text = flat(verdict_text(scores(snn=(0.166, 0.93),
                                    persistence=(0.126, 0.525)), caplog))
    assert "DISAGREE" in text
    assert "r2_rv ranks snn first at 0.166" in text
    assert "QLIKE ranks persistence first at 0.5250" in text


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_an_unrankable_qlike_is_named_and_left_out_of_the_ranking(bad, caplog):
    results = scores(good=(0.1, 0.8), broken=(0.9, bad))
    #-inf would win a naive min() and NaN can win one too, depending on the order the
    #models happen to be in, so neither may reach the comparison at all
    assert _best_on(results, "qlike", False) == ("good", 0.8)
    assert _unrankable(results, "qlike") == [f"broken ({bad:.4g})"]
    text = flat(verdict_text(results, caplog))
    assert "qlike not rankable broken" in text
    assert "excluded from the ranking above" in text
    assert "best qlike good 0.8000" in text


def test_a_missing_qlike_column_is_reported_as_absent(caplog):
    #a results tree written before the metric existed, not a metric that blew up
    results = {"a": {"n": 10.0, "r2_rv": 0.5, "gate": 0.1}}
    assert _unrankable(results, "qlike") == ["a (absent)"]
    assert _best_on(results, "qlike", False) is None
    assert "cannot be compared" in flat(verdict_text(results, caplog))


def test_no_rankable_qlike_at_all_refuses_to_pick_a_winner(caplog):
    results = scores(a=(0.5, float("nan")), b=(0.2, float("nan")))
    text = flat(verdict_text(results, caplog))
    assert "best qlike none scored (lower is better)" in text
    assert "no model has a rankable qlike" in text
    #r2_rv is still perfectly rankable, so it is still reported
    assert "best r2_rv a 0.5000 (higher is better)" in text


def test_the_disagreement_note_survives_a_nan_on_the_other_metric():
    #b wins qlike; a wins r2_rv. b's own r2_rv is NaN, and _finite must not crash the
    #sentence that quotes it
    results = scores(a=(0.5, 0.9), b=(float("nan"), 0.4))
    note = _headline_note(results)
    assert note.startswith("DISAGREE")
    assert "-inf" in note or "inf" in note


def test_the_gate_and_dm_lines_still_report_after_the_metric_lines(caplog):
    results = scores(snn=(0.6, 0.3), har_rv=(0.5, 0.4), constant=(0.0, 0.5))
    results["snn"]["dm_har_rv_rv_lag0"] = 2.6
    results["snn"]["dm_har_rv_rv_lag0_p"] = 0.009
    results["constant"]["gate"] = 0.0
    text = flat(verdict_text(results, caplog))
    assert "clears the gate snn, har_rv" in text
    assert "beats har_rv snn (+2.600, p=0.009)" in text


def test_an_empty_result_set_prints_nothing(caplog):
    assert verdict_lines({}, caplog) == []


def test_a_long_note_wraps_under_the_label_rather_than_running_off(caplog):
    lines = verdict_lines(scores(snn=(0.6155, 0.3505),
                                 har_rv_direct=(0.5798, 0.3479)), caplog)
    body = [line for line in lines if line.startswith("  ")]
    assert body, "the verdict emitted no labelled lines"
    assert max(len(line) for line in body) <= 96, "a line ran past the wrap width"
    #the continuation lines line up under the text column, not under the label
    assert [line for line in body if line.startswith(" " * 25)], "the note did not wrap"


def test_qlike_marks_invalid_forecasts_non_rankable():
    """The reporting guards are for values metrics.py really produces, not for show."""
    from core.metrics import qlike

    with np.errstate(divide="ignore", invalid="ignore"):
        #every forecast non-positive: no row survives the mask
        assert np.isnan(qlike(np.array([1.0, 2.0]), np.array([0.0, -1.0])))
        #a forecast of near-zero vol: its squared value underflows, so it is invalid.
        assert np.isnan(qlike(np.array([1.0, 2.0]), np.array([1e-200, 1.0])))
        #and an infinite forecast is also not rankable.
        assert np.isnan(qlike(np.array([1.0, 2.0]), np.array([np.inf, 1.0])))
