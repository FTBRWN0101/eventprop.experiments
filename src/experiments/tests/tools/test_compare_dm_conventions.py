"""The DM convention comparison: does it actually produce the four conventions?"""

import numpy as np
import pytest

from core import metrics
from core.config import ExperimentConfig
from tools import compare_dm_conventions as tool


@pytest.fixture(scope="module")
def frame():
    return tool.compare(ExperimentConfig.load())


def test_every_model_and_space_gets_all_four_conventions(frame):
    assert set(frame["space"]) == {"target", "rv_fwd"}
    assert set(frame["model"]) == {"constant", "persistence"}
    for (space, model), block in frame.groupby(["space", "model"]):
        assert len(block) == 4, (space, model)
    assert frame["stat"].notna().all() and frame["p"].notna().all()


def test_the_benchmark_is_not_compared_against_itself(frame):
    assert tool.BENCHMARK not in set(frame["model"])


def test_conventions_are_labelled_with_their_lag_and_distribution():
    labels = list(tool.conventions(4))
    assert labels[0].startswith("old") and "lag=4" in labels[0] and "normal" in labels[0]
    assert any("lag=0" in label and "HLN+t" in label for label in labels)


def test_the_old_convention_reproduces_the_uncorrected_statistic(frame):
    #the point of the tool is that "old" is the pre-D39/D50 code path, so it must be
    #exactly the normal-distribution, lag h-1 number, not an approximation of it
    errors, _ = tool.error_series(ExperimentConfig.load(), tool.MODELS)
    expected, _ = metrics.diebold_mariano(
        errors[tool.BENCHMARK], errors["constant"], lag=4, hln=False)
    row = frame[(frame["space"] == "target") & (frame["model"] == "constant")
                & (frame["convention"].str.startswith("old"))].iloc[0]
    assert row["stat"] == pytest.approx(expected)


def test_hln_shrinks_the_statistic_at_every_lag(frame):
    for model in ("constant", "persistence"):
        block = frame[(frame["space"] == "target") & (frame["model"] == model)]
        by_label = dict(zip(block["convention"], block["stat"]))
        assert abs(by_label["new   lag=0 HLN+t"]) < abs(by_label["D39   lag=0 normal"])
        assert abs(by_label["new   lag=4 HLN+t"]) < abs(by_label["old   lag=4 normal"])


def test_the_two_spaces_agree_on_a_vrp_target(frame):
    #vrp = iv - rv_fwd is affine with a unit coefficient on the same iv the prediction
    #uses, so the target-space error is minus the rv-space error and the squares match.
    #This is why the extra DM columns are free here and are NOT on vvrp.
    target = frame[frame["space"] == "target"].set_index(["model", "convention"])["stat"]
    rv = frame[frame["space"] == "rv_fwd"].set_index(["model", "convention"])["stat"]
    assert np.allclose(target.to_numpy(), rv.reindex(target.index).to_numpy())


def test_the_two_spaces_disagree_on_a_variance_target():
    #vvrp squares the map, so the cancellation above does not happen there
    vvrp = tool.compare(ExperimentConfig.load(target="vvrp"), ("har_rv", "constant"))
    target = vvrp[vvrp["space"] == "target"]["stat"].to_numpy()
    rv = vvrp[vvrp["space"] == "rv_fwd"]["stat"].to_numpy()
    assert not np.allclose(target, rv)


def test_the_tool_runs_end_to_end(capsys):
    tool.main([])
    out = capsys.readouterr().out
    assert "target space" in out and "rv_fwd space" in out
    assert "HLN factor" in out
