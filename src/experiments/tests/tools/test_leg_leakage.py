"""The leg-leakage diagnostic: is the arithmetic behind the D28 verdict right?"""

import numpy as np
import pandas as pd
import pytest

from core.config import ExperimentConfig
from core.dataset import VrpDataset
from tools import leg_leakage as tool

BASE = dict(iv_leg="vix9d", feature_set="price-only")


@pytest.fixture(scope="module")
def vrp():
    return tool.diagnose(ExperimentConfig.load(target="vrp", **BASE))


@pytest.fixture(scope="module")
def rv():
    return tool.diagnose(ExperimentConfig.load(target="rv_fwd", **BASE))


def test_the_variance_shares_sum_to_one():
    index = pd.date_range("2020-01-01", periods=200)
    rng = np.random.default_rng(0)
    iv = pd.Series(rng.normal(20, 5, 200), index=index)
    rv_hat = pd.Series(rng.normal(15, 4, 200) + 0.5 * iv, index=index)
    out = tool.decompose(iv - rv_hat, iv, rv_hat)
    assert out["share_iv"] + out["share_rv_hat"] == pytest.approx(1.0)


def test_an_iv_free_prediction_puts_no_share_on_the_leg():
    #rv_hat tracks iv exactly and adds independent noise z, so pred = iv - rv_hat = z
    #carries none of the leg. This is the shape a genuinely leg-free arm would have.
    index = pd.date_range("2020-01-01", periods=1000)
    rng = np.random.default_rng(1)
    iv = pd.Series(rng.normal(20, 5, 1000), index=index)
    z = pd.Series(rng.normal(0, 2, 1000), index=index)
    out = tool.decompose(iv - (iv - z), iv, iv - z)
    assert abs(out["share_iv"]) < 0.15
    assert out["share_rv_hat"] == pytest.approx(1.0 - out["share_iv"])


def test_the_leg_cancels_out_of_the_loss_on_a_vrp_target(vrp):
    #the load-bearing claim of the whole diagnostic
    assert vrp["cancellation"]["max_abs_gap"] < 1e-8
    assert vrp["cancellation"]["mean_target"] == pytest.approx(
        vrp["cancellation"]["mean_rv"])


def test_the_leg_does_not_cancel_on_a_variance_target():
    data = VrpDataset(ExperimentConfig.load(target="vvrp", **BASE))
    split = data.split("test")
    pred = split.to_target(split.rv_fwd * 1.1)
    assert tool.cancellation(pred, data)["max_abs_gap"] > 1e-6


def test_the_constant_baseline_is_an_affine_map_of_the_vix_on_a_vrp_target(vrp):
    #this is the leak: the "zero-skill" benchmark is secretly a VIX-based RV forecast
    assert vrp["implied_corr_iv"]["constant"] == pytest.approx(1.0)


def test_the_constant_baseline_is_leg_free_on_an_rv_fwd_target(rv):
    #a flat forecast has no variance, so pandas corr gives NaN; 0.0 after the fillna
    #that Split.to_rv_fwd's identity path implies. Either way, not 1.0.
    assert not np.isclose(rv["implied_corr_iv"]["constant"], 1.0)


def test_the_target_choice_flips_the_constant_vs_har_ranking(vrp, rv):
    model = vrp["model"]
    assert vrp["scores"]["constant"]["r2_rv"] > vrp["scores"][model]["r2_rv"]
    assert rv["scores"]["constant"]["r2_rv"] < rv["scores"][model]["r2_rv"]


def test_the_model_scores_identically_in_rv_space_on_both_targets(vrp, rv):
    model = vrp["model"]
    assert vrp["scores"][model]["r2_rv"] == pytest.approx(rv["scores"][model]["r2_rv"])
    assert vrp["scores"][model]["mse"] == pytest.approx(rv["scores"][model]["mse"])


def test_the_oracle_is_perfect_and_the_trivial_model_is_the_constant_rv_forecast(vrp):
    assert vrp["scores"]["oracle_iv"]["mse"] == pytest.approx(0.0, abs=1e-18)
    assert vrp["scores"]["oracle_iv"]["r2_rv"] == pytest.approx(1.0)
    #the trivial model IS the train-mean rv constant, so its r2_rv is exactly zero
    assert vrp["scores"]["trivial_iv"]["r2_rv"] == pytest.approx(0.0, abs=1e-12)


def test_informativeness_is_r2_rv_by_construction(vrp):
    #stated in the tool as a consistency check; assert it so a change to either side
    #that breaks the identity is caught
    assert vrp["informativeness"] == pytest.approx(vrp["scores"][vrp["model"]]["r2_rv"])


def test_iv_is_largely_reconstructible_from_realised_vol_alone(vrp):
    rec = vrp["reconstruction"]
    assert 0.5 < rec["iv_r2"] < 1.0
    assert rec["iv_rmse"] > 0.0
    #stacking two separately fitted pieces cannot beat fitting them jointly
    assert rec["mse_direct_stacked"] > rec["mse_direct_joint"]


def test_the_verdict_names_the_cancellation_and_the_recommendation(vrp, rv):
    lines = tool._verdict(vrp, rv)
    assert any("cancels EXACTLY" in line for line in lines)
    assert any("FLIPS" in line for line in lines)
    assert lines[-1].startswith("VERDICT")


def test_the_tool_runs_end_to_end(capsys):
    tool.main([])
    out = capsys.readouterr().out
    assert "CANCELS" in out and "VERDICT" in out
