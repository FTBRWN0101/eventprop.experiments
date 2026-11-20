"""End-to-end runs through the CLI: argv -> config -> data -> models -> metrics -> disk.

Baselines only. The SNN needs the mlgenn env and is covered by the gpu-marked tests.
"""

import json

import numpy as np
import pandas as pd
import pytest

import run_experiment
from core import results as results_module

BASELINES = "har_rv,constant,persistence"
#both benchmarks, so DM is reported against the stronger econometric baseline too
BOTH_BENCHMARKS = "har_rv,garch,constant,persistence"

#lag 0 and lag h-1, in target space and rv space, each with its p-value.
#One such block per benchmark in BENCHMARKS.
DM_COLUMNS = tuple(f"dm_har_rv_{space}_{lag}{suffix}"
                   for space in ("target", "rv")
                   for lag in ("lag0", "lagh")
                   for suffix in ("", "_p"))


@pytest.fixture
def results_root(tmp_path, monkeypatch):
    """Redirect run output into tmp_path so a test run cannot pollute results/."""
    monkeypatch.setattr(results_module, "RESULTS_ROOT", tmp_path)
    return tmp_path


def _only_run_dir(root):
    runs = [p for p in root.iterdir() if p.is_dir()]
    assert len(runs) == 1, f"expected exactly one run directory, got {runs}"
    return runs[0]


def test_baseline_run_writes_a_complete_result(results_root):
    run_experiment.main(["--models", BASELINES])
    run_dir = _only_run_dir(results_root)

    config = json.loads((run_dir / "config.json").read_text(encoding="utf8"))
    assert config["target"] == "vrp" and config["iv_leg"] == "vix9d"
    assert config["model"] == "multiple"
    assert config["requested_models"] == BASELINES.split(",")
    assert config["scored_models"] == BASELINES.split(",")

    metrics = pd.read_csv(run_dir / "metrics.csv", index_col="model")
    assert set(metrics.index) == {"har_rv", "constant", "persistence"}
    for column in ("n", "r2_rv", "gate", "mse", "qlike", "mz_r2", "dispersion"):
        assert column in metrics.columns
    #DM is reported at both lags in both spaces, and nothing is picked for you
    for column in DM_COLUMNS:
        assert column in metrics.columns, column
        assert metrics.loc[["constant", "persistence"], column].notna().all()
        assert pd.isna(metrics.loc["har_rv", column]), "the benchmark has no DM vs self"
    assert metrics.loc["har_rv", "n"] > 0

    predictions = pd.read_csv(run_dir / "predictions.csv", index_col="date")
    assert "actual" in predictions.columns
    assert set(BASELINES.split(",")) <= set(predictions.columns)


def test_run_directory_name_identifies_the_arm(results_root):
    run_experiment.main(["--models", "constant", "--seed", "7", "--input-window", "20"])
    run_dir = _only_run_dir(results_root)
    name = run_dir.name
    assert "_L20_" in name and name.endswith("_s7")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf8"))
    assert config["model"] == "constant"
    assert config["requested_models"] == ["constant"]
    assert config["scored_models"] == ["constant"]


def test_campaign_id_is_persisted_and_rejects_path_syntax(results_root):
    run_experiment.main(["--models", "constant", "--campaign-id", "final-20260831"])
    config = json.loads((_only_run_dir(results_root) / "config.json").read_text(encoding="utf8"))
    assert config["campaign_id"] == "final-20260831"
    with pytest.raises(ValueError, match="campaign_id"):
        run_experiment.main(["--models", "constant", "--campaign-id", "../outside"])


def test_rv_fwd_arm_runs_and_degrades_gracefully(results_root):
    run_experiment.main(["--models", BASELINES, "--target", "rv_fwd"])
    metrics = pd.read_csv(_only_run_dir(results_root) / "metrics.csv", index_col="model")
    #identity conversion: gate and r2_rv are the same quantity here
    assert (metrics["gate"] - metrics["r2_rv"]).abs().max() < 1e-9
    #and direction is undefined on a strictly positive target
    assert metrics["dir_acc"].isna().all()


def test_variance_arm_runs(results_root):
    run_experiment.main(["--models", "har_rv,constant", "--target", "vvrp"])
    metrics = pd.read_csv(_only_run_dir(results_root) / "metrics.csv", index_col="model")
    #variance units are ~100x the vol-space ones, so this is a scale check, not a bug
    assert metrics.loc["har_rv", "mse"] > metrics.loc["har_rv", "n"]


def test_surface_arm_costs_test_rows_but_still_runs(results_root):
    run_experiment.main(["--models", "har_rv",
                         "--feature-set", "options+price+surface"])
    metrics = pd.read_csv(_only_run_dir(results_root) / "metrics.csv", index_col="model")
    assert 0 < metrics.loc["har_rv", "n"] < 302


def test_every_model_fits_from_the_same_start(results_root):
    run_experiment.main(["--models", BASELINES])
    diagnostics = json.loads(
        (_only_run_dir(results_root) / "diagnostics.json").read_text(encoding="utf8"))
    starts = {name: diag["fitted_range"][0]
              for name, diag in diagnostics.items() if "fitted_range" in diag}
    assert len(set(starts.values())) == 1, f"D27 violated: {starts}"


def test_monthly_horizon_rejects_the_vix9d_leg():
    with pytest.raises(ValueError, match="monthly panel has no VIX9D"):
        run_experiment.main(["--horizon", "monthly", "--models", "constant"])


def test_both_dm_lags_are_reported_and_differ(results_root):
    run_experiment.main(["--models", BASELINES])
    metrics = pd.read_csv(_only_run_dir(results_root) / "metrics.csv", index_col="model")
    lag0 = metrics.loc["persistence", "dm_har_rv_target_lag0"]
    lagh = metrics.loc["persistence", "dm_har_rv_target_lagh"]
    #vol is persistent, so the HAC window does something even on disjoint rows
    assert not np.isclose(lag0, lagh), "reporting both lags is pointless if they agree"


def test_dm_columns_agree_across_spaces_on_a_vrp_target(results_root):
    #vrp's map is affine with a unit coefficient on the same iv the prediction uses,
    #so the target-space and rv-space error series are negatives of each other
    run_experiment.main(["--models", BASELINES])
    metrics = pd.read_csv(_only_run_dir(results_root) / "metrics.csv", index_col="model")
    for suffix in ("lag0", "lagh"):
        assert np.allclose(metrics[f"dm_har_rv_target_{suffix}"].dropna(),
                           metrics[f"dm_har_rv_rv_{suffix}"].dropna())


def test_the_unrestricted_benchmarks_run_and_break_the_shared_start(results_root):
    run_experiment.main(["--models", "har_rv,har_rv_full,garch_full"])
    run_dir = _only_run_dir(results_root)
    metrics = pd.read_csv(run_dir / "metrics.csv", index_col="model")
    assert set(metrics.index) == {"har_rv", "har_rv_full", "garch_full"}
    assert metrics["n"].nunique() == 1, "the full arms must score the same test rows"

    diagnostics = json.loads((run_dir / "diagnostics.json").read_text(encoding="utf8"))
    starts = {name: diag["fitted_range"][0]
              for name, diag in diagnostics.items() if "fitted_range" in diag}
    #deliberately NOT sample-matched: that is the entire point of these two arms
    assert starts["har_rv_full"].startswith("1990")
    assert starts["garch_full"].startswith("1990")
    assert starts["har_rv"] != starts["har_rv_full"]

    #and because they are not sample-matched, their DM against har_rv is refused
    suppressed = diagnostics["__dm_suppressed__"]["har_rv"]
    assert set(suppressed) == {"har_rv_full", "garch_full"}
    for column in DM_COLUMNS:
        assert metrics.loc[["har_rv_full", "garch_full"], column].isna().all(), column


def test_garch_is_a_benchmark_in_its_own_right(results_root):
    """D27 leaves garch the stronger baseline, so DM must be reported against it too."""
    run_experiment.main(["--models", BOTH_BENCHMARKS])
    metrics = pd.read_csv(_only_run_dir(results_root) / "metrics.csv", index_col="model")
    for column in DM_COLUMNS:
        assert column.replace("har_rv", "garch") in metrics.columns, column
    #each benchmark has no DM against itself, and a real one against the other
    assert pd.isna(metrics.loc["garch", "dm_garch_rv_lag0"])
    assert pd.notna(metrics.loc["garch", "dm_har_rv_rv_lag0"])
    assert pd.notna(metrics.loc["har_rv", "dm_garch_rv_lag0"])
    #and the two are the same comparison from opposite sides
    assert np.isclose(metrics.loc["har_rv", "dm_garch_rv_lag0"],
                      -metrics.loc["garch", "dm_har_rv_rv_lag0"])


def test_the_validation_split_is_scored_on_validation_dates(results_root):
    """--evaluation-split val must score 2017-2019, not the 2020 test window.

    garch forecast from the test boundary regardless of the split it was asked for,
    so every val forecast was NaN and the strict scoring guard killed the whole run.
    """
    run_experiment.main(["--models", BOTH_BENCHMARKS, "--target", "rv_fwd",
                         "--holdout-val", "--evaluation-split", "val"])
    run_dir = _only_run_dir(results_root)

    config = json.loads((run_dir / "config.json").read_text(encoding="utf8"))
    assert config["evaluation_split"] == "val"
    assert config["scored_models"] == BOTH_BENCHMARKS.split(",")

    predictions = pd.read_csv(run_dir / "predictions.csv", parse_dates=["date"])
    assert predictions["date"].min().year == 2017
    assert predictions["date"].max().year < 2020
    metrics = pd.read_csv(run_dir / "metrics.csv", index_col="model")
    assert metrics.loc["garch", "n"] > 0


def test_one_model_failing_does_not_discard_the_others(results_root, monkeypatch):
    """A broken model leaves the comparison. The run is still written, and still fails."""
    from models.garch import GarchForecaster

    def explode(self, data):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(GarchForecaster, "fit", explode)

    with pytest.raises(SystemExit):
        run_experiment.main(["--models", BOTH_BENCHMARKS, "--target", "rv_fwd"])

    run_dir = _only_run_dir(results_root)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf8"))
    #requested records the ask, scored records what survived, so the gap is visible
    assert config["requested_models"] == BOTH_BENCHMARKS.split(",")
    assert "garch" not in config["scored_models"]

    metrics = pd.read_csv(run_dir / "metrics.csv", index_col="model")
    assert "garch" not in metrics.index
    assert {"har_rv", "constant", "persistence"} <= set(metrics.index)

    diagnostics = json.loads((run_dir / "diagnostics.json").read_text(encoding="utf8"))
    assert "synthetic failure" in diagnostics["__failed__"]["garch"]
