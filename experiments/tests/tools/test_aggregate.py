"""The seed aggregator: does a group mean rest on the seeds it claims to?"""

import json
import math

import pandas as pd
import pytest

from tools import aggregate as tool

#a flat config of the shape core.results.save_run writes
BASE_CONFIG = {
    "repo_root": "C:\\Users\\user\\UNI\\eventprop.experiments",
    "horizon": "weekly", "target": "vrp", "iv_leg": "vix9d",
    "feature_set": "options+price", "model": "snn", "encoding": "rate",
    "algorithm": "eventprop", "test_sampling": "nonoverlap", "seed": 1,
    "num_epochs": 50, "learning_rate": 0.001, "w_in_scale": 2.5,
    "drop_features": [], "restore_features": [],
}


def _write_run(root, name, metrics, diagnostics=None, **config):
    """Write one synthetic run directory the way core.results.save_run does."""
    run = root / name
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({**BASE_CONFIG, **config}),
                                     encoding="utf8")
    pd.DataFrame(metrics).T.to_csv(run / "metrics.csv", index_label="model")
    if diagnostics is not None:
        (run / "diagnostics.json").write_text(json.dumps(diagnostics), encoding="utf8")
    return run


def _metrics(**values):
    """One model row, with the n every real metrics.csv carries."""
    return {"snn": {"n": 302.0, **values}}


def _sweep(root, values, key="r2_rv", **config):
    """One run per (seed, value) pair, all in the same group."""
    for seed, value in values:
        _write_run(root, f"run_s{seed}", _metrics(**{key: value}), seed=seed, **config)
    return tool.aggregate(tool.load_runs(root), min_seeds=1)


def test_grouping_ignores_seed_and_repo_root(tmp_path):
    for seed, repo in ((1, "/home/a"), (2, "D:\\b"), (3, "/mnt/c")):
        _write_run(tmp_path, f"run{seed}", _metrics(r2_rv=0.5), seed=seed, repo_root=repo)
    frame = tool.aggregate(tool.load_runs(tmp_path), min_seeds=1)
    assert frame["group"].nunique() == 1
    assert frame["n_seeds"].tolist() == [3]


def test_any_other_config_field_starts_a_new_group(tmp_path):
    _write_run(tmp_path, "a", _metrics(r2_rv=0.5), seed=1, learning_rate=0.001)
    _write_run(tmp_path, "b", _metrics(r2_rv=0.5), seed=2, learning_rate=0.01)
    frame = tool.aggregate(tool.load_runs(tmp_path), min_seeds=1)
    assert frame["group"].nunique() == 2
    #the field that split them is named in the table, so the groups can be told apart
    assert sorted(frame["learning_rate"]) == [0.001, 0.01]


def test_group_identity_comes_from_the_config_not_the_directory_name(tmp_path):
    #names that disagree about the arm, configs that agree: one group
    _write_run(tmp_path, "20260818-120000_weekly_vrp_vix9d_rate_eventprop_L5_s1",
               _metrics(r2_rv=0.4), seed=1)
    _write_run(tmp_path, "20260818-120001_monthly_rvrp_vix_delta_eprop_L45_s2",
               _metrics(r2_rv=0.6), seed=2)
    frame = tool.aggregate(tool.load_runs(tmp_path), min_seeds=1)
    assert frame["group"].nunique() == 1
    assert frame["r2_rv_mean"].iloc[0] == pytest.approx(0.5)


def test_matching_names_still_split_when_the_configs_differ(tmp_path):
    #the name never shows num_epochs, so only config.json can tell these apart
    _write_run(tmp_path, "20260818-120000_weekly_vrp_vix9d_rate_eventprop_L5_s1",
               _metrics(r2_rv=0.4), seed=1, num_epochs=50)
    _write_run(tmp_path, "20260818-130000_weekly_vrp_vix9d_rate_eventprop_L5_s1",
               _metrics(r2_rv=0.6), seed=2, num_epochs=200)
    assert tool.aggregate(tool.load_runs(tmp_path), min_seeds=1)["group"].nunique() == 2


def test_mean_and_std_are_the_sample_statistics_over_seeds(tmp_path):
    row = _sweep(tmp_path, ((1, 0.1), (2, 0.2), (3, 0.6))).iloc[0]
    assert row["r2_rv_mean"] == pytest.approx(0.3)
    #sqrt(((0.1-0.3)^2 + (0.2-0.3)^2 + (0.6-0.3)^2) / 2) = sqrt(0.07)
    assert row["r2_rv_std"] == pytest.approx(0.2645751311064591)
    assert row["r2_rv_count"] == 3


def test_a_single_seed_has_no_sample_std(tmp_path):
    row = _sweep(tmp_path, ((1, 0.42),)).iloc[0]
    assert row["r2_rv_mean"] == pytest.approx(0.42)
    assert math.isnan(row["r2_rv_std"]), "ddof=1 on one observation is undefined"
    assert row["r2_rv_count"] == 1


def test_a_metric_that_is_nan_for_every_seed_aggregates_to_nan(tmp_path):
    nan = float("nan")
    row = _sweep(tmp_path, ((1, nan), (2, nan), (3, nan)), key="dm_lag0").iloc[0]
    assert math.isnan(row["dm_lag0_mean"])
    assert math.isnan(row["dm_lag0_std"])
    assert row["dm_lag0_count"] == 0
    #the rest of the row is untouched by the dead column
    assert row["n_mean"] == pytest.approx(302.0)


def test_a_partly_nan_metric_averages_only_the_seeds_that_have_it(tmp_path):
    row = _sweep(tmp_path, ((1, 2.0), (2, 4.0), (3, float("nan"))),
                 key="dm_lagh").iloc[0]
    assert row["dm_lagh_mean"] == pytest.approx(3.0)
    #sqrt(((2-3)^2 + (4-3)^2) / 1) = sqrt(2), over the two seeds present
    assert row["dm_lagh_std"] == pytest.approx(math.sqrt(2.0))
    assert row["dm_lagh_count"] == 2, "the count must not claim the missing seed"
    assert row["n_count"] == 3 and row["n_seeds"] == 3


def test_dm_columns_aggregate_like_any_other_metric(tmp_path):
    for seed, value in ((1, 1.0), (2, 3.0)):
        _write_run(tmp_path, f"run{seed}",
                   _metrics(r2_rv=0.5, dm_rv_lag0=value, dm_rv_lag0_p=value / 10),
                   seed=seed)
    row = tool.aggregate(tool.load_runs(tmp_path), min_seeds=1).iloc[0]
    assert row["dm_rv_lag0_mean"] == pytest.approx(2.0)
    assert row["dm_rv_lag0_std"] == pytest.approx(math.sqrt(2.0))
    assert row["dm_rv_lag0_p_mean"] == pytest.approx(0.2)
    assert row["dm_rv_lag0_p_count"] == 2


def test_a_model_missing_from_one_seed_counts_only_where_it_ran(tmp_path):
    _write_run(tmp_path, "a", {"snn": {"r2_rv": 0.4}, "har_rv": {"r2_rv": 0.2}}, seed=1)
    _write_run(tmp_path, "b", {"snn": {"r2_rv": 0.6}}, seed=2)
    frame = tool.aggregate(tool.load_runs(tmp_path), min_seeds=1).set_index("model")
    assert frame.at["snn", "r2_rv_count"] == 2
    assert frame.at["har_rv", "r2_rv_count"] == 1
    assert frame.at["har_rv", "r2_rv_mean"] == pytest.approx(0.2)


def test_a_group_below_the_seed_floor_warns(tmp_path, caplog):
    for seed in (1, 2):
        _write_run(tmp_path, f"run{seed}", _metrics(r2_rv=0.5), seed=seed)
    runs = tool.load_runs(tmp_path)
    with caplog.at_level("WARNING"):
        tool.aggregate(runs, min_seeds=8)
    assert "2 seed(s) [1, 2]" in caplog.text
    assert "fewer than the 8" in caplog.text


def test_a_group_at_the_seed_floor_is_quiet(tmp_path, caplog):
    for seed in range(1, 9):
        _write_run(tmp_path, f"run{seed}", _metrics(r2_rv=0.5), seed=seed)
    runs = tool.load_runs(tmp_path)
    with caplog.at_level("WARNING"):
        tool.aggregate(runs, min_seeds=8)
    assert caplog.text == ""


def test_a_repeated_seed_warns_about_double_counting(tmp_path, caplog):
    for name, seed in (("a", 1), ("b", 1), ("c", 2)):
        _write_run(tmp_path, name, _metrics(r2_rv=0.5), seed=seed)
    runs = tool.load_runs(tmp_path)
    with caplog.at_level("WARNING"):
        frame = tool.aggregate(runs, min_seeds=1)
    assert "repeats seed(s) [1] over 3 runs" in caplog.text
    #the row shows both numbers, so the double count is visible in the table too
    assert frame["n_seeds"].iloc[0] == 2 and frame["n_runs"].iloc[0] == 3


def test_malformed_runs_are_skipped_rather_than_fatal(tmp_path, caplog):
    good = _write_run(tmp_path, "good", _metrics(r2_rv=0.5), seed=1)
    (tmp_path / "bad_json").mkdir()
    (tmp_path / "bad_json" / "config.json").write_text("{not json", encoding="utf8")
    (tmp_path / "bad_json" / "metrics.csv").write_text("model,r2_rv\nsnn,0.5\n",
                                                       encoding="utf8")
    (tmp_path / "no_metrics").mkdir()
    (tmp_path / "no_metrics" / "config.json").write_text("{}", encoding="utf8")
    with caplog.at_level("WARNING"):
        runs = tool.load_runs(tmp_path)
    assert [run.path for run in runs] == [good]
    assert "bad_json" in caplog.text and "no_metrics" in caplog.text


def test_silent_final_is_the_mean_last_checkpoint_over_seeds(tmp_path):
    for seed, series in ((1, [0.5, 0.3, 0.2]), (2, [0.1, 0.4])):
        _write_run(tmp_path, f"run{seed}", _metrics(r2_rv=0.5), seed=seed,
                   diagnostics={"snn": {"silent_neuron_fraction": series}})
    row = tool.aggregate(tool.load_runs(tmp_path), min_seeds=1).iloc[0]
    #last values are 0.2 and 0.4
    assert row["silent_final"] == pytest.approx(0.3)


def test_silent_final_is_absent_when_no_run_records_it(tmp_path):
    _write_run(tmp_path, "run1", _metrics(r2_rv=0.5), seed=1,
               diagnostics={"snn": {"fitted_range": ["2011-01-04", "2019-12-31"]}})
    assert "silent_final" not in tool.aggregate(tool.load_runs(tmp_path), min_seeds=1)


def test_main_writes_every_metric_but_prints_only_the_asked_for_ones(tmp_path, caplog):
    for seed in (1, 2):
        _write_run(tmp_path, f"run{seed}",
                   _metrics(r2_rv=0.5, gate=-0.1, qlike=0.4), seed=seed)
    out = tmp_path / "agg.csv"
    with caplog.at_level("INFO"):
        tool.main(["--results-root", str(tmp_path), "--out", str(out),
                   "--min-seeds", "1", "--metrics", "r2_rv"])
    written = pd.read_csv(out)
    assert {"r2_rv_mean", "gate_mean", "qlike_mean"} <= set(written.columns)
    assert "r2_rv_mean" in caplog.text and "qlike_mean" not in caplog.text


def test_main_warns_on_a_metric_no_run_carries(tmp_path, caplog):
    _write_run(tmp_path, "run1", _metrics(r2_rv=0.5), seed=1)
    with caplog.at_level("WARNING"):
        tool.main(["--results-root", str(tmp_path), "--out", str(tmp_path / "a.csv"),
                   "--min-seeds", "1", "--metrics", "r2_rv,nonesuch"])
    assert "nonesuch" in caplog.text


def test_main_on_an_empty_root_warns_rather_than_raising(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        tool.main(["--results-root", str(tmp_path)])
    assert "no run under" in caplog.text
