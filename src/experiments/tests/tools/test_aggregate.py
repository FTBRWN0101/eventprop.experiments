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


#--- attempted versus scored -------------------------------------------------------
#An arm that trips the D67 silent-neuron abort writes no result directory, so a cell
#that ran eight seeds and scored two is indistinguishable from a clean two-seed cell by
#any walk of the results tree. The counts therefore come from the sweep manifests, which
#record every arm with its returncode whether or not it produced anything. This is a
#real case, not a hypothetical: delta_adaptive at L=5 in sweep_manifest.json aborted six
#of its eight seeds, and its published mean rests on the two that survived.


def _write_manifest(root, name, records):
    """One sweep manifest of the shape tools/sweep.py writes."""
    path = root / name
    path.write_text(json.dumps(records), encoding="utf8")
    return path


def _arm(seed, run_dir="", returncode=0, **fields):
    """One manifest record: the arm's axes, its exit code, and where it wrote."""
    record = {"key": f"rate_eventprop_L5_s{seed}", "encoding": "rate",
              "algorithm": "eventprop", "input_window": 5, **fields, "seed": seed,
              "returncode": returncode, "seconds": 1.0, "run_dir": run_dir}
    if returncode != 0:
        #what the abort really leaves: no run_dir, and the breach in the error tail
        record["error"] = "RuntimeError: [snn] spike deletion: 13.7% of hidden neurons"
    return record


def _sweep_cell(root, scored, aborted, manifest="sweep_manifest.json", **config):
    """A sweep cell where *scored* seeds wrote runs and *aborted* seeds wrote nothing."""
    records = []
    for seed in scored:
        run = _write_run(root, f"run_s{seed}", _metrics(r2_rv=0.5, qlike=0.4),
                         seed=seed, **config)
        records.append(_arm(seed, run_dir=str(run)))
    records.extend(_arm(seed, returncode=1) for seed in aborted)
    _write_manifest(root, manifest, records)
    return records


def _row(tmp_path, attempts):
    """The first aggregated row, with the manifest-derived counts attached."""
    return tool.aggregate(tool.load_runs(tmp_path), min_seeds=1,
                          attempts=attempts).iloc[0]


def test_the_arm_identity_ignores_the_seed_and_the_per_run_bookkeeping():
    """Two seeds of one arm are one cell; a different window is a different cell."""
    one = tool._arm_identity(_arm(1, run_dir="C:/runs/a"))
    two = tool._arm_identity(_arm(2, run_dir="C:/runs/b", returncode=1))
    assert one == two
    assert one != tool._arm_identity(_arm(1, input_window=45))


def test_attempted_counts_the_aborted_seeds_the_results_tree_cannot_see(tmp_path):
    #the delta_adaptive L=5 case: 8 attempted, 2 scored, 6 gone without a directory
    _sweep_cell(tmp_path, scored=(1, 7), aborted=(2, 3, 4, 5, 6, 8))
    row = _row(tmp_path, tool.read_manifests(tmp_path))
    assert row["n_scored"] == 2 and row["n_attempted"] == 8
    assert row["n_aborted"] == 6
    assert row["missing_seeds"] == "2,3,4,5,6,8"
    #n_seeds keeps its historical meaning, so existing analysis still reads
    assert row["n_seeds"] == 2


def test_a_partial_cell_is_flagged_rather_than_silently_averaged(tmp_path, caplog):
    _sweep_cell(tmp_path, scored=(1, 7), aborted=(2, 3, 4, 5, 6, 8))
    attempts = tool.read_manifests(tmp_path)
    with caplog.at_level("WARNING"):
        tool.aggregate(tool.load_runs(tmp_path), min_seeds=1, attempts=attempts)
    assert "PARTIAL" in caplog.text
    assert "2 of 8 attempted seed(s) scored" in caplog.text
    assert "['2', '3', '4', '5', '6', '8']" in caplog.text


def test_a_complete_cell_is_not_flagged(tmp_path, caplog):
    _sweep_cell(tmp_path, scored=(1, 2, 3), aborted=())
    attempts = tool.read_manifests(tmp_path)
    with caplog.at_level("WARNING"):
        row = tool.aggregate(tool.load_runs(tmp_path), min_seeds=1,
                             attempts=attempts).iloc[0]
    assert "PARTIAL" not in caplog.text
    assert row["n_scored"] == 3 and row["n_attempted"] == 3
    assert row["n_aborted"] == 0 and row["missing_seeds"] == ""


def test_without_a_manifest_the_attempted_count_is_unknown_not_assumed(tmp_path):
    """Most of the results tree predates the sweep runner; guessing there would lie."""
    for seed in (1, 2):
        _write_run(tmp_path, f"run{seed}", _metrics(r2_rv=0.5), seed=seed)
    row = tool.aggregate(tool.load_runs(tmp_path), min_seeds=1).iloc[0]
    assert pd.isna(row["n_attempted"]) and pd.isna(row["n_aborted"])
    assert row["missing_seeds"] == "?"
    assert row["n_scored"] == 2


def test_a_group_no_manifest_covers_is_unknown_even_when_manifests_exist(tmp_path):
    _sweep_cell(tmp_path, scored=(1,), aborted=(2,))
    #a run from a different arm entirely, which no manifest record points at
    _write_run(tmp_path, "loose", _metrics(r2_rv=0.9), seed=1, num_epochs=999)
    frame = tool.aggregate(tool.load_runs(tmp_path), min_seeds=1,
                           attempts=tool.read_manifests(tmp_path))
    loose = frame[frame["num_epochs"] == 999].iloc[0]
    assert pd.isna(loose["n_attempted"]) and loose["missing_seeds"] == "?"


def test_a_scored_seed_the_manifest_never_recorded_still_counts_as_attempted(tmp_path):
    """n_attempted must never fall below n_scored, or n_aborted goes negative."""
    _sweep_cell(tmp_path, scored=(1,), aborted=(2,))
    #a hand-run repeat of the same arm, absent from the manifest
    _write_run(tmp_path, "hand_run", _metrics(r2_rv=0.6, qlike=0.4), seed=3)
    row = _row(tmp_path, tool.read_manifests(tmp_path))
    assert row["n_scored"] == 2 and row["n_attempted"] == 3
    assert row["missing_seeds"] == "2"


def test_a_cell_where_every_seed_aborted_is_reported_as_having_no_row(tmp_path):
    """delta at L=5 lost all eight seeds, so nothing in the table can carry the flag."""
    _sweep_cell(tmp_path, scored=(1, 2), aborted=(), manifest="a_manifest.json")
    _write_manifest(tmp_path, "b_manifest.json",
                    [_arm(seed, returncode=1, encoding="delta")
                     for seed in range(1, 9)])
    attempts = tool.read_manifests(tmp_path)
    assert [o["attempted"] for o in attempts.orphans] == [8]
    assert "delta" in attempts.orphans[0]["arm"]
    assert attempts.orphans[0]["manifest"] == "b_manifest.json"


def test_the_manifest_file_is_part_of_the_cell_identity(tmp_path):
    """window_fraction_0.05 and _0.25 record identical arm fields and differ only in a
    --readout-window the record never mentions. Keying on the fields alone would merge
    four separate sweeps into one cell.
    """
    for fraction, manifest in ((0.05, "w05_manifest.json"),
                               (0.25, "w25_manifest.json")):
        run = _write_run(tmp_path, f"run_{fraction}", _metrics(r2_rv=0.5, qlike=0.4),
                         seed=1, readout_window_fraction=fraction)
        _write_manifest(tmp_path, manifest,
                        [_arm(1, run_dir=str(run)), _arm(2, returncode=1)])
    frame = tool.aggregate(tool.load_runs(tmp_path), min_seeds=1,
                           attempts=tool.read_manifests(tmp_path))
    assert frame["group"].nunique() == 2
    #each sweep keeps its own failure rather than both inheriting two
    assert frame["n_attempted"].tolist() == [2, 2]
    assert frame["n_scored"].tolist() == [1, 1]


def test_a_moved_run_directory_is_found_again_by_name(tmp_path):
    """The manifest stores an absolute path, which dies when the repo moves machine -
    the same reason group_key ignores repo_root.
    """
    run = _write_run(tmp_path, "run_s1", _metrics(r2_rv=0.5, qlike=0.4), seed=1)
    _write_manifest(tmp_path, "sweep_manifest.json",
                    [_arm(1, run_dir="D:/an/old/machine/run_s1"),
                     _arm(2, returncode=1)])
    assert tool._resolve_run_dir("D:/an/old/machine/run_s1", tmp_path) == run
    row = _row(tmp_path, tool.read_manifests(tmp_path))
    assert row["n_scored"] == 1 and row["n_attempted"] == 2


def test_a_run_directory_that_is_gone_does_not_place_its_arm(tmp_path):
    _write_run(tmp_path, "kept", _metrics(r2_rv=0.5, qlike=0.4), seed=1)
    assert tool._resolve_run_dir("D:/gone/never_existed", tmp_path) is None
    assert tool._resolve_run_dir("", tmp_path) is None


def test_a_cell_spanning_two_config_groups_reports_unknown_rather_than_wrong(
        tmp_path, caplog):
    """The config schema gained a field partway through a sweep, so one sweep cell
    became two rows here and its failed seeds belong to neither.

    This is live in the results root: window_fraction_0.10's latency L=45 cell has two
    seeds written before ``reset_synapses_between_trials`` existed and six after.
    """
    for seed, epochs in ((1, 50), (2, 200)):
        _write_run(tmp_path, f"run_s{seed}", _metrics(r2_rv=0.5, qlike=0.4),
                   seed=seed, num_epochs=epochs)
    _write_manifest(tmp_path, "sweep_manifest.json",
                    [_arm(1, run_dir=str(tmp_path / "run_s1")),
                     _arm(2, run_dir=str(tmp_path / "run_s2")),
                     _arm(3, returncode=1)])
    with caplog.at_level("WARNING"):
        attempts = tool.read_manifests(tmp_path)
    assert "spans 2 config groups" in caplog.text
    assert "num_epochs" in caplog.text
    frame = tool.aggregate(tool.load_runs(tmp_path), min_seeds=1, attempts=attempts)
    assert frame["n_attempted"].isna().all()
    assert (frame["missing_seeds"] == "?").all()


def test_an_unreadable_manifest_is_skipped_rather_than_fatal(tmp_path, caplog):
    _sweep_cell(tmp_path, scored=(1,), aborted=(2,))
    (tmp_path / "broken_manifest.json").write_text("{not json", encoding="utf8")
    with caplog.at_level("WARNING"):
        attempts = tool.read_manifests(tmp_path)
    assert "broken_manifest.json" in caplog.text
    assert _row(tmp_path, attempts)["n_attempted"] == 2


def test_main_prints_the_partial_cells_and_the_orphans(tmp_path, caplog):
    _sweep_cell(tmp_path, scored=(1, 7), aborted=(2, 3, 4, 5, 6, 8))
    _write_manifest(tmp_path, "dead_manifest.json",
                    [_arm(s, returncode=1, encoding="delta") for s in (1, 2)])
    with caplog.at_level("INFO"):
        tool.main(["--results-root", str(tmp_path), "--out", str(tmp_path / "a.csv"),
                   "--min-seeds", "1"])
    assert "PARTIAL cells" in caplog.text
    assert "attempted counts from 2 manifest(s)" in caplog.text
    assert "scored nothing at all" in caplog.text
    assert "0 of 2 seed(s) scored" in caplog.text


def test_main_can_be_told_to_ignore_the_manifests(tmp_path, caplog):
    _sweep_cell(tmp_path, scored=(1, 7), aborted=(2, 3, 4, 5, 6, 8))
    with caplog.at_level("INFO"):
        tool.main(["--results-root", str(tmp_path), "--out", str(tmp_path / "a.csv"),
                   "--min-seeds", "1", "--manifests", "none"])
    assert "manifests skipped" in caplog.text
    assert "PARTIAL" not in caplog.text
    assert pd.read_csv(tmp_path / "a.csv")["n_attempted"].isna().all()


def test_named_manifests_are_read_instead_of_the_glob(tmp_path):
    _sweep_cell(tmp_path, scored=(1,), aborted=(2,), manifest="one_manifest.json")
    _write_manifest(tmp_path, "two_manifest.json",
                    [_arm(s, returncode=1) for s in (3, 4)])
    only = tool.read_manifests(tmp_path, [tmp_path / "one_manifest.json"])
    assert [p.name for p in only.manifests] == ["one_manifest.json"]
    assert _row(tmp_path, only)["n_attempted"] == 2


#--- QLIKE as a co-primary metric --------------------------------------------------


def test_qlike_is_summarised_by_the_median_because_its_mean_is_not_a_summary(tmp_path):
    """One arm forecasting near-zero vol drags the mean past every other seed."""
    row = _sweep(tmp_path, ((1, 1.0), (2, 1.4), (3, 1.5), (4, 26836.0)),
                 key="qlike").iloc[0]
    assert row["qlike_median"] == pytest.approx(1.45)
    assert row["qlike_mean"] == pytest.approx(6709.975)
    #the tail seed is a real under-dispersed forecaster, so it stays in the sample
    assert row["qlike_count"] == 4


def test_the_printed_table_gives_qlike_its_median_and_r2_rv_its_mean(tmp_path):
    _write_run(tmp_path, "run1", _metrics(r2_rv=0.5, qlike=0.4), seed=1)
    frame = tool.aggregate(tool.load_runs(tmp_path), min_seeds=1)
    shown = tool.shown_columns(frame, ["r2_rv", "qlike"])
    assert "qlike_median" in shown and "qlike_std" not in shown
    assert "r2_rv_mean" in shown and "r2_rv_median" not in shown
    #the mean is printed beside the median so the size of the tail stays visible
    assert "qlike_mean" in shown


def test_qlike_is_a_default_printed_metric(tmp_path, caplog):
    _write_run(tmp_path, "run1", _metrics(r2_rv=0.5, qlike=0.4), seed=1)
    with caplog.at_level("INFO"):
        tool.main(["--results-root", str(tmp_path), "--out", str(tmp_path / "a.csv"),
                   "--min-seeds", "1"])
    assert "qlike_median" in caplog.text


def test_the_headline_names_a_winner_on_each_metric(tmp_path):
    _write_run(tmp_path, "run1",
               {"snn": {"n": 302.0, "r2_rv": 0.6, "qlike": 0.9},
                "garch": {"n": 302.0, "r2_rv": 0.5, "qlike": 0.3}}, seed=1)
    row = tool.headline(tool.aggregate(tool.load_runs(tmp_path), min_seeds=1)).iloc[0]
    assert row["best_r2_rv"] == "snn" and row["best_qlike"] == "garch"
    assert row["headline"] == "DISAGREE"


def test_the_headline_says_so_when_the_metrics_agree(tmp_path):
    _write_run(tmp_path, "run1",
               {"snn": {"n": 302.0, "r2_rv": 0.1, "qlike": 0.9},
                "garch": {"n": 302.0, "r2_rv": 0.5, "qlike": 0.3}}, seed=1)
    assert tool.headline(tool.aggregate(tool.load_runs(tmp_path),
                                        min_seeds=1)).iloc[0]["headline"] == "agree"


def test_the_headline_refuses_a_qlike_that_cannot_be_ranked(tmp_path):
    """+inf is what QLIKE returns for a forecast far above the truth; it must not win a
    min(), and the group has to say the ranking is undecided rather than pick anyway.
    """
    _write_run(tmp_path, "run1",
               {"snn": {"n": 302.0, "r2_rv": 0.6, "qlike": float("inf")},
                "garch": {"n": 302.0, "r2_rv": 0.5, "qlike": float("nan")}}, seed=1)
    row = tool.headline(tool.aggregate(tool.load_runs(tmp_path), min_seeds=1)).iloc[0]
    assert row["best_r2_rv"] == "snn"
    assert row["best_qlike"] == "none" and row["headline"] == "undecided"


def test_the_headline_carries_the_scored_and_attempted_counts(tmp_path):
    """A winner drawn from two surviving seeds of eight must not read as an 8-seed one."""
    _sweep_cell(tmp_path, scored=(1, 7), aborted=(2, 3, 4, 5, 6, 8))
    frame = tool.aggregate(tool.load_runs(tmp_path), min_seeds=1,
                           attempts=tool.read_manifests(tmp_path))
    row = tool.headline(frame).iloc[0]
    assert row["n_scored"] == 2 and row["n_attempted"] == 8
