import json

import pandas as pd

from core.config import ExperimentConfig
from core.results import save_run


def test_save_run_writes_everything(tmp_path):
    cfg = ExperimentConfig.load()
    results = {"har_rv": {"mse": 1.0, "n": 3.0}}
    preds = {"har_rv": pd.Series([1.0, 2.0],
                                 index=pd.date_range("2020-01-01", periods=2))}
    diags = {"har_rv": {"fitted_range": ["1990-01-03", "2019-12-31"]}}

    run_dir = save_run(cfg, results, preds, diags, root=tmp_path)

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "predictions.csv").exists()
    assert (run_dir / "diagnostics.json").exists()
    saved_cfg = json.loads((run_dir / "config.json").read_text())
    assert saved_cfg["seed"] == 1
    assert saved_cfg["num_epochs"] == 50


def test_predictions_carry_the_realised_target(tmp_path):
    """predictions.csv must be self-contained: metrics recomputable without the panel."""
    cfg = ExperimentConfig.load()
    dates = pd.date_range("2020-01-01", periods=3)
    #two dates of three, as the SNN scores when it drops a partial batch
    preds = {"snn": pd.Series([1.0, 2.0], index=dates[:2])}
    actual = pd.Series([10.0, 20.0, 30.0], index=dates)

    run_dir = save_run(cfg, {"snn": {"mse": 1.0}}, preds, None, actual, root=tmp_path)

    frame = pd.read_csv(run_dir / "predictions.csv", parse_dates=["date"])
    assert list(frame.columns) == ["date", "actual", "snn"]
    assert frame["actual"].tolist() == [10.0, 20.0]
