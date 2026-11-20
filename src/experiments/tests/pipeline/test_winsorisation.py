"""Regression tests for train-only winsorisation in the data-process stage."""

from __future__ import annotations

import pandas as pd
import pytest


def test_winsorisation_rejects_test_only_target_values(data_process_import):
    """Quantile bounds must never be estimated from test-period target values."""
    assembler_module = data_process_import("assembler")
    config_module = data_process_import("core.config")

    config = config_module.ProcessConfig.load(split_date="2020-01-01")
    assembler = assembler_module.DatasetAssembler(config, builders=[])
    index = pd.to_datetime(["2019-12-31", "2020-01-01"])
    panel = pd.DataFrame({"rvrp_vix": [float("nan"), 0.5]}, index=index)
    #the caller now names the training rows, and only 2019-12-31 is one
    train_index = index[index < config.split_ts]

    with pytest.raises(ValueError, match="rvrp_vix.*no non-null training observations"):
        assembler._winsorise(panel, ["rvrp_vix"], train_index)
