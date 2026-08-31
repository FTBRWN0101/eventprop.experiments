"""Regression tests for train-only winsorisation in the data-process stage."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator

import pandas as pd
import pytest


DATA_PROCESS = Path(__file__).resolve().parents[4] / "src" / "data_process"


@contextmanager
def _data_process_modules() -> Iterator[tuple[ModuleType, ModuleType]]:
    """Temporarily load data-process's absolute ``core`` imports in isolation."""
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "core" or name.startswith("core.") or name == "assembler"
    }
    for name in saved:
        del sys.modules[name]
    sys.path.insert(0, str(DATA_PROCESS))
    try:
        yield importlib.import_module("assembler"), importlib.import_module("core.config")
    finally:
        sys.path.remove(str(DATA_PROCESS))
        for name in tuple(sys.modules):
            if name == "core" or name.startswith("core.") or name == "assembler":
                del sys.modules[name]
        sys.modules.update(saved)


def test_winsorisation_rejects_test_only_target_values():
    """Quantile bounds must never be estimated from test-period target values."""
    with _data_process_modules() as (assembler_module, config_module):
        config = config_module.ProcessConfig.load(split_date="2020-01-01")
        assembler = assembler_module.DatasetAssembler(config, builders=[])
        index = pd.to_datetime(["2019-12-31", "2020-01-01"])
        panel = pd.DataFrame({"rvrp_vix": [float("nan"), 0.5]}, index=index)

        with pytest.raises(ValueError, match="rvrp_vix.*no non-null training observations"):
            assembler._winsorise(panel, ["rvrp_vix"])
