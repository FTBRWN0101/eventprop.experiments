"""Load ``data-process`` modules by path.

The directory name contains a hyphen, so it is not importable, and its ``core``
package would shadow ``experiments/core`` if it went on ``sys.path``. Loading the
one module under its own name avoids both problems.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PROCESS = REPO_ROOT / "data-process"
PROCESSED = REPO_ROOT / "data-save" / "processed"
SPX_CSV = REPO_ROOT / "data-save" / "index_returns" / "spx_daily.csv"


def load_module(relative_path: str, name: str) -> ModuleType:
    """Import a ``data-process`` module from its file, bypassing sys.path."""
    path = DATA_PROCESS / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def volatility() -> ModuleType:
    return load_module("core/volatility.py", "_dp_volatility")


@pytest.fixture(scope="session")
def spx_prices():
    """Adjusted close from the raw SPX file, date-indexed."""
    import pandas as pd

    if not SPX_CSV.exists():
        pytest.skip(f"raw SPX file absent: {SPX_CSV}")
    frame = pd.read_csv(SPX_CSV, parse_dates=["date"]).set_index("date").sort_index()
    return frame["adj_close"]


@pytest.fixture(scope="session")
def panels():
    """``{(horizon, split): frame}`` for every processed panel on disk."""
    import pandas as pd

    frames = {}
    for horizon in ("weekly", "monthly"):
        for split in ("train", "test", "full"):
            path = PROCESSED / horizon / f"{split}.csv"
            if path.exists():
                frames[(horizon, split)] = (
                    pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index())
    if not frames:
        pytest.skip(f"no processed panels under {PROCESSED}")
    return frames
