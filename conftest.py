"""Skip the tests that need data the run does not have.

Most of the suite reads the processed panels under data-save/, which the repo does not ship.
Without them those tests skip instead of failing, so a fresh clone and CI run the rest.
Two tests also need the OptionMetrics volatility surface, which needs a WRDS licence.
"""
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "data-save"

NEEDS_SURFACE = {
    "src/experiments/tests/core/test_vvrp.py::test_surface_columns_stay_out_of_the_default_set",
    "src/experiments/tests/test_end_to_end.py::test_surface_arm_costs_test_rows_but_still_runs",
}


def pytest_collection_modifyitems(config, items):
    if any((DATA / "optionmetrics").glob("spx_vsurf_*.csv")):
        return
    skip = pytest.mark.skip(reason="needs the OptionMetrics surface, which needs a WRDS licence")
    for item in items:
        if item.nodeid in NEEDS_SURFACE:
            item.add_marker(skip)


def _skip_if_data_absent(exc: FileNotFoundError) -> None:
    """Skip the test if it failed only because the panels have not been built."""
    missing = Path(exc.filename).absolute() if exc.filename else None
    if missing and missing.is_relative_to(DATA) and not (DATA / "processed").exists():
        pytest.skip("needs the data: run fetch_all.py, then build_vrp.py")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    try:
        return (yield)
    except FileNotFoundError as exc:
        _skip_if_data_absent(exc)
        raise


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except FileNotFoundError as exc:
        _skip_if_data_absent(exc)
        raise
