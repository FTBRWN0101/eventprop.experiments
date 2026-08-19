"""--holdout-val must shrink every baseline's fit sample, not just HAR's (P2-2).

garch reads the full return frame and bounded its fit by the first *test* date, so
under --holdout-val it kept the three validation years that har_rv and the sequence
models lose. A DM between them was then comparing fits on different samples, which
is the statistic the project's own rule forbids.
"""

import pandas as pd
import pytest

from core.config import ExperimentConfig
from core.dataset import VrpDataset
from models.garch import GarchForecaster
from models.har_rv import HarRvForecaster
from run_experiment import _samples_comparable


@pytest.fixture(scope="module")
def fitted():
    """Both baselines fitted with and without the validation holdout."""
    out = {}
    for holdout in (False, True):
        config = ExperimentConfig.load(target="rv_fwd", iv_leg="vix9d",
                                       holdout_val=holdout)
        data = VrpDataset(config)
        models = {}
        for name, cls in (("garch", GarchForecaster), ("har_rv", HarRvForecaster)):
            model = cls(config)
            model.fit(data)
            models[name] = model
        out[holdout] = (data, models)
    return out


def test_baselines_share_a_fit_sample_without_the_holdout(fitted):
    _, models = fitted[False]
    assert models["garch"].fitted_range == models["har_rv"].fitted_range


def test_baselines_share_a_fit_sample_under_the_holdout(fitted):
    _, models = fitted[True]
    assert models["garch"].fitted_range == models["har_rv"].fitted_range


def test_holdout_actually_moves_the_garch_boundary(fitted):
    """The fix must bite: the held-out years leave garch's fit sample."""
    _, plain = fitted[False]
    _, held = fitted[True]
    assert held["garch"].fitted_range[1] < plain["garch"].fitted_range[1]
    assert pd.Timestamp(held["garch"].fitted_range[1]) < pd.Timestamp(VrpDataset.VAL_START)


def test_without_the_holdout_the_boundary_is_the_first_test_date(fitted):
    """The no-op property: no dates sit between the train end and the test start,
    so every result recorded before this fix stays valid."""
    _, models = fitted[False]
    assert models["garch"]._last_obs == models["garch"]._first_test


def test_the_dm_guard_admits_the_pair_once_the_samples_match(fitted):
    data, models = fitted[True]
    diagnostics = {name: {"fitted_range": list(m.fitted_range),
                          "fit_tolerance_days": m.fit_tolerance_days}
                   for name, m in models.items()}
    comparable, reason = _samples_comparable(
        diagnostics, data.daily("full").index, "garch", "har_rv")
    assert comparable, reason
