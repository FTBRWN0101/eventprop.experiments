"""har_rv_full / garch_full: the unrestricted siblings of the D27-clipped benchmarks."""

import pandas as pd
import pytest

from core.config import ExperimentConfig
from core.dataset import VrpDataset
from models.base import MODELS
from models.garch import GarchForecaster, GarchFullForecaster
from models.har_rv import HarRvForecaster, HarRvFullForecaster


@pytest.fixture(scope="module")
def data():
    return VrpDataset(ExperimentConfig.load())


def test_both_variants_self_register():
    #__init_subclass__ registers on import, so importing the modules is the whole test
    assert MODELS.get("har_rv_full") is HarRvFullForecaster
    assert MODELS.get("garch_full") is GarchFullForecaster
    assert {"har_rv", "garch"} <= set(MODELS.names())


def test_full_har_fits_from_the_panel_start_not_the_shared_start(data):
    restricted, full = HarRvForecaster(data.config), HarRvFullForecaster(data.config)
    restricted.fit(data)
    full.fit(data)
    assert pd.Timestamp(restricted.fitted_range[0]) == data.fit_start()
    assert pd.Timestamp(full.fitted_range[0]) < data.fit_start()
    assert full.fitted_range[0].startswith("1990")


def test_the_two_har_variants_are_genuinely_different_models(data):
    restricted, full = HarRvForecaster(data.config), HarRvFullForecaster(data.config)
    restricted.fit(data)
    full.fit(data)
    assert not (restricted._coef == full._coef).all()
    a = restricted.predict(data, "test")
    b = full.predict(data, "test")
    assert not a.equals(b)


def test_restricting_har_does_not_weaken_it_on_this_test_set(data):
    #HARNESS-REVIEW R2 assumed D27 weakens the benchmark. It does not: on the 2020-2025
    #test set the ~2011-fitted HAR beats the 1990-fitted one. tools/har_sample_curve.py
    #reports the whole curve; this pins the direction so a refactor cannot silently
    #flip the claim the write-up rests on.
    from core import metrics

    split = data.split("test")
    scores = []
    for cls in (HarRvForecaster, HarRvFullForecaster):
        model = cls(data.config)
        model.fit(data)
        scores.append(metrics.squared_errors(split, model.predict(data, "test")).mean())
    restricted_mse, full_mse = scores
    assert restricted_mse < full_mse


def test_full_garch_fits_from_the_panel_start(data):
    restricted, full = GarchForecaster(data.config), GarchFullForecaster(data.config)
    restricted.fit(data)
    full.fit(data)
    assert pd.Timestamp(full.fitted_range[0]) < pd.Timestamp(restricted.fitted_range[0])
    assert full.fitted_range[0].startswith("1990")
    #same upper bound: last_obs, not the sample floor, is what keeps test out of the fit
    assert full.fitted_range[1] == restricted.fitted_range[1]


def test_the_restricted_models_are_untouched(data):
    #these are additions, not changes: the default arms must behave exactly as before
    assert HarRvForecaster.restrict_sample is True
    assert GarchForecaster.restrict_sample is True
    assert HarRvForecaster.sample_floor is None
    model = HarRvForecaster(data.config)
    model.fit(data)
    assert pd.Timestamp(model.fitted_range[0]) == data.fit_start()


def test_sample_floor_overrides_the_shared_start(data):
    model = HarRvForecaster(data.config)
    model.sample_floor = pd.Timestamp("2005-01-01")
    model.fit(data)
    assert model.fitted_range[0].startswith("2005")
