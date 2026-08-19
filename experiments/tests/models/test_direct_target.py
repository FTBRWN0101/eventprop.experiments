"""har_rv_direct: the baseline made to solve the sequence models' problem (O8, D61).

The standard arm forecasts rv_fwd and lets Split.to_target convert, which hands it
iv_t exactly whatever the feature set. snn and lstm predict the target column and must
rebuild that from features. This sibling closes the gap so it can be measured.

There is deliberately no garch_direct. A constant-mean GARCH fitted on a target series
forecasts a single constant for every date, so it collapses onto the constant baseline
rather than forming a distinct model. GARCH earns its place here through its variance
forecast, which only exists in forward-vol space.
"""

import numpy as np
import pytest

from core.config import ExperimentConfig
from core.dataset import VrpDataset
from models.base import MODELS
from models.har_rv import HarRvDirectForecaster, HarRvForecaster


def fit_both(**overrides):
    config = ExperimentConfig.load(**overrides)
    data = VrpDataset(config)
    out = {}
    for cls in (HarRvForecaster, HarRvDirectForecaster):
        model = cls(config)
        model.fit(data)
        out[cls.name] = (model, model.predict(data, "test"))
    return data, out


def test_the_direct_arm_is_registered():
    assert MODELS.get("har_rv_direct") is HarRvDirectForecaster
    assert "har_rv_direct" in set(MODELS.names())


def test_identical_on_rv_fwd_where_the_map_is_the_identity():
    #target_column is rv_fwd and to_target is the identity, so the two arms coincide
    _, out = fit_both(target="rv_fwd", iv_leg="vix9d")
    plain = out["har_rv"][1]
    direct = out["har_rv_direct"][1]
    assert np.allclose(plain.to_numpy(), direct.to_numpy())


def test_they_differ_on_a_vrp_target():
    _, out = fit_both(target="vrp", iv_leg="vix9d")
    plain = out["har_rv"][1]
    direct = out["har_rv_direct"][1]
    assert not np.allclose(plain.to_numpy(), direct.to_numpy())


def test_the_direct_arm_predicts_in_target_space():
    """A direct fit must land near the target's own scale, not forward vol's."""
    data, out = fit_both(target="vrp", iv_leg="vix9d")
    direct = out["har_rv_direct"][1]
    train_target_mean = float(data.split("train").target.mean())
    rv_mean = float(data.split("train").rv_fwd.mean())
    assert abs(direct.mean() - train_target_mean) < abs(direct.mean() - rv_mean)


@pytest.mark.parametrize("target", ["vrp", "rvrp", "rv_fwd"])
def test_fit_samples_match_so_the_dm_guard_admits_the_pair(target):
    _, out = fit_both(target=target, iv_leg="vix9d")
    assert out["har_rv"][0].fitted_range == out["har_rv_direct"][0].fitted_range


def test_the_direct_arm_still_honours_the_shared_fit_start():
    data, out = fit_both(target="vrp", iv_leg="vix9d")
    model = out["har_rv_direct"][0]
    assert model.fitted_range[0] == str(data.fit_start().date())
