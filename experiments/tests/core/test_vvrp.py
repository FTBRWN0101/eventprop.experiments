"""Variance-space VRP target: panel column, conversion, and its inverse."""

import numpy as np
import pandas as pd
import pytest

from core.config import ExperimentConfig
from core.dataset import VrpDataset


@pytest.fixture(scope="module")
def vvrp_data():
    return VrpDataset(ExperimentConfig.load(target="vvrp"))


def test_panel_carries_the_variance_premium(vvrp_data):
    frame = vvrp_data.daily("test")
    expected = frame["iv_vix9d"] ** 2 - frame["rv_fwd"] ** 2
    assert np.allclose(frame["vvrp_vix9d"], expected, equal_nan=True)


def test_variance_premium_is_not_a_feature(vvrp_data):
    frame = vvrp_data.daily("test")
    assert "vvrp_vix9d" not in vvrp_data.feature_columns(frame)


def test_conversion_round_trips(vvrp_data):
    split = vvrp_data.split("test")
    assert np.allclose(split.to_rv_fwd(split.to_target(split.rv_fwd)), split.rv_fwd)


def test_realised_target_equals_the_conversion_of_realised_rv(vvrp_data):
    split = vvrp_data.split("test")
    assert np.allclose(split.to_target(split.rv_fwd), split.target)


def test_inverse_clips_impossible_variance(vvrp_data):
    #a forecast above iv^2 implies negative variance; the inverse must stay real
    split = vvrp_data.split("test")
    absurd = pd.Series(1e9, index=split.iv.index)
    assert (split.to_rv_fwd(absurd) == 0.0).all()


def test_surface_columns_stay_out_of_the_default_set(vvrp_data):
    frame = vvrp_data.daily("test")
    assert "om_atm30" in frame.columns
    assert "om_atm30" not in vvrp_data.feature_columns(frame)

    with_surface = VrpDataset(
        ExperimentConfig.load(feature_set="options+price+surface"))
    assert "om_atm30" in with_surface.feature_columns(frame)
