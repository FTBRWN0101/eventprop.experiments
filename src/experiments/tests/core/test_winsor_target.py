"""The winsorised rVRP arm the proposal specifies must be selectable and mapped."""

import numpy as np
import pandas as pd
import pytest

from core.config import TARGETS, ExperimentConfig
from core.dataset import Split, VrpDataset


def _split(kind: str) -> Split:
    index = pd.date_range("2020-01-01", periods=5)
    return Split(
        name="test", features=pd.DataFrame(index=index),
        target=pd.Series(1.0, index=index), rv_fwd=pd.Series(12.0, index=index),
        iv=pd.Series(20.0, index=index), denom=pd.Series(19.0, index=index),
        target_kind=kind, horizon_days=5)


def test_winsor_is_a_selectable_target():
    assert "rvrp_winsor" in TARGETS
    config = ExperimentConfig.load(target="rvrp_winsor")
    assert config.target_column == "rvrp_vix9d_winsor"


def test_winsor_column_exists_in_both_panels():
    for horizon, leg in (("weekly", "vix9d"), ("monthly", "vix")):
        config = ExperimentConfig.load(horizon=horizon, iv_leg=leg, target="rvrp_winsor")
        frame = pd.read_csv(config.processed_path("train"), nrows=1)
        assert config.target_column in frame.columns


def test_winsor_shares_the_rvrp_map():
    rv_hat = pd.Series(11.0, index=pd.date_range("2020-01-01", periods=5))
    plain, winsor = _split("rvrp"), _split("rvrp_winsor")
    assert np.allclose(plain.to_target(rv_hat), winsor.to_target(rv_hat))
    target_hat = pd.Series(0.4, index=rv_hat.index)
    assert np.allclose(plain.to_rv_fwd(target_hat), winsor.to_rv_fwd(target_hat))


def test_round_trip_is_exact_in_target_space():
    split = _split("rvrp_winsor")
    rv_hat = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=split.target.index)
    assert np.allclose(split.to_rv_fwd(split.to_target(rv_hat)), rv_hat)


def test_unknown_target_kind_raises_rather_than_defaulting_to_rvrp():
    split = _split("not_a_target")
    series = pd.Series(1.0, index=split.target.index)
    with pytest.raises(ValueError, match="no target map"):
        split.to_target(series)
    with pytest.raises(ValueError, match="no inverse map"):
        split.to_rv_fwd(series)


def test_winsor_target_is_not_served_as_a_feature():
    config = ExperimentConfig.load(target="rvrp_winsor")
    data = VrpDataset(config)
    columns = data.feature_columns(data.daily("train"))
    assert not any(c.startswith("rvrp_") for c in columns)


def test_winsor_split_is_clipped_relative_to_the_plain_target():
    """The winsorised series must have no wider a range than the plain one."""
    config = ExperimentConfig.load(target="rvrp_winsor")
    data = VrpDataset(config)
    frame = data.daily("train")
    plain = frame[f"rvrp_{config.iv_leg}"].dropna()
    winsor = frame[config.target_column].dropna()
    assert winsor.min() >= plain.min()
    assert winsor.max() <= plain.max()
    assert (winsor != plain.reindex(winsor.index)).any(), "clip never bound on train"
