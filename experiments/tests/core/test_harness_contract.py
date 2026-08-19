"""Harness-level guarantees the experiment matrix depends on.

Run these first: they are the cheapest check that the matrix changes landed intact.
"""

import numpy as np
import pandas as pd
import pytest

from core.config import TARGETS, ExperimentConfig
from core.dataset import VrpDataset
from core.results import save_run


def test_rv_fwd_target_drops_the_leg_suffix():
    assert ExperimentConfig.load(target="rv_fwd").target_column == "rv_fwd"
    assert ExperimentConfig.load(target="vrp").target_column == "vrp_vix9d"


def test_unknown_target_is_rejected():
    with pytest.raises(ValueError, match="unknown target"):
        ExperimentConfig.load(target="not_a_target")


def test_rv_fwd_conversions_are_identity():
    data = VrpDataset(ExperimentConfig.load(target="rv_fwd"))
    split = data.split("test")
    assert np.allclose(split.to_target(split.rv_fwd), split.rv_fwd)
    assert np.allclose(split.to_rv_fwd(split.target), split.target)


def test_rv_fwd_target_keeps_the_leg_sample():
    #the arm is leg-free in its VALUES but not in its DATES: still gated on the leg,
    #so it stays comparable with the vrp arms instead of gaining 1990-2010
    rv = VrpDataset(ExperimentConfig.load(target="rv_fwd")).split("test")
    vrp = VrpDataset(ExperimentConfig.load(target="vrp")).split("test")
    assert list(rv.target.index) == list(vrp.target.index)


def test_fit_start_matches_the_sequence_models():
    data = VrpDataset(ExperimentConfig.load())
    assert data.fit_start() == data.split("train").target.index.min()


def test_har_fits_from_the_shared_start():
    from models.har_rv import HarRvForecaster

    data = VrpDataset(ExperimentConfig.load())
    model = HarRvForecaster(ExperimentConfig.load())
    model.fit(data)
    assert pd.Timestamp(model.fitted_range[0]) >= data.fit_start()


def test_results_dir_separates_arms(tmp_path):
    base = dict(horizon="weekly", target="vrp", iv_leg="vix9d")
    a = save_run(ExperimentConfig.load(**base, seed=1, encoding="delta"), {}, root=tmp_path)
    b = save_run(ExperimentConfig.load(**base, seed=2, encoding="delta"), {}, root=tmp_path)
    c = save_run(ExperimentConfig.load(**base, seed=1, encoding="delta_adaptive"),
                 {}, root=tmp_path)
    assert a.name != b.name and a.name != c.name
    assert "_s1" in a.name and "_s2" in b.name


def test_every_declared_target_builds_a_split():
    for target in TARGETS:
        data = VrpDataset(ExperimentConfig.load(target=target))
        assert len(data.split("test").target) > 0, target
