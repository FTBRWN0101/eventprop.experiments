"""Weight scales are config, and a sweep over them must not share checkpoints."""

import pytest

from core.config import ExperimentConfig
from models.snn import W_IN_SCALE, W_REC_SCALE, SnnForecaster


def test_defaults_match_the_module_constants():
    #the constants stay the documented default, so an unset run is unchanged
    config = ExperimentConfig.load()
    assert config.w_in_scale == W_IN_SCALE
    assert config.w_rec_scale == W_REC_SCALE


@pytest.mark.parametrize("field", ["w_in_scale", "w_rec_scale"])
@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_scales_are_rejected(field, bad):
    with pytest.raises(ValueError, match=field):
        ExperimentConfig.load(**{field: bad})


def test_default_scales_leave_the_checkpoint_path_unsuffixed():
    path = SnnForecaster(ExperimentConfig.load())._checkpoint_dir()
    assert "_wi" not in path.name and "_wr" not in path.name


def test_changed_scales_separate_checkpoints():
    #D42's failure mode: without this a drive sweep silently overwrites itself
    base = SnnForecaster(ExperimentConfig.load())._checkpoint_dir()
    louder = SnnForecaster(
        ExperimentConfig.load(w_in_scale=10.0))._checkpoint_dir()
    quieter = SnnForecaster(
        ExperimentConfig.load(w_rec_scale=0.15))._checkpoint_dir()
    assert len({base.name, louder.name, quieter.name}) == 3
    assert "_wi10" in louder.name
    assert "_wr0.15" in quieter.name
