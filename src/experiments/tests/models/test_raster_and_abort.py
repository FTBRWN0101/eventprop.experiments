"""Rasters must reach disk, and the 10% silent threshold must stop training.

The proposal builds raster monitoring into the training loop from the first epoch and
pauses when more than 10% of hidden neurons fall silent. A warning in a background
run is not a pause, so the default here raises.

Both paths are pure Python and numpy, so they run without the mlgenn env.
"""

import numpy as np
import pytest

from core.config import ExperimentConfig
from models.snn import SILENT_NEURON_WARN_THRESHOLD, SnnForecaster


def _forecaster(tmp_path, **overrides) -> SnnForecaster:
    model = SnnForecaster(ExperimentConfig.load(**overrides))
    model._timesteps = 5
    #keep every artefact inside tmp_path
    model._checkpoint_dir = lambda create=False: tmp_path  #type: ignore[method-assign]
    return model


def test_raster_is_written_with_times_ids_and_example(tmp_path):
    model = _forecaster(tmp_path)
    times = [np.array([0.0, 2.0]), np.array([1.0])]
    ids = [np.array([3, 7]), np.array([5])]

    path = model._save_raster((times, ids), epoch=10)
    assert path is not None and path.exists()
    assert path.name == "epoch10.npz"

    saved = np.load(path)
    assert np.array_equal(saved["times"], [0.0, 2.0, 1.0])
    assert np.array_equal(saved["ids"], [3, 7, 5])
    #the example column is what makes one window separable from the next
    assert np.array_equal(saved["example"], [0, 0, 1])
    assert int(saved["timesteps"]) == 5


def test_raster_lands_under_the_checkpoint_identity(tmp_path):
    model = _forecaster(tmp_path)
    path = model._save_raster(([np.array([0.0])], [np.array([1])]), epoch=20)
    assert path.parent == tmp_path / "rasters"


def test_empty_raster_writes_nothing(tmp_path):
    model = _forecaster(tmp_path)
    assert model._save_raster(([], []), epoch=10) is None


def test_below_threshold_does_not_raise(tmp_path):
    model = _forecaster(tmp_path)
    model._check_silent_fraction(SILENT_NEURON_WARN_THRESHOLD, 10, 50)


def test_above_threshold_stops_training(tmp_path):
    model = _forecaster(tmp_path)
    with pytest.raises(RuntimeError, match="spike deletion"):
        model._check_silent_fraction(0.5, 10, 50)


def test_abort_message_points_at_the_rasters(tmp_path):
    model = _forecaster(tmp_path)
    with pytest.raises(RuntimeError, match="rasters"):
        model._check_silent_fraction(0.5, 10, 50)


def test_allow_silent_neurons_downgrades_to_a_warning(tmp_path, caplog):
    model = _forecaster(tmp_path, silent_neuron_abort=False)
    with caplog.at_level("WARNING"):
        model._check_silent_fraction(0.5, 10, 50)
    assert "allow-silent-neurons" in caplog.text


def test_nan_fraction_is_not_treated_as_a_breach(tmp_path):
    """An empty spike-count array yields NaN; that is missing data, not a breach."""
    model = _forecaster(tmp_path)
    model._check_silent_fraction(float("nan"), 10, 50)
