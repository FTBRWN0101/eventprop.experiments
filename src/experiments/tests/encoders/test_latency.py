"""Latency must survive the dt=1.0 grid that made the original version degenerate.

D16's evidence was that every spike time quantised to the same step. These tests pin
the property that fixes it: distinct values must land on distinct whole steps.
"""

import numpy as np
import pytest

from encoders.latency import LatencyEncoder


def _windows(values: np.ndarray, timesteps: int = 5) -> np.ndarray:
    """``[N, T, F]`` windows whose last day carries *values*."""
    n, f = values.shape
    X = np.zeros((n, timesteps, f))
    X[:, -1, :] = values
    return X


def test_larger_values_spike_earlier():
    X = _windows(np.array([[0.0], [0.5], [1.0]]))
    times = LatencyEncoder()._spike_times(X)
    assert times[2, 0] < times[1, 0] < times[0, 0]


def test_endpoints_span_the_whole_trial():
    timesteps = 21
    X = _windows(np.array([[1.0], [0.0]]), timesteps=timesteps)
    times = LatencyEncoder()._spike_times(X)
    assert times[0, 0] == 0
    assert times[1, 0] == timesteps - 1


def test_spike_times_are_whole_steps():
    rng = np.random.default_rng(3)
    X = _windows(rng.uniform(size=(40, 3)), timesteps=21)
    times = LatencyEncoder()._spike_times(X)
    assert np.all(times == np.rint(times)), "a fractional time is lost to the dt grid"


def test_distinct_values_resolve_to_distinct_steps():
    """The D16 failure was 720 spikes collapsing onto one step. This is the fix."""
    timesteps = 21
    values = np.linspace(0.0, 1.0, timesteps).reshape(-1, 1)
    times = LatencyEncoder()._spike_times(_windows(values, timesteps=timesteps))
    assert len(np.unique(times)) == timesteps


def test_one_spike_per_feature_per_window():
    rng = np.random.default_rng(5)
    X = _windows(rng.uniform(size=(7, 4)))
    times, ids, num_neurons = LatencyEncoder()._events(X)
    assert num_neurons == 4
    for row_times, row_ids in zip(times, ids):
        assert len(row_times) == 4
        assert sorted(row_ids) == [0, 1, 2, 3]


def test_events_are_sorted_in_time():
    rng = np.random.default_rng(9)
    X = _windows(rng.uniform(size=(20, 6)), timesteps=21)
    times, _, _ = LatencyEncoder()._events(X)
    for row in times:
        assert np.all(np.diff(row) >= 0), "preprocess_spikes needs sorted times"


def test_values_are_clipped_into_the_trial():
    X = _windows(np.array([[-2.0], [3.0]]), timesteps=5)
    times = LatencyEncoder()._spike_times(X)
    assert times.min() >= 0
    assert times.max() <= 4


def test_mean_summary_differs_from_last():
    X = np.zeros((1, 5, 1))
    X[0, :, 0] = [1.0, 1.0, 1.0, 1.0, 0.0]
    last = LatencyEncoder(summary="last")._spike_times(X)
    mean = LatencyEncoder(summary="mean")._spike_times(X)
    assert last[0, 0] == 4      #last day is 0.0, so latest possible
    assert mean[0, 0] < last[0, 0]


def test_unknown_summary_is_rejected():
    with pytest.raises(ValueError, match="unknown summary"):
        LatencyEncoder(summary="median")


def test_encoder_is_registered_under_its_name():
    from core.registry import Registry
    from encoders.base import ENCODERS
    from pathlib import Path

    ENCODERS.discover(Path(__file__).resolve().parents[2] / "encoders", "encoders")
    assert isinstance(ENCODERS, Registry)
    assert ENCODERS.get("latency") is LatencyEncoder
