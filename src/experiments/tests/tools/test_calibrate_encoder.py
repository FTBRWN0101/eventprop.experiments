"""Reconstruction maths, and the monotonicity the old bisection took on faith."""

import numpy as np
import pytest

from core.config import ExperimentConfig
from core.dataset import VrpDataset
from encoders.base import ENCODERS
from encoders.delta import AdaptiveDeltaEncoder, DeltaEncoder
from tools import calibrate_encoder as calib


@pytest.fixture(scope="module")
def train_windows():
    data = VrpDataset(ExperimentConfig.load(input_window=45))
    X, _, _ = data.sequences("train", space="raw")
    return X, data.window_scale


def test_reconstruction_is_exact_on_a_clean_staircase():
    #steps of exactly one threshold: step-forward must track the signal perfectly
    X = np.array([[[0.0], [1.0], [2.0], [3.0]]])
    encoder = DeltaEncoder(accumulate=True)
    encoder._threshold = np.array([1.0])
    up, down, theta = encoder.spike_masks(X)
    recon = calib.reconstruct(X, up, down, theta)
    assert np.allclose(recon, X)
    assert np.isposinf(calib.snr_db(X, recon))


def test_reconstruction_lags_a_signal_that_moves_faster_than_one_step():
    #one spike per timestep, so a three-threshold jump cannot be tracked immediately
    X = np.array([[[0.0], [3.0], [3.0], [3.0]]])
    encoder = DeltaEncoder(accumulate=True)
    encoder._threshold = np.array([1.0])
    up, down, theta = encoder.spike_masks(X)
    recon = calib.reconstruct(X, up, down, theta)[0, :, 0]
    assert list(recon) == [0.0, 1.0, 2.0, 3.0], "drains one threshold per step"


def test_accumulator_reconstruction_beats_the_static_arm_on_a_ramp():
    #the whole argument for carrying the residual: sub-threshold drift is not lost
    ramp = np.cumsum(np.full((1, 40, 1), 0.2), axis=1)
    static, accumulate = DeltaEncoder(), DeltaEncoder(accumulate=True)
    static._threshold = accumulate._threshold = np.array([0.5])
    snrs = []
    for encoder in (static, accumulate):
        up, down, theta = encoder.spike_masks(ramp)
        snrs.append(calib.snr_db(ramp, calib.reconstruct(ramp, up, down, theta)))
    assert snrs[1] > snrs[0]


def test_spikes_per_cell_counts_both_channels():
    up = np.array([[[True, False], [False, False]]])
    down = np.array([[[False, False], [True, True]]])
    assert calib.spikes_per_cell(up, down) == 3 / 4


def test_monotonicity_violations_detects_a_rising_rate():
    assert calib.monotonicity_violations(np.array([0.5, 0.4, 0.3])) == 0
    assert calib.monotonicity_violations(np.array([0.5, 0.6, 0.3])) == 1


def test_firing_rate_falls_as_the_threshold_rises_static(train_windows):
    #the assumption the previous bisection rested on, checked rather than assumed
    X, _ = train_windows
    curve = calib.sweep("delta", X, None)
    assert calib.monotonicity_violations(curve["rate"]) == 0


def test_firing_rate_falls_as_the_threshold_rises_accumulator(train_windows):
    #the case I could not prove: the accumulator changes *which* steps fire, so a
    #higher threshold could in principle defer a spike into a step that fires anyway
    X, scale = train_windows
    curve = calib.sweep("delta_adaptive", X, scale)
    assert calib.monotonicity_violations(curve["rate"]) == 0


def test_rate_matching_lands_close_to_the_reference(train_windows):
    X, scale = train_windows
    reference = DeltaEncoder()
    reference.fit(X)
    up, down, _ = reference.spike_masks(X)
    goal = calib.spikes_per_cell(up, down)

    curve = calib.sweep("delta_adaptive", X, scale)
    multiplier = calib.match_rate("delta_adaptive", X, scale, goal, curve)
    encoder = AdaptiveDeltaEncoder(multiplier=multiplier)
    encoder.fit(X, scale)
    matched_up, matched_down, _ = encoder.spike_masks(X, scale)
    achieved = calib.spikes_per_cell(matched_up, matched_down)
    assert abs(achieved - goal) < 0.01, f"wanted {goal:.4f}, got {achieved:.4f}"


def test_matching_actually_moves_the_multiplier(train_windows):
    #if the arms already fired at the same rate there would be nothing to control for
    X, scale = train_windows
    static, adaptive = DeltaEncoder(), AdaptiveDeltaEncoder()
    static.fit(X)
    adaptive.fit(X, scale)
    su, sd, _ = static.spike_masks(X)
    au, ad, _ = adaptive.spike_masks(X, scale)
    assert calib.spikes_per_cell(au, ad) > calib.spikes_per_cell(su, sd)


def test_spike_masks_agree_with_the_event_arrays(train_windows):
    X, scale = train_windows
    encoder = AdaptiveDeltaEncoder()
    encoder.fit(X, scale)
    up, down, _ = encoder.spike_masks(X, scale)
    times, ids, num_neurons = encoder._events(X, scale)
    assert num_neurons == 2 * X.shape[2]
    assert sum(len(t) for t in times) == int(up.sum() + down.sum())
    #even ids are the up channel, odd the down
    assert sum(int((i % 2 == 0).sum()) for i in ids) == int(up.sum())


def test_encoders_are_discoverable_by_name():
    ENCODERS.discover(calib._ROOT / "encoders", "encoders")
    for name in ("delta", "delta_adaptive", "rate", "population"):
        assert ENCODERS.get(name) is not None
