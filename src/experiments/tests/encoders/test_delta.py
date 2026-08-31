"""Accumulator and adaptive-threshold behaviour of the delta encoder."""

import numpy as np
import pytest

from encoders.delta import AdaptiveDeltaEncoder, DeltaEncoder


def _fixed(cls, threshold, **kwargs):
    enc = cls(**kwargs)
    enc._threshold = np.asarray([threshold], dtype=float)
    return enc


def test_static_arm_unchanged():
    #the original behaviour: one spike on the jump, on the even 'up' channel
    X = np.array([[[0.0], [0.0], [0.0], [10.0], [10.0]]])
    enc = DeltaEncoder()
    enc.fit(X)
    times, ids, num_neurons = enc._events(X)
    assert num_neurons == 2
    assert times[0].tolist() == [3.0]
    assert ids[0].tolist() == [0]


def test_static_discards_subthreshold_ramp():
    ramp = np.array([[[0.0], [0.2], [0.4], [0.6], [0.8]]])
    assert len(_fixed(DeltaEncoder, 0.5)._events(ramp)[0][0]) == 0


def test_accumulator_fires_on_a_subthreshold_ramp():
    #+0.2 x4 against a 0.5 threshold: crosses at step 3 (0.6) and carries 0.1 on
    ramp = np.array([[[0.0], [0.2], [0.4], [0.6], [0.8]]])
    times, ids, _ = _fixed(DeltaEncoder, 0.5, accumulate=True)._events(ramp)
    assert times[0].tolist() == [3.0]
    assert ids[0].tolist() == [0]


def test_accumulator_is_sign_symmetric():
    ramp = np.array([[[0.0], [-0.2], [-0.4], [-0.6], [-0.8]]])
    times, ids, _ = _fixed(DeltaEncoder, 0.5, accumulate=True)._events(ramp)
    assert times[0].tolist() == [3.0]
    assert ids[0].tolist() == [1], "a downward ramp must fire the odd channel"


def test_residual_carries_rather_than_resetting():
    #a +0.9 jump then +0.2: reset-to-zero fires once, carrying 0.4 fires twice
    X = np.array([[[0.0], [0.9], [1.1], [1.1]]])
    times, _, _ = _fixed(DeltaEncoder, 0.5, accumulate=True)._events(X)
    assert times[0].tolist() == [1.0, 2.0]


def test_accumulator_conserves_movement():
    #nothing is lost: net move = what the spikes carried + what is left in the bucket
    rng = np.random.default_rng(0)
    X = np.cumsum(rng.normal(size=(4, 40, 3)), axis=1)
    theta = 0.5
    enc = DeltaEncoder(accumulate=True)
    enc._threshold = np.full(3, theta)
    diffs = np.diff(X, axis=1)
    up, down, residual = enc._accumulated(
        diffs, np.broadcast_to(enc._threshold, diffs.shape))
    fired = (up.sum(axis=1) - down.sum(axis=1)) * theta
    assert np.allclose(diffs.sum(axis=1), fired + residual)


def test_bucket_drains_one_threshold_per_step():
    #a single move worth three thresholds fires on three consecutive steps
    X = np.array([[[0.0], [1.6], [1.6], [1.6], [1.6]]])
    times, ids, _ = _fixed(DeltaEncoder, 0.5, accumulate=True)._events(X)
    assert times[0].tolist() == [1.0, 2.0, 3.0]
    assert ids[0].tolist() == [0, 0, 0]


def test_adaptive_threshold_tracks_the_scale():
    #identical increments, different supplied scale -> different spike counts
    X = np.tile(np.arange(6.0)[None, :, None], (1, 1, 1))
    enc = _fixed(DeltaEncoder, 0.5, adaptive=True)
    calm = np.full((1, 6, 1), 0.1)
    loud = np.full((1, 6, 1), 10.0)
    assert len(enc._events(X, calm)[0][0]) == 5
    assert len(enc._events(X, loud)[0][0]) == 0


def test_adaptive_falls_back_where_scale_is_missing():
    X = np.array([[[0.0], [0.0], [0.0], [10.0], [10.0]]])
    enc = _fixed(DeltaEncoder, 1.0, adaptive=True)
    scale = np.full((1, 5, 1), np.nan)
    assert enc._events(X, scale)[0][0].tolist() == [3.0]


def test_adaptive_requires_a_scale():
    enc = _fixed(DeltaEncoder, 1.0, adaptive=True)
    with pytest.raises(ValueError, match="scale"):
        enc._events(np.zeros((1, 4, 1)))


def test_adaptive_arm_has_both_flags_on():
    enc = AdaptiveDeltaEncoder()
    assert enc.accumulate and enc.adaptive and enc.wants_scale
    assert not DeltaEncoder().wants_scale
