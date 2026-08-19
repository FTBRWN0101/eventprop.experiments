"""Latency encoding: one spike per feature per window, earlier means larger.

D16 cut an earlier latency encoder and the measurement behind that cut was sound.
That version put the whole value into a *within-day* offset, firing at ``t + (1 -
value)``. GeNN fires on ``t >= spikeTimes[...]`` with ``t`` on the ``dt`` grid
(``neuronModels.h:279-281``), so at ``dt = 1.0`` every offset in ``(0, 1)`` lands on
the same step and the code conveys nothing. 139,641 distinct spike times collapsed
to 60.

What does not follow is that latency itself is impossible at one timestep per day.
Spreading the spike time across the whole ``T``-step trial gives ``T`` resolvable
levels, which is 5 weekly and 21 monthly, and needs no change to the timebase. That
is what this encoder does.

The cost is real and belongs in the write-up: one spike per feature per trial
discards within-window temporal structure for that feature. Rate and population keep
a value per day; this keeps a single summary per window. It is the fourth arm of the
encoding comparison the proposal specifies, not a candidate for the primary
hypothesis.

Takes the rank-uniform representation, so the value already spans [0, 1].
"""

from __future__ import annotations

import numpy as np

from encoders.base import Encoder, EncodedInput

#the value a window is reduced to before its spike time is chosen
SUMMARIES: tuple[str, ...] = ("last", "mean")
DEFAULT_SUMMARY = "last"


class LatencyEncoder(Encoder):
    """Emit one spike per feature, timed across the trial so earlier means larger."""

    name = "latency"
    description = "Time-to-first-spike across the trial; earlier spike = larger value"
    kind = "spikes"
    input_space = "unit"

    def __init__(self, summary: str = DEFAULT_SUMMARY) -> None:
        if summary not in SUMMARIES:
            raise ValueError(f"unknown summary {summary!r}, expected one of {SUMMARIES}")
        self.summary = summary

    def fit(self, X: np.ndarray, scale: np.ndarray | None = None) -> None:
        """No-op: the rank transform is fitted upstream, per column, on the daily frame."""

    def _summarise(self, X: np.ndarray) -> np.ndarray:
        """Reduce ``[N, T, F]`` windows to the ``[N, F]`` value that sets the spike time."""
        if self.summary == "mean":
            return np.nanmean(X, axis=1)
        #"last": the most recent day, which is the one the forecast is made from
        return X[:, -1, :]

    def _spike_times(self, X: np.ndarray) -> np.ndarray:
        """Spike time per (example, feature), in ``[0, T-1]``.

        Value 1 fires at step 0 and value 0 at step ``T-1``, so a larger value is
        always earlier. Times are whole steps because anything finer is quantised
        away by the ``dt = 1.0`` grid, which is what made the original encoder
        degenerate.
        """
        value = np.clip(self._summarise(X), 0.0, 1.0)
        last_step = X.shape[1] - 1
        return np.rint((1.0 - value) * last_step)

    def _events(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], int]:
        """Pure-numpy (times, ids) event arrays per example. No mlGeNN dependency."""
        n, _, f = X.shape
        times = self._spike_times(X)
        feature_ids = np.arange(f, dtype=np.int64)

        times_list, ids_list = [], []
        for i in range(n):
            row = times[i]
            finite = np.isfinite(row)
            #spikes must be sorted in time for preprocess_spikes
            order = np.argsort(row[finite], kind="stable")
            times_list.append(row[finite][order].astype(np.float32))
            ids_list.append(feature_ids[finite][order])
        return times_list, ids_list, f

    def encode(self, X: np.ndarray, scale: np.ndarray | None = None) -> EncodedInput:
        from ml_genn.utils.data import preprocess_spikes

        times_list, ids_list, num_neurons = self._events(X)
        spikes = [preprocess_spikes(t, i, num_neurons)
                  for t, i in zip(times_list, ids_list)]
        return EncodedInput(kind="spikes", data=spikes, num_neurons=num_neurons)
