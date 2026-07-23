"""Latency encoding: earlier spike means larger value."""

from __future__ import annotations

import numpy as np

from encoders.base import Encoder, EncodedInput


class LatencyEncoder(Encoder):
    """Normalise each feature to ``[0, 1]``; spike offset = ``1 - value`` within the day."""

    name = "latency"
    description = "Earlier spike = larger value (spike deletion risk near zero)"
    kind = "spikes"

    def fit(self, X: np.ndarray) -> None:
        self._min = X.min(axis=(0, 1))
        self._max = X.max(axis=(0, 1))

    def _events(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], int]:
        """Pure-numpy (times, ids) event arrays per example. No mlGeNN dependency."""
        n, t, f = X.shape
        span = self._max - self._min
        span = np.where(span == 0, 1.0, span)
        norm = np.clip((X - self._min) / span, 0.0, 1.0)
        offset = 1.0 - norm  #larger value -> earlier spike

        tt, ff = np.meshgrid(np.arange(t), np.arange(f), indexing="ij")
        times_list, ids_list = [], []
        for i in range(n):
            times_list.append((tt + offset[i]).ravel().astype(np.float32))
            ids_list.append(ff.ravel().astype(np.int64))
        return times_list, ids_list, f

    def encode(self, X: np.ndarray) -> EncodedInput:
        from ml_genn.utils.data import preprocess_spikes

        times_list, ids_list, num_neurons = self._events(X)
        spikes = [preprocess_spikes(t, i, num_neurons)
                  for t, i in zip(times_list, ids_list)]
        return EncodedInput(kind="spikes", data=spikes, num_neurons=num_neurons)


def demo() -> None:
    #two features over [0, 1]; fit needs a spread to normalise against
    X = np.array([[[0.0, 0.0]], [[1.0, 1.0]]])
    enc = LatencyEncoder()
    enc.fit(X)
    #feature 0 low, feature 1 high -> feature 1 must spike earlier
    probe = np.array([[[0.2, 0.9]]])
    times_list, ids_list, num_neurons = enc._events(probe)
    assert num_neurons == 2
    times, ids = times_list[0], ids_list[0]
    time_of = dict(zip(ids.tolist(), times.tolist()))
    assert time_of[1] < time_of[0], "feature with the larger value (id 1) must spike first"
    assert (times >= 0).all()
    print("latency encoder demo OK")


if __name__ == "__main__":
    demo()
