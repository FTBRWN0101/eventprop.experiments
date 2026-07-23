"""Delta encoding: spike when a feature moves more than a threshold.

Two channels per feature, up and down, so the sign survives. Takes raw
features, not the rank-uniform ones.
"""

from __future__ import annotations

import numpy as np

from encoders.base import Encoder, EncodedInput

DEFAULT_MULTIPLIER = 1.0


class DeltaEncoder(Encoder):
    """Spike when ``|x[t] - x[t-1]| > multiplier * std(diffs)``, split by sign."""

    name = "delta"
    description = "Spike on significant change; adaptive threshold (primary hypothesis)"
    kind = "spikes"

    def __init__(self, multiplier: float = DEFAULT_MULTIPLIER) -> None:
        self.multiplier = multiplier

    def fit(self, X: np.ndarray) -> None:
        diffs = np.diff(X, axis=1)
        threshold = self.multiplier * diffs.std(axis=(0, 1))
        threshold[threshold == 0] = 1.0
        self._threshold = threshold

    def _events(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], int]:
        """Pure-numpy (times, ids) event arrays per example. No mlGeNN dependency."""
        n, _, f = X.shape
        diffs = np.diff(X, axis=1)  #[N, T-1, F]; diffs[:, k] = X[:, k+1] - X[:, k]
        up = diffs > self._threshold
        down = diffs < -self._threshold

        times_list, ids_list = [], []
        for i in range(n):
            up_t, up_f = np.nonzero(up[i])
            down_t, down_f = np.nonzero(down[i])
            times = np.concatenate([up_t + 1, down_t + 1]).astype(np.float32)
            ids = np.concatenate([up_f * 2, down_f * 2 + 1]).astype(np.int64)
            times_list.append(times)
            ids_list.append(ids)
        return times_list, ids_list, 2 * f

    def encode(self, X: np.ndarray) -> EncodedInput:
        from ml_genn.utils.data import preprocess_spikes

        times_list, ids_list, num_neurons = self._events(X)
        spikes = [preprocess_spikes(t, i, num_neurons)
                  for t, i in zip(times_list, ids_list)]
        return EncodedInput(kind="spikes", data=spikes, num_neurons=num_neurons)


def demo() -> None:
    #flat run then a jump bigger than any in-window std
    X = np.array([[[0.0], [0.0], [0.0], [10.0], [10.0]]])
    enc = DeltaEncoder(multiplier=1.0)
    enc.fit(X)
    times_list, ids_list, num_neurons = enc._events(X)
    assert num_neurons == 2
    times, ids = times_list[0], ids_list[0]
    assert len(times) == 1, "expected exactly one spike (the jump at t=3)"
    assert times[0] == 3.0
    assert ids[0] == 0, "positive jump must fire the 'up' channel (even id)"

    #a symmetric drop must fire the down channel
    X_down = np.array([[[10.0], [10.0], [10.0], [0.0], [0.0]]])
    enc2 = DeltaEncoder(multiplier=1.0)
    enc2.fit(X_down)
    times2, ids2, _ = enc2._events(X_down)
    assert ids2[0][0] == 1, "negative jump must fire the 'down' channel (odd id)"
    print("delta encoder demo OK")


if __name__ == "__main__":
    demo()
