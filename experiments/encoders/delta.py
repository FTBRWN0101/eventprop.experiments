"""Delta encoding: spike when a feature moves more than a threshold, split by sign.

Two flags, both off by default so delta stays the static control arm:
accumulate carries the sub-threshold residual, adaptive takes the threshold
from a rolling std instead of one global scalar.

These are not ad-hoc variants. In the temporal-contrast taxonomy of Petro, Kasabov &
Kiss (IEEE TNNLS 2020) the two arms are named published methods:

* ``accumulate=False`` is threshold-based representation (TBR) -- fire on
  ``|x[k] - x[k-1]| > theta``, and discard whatever falls short.
* ``accumulate=True`` is step-forward (SF) -- keep a baseline ``B``, fire when
  ``x[k]`` leaves ``B +/- theta``, then move ``B`` by ``+/-theta``. Subtracting the
  threshold from a running accumulator is algebraically the same thing, since the
  accumulator holds ``x[k] - B``.

That paper found SF the most robust and easiest to optimise of the four it compared,
which is the prior for expecting the accumulator to win here. It also sets the
calibration convention used in ``tools/calibrate_encoder.py``: pick the threshold by
the reconstruction error between the original and the spike-reconstructed signal
(SNR), not by downstream task score.

Takes raw features, not the rank-uniform ones.
"""

from __future__ import annotations

import numpy as np

from encoders.base import Encoder, EncodedInput

DEFAULT_MULTIPLIER = 1.0


class DeltaEncoder(Encoder):
    """Spike when the (optionally accumulated) change clears the threshold, split by sign."""

    name = "delta"
    description = "Spike on significant change; static global threshold (control arm)"
    kind = "spikes"
    input_space = "raw"

    def __init__(self, multiplier: float = DEFAULT_MULTIPLIER,
                 accumulate: bool = False, adaptive: bool = False) -> None:
        self.multiplier = multiplier
        self.accumulate = accumulate
        self.adaptive = adaptive

    @property
    def wants_scale(self) -> bool:  # type: ignore[override]
        return self.adaptive

    def fit(self, X: np.ndarray, scale: np.ndarray | None = None) -> None:
        #kept even when adaptive: it is the fallback wherever the rolling window is short
        diffs = np.diff(X, axis=1)
        threshold = self.multiplier * diffs.std(axis=(0, 1))
        threshold[threshold == 0] = 1.0
        self._threshold = threshold

    def _thresholds(self, X: np.ndarray, scale: np.ndarray | None) -> np.ndarray:
        """Per-(example, step, feature) threshold, aligned to ``diffs``."""
        n, t, f = X.shape
        if not self.adaptive:
            return np.broadcast_to(self._threshold, (n, t - 1, f))
        if scale is None:
            raise ValueError("adaptive delta needs the per-window scale from sequences()")
        #scale[:, k] belongs to day k, and diffs[:, k] lands on day k+1
        theta = self.multiplier * np.asarray(scale)[:, 1:, :]
        usable = np.isfinite(theta) & (theta > 0)
        return np.where(usable, theta, self._threshold)

    @staticmethod
    def _accumulated(diffs: np.ndarray, theta: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Signed bucket per feature: fire on crossing, keep the remainder.

        Returns ``(up, down, residual)``. One spike per step, so a big move drains
        over later steps rather than being lost.

        The comparison is ``>=``, not ``>``. Under a strict inequality a bucket holding
        exactly one threshold can never discharge, so a signal rising by exactly theta
        per step tracks one threshold behind for ever and a jump of exactly n*theta
        strands its last unit permanently. On float data the two rules differ only on
        exact equality and the spike counts are identical, but ``>=`` is what makes
        "deferred, never destroyed" true rather than nearly true.
        """
        up = np.zeros(diffs.shape, dtype=bool)
        down = np.zeros(diffs.shape, dtype=bool)
        acc = np.zeros((diffs.shape[0], diffs.shape[2]))
        for k in range(diffs.shape[1]):
            acc = acc + diffs[:, k]
            step = theta[:, k]
            fire_up, fire_down = acc >= step, acc <= -step
            #subtract, do not reset: the overshoot is real movement and stays in the bucket
            acc = np.where(fire_up, acc - step, np.where(fire_down, acc + step, acc))
            up[:, k], down[:, k] = fire_up, fire_down
        return up, down, acc

    def spike_masks(self, X: np.ndarray, scale: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(up, down, theta)`` aligned to ``diffs``, i.e. shape ``[N, T-1, F]``.

        Public because the reconstruction diagnostics need the per-step threshold that
        each spike stands for, not just the event times.
        """
        diffs = np.diff(X, axis=1)  #diffs[:, k] = X[:, k+1] - X[:, k]
        theta = self._thresholds(X, scale)
        if self.accumulate:
            up, down, _ = self._accumulated(diffs, theta)
        else:
            #>= to match the accumulator's rule; on float data they never differ
            up, down = diffs >= theta, diffs <= -theta
        return up, down, theta

    def _events(self, X: np.ndarray, scale: np.ndarray | None = None
                ) -> tuple[list[np.ndarray], list[np.ndarray], int]:
        """Pure-numpy (times, ids) event arrays per example. No mlGeNN dependency."""
        n, _, f = X.shape
        up, down, _ = self.spike_masks(X, scale)

        times_list, ids_list = [], []
        for i in range(n):
            up_t, up_f = np.nonzero(up[i])
            down_t, down_f = np.nonzero(down[i])
            times = np.concatenate([up_t + 1, down_t + 1]).astype(np.float32)
            ids = np.concatenate([up_f * 2, down_f * 2 + 1]).astype(np.int64)
            times_list.append(times)
            ids_list.append(ids)
        return times_list, ids_list, 2 * f

    def encode(self, X: np.ndarray, scale: np.ndarray | None = None) -> EncodedInput:
        from ml_genn.utils.data import preprocess_spikes

        times_list, ids_list, num_neurons = self._events(X, scale)
        spikes = [preprocess_spikes(t, i, num_neurons)
                  for t, i in zip(times_list, ids_list)]
        return EncodedInput(kind="spikes", data=spikes, num_neurons=num_neurons)


class AdaptiveDeltaEncoder(DeltaEncoder):
    """Delta with both upgrades on: residual carry and a causal rolling threshold."""

    name = "delta_adaptive"
    description = "Delta with residual accumulator and causal rolling threshold"

    def __init__(self, multiplier: float = DEFAULT_MULTIPLIER,
                 accumulate: bool = True, adaptive: bool = True) -> None:
        super().__init__(multiplier, accumulate, adaptive)


def demo() -> None:
    #static: a flat run then a jump larger than any in-window std
    X = np.array([[[0.0], [0.0], [0.0], [10.0], [10.0]]])
    enc = DeltaEncoder()
    enc.fit(X)
    times, ids, num_neurons = enc._events(X)
    assert num_neurons == 2
    assert len(times[0]) == 1 and times[0][0] == 3.0 and ids[0][0] == 0

    X_down = np.array([[[10.0], [10.0], [10.0], [0.0], [0.0]]])
    enc2 = DeltaEncoder()
    enc2.fit(X_down)
    assert enc2._events(X_down)[1][0][0] == 1, "a drop must fire the odd 'down' channel"

    #accumulator: three +0.2 steps against a 0.5 threshold fire once, static fires never
    ramp = np.array([[[0.0], [0.2], [0.4], [0.6], [0.8]]])
    acc = DeltaEncoder(accumulate=True)
    acc._threshold = np.array([0.5])
    static = DeltaEncoder()
    static._threshold = np.array([0.5])
    assert len(static._events(ramp)[0][0]) == 0
    assert len(acc._events(ramp)[0][0]) == 1
    print("delta encoder demo OK")


if __name__ == "__main__":
    demo()
