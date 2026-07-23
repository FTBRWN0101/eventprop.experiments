"""Rate encoding: value -> Poisson spike rate."""

from __future__ import annotations

import numpy as np

from encoders.base import Encoder, EncodedInput


class RateEncoder(Encoder):
    """Min-max scale each feature to ``[0, 1]`` for mlGeNN's ``PoissonInput``."""

    name = "rate"
    description = "Spike rate proportional to signal value (baseline, discards temporal structure)"
    kind = "rate"

    def fit(self, X: np.ndarray) -> None:
        self._min = X.min(axis=(0, 1))
        self._max = X.max(axis=(0, 1))

    def encode(self, X: np.ndarray) -> EncodedInput:
        span = self._max - self._min
        span = np.where(span == 0, 1.0, span)
        rates = np.clip((X - self._min) / span, 0.0, 1.0)
        return EncodedInput(kind="rate", data=rates, num_neurons=X.shape[2])


def demo() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 5, 3))
    enc = RateEncoder()
    enc.fit(X)
    out = enc.encode(X)
    assert out.kind == "rate"
    assert out.num_neurons == 3
    assert out.data.shape == X.shape
    assert out.data.min() >= 0.0 and out.data.max() <= 1.0
    #a constant feature must not divide by zero
    X_const = np.zeros((2, 5, 3))
    enc2 = RateEncoder()
    enc2.fit(X_const)
    out2 = enc2.encode(X_const)
    assert np.isfinite(out2.data).all()
    print("rate encoder demo OK")


if __name__ == "__main__":
    demo()
