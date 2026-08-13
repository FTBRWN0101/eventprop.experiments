"""Rate encoding: value -> Poisson spike rate.

Takes the rank-uniform representation, so no scaling happens here.
"""

from __future__ import annotations

import numpy as np

from encoders.base import Encoder, EncodedInput


class RateEncoder(Encoder):
    """Pass rank-uniform features straight to mlGeNN's ``PoissonInput``."""

    name = "rate"
    description = "Spike rate proportional to signal value (baseline, discards temporal structure)"
    kind = "rate"
    input_space = "unit"

    def fit(self, X: np.ndarray) -> None:
        """No-op: the rank transform is fitted upstream, per column, on the daily frame."""

    def encode(self, X: np.ndarray) -> EncodedInput:
        #clip defensively, PoissonInput must never get a negative rate
        rates = np.clip(X, 0.0, 1.0)
        return EncodedInput(kind="rate", data=rates, num_neurons=X.shape[2])
