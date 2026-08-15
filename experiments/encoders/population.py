"""Population encoding: Gaussian receptive fields tiling the value range.

Takes the rank-uniform representation, which tiles [0, 1] evenly.
"""

from __future__ import annotations

import numpy as np

from encoders.base import Encoder, EncodedInput

NUM_CENTERS = 8
RESPONSE_THRESHOLD = 0.5  #minimum receptive-field response to spike


class PopulationEncoder(Encoder):
    """``NUM_CENTERS`` Gaussian-tuned neurons per feature, tiling the unit interval."""

    name = "population"
    description = f"Gaussian receptive fields ({NUM_CENTERS}/feature) across the value range"
    kind = "spikes"
    input_space = "unit"

    def __init__(self) -> None:
        self._centers = np.linspace(0.0, 1.0, NUM_CENTERS)
        self._sigma = 1.0 / (NUM_CENTERS - 1)

    def fit(self, X: np.ndarray, scale: np.ndarray | None = None) -> None:
        """No-op: fields tile the unit interval, which the rank transform already targets."""

    def _events(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], int]:
        """Pure-numpy (times, ids) event arrays per example. No mlGeNN dependency."""
        n, t, f = X.shape
        norm = np.clip(X, 0.0, 1.0)

        #response[..., c]: closeness to receptive-field centre c
        response = np.exp(-((norm[..., None] - self._centers) ** 2) / (2 * self._sigma ** 2))
        offset = 1.0 - response  #stronger response -> earlier spike
        num_neurons = f * NUM_CENTERS

        times_list, ids_list = [], []
        for i in range(n):
            tt, ff, cc = np.nonzero(response[i] > RESPONSE_THRESHOLD)
            times_list.append((tt + offset[i, tt, ff, cc]).astype(np.float32))
            ids_list.append((ff * NUM_CENTERS + cc).astype(np.int64))
        return times_list, ids_list, num_neurons

    def encode(self, X: np.ndarray, scale: np.ndarray | None = None) -> EncodedInput:
        from ml_genn.utils.data import preprocess_spikes

        times_list, ids_list, num_neurons = self._events(X)
        spikes = [preprocess_spikes(t, i, num_neurons)
                  for t, i in zip(times_list, ids_list)]
        return EncodedInput(kind="spikes", data=spikes, num_neurons=num_neurons)
