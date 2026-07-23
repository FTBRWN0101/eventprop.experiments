"""Population encoding: Gaussian receptive fields across the value range."""

from __future__ import annotations

import numpy as np

from encoders.base import Encoder, EncodedInput

NUM_CENTERS = 8
RESPONSE_THRESHOLD = 0.5  #minimum receptive-field response to spike


class PopulationEncoder(Encoder):
    """``NUM_CENTERS`` Gaussian-tuned neurons per feature, latency-coded by response."""

    name = "population"
    description = f"Gaussian receptive fields ({NUM_CENTERS}/feature) across the value range"
    kind = "spikes"

    def fit(self, X: np.ndarray) -> None:
        self._min = X.min(axis=(0, 1))
        self._max = X.max(axis=(0, 1))
        self._centers = np.linspace(0.0, 1.0, NUM_CENTERS)
        self._sigma = 1.0 / (NUM_CENTERS - 1)

    def _events(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], int]:
        """Pure-numpy (times, ids) event arrays per example. No mlGeNN dependency."""
        n, t, f = X.shape
        span = self._max - self._min
        span = np.where(span == 0, 1.0, span)
        norm = np.clip((X - self._min) / span, 0.0, 1.0)  #[N, T, F]

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

    def encode(self, X: np.ndarray) -> EncodedInput:
        from ml_genn.utils.data import preprocess_spikes

        times_list, ids_list, num_neurons = self._events(X)
        spikes = [preprocess_spikes(t, i, num_neurons)
                  for t, i in zip(times_list, ids_list)]
        return EncodedInput(kind="spikes", data=spikes, num_neurons=num_neurons)


def demo() -> None:
    #one feature at its max: only the top receptive field fires, at offset 0
    X = np.array([[[1.0]], [[0.0]]])  #two examples spanning the full range
    enc = PopulationEncoder()
    enc.fit(X)
    times_list, ids_list, num_neurons = enc._events(X)
    assert num_neurons == NUM_CENTERS
    #value=max: the last receptive field must fire at about t=0
    ids0 = ids_list[0]
    assert (NUM_CENTERS - 1) in ids0
    top_time = times_list[0][list(ids0).index(NUM_CENTERS - 1)]
    assert top_time < 0.1
    print("population encoder demo OK")


if __name__ == "__main__":
    demo()
