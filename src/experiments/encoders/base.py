"""The contract every spike encoder implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from core.registry import Registry

ENCODERS: Registry["Encoder"] = Registry("encoder")


@dataclass
class EncodedInput:
    """What an :class:`Encoder` hands the network for one split."""

    kind: str          #"rate" | "spikes" | "dense"
    data: object        #[N,T,F] for rate/dense; list[PreprocessedSpikes] for spikes
    num_neurons: int    #input population width, may exceed n_features


class Encoder(ABC):
    """Abstract base for every input-encoding strategy."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    kind: ClassVar[str] = "spikes"
    #what the encoder wants: unit, zscore or raw
    input_space: ClassVar[str] = "unit"

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        ENCODERS.register(cls)

    #set by encoders that need the causal scale; only adaptive delta uses it
    wants_scale: ClassVar[bool] = False

    @abstractmethod
    def fit(self, X: np.ndarray, scale: np.ndarray | None = None) -> None:
        """Fit encoder statistics (value range, adaptive thresholds, ...) on train windows."""

    @abstractmethod
    def encode(self, X: np.ndarray, scale: np.ndarray | None = None) -> EncodedInput:
        """Encode standardised windows ``X[N, T, F]`` into network input."""


def rasterise(times_list: list[np.ndarray], ids_list: list[np.ndarray],
              num_neurons: int, timesteps: int) -> np.ndarray:
    """Turn pure encoder events into a binary ``[N, T, neurons]`` representation.

    GeNN receives an event at the first integer timestep not earlier than its event
    time, hence ``ceil`` rather than rounding. Multiple events in one bin remain one
    binary activation for the non-spiking control.
    """
    if len(times_list) != len(ids_list):
        raise ValueError("times_list and ids_list must contain one row per example")
    if num_neurons < 1 or timesteps < 1:
        raise ValueError("num_neurons and timesteps must be positive")
    out = np.zeros((len(times_list), timesteps, num_neurons), dtype=np.float32)
    for row, (times, ids) in enumerate(zip(times_list, ids_list)):
        times = np.asarray(times)
        ids = np.asarray(ids)
        if times.shape != ids.shape:
            raise ValueError("event times and ids must have the same shape")
        steps = np.ceil(times).astype(np.int64)
        valid = ((steps >= 0) & (steps < timesteps) & (ids >= 0)
                 & (ids < num_neurons))
        out[row, steps[valid], ids[valid]] = 1.0
    return out
