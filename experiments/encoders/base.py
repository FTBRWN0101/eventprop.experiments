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

    kind: str          #"rate" | "spikes"
    data: object        #[N,T,F] in [0,1] for rate; list[PreprocessedSpikes] for spikes
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
