"""The contract every forecaster implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd

from core.config import ExperimentConfig
from core.dataset import VrpDataset
from core.registry import Registry

MODELS: Registry["Forecaster"] = Registry("model")


class Forecaster(ABC):
    """Abstract base for every forecasting model."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        MODELS.register(cls)

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    @property
    def fit_tolerance_days(self) -> int:
        """Trading days this model's fitted range may sit inside the sample by (D81).

        Zero for a model that fits every training row. A sequence model cannot form
        a window until it holds L days of history, and a trainer wanting full batches
        drops the remainder, so its fitted range is a strict subset of the sample it
        was handed. That is a handicap rather than an advantage, so the DM guard
        admits it instead of refusing the comparison. It is not a licence to fit a
        different sample: a range reaching *beyond* the other is refused at any size.
        """
        return 0

    @abstractmethod
    def fit(self, data: VrpDataset) -> None:
        """Fit the model on the training split of *data*."""

    @abstractmethod
    def predict(self, data: VrpDataset, split: str) -> pd.Series:
        """Predict the target for *split*, indexed by that split's evaluation dates."""
