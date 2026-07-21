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

    @abstractmethod
    def fit(self, data: VrpDataset) -> None:
        """Fit the model on the training split of *data*."""

    @abstractmethod
    def predict(self, data: VrpDataset, split: str) -> pd.Series:
        """Predict the target for *split*, indexed by that split's evaluation dates."""
