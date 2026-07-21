"""The contract every feature builder implements."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar

import pandas as pd

from core import registry

if TYPE_CHECKING:
    from core.config import ProcessConfig
    from core.loaders import RawData


@dataclass(frozen=True)
class Horizon:
    """A named forecast horizon measured in trading days."""

    name: str
    days: int

    def __post_init__(self) -> None:
        if self.days <= 0:
            raise ValueError(f"Horizon {self.name!r} must have positive days, got {self.days}")


class FeatureBuilder(ABC):
    """Abstract base for every feature builder."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    requires: ClassVar[tuple[str, ...]] = ()
    #rows can stay even when these are NaN
    optional: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        registry.register(cls)

    def __init__(self, config: "ProcessConfig") -> None:
        self.config = config

    @cached_property
    def logger(self) -> logging.Logger:
        return logging.getLogger(f"data-process.{self.name or type(self).__name__}")

    @abstractmethod
    def build(self, raw: "RawData") -> pd.DataFrame:
        """Return a daily, date-indexed frame of this builder's feature columns.

        Leave gaps as NaN; the assembler decides the usable sample.
        """

    def run(self, raw: "RawData") -> pd.DataFrame:
        """Validate inputs, build the columns, and sanity-check the result."""
        missing = [name for name in self.requires if name not in raw.columns]
        if missing:
            raise KeyError(
                f"feature {self.name!r} requires missing raw series: {', '.join(missing)}"
            )
        frame = self.build(raw)
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(f"feature {self.name!r} must return a DatetimeIndex frame")
        return frame
