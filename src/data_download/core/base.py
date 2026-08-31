"""The contract every data source implements."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from core import registry
from core.config import Config

if TYPE_CHECKING:
    from core.http import Downloader


@dataclass(frozen=True)
class Dataset:
    """One downloadable artifact within a source."""

    key: str
    url: str
    filename: str


@dataclass
class FetchResult:
    """Outcome of fetching a single artifact (or a skipped/failed source)."""

    source: str
    key: str
    path: Path
    rows: int | None = None
    bytes: int | None = None
    skipped: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the artifact was downloaded (or already present) without error."""
        return self.error is None and not self.skipped


class DataSource(ABC):
    """Abstract base for every data source."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    datasets: ClassVar[list[Dataset]] = []

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        registry.register(cls)

    def __init__(self, config: Config) -> None:
        self.config = config

    @cached_property
    def logger(self) -> logging.Logger:
        return logging.getLogger(f"data-download.{self.name or type(self).__name__}")

    @cached_property
    def downloader(self) -> "Downloader":
        #lazy import, avoids a cycle
        from core.http import Downloader

        return Downloader()

    @property
    def save_dir(self) -> Path:
        """The ``data-save/<name>/`` directory for this source (created on demand)."""
        return self.config.save_dir(self.name)

    def available(self) -> bool:
        """Whether preconditions (credentials, optional deps, ...) are satisfied."""
        return True

    def skip_reason(self) -> str:
        """Message explaining why the source is skipped when :meth:`available` is False."""
        return "preconditions not met"

    @abstractmethod
    def fetch(self) -> list[FetchResult]:
        """Download this source's artifacts into :attr:`save_dir`."""

    def run(self) -> list[FetchResult]:
        """Run the source, gating on :meth:`available` and isolating failures."""
        if not self.available():
            self.logger.warning("[skip] %s: %s", self.name, self.skip_reason())
            return [FetchResult(self.name, self.name, self.save_dir, skipped=True)]

        self.logger.info("fetching %s", self.name)
        try:
            results = self.fetch()
        except Exception as exc:  # noqa: BLE001 - deliberate: isolate per-source failures
            self.logger.exception("[error] %s: fetch failed", self.name)
            return [FetchResult(self.name, self.name, self.save_dir, error=str(exc))]

        for result in results:
            if result.rows is not None:
                self.logger.info("  %s: %s rows -> %s", result.key, f"{result.rows:,}",
                                 result.path.name)
        return results


class CredentialedSource(DataSource):
    """Base for sources needing secrets. Skipped, not failed, when they are missing."""

    required_secrets: ClassVar[list[str]] = []

    def available(self) -> bool:
        return self.config.has_secrets(self.required_secrets)

    def skip_reason(self) -> str:
        missing = [k for k in self.required_secrets if not self.config.secret(k)]
        return f"missing credentials: {', '.join(missing)} (see .env.example)"
