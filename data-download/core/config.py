"""Runtime configuration: filesystem layout and secret resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]   #<repo>/data-download
_REPO_ROOT = _PACKAGE_ROOT.parent                      #<repo>

DATA_SAVE_DIRNAME = "data-save"
ENV_FILENAME = ".env"


def _parse_env_file(path: Path) -> dict[str, str]:
    """Minimal ``KEY=VALUE`` parser so ``.env`` works without ``python-dotenv``."""
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _load_dotenv(env_path: Path) -> dict[str, str]:
    """Return key/values from *env_path*, preferring ``python-dotenv`` if installed."""
    if not env_path.is_file():
        return {}
    try:
        from dotenv import dotenv_values  # type: ignore import-not-found
    except ModuleNotFoundError:
        return _parse_env_file(env_path)
    return {k: v for k, v in dotenv_values(env_path).items() if v is not None}


@dataclass(frozen=True)
class Config:
    """Immutable view of paths and resolved secrets for a pipeline run."""

    repo_root: Path = _REPO_ROOT
    package_root: Path = _PACKAGE_ROOT
    secrets: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Config":
        """Build a config, merging ``.env`` values under real environment variables."""
        dotenv_values = _load_dotenv(_PACKAGE_ROOT / ENV_FILENAME)
        merged = {**dotenv_values, **os.environ}
        return cls(secrets=merged)

    @property
    def data_save_dir(self) -> Path:
        """Root directory all sources write under (``<repo>/data-save``)."""
        return self.repo_root / DATA_SAVE_DIRNAME

    def save_dir(self, source: str) -> Path:
        """Return ``data-save/<source>/``, creating it on first use."""
        path = self.data_save_dir / source
        path.mkdir(parents=True, exist_ok=True)
        return path

    def secret(self, key: str, default: str | None = None) -> str | None:
        """Return a secret by key, or *default* if unset/blank."""
        value = self.secrets.get(key)
        return value if value else default

    def has_secrets(self, keys: list[str]) -> bool:
        """True only if every key in *keys* resolves to a non-empty value."""
        return all(self.secret(key) for key in keys)
