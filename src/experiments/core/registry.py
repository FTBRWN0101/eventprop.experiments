"""One generalised registry, reused for models and encoders."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name -> class registry for one family of plugins (e.g. ``"model"``)."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, cls: type[T]) -> None:
        """Register a concrete, named subclass; ignore abstract/anonymous ones."""
        if inspect.isabstract(cls):
            return
        name = getattr(cls, "name", "")
        if not name:
            return
        existing = self._items.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Duplicate {self.kind} name {name!r}: "
                f"{existing.__module__} and {cls.__module__}")
        self._items[name] = cls

    def get(self, name: str) -> type[T]:
        if name not in self._items:
            raise KeyError(
                f"unknown {self.kind} {name!r}; known: {sorted(self._items)}")
        return self._items[name]

    def names(self) -> list[str]:
        return list(self._items)

    def discover(self, package_dir: Path, package: str) -> "Registry[T]":
        """Import every module under *package_dir* to trigger registration."""
        for module_path in sorted(package_dir.glob("*.py")):
            if module_path.stem == "__init__":
                continue
            importlib.import_module(f"{package}.{module_path.stem}")
        return self
