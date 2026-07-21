"""Auto-registration and discovery of data sources."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.base import DataSource

#insertion-ordered: sources run in discovery order
_REGISTRY: dict[str, type["DataSource"]] = {}

#framework code, not data sources
_NON_SOURCE_DIRS = frozenset({"core", "__pycache__"})


def register(source_cls: type["DataSource"]) -> None:
    """Register a concrete source class under its ``name``."""
    if inspect.isabstract(source_cls):
        return
    name = getattr(source_cls, "name", "")
    if not name:
        return
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not source_cls:
        raise ValueError(
            f"Duplicate data-source name {name!r}: "
            f"{existing.__module__} and {source_cls.__module__}"
        )
    _REGISTRY[name] = source_cls


def registered() -> list[type["DataSource"]]:
    """Return all registered source classes in discovery order."""
    return list(_REGISTRY.values())


def discover(package_dir: Path) -> list[type["DataSource"]]:
    """Import every ``<name>/fetch_<name>.py`` under *package_dir* to register sources."""
    for sub in sorted(package_dir.iterdir()):
        if not sub.is_dir() or sub.name in _NON_SOURCE_DIRS:
            continue
        if not (sub / "__init__.py").exists():
            continue
        for module_path in sorted(sub.glob("fetch_*.py")):
            importlib.import_module(f"{sub.name}.{module_path.stem}")
    return registered()
