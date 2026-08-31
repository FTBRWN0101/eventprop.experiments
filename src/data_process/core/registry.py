"""Auto-registration and discovery of feature builders."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.base import FeatureBuilder

#insertion-ordered: features build in discovery order
_REGISTRY: dict[str, type["FeatureBuilder"]] = {}


def register(builder_cls: type["FeatureBuilder"]) -> None:
    """Register a concrete feature-builder class under its ``name``."""
    if inspect.isabstract(builder_cls):
        return
    name = getattr(builder_cls, "name", "")
    if not name:
        return
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not builder_cls:
        raise ValueError(
            f"Duplicate feature-builder name {name!r}: "
            f"{existing.__module__} and {builder_cls.__module__}"
        )
    _REGISTRY[name] = builder_cls


def registered() -> list[type["FeatureBuilder"]]:
    """Return all registered feature-builder classes in discovery order."""
    return list(_REGISTRY.values())


def discover(package_dir: Path, package: str = "features") -> list[type["FeatureBuilder"]]:
    """Import every module under *package_dir* to trigger feature registration."""
    for module_path in sorted(package_dir.glob("*.py")):
        if module_path.stem == "__init__":
            continue
        importlib.import_module(f"{package}.{module_path.stem}")
    return registered()
