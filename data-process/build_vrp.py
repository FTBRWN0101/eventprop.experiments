"""Orchestrator: build the VRP train/test panels from the downloaded raw CSVs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

#importable when run as a script
_PIPELINE_ROOT = Path(__file__).resolve().parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from core import registry  # noqa: E402
from core.base import FeatureBuilder, Horizon  # noqa: E402
from core.config import ProcessConfig  # noqa: E402
from core.loaders import RawData  # noqa: E402
from assembler import DatasetAssembler, HorizonPanel  # noqa: E402


class VrpPipeline:
    """Wires raw loading, feature discovery, assembly and output together."""

    def __init__(self, config: ProcessConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("data-process.build_vrp")

    def builders(self) -> list[FeatureBuilder]:
        """Instantiate every discovered feature builder against the config."""
        classes = registry.discover(_PIPELINE_ROOT / "features")
        return [cls(self.config) for cls in classes]

    def run(self) -> list[HorizonPanel]:
        """Build, write and summarise every configured horizon."""
        raw = RawData.load(self.config)
        self.logger.info("loaded raw panel: %d rows, columns: %s",
                         len(raw.frame), ", ".join(raw.columns))

        builders = self.builders()
        self.logger.info("feature builders: %s",
                         ", ".join(b.name for b in builders))

        assembler = DatasetAssembler(self.config, builders)
        panels: list[HorizonPanel] = []
        for horizon in self.config.horizons:
            panel = assembler.write(assembler.assemble(raw, horizon))
            self._summarise(panel)
            panels.append(panel)
        return panels

    def _summarise(self, panel: HorizonPanel) -> None:
        self.logger.info("%s (%d-day)", panel.horizon.name, panel.horizon.days)
        self.logger.info("  span:     %s", panel.span)
        self.logger.info("  rows:     %d total (%d train / %d test)",
                         len(panel.full), len(panel.train), len(panel.test))
        self.logger.info("  features: %d (%s)", len(panel.feature_columns),
                         ", ".join(panel.feature_columns))
        self.logger.info("  targets:  %s", ", ".join(panel.target_columns))
        self.logger.info("  written:  %s", panel.paths["full"].parent)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VRP train/test panels.")
    parser.add_argument("--split-date", default=None,
                        help="First out-of-sample date (default: config default).")
    parser.add_argument("--horizons", nargs="*", default=None,
                        help="Subset of horizon names to build (default: all).")
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> ProcessConfig:
    overrides: dict[str, object] = {}
    if args.split_date:
        overrides["split_date"] = args.split_date
    if args.horizons:
        wanted = set(args.horizons)
        selected = tuple(h for h in ProcessConfig().horizons if h.name in wanted)
        if not selected:
            raise SystemExit(f"no known horizons match {sorted(wanted)}")
        overrides["horizons"] = selected
    return ProcessConfig.load(**overrides)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    VrpPipeline(_config_from_args(args)).run()


if __name__ == "__main__":
    main()
