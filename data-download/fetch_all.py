"""Run every registered data source and summarise what landed in data-save/."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

#importable when run as a script
_PIPELINE_ROOT = Path(__file__).resolve().parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from core.base import FetchResult  # noqa: E402
from core.config import Config  # noqa: E402
from core.registry import discover  # noqa: E402

logger = logging.getLogger("data-download")


def run_all(config: Config) -> list[FetchResult]:
    """Discover and run every source, returning the flattened results."""
    results: list[FetchResult] = []
    for source_cls in discover(_PIPELINE_ROOT):
        results.extend(source_cls(config).run())
    return results


def _summarise(results: list[FetchResult], config: Config) -> None:
    logger.info("\nsummary (saved under %s)", config.data_save_dir)
    downloaded = skipped = failed = 0
    for result in results:
        if result.error is not None:
            failed += 1
            logger.info("  [FAIL] %-14s %s: %s", result.source, result.key, result.error)
        elif result.skipped:
            skipped += 1
            logger.info("  [SKIP] %-14s (source not run)", result.source)
        else:
            downloaded += 1
            rows = "?" if result.rows is None else f"{result.rows:,}"
            logger.info("  [ OK ] %-14s %-18s %8s rows  %s",
                        result.source, result.key, rows, result.path.name)
    logger.info("  %d ok, %d skipped, %d failed", downloaded, skipped, failed)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = Config.load()
    results = run_all(config)
    _summarise(results, config)
    return 1 if any(r.error for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
