"""CBOE index history downloader (VIX, VVIX, SKEW, term structure). Free CSVs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

#importable when run as a script
_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from core.base import DataSource, Dataset, FetchResult  # noqa: E402
from core.config import Config  # noqa: E402


class CboeSource(DataSource):
    """Free CBOE daily index history."""

    name = "cboe"
    description = "CBOE VIX/VVIX/SKEW and VIX term-structure daily history (free CSV)"

    #CDN with the history CSVs
    DEFAULT_BASE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices"

    #output key -> CBOE symbol
    SYMBOLS: dict[str, str] = {
        "vix_daily": "VIX",     #30-day S&P 500 implied vol (from 1990)
        "vvix_daily": "VVIX",   #implied vol of VIX options (from 2006)
        "skew_daily": "SKEW",   #tail-risk index (from 1990)
        "vix9d_daily": "VIX9D",  #9-day term point (from 2011)
        "vix3m_daily": "VIX3M",  #3-month term point (from 2011)
        "vix6m_daily": "VIX6M",  #6-month term point (from 2011)
    }

    @property
    def base_url(self) -> str:
        return self.config.secret("CBOE_BASE_URL", self.DEFAULT_BASE_URL)

    @property
    def datasets(self) -> list[Dataset]:  # type: ignore[override]
        """Build the artifact list from :attr:`SYMBOLS` and the active base URL."""
        return [
            Dataset(key=key, url=f"{self.base_url}/{symbol}_History.csv",
                    filename=f"{key}.csv")
            for key, symbol in self.SYMBOLS.items()
        ]

    def fetch(self) -> list[FetchResult]:
        results: list[FetchResult] = []
        for dataset in self.datasets:
            result = self.downloader.fetch(dataset, self.save_dir, source=self.name)
            result.rows = _count_data_rows(result.path)
            results.append(result)
        return results


def _count_data_rows(path: Path) -> int:
    """Count CSV data rows (total lines minus a one-line header), cheaply."""
    with path.open("rb") as handle:
        lines = sum(1 for _ in handle)
    return max(0, lines - 1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    source = CboeSource(Config.load())
    source.run()


if __name__ == "__main__":
    main()
