"""OptionMetrics SPX vol surface and underlying prices, per year, from WRDS."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

#importable when run as a script
_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from core.base import Dataset, FetchResult  # noqa: E402
from core.config import Config  # noqa: E402
from core.wrds import WrdsSourceBase  # noqa: E402

if TYPE_CHECKING:
    import pandas as pd


class OptionMetricsSource(WrdsSourceBase):
    """SPX implied-volatility surface and underlying prices via WRDS OptionMetrics."""

    name = "optionmetrics"
    description = ("OptionMetrics IvyDB US: SPX implied-vol surface + underlying, "
                   "1996+ (institutional; requires WRDS login)")

    LIBRARY: ClassVar[str] = "optionm"
    #SPX security id in OptionMetrics
    SECID: ClassVar[int] = 108105

    #(table prefix, output stem, columns) per year-family
    TABLES: ClassVar[tuple[tuple[str, str, tuple[str, ...]], ...]] = (
        ("vsurfd", "spx_vsurf",
         ("date", "days", "delta", "cp_flag", "impl_volatility", "impl_strike",
          "impl_premium", "dispersion")),
        ("secprd", "spx_secpr",
         ("date", "open", "high", "low", "close", "volume", "return")),
    )

    @property
    def datasets(self) -> list[Dataset]:  # type: ignore[override]
        return [Dataset(key=stem, url=f"wrds:{self.LIBRARY}.{prefix}<year>",
                        filename=f"{stem}_<year>.csv")
                for prefix, stem, _ in self.TABLES]

    def fetch(self) -> list[FetchResult]:
        connection = self._connect()
        try:
            results: list[FetchResult] = []
            for prefix, stem, columns in self.TABLES:
                years = self._available_years(connection, prefix)
                self.logger.info("  %s: %d yearly tables (%d-%d)",
                                 prefix, len(years), years[0], years[-1])
                for year in years:
                    results.append(
                        self._fetch_year(connection, prefix, stem, columns, year))
            return results
        finally:
            connection.close()

    def _available_years(self, connection, prefix: str) -> list[int]:
        """Years for which ``<prefix><year>`` exists, ascending."""
        pattern = re.compile(rf"^{prefix}(\d{{4}})$")
        years = [int(m.group(1))
                 for table in connection.list_tables(library=self.LIBRARY)
                 if (m := pattern.match(table))]
        if not years:
            raise RuntimeError(
                f"no {self.LIBRARY}.{prefix}<year> tables visible, "
                f"check the WRDS subscription covers OptionMetrics")
        return sorted(years)

    def _fetch_year(self, connection, prefix: str, stem: str,
                    columns: tuple[str, ...], year: int) -> FetchResult:
        """Query one year's table for SPX and write it, skipping an existing file."""
        dest = self.save_dir / f"{stem}_{year}.csv"
        if dest.is_file() and dest.stat().st_size > 0:
            self.logger.info("  [cached] %s", dest.name)
            return FetchResult(self.name, f"{stem}_{year}", dest,
                               bytes=dest.stat().st_size)

        frame = self._query(connection, prefix, columns, year)
        frame.to_csv(dest, index=False)
        self.logger.info("  [saved] %s (%s rows, %s bytes)", dest.name,
                         f"{len(frame):,}", f"{dest.stat().st_size:,}")
        return FetchResult(self.name, f"{stem}_{year}", dest, rows=len(frame),
                           bytes=dest.stat().st_size)

    def _query(self, connection, prefix: str, columns: tuple[str, ...],
               year: int) -> "pd.DataFrame":
        selected = ", ".join(columns)
        sql = (f"SELECT {selected} FROM {self.LIBRARY}.{prefix}{year} "
               f"WHERE secid = %(secid)s ORDER BY date")
        return connection.raw_sql(sql, params={"secid": self.SECID},
                                  date_cols=["date"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OptionMetricsSource(Config.load()).run()


if __name__ == "__main__":
    main()
