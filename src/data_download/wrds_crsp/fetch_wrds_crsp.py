"""WRDS CRSP downloader for daily S&P 500 index returns. Needs an account."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

#importable when run as a script
_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from core.base import Dataset, FetchResult  # noqa: E402
from core.config import Config  # noqa: E402
from core.wrds import WrdsSourceBase  # noqa: E402

if TYPE_CHECKING:
    import pandas as pd


class WrdsSource(WrdsSourceBase):
    """CRSP daily S&P 500 index returns via WRDS."""

    name = "wrds"
    description = "CRSP daily S&P 500 index returns (institutional; requires WRDS login)"

    #daily S&P 500 index file; market returns live in crsp.dsi instead
    LIBRARY = "crsp"
    TABLE = "dsp500"
    COLUMNS = ["caldt", "spindx", "sprtrn"]
    DATE_COLUMN = "caldt"
    START_DATE = "1990-01-01"
    OUTPUT_FILENAME = "sp500_daily_returns.csv"

    @property
    def datasets(self) -> list[Dataset]:  # type: ignore[override]
        return [Dataset(key="sp500_daily",
                        url=f"wrds:{self.LIBRARY}.{self.TABLE}",
                        filename=self.OUTPUT_FILENAME)]

    def fetch(self) -> list[FetchResult]:
        dataset = self.datasets[0]
        connection = self._connect()
        try:
            frame = self._query(connection)
        finally:
            connection.close()

        dest = self.save_dir / dataset.filename
        frame.to_csv(dest, index=False)
        self.logger.info("  [saved] %s (%s bytes)", dest.name, f"{dest.stat().st_size:,}")
        return [FetchResult(self.name, dataset.key, dest, rows=len(frame),
                            bytes=dest.stat().st_size)]

    def _query(self, connection) -> "pd.DataFrame":
        columns = ", ".join(self.COLUMNS)
        sql = (
            f"SELECT {columns} FROM {self.LIBRARY}.{self.TABLE} "
            f"WHERE {self.DATE_COLUMN} >= %(start)s ORDER BY {self.DATE_COLUMN}"
        )
        return connection.raw_sql(sql, params={"start": self.START_DATE},
                                  date_cols=[self.DATE_COLUMN])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    WrdsSource(Config.load()).run()


if __name__ == "__main__":
    main()
