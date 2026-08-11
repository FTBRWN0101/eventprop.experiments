"""WRDS CRSP downloader for daily S&P 500 index returns. Needs an account."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

#importable when run as a script
_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from core.base import CredentialedSource, Dataset, FetchResult  # noqa: E402
from core.config import Config  # noqa: E402

if TYPE_CHECKING:
    import pandas as pd


class WrdsSource(CredentialedSource):
    """CRSP daily S&P 500 index returns via WRDS."""

    name = "wrds"
    description = "CRSP daily S&P 500 index returns (institutional; requires WRDS login)"
    required_secrets = ["WRDS_USERNAME", "WRDS_PASSWORD"]

    #daily S&P 500 index file; market returns live in crsp.dsi instead
    LIBRARY = "crsp"
    TABLE = "dsp500"
    COLUMNS = ["caldt", "spindx", "sprtrn"]
    DATE_COLUMN = "caldt"
    START_DATE = "1990-01-01"
    OUTPUT_FILENAME = "sp500_daily_returns.csv"

    #WRDS Postgres endpoint, for ~/.pgpass
    PG_HOST = "wrds-pgdata.wharton.upenn.edu"
    PG_PORT = 9737
    PG_DATABASE = "wrds"

    @property
    def datasets(self) -> list[Dataset]:  # type: ignore[override]
        return [Dataset(key="sp500_daily",
                        url=f"wrds:{self.LIBRARY}.{self.TABLE}",
                        filename=self.OUTPUT_FILENAME)]

    def available(self) -> bool:
        #needs credentials and the wrds client
        return super().available() and importlib.util.find_spec("wrds") is not None

    def skip_reason(self) -> str:
        if not super().available():
            return super().skip_reason()
        return "the 'wrds' package is not installed, run 'pip install wrds'"

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

    def _credentials(self) -> tuple[str, str]:
        """WRDS username and password, username lower-cased.

        PAM is case-sensitive, so a mixed-case username is rejected.
        """
        return ((self.config.secret("WRDS_USERNAME") or "").lower(),
                self.config.secret("WRDS_PASSWORD") or "")

    def _connect(self):
        """Open a WRDS connection for headless auth."""
        try:
            import wrds
        except ModuleNotFoundError as exc:  #optional heavy dependency
            raise RuntimeError(
                "the 'wrds' package is not installed, run 'pip install wrds'"
            ) from exc
        username, password = self._credentials()
        self._ensure_pgpass()
        return wrds.Connection(wrds_username=username, wrds_password=password)

    @staticmethod
    def _pgpass_path() -> Path:
        """Location libpq actually reads: ``~/.pgpass`` on Unix, a different path on Windows."""
        if sys.platform == "win32":
            appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
            return appdata / "postgresql" / "pgpass.conf"
        return Path.home() / ".pgpass"

    def _ensure_pgpass(self) -> None:
        """Write/refresh this user's WRDS line in the platform's pgpass file, idempotently."""
        username, password = self._credentials()
        if not (username and password):
            return  #fall back to an existing pgpass or a prompt

        pgpass = self._pgpass_path()
        pgpass.parent.mkdir(parents=True, exist_ok=True)
        prefix = f"{self.PG_HOST}:{self.PG_PORT}:{self.PG_DATABASE}:{username}:"
        entry = f"{prefix}{password}"

        existing = pgpass.read_text().splitlines() if pgpass.exists() else []
        kept = [line for line in existing if line and not line.startswith(prefix)]
        pgpass.write_text("\n".join([*kept, entry]) + "\n")
        if sys.platform != "win32":
            pgpass.chmod(0o600)

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
