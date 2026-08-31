"""Daily S&P 500 prices from Yahoo, for realised vol. No API key needed."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import pandas as pd

#importable when run as a script
_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from core.base import DataSource, Dataset, FetchResult  # noqa: E402
from core.config import Config  # noqa: E402

_KEEP_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]


class IndexReturnsSource(DataSource):
    """Free daily index prices (S&P 500) as a realised-volatility input."""

    name = "index_returns"
    description = "Daily S&P 500 index prices for realised-volatility computation (free)"

    #output key -> Yahoo ticker
    TICKERS: dict[str, str] = {"spx_daily": "^GSPC"}
    PERIOD = "max"      #full history
    INTERVAL = "1d"     #daily bars

    @property
    def datasets(self) -> list[Dataset]:  # type: ignore[override]
        return [
            Dataset(key=key, url=f"yfinance:{ticker}", filename=f"{key}.csv")
            for key, ticker in self.TICKERS.items()
        ]

    def available(self) -> bool:
        return importlib.util.find_spec("yfinance") is not None

    def skip_reason(self) -> str:
        return "the 'yfinance' package is not installed, run 'pip install yfinance'"

    def fetch(self) -> list[FetchResult]:
        import yfinance as yf

        results: list[FetchResult] = []
        for key, ticker in self.TICKERS.items():
            self.logger.info("  [download] %s via yfinance", ticker)
            raw = yf.Ticker(ticker).history(
                period=self.PERIOD, interval=self.INTERVAL, auto_adjust=False,
            )
            if raw.empty:
                raise RuntimeError(f"yfinance returned no rows for {ticker!r}")
            frame = _tidy(raw)

            dest = self.save_dir / f"{key}.csv"
            frame.to_csv(dest, index=False)
            self.logger.info("  [saved] %s (%s rows)", dest.name, f"{len(frame):,}")
            results.append(FetchResult(self.name, key, dest, rows=len(frame),
                                       bytes=dest.stat().st_size))
        return results


def _tidy(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise a yfinance OHLCV frame to lower_snake columns with a ``date`` string."""
    frame = raw.reset_index()
    frame.columns = [str(col).lower().replace(" ", "_") for col in frame.columns]
    date_col = "date" if "date" in frame.columns else frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col], utc=True).dt.strftime("%Y-%m-%d")
    frame = frame.rename(columns={date_col: "date"})
    return frame[[col for col in _KEEP_COLUMNS if col in frame.columns]]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    IndexReturnsSource(Config.load()).run()


if __name__ == "__main__":
    main()
