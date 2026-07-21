# data-download

Per-source download pipeline for the SNN volatility-forecasting project. Each source
lives in its own subfolder with a runnable fetch script; everything writes raw data to
the sibling [`data-save/`](../data-save) tree (`data-save/<source>/`).

> The proposal's Oxford-Man Realised Library is intentionally omitted - it was
> discontinued (~2022). Realised volatility is computed from the `index_returns`
> daily prices instead (or CRSP via `wrds`).

## Setup

```bash
pip install -r ../requirements.txt          # repo dependencies
cp .env.example .env                          # then fill in any credentials you have
```

`.env` is git-ignored and read automatically. Real environment variables take
precedence over `.env`. CBOE and Yahoo (`index_returns`) need **no key**; only WRDS
requires credentials.

## Running

```bash
# Everything (sources lacking credentials are skipped, not failed):
python data-download/fetch_all.py

# A single source:
python data-download/cboe/fetch_cboe.py
python data-download/index_returns/fetch_index_returns.py
python data-download/wrds_crsp/fetch_wrds_crsp.py
```

Output CSVs appear under `data-save/<source>/`. CBOE files are skipped if already
present; index prices are always re-pulled (the feed appends daily).

## Sources

- `cboe`: `vix_daily.csv`, `vvix_daily.csv`, `skew_daily.csv`, `vix9d/3m/6m_daily.csv` (auth: none)
- `index_returns`: `spx_daily.csv` (auth: none (yfinance))
- `wrds`: `sp500_daily_returns.csv` (auth: `WRDS_USERNAME` + `WRDS_PASSWORD`)

## Adding a new source

The pipeline is auto-discovering, so a new source is a drop-in - no central file to
edit:

1. Create `data-download/<name>/` with an `__init__.py` and a `fetch_<name>.py`.
2. In `fetch_<name>.py`, subclass `DataSource` (or `CredentialedSource` if it needs
   secrets), set `name = "<name>"`, declare its `datasets`, and implement `fetch()`.
3. That's it - `fetch_all.py` discovers and runs it automatically.

```python
from core.base import DataSource, Dataset, FetchResult

class MySource(DataSource):
    name = "mysource"
    description = "what it provides"

    @property
    def datasets(self):
        return [Dataset(key="thing", url="https://...", filename="thing.csv")]

    def fetch(self):
        return [self.downloader.fetch(ds, self.save_dir, source=self.name)
                for ds in self.datasets]
```

### Framework (`core/`)

- `base.py`: `DataSource` / `CredentialedSource` ABCs, `Dataset` & `FetchResult`
- `config.py`: `Config`: repo paths, `data-save` dirs, `.env` / secret loading
- `http.py`: `Downloader`: pooled session, streamed + skip-existing downloads
- `registry.py`: auto-registration (`__init_subclass__`) + `discover()`
