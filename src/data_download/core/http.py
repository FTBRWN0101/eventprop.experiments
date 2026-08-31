"""A small, reusable HTTP downloader shared by file-based sources."""

from __future__ import annotations

import logging
from pathlib import Path

import requests

from core.base import Dataset, FetchResult

logger = logging.getLogger("data-download.http")

#CBOE and Stooq reject the default requests user agent
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; eventprop-experiments/1.0; +data-download pipeline)"
)


class Downloader:
    """Streamed, connection-pooled file downloader."""

    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        timeout: float = 30.0,
        chunk_size: int = 1 << 16,
    ) -> None:
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", _DEFAULT_USER_AGENT)
        self.timeout = timeout
        self.chunk_size = chunk_size

    def fetch(
        self,
        dataset: Dataset,
        dest_dir: Path,
        *,
        source: str = "",
        skip_existing: bool = True,
    ) -> FetchResult:
        """Download *dataset* into *dest_dir*, skipping a non-empty existing file."""
        dest = dest_dir / dataset.filename

        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            size = dest.stat().st_size
            logger.info("  [cached] %s (%s bytes)", dest.name, f"{size:,}")
            return FetchResult(source, dataset.key, dest, bytes=size)

        logger.info("  [download] %s -> %s", dataset.url, dest.name)
        with self.session.get(dataset.url, stream=True, timeout=self.timeout) as resp:
            resp.raise_for_status()
            self._stream_to_file(resp, dest)

        size = dest.stat().st_size
        logger.info("  [saved] %s (%s bytes)", dest.name, f"{size:,}")
        return FetchResult(source, dataset.key, dest, bytes=size)

    def _stream_to_file(self, resp: requests.Response, dest: Path) -> None:
        """Stream a response body to *dest* via a temp file (atomic-ish replace)."""
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=self.chunk_size):
                if chunk:
                    handle.write(chunk)
        tmp.replace(dest)

    def close(self) -> None:
        self.session.close()
