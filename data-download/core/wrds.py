"""Shared WRDS connection handling for authenticated sources.

Abstract on purpose, so the registry skips it and only subclasses run.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import ClassVar

from core.base import CredentialedSource


class WrdsSourceBase(CredentialedSource):
    """Base for any source that queries WRDS over its Postgres endpoint."""

    required_secrets: ClassVar[list[str]] = ["WRDS_USERNAME", "WRDS_PASSWORD"]

    #WRDS Postgres endpoint, for the pgpass file
    PG_HOST: ClassVar[str] = "wrds-pgdata.wharton.upenn.edu"
    PG_PORT: ClassVar[int] = 9737
    PG_DATABASE: ClassVar[str] = "wrds"

    def available(self) -> bool:
        #needs credentials and the wrds client
        return super().available() and importlib.util.find_spec("wrds") is not None

    def skip_reason(self) -> str:
        if not super().available():
            return super().skip_reason()
        return "the 'wrds' package is not installed, run 'pip install wrds'"

    def _credentials(self) -> tuple[str, str]:
        """WRDS username and password, username lower-cased.

        PAM is case-sensitive, so a mixed-case username is rejected.
        """
        return ((self.config.secret("WRDS_USERNAME") or "").lower(),
                self.config.secret("WRDS_PASSWORD") or "")

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
