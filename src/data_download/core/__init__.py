"""Core framework shared by every data source."""

from core.base import CredentialedSource, DataSource, Dataset, FetchResult
from core.config import Config
from core.http import Downloader
from core.registry import discover, registered

__all__ = [
    "Config",
    "CredentialedSource",
    "DataSource",
    "Dataset",
    "Downloader",
    "FetchResult",
    "discover",
    "registered",
]
