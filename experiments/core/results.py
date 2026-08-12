"""Persist one experiment run to a timestamped directory with the config that made it."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import ExperimentConfig

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"


def save_run(config: ExperimentConfig,
             results: dict[str, dict[str, float]],
             predictions: dict[str, pd.Series] | None = None,
             diagnostics: dict[str, Any] | None = None,
             actual: pd.Series | None = None,
             root: Path | None = None) -> Path:
    """Write one run to ``<root>/<timestamp>_<horizon>_<target>_<leg>/`` and return it."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = RESULTS_ROOT if root is None else root
    run_dir = root / f"{stamp}_{config.horizon}_{config.target}_{config.iv_leg}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf8")
    pd.DataFrame(results).T.to_csv(run_dir / "metrics.csv", index_label="model")
    if predictions:
        frame = pd.DataFrame(predictions)
        if actual is not None:
            #models score different dates, so align
            frame.insert(0, "actual", actual.reindex(frame.index))
        frame.to_csv(run_dir / "predictions.csv", index_label="date")
    if diagnostics:
        (run_dir / "diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, default=str), encoding="utf8")
    return run_dir
