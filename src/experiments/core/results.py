"""Persist one experiment run to a timestamped directory with the config that made it."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import pandas as pd

from core.config import ExperimentConfig

RESULTS_ROOT = (Path(__file__).resolve().parents[3] / "artifacts" / "results" /
                "runs")


def save_run(config: ExperimentConfig,
             results: dict[str, dict[str, float]],
             predictions: dict[str, pd.Series] | None = None,
             diagnostics: dict[str, Any] | None = None,
             actual: pd.Series | None = None,
             requested_models: Sequence[str] | None = None,
             root: Path | None = None) -> Path:
    """Write one run to a timestamped, arm-identifying directory and return it.

    Encoding/algorithm/L/seed are in the name because the matrix produces dozens of
    runs that differ only in those. They are meaningless for non-SNN models, but the
    authoritative record is config.json either way.  ``requested_models`` records
    the CLI request and ``scored_models`` the rows actually written, because one
    directory can hold several fitted models.
    """
    root = RESULTS_ROOT if root is None else root
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    prefix = (f"{stamp}_{config.horizon}_{config.target}_{config.iv_leg}"
              f"_{config.encoding}_{config.algorithm}"
              f"_L{config.sequence_length}")
    #A UUID prevents two processes with identical configs and clocks from claiming
    #the same directory.  mkdir(exist_ok=False) makes ownership atomic rather than
    #letting the second process overwrite the first run's evidence.
    for _ in range(10):
        run_id = uuid4().hex
        #Keep the seed as the suffix for existing human/tooling conventions while
        #putting the collision-proof identifier before it.
        run_dir = root / f"{prefix}_{run_id}_s{config.seed}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            continue
    else:  # pragma: no cover - requires ten UUID collisions or a hostile filesystem
        raise RuntimeError("could not allocate a unique experiment result directory")

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()}
    cfg["run_id"] = run_id
    cfg["requested_models"] = list(requested_models or tuple(results))
    cfg["scored_models"] = list(results)
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
