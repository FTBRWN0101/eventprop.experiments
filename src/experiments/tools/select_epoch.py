"""Select saved SNN epochs on validation data without retraining.

This tool deliberately emits two files instead of changing a run's ``metrics.csv``:
``epoch_curves.csv`` preserves every scored checkpoint, while ``epoch_selection.csv``
states the validation rule, selected checkpoint, and its held-out test score. The
original final-epoch run remains intact as provenance.

Only runs with ``holdout_val=true`` are eligible. Selecting by test performance would
turn a held-out test set into a tuning set, so ``--selection-metric`` must name a
validation column such as ``val_r2`` or ``val_qlike``.

    python src/experiments/tools/select_epoch.py \\
        --campaign-id final_weekly_20260831_v3_holdout --selection-metric val_r2
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from core.config import ExperimentConfig  # noqa: E402
from core.results import RESULTS_ROOT  # noqa: E402
from tools.epoch_curve import score_epochs  # noqa: E402

logger = logging.getLogger("select_epoch")

_PATH_FIELDS = frozenset(("repo_root", "encoder_checkpoint"))
_TUPLE_FIELDS = frozenset(("drop_features", "restore_features"))
_CONFIG_FIELDS = frozenset(field.name for field in fields(ExperimentConfig))


def config_from_run(path: Path) -> ExperimentConfig:
    """Reconstruct a run's immutable configuration, rejecting incomplete records."""
    loaded = json.loads((path / "config.json").read_text(encoding="utf8"))
    if not isinstance(loaded, dict):
        raise ValueError("config.json is not an object")
    values = {name: loaded[name] for name in _CONFIG_FIELDS if name in loaded}
    missing = _CONFIG_FIELDS - values.keys()
    if missing:
        raise ValueError(f"config.json lacks {', '.join(sorted(missing))}")
    for name in _PATH_FIELDS:
        if values[name] is not None:
            values[name] = Path(values[name])
    for name in _TUPLE_FIELDS:
        values[name] = tuple(values[name])
    return ExperimentConfig.load(**values)


def eligible_runs(root: Path, campaign_id: str) -> list[tuple[Path, ExperimentConfig]]:
    """Runs in the named holdout campaign, in a stable order."""
    matches: list[tuple[Path, ExperimentConfig]] = []
    for config_path in sorted(root.rglob("config.json")):
        try:
            raw = json.loads(config_path.read_text(encoding="utf8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            logger.warning("[select] skipping %s: %s", config_path.parent.name, error)
            continue
        #Most historic run records predate either campaign IDs or the full current
        #configuration. Filter them before reconstructing a config, so a precise
        #campaign request does not turn the terminal into warnings about old data.
        if not isinstance(raw, dict) or raw.get("campaign_id") != campaign_id:
            continue
        if "snn" not in raw.get("scored_models", []):
            continue
        try:
            config = config_from_run(config_path.parent)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            logger.warning("[select] skipping %s: %s", config_path.parent.name, error)
            continue
        if not config.holdout_val:
            raise ValueError(f"{config_path.parent} belongs to {campaign_id!r} but has no "
                             "validation holdout")
        matches.append((config_path.parent, config))
    return matches


def _select(frame: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.Series:
    """Best finite validation row, with the earlier epoch resolving an exact tie."""
    if metric not in frame:
        raise ValueError(f"checkpoint curve has no {metric!r} column")
    usable = frame.loc[pd.to_numeric(frame[metric], errors="coerce").map(math.isfinite)]
    if usable.empty:
        raise ValueError(f"checkpoint curve has no finite {metric!r} value")
    ordered = usable.sort_values([metric, "epoch"], ascending=[not higher_is_better, True])
    return ordered.iloc[0]


def select_campaign(root: Path, campaign_id: str, metric: str,
                    higher_is_better: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every checkpoint and return ``(curves, selections)`` for a campaign."""
    curves: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    for number, (run_dir, config) in enumerate(eligible_runs(root, campaign_id), start=1):
        logger.info("[select] %d: %s", number, run_dir.name)
        curve = score_epochs(config, ("val", "test")).copy()
        curve.insert(0, "run_dir", str(run_dir))
        curve.insert(1, "campaign_id", campaign_id)
        chosen = _select(curve, metric, higher_is_better)
        final = curve.loc[curve["epoch"].idxmax()]
        selections.append({
            "run_dir": str(run_dir),
            "campaign_id": campaign_id,
            "selection_metric": metric,
            "selection_direction": "max" if higher_is_better else "min",
            "selected_epoch": int(chosen["epoch"]),
            "final_epoch": int(final["epoch"]),
            **{f"selected_{column}": chosen[column] for column in curve.columns
               if column.endswith(("_r2", "_qlike"))},
            **{f"final_{column}": final[column] for column in curve.columns
               if column.endswith(("_r2", "_qlike"))},
        })
        curves.append(curve)
        gc.collect()
    return (pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
            pd.DataFrame(selections))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign-id", required=True,
                        help="Exact holdout campaign identifier to score.")
    parser.add_argument("--selection-metric", required=True,
                        help="Validation column used to choose the checkpoint, e.g. val_r2.")
    parser.add_argument("--lower-is-better", action="store_true",
                        help="Use for a loss such as val_qlike; the default maximises.")
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--out-dir", default=None,
                        help="Defaults to <results-root>/<campaign-id>_epoch_selection.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    args = _parse_args(argv)
    root = Path(args.results_root)
    runs = eligible_runs(root, args.campaign_id)
    if not runs:
        raise FileNotFoundError(f"no completed SNN runs for campaign {args.campaign_id!r}")
    logger.info("[select] %d eligible run(s) in %s", len(runs), args.campaign_id)
    if args.dry_run:
        for run_dir, _ in runs:
            logger.info("[select] would score %s", run_dir.name)
        return
    curves, selected = select_campaign(root, args.campaign_id, args.selection_metric,
                                       not args.lower_is_better)
    out_dir = (Path(args.out_dir) if args.out_dir else
               root / f"{args.campaign_id}_epoch_selection")
    out_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(out_dir / "epoch_curves.csv", index=False)
    selected.to_csv(out_dir / "epoch_selection.csv", index=False)
    logger.info("[select] %d selected run(s) written to %s", len(selected), out_dir)


if __name__ == "__main__":
    main()
