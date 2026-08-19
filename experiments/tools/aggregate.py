"""Reduce a directory of single-seed runs to mean and standard deviation over seeds.

The seed sweep writes one directory per run, so a single reported figure lives spread
across dozens of them. This walks a results root, groups runs by their config with
``seed`` removed, and aggregates every metric column per model: mean, sample standard
deviation, and the number of seeds that actually contributed.

Group identity comes from ``config.json`` alone. The directory name carries only part of
the arm, so two runs whose names differ can still be the same arm, and two runs whose
names agree can differ in a field the name never shows. ``repo_root`` is ignored because
it is an absolute path that moves between machines without changing the experiment.

A metric that is NaN for some seeds is averaged over the seeds that have it, and its
``_count`` column says how many those were. That is the honest reading of a DM test that
could not be run against its own baseline, but it also means a mean can quietly rest on
fewer seeds than the row's seed count, so read the counts before quoting a figure.

    python experiments/tools/aggregate.py
    python experiments/tools/aggregate.py --metrics r2_rv,qlike --min-seeds 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  #noqa: E402

from core.results import RESULTS_ROOT  #noqa: E402

logger = logging.getLogger("aggregate")

#seed is what we aggregate over; repo_root is an absolute path that varies by machine
IGNORED_FIELDS: tuple[str, ...] = ("seed", "repo_root")

#the metrics contract: fewer than this many seeds is not a reportable figure
MIN_SEEDS = 8

#printed by default; the rest still reach the CSV
DEFAULT_METRICS: tuple[str, ...] = ("r2_rv", "gate", "dispersion")

#per-metric suffixes, so identity columns can be told from aggregated ones
SUFFIXES: tuple[str, ...] = ("_mean", "_std", "_count")


@dataclass(frozen=True)
class Run:
    """One run directory, already parsed."""

    path: Path
    config: dict[str, Any]
    metrics: pd.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _cell(value: Any) -> Any:
    """Render one config value as something hashable and printable."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value)


def group_key(config: dict[str, Any]) -> str:
    """Canonical JSON of *config* without the fields a seed sweep is allowed to vary."""
    trimmed = {k: v for k, v in config.items() if k not in IGNORED_FIELDS}
    return json.dumps(trimmed, sort_keys=True, default=str)


def varying_fields(configs: list[dict[str, Any]]) -> list[str]:
    """Config fields that differ across *configs*, so the table can tell groups apart."""
    names = sorted({k for c in configs for k in c} - set(IGNORED_FIELDS))
    return [n for n in names if len({_cell(c.get(n)) for c in configs}) > 1]


def _load_diagnostics(run_dir: Path) -> dict[str, Any]:
    """Read diagnostics.json if it is there and readable, else an empty dict."""
    path = run_dir / "diagnostics.json"
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf8"))
    except (OSError, ValueError) as error:
        logger.warning("[aggregate] %s has an unreadable diagnostics.json, "
                       "continuing without it: %s", run_dir.name, error)
        return {}
    return loaded if isinstance(loaded, dict) else {}


def load_run(run_dir: Path) -> Run | None:
    """Parse one run directory, or None if it is incomplete or malformed."""
    config_path, metrics_path = run_dir / "config.json", run_dir / "metrics.csv"
    if not config_path.is_file():
        return None
    if not metrics_path.is_file():
        logger.warning("[aggregate] skipping %s: config.json but no metrics.csv",
                       run_dir.name)
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf8"))
        metrics = pd.read_csv(metrics_path, index_col=0)
    except (OSError, ValueError) as error:
        #ValueError covers JSONDecodeError and the pandas parser and empty-file errors
        logger.warning("[aggregate] skipping %s: %s", run_dir.name, error)
        return None
    if not isinstance(config, dict) or metrics.empty:
        logger.warning("[aggregate] skipping %s: config.json is not an object, or "
                       "metrics.csv has no model rows", run_dir.name)
        return None
    return Run(path=run_dir, config=config, metrics=metrics,
               diagnostics=_load_diagnostics(run_dir))


def load_runs(root: Path) -> list[Run]:
    """Every readable run under *root*, in path order."""
    runs = [run for path in sorted(root.rglob("config.json"))
            if (run := load_run(path.parent)) is not None]
    if not runs:
        logger.warning("[aggregate] no run under %s has both config.json and "
                       "metrics.csv", root)
    return runs


def _label(config: dict[str, Any], fields: list[str]) -> str:
    """Short human name for a group, from the fields that differ across the sweep."""
    if not fields:
        return f"{config.get('horizon')}/{config.get('target')}/{config.get('iv_leg')}"
    return ", ".join(f"{name}={_cell(config.get(name))}" for name in fields)


def _check_seeds(index: int, label: str, seeds: list[Any], min_seeds: int,
                 members: list[Run]) -> None:
    """Warn when a group is too thin to report, or repeats a seed."""
    unique = sorted(set(seeds), key=str)
    if len(unique) < min_seeds:
        logger.warning("[aggregate] group %d (%s) has %d seed(s) %s, fewer than the %d "
                       "a reported figure needs; e.g. %s", index, label, len(unique),
                       unique, min_seeds, members[0].path.name)
    repeated = [s for s in unique if seeds.count(s) > 1]
    if repeated:
        logger.warning("[aggregate] group %d (%s) repeats seed(s) %s over %d runs, so "
                       "those seeds are counted twice; e.g. %s", index, label, repeated,
                       len(members), members[0].path.name)


def _silent_final(members: list[Run], model: str) -> float:
    """Mean over seeds of the last silent-neuron fraction recorded for *model*."""
    finals = []
    for run in members:
        entry = run.diagnostics.get(model)
        series = entry.get("silent_neuron_fraction") if isinstance(entry, dict) else None
        if isinstance(series, list) and series:
            finals.append(float(series[-1]))
    return sum(finals) / len(finals) if finals else float("nan")


def _group_rows(index: int, members: list[Run], fields: list[str],
                seeds: list[Any]) -> list[dict[str, Any]]:
    """One row per model in one group, aggregated over that group's seeds."""
    stacked = pd.concat([run.metrics for run in members]).select_dtypes(include="number")
    #sort=False keeps the model order metrics.csv was written in
    by_model = stacked.groupby(level=0, sort=False)
    mean, std, count = by_model.mean(), by_model.std(ddof=1), by_model.count()
    identity = {name: _cell(members[0].config.get(name)) for name in fields}

    rows = []
    for model in mean.index:
        row: dict[str, Any] = {
            "group": index, **identity, "model": str(model),
            "n_seeds": len(set(seeds)), "n_runs": len(members),
            "seeds": ",".join(str(s) for s in sorted(seeds, key=str)),
            "silent_final": _silent_final(members, str(model)),
        }
        for column in stacked.columns:
            row[f"{column}_mean"] = float(mean.at[model, column])
            row[f"{column}_std"] = float(std.at[model, column])
            row[f"{column}_count"] = int(count.at[model, column])
        rows.append(row)
    return rows


def aggregate(runs: list[Run], min_seeds: int = MIN_SEEDS) -> pd.DataFrame:
    """Group *runs* by config-without-seed and summarise each group over its seeds."""
    if not runs:
        return pd.DataFrame()

    groups: dict[str, list[Run]] = {}
    for run in runs:
        groups.setdefault(group_key(run.config), []).append(run)
    fields = varying_fields([run.config for run in runs])

    rows: list[dict[str, Any]] = []
    for index, key in enumerate(sorted(groups)):
        members = groups[key]
        seeds = [_cell(run.config.get("seed")) for run in members]
        _check_seeds(index, _label(members[0].config, fields), seeds, min_seeds, members)
        rows.extend(_group_rows(index, members, fields, seeds))

    frame = pd.DataFrame(rows)
    #the column is noise on a sweep of models that never report it
    if frame["silent_final"].isna().all():
        frame = frame.drop(columns="silent_final")
    return frame


#always printed, however little they vary: without these a row cannot be identified
IDENTITY_ALWAYS: tuple[str, ...] = ("group", "model", "n_seeds", "n_runs", "seeds")


def shown_columns(frame: pd.DataFrame, wanted: list[str]) -> list[str]:
    """Identity columns, plus mean/std/count for each metric named in *wanted*.

    Config columns that hold one value across every row are dropped from the printed
    table. They are still in the CSV. A 160-run sweep varies three or four fields and
    holds twenty constant, so printing the constants buries the comparison.
    """
    identity = [c for c in frame.columns if not c.endswith(SUFFIXES)]
    shown = [c for c in identity
             if c in IDENTITY_ALWAYS or frame[c].astype(str).nunique(dropna=False) > 1]
    for metric in wanted:
        columns = [f"{metric}{suffix}" for suffix in SUFFIXES]
        if any(c not in frame.columns for c in columns):
            logger.warning("[aggregate] no metric named %r in the loaded runs", metric)
            continue
        shown.extend(columns)
    return shown


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", default=str(RESULTS_ROOT),
                        help="Directory of run directories to walk.")
    parser.add_argument("--out", default=None,
                        help="CSV path; defaults to aggregate.csv in the results root.")
    parser.add_argument("--min-seeds", type=int, default=MIN_SEEDS,
                        help=f"Warn below this many seeds in a group "
                             f"(default {MIN_SEEDS}).")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                        help="Comma-separated metrics to print. The CSV always carries "
                             "all of them.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    root = Path(args.results_root)

    frame = aggregate(load_runs(root), args.min_seeds)
    if frame.empty:
        logger.warning("[aggregate] nothing to aggregate under %s", root)
        return

    wanted = [m.strip() for m in args.metrics.split(",") if m.strip()]
    logger.info("\n%s", frame[shown_columns(frame, wanted)].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    out = Path(args.out) if args.out else root / "aggregate.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    logger.info("\n%d group(s), %d row(s), %d column(s) written to %s",
                frame["group"].nunique(), len(frame), frame.shape[1], out)


if __name__ == "__main__":
    main()
