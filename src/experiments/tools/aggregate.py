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

**Attempted is not the same as scored.** An arm that trips the D67 silent-neuron abort
writes no result directory at all, so it is invisible to a walk of the results tree: a
cell that ran eight seeds and scored two looks like a clean two-seed cell here, and sits
in the table beside a genuine eight-seed cell with nothing to tell them apart. The
attempted counts therefore come from the sweep manifests, which record every arm with
its ``returncode`` whether or not it produced anything, and any cell where ``n_scored``
is below ``n_attempted`` is flagged rather than quietly averaged. Manifests are optional:
without one a cell reports its attempted count as unknown instead of guessing.

QLIKE is summarised by the median, not the mean; see ``HEAVY_TAILED`` for why.

    python src/experiments/tools/aggregate.py
    python src/experiments/tools/aggregate.py --metrics r2_rv,qlike --min-seeds 8
    python src/experiments/tools/aggregate.py --manifests none
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

#printed by default; the rest still reach the CSV. qlike is here as a co-primary
#alongside r2_rv rather than as one more column, because the two rank models
#differently often enough that quoting either alone decides the conclusion.
DEFAULT_METRICS: tuple[str, ...] = ("r2_rv", "qlike", "gate", "dispersion")

#per-metric suffixes, so identity columns can be told from aggregated ones
SUFFIXES: tuple[str, ...] = ("_mean", "_std", "_count", "_median")

#metrics whose group figure is the median. QLIKE has a right tail that makes its mean a
#summary of nothing: over the 604 EventProp arms on disk the median is 1.44, the mean
#50.7, the 99th percentile 119.6 and the max 26836. The tail is not contamination to be
#filtered out - it is arms forecasting vol at or near zero, which the ratio term in
#QLIKE punishes without bound, and being under-dispersed is a real property of those
#forecasters. So the tail stays in the sample and the statistic reported over it is one
#the tail cannot drag: the median. The mean is printed beside it so the size of the gap
#is visible rather than something the reader has to be told about.
HEAVY_TAILED: tuple[str, ...] = ("qlike",)

#printed columns per metric; every suffix reaches the CSV either way
MEAN_COLUMNS: tuple[str, ...] = ("_mean", "_std", "_count")
MEDIAN_COLUMNS: tuple[str, ...] = ("_median", "_mean", "_count")

#the two co-primary metrics and the direction that wins on each, matching
#run_experiment.HEADLINE_METRICS. Kept as a local literal rather than imported: this
#tool reads finished CSVs and has no business pulling in the model registry to do it.
HEADLINE_METRICS: tuple[tuple[str, bool], ...] = (("r2_rv", True), ("qlike", False))

#sweep manifests beside the results root, which is where sweep.py writes them
MANIFEST_GLOB = "*_manifest.json"

#manifest record fields that identify the run rather than the arm. Everything else in a
#record is an axis of the grid, so stripping these leaves the cell the arm belongs to
#without this tool needing to know what any given sweep swept.
MANIFEST_BOOKKEEPING: tuple[str, ...] = ("key", "seed", "returncode", "seconds",
                                         "run_dir", "error")


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


@dataclass(frozen=True)
class Attempts:
    """What the sweep manifests say was attempted, indexed by results group.

    ``seeds`` maps a ``group_key`` to the seeds a manifest records as having been run
    for it, aborted arms included. ``ambiguous`` holds groups whose attempted count
    could not be settled, ``orphans`` the manifest cells that produced no scored run at
    all and so have no row in the table to be flagged on.
    """

    seeds: dict[str, set[str]] = field(default_factory=dict)
    #why a group's failed arms failed, as a set of abort / transient / unknown. Kept
    #beside the seeds because the two failure modes have opposite consequences for a
    #reported mean and only the manifest knows which a cell suffered.
    reasons: dict[str, set[str]] = field(default_factory=dict)
    ambiguous: frozenset[str] = frozenset()
    orphans: tuple[dict[str, Any], ...] = ()
    manifests: tuple[Path, ...] = ()

    def for_group(self, key: str, scored: set[str]) -> tuple[set[str], bool]:
        """``(seeds attempted, whether that is known)`` for one results group.

        Unknown rather than assumed-complete when no manifest covers the group: most of
        the results tree predates the sweep runner, and reporting those cells as
        "attempted == scored" would be inventing the very reassurance this is for.
        """
        if key not in self.seeds or key in self.ambiguous:
            return scored, False
        #a run on disk was attempted whether or not a manifest recorded it, so the union
        #keeps n_attempted >= n_scored instead of producing a negative abort count
        return self.seeds[key] | scored, True


def _failure_reason(record: dict[str, Any]) -> str:
    """One failed record's failure mode: ``abort``, ``transient`` or ``unknown``.

    The silent-neuron abort is selective, so a cell that lost seeds to it reports a mean
    over survivors. The error:4551 library failure strikes before training reaches an
    epoch and is missing at random, so it costs sample size and biases nothing. Calling
    both "aborted" would manufacture a caveat where none applies.
    """
    blob = f"{record.get('error', '')} {record.get('stderr_tail', '')}".lower()
    if "silent" in blob:
        return "abort"
    if "4551" in blob or "unable to load library" in blob:
        return "transient"
    return "unknown"


def _arm_identity(record: dict[str, Any]) -> str:
    """One manifest record's arm, with the seed and the per-run bookkeeping stripped.

    Read off the record's own fields rather than by parsing ``key``, so a sweep with
    axes this tool has never heard of (drive_sweep varies ``ratio`` and two weight
    scales) still groups correctly.
    """
    return json.dumps({k: _cell(v) for k, v in record.items()
                       if k not in MANIFEST_BOOKKEEPING}, sort_keys=True, default=str)


def _resolve_run_dir(raw: str, root: Path) -> Path | None:
    """The directory a manifest record points at, or None when it is not there.

    A manifest stores the absolute path the arm wrote to. That path dies the moment the
    repo moves machine, which is the same reason ``group_key`` ignores ``repo_root``, so
    the directory name is retried under *root* before giving up.
    """
    if not raw:
        return None
    for candidate in (Path(raw), root / Path(raw).name):
        if (candidate / "config.json").is_file():
            return candidate
    return None


def _group_of(record: dict[str, Any], root: Path) -> str | None:
    """The results group a manifest record landed in, or None if it left none.

    None for every arm that aborted, because a failed arm writes no result directory -
    which is the whole reason attempted counts cannot be read off the results tree - and
    None for a successful arm whose directory has since been moved or deleted.
    """
    if record.get("returncode") != 0:
        return None
    run_dir = _resolve_run_dir(str(record.get("run_dir") or ""), root)
    if run_dir is None:
        return None
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf8"))
    except (OSError, ValueError) as error:
        logger.warning("[aggregate] manifest points at %s but its config.json is "
                       "unreadable, so that arm cannot be placed: %s", run_dir, error)
        return None
    return group_key(config) if isinstance(config, dict) else None


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    """Records from one manifest, or an empty list if it is missing or malformed."""
    try:
        records = json.loads(path.read_text(encoding="utf8"))
    except (OSError, ValueError) as error:
        logger.warning("[aggregate] manifest %s is unreadable, so its arms cannot be "
                       "counted as attempted: %s", path.name, error)
        return []
    if not isinstance(records, list):
        logger.warning("[aggregate] manifest %s is not a list of records", path.name)
        return []
    return [r for r in records if isinstance(r, dict)]


def read_manifests(root: Path, paths: list[Path] | None = None) -> Attempts:
    """Attempted-seed counts per results group, read from the sweep manifests.

    A manifest record names an arm and its exit code; it does not name a results group,
    and a failed record has no run directory to look one up from. The association is
    therefore made a cell at a time, not a record at a time. Records sharing a manifest
    and every non-seed field are one cell of one sweep, and a cell's successful members
    do carry a ``run_dir``, so reading the ``config.json`` there gives the group the
    whole cell belongs to. The failures are then attributed to that group.

    The manifest file is part of the cell identity on purpose. ``window_fraction_0.05``
    and ``window_fraction_0.25`` record byte-identical arm fields and differ only in a
    ``--readout-window`` the record never mentions, so keying on the arm fields alone
    would merge four separate sweeps into one.

    Two things this cannot do. A cell where every seed failed has no successful member,
    so it cannot be placed at all; those are returned as ``orphans`` and reported
    separately, because there is no row in the table to flag them on. And a cell whose
    successful members land in *different* groups - which happens when the config schema
    gained a field partway through a sweep - cannot have its failures attributed to
    either, so those groups are marked ambiguous and report an unknown attempted count
    rather than a wrong one.
    """
    paths = sorted(root.rglob(MANIFEST_GLOB)) if paths is None else list(paths)
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path in paths:
        for record in _read_manifest(path):
            cells.setdefault((str(path), _arm_identity(record)), []).append(record)

    seeds: dict[str, set[str]] = {}
    reasons: dict[str, set[str]] = {}
    ambiguous: set[str] = set()
    orphans: list[dict[str, Any]] = []
    for (manifest, identity), records in sorted(cells.items()):
        placed: dict[str, str] = {}
        unplaced: set[str] = set()
        for record in records:
            seed = str(_cell(record.get("seed")))
            key = _group_of(record, root)
            if key is None:
                unplaced.add(seed)
            else:
                placed[seed] = key
                seeds.setdefault(key, set()).add(seed)
            if record.get("returncode") not in (0, None):
                for group in {k for k in placed.values()} or set():
                    reasons.setdefault(group, set()).add(_failure_reason(record))

        groups = set(placed.values())
        name = Path(manifest).name
        if not groups:
            orphans.append({"manifest": name, "arm": identity,
                            "attempted": len(records), "scored": 0})
            continue
        if len(groups) == 1:
            seeds[next(iter(groups))] |= unplaced
            continue
        #one sweep cell, several config groups: the schema drifted mid-sweep, so the
        #table already splits this cell in two and the failures belong to neither half
        differing = varying_fields([json.loads(key) for key in groups])
        logger.warning("[aggregate] %s cell %s spans %d config groups differing in %s. "
                       "The sweep ran it as one cell, so its rows here are each a slice "
                       "of it%s", name, identity, len(groups), differing or "nothing",
                       ", and its failed seeds cannot be attributed to either"
                       if unplaced else "")
        if unplaced:
            ambiguous |= groups

    return Attempts(seeds=seeds, reasons=reasons, ambiguous=frozenset(ambiguous),
                    orphans=tuple(orphans), manifests=tuple(paths))


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


def _check_attempts(index: int, label: str, scored: set[str], attempted: set[str],
                    known: bool, reasons: set[str] | None = None) -> None:
    """Warn when a group scored fewer seeds than the sweep started.

    Reported next to a cell that scored everything it ran, a partial cell is
    survivorship bias: what is being averaged is the seeds that survived, and the ones
    that aborted are gone from the tree rather than recorded as zeros. Saying so is the
    whole point, so this is a warning and not a debug line.
    """
    if not known or scored >= attempted:
        return
    missing = sorted(attempted - scored, key=str)
    #not every missing seed is survivorship bias, and calling them all aborts would
    #manufacture a caveat where none applies. A silent-neuron abort is selective: the
    #seeds that survive are the ones that did not collapse, so the mean is over
    #survivors. The error:4551 library failure is not selective; it strikes before
    #training reaches an epoch and is missing at random, so the mean is unbiased and
    #only the sample is smaller. The manifest carries the reason, so it is read rather
    #than assumed.
    reasons = reasons or {"unknown"}
    if reasons == {"abort"}:
        note = ("Those seeds hit the silent-neuron abort, which is selective, so every "
                "figure in this row is a mean over the seeds that survived.")
    elif reasons == {"transient"}:
        note = ("Those seeds hit the transient library failure before training began, "
                "which is missing at random, so the figures are unbiased on a smaller "
                "sample. Rerun with --retry-failed to close the gap.")
    else:
        note = ("Those seeds failed for mixed or unrecorded reasons, so whether the "
                "figures are survivorship biased cannot be settled from the manifest.")
    logger.warning("[aggregate] group %d (%s) is PARTIAL: %d of %d attempted seed(s) "
                   "scored, missing %s. %s",
                   index, label, len(scored), len(attempted), missing, note)


def _group_rows(index: int, members: list[Run], fields: list[str], seeds: list[Any],
                attempted: set[str], known: bool) -> list[dict[str, Any]]:
    """One row per model in one group, aggregated over that group's seeds."""
    stacked = pd.concat([run.metrics for run in members]).select_dtypes(include="number")
    #sort=False keeps the model order metrics.csv was written in
    by_model = stacked.groupby(level=0, sort=False)
    mean, std = by_model.mean(), by_model.std(ddof=1)
    count, median = by_model.count(), by_model.median()
    identity = {name: _cell(members[0].config.get(name)) for name in fields}
    scored = {str(seed) for seed in seeds}
    missing = sorted(attempted - scored, key=str) if known else []

    rows = []
    for model in mean.index:
        row: dict[str, Any] = {
            "group": index, **identity, "model": str(model),
            #n_seeds is n_scored under its historical name, kept because existing
            #analysis reads it; n_scored is the name that says what it counts
            "n_seeds": len(scored), "n_runs": len(members),
            "n_scored": len(scored),
            "n_attempted": len(attempted) if known else None,
            "n_aborted": len(missing) if known else None,
            "missing_seeds": ",".join(missing) if known else "?",
            "seeds": ",".join(str(s) for s in sorted(seeds, key=str)),
            "silent_final": _silent_final(members, str(model)),
        }
        for column in stacked.columns:
            row[f"{column}_mean"] = float(mean.at[model, column])
            row[f"{column}_std"] = float(std.at[model, column])
            row[f"{column}_count"] = int(count.at[model, column])
            row[f"{column}_median"] = float(median.at[model, column])
        rows.append(row)
    return rows


def aggregate(runs: list[Run], min_seeds: int = MIN_SEEDS,
              attempts: Attempts | None = None) -> pd.DataFrame:
    """Group *runs* by config-without-seed and summarise each group over its seeds.

    *attempts* carries what the sweep manifests say was started. Without it every group
    reports its attempted count as unknown, which is the honest answer for a results
    tree with no manifest beside it and keeps the older two-argument call working.
    """
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
        scored = {str(seed) for seed in seeds}
        attempted, known = (attempts.for_group(key, scored) if attempts
                            else (scored, False))
        label = _label(members[0].config, fields)
        _check_seeds(index, label, seeds, min_seeds, members)
        _check_attempts(index, label, scored, attempted, known,
                        attempts.reasons.get(key) if attempts else None)
        rows.extend(_group_rows(index, members, fields, seeds, attempted, known))

    frame = pd.DataFrame(rows)
    #None means "no manifest could tell us", which is not the same number as zero, so
    #these stay nullable integers rather than becoming NaN floats
    for column in ("n_attempted", "n_aborted"):
        frame[column] = frame[column].astype("Int64")
    #the column is noise on a sweep of models that never report it
    if frame["silent_final"].isna().all():
        frame = frame.drop(columns="silent_final")
    return frame


def headline(frame: pd.DataFrame) -> pd.DataFrame:
    """Per group, the best model on each co-primary metric and whether they agree.

    Ranking is within a group and never across groups: the models in one group were
    scored on one test sample, whereas two groups can differ in horizon or target and
    have no common scale. r2_rv is read as a mean over seeds and QLIKE as a median, per
    HEAVY_TAILED. Non-finite values are excluded from ranking; in particular, QLIKE
    is NaN for a run with an invalid forward-vol forecast, so it cannot win a min()
    by accident.
    """
    rows = []
    for index in sorted(frame["group"].unique()):
        block = frame[frame["group"] == index]
        row: dict[str, Any] = {"group": int(index),
                               "n_scored": int(block["n_scored"].iloc[0]),
                               "n_attempted": block["n_attempted"].iloc[0]}
        winners = {}
        for metric, higher in HEADLINE_METRICS:
            column = f"{metric}_median" if metric in HEAVY_TAILED else f"{metric}_mean"
            #a results tree written before the metric existed simply has no column, and
            #that is a reason to say "undecided", not to take the report down
            usable = (block[["model", column]].replace([float("inf"), float("-inf")],
                                                       pd.NA).dropna()
                      if column in block.columns else block.iloc[0:0])
            if usable.empty:
                winners[metric] = None
                row[f"best_{metric}"], row[metric] = "none", float("nan")
                continue
            pick = usable[column].idxmax() if higher else usable[column].idxmin()
            winners[metric] = str(usable.at[pick, "model"])
            row[f"best_{metric}"] = winners[metric]
            row[metric] = float(usable.at[pick, column])
        named = [w for w in winners.values() if w is not None]
        row["headline"] = ("undecided" if len(named) < len(winners)
                           else "agree" if len(set(named)) == 1 else "DISAGREE")
        rows.append(row)
    return pd.DataFrame(rows)


#always printed, however little they vary: without these a row cannot be identified.
#n_scored and n_attempted are here so a partial cell cannot be read as a whole one.
IDENTITY_ALWAYS: tuple[str, ...] = ("group", "model", "n_scored", "n_attempted",
                                    "n_seeds", "n_runs", "seeds")


def shown_columns(frame: pd.DataFrame, wanted: list[str]) -> list[str]:
    """Identity columns, plus the printed statistics for each metric named in *wanted*.

    Config columns that hold one value across every row are dropped from the printed
    table. They are still in the CSV. A 160-run sweep varies three or four fields and
    holds twenty constant, so printing the constants buries the comparison.

    A heavy-tailed metric is printed as median/mean/count and every other one as
    mean/std/count. The CSV carries all four either way; this is about which number a
    reader's eye lands on.
    """
    identity = [c for c in frame.columns if not c.endswith(SUFFIXES)]
    shown = [c for c in identity
             if c in IDENTITY_ALWAYS or frame[c].astype(str).nunique(dropna=False) > 1]
    for metric in wanted:
        if any(f"{metric}{suffix}" not in frame.columns for suffix in SUFFIXES):
            logger.warning("[aggregate] no metric named %r in the loaded runs", metric)
            continue
        printed = MEDIAN_COLUMNS if metric in HEAVY_TAILED else MEAN_COLUMNS
        shown.extend(f"{metric}{suffix}" for suffix in printed)
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
    parser.add_argument("--manifests", default="auto",
                        help=f"Sweep manifests to read attempted counts from: 'auto' "
                             f"for every {MANIFEST_GLOB} under the results root, "
                             f"'none' to skip them and report every attempted count as "
                             f"unknown, or a comma-separated list of paths.")
    return parser.parse_args(argv)


def _manifest_paths(root: Path, argument: str) -> list[Path] | None:
    """Manifests named on the command line, or None to let read_manifests glob."""
    if argument.strip().lower() == "auto":
        return None
    return [Path(p.strip()) for p in argument.split(",") if p.strip()]


def _report_attempts(frame: pd.DataFrame, attempts: Attempts | None) -> None:
    """Say what the attempted counts rest on, and name every cell they flag."""
    if attempts is None:
        logger.info("\n[aggregate] manifests skipped, so n_attempted is unknown on "
                    "every row: a cell that aborted seeds cannot be told from one that "
                    "did not.")
        return

    logger.info("\n[aggregate] attempted counts from %d manifest(s): %s",
                len(attempts.manifests),
                ", ".join(p.name for p in attempts.manifests) or "none found")
    unknown = frame.loc[frame["n_attempted"].isna(), "group"].nunique()
    if unknown:
        logger.info("[aggregate] %d group(s) have no manifest covering them, so their "
                    "attempted counts are unknown rather than complete.", unknown)

    partial = frame[frame["n_attempted"].notna()
                    & (frame["n_scored"] < frame["n_attempted"])]
    if partial.empty:
        logger.info("[aggregate] no group scored fewer seeds than it attempted.")
    else:
        columns = ["group", "n_scored", "n_attempted", "n_aborted", "missing_seeds"]
        logger.info("\n[aggregate] PARTIAL cells - the row is a mean over the seeds "
                    "that survived, not over the seeds that ran:\n%s",
                    partial.drop_duplicates("group")[columns].to_string(index=False))

    if attempts.orphans:
        #these have no row to be flagged on: every seed aborted, so the cell is absent
        #from the table entirely and only the manifest knows it was ever tried
        logger.info("\n[aggregate] %d sweep cell(s) scored nothing at all, so they have "
                    "no row above:", len(attempts.orphans))
        for orphan in attempts.orphans:
            logger.info("[aggregate]   %s %s: 0 of %d seed(s) scored",
                        orphan["manifest"], orphan["arm"], orphan["attempted"])


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    root = Path(args.results_root)

    attempts = (None if args.manifests.strip().lower() == "none"
                else read_manifests(root, _manifest_paths(root, args.manifests)))
    frame = aggregate(load_runs(root), args.min_seeds, attempts)
    if frame.empty:
        logger.warning("[aggregate] nothing to aggregate under %s", root)
        return

    wanted = [m.strip() for m in args.metrics.split(",") if m.strip()]
    logger.info("\n%s", frame[shown_columns(frame, wanted)].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    #which model wins on each co-primary metric, and whether the two agree. A
    #disagreement is the one thing a table of means cannot show you by itself.
    logger.info("\n[aggregate] headline metrics per group, r2_rv by mean and QLIKE by "
                "median:\n%s", headline(frame).to_string(
                    index=False, float_format=lambda v: f"{v:.4f}"))
    _report_attempts(frame, attempts)

    out = Path(args.out) if args.out else root / "aggregate.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    logger.info("\n%d group(s), %d row(s), %d column(s) written to %s",
                frame["group"].nunique(), len(frame), frame.shape[1], out)


if __name__ == "__main__":
    main()
