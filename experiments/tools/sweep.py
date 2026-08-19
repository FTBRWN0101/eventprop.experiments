"""Run the encoding grid across seeds, one subprocess per arm, resumable.

The encoding comparison needs every arm at enough seeds to report a spread, which is
160 runs and several hours. Three things follow from that length. A failed arm must not
take the remaining hours with it, so each arm runs in its own process and a non-zero
exit is recorded and stepped over. The sweep must survive being killed, so a manifest is
written after every arm and completed arms are skipped on the next invocation. And it
must be legible while it runs, so every arm is logged before it starts and again with
its duration when it ends.

One subprocess per arm rather than one process for all of them: GeNN builds and caches
native code per compiler, and a segfault in one arm would otherwise end the sweep.

    python experiments/tools/sweep.py --dry-run
    python experiments/tools/sweep.py
    python experiments/tools/sweep.py --limit 4
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  #noqa: E402

logger = logging.getLogger("sweep")

#the proposal's four strategies plus delta's static control
ENCODINGS: tuple[str, ...] = ("rate", "latency", "population", "delta", "delta_adaptive")
ALGORITHMS: tuple[str, ...] = ("eventprop", "eprop")
#contiguous and stated, so no one can suspect a seed was chosen after the fact
SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
#L=5 is the weekly horizon default; L=45 gives latency 45 resolvable spike times
#instead of 5, so its arm tests the encoding rather than the timebase
WINDOWS: tuple[int, ...] = (5, 45)

#baselines ride along in every run so each result directory carries its own DM against
#both benchmarks. They are deterministic, so the repetition costs seconds and buys a
#self-contained comparison.
MODELS = "snn,har_rv,garch,constant,persistence"

#D65 measured 28.9% of hidden neurons silent on latency, which trips the D67 abort.
#The arm runs anyway with the fraction recorded, per the decision to measure it rather
#than hide it. Every other encoding keeps the safeguard armed.
ALLOW_SILENT_ENCODINGS: tuple[str, ...] = ("latency",)

MANIFEST_NAME = "sweep_manifest.json"
RESULT_MARKER = "results written to "


@dataclass(frozen=True)
class Arm:
    """One cell of the grid."""

    encoding: str
    algorithm: str
    seed: int
    input_window: int

    @property
    def key(self) -> str:
        """Stable identifier, used for resume and for log lines."""
        return f"{self.encoding}_{self.algorithm}_L{self.input_window}_s{self.seed}"


def build_grid(encodings: tuple[str, ...] = ENCODINGS,
               algorithms: tuple[str, ...] = ALGORITHMS,
               seeds: tuple[int, ...] = SEEDS,
               windows: tuple[int, ...] = WINDOWS) -> list[Arm]:
    """Every arm of the grid, in a deterministic order."""
    return [Arm(encoding=e, algorithm=a, seed=s, input_window=w)
            for w, e, a, s in product(windows, encodings, algorithms, seeds)]


def arm_command(arm: Arm, python: str, target: str, iv_leg: str,
                horizon: str, models: str) -> list[str]:
    """The exact argv for one arm, including the safeguard override where it applies."""
    command = [python, str(_ROOT / "run_experiment.py"),
               "--models", models,
               "--horizon", horizon,
               "--target", target,
               "--iv-leg", iv_leg,
               "--encoding", arm.encoding,
               "--algorithm", arm.algorithm,
               "--seed", str(arm.seed),
               "--input-window", str(arm.input_window)]
    if arm.encoding in ALLOW_SILENT_ENCODINGS:
        command.append("--allow-silent-neurons")
    return command


def result_dir(*streams: str) -> str:
    """The run's own closing line, pulled back out of its output for the manifest.

    Both streams are searched because the harness reports through ``logging``, whose
    default handler writes to stderr, not stdout. Reading stdout alone silently left
    every manifest record without the directory it produced.

    The marker is matched anywhere in the line rather than at the start, so a log
    format carrying a timestamp or level prefix does not silently break this again.
    """
    for stream in streams:
        for line in reversed(stream.splitlines()):
            _, marker, tail = line.partition(RESULT_MARKER)
            if marker:
                return tail.strip()
    return ""


def run_arm(arm: Arm, command: list[str], timeout: float | None) -> dict:
    """Run one arm to completion and return its manifest record."""
    logger.info("[sweep] starting %s", arm.key)
    logger.debug("[sweep] %s", " ".join(command))
    started = time.time()
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=timeout)
        code, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired:
        code, stdout, stderr = -1, "", f"timed out after {timeout}s"
    elapsed = time.time() - started

    record = {"key": arm.key, **asdict(arm), "returncode": code,
              "seconds": round(elapsed, 1), "run_dir": result_dir(stdout, stderr)}
    if code == 0:
        logger.info("[sweep] finished %s in %.1fs, wrote %s",
                    arm.key, elapsed, record["run_dir"] or "(no run dir reported)")
    else:
        #the last stderr line is usually the exception that ended it
        tail = (stderr.strip().splitlines() or ["(no stderr)"])[-1]
        record["error"] = tail
        logger.warning("[sweep] FAILED %s after %.1fs, exit %d: %s",
                       arm.key, elapsed, code, tail)
    return record


def load_manifest(path: Path) -> dict[str, dict]:
    """Records from a previous invocation, keyed by arm key."""
    if not path.exists():
        return {}
    try:
        records = json.loads(path.read_text(encoding="utf8"))
    except json.JSONDecodeError as error:
        logger.warning("[sweep] manifest at %s is unreadable, starting fresh: %s",
                       path, error)
        return {}
    return {r["key"]: r for r in records if isinstance(r, dict) and "key" in r}


def save_manifest(path: Path, records: dict[str, dict]) -> None:
    """Write the manifest after every arm, so a kill loses at most one run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(records.values()), indent=2), encoding="utf8")


def pending(grid: list[Arm], records: dict[str, dict], retry_failed: bool) -> list[Arm]:
    """Arms still to run: never recorded, or recorded as failed when retrying."""
    todo = []
    for arm in grid:
        record = records.get(arm.key)
        if record is None:
            todo.append(arm)
        elif retry_failed and record.get("returncode") != 0:
            todo.append(arm)
    return todo


def summarise(records: dict[str, dict]) -> pd.DataFrame:
    """One row per recorded arm, for the closing report.

    Sorts on whichever arm fields are present. A manifest carrying records from an
    older version, or one hand-edited between resumes, must not take the closing
    report down with it: by the time this runs the expensive work is already done and
    on disk, so a crash here would destroy the summary of a sweep that succeeded.
    """
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(list(records.values()))
    order = [c for c in ("input_window", "encoding", "algorithm", "seed")
             if c in frame.columns]
    return frame.sort_values(order) if order else frame


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", default=sys.executable,
                        help="Interpreter for each arm. Must be the mlgenn env.")
    parser.add_argument("--horizon", default="weekly")
    parser.add_argument("--target", default="rv_fwd")
    parser.add_argument("--iv-leg", default="vix9d")
    parser.add_argument("--models", default=MODELS)
    parser.add_argument("--encodings", default=",".join(ENCODINGS))
    parser.add_argument("--algorithms", default=",".join(ALGORITHMS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--windows", default=",".join(str(w) for w in WINDOWS))
    parser.add_argument("--manifest", default=None,
                        help=f"Defaults to {MANIFEST_NAME} beside the results root.")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Per-arm timeout in seconds. None waits indefinitely.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run at most this many arms, for a smoke test.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Rerun arms the manifest records as failed.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log the plan and exit without running anything.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    args = _parse_args(argv)

    grid = build_grid(
        encodings=tuple(e.strip() for e in args.encodings.split(",") if e.strip()),
        algorithms=tuple(a.strip() for a in args.algorithms.split(",") if a.strip()),
        seeds=tuple(int(s) for s in args.seeds.split(",") if s.strip()),
        windows=tuple(int(w) for w in args.windows.split(",") if w.strip()))

    from core.results import RESULTS_ROOT
    #inside the results root, which is gitignored: the manifest is a run artefact and
    #must not land in a tracked directory. aggregate.py globs for config.json, so a
    #loose file here is invisible to it.
    manifest_path = (Path(args.manifest) if args.manifest
                     else RESULTS_ROOT / MANIFEST_NAME)
    records = load_manifest(manifest_path)
    todo = pending(grid, records, args.retry_failed)
    if args.limit is not None:
        todo = todo[:args.limit]

    logger.info("[sweep] grid of %d arms, %d already recorded, %d to run",
                len(grid), len(records), len(todo))
    logger.info("[sweep] cell: %s / %s / %s, models %s",
                args.horizon, args.target, args.iv_leg, args.models)
    logger.info("[sweep] manifest at %s", manifest_path)
    if ALLOW_SILENT_ENCODINGS:
        logger.info("[sweep] silent-neuron abort disabled for %s, per D65 and D67. "
                    "Every other arm keeps it armed.",
                    ", ".join(ALLOW_SILENT_ENCODINGS))

    if args.dry_run:
        for arm in todo:
            logger.info("[sweep] would run %s", arm.key)
        return

    started = time.time()
    for index, arm in enumerate(todo, start=1):
        command = arm_command(arm, args.python, args.target, args.iv_leg,
                              args.horizon, args.models)
        logger.info("[sweep] arm %d of %d", index, len(todo))
        records[arm.key] = run_arm(arm, command, args.timeout)
        save_manifest(manifest_path, records)

    elapsed = time.time() - started
    frame = summarise(records)
    failed = frame[frame["returncode"] != 0] if not frame.empty else frame
    logger.info("[sweep] %d arm(s) run in %.1f min, %d failed of %d recorded",
                len(todo), elapsed / 60.0, len(failed), len(records))
    for _, row in failed.iterrows():
        logger.warning("[sweep]   %s: %s", row["key"], row.get("error", row["returncode"]))


if __name__ == "__main__":
    main()
