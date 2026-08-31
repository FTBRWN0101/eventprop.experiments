"""Cross the four factors instead of sweeping them one at a time.

Every intervention in this project was measured alone, with the others left at their
defaults: the drive ratio in D84, the readout window in D92, its fraction in D93, the
trial reset in D98. So every conclusion is conditional on the rest sitting where they
happened to sit, and no interaction has ever been measured. Two in particular are
unmeasured and load-bearing:

* **window x reset.** The reset was tested unwindowed and the window was tested without
  the reset. The monthly programme ran both together but never factorially, so nothing
  separates "both help" from "either helps and the second adds nothing".
* **window x drive.** D84 rejected recurrence dominance using unwindowed arms. If the
  window changes what the network can learn, that rejection was made under an objective
  now known to be misspecified (D85), and it deserves re-asking with the window on.

This is a full 2^4: window, reset, drive ratio and encoding, at the sequence length where
the window works (L=45, per the corrected D100), with 8 seeds per cell. 128 arms.

Full rather than fractional on purpose. A fractional design would alias the two-factor
interactions against each other, and the two-factor interactions are the entire question.

Levels are chosen so that one corner reproduces work already on disk. The
(window off, reset off, ratio 3.43, delta) cell is the original sweep's arm, so if the
factorial disagrees with it the factorial is wrong.

    python src/experiments/tools/factorial.py --dry-run
    run_factorial.bat
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.snn import W_IN_SCALE, W_REC_SCALE  # noqa: E402
from tools.drive_sweep import gains_for_ratio  # noqa: E402
from tools.sweep import (  # noqa: E402
    load_manifest, pending, run_arm, save_manifest, summarise)

logger = logging.getLogger("factorial")

MANIFEST_NAME = "factorial_manifest.json"
MODELS = "snn,har_rv,garch,constant,persistence"

#L=45 because that is where the window has an effect at all. The corrected D100 found it
#does nothing at L=21, so crossing factors at a length where the headline factor is inert
#would spend 128 arms measuring interactions with zero.
INPUT_WINDOW = 45
HORIZON, TARGET, IV_LEG, ALGORITHM = "weekly", "rv_fwd", "vix9d", "eventprop"
SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)

#the four factors, two levels each
WINDOWS: tuple[float | None, ...] = (None, 0.10)   #off, and D93's optimum
RESETS: tuple[bool, ...] = (False, True)
DEFAULT_RATIO = 3.43
RATIOS: tuple[float, ...] = (DEFAULT_RATIO, 1.0)   #the default, and input-dominated
ENCODINGS: tuple[str, ...] = ("delta", "rate")     #the two that reward the window


@dataclass(frozen=True)
class Cell:
    """One corner of the design at one seed."""

    encoding: str
    ratio: float
    reset: bool
    window: float | None
    seed: int
    w_in_scale: float
    w_rec_scale: float

    @property
    def key(self) -> str:
        win = "off" if self.window is None else f"{self.window:g}"
        return (f"{self.encoding}_r{self.ratio:g}_rst{int(self.reset)}"
                f"_win{win}_s{self.seed}")


def build_grid(encodings: tuple[str, ...] = ENCODINGS,
               ratios: tuple[float, ...] = RATIOS,
               resets: tuple[bool, ...] = RESETS,
               windows: tuple[float | None, ...] = WINDOWS,
               seeds: tuple[int, ...] = SEEDS) -> list[Cell]:
    """Every corner by every seed, in a deterministic order."""
    cells = []
    for encoding, ratio, reset, window, seed in product(
            encodings, ratios, resets, windows, seeds):
        w_in, w_rec = gains_for_ratio(ratio)
        #the default ratio takes the exact model defaults rather than the 2.4992 and
        #1.5001 the ratio arithmetic returns for it. Same ratio to four decimals, but
        #the offset would put this corner outside the `w_in_scale == 2.5` filter every
        #existing analysis uses, so the baseline corner would silently fail to line up
        #with the sweep arm it is here to reproduce. The drive sweep has exactly this
        #offset and is excluded from every window analysis because of it.
        if abs(ratio - DEFAULT_RATIO) < 0.005:
            w_in, w_rec = W_IN_SCALE, W_REC_SCALE
        cells.append(Cell(encoding=encoding, ratio=ratio, reset=reset, window=window,
                          seed=seed, w_in_scale=round(w_in, 4),
                          w_rec_scale=round(w_rec, 4)))
    return cells


def arm_command(cell: Cell, python: str, models: str) -> list[str]:
    """The exact argv for one corner."""
    command = [python, str(_ROOT / "run_experiment.py"),
               "--models", models,
               "--horizon", HORIZON,
               "--target", TARGET,
               "--iv-leg", IV_LEG,
               "--encoding", cell.encoding,
               "--algorithm", ALGORITHM,
               "--input-window", str(INPUT_WINDOW),
               "--seed", str(cell.seed),
               "--w-in-scale", str(cell.w_in_scale),
               "--w-rec-scale", str(cell.w_rec_scale)]
    if cell.window is not None:
        command += ["--readout-window", "--readout-window-fraction", str(cell.window)]
    if cell.reset:
        command.append("--reset-synapses-between-trials")
    return command


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", default=sys.executable,
                        help="Interpreter for each arm. Must be the mlgenn env.")
    parser.add_argument("--models", default=MODELS)
    parser.add_argument("--encodings", default=",".join(ENCODINGS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    args = _parse_args(argv)

    grid = build_grid(
        encodings=tuple(e for e in args.encodings.split(",") if e.strip()),
        seeds=tuple(int(s) for s in args.seeds.split(",") if s.strip()))

    from core.results import RESULTS_ROOT
    manifest_path = (Path(args.manifest) if args.manifest
                     else RESULTS_ROOT / MANIFEST_NAME)
    records = load_manifest(manifest_path)
    todo = pending(grid, records, args.retry_failed)
    if args.limit is not None:
        todo = todo[:args.limit]

    logger.info("[factorial] 2^4 at %s/%s/L%d: window x reset x drive x encoding",
                HORIZON, TARGET, INPUT_WINDOW)
    for ratio in RATIOS:
        w_in, w_rec = gains_for_ratio(ratio)
        logger.info("[factorial]   ratio %-5g -> w_in %.4f, w_rec %.4f",
                    ratio, w_in, w_rec)
    logger.info("[factorial] grid of %d arms, %d already recorded, %d to run",
                len(grid), len(records), len(todo))
    logger.info("[factorial] manifest at %s", manifest_path)
    if args.dry_run:
        for cell in todo:
            logger.info("[factorial] would run %s", cell.key)
        return

    started = time.time()
    for index, cell in enumerate(todo, start=1):
        command = arm_command(cell, args.python, args.models)
        logger.info("[factorial] arm %d of %d", index, len(todo))
        record = run_arm(cell, command, args.timeout)
        record.update(asdict(cell))
        records[cell.key] = record
        save_manifest(manifest_path, records)

    elapsed = time.time() - started
    frame = summarise(records)
    failed = frame[frame["returncode"] != 0] if not frame.empty else frame
    logger.info("[factorial] %d arm(s) in %.1f min, %d failed of %d recorded",
                len(todo), elapsed / 60.0, len(failed), len(records))


if __name__ == "__main__":
    main()
