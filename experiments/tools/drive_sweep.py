"""Sweep the input/recurrent drive ratio at fixed total drive, across seeds.

`sec:notdeletion` reports a five-point, single-seed probe of the two initialisation
gains, and it is the evidence behind the leading hypothesis for why the network does
not learn: the hidden layer is driven harder by its own recurrence than by its input.
That probe varied both gains independently, so it confounded the *ratio* with the
*total* drive and cannot separate them.

This sweep holds total drive fixed and varies only the ratio, at eight seeds per point,
which is the experiment `sec:furtherwork` asks for. The first-order drive estimate is
`eq:driveratio`:

    external  = sqrt(n_in)  * duty_in  * w_in_scale
    recurrent = sqrt(n_hid) * duty_hid * w_rec_scale

Both duty cycles are themselves functions of the gains, so "fixed total drive" is a
first-order construction, not an exact one. That is the same approximation the equation
already makes and it is recorded with the result rather than hidden.

    python experiments/tools/drive_sweep.py --dry-run
    python experiments/tools/drive_sweep.py
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#the encoding sweep already solved resumability, manifests and per-arm subprocessing;
#reusing it keeps one implementation of that rather than two that drift
from tools.sweep import (  # noqa: E402
    load_manifest, pending, run_arm, save_manifest, summarise)

logger = logging.getLogger("drive_sweep")

MANIFEST_NAME = "drive_sweep_manifest.json"

#the cell the original probe ran on, and the one the primary hypothesis lives in
ENCODING = "delta_adaptive"
ALGORITHM = "eventprop"
INPUT_WINDOW = 45
SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
MODELS = "snn,har_rv,garch,constant,persistence"

#duty cycles from the measured defaults: delta fires ~21% of input cells, and the
#regulariser holds the hidden population at its 0.3 target
NUM_INPUT = 16          #delta encodings allocate two neurons per feature, 2 * 8
NUM_HIDDEN = 256
DUTY_INPUT = 0.21
DUTY_HIDDEN = 0.30

#defaults, which are also the r = 3.43 point of the grid
W_IN_DEFAULT = 2.5
W_REC_DEFAULT = 1.5

EXTERNAL_PER_GAIN = math.sqrt(NUM_INPUT) * DUTY_INPUT      #0.84
RECURRENT_PER_GAIN = math.sqrt(NUM_HIDDEN) * DUTY_HIDDEN   #4.8

TOTAL_DRIVE = (EXTERNAL_PER_GAIN * W_IN_DEFAULT
               + RECURRENT_PER_GAIN * W_REC_DEFAULT)       #9.3
DEFAULT_RATIO = (RECURRENT_PER_GAIN * W_REC_DEFAULT
                 / (EXTERNAL_PER_GAIN * W_IN_DEFAULT))     #3.43

#spans an input-dominated regime through the default and past it, roughly geometric
RATIOS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, round(DEFAULT_RATIO, 2), 6.0)


def gains_for_ratio(ratio: float, total: float = TOTAL_DRIVE) -> tuple[float, float]:
    """The two gains giving this recurrent-to-external ratio at the given total drive."""
    external = total / (1.0 + ratio)
    recurrent = total - external
    return (external / EXTERNAL_PER_GAIN, recurrent / RECURRENT_PER_GAIN)


@dataclass(frozen=True)
class DriveArm:
    """One cell: a drive ratio at one seed."""

    ratio: float
    seed: int
    w_in_scale: float
    w_rec_scale: float

    @property
    def key(self) -> str:
        return f"r{self.ratio:g}_s{self.seed}"


def build_grid(ratios: tuple[float, ...] = RATIOS,
               seeds: tuple[int, ...] = SEEDS) -> list[DriveArm]:
    """Every arm, in a deterministic order."""
    arms = []
    for ratio, seed in product(ratios, seeds):
        w_in, w_rec = gains_for_ratio(ratio)
        arms.append(DriveArm(ratio=ratio, seed=seed,
                             w_in_scale=round(w_in, 4),
                             w_rec_scale=round(w_rec, 4)))
    return arms


def arm_command(arm: DriveArm, python: str, target: str, iv_leg: str,
                horizon: str, models: str) -> list[str]:
    """The exact argv for one arm."""
    return [python, str(_ROOT / "run_experiment.py"),
            "--models", models,
            "--horizon", horizon,
            "--target", target,
            "--iv-leg", iv_leg,
            "--encoding", ENCODING,
            "--algorithm", ALGORITHM,
            "--input-window", str(INPUT_WINDOW),
            "--seed", str(arm.seed),
            "--w-in-scale", str(arm.w_in_scale),
            "--w-rec-scale", str(arm.w_rec_scale)]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", default=sys.executable,
                        help="Interpreter for each arm. Must be the mlgenn env.")
    parser.add_argument("--horizon", default="weekly")
    parser.add_argument("--target", default="rv_fwd")
    parser.add_argument("--iv-leg", default="vix9d")
    parser.add_argument("--models", default=MODELS)
    parser.add_argument("--ratios", default=",".join(str(r) for r in RATIOS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--timeout", type=float, default=1800.0,
                        help="Per-arm timeout in seconds. The main sweep averaged 150s.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    args = _parse_args(argv)

    grid = build_grid(
        ratios=tuple(float(r) for r in args.ratios.split(",") if r.strip()),
        seeds=tuple(int(s) for s in args.seeds.split(",") if s.strip()))

    from core.results import RESULTS_ROOT
    manifest_path = (Path(args.manifest) if args.manifest
                     else RESULTS_ROOT / MANIFEST_NAME)
    records = load_manifest(manifest_path)
    todo = pending(grid, records, args.retry_failed)
    if args.limit is not None:
        todo = todo[:args.limit]

    logger.info("[drive] cell: %s / %s / L%d, %s / %s / %s",
                ENCODING, ALGORITHM, INPUT_WINDOW,
                args.horizon, args.target, args.iv_leg)
    logger.info("[drive] total drive held at %.3f; default ratio is %.2f",
                TOTAL_DRIVE, DEFAULT_RATIO)
    for ratio in sorted({a.ratio for a in grid}):
        w_in, w_rec = gains_for_ratio(ratio)
        logger.info("[drive]   ratio %-5g -> w_in %.4f, w_rec %.4f", ratio, w_in, w_rec)
    logger.info("[drive] grid of %d arms, %d already recorded, %d to run",
                len(grid), len(records), len(todo))
    logger.info("[drive] manifest at %s", manifest_path)

    if args.dry_run:
        for arm in todo:
            logger.info("[drive] would run %s (w_in %.4f, w_rec %.4f)",
                        arm.key, arm.w_in_scale, arm.w_rec_scale)
        return

    started = time.time()
    for index, arm in enumerate(todo, start=1):
        command = arm_command(arm, args.python, args.target, args.iv_leg,
                              args.horizon, args.models)
        logger.info("[drive] arm %d of %d", index, len(todo))
        record = run_arm(arm, command, args.timeout)
        record.update(asdict(arm))
        records[arm.key] = record
        save_manifest(manifest_path, records)

    elapsed = time.time() - started
    frame = summarise(records)
    failed = frame[frame["returncode"] != 0] if not frame.empty else frame
    logger.info("[drive] %d arm(s) run in %.1f min, %d failed of %d recorded",
                len(todo), elapsed / 60.0, len(failed), len(records))


if __name__ == "__main__":
    main()
