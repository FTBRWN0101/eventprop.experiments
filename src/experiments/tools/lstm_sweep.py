"""Give the LSTM control the search budget the SNN got.

The LSTM is the proposal's neural baseline and the only model the SNN currently beats
(D101). Until now it ran at one fixed configuration -- 64 hidden units, one layer, no
dropout -- while the SNN was swept over five encodings, two sequence lengths, six window
fractions, six drive ratios and a trial-reset flag. "Tuned model beats untuned baseline"
is a much weaker claim than "SNN beats LSTM", and it is the flaw most likely to be raised
against the one positive result in the project.

This sweep removes that asymmetry. Architecture and learning rate are crossed at both
sequence lengths and both horizons, at full seed depth, so the comparison is against the
best LSTM rather than the first one.

No GPU and no GeNN: the LSTM is torch on CPU here (`torch.cuda.is_available()` is False
in the plain .venv), so this runs safely alongside a GeNN sweep. Confirmed on 2026-08-25.

**Selection caveat, stated rather than hidden.** Like every other sweep in this project,
this one scores on the test split, because the validation split has never been usable
(O4, and 934 of 935 runs to date). So the LSTM is being tuned on test exactly as the SNN
was. That keeps the comparison symmetric, which is the point here, but it does not make
either side clean. Fixing that is a separate piece of work.

    python src/experiments/tools/lstm_sweep.py --dry-run
    run_lstm_sweep.bat
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

#the encoding sweep already solved manifests, resume and per-arm subprocessing
from tools.sweep import (  # noqa: E402
    load_manifest, pending, run_arm, save_manifest, summarise)

logger = logging.getLogger("lstm_sweep")

MANIFEST_NAME = "lstm_sweep_manifest.json"

#only the LSTM and the deterministic econometric baselines. No snn, so no GeNN, so this
#can share a machine with a running GeNN sweep.
MODELS = "lstm,har_rv,garch,constant,persistence"

#architecture. 64/1/0.0 is the historical default and is included so the sweep contains
#the configuration every published LSTM number came from.
HIDDEN: tuple[int, ...] = (32, 64, 128, 256)
LAYERS: tuple[int, ...] = (1, 2)
LEARNING_RATES: tuple[float, ...] = (0.001, 0.003)
SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)

#dropout only exists as a knob with two or more layers; torch ignores it on one, and
#config.py raises rather than let a run claim a setting that does nothing
DROPOUT_MULTI = 0.2

#(horizon, iv leg, sequence lengths). The monthly panel carries no VIX9D leg.
CELLS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("weekly", "vix9d", (5, 45)),
    ("monthly", "vix", (21, 45)),
)


@dataclass(frozen=True)
class LstmArm:
    """One LSTM configuration at one cell and seed."""

    horizon: str
    iv_leg: str
    input_window: int
    hidden: int
    layers: int
    learning_rate: float
    dropout: float
    seed: int

    @property
    def key(self) -> str:
        return (f"{self.horizon}_L{self.input_window}_h{self.hidden}"
                f"_l{self.layers}_lr{self.learning_rate:g}_d{self.dropout:g}"
                f"_s{self.seed}")


def build_grid(hidden: tuple[int, ...] = HIDDEN,
               layers: tuple[int, ...] = LAYERS,
               learning_rates: tuple[float, ...] = LEARNING_RATES,
               seeds: tuple[int, ...] = SEEDS,
               cells: tuple = CELLS) -> list[LstmArm]:
    """Every arm, in a deterministic order."""
    arms = []
    for horizon, iv_leg, windows in cells:
        for window, h, layer_count, lr, seed in product(
                windows, hidden, layers, learning_rates, seeds):
            arms.append(LstmArm(horizon=horizon, iv_leg=iv_leg, input_window=window,
                                hidden=h, layers=layer_count, learning_rate=lr,
                                dropout=DROPOUT_MULTI if layer_count > 1 else 0.0,
                                seed=seed))
    return arms


def arm_command(arm: LstmArm, python: str, target: str, models: str) -> list[str]:
    """The exact argv for one arm."""
    return [python, str(_ROOT / "run_experiment.py"),
            "--models", models,
            "--horizon", arm.horizon,
            "--target", target,
            "--iv-leg", arm.iv_leg,
            "--input-window", str(arm.input_window),
            "--seed", str(arm.seed),
            "--learning-rate", str(arm.learning_rate),
            "--lstm-hidden-size", str(arm.hidden),
            "--lstm-num-layers", str(arm.layers),
            "--lstm-dropout", str(arm.dropout)]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", default=sys.executable,
                        help="Interpreter for each arm. The plain .venv, which has torch.")
    parser.add_argument("--target", default="rv_fwd")
    parser.add_argument("--models", default=MODELS)
    parser.add_argument("--hidden", default=",".join(str(h) for h in HIDDEN))
    parser.add_argument("--layers", default=",".join(str(n) for n in LAYERS))
    parser.add_argument("--learning-rates",
                        default=",".join(str(r) for r in LEARNING_RATES))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    args = _parse_args(argv)

    grid = build_grid(
        hidden=tuple(int(h) for h in args.hidden.split(",") if h.strip()),
        layers=tuple(int(n) for n in args.layers.split(",") if n.strip()),
        learning_rates=tuple(float(r) for r in args.learning_rates.split(",") if r.strip()),
        seeds=tuple(int(s) for s in args.seeds.split(",") if s.strip()))

    from core.results import RESULTS_ROOT
    manifest_path = (Path(args.manifest) if args.manifest
                     else RESULTS_ROOT / MANIFEST_NAME)
    records = load_manifest(manifest_path)
    todo = pending(grid, records, args.retry_failed)
    if args.limit is not None:
        todo = todo[:args.limit]

    logger.info("[lstm] architecture x learning rate at both horizons, %s", args.target)
    logger.info("[lstm] grid of %d arms, %d already recorded, %d to run",
                len(grid), len(records), len(todo))
    logger.info("[lstm] manifest at %s", manifest_path)
    if args.dry_run:
        for arm in todo:
            logger.info("[lstm] would run %s", arm.key)
        return

    started = time.time()
    for index, arm in enumerate(todo, start=1):
        command = arm_command(arm, args.python, args.target, args.models)
        logger.info("[lstm] arm %d of %d", index, len(todo))
        record = run_arm(arm, command, args.timeout)
        record.update(asdict(arm))
        records[arm.key] = record
        save_manifest(manifest_path, records)

    elapsed = time.time() - started
    frame = summarise(records)
    failed = frame[frame["returncode"] != 0] if not frame.empty else frame
    logger.info("[lstm] %d arm(s) in %.1f min, %d failed of %d recorded",
                len(todo), elapsed / 60.0, len(failed), len(records))


if __name__ == "__main__":
    main()
