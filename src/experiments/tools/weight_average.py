"""Average the weights across training checkpoints, and score the average.

D110 found the SNN never converges: the trajectory swings by up to 0.26 inside a single
run and every seed peaks at a different epoch, so `predict()` reading back the last epoch
is closer to a coin flip than a choice. D115 then ruled out the obvious remedy, since
lowering the learning rate triggers spike deletion rather than stabilising anything.

If no single epoch is trustworthy, the natural response is not to pick one. Averaging the
weights over a tail of checkpoints is the standard treatment for exactly this, and here it
costs nothing to try: training already wrote the checkpoints, so this is arithmetic plus
one inference pass rather than a rerun.

The averaged weights are written back under a synthetic key well outside the epoch range,
so they sit beside the originals rather than replacing any of them. Nothing is deleted.

    scripts\\windows\\run_in_mlgenn.bat src\\experiments\\tools\\weight_average.py --encoding delta
        --input-window 45 --seed 1 --readout-window --tail 4
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.config import ExperimentConfig  # noqa: E402
from core.dataset import VrpDataset  # noqa: E402
from core.metrics import evaluate  # noqa: E402
from tools.epoch_curve import saved_epochs  # noqa: E402

logger = logging.getLogger("weight_average")

#far outside any real epoch, so an averaged checkpoint can never be mistaken for one and
#can never collide with a longer run
AVERAGED_KEY = 90000


def write_average(checkpoint_dir: Path, epochs: list[int],
                  key: int = AVERAGED_KEY) -> list[Path]:
    """Mean the weight arrays over *epochs*, written under *key*. Returns what it wrote.

    Every connection is averaged independently and shapes are checked, because a mismatch
    means the checkpoints came from different architectures sharing a directory. That has
    happened twice in this project, so it is verified rather than trusted.
    """
    written = []
    names = {p.name.split("-", 1)[1] for p in checkpoint_dir.glob(f"{epochs[0]}-Conn_*")}
    for name in sorted(names):
        stack = []
        for epoch in epochs:
            path = checkpoint_dir / f"{epoch}-{name}"
            if not path.exists():
                raise FileNotFoundError(f"{path} missing; cannot average over {epochs}")
            stack.append(np.load(path))
        shapes = {a.shape for a in stack}
        if len(shapes) != 1:
            raise ValueError(
                f"{name} has shapes {shapes} across {epochs}. The checkpoints in "
                f"{checkpoint_dir} are not all from one architecture.")
        out = checkpoint_dir / f"{key}-{name}"
        np.save(out, np.mean(stack, axis=0))
        written.append(out)
    return written


def score(config: ExperimentConfig, data: VrpDataset, epoch: int,
          train_target_mean: float, train_rv_mean: float, split: str) -> dict:
    from models.snn import SnnForecaster

    model = SnnForecaster(replace(config, predict_epoch=epoch))
    model.warm_start(data)
    return evaluate(data.split(split), model.predict(data, split),
                    train_target_mean, train_rv_mean)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--horizon", default="weekly")
    parser.add_argument("--target", default="rv_fwd")
    parser.add_argument("--iv-leg", default="vix9d")
    parser.add_argument("--encoding", default="delta")
    parser.add_argument("--input-window", type=int, default=45)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--readout-window", action="store_true")
    parser.add_argument("--readout-window-fraction", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--tail", type=int, default=4,
                        help="How many of the latest checkpoints to average.")
    parser.add_argument("--split", default="test")
    args = parser.parse_args(argv)

    config = ExperimentConfig.load(
        model="snn", horizon=args.horizon, target=args.target, iv_leg=args.iv_leg,
        encoding=args.encoding, algorithm="eventprop", input_window=args.input_window,
        seed=args.seed, readout_window=args.readout_window,
        readout_window_fraction=args.readout_window_fraction,
        learning_rate=args.learning_rate)

    from models.snn import SnnForecaster
    data = VrpDataset(config)
    checkpoint_dir = SnnForecaster(config)._checkpoint_dir()
    epochs = [e for e in saved_epochs(checkpoint_dir) if e < AVERAGED_KEY]
    if len(epochs) < 2:
        raise FileNotFoundError(
            f"{checkpoint_dir} has {len(epochs)} checkpoint(s); averaging needs at least 2")
    tail = epochs[-args.tail:]
    logger.info("[avg] %s", checkpoint_dir.name)
    logger.info("[avg] saved epochs %s, averaging the last %d: %s",
                epochs, len(tail), tail)

    train = data.split("train")
    mt, mrv = float(train.target.mean()), float(train.rv_fwd.mean())

    rows = [{"weights": str(e),
             **{k: score(config, data, e, mt, mrv, args.split)[k]
                for k in ("r2_rv", "qlike")}} for e in epochs]
    write_average(checkpoint_dir, tail)
    avg = score(config, data, AVERAGED_KEY, mt, mrv, args.split)
    rows.append({"weights": f"mean of {len(tail)}", "r2_rv": avg["r2_rv"],
                 "qlike": avg["qlike"]})

    frame = pd.DataFrame(rows)
    print(frame.round(4).to_string(index=False))
    last = frame.iloc[-2]
    logger.info("[avg] last epoch %s: r2 %.4f | average: r2 %.4f (%+.4f)",
                last["weights"], last["r2_rv"], avg["r2_rv"],
                avg["r2_rv"] - last["r2_rv"])
    best_single = frame.iloc[:-1]["r2_rv"].max()
    logger.info("[avg] best single epoch %.4f, so the average is %s it",
                best_single, "above" if avg["r2_rv"] > best_single else "below")


if __name__ == "__main__":
    main()
