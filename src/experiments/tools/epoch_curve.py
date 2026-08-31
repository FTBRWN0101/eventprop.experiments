"""Score every saved training epoch, to ask whether the SNN should have stopped earlier.

The SNN has no validation logic and no epoch selection. `predict()` reads back
`num_epochs - 1` unconditionally, so every result in this project is the network as it
stood after the last epoch, whether or not that was its best. Nothing has ever checked.

The LSTM lost 0.35 in `r2_rv` to exactly this: without `--holdout-val` it has no early
stopping signal, trains 50 epochs blind and overfits (D108). The SNN's protocol is the
same shape and has never been tested.

It can be tested for nothing. Training already writes a checkpoint every
`CHECKPOINT_EPOCHS`, so keys 9, 19, 29, 39 and 49 are sitting in
`artifacts/checkpoints/live` for every run ever done. This loads each in turn and
scores it, which costs inference rather than a retrain.

**Read the validation column, not the test column.** Choosing an epoch by test score is
the selection-on-test problem this whole exercise is about. The test column is here to
show what the choice would have been worth, not to make it.

    run_epoch_curve.bat --encoding delta --input-window 45 --seed 1 --readout-window
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

import pandas as pd  # noqa: E402

from core.config import ExperimentConfig  # noqa: E402
from core.dataset import VrpDataset  # noqa: E402
from core.metrics import evaluate  # noqa: E402

logger = logging.getLogger("epoch_curve")


def saved_epochs(checkpoint_dir: Path) -> list[int]:
    """Epoch keys with a full set of weights on disk, ascending.

    A key counts only when every connection is present. A run killed mid-save leaves a
    partial set, and scoring that would silently read a half-written network.
    """
    if not checkpoint_dir.is_dir():
        return []
    keys: dict[int, int] = {}
    for path in checkpoint_dir.glob("*-Conn_*-g.npy"):
        head = path.name.split("-", 1)[0]
        if head.isdigit():
            keys[int(head)] = keys.get(int(head), 0) + 1
    expected = max(keys.values(), default=0)
    return sorted(k for k, n in keys.items() if n == expected)


def score_epochs(config: ExperimentConfig, splits: tuple[str, ...]) -> pd.DataFrame:
    """One row per saved epoch, scored on each split."""
    from models.snn import SnnForecaster

    data = VrpDataset(config)
    probe = SnnForecaster(config)
    epochs = saved_epochs(probe._checkpoint_dir())
    if not epochs:
        raise FileNotFoundError(
            f"no complete checkpoints under {probe._checkpoint_dir()}; "
            f"this scores saved weights and cannot train them")
    logger.info("[epoch] %s", probe._checkpoint_dir().name)
    logger.info("[epoch] saved epochs: %s", epochs)

    #the same train-time constants run_experiment.py scores against, so a number here
    #is comparable to a number in any metrics.csv rather than to itself alone
    train = data.split("train")
    train_target_mean = float(train.target.mean())
    train_rv_mean = float(train.rv_fwd.mean())

    rows = []
    for epoch in epochs:
        #a fresh forecaster per epoch: warm_start is cheap and this stops one epoch's
        #encoder or normalisation leaking into the next
        model = SnnForecaster(replace(config, predict_epoch=epoch))
        model.warm_start(data)
        row = {"epoch": epoch}
        for name in splits:
            if data.daily(name).empty:
                continue
            pred = model.predict(data, name)
            scored = evaluate(data.split(name), pred, train_target_mean, train_rv_mean)
            row[f"{name}_r2"] = scored.get("r2_rv")
            row[f"{name}_qlike"] = scored.get("qlike")
        rows.append(row)
    return pd.DataFrame(rows)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--horizon", default="weekly")
    parser.add_argument("--target", default="rv_fwd")
    parser.add_argument("--iv-leg", default="vix9d")
    parser.add_argument("--encoding", default="delta")
    parser.add_argument("--algorithm", default="eventprop")
    parser.add_argument("--input-window", type=int, default=45)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--readout-window", action="store_true")
    parser.add_argument("--readout-window-fraction", type=float, default=0.1)
    parser.add_argument("--reset-synapses-between-trials", action="store_true")
    parser.add_argument("--holdout-val", action="store_true",
                        help="Carve 2017-2019 out of train, so there is a validation "
                             "column to choose an epoch on rather than choosing on test.")
    parser.add_argument("--splits", default="val,test")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    args = _parse_args(argv)
    config = ExperimentConfig.load(
        model="snn", horizon=args.horizon, target=args.target, iv_leg=args.iv_leg,
        encoding=args.encoding, algorithm=args.algorithm,
        input_window=args.input_window, seed=args.seed,
        readout_window=args.readout_window,
        readout_window_fraction=args.readout_window_fraction,
        reset_synapses_between_trials=args.reset_synapses_between_trials,
        holdout_val=args.holdout_val)

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    frame = score_epochs(config, splits)
    print(frame.round(4).to_string(index=False))

    if "test_r2" in frame and frame["test_r2"].notna().any():
        last = frame.iloc[-1]
        best = frame.loc[frame["test_r2"].idxmax()]
        logger.info("[epoch] final epoch %d: test r2 %.4f", last["epoch"], last["test_r2"])
        logger.info("[epoch] best epoch  %d: test r2 %.4f (%+.4f)",
                    best["epoch"], best["test_r2"], best["test_r2"] - last["test_r2"])
        if "val_r2" in frame and frame["val_r2"].notna().any():
            chosen = frame.loc[frame["val_r2"].idxmax()]
            logger.info("[epoch] chosen on VALIDATION: epoch %d, which scores test "
                        "%.4f (%+.4f against the final epoch)",
                        chosen["epoch"], chosen["test_r2"],
                        chosen["test_r2"] - last["test_r2"])


if __name__ == "__main__":
    main()
