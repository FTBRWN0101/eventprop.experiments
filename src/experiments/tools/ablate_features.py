"""Leave-one-out ablation over the served input signals, for Contribution #3.

The proposal promises the forecast improvement attributable to each options-derived
signal. That number only means something on the ``rv_fwd`` target: on any VRP target
the implied-vol leg re-enters the prediction through ``Split.to_target``, so a
price-only arm is options-free in neither its features nor its target (D28). This
tool therefore defaults to ``--target rv_fwd`` and warns on anything else.

Each arm withholds exactly one column and refits every named model. The full arm is
run first and every ablated arm is scored against it, so the reported delta is the
cost of removing that signal and nothing else.

Only ``lstm`` and ``snn`` can be ablated. ``har_rv`` reads a fixed three columns,
``garch`` reads the return series and ``constant``/``persistence`` read the target,
so withholding a feature either changes nothing for them or removes a column the
model is defined by. Naming one of those raises rather than reporting zeroes.

    python src/experiments/tools/ablate_features.py --models lstm
    python src/experiments/tools/ablate_features.py --restore-features skew,ts_slope
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  #noqa: E402

from core import metrics  #noqa: E402
from core.config import ExperimentConfig  #noqa: E402
from core.dataset import VrpDataset  #noqa: E402
from run_experiment import fit_predict  #noqa: E402

logger = logging.getLogger("ablate")

#the only models whose inputs are the served feature vector
ABLATABLE_MODELS: tuple[str, ...] = ("lstm", "snn")


def check_ablatable(model_names: list[str]) -> None:
    """Reject models that do not read the served feature vector."""
    fixed = [n for n in model_names if n not in ABLATABLE_MODELS]
    if fixed:
        raise ValueError(
            f"{fixed} do not read the served feature columns, so withholding one "
            f"either changes nothing or removes a column the model is defined by "
            f"(har_rv is exactly rv_1/rv_5/rv_21). Ablatable models: "
            f"{list(ABLATABLE_MODELS)}.")


def served_columns(config: ExperimentConfig) -> list[str]:
    """Feature columns the configured arm actually serves."""
    data = VrpDataset(config)
    return data.feature_columns(data.daily("train"))


def score(config: ExperimentConfig, model_names: list[str]) -> dict[str, dict]:
    """Fit and evaluate *model_names* under *config*, returning the metric rows."""
    data, predictions, _ = fit_predict(config, model_names)
    train = data.split("train")
    split = data.split("test")
    train_target_mean = float(train.target.mean())
    train_rv_mean = float(train.rv_fwd.mean())
    return {name: metrics.evaluate(split, pred, train_target_mean, train_rv_mean)
            for name, pred in predictions.items()}


def run_ablation(config: ExperimentConfig, model_names: list[str]) -> pd.DataFrame:
    """Score the full arm, then one arm per withheld column."""
    check_ablatable(model_names)
    columns = served_columns(config)
    logger.info("[ablate] %d served columns: %s", len(columns), ", ".join(columns))

    rows = []
    baseline = score(config, model_names)
    for name, row in baseline.items():
        rows.append({"dropped": "(none)", "model": name, **row})

    for column in columns:
        arm = ExperimentConfig.load(
            **{**{f.name: getattr(config, f.name)
                  for f in config.__dataclass_fields__.values()},
               "drop_features": (column,)})
        logger.info("[ablate] withholding %s", column)
        for name, row in score(arm, model_names).items():
            #both co-primary metrics, and note the sign convention differs: a signal
            #that mattered shows up as d_r2_rv < 0 (accuracy lost) and d_qlike > 0
            #(loss gained). The two can name different signals as the most damaging,
            #which is the same split _print_verdict reports between models.
            rows.append({"dropped": column, "model": name,
                         "d_r2_rv": row["r2_rv"] - baseline[name]["r2_rv"],
                         "d_qlike": row["qlike"] - baseline[name]["qlike"], **row})

    frame = pd.DataFrame(rows)
    #most damaging signal first: the largest drop in r2_rv when withheld
    return frame.sort_values(["model", "d_r2_rv"], na_position="first")


def _report_most_damaging(frame: pd.DataFrame) -> None:
    """Name the costliest withheld signal under each co-primary metric, per model.

    r2_rv has been this project's headline throughout and QLIKE is the loss the
    volatility literature scores on. They rank forecasts differently often enough that
    "the most important signal" is not a well-formed claim until it says which metric it
    is measured on, so both are named and a split between them is stated outright.
    """
    ablated = frame[frame["dropped"] != "(none)"]
    for model, block in ablated.groupby("model", sort=False):
        usable = block.replace([float("inf"), float("-inf")], pd.NA)
        by_r2 = usable[["dropped", "d_r2_rv"]].dropna()
        by_qlike = usable[["dropped", "d_qlike"]].dropna()
        if by_r2.empty or by_qlike.empty:
            logger.info("[ablate] %s: not enough finite deltas to rank the signals "
                        "(r2_rv %d, QLIKE %d usable)", model, len(by_r2), len(by_qlike))
            continue
        #costliest = the largest accuracy lost, or the largest loss gained
        worst_r2 = by_r2.loc[by_r2["d_r2_rv"].idxmin()]
        worst_qlike = by_qlike.loc[by_qlike["d_qlike"].idxmax()]
        if worst_r2["dropped"] == worst_qlike["dropped"]:
            logger.info("[ablate] %s: %s is the costliest signal on both metrics "
                        "(d_r2_rv %+.4f, d_qlike %+.4f)", model, worst_r2["dropped"],
                        worst_r2["d_r2_rv"], worst_qlike["d_qlike"])
        else:
            logger.info("[ablate] %s: the metrics DISAGREE about the costliest signal. "
                        "r2_rv says %s (%+.4f), QLIKE says %s (%+.4f).", model,
                        worst_r2["dropped"], worst_r2["d_r2_rv"],
                        worst_qlike["dropped"], worst_qlike["d_qlike"])


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--horizon", default="weekly", choices=["weekly", "monthly"])
    parser.add_argument("--iv-leg", default="vix9d", choices=["vix", "vix9d"])
    parser.add_argument("--target", default="rv_fwd",
                        help="Defaults to rv_fwd; see D28 on why a VRP target cannot "
                             "answer this question.")
    parser.add_argument("--feature-set", default="options+price")
    parser.add_argument("--models", default="lstm",
                        help=f"Comma-separated models to refit on every arm. "
                             f"One of {list(ABLATABLE_MODELS)}.")
    parser.add_argument("--restore-features", default="",
                        help="Columns to re-admit from EXCLUDED_FEATURES first.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=None, help="Optional CSV path for the table.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    if args.target != "rv_fwd":
        logger.warning("[ablate] target=%s routes the IV leg back into every "
                       "prediction, so the deltas below understate the options "
                       "contribution. See D28.", args.target)
    restore = tuple(c.strip() for c in args.restore_features.split(",") if c.strip())
    config = ExperimentConfig.load(
        horizon=args.horizon, iv_leg=args.iv_leg, target=args.target,
        feature_set=args.feature_set, seed=args.seed, restore_features=restore)

    frame = run_ablation(config, [m.strip() for m in args.models.split(",") if m.strip()])
    show = ["model", "dropped", "d_r2_rv", "d_qlike", "r2_rv", "gate", "mse", "qlike", "n"]
    logger.info("\n%s", frame[show].to_string(index=False,
                                              float_format=lambda v: f"{v:.4f}"))
    _report_most_damaging(frame)
    if args.out:
        frame.to_csv(args.out, index=False)
        logger.info("\nwritten to %s", args.out)


if __name__ == "__main__":
    main()
