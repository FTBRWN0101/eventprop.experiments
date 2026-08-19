"""Price the D27/D38 fit-sample restriction on HAR, as a number rather than a claim.

D27 forces every model to fit from ``VrpDataset.fit_start()`` (~2011 on the vix9d leg)
so the DM test compares like with like. That is correct *and* it weakens the benchmark,
which flatters every model scored against it. This sweeps the fit start and reports the
test-split score at each, so the size of the handicap is measurable.

Every arm is scored on the **same test rows** against the **same train-mean constant**,
so only the fit sample moves. Sample starts before the leg's own start still score on
the leg-gated test set -- the curve is about HAR's training data, not the evaluation.

    python experiments/tools/har_sample_curve.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  #noqa: E402

from core import metrics  #noqa: E402
from core.config import ExperimentConfig  #noqa: E402
from core.dataset import VrpDataset  #noqa: E402
from models.har_rv import HarRvForecaster  #noqa: E402

#1990 is the panel start; 2011 is where the vix9d leg (and so fit_start) begins
STARTS: tuple[str, ...] = ("1990-01-01", "1996-01-01", "2000-01-01", "2005-01-01",
                           "2008-01-01", "2011-01-01", "2013-01-01")


def curve(config: ExperimentConfig,
          starts: tuple[str, ...] = STARTS) -> pd.DataFrame:
    """One row per fit start: fitted range, coefficients, and test-split scores."""
    data = VrpDataset(config)
    train, split = data.split("train"), data.split("test")
    train_rv_mean = float(train.rv_fwd.mean())
    train_target_mean = float(train.target.mean())
    shared_start = data.fit_start()

    rows = []
    for start in (*starts, str(shared_start.date())):
        model = HarRvForecaster(config)
        model.sample_floor = pd.Timestamp(start)
        model.fit(data)
        pred = model.predict(data, "test")
        scores = metrics.evaluate(split, pred, train_target_mean, train_rv_mean)
        fit_rows = len(data.daily("train")[[*model.COMPONENTS, "rv_fwd"]]
                       .dropna().loc[lambda f: f.index >= pd.Timestamp(start)])
        intercept, beta_1, beta_5, beta_21 = model._coef
        rows.append({
            "start": start,
            "is_d27": start == str(shared_start.date()),
            "fit_from": model.fitted_range[0], "fit_to": model.fitted_range[1],
            "fit_rows": fit_rows,
            "r2_rv": scores["r2_rv"], "mse_target": scores["mse"],
            "qlike": scores["qlike"], "mz_r2": scores["mz_r2"],
            "gate": scores["gate"],
            "b0": intercept, "b_rv1": beta_1, "b_rv5": beta_5, "b_rv21": beta_21,
        })
    return pd.DataFrame(rows).drop_duplicates(subset="fit_from", keep="last")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="HAR score vs fit-sample start.")
    parser.add_argument("--horizon", default="weekly")
    parser.add_argument("--target", default="vrp")
    parser.add_argument("--iv-leg", default="vix9d")
    parser.add_argument("--test-sampling", default="nonoverlap")
    parser.add_argument("--starts", default=",".join(STARTS),
                        help="Comma-separated ISO fit-start dates.")
    args = parser.parse_args(argv)

    config = ExperimentConfig.load(
        horizon=args.horizon, target=args.target, iv_leg=args.iv_leg,
        test_sampling=args.test_sampling)
    frame = curve(config, tuple(s.strip() for s in args.starts.split(",") if s.strip()))

    print(f"{config.horizon} | target={config.target} | leg={config.iv_leg} | "
          f"test={config.test_sampling}")
    print("r2_rv is the primary metric: OOS R2 on forward realised vol against the "
          "train-mean\nconstant, which is held fixed across rows. Only the fit sample "
          "moves.\n")
    print(f"{'fit from':<12} {'to':<12} {'rows':>6} {'R2rv':>8} {'MSEtgt':>9} "
          f"{'QLIKE':>8} {'corr2':>7} {'b0':>7} {'rv_1':>7} {'rv_5':>7} {'rv_21':>7}")
    for _, r in frame.iterrows():
        mark = "  <- D27" if r["is_d27"] else ""
        print(f"{r['fit_from']:<12} {r['fit_to']:<12} {int(r['fit_rows']):>6} "
              f"{r['r2_rv']:>8.4f} {r['mse_target']:>9.3f} {r['qlike']:>8.4f} "
              f"{r['mz_r2']:>7.3f} {r['b0']:>7.3f} {r['b_rv1']:>7.3f} "
              f"{r['b_rv5']:>7.3f} {r['b_rv21']:>7.3f}{mark}")

    full = frame.iloc[0]
    d27 = frame[frame["is_d27"]].iloc[0]
    print(f"\ncost of D27 on r2_rv: {full['r2_rv']:.4f} (from {full['fit_from']}) "
          f"-> {d27['r2_rv']:.4f} (from {d27['fit_from']}), "
          f"delta {d27['r2_rv'] - full['r2_rv']:+.4f}")
    print(f"cost on target-space MSE: {full['mse_target']:.3f} -> "
          f"{d27['mse_target']:.3f}, delta {d27['mse_target'] - full['mse_target']:+.3f}")


if __name__ == "__main__":
    main()
