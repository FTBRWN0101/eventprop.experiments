"""How much of a VRP-target forecast is the model, and how much is just the IV leg?

D28 argues Contribution #3 (options vs price-only) cannot be answered on a VRP target,
because the target is *built from* the implied-vol leg and RV-based models inject
``iv_t`` back through ``Split.to_target``. That is an argument, not a measurement. This
is the measurement, in four parts.

1. **Cancellation.** For the ``vrp`` target the map is affine with a unit coefficient on
   ``iv_t``, and the realised target uses the *same* ``iv_t``, so the error is
   ``rv_hat - rv_fwd`` exactly and the leg drops out. Any MSE/QLIKE/DM comparison on a
   ``vrp`` target is therefore algebraically the same comparison as on ``rv_fwd``. This
   is checked numerically, not assumed, and it does *not* hold for ``rvrp``/``vvrp``.
2. **Variance decomposition** of the prediction. ``var(pred) = cov(pred, iv) -
   cov(pred, rv_hat)``, so the two shares sum to 1 exactly and the IV share is readable.
3. **Trivial and oracle IV models.** ``iv_t - mean(rv_fwd_train)`` is the VIX with a
   constant RV forecast; ``iv_t - rv_fwd`` is a perfect one. They bracket the range any
   price-only model can occupy, so the model's position between them is its real skill.
4. **The direct-target penalty.** ``har_rv``/``garch`` are handed ``iv_t`` by
   ``to_target`` whatever the feature set. ``snn`` and ``lstm`` predict the target
   *directly*, so on a price-only VRP arm they must reconstruct ``iv_t`` from realised
   vol alone. That is the asymmetry that actually breaks the ablation, and it is priced
   here by regressing ``iv_t`` on the price-only columns.

``har_rv`` stands in for the price-only arm: its inputs rv_1/rv_5/rv_21 are exactly the
price-only feature set, and it runs on CPU.

    python experiments/tools/leg_leakage.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  #noqa: E402
import pandas as pd  #noqa: E402

from core import metrics  #noqa: E402
from core.config import ExperimentConfig  #noqa: E402
from core.dataset import VrpDataset  #noqa: E402
from run_experiment import fit_predict  #noqa: E402

MODEL = "har_rv"


def decompose(pred: pd.Series, iv: pd.Series, rv_hat: pd.Series) -> dict[str, float]:
    """Split var(pred) into an IV-leg share and a model-forecast share.

    ``pred = iv - rv_hat``, so ``var(pred) = cov(pred, iv) - cov(pred, rv_hat)`` and the
    shares sum to 1 by construction. A share above 1 means rv_hat co-moves with iv and
    so *damps* the prediction rather than adding independent movement.
    """
    frame = pd.concat({"pred": pred, "iv": iv, "rv_hat": rv_hat}, axis=1).dropna()
    var_pred = float(frame["pred"].var(ddof=1))
    cov = frame.cov(ddof=1)
    return {
        "var_iv": float(frame["iv"].var(ddof=1)),
        "var_rv_hat": float(frame["rv_hat"].var(ddof=1)),
        "var_pred": var_pred,
        "corr_iv_rv_hat": float(frame["iv"].corr(frame["rv_hat"])),
        "share_iv": float(cov.loc["pred", "iv"]) / var_pred,
        "share_rv_hat": -float(cov.loc["pred", "rv_hat"]) / var_pred,
        #a weaker, separate view: how much of pred iv alone linearly explains
        "r2_pred_on_iv": float(frame["pred"].corr(frame["iv"])) ** 2,
    }


def cancellation(pred: pd.Series, data: VrpDataset) -> dict[str, float]:
    """Max gap between the target-space and rv-space squared-error series.

    Zero means the IV leg cancels out of the loss and the two DM tests are one test.
    """
    split = data.split("test")
    target_space = metrics.squared_errors(split, pred)
    rv_space = metrics.squared_errors_rv(split, pred)
    gap = (target_space - rv_space).abs()
    return {"max_abs_gap": float(gap.max()), "mean_target": float(target_space.mean()),
            "mean_rv": float(rv_space.mean())}


def _ols_forecast(data: VrpDataset, columns: list[str],
                  target: str) -> pd.Series:
    """OLS of *target* on *columns*, fitted from the shared start, predicted on test."""
    train = data.daily("train")[[*columns, target]].dropna()
    train = train.loc[train.index >= data.fit_start()]
    design = np.column_stack([np.ones(len(train)), train[columns].to_numpy()])
    coef, *_ = np.linalg.lstsq(design, train[target].to_numpy(), rcond=None)
    features = data.split("test").features
    test_design = np.column_stack([np.ones(len(features)),
                                   features[columns].to_numpy()])
    return pd.Series(test_design @ coef, index=features.index)


def iv_reconstruction(config: ExperimentConfig, data: VrpDataset,
                      rv_hat: pd.Series) -> dict[str, float]:
    """Price the handicap a *direct-target* price-only model carries on a VRP target.

    ``snn``/``lstm`` output the target itself, so a price-only arm has to produce
    ``iv_t - rv_hat`` without ever seeing ``iv_t``. Two prices are reported:

    * ``mse_direct_joint`` -- the best *linear* price-only model of the vrp target,
      fitted jointly. This is the honest one, and the fair comparison against the
      ``to_target`` model's ``mse_rv_space``, since both are MSEs on the vrp target.
    * ``mse_direct_stacked`` -- rebuild iv_t by OLS, then subtract har_rv's rv_hat. An
      upper bound: fitting the two pieces separately cannot beat fitting them together.

    Both are linear, so both understate what a sequence model over a whole window could
    do. They bound the handicap, not measure it exactly.
    """
    columns = data.feature_columns(data.daily("full"))
    split = data.split("test")
    iv_hat = _ols_forecast(data, columns, config.iv_column)
    vrp_hat = _ols_forecast(data, columns, f"vrp_{config.iv_leg}")

    frame = pd.concat({"iv": split.iv, "iv_hat": iv_hat, "rv_hat": rv_hat,
                       "rv_fwd": split.rv_fwd, "vrp_hat": vrp_hat,
                       "vrp": split.iv - split.rv_fwd}, axis=1).dropna()
    rv_error = frame["rv_hat"] - frame["rv_fwd"]
    iv_error = frame["iv_hat"] - frame["iv"]
    return {
        "columns": len(columns),
        "iv_r2": float(1.0 - (iv_error ** 2).sum()
                       / ((frame["iv"] - frame["iv"].mean()) ** 2).sum()),
        "iv_rmse": float(np.sqrt((iv_error ** 2).mean())),
        "mse_rv_space": float((rv_error ** 2).mean()),
        "mse_direct_joint": float(((frame["vrp_hat"] - frame["vrp"]) ** 2).mean()),
        "mse_direct_stacked": float(((rv_error + iv_error) ** 2).mean()),
    }


def diagnose(config: ExperimentConfig, model: str = MODEL) -> dict[str, Any]:
    """Fit the price-only model and score it beside the trivial and oracle IV models."""
    data, predictions, diagnostics = fit_predict(config, [model])
    split = data.split("test")
    train = data.split("train")
    train_rv_mean = float(train.rv_fwd.mean())
    train_target_mean = float(train.target.mean())

    pred = predictions[model]
    rv_hat = split.to_rv_fwd(pred)
    #the VIX with a constant RV forecast, and the same with a perfect one
    trivial = split.to_target(pd.Series(train_rv_mean, index=split.rv_fwd.index))
    oracle = split.to_target(split.rv_fwd)
    constant = pd.Series(train_target_mean, index=split.target.index)

    arms = {model: pred, "trivial_iv": trivial, "oracle_iv": oracle,
            "constant": constant}
    scores = {name: metrics.evaluate(split, p, train_target_mean, train_rv_mean)
              for name, p in arms.items()}

    lag_h = max(config.horizon_days - 1, 0)
    dm = {f"lag{lag}": metrics.diebold_mariano(
              metrics.squared_errors(split, trivial),
              metrics.squared_errors(split, pred), lag=lag)
          for lag in (0, lag_h)}

    span = scores["oracle_iv"]["gate"] - scores["trivial_iv"]["gate"]
    #every arm's IMPLIED rv forecast, and how much of it is just the leg
    implied = {name: float(split.to_rv_fwd(p).corr(split.iv)) for name, p in arms.items()}
    return {
        "model": model,
        "target": config.target,
        "fitted_range": diagnostics.get(model, {}).get("fitted_range"),
        "decomposition": decompose(pred, split.iv, rv_hat),
        "cancellation": cancellation(pred, data),
        "reconstruction": iv_reconstruction(config, data, rv_hat),
        "scores": scores,
        "implied_corr_iv": implied,
        "dm_vs_trivial": dm,
        "lag_h": lag_h,
        "corr_pred_trivial": float(pred.corr(trivial)),
        #how far along the road from "just the VIX" to "a perfect RV forecast"
        "informativeness": ((scores[model]["gate"] - scores["trivial_iv"]["gate"]) / span
                            if span > 0 else float("nan")),
        "var_target": float(split.target.var(ddof=1)),
        "var_rv_fwd": float(split.rv_fwd.var(ddof=1)),
    }


def _print_scores(out: dict[str, Any]) -> None:
    scores = out["scores"]
    implied = out["implied_corr_iv"]
    print(f"\n  {'arm':<12} {'n':>5} {'MSE':>9} {'gate':>8} {'R2rv':>8} "
          f"{'corr2':>7} {'QLIKE':>8} {'r(rv_hat,iv)':>13}")
    for name, s in scores.items():
        print(f"  {name:<12} {int(s['n']):>5} {s['mse']:>9.3f} {s['gate']:>8.4f} "
              f"{s['r2_rv']:>8.4f} {s['mz_r2']:>7.4f} {s['qlike']:>8.4f} "
              f"{implied[name]:>13.4f}")


def _print_block(out: dict[str, Any], config: ExperimentConfig) -> None:
    model, dec = str(out["model"]), out["decomposition"]
    print(f"target={config.target} | leg={config.iv_leg} | "
          f"features={config.feature_set} | model={model} "
          f"(fitted {out['fitted_range']})")
    print(f"  var(target)={out['var_target']:.3f}  var(rv_fwd)={out['var_rv_fwd']:.3f}")

    if config.target != "rv_fwd":
        print("\n  1. does the IV leg cancel out of the loss?")
        can = out["cancellation"]
        print(f"     mean sq err, target space {can['mean_target']:>9.4f}")
        print(f"     mean sq err, rv space     {can['mean_rv']:>9.4f}")
        print(f"     max per-date |difference| {can['max_abs_gap']:>9.2e}"
              f"   -> {'CANCELS' if can['max_abs_gap'] < 1e-8 else 'does NOT cancel'}")

        print("\n  2. prediction decomposition, pred = iv_t - rv_hat")
        print(f"     var(iv_t)          {dec['var_iv']:>10.3f}")
        print(f"     var(rv_hat)        {dec['var_rv_hat']:>10.3f}")
        print(f"     var(pred)          {dec['var_pred']:>10.3f}")
        print(f"     corr(iv, rv_hat)   {dec['corr_iv_rv_hat']:>10.3f}")
        print(f"     share from IV leg  {dec['share_iv']:>10.3f}")
        print(f"     share from rv_hat  {dec['share_rv_hat']:>10.3f}")
        print(f"     corr2(pred, iv)    {dec['r2_pred_on_iv']:>10.3f}")

    print("\n  3. scores against the trivial and oracle IV models")
    print("     r(rv_hat, iv) is the correlation between the arm's IMPLIED forward-vol")
    print("     forecast and the IV leg. 1.000 means the arm is an affine map of VIX.")
    _print_scores(out)

    if config.target != "rv_fwd":
        print(f"\n     corr(pred, trivial_iv) = {out['corr_pred_trivial']:.4f}")
        print(f"     DM, trivial_iv vs {model} (stat > 0 => {model} wins):")
        for label, (stat, p) in out["dm_vs_trivial"].items():
            print(f"       {label:<6} stat {stat:>7.3f}   p {p:>6.3f}")
        #identical to r2_rv by construction: the trivial model's rv-space forecast IS
        #the train-mean constant, so this normalises to the same ratio. A consistency
        #check on the arithmetic, not a second measurement.
        print(f"     informativeness (model - trivial)/(oracle - trivial) = "
              f"{out['informativeness']:.4f}  (== R2rv by construction)")

    if config.target == "rv_fwd":
        print("\n  4. direct-target penalty: zero here by construction, there is no "
              "iv_t to rebuild.")
        return
    rec = out["reconstruction"]
    print("\n  4. direct-target penalty (snn/lstm predict the target, not rv_fwd)")
    print(f"     OLS of iv_t on the {int(rec['columns'])} price-only columns:"
          f"  test R2 {rec['iv_r2']:.4f}, RMSE {rec['iv_rmse']:.3f} vol points")
    print(f"     MSE via to_target, iv_t handed over    {rec['mse_rv_space']:>9.3f}")
    print(f"     MSE direct on vrp, jointly fitted      {rec['mse_direct_joint']:>9.3f}"
          f"  ({rec['mse_direct_joint'] / rec['mse_rv_space']:.2f}x)")
    print(f"     MSE direct on vrp, stacked (bound)     {rec['mse_direct_stacked']:>9.3f}"
          f"  ({rec['mse_direct_stacked'] / rec['mse_rv_space']:.2f}x)")


def _verdict(vrp: dict[str, Any], rv: dict[str, Any]) -> list[str]:
    """Plain-language reading of the numbers, one finding per line."""
    model = str(vrp["model"])
    can, dec, rec = vrp["cancellation"], vrp["decomposition"], vrp["reconstruction"]
    vrp_s, rv_s = vrp["scores"], rv["scores"]
    lines = []

    lines.append(
        "The IV leg cancels EXACTLY out of the loss. vrp = iv_t - rv_fwd is affine with "
        "a unit coefficient on the same iv_t the prediction uses, so the error is "
        f"rv_hat - rv_fwd whatever the target (max gap {can['max_abs_gap']:.1e}). "
        "MSE, QLIKE and every DM statistic on the vrp target are numerically identical "
        "to the same statistic on rv_fwd -- for every model, not just this one. D40's "
        "two DM columns are one column on this target."
        if can["max_abs_gap"] < 1e-8 else
        "The IV leg does NOT cancel out of the loss on this target.")

    flip = (vrp_s["constant"]["r2_rv"] > vrp_s[model]["r2_rv"]
            and rv_s["constant"]["r2_rv"] < rv_s[model]["r2_rv"])
    lines.append(
        f"But the BENCHMARK changes with the target, and that is the real leak. On the "
        f"vrp target the constant predicts mean(vrp_train), whose implied forward-vol "
        f"forecast is iv_t minus a scalar -- correlation with the IV leg exactly "
        f"{vrp['implied_corr_iv']['constant']:.3f}. So the 'zero-skill' baseline is "
        f"secretly a VIX-based RV forecast and scores R2rv "
        f"{vrp_s['constant']['r2_rv']:.4f}, against {model}'s "
        f"{vrp_s[model]['r2_rv']:.4f}. On the rv_fwd target the same constant scores "
        f"{rv_s['constant']['r2_rv']:.4f} against {model}'s "
        f"{rv_s[model]['r2_rv']:.4f}."
        + (" The ranking FLIPS on identical test dates, purely from the target choice."
           if flip else ""))

    lines.append(
        f"The prediction is mostly the leg: the IV term supplies "
        f"{dec['share_iv']:.2f} of the prediction's variance and the model's own "
        f"forecast {dec['share_rv_hat']:+.2f}, because rv_hat is already "
        f"{dec['corr_iv_rv_hat']:.3f}-correlated with iv_t. gate reads "
        f"{vrp_s[model]['gate']:+.4f} and corr2 {vrp_s[model]['mz_r2']:.4f} on vrp "
        f"against {rv_s[model]['gate']:+.4f} and {rv_s[model]['mz_r2']:.4f} on rv_fwd. "
        f"Same forecasts, same dates. Neither vrp figure is readable as skill.")

    lines.append(
        f"Second asymmetry: har_rv and garch route through to_target and are handed "
        f"iv_t whatever the feature set, while snn and lstm predict the target "
        f"directly. A price-only sequence model on a vrp target must therefore rebuild "
        f"iv_t from realised vol alone (OLS test R2 {rec['iv_r2']:.3f}, RMSE "
        f"{rec['iv_rmse']:.2f} vol points). Its best linear price-only fit scores MSE "
        f"{rec['mse_direct_joint']:.3f}, which BEATS the to_target route's "
        f"{rec['mse_rv_space']:.3f} -- not because it forecasts better but because it "
        f"can collapse toward the constant, which the vrp target rewards (D7/D8). The "
        f"two model families are not solving the same problem on this target.")

    lines.append(
        "VERDICT: the ablation on a VRP target does NOT carry the information "
        "Contribution #3 needs. The loss itself is clean -- the leg cancels -- but the "
        "reported quantities are not: the benchmark is options-informed, gate and corr2 "
        "are uninterpretable, and the direct-target and to_target model families face "
        "different problems. Every one of those defects disappears on target=rv_fwd, "
        "and the model-vs-model losses and DM statistics are unchanged by the switch. "
        "The cost of moving is zero in the loss and the gain is that the headline "
        "numbers mean what they say. Recommend switching Contribution #3 to rv_fwd.")
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Is the price-only arm on a VRP target just reporting the VIX?")
    parser.add_argument("--horizon", default="weekly")
    parser.add_argument("--iv-leg", default="vix9d")
    parser.add_argument("--feature-set", default="price-only")
    parser.add_argument("--test-sampling", default="nonoverlap")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args(argv)

    base = dict(horizon=args.horizon, iv_leg=args.iv_leg,
                feature_set=args.feature_set, test_sampling=args.test_sampling)
    vrp_config = ExperimentConfig.load(target="vrp", **base)
    vrp = diagnose(vrp_config, args.model)
    _print_block(vrp, vrp_config)

    #the contrast D28 proposes: the same model on the leg-free target
    print("\n" + "=" * 74)
    print("contrast: the same model on the rv_fwd target (no IV leg in the prediction)")
    print("=" * 74)
    rv_config = ExperimentConfig.load(target="rv_fwd", **base)
    rv = diagnose(rv_config, args.model)
    _print_block(rv, rv_config)

    print("\n" + "-" * 74)
    for line in _verdict(vrp, rv):
        print(f"* {line}\n")


if __name__ == "__main__":
    main()
