"""Orchestrator: train and evaluate forecasters on a processed VRP panel."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

#importable when run as a script
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  #noqa: E402
import pandas as pd  #noqa: E402

from core import metrics  #noqa: E402
from core.config import ALGORITHMS, TARGETS, ExperimentConfig  #noqa: E402
from core.dataset import VrpDataset  #noqa: E402
from core.results import save_run  #noqa: E402
from models.base import MODELS  #noqa: E402

logger = logging.getLogger("experiments")

#benchmarks for Diebold-Mariano. GARCH is here as well as HAR-RV because on the
#D27-matched sample GARCH is the stronger econometric baseline, so testing only
#against HAR-RV tests against the weaker of the two.
BENCHMARKS: tuple[str, ...] = ("har_rv", "garch")
#the space and lag the verdict block reads, out of the four combinations reported
HEADLINE_SPACE, HEADLINE_LAG = "rv", "lag0"
ALPHA = 0.05


def fit_predict(config: ExperimentConfig, model_names: list[str]) -> tuple[
        VrpDataset, dict[str, pd.Series], dict[str, dict]]:
    """Fit every named model and predict the test split.

    Split out so tools/ can reuse one fit loop instead of copying it.
    """
    MODELS.discover(_ROOT / "models", "models")
    data = VrpDataset(config)
    predictions: dict[str, pd.Series] = {}
    diagnostics: dict[str, dict] = {}
    for name in model_names:
        model = MODELS.get(name)(config)
        model.fit(data)
        predictions[name] = model.predict(data, "test")
        diag = {}
        for attribute in ("fitted_range", "silent_neuron_fraction", "pinned_fraction"):
            if hasattr(model, attribute):
                diag[attribute] = getattr(model, attribute)
        if diag:
            diagnostics[name] = diag
    return data, predictions, diagnostics


def run(config: ExperimentConfig, model_names: list[str]) -> tuple[
        dict[str, dict[str, float]], dict[str, pd.Series], dict[str, dict], pd.Series]:
    """Fit and evaluate each model; return ``(metrics, predictions, diagnostics, actual)``."""
    data, predictions, diagnostics = fit_predict(config, model_names)

    #constants available at train time only
    train = data.split("train")
    train_target_mean = float(train.target.mean())
    train_rv_mean = float(train.rv_fwd.mean())
    split = data.split("test")

    results: dict[str, dict[str, float]] = {}
    errors: dict[str, object] = {}
    errors_rv: dict[str, object] = {}
    for name, pred in predictions.items():
        results[name] = metrics.evaluate(split, pred, train_target_mean, train_rv_mean)
        errors[name] = metrics.squared_errors(split, pred)
        errors_rv[name] = metrics.squared_errors_rv(split, pred)

    #Diebold-Mariano vs the benchmark (positive => benchmark worse), both lags.
    #Neither lag is obviously right under nonoverlap sampling: the test rows are already
    #stride-N disjoint so the mechanical MA(h-1) structure is gone (lag 0), but vol is
    #persistent so some serial correlation can survive (lag h-1). Reporting both means
    #no arbitrary choice is being made. Every stat carries the HLN small-sample
    #correction and a t(n-1) p-value.
    suppressed = _add_dm_columns(results, list(predictions), errors, errors_rv,
                                 max(config.horizon_days - 1, 0), diagnostics)
    if suppressed:
        diagnostics["__dm_suppressed__"] = suppressed
    binding = _winsor_binding(data, config)
    if binding is not None:
        diagnostics["__winsor__"] = binding
    return results, predictions, diagnostics, split.target


def _winsor_binding(data: VrpDataset, config: ExperimentConfig) -> dict | None:
    """Fraction of scored test rows where the winsor clip actually bound.

    ``Split.to_rv_fwd`` is not an exact inverse on a clipped row, so ``r2_rv`` and
    QLIKE on this arm carry an error proportional to how often the bound binds.
    Returns None on every other target.
    """
    if config.target != "rvrp_winsor":
        return None
    frame = data.daily("test")
    raw_column = f"rvrp_{config.iv_leg}"
    if raw_column not in frame.columns:
        return None
    scored = data.split("test").target.index
    pair = frame.loc[frame.index.intersection(scored), [raw_column, config.target_column]]
    pair = pair.dropna()
    if pair.empty:
        return None
    clipped = ~np.isclose(pair[raw_column], pair[config.target_column])
    fraction = float(clipped.mean())
    logger.info("[winsor] clip binds on %d/%d scored test rows (%.1f%%)",
                int(clipped.sum()), len(pair), fraction * 100)
    return {"rows": int(len(pair)), "clipped": int(clipped.sum()),
            "fraction": fraction}


def _fit_ranges_match(diagnostics: dict[str, dict], a: str, b: str) -> bool:
    """True when two models report the same fitted date range.

    Absent diagnostics count as matching: a model that reports no range is not
    evidence of a mismatch, and refusing to test it would silence the baselines.
    """
    range_a = diagnostics.get(a, {}).get("fitted_range")
    range_b = diagnostics.get(b, {}).get("fitted_range")
    if range_a is None or range_b is None:
        return True
    return list(range_a) == list(range_b)


def _add_dm_columns(results: dict[str, dict[str, float]], model_names: list[str],
                    errors: dict[str, object], errors_rv: dict[str, object],
                    lag_h: int, diagnostics: dict[str, dict]) -> dict[str, list[str]]:
    """Write the DM stat/p column pairs onto *results*, in place.

    Two benchmarks x two error spaces x two lags, so sixteen columns per model.
    Returns ``{benchmark: [models whose DM was suppressed]}``.

    A DM statistic between models fitted on different samples is one the project's
    own rule forbids, and it was previously computed and printed anyway for the
    ``_full`` arms. Those cells are now NaN with a warning. Suppressing rather than
    raising keeps ``har_rv,har_rv_full`` runnable, which is the whole point of those
    arms: they price the D27 restriction, they are not candidates for a DM.
    """
    lags = {"lag0": 0, "lagh": lag_h}
    suppressed: dict[str, list[str]] = {}
    for benchmark in BENCHMARKS:
        if benchmark not in errors:
            continue
        for name in model_names:
            comparable = _fit_ranges_match(diagnostics, benchmark, name)
            if not comparable:
                suppressed.setdefault(benchmark, []).append(name)
                logger.warning(
                    "[dm] %s vs %s suppressed: fitted %s against %s. A DM between "
                    "models fitted on different samples is not a valid comparison.",
                    name, benchmark,
                    diagnostics.get(name, {}).get("fitted_range"),
                    diagnostics.get(benchmark, {}).get("fitted_range"))
            for space, series in (("target", errors), ("rv", errors_rv)):
                for suffix, lag in lags.items():
                    if name == benchmark or not comparable:
                        stat, p = float("nan"), float("nan")
                    else:
                        stat, p = metrics.diebold_mariano(
                            series[benchmark], series[name], lag=lag)
                    results[name][f"dm_{benchmark}_{space}_{suffix}"] = stat
                    results[name][f"dm_{benchmark}_{space}_{suffix}_p"] = p
    return suppressed


def _print_table(config: ExperimentConfig, results: dict[str, dict[str, float]],
                 diagnostics: dict[str, dict] | None = None) -> None:
    logger.info("\n%s | target=%s | iv_leg=%s | features=%s | test=%s",
                config.horizon, config.target, config.iv_leg,
                config.feature_set, config.test_sampling)
    lag_h = max(config.horizon_days - 1, 0)
    #r2_rv is primary, gate must be > 0; corr2 is Mincer-Zarnowitz, not OOS
    logger.info(f"{'model':<12} {'n':>5} {'R2rv':>7} {'gate':>7} {'MSE':>9} {'QLIKE':>8} "
                f"{'corr2':>7} {'disp':>6}")
    for name, m in results.items():
        logger.info(
            f"{name:<12} {int(m['n']):>5} {m['r2_rv']:>7.3f} {m['gate']:>7.3f} "
            f"{m['mse']:>9.3f} {m['qlike']:>8.4f} {m['mz_r2']:>7.3f} "
            f"{m['dispersion']:>6.2f}")

    #four DM tables: two benchmarks x two error spaces, both lags in each. Neither
    #lag is obviously right under nonoverlap sampling - the test rows are already
    #stride-N disjoint so the mechanical MA(h-1) structure is gone (lag 0), but vol
    #is persistent so some serial correlation can survive (lag h-1). Reporting both
    #means no arbitrary choice is being made. Every stat carries the HLN small-sample
    #correction and a t(n-1) p-value.
    spaces = (("target", f"{config.target}-space errors"),
              ("rv", "rv_fwd-space errors (the loss r2_rv reports)"))
    for benchmark in BENCHMARKS:
        for space, label in spaces:
            logger.info("\nDM vs %s on %s, HLN + t(n-1). Positive => %s is worse.",
                        benchmark, label, benchmark)
            logger.info(f"{'model':<12} {'DM_L0':>8} {'p':>6} "
                        f"{'DM_L' + str(lag_h):>8} {'p':>6}")
            for name, m in results.items():
                logger.info(
                    f"{name:<12} "
                    f"{m.get(f'dm_{benchmark}_{space}_lag0', float('nan')):>8.3f} "
                    f"{m.get(f'dm_{benchmark}_{space}_lag0_p', float('nan')):>6.3f} "
                    f"{m.get(f'dm_{benchmark}_{space}_lagh', float('nan')):>8.3f} "
                    f"{m.get(f'dm_{benchmark}_{space}_lagh_p', float('nan')):>6.3f}")

    _print_verdict(config, results, diagnostics)


def _significant_wins(results: dict[str, dict[str, float]], benchmark: str) -> list[str]:
    """Models that beat *benchmark* significantly, formatted for the verdict block."""
    wins = []
    for name, m in results.items():
        stat = m.get(f"dm_{benchmark}_{HEADLINE_SPACE}_{HEADLINE_LAG}", float("nan"))
        p = m.get(f"dm_{benchmark}_{HEADLINE_SPACE}_{HEADLINE_LAG}_p", float("nan"))
        if np.isfinite(stat) and np.isfinite(p) and stat > 0 and p < ALPHA:
            wins.append(f"{name} ({stat:+.3f}, p={p:.3f})")
    return wins


def _print_verdict(config: ExperimentConfig, results: dict[str, dict[str, float]],
                   diagnostics: dict[str, dict] | None) -> None:
    """Summarise the four tables into the handful of lines worth acting on.

    Sixteen DM columns is more than anyone reads. This says which model won, whether
    anything cleared the constant, and which comparisons were refused.
    """
    if not results:
        return
    logger.info("\nVerdict  (%s space, %s, HLN + t(n-1), alpha = %.2f)",
                "rv_fwd" if HEADLINE_SPACE == "rv" else config.target,
                HEADLINE_LAG.replace("lag", "lag "), ALPHA)

    best = max(results.items(), key=lambda kv: _finite(kv[1].get("r2_rv")))
    logger.info("  %-22s %s %.3f", "best r2_rv", best[0], best[1]["r2_rv"])

    #the gate is R2 against the train-mean constant, so > 0 is the bar for any claim
    #of the form "we beat the benchmark"
    cleared = [n for n, m in results.items()
               if n != "constant" and _finite(m.get("gate")) > 0]
    logger.info("  %-22s %s", "clears the gate", ", ".join(cleared) or "none")

    for benchmark in BENCHMARKS:
        if benchmark not in results:
            continue
        wins = _significant_wins(results, benchmark)
        logger.info("  %-22s %s", f"beats {benchmark}", ", ".join(wins) or "none")

    suppressed = (diagnostics or {}).get("__dm_suppressed__") or {}
    for benchmark, names in suppressed.items():
        logger.info("  %-22s %s  (fit sample differs from %s)",
                    "DM suppressed", ", ".join(names), benchmark)

    if config.target == "rv_fwd":
        #both columns are R2 against the train mean of the same quantity, so they are
        #the same number here by construction, not by coincidence
        logger.info("  note: on the rv_fwd arm gate == R2rv by construction, the two "
                    "error spaces coincide, and dir_acc is NaN because the target "
                    "never changes sign.")


def _finite(value: float | None) -> float:
    """Value, or -inf when missing or NaN, so max() and comparisons stay well defined."""
    if value is None or not np.isfinite(value):
        return float("-inf")
    return float(value)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate VRP forecasters.")
    parser.add_argument("--horizon", default="weekly", choices=["weekly", "monthly"])
    parser.add_argument("--target", default="vrp", choices=list(TARGETS),
                        help="rv_fwd is the leg-free arm for the options-vs-price "
                             "ablation; it is still sampled on the configured leg.")
    parser.add_argument("--iv-leg", default="vix9d", choices=["vix", "vix9d"])
    parser.add_argument("--feature-set", default="options+price",
                        choices=["options+price", "price-only", "options+price+surface"])
    parser.add_argument("--test-sampling", default="nonoverlap",
                        choices=["nonoverlap", "daily"])
    parser.add_argument("--encoding", default="rate",
                        choices=["rate", "latency", "population", "delta",
                                 "delta_adaptive"],
                        help="Spike encoding, SNN only. The first four are the "
                             "proposal's four strategies; delta_adaptive is the "
                             "primary hypothesis and delta its static control.")
    parser.add_argument("--algorithm", default="eventprop",
                        choices=list(ALGORITHMS),
                        help="SNN training algorithm. The two arms are not the same network, "
                             "see models/snn.py.")
    parser.add_argument("--models", default="har_rv,garch",
                        help="Comma-separated model names to run.")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed for the GPU RNG and tau sampling. 0 is rejected.")
    parser.add_argument("--input-window", type=int, default=None,
                        help="Input sequence length T in trading days "
                             "(default: same as the forecast horizon).")
    parser.add_argument("--sample-start", default=None,
                        help="Uniform sample start date (ISO) for every split/model.")
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--delta-multiplier", type=float, default=1.0)
    parser.add_argument("--holdout-val", action="store_true",
                        help="Hold out 2017-2019 from train as a validation split.")
    parser.add_argument("--drop-features", default="",
                        help="Comma-separated feature columns to withhold, for the "
                             "per-signal ablation.")
    parser.add_argument("--restore-features", default="",
                        help="Comma-separated columns to re-admit from "
                             "EXCLUDED_FEATURES, e.g. skew,ts_slope.")
    return parser.parse_args(argv)


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    config = ExperimentConfig.load(
        horizon=args.horizon, target=args.target, iv_leg=args.iv_leg,
        feature_set=args.feature_set, test_sampling=args.test_sampling,
        encoding=args.encoding, algorithm=args.algorithm, seed=args.seed,
        input_window=args.input_window,
        sample_start=args.sample_start, num_epochs=args.num_epochs,
        learning_rate=args.learning_rate, delta_multiplier=args.delta_multiplier,
        holdout_val=args.holdout_val,
        drop_features=_split_csv(args.drop_features),
        restore_features=_split_csv(args.restore_features))
    results, predictions, diagnostics, actual = run(
        config, [m.strip() for m in args.models.split(",") if m.strip()])
    _print_table(config, results, diagnostics)
    run_dir = save_run(config, results, predictions, diagnostics, actual)
    logger.info("\nresults written to %s", run_dir)


if __name__ == "__main__":
    main()
