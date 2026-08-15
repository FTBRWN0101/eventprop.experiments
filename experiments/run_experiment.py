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

import pandas as pd  #noqa: E402

from core import metrics  #noqa: E402
from core.config import ALGORITHMS, ExperimentConfig  #noqa: E402
from core.dataset import VrpDataset  #noqa: E402
from core.results import save_run  #noqa: E402
from models.base import MODELS  #noqa: E402

logger = logging.getLogger("experiments")

#benchmark for Diebold-Mariano
BENCHMARK = "har_rv"


def run(config: ExperimentConfig, model_names: list[str]) -> tuple[
        dict[str, dict[str, float]], dict[str, pd.Series], dict[str, dict], pd.Series]:
    """Fit and evaluate each model; return ``(metrics, predictions, diagnostics, actual)``."""
    MODELS.discover(_ROOT / "models", "models")
    data = VrpDataset(config)

    #constants available at train time only
    train = data.split("train")
    train_target_mean = float(train.target.mean())
    train_rv_mean = float(train.rv_fwd.mean())

    results: dict[str, dict[str, float]] = {}
    errors: dict[str, object] = {}
    predictions: dict[str, pd.Series] = {}
    diagnostics: dict[str, dict] = {}
    for name in model_names:
        model = MODELS.get(name)(config)
        model.fit(data)
        pred = model.predict(data, "test")
        predictions[name] = pred
        split = data.split("test")
        results[name] = metrics.evaluate(split, pred, train_target_mean, train_rv_mean)
        errors[name] = metrics.squared_errors(split, pred)
        diag = {}
        for attribute in ("fitted_range", "silent_neuron_fraction", "pinned_fraction"):
            if hasattr(model, attribute):
                diag[attribute] = getattr(model, attribute)
        if diag:
            diagnostics[name] = diag

    #Diebold-Mariano vs the benchmark (positive => benchmark worse)
    lag = max(config.horizon_days - 1, 0)
    if BENCHMARK in errors:
        for name in model_names:
            if name == BENCHMARK:
                results[name]["dm_vs_har"] = float("nan")
                continue
            stat, p = metrics.diebold_mariano(errors[BENCHMARK], errors[name], lag=lag)
            results[name]["dm_vs_har"] = stat
            results[name]["dm_p"] = p
    return results, predictions, diagnostics, data.split("test").target


def _print_table(config: ExperimentConfig, results: dict[str, dict[str, float]]) -> None:
    logger.info("\n%s | target=%s | iv_leg=%s | features=%s | test=%s",
                config.horizon, config.target, config.iv_leg,
                config.feature_set, config.test_sampling)
    #r2_rv is primary, gate must be > 0; corr2 is Mincer-Zarnowitz, not OOS
    header = f"{'model':<12} {'n':>5} {'R2rv':>7} {'gate':>7} {'MSE':>9} {'QLIKE':>8} " \
             f"{'corr2':>7} {'disp':>6} {'DMvsHAR':>8} {'p':>6}"
    logger.info(header)
    for name, m in results.items():
        logger.info(
            f"{name:<12} {int(m['n']):>5} {m['r2_rv']:>7.3f} {m['gate']:>7.3f} "
            f"{m['mse']:>9.3f} {m['qlike']:>8.4f} {m['mz_r2']:>7.3f} "
            f"{m['dispersion']:>6.2f} "
            f"{m.get('dm_vs_har', float('nan')):>8.3f} {m.get('dm_p', float('nan')):>6.3f}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate VRP forecasters.")
    parser.add_argument("--horizon", default="weekly", choices=["weekly", "monthly"])
    parser.add_argument("--target", default="vrp", choices=["vrp", "rvrp", "vvrp"])
    parser.add_argument("--iv-leg", default="vix9d", choices=["vix", "vix9d"])
    parser.add_argument("--feature-set", default="options+price",
                        choices=["options+price", "price-only"])
    parser.add_argument("--test-sampling", default="nonoverlap",
                        choices=["nonoverlap", "daily"])
    parser.add_argument("--encoding", default="rate",
                        choices=["rate", "population", "delta"],
                        help="Spike encoding, SNN only.")
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
    return parser.parse_args(argv)


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
        holdout_val=args.holdout_val)
    results, predictions, diagnostics, actual = run(
        config, [m.strip() for m in args.models.split(",") if m.strip()])
    _print_table(config, results)
    run_dir = save_run(config, results, predictions, diagnostics, actual)
    logger.info("\nresults written to %s", run_dir)


if __name__ == "__main__":
    main()
