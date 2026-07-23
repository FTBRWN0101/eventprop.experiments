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

from core import metrics  # noqa: E402
from core.config import ExperimentConfig  # noqa: E402
from core.dataset import VrpDataset  # noqa: E402
from models.base import MODELS  # noqa: E402

logger = logging.getLogger("experiments")

#benchmark for Diebold-Mariano
BENCHMARK = "har_rv"


def run(config: ExperimentConfig, model_names: list[str]) -> dict[str, dict[str, float]]:
    """Fit and evaluate each model; return ``{model: metrics}``."""
    MODELS.discover(_ROOT / "models", "models")
    data = VrpDataset(config)

    results: dict[str, dict[str, float]] = {}
    errors: dict[str, object] = {}
    for name in model_names:
        model = MODELS.get(name)(config)
        model.fit(data)
        pred = model.predict(data, "test")
        split = data.split("test")
        results[name] = metrics.evaluate(split, pred)
        errors[name] = metrics.squared_errors(split, pred)

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
    return results


def _print_table(config: ExperimentConfig, results: dict[str, dict[str, float]]) -> None:
    logger.info("\n%s | target=%s | iv_leg=%s | features=%s | test=%s",
                config.horizon, config.target, config.iv_leg,
                config.feature_set, config.test_sampling)
    header = f"{'model':<10} {'n':>5} {'MSE':>9} {'MAE':>8} {'dirAcc':>7} " \
             f"{'QLIKE':>8} {'MZ-R2':>7} {'DMvsHAR':>8} {'p':>6}"
    logger.info(header)
    for name, m in results.items():
        logger.info(
            f"{name:<10} {int(m['n']):>5} {m['mse']:>9.3f} {m['mae']:>8.3f} "
            f"{m['dir_acc']:>7.3f} {m['qlike']:>8.4f} {m['mz_r2']:>7.3f} "
            f"{m.get('dm_vs_har', float('nan')):>8.3f} {m.get('dm_p', float('nan')):>6.3f}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate VRP forecasters.")
    parser.add_argument("--horizon", default="weekly", choices=["weekly", "monthly"])
    parser.add_argument("--target", default="vrp", choices=["vrp", "rvrp"])
    parser.add_argument("--iv-leg", default="vix9d", choices=["vix", "vix9d"])
    parser.add_argument("--feature-set", default="options+price",
                        choices=["options+price", "price-only"])
    parser.add_argument("--test-sampling", default="nonoverlap",
                        choices=["nonoverlap", "daily"])
    parser.add_argument("--encoding", default="rate",
                        choices=["rate", "latency", "population", "delta"],
                        help="Spike encoding for the SNN (ignored by other models).")
    parser.add_argument("--models", default="har_rv,garch",
                        help="Comma-separated model names to run.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    config = ExperimentConfig.load(
        horizon=args.horizon, target=args.target, iv_leg=args.iv_leg,
        feature_set=args.feature_set, test_sampling=args.test_sampling,
        encoding=args.encoding)
    results = run(config, [m.strip() for m in args.models.split(",") if m.strip()])
    _print_table(config, results)


if __name__ == "__main__":
    main()
