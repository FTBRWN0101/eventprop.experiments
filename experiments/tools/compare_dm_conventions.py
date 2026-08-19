"""Quantify what the DM convention changes did to previously recorded numbers.

Three conventions on one fixed configuration:

* **old** -- ``lag = h-1`` always, normal distribution. Every DM number recorded before
  D39 is on this convention.
* **D39** -- ``lag = 0`` under nonoverlap, normal distribution. The intermediate state
  HARNESS-REVIEW R1 flagged as unmeasured.
* **new** -- lag 0 *and* lag h-1, both with the Harvey-Leybourne-Newbold small-sample
  correction and a ``t(n-1)`` p-value.

The point is that R1 asked for one known configuration to be re-measured before any DM
statistic is quoted. This is that measurement.

    python experiments/tools/compare_dm_conventions.py
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
from run_experiment import fit_predict  #noqa: E402

#pinned locally rather than read from run_experiment.BENCHMARKS. This tool exists to
#price one specific change, the DM convention move recorded in D56, and every number
#it is compared against was measured against har_rv. Following the harness when a
#second benchmark was added would silently change what the tool measures.
BENCHMARK = "har_rv"

#fixed configuration: the harness defaults, baselines only
MODELS = ("har_rv", "constant", "persistence")


def error_series(config: ExperimentConfig, model_names: tuple[str, ...]) -> tuple[
        dict[str, pd.Series], dict[str, pd.Series]]:
    """Per-date squared errors in target space and rv_fwd space, per model."""
    data, predictions, _ = fit_predict(config, list(model_names))
    split = data.split("test")
    return ({n: metrics.squared_errors(split, p) for n, p in predictions.items()},
            {n: metrics.squared_errors_rv(split, p) for n, p in predictions.items()})


def conventions(lag_h: int) -> dict[str, dict[str, object]]:
    """(label, lag, hln) for each convention, in reporting order."""
    return {
        f"old   lag={lag_h} normal": {"lag": lag_h, "hln": False},
        "D39   lag=0 normal": {"lag": 0, "hln": False},
        "new   lag=0 HLN+t": {"lag": 0, "hln": True},
        f"new   lag={lag_h} HLN+t": {"lag": lag_h, "hln": True},
    }


def compare(config: ExperimentConfig,
            model_names: tuple[str, ...] = MODELS) -> pd.DataFrame:
    """One row per (model, space, convention) with the stat and p-value."""
    errors, errors_rv = error_series(config, model_names)
    lag_h = max(config.horizon_days - 1, 0)
    rows = []
    for space, series in (("target", errors), ("rv_fwd", errors_rv)):
        for name in model_names:
            if name == BENCHMARK:
                continue
            for label, kwargs in conventions(lag_h).items():
                stat, p = metrics.diebold_mariano(
                    series[BENCHMARK], series[name], **kwargs)  #type: ignore[arg-type]
                rows.append({"space": space, "model": name, "convention": label,
                             "n": len(series[name]), "stat": stat, "p": p})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="DM statistic under the old and new conventions.")
    parser.add_argument("--horizon", default="weekly")
    parser.add_argument("--target", default="vrp")
    parser.add_argument("--iv-leg", default="vix9d")
    parser.add_argument("--test-sampling", default="nonoverlap")
    parser.add_argument("--models", default=",".join(MODELS))
    args = parser.parse_args(argv)

    config = ExperimentConfig.load(
        horizon=args.horizon, target=args.target, iv_leg=args.iv_leg,
        test_sampling=args.test_sampling)
    names = tuple(m.strip() for m in args.models.split(",") if m.strip())
    frame = compare(config, names)

    print(f"{config.horizon} | target={config.target} | leg={config.iv_leg} | "
          f"test={config.test_sampling} | benchmark={BENCHMARK}")
    print(f"stat > 0 means {BENCHMARK} has the larger loss, i.e. the row model wins.\n")
    lag_h = max(config.horizon_days - 1, 0)
    for space in ("target", "rv_fwd"):
        block = frame[frame["space"] == space]
        print(f"--- {space} space " + "-" * 52)
        print(f"{'convention':<22}" + "".join(
            f"{m:>22}" for m in block["model"].unique()))
        for label in conventions(lag_h):
            cells = ""
            for model in block["model"].unique():
                row = block[(block["convention"] == label)
                            & (block["model"] == model)].iloc[0]
                cells += f"{row['stat']:>13.3f} (p{row['p']:>6.3f})"
            print(f"{label:<22}{cells}")
        n = int(block["n"].iloc[0])
        print(f"n = {n}, HLN factor at lag 0 = {metrics.hln_factor(n, 0):.5f}, "
              f"at lag {lag_h} = {metrics.hln_factor(n, lag_h):.5f}\n")


if __name__ == "__main__":
    main()
