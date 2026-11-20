"""Orchestrator: train and evaluate forecasters on a processed VRP panel."""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

#importable when run as a script
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  #noqa: E402
import pandas as pd  #noqa: E402

from core import metrics  #noqa: E402
from core.config import ALGORITHMS, ISO_ACTIVATIONS, TARGETS, ExperimentConfig  #noqa: E402
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

#the two co-primary metrics and the direction that wins on each. r2_rv has been the
#project's headline throughout; QLIKE is the loss the volatility forecasting literature
#actually scores on, and core/metrics.py has written it to every metrics.csv since the
#harness was built without it ever being headlined. Reporting only one of them is a
#choice about the conclusion rather than a presentational one: on the monthly rv_fwd
#cell the best SNN takes r2_rv at 0.166 against persistence's 0.126 and then loses
#QLIKE 0.93 to 0.525, so "the SNN beats persistence" is true of one metric and false of
#the other. D82 settled the same question for MAE: a metric that splits from the rest
#earns its place precisely because the split is the diagnostic.
HEADLINE_METRICS: tuple[tuple[str, bool], ...] = (("r2_rv", True), ("qlike", False))

#label column and wrap width for the verdict block, which now carries notes long
#enough to need a hanging indent
_LABEL_WIDTH, _LINE_WIDTH = 22, 92


def fit_predict(config: ExperimentConfig, model_names: list[str]) -> tuple[
        VrpDataset, dict[str, pd.Series], dict[str, dict]]:
    """Fit every named model and predict the test split.

    Split out so tools/ can reuse one fit loop instead of copying it.
    """
    MODELS.discover(_ROOT / "models", "models")
    data = VrpDataset(config)
    predictions: dict[str, pd.Series] = {}
    diagnostics: dict[str, dict] = {}
    failed: dict[str, str] = {}
    for name in model_names:
        try:
            model = MODELS.get(name)(config)
            model.fit(data)
            predictions[name] = model.predict(data, config.evaluation_split)
        except Exception as error:
            #one model's failure must not discard every other model's completed work.
            #The arm still exits non-zero, so the sweep records it as failed rather
            #than banking a run that is quietly missing a model.
            logger.exception("[run] %s failed and is dropped from this run", name)
            failed[name] = f"{type(error).__name__}: {error}"
            continue
        diag = {}
        for attribute in ("fitted_range", "fit_tolerance_days",
                          "silent_neuron_fraction", "pinned_fraction"):
            if hasattr(model, attribute):
                diag[attribute] = getattr(model, attribute)
        if diag:
            diagnostics[name] = diag
    if failed:
        diagnostics["__failed__"] = failed
    return data, predictions, diagnostics


def run(config: ExperimentConfig, model_names: list[str]) -> tuple[
        dict[str, dict[str, float]], dict[str, pd.Series], dict[str, dict], pd.Series]:
    """Fit and evaluate each model; return ``(metrics, predictions, diagnostics, actual)``."""
    data, predictions, diagnostics = fit_predict(config, model_names)

    #constants available at train time only
    train = data.split("train")
    train_target_mean = float(train.target.mean())
    train_rv_mean = float(train.rv_fwd.mean())
    split = data.split(config.evaluation_split)

    results: dict[str, dict[str, float]] = {}
    errors: dict[str, object] = {}
    errors_rv: dict[str, object] = {}
    for name in list(predictions):
        pred = predictions[name]
        try:
            results[name] = metrics.evaluate(split, pred, train_target_mean,
                                             train_rv_mean)
            errors[name] = metrics.squared_errors(split, pred)
            errors_rv[name] = metrics.squared_errors_rv(split, pred)
        except Exception as error:
            logger.exception("[run] %s cannot be scored on the %s split",
                             name, config.evaluation_split)
            diagnostics.setdefault("__failed__", {})[name] = (
                f"{type(error).__name__}: {error}")
            results.pop(name, None)
            del predictions[name]

    #Diebold-Mariano vs the benchmark (positive => benchmark worse), both lags.
    #Neither lag is obviously right under nonoverlap sampling: the test rows are already
    #stride-N disjoint so the mechanical MA(h-1) structure is gone (lag 0), but vol is
    #persistent so some serial correlation can survive (lag h-1). Reporting both means
    #no arbitrary choice is being made. Every stat carries the HLN small-sample
    #correction and a t(n-1) p-value.
    suppressed = _add_dm_columns(results, list(predictions), errors, errors_rv,
                                 max(config.horizon_days - 1, 0), diagnostics,
                                 data.daily("full").index)
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
    frame = data.daily(config.evaluation_split)
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


def _trading_gap(calendar: pd.DatetimeIndex, one: str, two: str) -> int:
    """Trading days between two dates, counted on the panel's own calendar."""
    positions = calendar.searchsorted([pd.Timestamp(one), pd.Timestamp(two)])
    return int(abs(positions[1] - positions[0]))


def _samples_comparable(diagnostics: dict[str, dict], calendar: pd.DatetimeIndex,
                        a: str, b: str) -> tuple[bool, str]:
    """Whether a DM between *a* and *b* compares fits on the same sample (D81).

    Returns ``(comparable, reason)``; *reason* is empty when comparable.

    Identical ranges pass. So does a range sitting *inside* the other by no more than
    the **inner** model's own declared allowance, because that is what windowing and
    batch truncation do to a sequence model: they carve a subset out of the sample it
    was given, which handicaps it rather than crediting it. The allowance is read from
    the inner model on purpose. Taking the wider one's would let a model excuse the
    extra data it holds, which inverts the rule.

    A range reaching *beyond* the other fails at any size. har_rv_full starts in
    1990 against har_rv's 2011, and that is the comparison the rule exists to stop.
    A range differing by more than the allowance fails too, which is what catches a
    model that quietly fitted a different sample: under ``--holdout-val``, garch
    keeps 2017-2019 while har_rv loses it, and Option A would have waved that
    through because both were *permitted* the same span.

    Absent diagnostics count as matching. A model reporting no range is not evidence
    of a mismatch, and refusing it would silence every baseline that reports none.
    """
    range_a = diagnostics.get(a, {}).get("fitted_range")
    range_b = diagnostics.get(b, {}).get("fitted_range")
    if range_a is None or range_b is None:
        return True, ""
    start_a, end_a = str(range_a[0]), str(range_a[1])
    start_b, end_b = str(range_b[0]), str(range_b[1])
    if (start_a, end_a) == (start_b, end_b):
        return True, ""

    #one range must sit inside the other; a straddle is two different samples
    inside_a = start_a >= start_b and end_a <= end_b
    inside_b = start_b >= start_a and end_b <= end_a
    if not (inside_a or inside_b):
        return False, (f"fitted {range_a} against {range_b}, neither inside the other")

    #the allowance belongs to whichever model sits inside, because it is that model's
    #own windowing that carved the subset. Taking the wider model's tolerance would
    #let it excuse the extra data it holds, which is the opposite of the rule.
    inner = a if inside_a else b
    allowance = diagnostics.get(inner, {}).get("fit_tolerance_days", 0)
    start_gap = _trading_gap(calendar, start_a, start_b)
    end_gap = _trading_gap(calendar, end_a, end_b)
    if max(start_gap, end_gap) > allowance:
        return False, (f"fitted {range_a} against {range_b}, inside by "
                       f"{start_gap}/{end_gap} trading days against an allowance "
                       f"of {allowance}")
    return True, ""


def _scored_dates_match(errors: dict[str, object], a: str, b: str) -> bool:
    """True when two models were scored on identical dates.

    The project rule is "fitted on identical dates and scored on identical dates".
    Only the first clause was ever enforced, which D77 recorded as shaky. The error
    series carry the scored index, so this is the second clause.
    """
    index_a = getattr(errors.get(a), "index", None)
    index_b = getattr(errors.get(b), "index", None)
    if index_a is None or index_b is None:
        return True
    return len(index_a) == len(index_b) and bool((index_a == index_b).all())


def _add_dm_columns(results: dict[str, dict[str, float]], model_names: list[str],
                    errors: dict[str, object], errors_rv: dict[str, object],
                    lag_h: int, diagnostics: dict[str, dict],
                    calendar: pd.DatetimeIndex) -> dict[str, list[str]]:
    """Write the DM stat/p column pairs onto *results*, in place.

    Two benchmarks x two error spaces x two lags, so sixteen columns per model.
    Returns ``{benchmark: [models whose DM was suppressed]}``.

    A DM statistic between models fitted on different samples is one the project's
    own rule forbids, and it was previously computed and printed anyway for the
    ``_full`` arms. Those cells are now NaN with a warning. Suppressing rather than
    raising keeps ``har_rv,har_rv_full`` runnable, which is the whole point of those
    arms: they price the D27 restriction, they are not candidates for a DM.

    Both clauses of the rule are checked: the fit samples must be comparable per
    D81, and the scored dates must be identical.
    """
    lags = {"lag0": 0, "lagh": lag_h}
    suppressed: dict[str, list[str]] = {}
    for benchmark in BENCHMARKS:
        if benchmark not in errors:
            continue
        for name in model_names:
            comparable, reason = _samples_comparable(
                diagnostics, calendar, benchmark, name)
            if comparable and not _scored_dates_match(errors, benchmark, name):
                comparable, reason = False, "scored on different dates"
            if not comparable:
                suppressed.setdefault(benchmark, []).append(name)
                logger.warning(
                    "[dm] %s vs %s suppressed: %s. A DM between models fitted or "
                    "scored on different samples is not a valid comparison.",
                    name, benchmark, reason)
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


def _verdict_line(label: str, text: str) -> None:
    """One ``label   text`` line of the verdict, wrapped under a hanging indent.

    The block used to be one short phrase per label. The metric-disagreement note is a
    sentence, and a sentence that runs off the right of an 80-column terminal is a
    sentence nobody reads, so it wraps into the label column instead.
    """
    indent = " " * (2 + _LABEL_WIDTH + 1)
    head = f"  {label:<{_LABEL_WIDTH}} "
    for line in textwrap.wrap(text, width=_LINE_WIDTH, initial_indent=head,
                              subsequent_indent=indent) or [head + "none"]:
        logger.info("%s", line)


def _best_on(results: dict[str, dict[str, float]], metric: str,
             higher_is_better: bool) -> tuple[str, float] | None:
    """The model with the best rankable *metric*, or None when no model has one.

    Only finite values are ranked. A non-positive or non-finite forward-vol forecast
    makes QLIKE NaN for that model/run: it is an invalid forecast rather than a date
    the metric may discard. Filtering here stops an unrankable value from silently
    winning a comparison it never entered.
    """
    scored = [(name, float(m[metric])) for name, m in results.items()
              if m.get(metric) is not None and np.isfinite(m.get(metric, float("nan")))]
    if not scored:
        return None
    pick = max if higher_is_better else min
    return pick(scored, key=lambda kv: kv[1])


def _unrankable(results: dict[str, dict[str, float]], metric: str) -> list[str]:
    """Models whose *metric* cannot be ranked, each with the value it actually took."""
    out = []
    for name, m in results.items():
        value = m.get(metric)
        if value is None:
            out.append(f"{name} (absent)")
        elif not np.isfinite(value):
            out.append(f"{name} ({float(value):.4g})")
    return out


def _headline_note(results: dict[str, dict[str, float]]) -> str:
    """Whether the co-primary metrics agree on a winner, stated in a sentence.

    A disagreement is the single most useful thing this block can say. r2_rv rewards a
    forecast for tracking the variance of realised vol; QLIKE is a ratio loss that
    punishes an under-dispersed forecast on the low-vol days r2_rv barely notices. When
    they split, neither is the tie-break and the honest report is both.
    """
    winners = {metric: _best_on(results, metric, higher)
               for metric, higher in HEADLINE_METRICS}
    if any(win is None for win in winners.values()):
        missing = [m for m, win in winners.items() if win is None]
        return (f"cannot be compared: no model has a rankable "
                f"{', '.join(missing)}, so the ranking is undefined on it.")

    names = {metric: win[0] for metric, win in winners.items()}
    if len(set(names.values())) == 1:
        only = next(iter(set(names.values())))
        return f"agree: {only} is first on both r2_rv and QLIKE."

    r2_name, r2_value = winners["r2_rv"]
    ql_name, ql_value = winners["qlike"]
    #every figure at .4f, matching the "best <metric>" lines directly above: the same
    #number rounded two ways in adjacent lines reads as two different numbers
    return (f"DISAGREE: r2_rv ranks {r2_name} first at {r2_value:.4f}, QLIKE ranks "
            f"{ql_name} first at {ql_value:.4f}. {r2_name} scores QLIKE "
            f"{_finite(results[r2_name].get('qlike')):.4f} against {ql_name}'s "
            f"{ql_value:.4f}, and {ql_name} scores r2_rv "
            f"{_finite(results[ql_name].get('r2_rv')):.4f} against {r2_name}'s "
            f"{r2_value:.4f}. Neither metric is the tie-break; quote both or say which "
            f"one the claim rests on.")


def _print_verdict(config: ExperimentConfig, results: dict[str, dict[str, float]],
                   diagnostics: dict[str, dict] | None) -> None:
    """Summarise the four tables into the handful of lines worth acting on.

    Sixteen DM columns is more than anyone reads. This says which model won on each
    co-primary metric, whether the two agree, whether anything cleared the constant,
    and which comparisons were refused.
    """
    if not results:
        return
    logger.info("\nVerdict  (%s space, %s, HLN + t(n-1), alpha = %.2f)",
                "rv_fwd" if HEADLINE_SPACE == "rv" else config.target,
                HEADLINE_LAG.replace("lag", "lag "), ALPHA)

    for metric, higher in HEADLINE_METRICS:
        best = _best_on(results, metric, higher)
        direction = "higher is better" if higher else "lower is better"
        text = (f"{best[0]} {best[1]:.4f}  ({direction})" if best
                else f"none scored  ({direction})")
        _verdict_line(f"best {metric}", text)
        unrankable = _unrankable(results, metric)
        if unrankable:
            #named rather than dropped: an arm with non-rankable QLIKE made an invalid
            #forward-vol forecast, which is a finding about the arm, not a table gap
            _verdict_line(f"{metric} not rankable",
                          f"{', '.join(unrankable)} - excluded from the ranking above.")

    _verdict_line("headline metrics", _headline_note(results))

    #the gate is R2 against the train-mean constant, so > 0 is the bar for any claim
    #of the form "we beat the benchmark"
    cleared = [n for n, m in results.items()
               if n != "constant" and _finite(m.get("gate")) > 0]
    _verdict_line("clears the gate", ", ".join(cleared) or "none")

    for benchmark in BENCHMARKS:
        if benchmark not in results:
            continue
        wins = _significant_wins(results, benchmark)
        _verdict_line(f"beats {benchmark}", ", ".join(wins) or "none")

    suppressed = (diagnostics or {}).get("__dm_suppressed__") or {}
    for benchmark, names in suppressed.items():
        _verdict_line("DM suppressed",
                      f"{', '.join(names)}  (fit sample differs from {benchmark})")

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
                                 "delta_adaptive", "learnt_delta", "learnt_conv"],
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
    parser.add_argument("--evaluation-split", choices=("val", "test"), default="test",
                        help="Score validation while selecting; score final test only after "
                             "a configuration is predeclared.")
    parser.add_argument("--loss-shaping", action="store_true",
                        help="Contribution #6: exponential loss weight, EventProp only.")
    parser.add_argument("--w-in-scale", type=float, default=2.5,
                        help="Input->hidden weight scale (SNN). Raise it to give the "
                             "hidden layer more external drive relative to recurrence.")
    parser.add_argument("--w-rec-scale", type=float, default=1.5,
                        help="Hidden->hidden recurrent weight scale (SNN).")
    parser.add_argument("--num-hidden", type=int, default=256,
                        help="Hidden layer width. The proposal specifies 256 to "
                             "1024; only 256 was ever run.")
    parser.add_argument("--checkpoint-epochs", type=int, default=10,
                        help="Epochs between checkpoints. Lower it to resolve the "
                             "training trajectory; D110 found it swings inside a run.")
    parser.add_argument("--lstm-hidden-size", type=int, default=64,
                        help="LSTM hidden units. The control baseline is swept "
                             "too, so it is not judged at one fixed setting.")
    parser.add_argument("--lstm-num-layers", type=int, default=1)
    parser.add_argument("--lstm-dropout", type=float, default=0.0,
                        help="Only meaningful with two or more layers.")
    parser.add_argument("--reset-synapses-between-trials", action="store_true",
                        help="D96: clear synaptic OutPost at the start of every "
                             "trial, which EventProp does not do. SNN only.")
    parser.add_argument("--readout-window", action="store_true",
                        help="D85: score only the end of the trial instead of the "
                             "whole of it, EventProp only.")
    parser.add_argument("--readout-window-fraction", type=float, default=0.2,
                        help="Share of the trial the readout window scores, measured "
                             "back from the trial end (default: the final 20%%).")
    parser.add_argument("--loss-shaping-tau", type=float, default=None,
                        help="Decay constant for the shaped loss "
                             "(default: the sequence length, i.e. exp(-t/T)).")
    parser.add_argument("--reg-target-duty-cycle", type=float, default=0.3,
                        help="Regularisation spike-rate target as a duty cycle, "
                             "SNN only. Third knob of the targeted search.")
    parser.add_argument("--lr-ease-in-batches", type=int, default=None,
                        help="Ramp the learning rate from 1/1000 of its value to "
                             "full over this many batches (default: no ease-in).")
    parser.add_argument("--allow-silent-neurons", action="store_true",
                        help="Continue past the 10%% silent-hidden-neuron threshold "
                             "instead of stopping for review.")
    parser.add_argument("--drop-features", default="",
                        help="Comma-separated feature columns to withhold, for the "
                             "per-signal ablation.")
    parser.add_argument("--restore-features", default="",
                        help="Comma-separated columns to re-admit from "
                             "EXCLUDED_FEATURES, e.g. skew,ts_slope.")
    parser.add_argument("--iso-activation", choices=list(ISO_ACTIVATIONS), default="sigmoid",
                        help="Smooth hidden activation for the iso-architecture control.")
    parser.add_argument("--encoder-checkpoint", default=None,
                        help="Frozen B1 learnt-encoder artifact consumed by an SNN B2 run.")
    parser.add_argument("--campaign-id", default=None,
                        help="Namespace mutable checkpoints and learnt artifacts for one campaign.")
    parser.add_argument("--learnt-num-neurons", type=int, default=32,
                        help="Explicit learnt-encoder output width.")
    return parser.parse_args(argv)


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    config = ExperimentConfig.load(
        horizon=args.horizon, target=args.target, iv_leg=args.iv_leg,
        feature_set=args.feature_set, test_sampling=args.test_sampling,
        model=model_names[0] if len(model_names) == 1 else "multiple",
        encoding=args.encoding, algorithm=args.algorithm, seed=args.seed,
        input_window=args.input_window,
        sample_start=args.sample_start, num_epochs=args.num_epochs,
        learning_rate=args.learning_rate, delta_multiplier=args.delta_multiplier,
        holdout_val=args.holdout_val,
        evaluation_split=args.evaluation_split,
        loss_shaping=args.loss_shaping, loss_shaping_tau=args.loss_shaping_tau,
        num_hidden=args.num_hidden, checkpoint_epochs=args.checkpoint_epochs,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers, lstm_dropout=args.lstm_dropout,
        reset_synapses_between_trials=args.reset_synapses_between_trials,
        readout_window=args.readout_window,
        readout_window_fraction=args.readout_window_fraction,
        w_in_scale=args.w_in_scale, w_rec_scale=args.w_rec_scale,
        reg_target_duty_cycle=args.reg_target_duty_cycle,
        lr_ease_in_batches=args.lr_ease_in_batches,
        silent_neuron_abort=not args.allow_silent_neurons,
        drop_features=_split_csv(args.drop_features),
        restore_features=_split_csv(args.restore_features),
        iso_activation=args.iso_activation,
        encoder_checkpoint=Path(args.encoder_checkpoint) if args.encoder_checkpoint else None,
        learnt_num_neurons=args.learnt_num_neurons,
        campaign_id=args.campaign_id)
    results, predictions, diagnostics, actual = run(config, model_names)
    _print_table(config, results, diagnostics)
    run_dir = save_run(config, results, predictions, diagnostics, actual,
                       requested_models=model_names)
    logger.info("\nresults written to %s", run_dir)
    #the run is written either way, so no work is lost, but an arm missing a model
    #is not a success: sweep.py keys resume and --retry-failed on the exit code
    failed = diagnostics.get("__failed__") or {}
    if failed:
        logger.error("no result for %s", ", ".join(
            f"{name} ({reason})" for name, reason in sorted(failed.items())))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
