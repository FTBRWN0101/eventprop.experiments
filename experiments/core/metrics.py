"""Forecast evaluation metrics and the Diebold-Mariano test.

QLIKE is the exception: it needs the forward-vol forecast, not the signed VRP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from core.dataset import Split


def mse(real: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((real - pred) ** 2))


def mae(real: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(real - pred)))


def directional_accuracy(real: np.ndarray, pred: np.ndarray) -> float:
    """Fraction of forecasts whose sign matches the realised target's sign.

    NaN when the realised target never changes sign. On a single-signed sample any
    forecast with the right constant sign scores 1.0, which is not skill: ``rv_fwd`` is
    strictly positive, so the metric is undefined there by construction. D13 records
    the weaker version of the same problem on VRP, where the constant wins ``dir_acc``
    by recovering the unconditional positive rate.
    """
    mask = (real != 0) & (pred != 0)
    if not mask.any():
        return float("nan")
    signs = np.sign(real[mask])
    if np.unique(signs).size < 2:
        return float("nan")
    return float(np.mean(signs == np.sign(pred[mask])))


def qlike(rv_real: np.ndarray, rv_pred: np.ndarray) -> float:
    """QLIKE loss on variances. Rows with non-positive vol are dropped."""
    mask = (rv_real > 0) & (rv_pred > 0)
    if not mask.any():
        return float("nan")
    ratio = (rv_real[mask] ** 2) / (rv_pred[mask] ** 2)
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def mincer_zarnowitz_r2(real: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """R2, intercept ``a`` and slope ``b`` of ``real = a + b*pred``.

    Coefficients are NaN for a constant forecast, where the fit is rank-deficient.
    """
    if float(np.std(pred)) == 0.0:
        return 0.0, float("nan"), float("nan")
    design = np.column_stack([np.ones_like(pred), pred])
    coef, *_ = np.linalg.lstsq(design, real, rcond=None)
    resid = real - design @ coef
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((real - real.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, float(coef[0]), float(coef[1])


def hln_factor(n: int, lag: int) -> float:
    """Harvey-Leybourne-Newbold (1997) small-sample correction factor for DM.

    HLN eq. (9): ``DM* = [(n + 1 - 2h + h(h-1)/n) / n]^(1/2) * DM``, scored against
    ``t(n-1)`` instead of ``N(0,1)``. ``h`` is the forecast horizon in steps; an
    h-step-ahead forecast error is MA(h-1), so ``h = lag + 1`` for our HAC truncation
    and ``h = 1`` recovers the plain ``sqrt((n-1)/n)`` shrink.
    Harvey, D., Leybourne, S. & Newbold, P. (1997), "Testing the equality of
    prediction mean squared errors", International Journal of Forecasting 13(2),
    281-291. Cross-checked against R forecast::dm.test (R/DM2.R):
    ``k <- ((n + 1 - 2 * h + (h / n) * (h - 1)) / n)^(1 / 2)``, p from
    ``pt(..., df = n - 1)``.
    """
    h = lag + 1
    inner = (n + 1 - 2 * h + h * (h - 1) / n) / n
    #h > (n+1)/2 drives the factor imaginary; the sample is too short for the test
    return float(np.sqrt(inner)) if inner > 0 else float("nan")


def diebold_mariano(err_a: pd.Series, err_b: pd.Series,
                    lag: int = 0, hln: bool = True) -> tuple[float, float]:
    """Diebold-Mariano test on two aligned per-date loss series.

    Returns ``(stat, p_value)``; ``stat > 0`` means model A has the larger loss.
    ``hln=True`` applies the HLN small-sample correction and scores against
    ``t(n-1)``; ``hln=False`` is the uncorrected normal-distribution convention every
    result recorded before D56 used, kept so tools/compare_dm_conventions.py can price
    the change rather than assert it.
    """
    pair = pd.concat([err_a, err_b], axis=1).dropna()
    diff = (pair.iloc[:, 0] - pair.iloc[:, 1]).to_numpy()
    n = len(diff)
    if n < 2:
        return float("nan"), float("nan")
    centred = diff - diff.mean()
    var = float(np.dot(centred, centred) / n)
    for k in range(1, min(lag, n - 1) + 1):
        gamma = float(np.dot(centred[k:], centred[:-k]) / n)
        var += 2.0 * (1.0 - k / (lag + 1)) * gamma
    if var <= 0:
        return float("nan"), float("nan")
    stat = diff.mean() / np.sqrt(var / n)
    if not hln:
        return float(stat), float(2.0 * (1.0 - stats.norm.cdf(abs(stat))))
    factor = hln_factor(n, lag)
    if not np.isfinite(factor):
        return float("nan"), float("nan")
    stat *= factor
    return float(stat), float(2.0 * stats.t.cdf(-abs(stat), df=n - 1))


def r2_against(real: np.ndarray, pred: np.ndarray, benchmark: float) -> float:
    """Out-of-sample R2 against the train-mean constant.

    The train mean, not the test mean, so the benchmark was actually available
    at the time.
    """
    ss_res = float(np.sum((real - pred) ** 2))
    ss_tot = float(np.sum((real - benchmark) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def evaluate(split: Split, pred: pd.Series,
             train_target_mean: float | None = None,
             train_rv_mean: float | None = None) -> dict[str, float]:
    """Full metric set for one model's *pred* (target-space) against *split*.

    r2_rv is the primary metric, gate must be positive to beat the benchmark,
    and dispersion is descriptive only.
    """
    rv_pred = split.to_rv_fwd(pred)
    frame = pd.concat(
        {"real": split.target, "pred": pred,
         "rv_real": split.rv_fwd, "rv_pred": rv_pred}, axis=1).dropna()
    real, pred_a = frame["real"].to_numpy(), frame["pred"].to_numpy()
    rv_real, rv_pred_a = frame["rv_real"].to_numpy(), frame["rv_pred"].to_numpy()
    r2, a, b = mincer_zarnowitz_r2(real, pred_a)
    pred_std, real_std = float(pred_a.std()), float(real.std())
    return {
        "n": float(len(frame)),
        "mse": mse(real, pred_a),
        "mae": mae(real, pred_a),
        "dir_acc": directional_accuracy(real, pred_a),
        "qlike": qlike(rv_real, rv_pred_a),
        #primary: OOS R2 on forward realised vol vs the train-mean constant
        "r2_rv": (r2_against(rv_real, rv_pred_a, train_rv_mean)
                  if train_rv_mean is not None else float("nan")),
        #must be > 0 to beat the benchmark
        "gate": (r2_against(real, pred_a, train_target_mean)
                 if train_target_mean is not None else float("nan")),
        "pred_std": pred_std,
        "actual_std": real_std,
        "dispersion": pred_std / real_std if real_std > 0 else float("nan"),
        #corr2 of real on pred, affine-invariant
        "mz_r2": r2,
        "mz_a": a,
        "mz_b": b,
    }


def squared_errors(split: Split, pred: pd.Series) -> pd.Series:
    """Per-date squared error in target space, for Diebold-Mariano comparisons."""
    frame = pd.concat({"real": split.target, "pred": pred}, axis=1).dropna()
    return (frame["real"] - frame["pred"]) ** 2


def squared_errors_rv(split: Split, pred: pd.Series) -> pd.Series:
    """Per-date squared error in forward-vol space.

    The headline metric is r2_rv, so a DM run on target-space errors is testing a
    different loss from the one being reported. The target<->rv map is nonlinear
    (vvrp squares it), so the two can rank models differently; both are reported.
    """
    frame = pd.concat({"real": split.rv_fwd, "pred": split.to_rv_fwd(pred)},
                      axis=1).dropna()
    return (frame["real"] - frame["pred"]) ** 2
