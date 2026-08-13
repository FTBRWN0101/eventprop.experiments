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
    """Fraction of forecasts whose sign matches the realised target's sign."""
    mask = (real != 0) & (pred != 0)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(real[mask]) == np.sign(pred[mask])))


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


def diebold_mariano(err_a: pd.Series, err_b: pd.Series,
                    lag: int = 0) -> tuple[float, float]:
    """Diebold-Mariano test on two aligned per-date loss series.

    Returns ``(stat, p_value)``; ``stat > 0`` means model A has the larger loss.
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
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(stat)))
    return float(stat), float(p_value)


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
