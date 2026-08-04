"""Load processed VRP panels into model-ready views."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import PRICE_ONLY_FEATURES, ExperimentConfig

#targets, never inputs
_TARGET_PREFIXES = ("iv_", "vrp_", "rvrp_")


def _is_target_column(column: str) -> bool:
    return column == "rv_fwd" or column.startswith(_TARGET_PREFIXES)


@dataclass
class Split:
    """One evaluation split with everything needed to score any model."""

    name: str
    features: pd.DataFrame   #daily, date-indexed feature columns
    target: pd.Series        #realised VRP/rVRP to score against
    rv_fwd: pd.Series        #realised forward vol (for QLIKE)
    iv: pd.Series            #implied-vol leg level at t
    denom: pd.Series         #smoothed IV leg (rVRP denominator)
    target_kind: str         #"vrp" | "rvrp"
    horizon_days: int

    def to_target(self, rv_fwd_hat: pd.Series) -> pd.Series:
        """Convert a forward-vol forecast into a VRP/rVRP forecast (RV-based models)."""
        premium = self.iv - rv_fwd_hat
        if self.target_kind == "vrp":
            return premium
        return premium / self.denom

    def to_rv_fwd(self, target_hat: pd.Series) -> pd.Series:
        """Invert a VRP/rVRP forecast back to a forward-vol forecast (for QLIKE)."""
        if self.target_kind == "vrp":
            return self.iv - target_hat
        return self.iv - target_hat * self.denom


class VrpDataset:
    """Reads processed panels and serves daily frames, splits, and windowed tensors."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._cache: dict[str, pd.DataFrame] = {}
        self._scaler: tuple[np.ndarray, np.ndarray] | None = None

    def daily(self, split: str) -> pd.DataFrame:
        """The full daily frame for *split* (``train``/``test``/``full``), date-indexed."""
        if split not in self._cache:
            frame = pd.read_csv(self.config.processed_path(split), parse_dates=["date"])
            self._cache[split] = frame.set_index("date").sort_index()
        return self._cache[split]

    def full(self) -> pd.DataFrame:
        return self.daily("full")

    def feature_columns(self, frame: pd.DataFrame) -> list[str]:
        """Model-input columns for the configured feature set."""
        columns = [c for c in frame.columns if not _is_target_column(c)]
        if self.config.feature_set == "price-only":
            return [c for c in columns if c in PRICE_ONLY_FEATURES]
        return columns

    def split(self, name: str) -> Split:
        """Build the aligned :class:`Split` for *name*, sampled per the config."""
        cfg = self.config
        frame = self.daily(name)
        iv = frame[cfg.iv_column]
        #min_periods=1: tolerate isolated leg gaps
        denom = iv.rolling(cfg.rvrp_smooth_window, min_periods=1).mean()
        needed = self.feature_columns(frame)

        view = pd.concat(
            [frame[needed], frame[cfg.target_column].rename("__target__"),
             frame["rv_fwd"].rename("__rv__"), iv.rename("__iv__"),
             denom.rename("__denom__")], axis=1).dropna()

        if name == "test" and cfg.test_sampling == "nonoverlap":
            view = view.iloc[::cfg.horizon_days]

        return Split(
            name=name, features=view[needed], target=view["__target__"],
            rv_fwd=view["__rv__"], iv=view["__iv__"], denom=view["__denom__"],
            target_kind=cfg.target, horizon_days=cfg.horizon_days)

    def _fit_scaler(self, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if self._scaler is None:
            train = self.daily("train")[columns].dropna()
            mean = train.to_numpy().mean(axis=0)
            std = train.to_numpy().std(axis=0)
            std[std == 0] = 1.0
            self._scaler = (mean, std)
        return self._scaler

    def sequences(self, name: str) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """Return ``(X[N, T, F], y[N], dates)`` of length-``horizon_days`` windows."""
        cfg = self.config
        frame = self.daily(name)
        columns = self.feature_columns(frame)
        mean, std = self._fit_scaler(columns)

        feats = (frame[columns].to_numpy() - mean) / std
        target = frame[cfg.target_column].to_numpy()
        dates = frame.index
        window = cfg.horizon_days

        if name == "test" and cfg.test_sampling == "nonoverlap":
            #match split()'s dates: it strides after dropping NaN rows
            wanted = set(self.split(name).target.index)
            ends = [i for i in range(window - 1, len(frame)) if dates[i] in wanted]
        else:
            ends = range(window - 1, len(frame))

        x_rows, y_rows, end_dates = [], [], []
        for end in ends:
            y = target[end]
            if np.isnan(y) or np.isnan(feats[end - window + 1:end + 1]).any():
                continue
            x_rows.append(feats[end - window + 1:end + 1])
            y_rows.append(y)
            end_dates.append(dates[end])
        return (np.asarray(x_rows), np.asarray(y_rows),
                pd.DatetimeIndex(end_dates))
