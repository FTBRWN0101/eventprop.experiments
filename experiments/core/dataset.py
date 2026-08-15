"""Load processed VRP panels into model-ready views."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import EXCLUDED_FEATURES, PRICE_ONLY_FEATURES, ExperimentConfig

#targets, never inputs
_TARGET_PREFIXES = ("iv_", "vrp_", "rvrp_", "vvrp_")


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
    target_kind: str         #"vrp" | "rvrp" | "vvrp"
    horizon_days: int

    def to_target(self, rv_fwd_hat: pd.Series) -> pd.Series:
        """Convert a forward-vol forecast into a VRP/rVRP/vVRP forecast (RV-based models)."""
        if self.target_kind == "vvrp":
            return self.iv**2 - rv_fwd_hat**2
        premium = self.iv - rv_fwd_hat
        if self.target_kind == "vrp":
            return premium
        return premium / self.denom

    def to_rv_fwd(self, target_hat: pd.Series) -> pd.Series:
        """Invert a VRP/rVRP/vVRP forecast back to a forward-vol forecast (for QLIKE)."""
        if self.target_kind == "vvrp":
            #a forecast can imply negative variance; clip so the sqrt stays real
            return np.sqrt(np.maximum(self.iv**2 - target_hat, 0.0))
        if self.target_kind == "vrp":
            return self.iv - target_hat
        return self.iv - target_hat * self.denom


class VrpDataset:
    """Reads processed panels and serves daily frames, splits, and windowed tensors."""

    VAL_START = "2017-01-01"

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._cache: dict[str, pd.DataFrame] = {}
        self._transform: list | None = None
        self._transform_key: tuple | None = None
        #column -> fraction of values pinned to a bound
        self.pinned_fraction: dict[str, float] = {}

    def daily(self, split: str) -> pd.DataFrame:
        """The daily frame for *split* (``train``/``val``/``test``/``full``), date-indexed."""
        if split not in self._cache:
            cfg = self.config
            source = "train" if split == "val" else split
            frame = pd.read_csv(cfg.processed_path(source), parse_dates=["date"])
            frame = frame.set_index("date").sort_index()
            if split == "val":
                frame = frame.loc[frame.index >= self.VAL_START]
            elif split == "train" and cfg.holdout_val:
                frame = frame.loc[frame.index < self.VAL_START]
            if cfg.sample_start is not None:
                frame = frame.loc[frame.index >= cfg.sample_start]
            self._cache[split] = frame
        return self._cache[split]

    def full(self) -> pd.DataFrame:
        return self.daily("full")

    def feature_columns(self, frame: pd.DataFrame) -> list[str]:
        """Model-input columns for the configured feature set."""
        columns = [c for c in frame.columns
                   if not _is_target_column(c) and c not in EXCLUDED_FEATURES]
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

    def _fit_transform(self, columns: list[str], space: str) -> list:
        """Fit one transform per column on that column's own daily training rows.

        ``space`` picks the output representation: raw, zscore or unit.
        """
        key = (tuple(columns), space)
        if self._transform_key != key:
            train = self.daily("train")
            fitted = []
            for column in columns:
                values = train[column].dropna().to_numpy()
                if space == "unit":
                    fitted.append(np.sort(values))
                elif space == "zscore":
                    std = values.std()
                    fitted.append((values.mean(), std if std else 1.0))
                else:
                    fitted.append(None)
            self._transform_key, self._transform = key, fitted
        return self._transform

    def _apply_transform(self, values: np.ndarray, columns: list[str],
                         space: str) -> tuple[np.ndarray, dict[str, float]]:
        """Apply the fitted per-column transform; also return the pinning rate per column."""
        fitted = self._fit_transform(columns, space)
        out = np.empty_like(values, dtype=float)
        pinned: dict[str, float] = {}
        for index, column in enumerate(columns):
            column_values = values[:, index]
            if space == "unit":
                sorted_train = fitted[index]
                denominator = max(len(sorted_train) - 1, 1)
                transformed = np.clip(
                    np.searchsorted(sorted_train, column_values, side="left")
                    / denominator, 0.0, 1.0)
                finite = np.isfinite(column_values)
                pinned[column] = float(
                    np.mean((transformed[finite] <= 0.0) | (transformed[finite] >= 1.0))
                ) if finite.any() else float("nan")
            elif space == "zscore":
                mean, std = fitted[index]
                transformed = (column_values - mean) / std
                pinned[column] = 0.0
            else:
                transformed = column_values
                pinned[column] = 0.0
            out[:, index] = transformed
        return out, pinned

    def sequences(self, name: str,
                  space: str = "zscore") -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """Return ``(X[N, T, F], y[N], dates)`` of length-``sequence_length`` windows.

        Windows end on the dates :meth:`split` serves, so every model scores the
        same set. History may reach back across the split boundary, never forward.
        """
        cfg = self.config
        frame = self.daily("full")
        columns = self.feature_columns(frame)

        feats, pinned = self._apply_transform(
            frame[columns].to_numpy(), columns, space)
        target = frame[cfg.target_column].to_numpy()
        dates = frame.index
        window = cfg.sequence_length

        wanted = set(self.split(name).target.index)
        positions = {d: i for i, d in enumerate(dates)}
        ends = sorted(positions[d] for d in wanted if d in positions)

        x_rows, y_rows, end_dates = [], [], []
        for end in ends:
            if end - window + 1 < 0:
                continue
            y = target[end]
            if np.isnan(y) or np.isnan(feats[end - window + 1:end + 1]).any():
                continue
            x_rows.append(feats[end - window + 1:end + 1])
            y_rows.append(y)
            end_dates.append(dates[end])

        X = np.asarray(x_rows)
        #over the windows returned, not the whole frame
        if space == "unit" and X.size:
            flat = X.reshape(-1, X.shape[2])
            self.pinned_fraction = {
                column: float(np.mean((flat[:, i] <= 0.0) | (flat[:, i] >= 1.0)))
                for i, column in enumerate(columns)}
        else:
            self.pinned_fraction = dict.fromkeys(columns, 0.0)
        return X, np.asarray(y_rows), pd.DatetimeIndex(end_dates)
