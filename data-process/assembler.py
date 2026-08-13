"""Assemble features and targets into aligned, split panels, one per horizon."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from core.base import FeatureBuilder, Horizon
from core.config import ProcessConfig
from core.loaders import RawData
from targets import SHARED_TARGET, TargetBuilder


@dataclass
class HorizonPanel:
    """The assembled, split dataset for one forecast horizon."""

    horizon: Horizon
    full: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    target_columns: list[str]
    paths: dict[str, Path] = field(default_factory=dict)
    #column -> values carried forward that survived into full
    fill_counts: dict[str, int] = field(default_factory=dict)

    @property
    def span(self) -> str:
        """Human-readable date span of the full panel."""
        if self.full.empty:
            return "empty"
        return f"{self.full.index.min():%Y-%m-%d} .. {self.full.index.max():%Y-%m-%d}"


class DatasetAssembler:
    """Joins features and targets into per-horizon train/test panels."""

    WINSOR_SUFFIX = "_winsor"

    def __init__(self, config: ProcessConfig, builders: list[FeatureBuilder]) -> None:
        self.config = config
        self.builders = builders
        self.targets = TargetBuilder(config)

    def feature_frame(self, raw: RawData) -> tuple[pd.DataFrame, list[str]]:
        """Build the joined feature frame and the list of *required* feature columns."""
        parts: list[pd.DataFrame] = []
        required: list[str] = []
        for builder in self.builders:
            part = builder.run(raw)
            parts.append(part)
            if not builder.optional:
                required.extend(part.columns)
        frame = pd.concat(parts, axis=1).sort_index()
        #two builders must not emit the same column
        if frame.columns.duplicated().any():
            dupes = sorted(frame.columns[frame.columns.duplicated()].unique())
            raise ValueError(f"duplicate feature columns from builders: {', '.join(dupes)}")
        return frame, required

    def fill_optional_gaps(self, features: pd.DataFrame,
                           required: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Carry short gaps forward in optional feature columns, within their live range.

        Forward-only and bounded by ``config.max_fill_gap``, so no look-ahead.
        Returns ``(features, filled_mask)``.
        """
        filled_mask = pd.DataFrame(False, index=features.index, columns=features.columns)
        optional = [c for c in features.columns if c not in set(required)]
        if not optional or self.config.max_fill_gap <= 0:
            return features, filled_mask

        features = features.copy()
        for column in optional:
            series = features[column]
            first, last = series.first_valid_index(), series.last_valid_index()
            if first is None:
                continue
            live = series.loc[first:last]
            filled = live.ffill(limit=self.config.max_fill_gap)
            gained = live.isna() & filled.notna()
            if gained.any():
                features.loc[first:last, column] = filled
                filled_mask.loc[first:last, column] = gained
        return features, filled_mask

    def assemble(self, raw: RawData, horizon: Horizon) -> HorizonPanel:
        """Produce the aligned, winsorised, split panel for *horizon*."""
        features, required = self.feature_frame(raw)
        features, filled_mask = self.fill_optional_gaps(features, required)
        target = self.targets.build(raw, horizon)
        target_columns = list(target.columns)

        panel = features.join(target, how="outer").sort_index()
        #gate on required features only, per-leg targets start later
        essential = required + [SHARED_TARGET]
        panel = panel.dropna(subset=essential)

        #trim anything past the test window
        if self.config.test_end_ts is not None:
            panel = panel.loc[panel.index <= self.config.test_end_ts]

        panel, winsor_columns = self._winsorise(panel, target_columns)

        target_set = set(target_columns) | set(winsor_columns)
        train = panel.loc[panel.index < self.config.split_ts]
        test = panel.loc[panel.index >= self.config.split_ts]
        #only the filled cells that survived gating
        surviving = filled_mask.reindex(panel.index).fillna(False)
        fill_counts = {c: int(n) for c, n in surviving.sum().items() if n}
        return HorizonPanel(
            horizon=horizon, full=panel, train=train, test=test,
            feature_columns=[c for c in panel.columns if c not in target_set],
            target_columns=[c for c in panel.columns if c in target_set],
            fill_counts=fill_counts,
        )

    def _winsorise(self, panel: pd.DataFrame,
                   target_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
        """Add a winsorised twin for every ``rvrp_*`` column, clipped at train bounds."""
        lower_q, upper_q = self.config.winsor_quantiles
        panel = panel.copy()
        winsor_columns: list[str] = []
        for column in (c for c in target_columns if c.startswith("rvrp_")):
            train_vals = panel.loc[panel.index < self.config.split_ts, column].dropna()
            if train_vals.empty:
                train_vals = panel[column].dropna()
            if train_vals.empty:
                continue
            lower, upper = train_vals.quantile([lower_q, upper_q])
            winsor_column = f"{column}{self.WINSOR_SUFFIX}"
            panel[winsor_column] = panel[column].clip(lower, upper)
            winsor_columns.append(winsor_column)
        return panel, winsor_columns

    def write(self, panel: HorizonPanel) -> HorizonPanel:
        """Write ``full``/``train``/``test`` CSVs for *panel* and record their paths."""
        out_dir = self.config.horizon_dir(panel.horizon)
        for split_name, frame in (("full", panel.full), ("train", panel.train),
                                  ("test", panel.test)):
            path = out_dir / f"{split_name}.csv"
            frame.to_csv(path, index_label="date")
            panel.paths[split_name] = path
        return panel
