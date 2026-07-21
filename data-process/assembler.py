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

    def assemble(self, raw: RawData, horizon: Horizon) -> HorizonPanel:
        """Produce the aligned, winsorised, split panel for *horizon*."""
        features, required = self.feature_frame(raw)
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
        return HorizonPanel(
            horizon=horizon, full=panel, train=train, test=test,
            feature_columns=[c for c in panel.columns if c not in target_set],
            target_columns=[c for c in panel.columns if c in target_set],
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
