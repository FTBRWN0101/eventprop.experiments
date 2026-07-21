"""Configuration for one experiment run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#<repo>
_REPO_ROOT = Path(__file__).resolve().parents[2]

#horizon -> realised-vol window in trading days
HORIZON_DAYS: dict[str, int] = {"weekly": 5, "monthly": 21}

#no options inputs, for the ablation
PRICE_ONLY_FEATURES: tuple[str, ...] = ("rv_1", "rv_5", "rv_21", "spx_logret")


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable knobs for one train/evaluate run."""

    repo_root: Path = _REPO_ROOT
    horizon: str = "weekly"            #weekly | monthly
    target: str = "vrp"               #vrp | rvrp
    iv_leg: str = "vix9d"             #vix | vix9d (vix9d is weekly only, 2011+)
    feature_set: str = "options+price"  #options+price | price-only
    model: str = "har_rv"
    encoding: str = "rate"            #used by the SNN only
    test_sampling: str = "nonoverlap"  #nonoverlap | daily
    seed: int = 0

    annualisation: int = 252
    rvrp_smooth_window: int = 5

    @classmethod
    def load(cls, **overrides: object) -> "ExperimentConfig":
        return cls(**overrides)  # type: ignore[arg-type]

    @property
    def horizon_days(self) -> int:
        return HORIZON_DAYS[self.horizon]

    @property
    def target_column(self) -> str:
        """Realised target column to evaluate against, e.g. ``vrp_vix9d``."""
        return f"{self.target}_{self.iv_leg}"

    @property
    def iv_column(self) -> str:
        """Implied-vol leg level column, e.g. ``iv_vix9d``."""
        return f"iv_{self.iv_leg}"

    def processed_path(self, split: str) -> Path:
        """Path to ``data-save/processed/<horizon>/<split>.csv``."""
        return self.repo_root / "data-save" / "processed" / self.horizon / f"{split}.csv"
