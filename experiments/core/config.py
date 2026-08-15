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

#own arm, not folded into options+price: the surface has gaps and would reshape the sample
SURFACE_FEATURES: tuple[str, ...] = ("om_atm30", "om_skew30", "om_term")

#dropped: too noisy, or train/test distributions drift
EXCLUDED_FEATURES: tuple[str, ...] = ("spx_logret", "vix_logret", "skew", "ts_slope")

#not the same network either way, see models/snn.py
ALGORITHMS: tuple[str, ...] = ("eventprop", "eprop")


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable knobs for one train/evaluate run."""

    repo_root: Path = _REPO_ROOT
    horizon: str = "weekly"            #weekly | monthly
    target: str = "vrp"               #vrp | rvrp | vvrp (variance space)
    iv_leg: str = "vix9d"             #vix | vix9d (vix9d is weekly only, 2011+)
    feature_set: str = "options+price"  #options+price | price-only | options+price+surface
    model: str = "har_rv"
    encoding: str = "rate"            #used by the SNN only
    algorithm: str = "eventprop"      #eventprop | eprop; SNN only
    test_sampling: str = "nonoverlap"  #nonoverlap | daily
    seed: int = 1                     #0 means unseeded to GeNN
    input_window: int | None = None   #sequence length T; None ties it to the horizon
    num_epochs: int = 50
    learning_rate: float = 0.001
    sample_start: str | None = None   #ISO start date, same for every split and model
    delta_multiplier: float = 1.0     #delta encoder threshold multiplier
    holdout_val: bool = False         #carve 2017-2019 out of train as a validation set

    annualisation: int = 252
    rvrp_smooth_window: int = 5

    def __post_init__(self) -> None:
        if self.horizon == "monthly" and self.iv_leg == "vix9d":
            raise ValueError("the monthly panel has no VIX9D leg, "
                             "use --iv-leg vix with --horizon monthly")
        #GeNN reads 0 as unseeded
        if self.seed == 0:
            raise ValueError("seed 0 means 'unseeded' to GeNN, use any other value")
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"unknown algorithm {self.algorithm!r}, "
                             f"expected one of {sorted(ALGORITHMS)}")

    @classmethod
    def load(cls, **overrides: object) -> "ExperimentConfig":
        return cls(**overrides)  #type: ignore[arg-type]

    @property
    def horizon_days(self) -> int:
        return HORIZON_DAYS[self.horizon]

    @property
    def sequence_length(self) -> int:
        """Input window length T, independent of the forecast horizon."""
        return self.input_window if self.input_window is not None else self.horizon_days

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
