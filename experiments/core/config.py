"""Configuration for one experiment run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#<repo>
_REPO_ROOT = Path(__file__).resolve().parents[2]

#horizon -> realised-vol window in trading days
HORIZON_DAYS: dict[str, int] = {"weekly": 5, "monthly": 21}

#no options inputs, for the ablation. NOTE: spx_logret is also in EXCLUDED_FEATURES and
#exclusion wins, so this arm actually serves 3 columns, not 4. Listed anyway so clearing
#EXCLUDED_FEATURES restores the original arm.
PRICE_ONLY_FEATURES: tuple[str, ...] = ("rv_1", "rv_5", "rv_21", "spx_logret")

#own arm, not folded into options+price: the surface has gaps and would reshape the sample
SURFACE_FEATURES: tuple[str, ...] = ("om_atm30", "om_skew30", "om_term")

#dropped: too noisy, or train/test distributions drift
EXCLUDED_FEATURES: tuple[str, ...] = ("spx_logret", "vix_logret", "skew", "ts_slope")

#not the same network either way, see models/snn.py
ALGORITHMS: tuple[str, ...] = ("eventprop", "eprop")

#rv_fwd is the leg-free arm Contribution #3 needs: a price-only model scored on a VRP
#target still meets the IV leg through the target itself and through to_target.
#rvrp_winsor is the variant the proposal specifies, clipped at the train 1st/99th
#percentiles by the assembler; it shares rvrp's target<->rv_fwd map.
TARGETS: tuple[str, ...] = ("vrp", "rvrp", "rvrp_winsor", "vvrp", "rv_fwd")


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable knobs for one train/evaluate run."""

    repo_root: Path = _REPO_ROOT
    horizon: str = "weekly"            #weekly | monthly
    target: str = "vrp"               #vrp | rvrp | vvrp (variance space) | rv_fwd
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
    #Contribution #6, EventProp only. Off keeps the uniform per-timestep loss.
    loss_shaping: bool = False
    #exp(-t/tau) decay in timesteps; None means tau = sequence_length, i.e. exp(-t/T)
    loss_shaping_tau: float | None = None
    #initial weight scales, divided by sqrt(fan-in). Their RATIO sets whether hidden
    #listens to the input or to its own recurrence -- see D64.
    w_in_scale: float = 2.5
    w_rec_scale: float = 1.5
    #regularisation spike-rate target, as a duty cycle. Third knob of the proposal's
    #targeted search, with delta_multiplier and learning_rate. Feeds EventProp's
    #reg_nu_upper and eprop's f_target, which are different units of the same intent.
    reg_target_duty_cycle: float = 0.3
    #geometric per-batch learning-rate ease-in, as in Nowotny et al. None disables it.
    lr_ease_in_batches: int | None = None
    #the proposal pauses training when >10% of hidden neurons go silent. Off lets a
    #sweep run through the breach, with the raster still written for review.
    silent_neuron_abort: bool = True
    #per-signal ablation. drop_features removes served columns; restore_features
    #re-admits ones EXCLUDED_FEATURES drops, which is how skew and ts_slope get
    #measured without putting them back in every headline run. Tuples, so the
    #config stays hashable and frozen.
    drop_features: tuple[str, ...] = ()
    restore_features: tuple[str, ...] = ()

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
        if self.target not in TARGETS:
            raise ValueError(f"unknown target {self.target!r}, "
                             f"expected one of {sorted(TARGETS)}")
        for name in ("w_in_scale", "w_rec_scale", "reg_target_duty_cycle"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        if self.lr_ease_in_batches is not None and self.lr_ease_in_batches < 1:
            raise ValueError(f"lr_ease_in_batches must be at least 1, "
                             f"got {self.lr_ease_in_batches!r}")
        #restoring something that was never excluded is a typo, not a no-op
        unknown = set(self.restore_features) - set(EXCLUDED_FEATURES)
        if unknown:
            raise ValueError(f"restore_features names {sorted(unknown)}, which are not "
                             f"in EXCLUDED_FEATURES {sorted(EXCLUDED_FEATURES)}")
        if self.loss_shaping_tau is not None and not self.loss_shaping_tau > 0:
            raise ValueError(f"loss_shaping_tau must be positive, "
                             f"got {self.loss_shaping_tau!r}")
        #raise rather than ignore: eprop has its own gradient path and no drive to shape
        if self.loss_shaping and self.algorithm != "eventprop":
            raise ValueError(f"loss shaping patches the EventProp adjoint drive, "
                             f"it does nothing under algorithm={self.algorithm!r}")

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
        """Realised target column to evaluate against, e.g. ``vrp_vix9d``.

        ``rv_fwd`` is leg-independent, so it is the one target that does not take the
        leg suffix. It is still *sampled* on the leg (see dataset.split), which keeps
        it comparable with the VRP arms rather than silently gaining 1990-2010.

        ``rvrp_winsor`` carries its suffix after the leg, matching the twin column
        ``assembler._winsorise`` writes.
        """
        if self.target == "rv_fwd":
            return "rv_fwd"
        if self.target == "rvrp_winsor":
            return f"rvrp_{self.iv_leg}_winsor"
        return f"{self.target}_{self.iv_leg}"

    @property
    def iv_column(self) -> str:
        """Implied-vol leg level column, e.g. ``iv_vix9d``."""
        return f"iv_{self.iv_leg}"

    def processed_path(self, split: str) -> Path:
        """Path to ``data-save/processed/<horizon>/<split>.csv``."""
        return self.repo_root / "data-save" / "processed" / self.horizon / f"{split}.csv"
