"""Configuration for one experiment run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
import re

#<repo>
_REPO_ROOT = Path(__file__).resolve().parents[3]

#horizon -> realised-vol window in trading days
HORIZON_DAYS: dict[str, int] = {"weekly": 5, "monthly": 21}
IV_LEGS: tuple[str, ...] = ("vix", "vix9d")
FEATURE_SETS: tuple[str, ...] = (
    "options+price", "price-only", "options+price+surface")
ENCODINGS: tuple[str, ...] = (
    "rate", "latency", "population", "delta", "delta_adaptive", "learnt_delta",
    "learnt_conv")
TEST_SAMPLINGS: tuple[str, ...] = ("nonoverlap", "daily")
EVALUATION_SPLITS: tuple[str, ...] = ("val", "test")

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
ISO_ACTIVATIONS: tuple[str, ...] = ("sigmoid", "tanh", "relu", "identity")
CAMPAIGN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")

#default share of the trial the windowed readout scores, measured back from the trial
#end. D85 settled on the final 20%; it is a constant so the sweep and the CLI can name
#the default rather than repeat the number.
READOUT_WINDOW_FRACTION = 0.2

#rv_fwd is the leg-free arm Contribution #3 needs: a price-only model scored on a VRP
#target still meets the IV leg through the target itself and through to_target.
#rvrp_winsor is the variant the proposal specifies, clipped at the train 1st/99th
#percentiles by the assembler; it shares rvrp's target<->rv_fwd map.
TARGETS: tuple[str, ...] = ("vrp", "rvrp", "rvrp_winsor", "vvrp", "rv_fwd")


def _require_choice(name: str, value: object, choices: tuple[str, ...]) -> None:
    """Reject unknown enum-like configuration values before a run starts."""
    if value not in choices:
        raise ValueError(f"unknown {name} {value!r}, expected one of {sorted(choices)}")


def _require_integer(name: str, value: object, *, minimum: int = 1) -> None:
    """Require a non-boolean integer in a bounded configuration field."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        comparator = ">=" if minimum != 1 else "at least"
        if comparator == "at least":
            raise ValueError(f"{name} must be an integer of at least {minimum}, got {value!r}")
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _require_positive_finite(name: str, value: object) -> None:
    """Require a finite positive real number, never a NaN or infinity."""
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")


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
    evaluation_split: str = "test"      #val while selecting; final test after selection
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
    #D85's fix, EventProp only. Off keeps the loss integrated over the whole trial,
    #which is what every existing result was trained with, so the default leaves the
    #generated drive line untouched and reproduces those runs exactly.
    #D96. EventPropCompiler fires its ZeroOutPost update at the beginning of the LAST
    #timestep of each example (event_prop_compiler.py:1503-1509) rather than putting it
    #in a reset group at trial begin, the way InferenceCompiler does
    #(inference_compiler.py:445-450). Synaptic OutPost therefore carries from one trial
    #into the next. Off by default because every result on disk was trained with the
    #carry-over present, so the default reproduces them exactly.
    reset_synapses_between_trials: bool = False
    #LSTM architecture. Module-level constants until 2026-08-25, which meant the
    #control baseline ran at one fixed configuration while the SNN got encoding,
    #window, fraction, drive and reset sweeps. Beating an untuned baseline with a
    #tuned model is a weaker claim than it looks, so these are knobs now.
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.0
    #which training epoch predict() reads back. None means the last one, which is what
    #every run to date used. The SNN checkpoints every 10 epochs and has never once
    #asked whether an earlier one was better; the LSTM lost 0.35 to exactly that (D108).
    #how often training writes a checkpoint. 10 was fine while nobody looked between
    #them; D110 showed the trajectory swings 0.26 inside a single run, which 10-epoch
    #sampling cannot resolve. Lower it to see the shape.
    #hidden layer width. Fixed at 256 for the whole project against a proposal that
    #specified 256 to 1024, so architecture is the one axis never varied (D113). The
    #recurrent matrix is num_hidden squared, so cost does not scale linearly.
    num_hidden: int = 256
    checkpoint_epochs: int = 10
    predict_epoch: int | None = None
    readout_window: bool = False
    #fraction of the trial that is scored, measured back from the trial end. 0.2 is
    #the final 20%, agreed with Fin in D85. Swept from the CLI like w_in_scale.
    readout_window_fraction: float = READOUT_WINDOW_FRACTION
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
    #Non-spiking iso-architecture control. identity is test-only but stays here so the
    #parity test exercises the same validation path as an experiment config.
    iso_activation: str = "sigmoid"
    #Learnt encoder identity. B2 must name the exact frozen B1 artifact.
    encoder_checkpoint: Path | None = None
    learnt_num_neurons: int = 32
    #Optional namespace for a campaign's mutable checkpoints and learnt artifacts.
    #It prevents a deliberate rerun from overwriting a prior campaign with the same
    #scientific settings.
    campaign_id: str | None = None

    annualisation: int = 252
    rvrp_smooth_window: int = 5

    def __post_init__(self) -> None:
        _require_choice("horizon", self.horizon, tuple(HORIZON_DAYS))
        _require_choice("iv_leg", self.iv_leg, IV_LEGS)
        _require_choice("feature_set", self.feature_set, FEATURE_SETS)
        _require_choice("encoding", self.encoding, ENCODINGS)
        _require_choice("test_sampling", self.test_sampling, TEST_SAMPLINGS)
        _require_choice("evaluation_split", self.evaluation_split, EVALUATION_SPLITS)
        _require_choice("target", self.target, TARGETS)
        if self.horizon == "monthly" and self.iv_leg == "vix9d":
            raise ValueError("the monthly panel has no VIX9D leg, "
                             "use --iv-leg vix with --horizon monthly")
        _require_choice("algorithm", self.algorithm, ALGORITHMS)
        _require_choice("iso_activation", self.iso_activation, ISO_ACTIVATIONS)

        #GeNN treats zero as an unseeded run and NumPy rejects negative seeds. Every
        #published run must therefore carry one explicit positive RNG seed.
        _require_integer("seed", self.seed)
        if self.input_window is not None:
            _require_integer("input_window", self.input_window)
        _require_integer("num_epochs", self.num_epochs)
        _require_positive_finite("learning_rate", self.learning_rate)
        _require_positive_finite("delta_multiplier", self.delta_multiplier)
        _require_integer("learnt_num_neurons", self.learnt_num_neurons)
        _require_integer("num_hidden", self.num_hidden)
        _require_integer("checkpoint_epochs", self.checkpoint_epochs)
        _require_integer("lstm_hidden_size", self.lstm_hidden_size)
        _require_integer("lstm_num_layers", self.lstm_num_layers)
        _require_integer("annualisation", self.annualisation)
        _require_integer("rvrp_smooth_window", self.rvrp_smooth_window)
        if self.predict_epoch is not None:
            _require_integer("predict_epoch", self.predict_epoch, minimum=0)
            if self.predict_epoch >= self.num_epochs:
                raise ValueError(f"predict_epoch must be less than num_epochs "
                                 f"({self.num_epochs}), got {self.predict_epoch}")
            saved_epoch = self.predict_epoch == self.num_epochs - 1
            periodic_epoch = ((self.predict_epoch + 1) % self.checkpoint_epochs == 0)
            if not (saved_epoch or periodic_epoch):
                raise ValueError(
                    f"predict_epoch {self.predict_epoch} is not written by "
                    f"checkpoint_epochs={self.checkpoint_epochs}; choose a completed "
                    "checkpoint or the final epoch")
        if self.sample_start is not None:
            if (not isinstance(self.sample_start, str)
                    or re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.sample_start) is None):
                raise ValueError(f"sample_start must be an ISO date (YYYY-MM-DD), got "
                                 f"{self.sample_start!r}")
            try:
                date.fromisoformat(self.sample_start)
            except ValueError as error:
                raise ValueError(f"sample_start must be an ISO date (YYYY-MM-DD), got "
                                 f"{self.sample_start!r}") from error
        if self.campaign_id is not None and not CAMPAIGN_ID_PATTERN.fullmatch(self.campaign_id):
            raise ValueError("campaign_id must contain only letters, digits, dot, dash, or "
                             "underscore and must start with a letter or digit")
        for name in ("w_in_scale", "w_rec_scale", "reg_target_duty_cycle"):
            _require_positive_finite(name, getattr(self, name))
        if self.reg_target_duty_cycle > 1.0:
            raise ValueError("reg_target_duty_cycle must be at most 1, got "
                             f"{self.reg_target_duty_cycle!r}")
        if self.lr_ease_in_batches is not None:
            _require_integer("lr_ease_in_batches", self.lr_ease_in_batches)
        #restoring something that was never excluded is a typo, not a no-op
        unknown = set(self.restore_features) - set(EXCLUDED_FEATURES)
        if unknown:
            raise ValueError(f"restore_features names {sorted(unknown)}, which are not "
                             f"in EXCLUDED_FEATURES {sorted(EXCLUDED_FEATURES)}")
        if self.loss_shaping_tau is not None:
            _require_positive_finite("loss_shaping_tau", self.loss_shaping_tau)
        #raise rather than ignore: eprop has its own gradient path and no drive to shape
        if self.loss_shaping and self.algorithm != "eventprop":
            raise ValueError(f"loss shaping patches the EventProp adjoint drive, "
                             f"it does nothing under algorithm={self.algorithm!r}")
        #a window of 0 scores nothing and a window above 1 reaches before the trial
        #starts; both are numbers that would reach generated CUDA and be wrong there
        #rather than raise, so they are caught on the host
        #torch silently ignores dropout on a single layer, so a nonzero value there is a
        #configuration that does not do what it says. Caught rather than tolerated.
        if (isinstance(self.lstm_dropout, bool) or not isinstance(self.lstm_dropout, Real)
                or not isfinite(self.lstm_dropout) or not 0.0 <= self.lstm_dropout < 1.0):
            raise ValueError(f"lstm_dropout must be in [0, 1), got {self.lstm_dropout}")
        if self.lstm_dropout and self.lstm_num_layers < 2:
            raise ValueError("lstm_dropout has no effect with a single LSTM layer, "
                             "so setting both is a mistake rather than a configuration")
        if (isinstance(self.readout_window_fraction, bool)
                or not isinstance(self.readout_window_fraction, Real)
                or not isfinite(self.readout_window_fraction)
                or not 0.0 < self.readout_window_fraction <= 1.0):
            raise ValueError(f"readout_window_fraction must be in (0, 1], got "
                             f"{self.readout_window_fraction!r}")
        #same argument as loss shaping: eprop's readout is not EventProp's windowed
        #Var, so a window there would be silently ignored rather than applied
        #EPropCompiler registers no ZeroOutPost custom update group, so the callback
        #raises at run time rather than being ignored. All 80 monthly e-prop arms died
        #this way on 2026-08-24 before this check existed.
        if self.reset_synapses_between_trials and self.algorithm != "eventprop":
            raise ValueError(f"the trial reset targets EventProp's ZeroOutPost update, "
                             f"which does not exist under algorithm={self.algorithm!r}")
        if self.readout_window and self.algorithm != "eventprop":
            raise ValueError(f"the readout window is an EventProp readout, "
                             f"it does nothing under algorithm={self.algorithm!r}")
        if self.evaluation_split == "val" and not self.holdout_val:
            raise ValueError("evaluation_split='val' requires holdout_val=True so the "
                             "validation dates were not used for fitting")

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

    def readout_window_times(self, dt: float) -> tuple[float, float] | None:
        """``(window_start, window_end)`` in units of TIME, or None for the whole trial.

        mlGeNN 3.x windows are times, not timesteps (``readouts/readout.py``), so the
        timebase has to come in from the model that owns it rather than be assumed here.

        ``window_end`` is pinned to the trial end and the assertion below says so out
        loud (D85). ``Var.get_readout`` returns the instantaneous membrane voltage at
        the final timestep in *both* library versions, so a window ending earlier would
        train the network on one quantity and then predict with another. The fraction is
        therefore measured back from the end and there is deliberately no knob for the
        far edge; the check exists so that adding one later fails here rather than
        producing a quietly mistrained arm.
        """
        if not self.readout_window:
            return None
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt!r}")
        trial_end = self.sequence_length * dt
        window = ((1.0 - self.readout_window_fraction) * trial_end, trial_end)
        #both halves are reachable rather than restatements of the line above. A
        #fraction small enough to round away leaves window_start == trial_end, which
        #generates a guard that scores nothing and trains on a zero gradient in
        #silence; and pinning window_end is the D85 rule, checked rather than assumed.
        if not 0.0 <= window[0] < window[1] or window[1] != trial_end:
            raise ValueError(
                f"readout window {window} is not a non-empty window ending at the "
                f"trial end {trial_end}. The Var readout reports the final timestep, "
                f"so a window ending earlier trains on a different quantity than it "
                f"predicts, and an empty one trains on nothing.")
        return window

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
