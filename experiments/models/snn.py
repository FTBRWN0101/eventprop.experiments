"""SNN forecaster on mlGeNN. EventProp or eprop, set by config.algorithm.

One timestep is one trading day. Needs the mlgenn env, imported lazily.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from core.dataset import VrpDataset
from encoders.base import ENCODERS, Encoder, EncodedInput
from models.base import Forecaster

NUM_HIDDEN = 256
BATCH_SIZE = 32
DT = 1.0
TAU_MEM = 3.0   #heterogeneous sampling scales from this
TAU_SYN = 1.0   #homogeneous: _get_tau_syn wants one scalar
CHECKPOINT_ROOT = Path(__file__).resolve().parents[1] / ".snn_checkpoints"
_ENCODERS_DIR = Path(__file__).resolve().parents[1] / "encoders"

CHECKPOINT_EPOCHS = 10                #spike counts are checked between chunks
SILENT_NEURON_WARN_THRESHOLD = 0.10   #raster review trigger
RASTER_EXAMPLES = 4                   #windows whose full spike train is kept per chunk

#from the mlGeNN eventprop example
V_THRESH = 0.61
#defaults for ExperimentConfig.w_in_scale / w_rec_scale, which override these
W_IN_SCALE = 2.5
W_REC_SCALE = 1.5
W_OUT_SCALE = 2.0
TAU_MEM_OUT = 20.0   #long output integrator
REG_LAMBDA = 1e-8
REG_TARGET_DUTY_CYCLE = 0.3   #scales reg_nu_upper to the trial length

SILENT_NEURON_DELTA_G = 0.002

#eprop needs one, 1 step = 1 day
TAU_REFRAC_EPROP = 1.0
#not comparable with EventProp's REG_LAMBDA
C_REG_EPROP = 3.0

logger = logging.getLogger(__name__)


def _sample_heterogeneous_tau_mem(num_neurons: int, seed: int) -> np.ndarray:
    """Per-neuron tau_mem from a gamma, clipped to [TAU_MEM, 3*TAU_MEM]."""
    rng = np.random.default_rng(seed)
    return np.clip(rng.gamma(3.0, TAU_MEM / 3.0, num_neurons), TAU_MEM, 3.0 * TAU_MEM)


def _make_silent_neuron_recovery_cls():
    """Build the SilentNeuronRecovery callback. Defined inside a function so
    importing this module does not need the mlgenn env.
    """
    from ml_genn.callbacks import Callback
    from ml_genn.utils.network import get_underlying_conn, get_underlying_pop

    class SilentNeuronRecovery(Callback):
        def __init__(self, hidden_pop, delta_g: float = SILENT_NEURON_DELTA_G):
            self._pop = get_underlying_pop(hidden_pop)
            self._delta_g = delta_g
            self._pull = False

        def set_params(self, data, compiled_network, **kwargs):
            self._compiled_network = compiled_network
            if compiled_network.num_recording_timesteps is None:
                raise RuntimeError("SilentNeuronRecovery requires "
                                   "num_recording_timesteps to be set")
            genn_pop = compiled_network.neuron_populations[self._pop]
            if not genn_pop.spike_recording_enabled:
                raise RuntimeError("SilentNeuronRecovery requires record_spikes=True "
                                   "on the hidden population")
            self._num_neurons = genn_pop.num_neurons
            self._epoch_spike_totals = np.zeros(self._num_neurons, dtype=np.int64)

            #weakrefs, so dereference explicitly
            self._incoming = []
            for conn_ref in self._pop.incoming_connections:
                conn = conn_ref()
                n_pre = conn.source().shape[0]
                genn_conn_pop = compiled_network.connection_populations[
                    get_underlying_conn(conn)]
                self._incoming.append((genn_conn_pop, n_pre))

        def set_first(self):
            #do not rely on callback order
            self._pull = True

        def on_epoch_begin(self, epoch):
            self._epoch_spike_totals[:] = 0

        def on_timestep_end(self, timestep):
            cn = self._compiled_network
            timestep = cn.genn_model.timestep
            if (timestep % cn.num_recording_timesteps) == 0:
                if self._pull:
                    cn.genn_model.pull_recording_buffers_from_device()
                genn_pop = cn.neuron_populations[self._pop]
                for _, neuron_ids in genn_pop.spike_recording_data:
                    self._epoch_spike_totals += np.bincount(
                        neuron_ids, minlength=self._num_neurons)

        def on_epoch_end(self, epoch, metrics):
            silent = np.flatnonzero(self._epoch_spike_totals == 0)
            if silent.size == 0:
                return
            logger.debug("[snn] silent-neuron recovery: bumping %d/%d hidden "
                        "neurons (epoch %d)", silent.size, self._num_neurons, epoch)
            for genn_conn_pop, n_pre in self._incoming:
                g_var = genn_conn_pop.vars["g"]
                g_var.pull_from_device()
                weights = g_var.view.reshape(n_pre, self._num_neurons)
                weights[:, silent] += self._delta_g
                g_var.push_to_device()

    return SilentNeuronRecovery


def _make_ease_in_schedule_cls():
    """Build the EaseInSchedule callback, following Nowotny et al. (2025).

    Defined inside a function so importing this module does not need the mlgenn env.
    """
    from ml_genn.callbacks import Callback

    class EaseInSchedule(Callback):
        """Ramp alpha geometrically from target/1000 to target over *num_batches*.

        The batch counter lives in a caller-owned dict on purpose. mlGeNN shallow
        copies every callback per ``train()`` call (``utils/module.py:50``), and
        ``fit`` calls ``train()`` once per checkpoint chunk, so an ``int`` attribute
        would reset at each chunk boundary. Worse, mlGeNN's own ``batch`` argument
        restarts at 0 every epoch (``compiled_training_network.py:184``), so the
        reference implementation in the mlGeNN SHD example only completes because
        that dataset has more batches per epoch than the ramp needs. This one does
        not, so the ramp is driven off a counter that never resets.
        """

        def __init__(self, target_alpha: float, num_batches: int, state: dict):
            self._target = target_alpha
            #span the ramp inclusively: batch 0 sits at target/1000 and batch
            #num_batches-1 at target exactly, so num_batches counts the ramp itself
            self._rate = 1000.0 ** (1.0 / (num_batches - 1)) if num_batches > 1 else 0.0
            self._state = state

        def set_params(self, compiled_network, **kwargs):
            self._optimisers = [o for o, _ in compiled_network.optimisers]

        def on_batch_begin(self, batch):
            step = self._state["batches"]
            self._state["batches"] = step + 1
            #num_batches == 1 means no ramp at all, so go straight to target
            alpha = (self._target if self._rate == 0.0 else
                     min(self._target, (self._target / 1000.0) * (self._rate ** step)))
            for optimiser in self._optimisers:
                optimiser.alpha = alpha

    return EaseInSchedule


def _silent_neuron_fraction(spike_counts) -> float:
    """Fraction of hidden neurons with zero recorded spikes. Pure numpy, so it
    runs without the mlgenn env.
    """
    counts = np.asarray(spike_counts)
    if counts.size == 0:
        return float("nan")
    total_per_neuron = counts.sum(axis=0)
    return float(np.mean(total_per_neuron == 0))


class SnnForecaster(Forecaster):
    """Recurrent LIF network, pluggable spike encoding, EventProp/MSE training."""

    name = "snn"
    description = "mlGeNN recurrent LIF + EventProp, pluggable encoding (config.encoding)"

    def _get_encoder(self) -> Encoder:
        ENCODERS.discover(_ENCODERS_DIR, "encoders")
        cls = ENCODERS.get(self.config.encoding)
        if self.config.encoding.startswith("delta"):
            return cls(multiplier=self.config.delta_multiplier)
        return cls()

    def _build_network(self, encoded: EncodedInput, timesteps: int, max_spikes: int | None):
        from ml_genn import Connection, Network, Population
        from ml_genn.connectivity import Dense
        from ml_genn.initializers import Normal
        from ml_genn.neurons import (LeakyIntegrate, LeakyIntegrateFire,
                                     PoissonInput, SpikeInput)
        from ml_genn.synapses import Delta, Exponential

        eprop = self.config.algorithm == "eprop"
        if eprop:
            #hard: eprop_compiler.py:474 raises on anything but Delta
            from ml_genn.compilers.eprop_compiler import default_params
            synapse = lambda: Delta()
            #hard: eprop_compiler.py:103-108 raises on differing time constants
            tau_mem = TAU_MEM
            #NOT hard. eprop_compiler.py:452-459 only logs a warning that eprop "works
            #best with" a refractory period and a relative reset. Both are followed
            #because the library recommends them, not because it enforces them -- so
            #the EventProp arm can be given the same settings as a matched control.
            tau_refrac = TAU_REFRAC_EPROP
        else:
            from ml_genn.compilers.event_prop_compiler import default_params
            synapse = lambda: Exponential(TAU_SYN)
            tau_mem = _sample_heterogeneous_tau_mem(NUM_HIDDEN, self.config.seed)
            tau_refrac = None

        num_neurons = encoded.num_neurons
        network = Network(default_params)
        with network:
            if encoded.kind == "rate":
                input_neuron = PoissonInput(signed_spikes=False, input_frames=timesteps,
                                            input_frame_timesteps=1)
            else:
                input_neuron = SpikeInput(max_spikes=max_spikes)
            #name them, or a second network gets different checkpoint names
            input_pop = Population(input_neuron, num_neurons, name="input")
            hidden = Population(
                LeakyIntegrateFire(v_thresh=V_THRESH, tau_mem=tau_mem,
                                   tau_refrac=tau_refrac),
                NUM_HIDDEN, record_spikes=True, name="hidden")
            output = Population(
                LeakyIntegrate(tau_mem=TAU_MEM_OUT, readout="var"), 1, name="output")

            #swept from the CLI: the input population is small and sparse while the
            #recurrent one is 256-wide, so the ratio of these two decides whether the
            #hidden layer listens to the input or to itself (D64)
            Connection(input_pop, hidden,
                      Dense(Normal(mean=0.0,
                                   sd=self.config.w_in_scale / np.sqrt(num_neurons))),
                      synapse())
            Connection(hidden, hidden,
                      Dense(Normal(mean=0.0,
                                   sd=self.config.w_rec_scale / np.sqrt(NUM_HIDDEN))),
                      synapse())
            Connection(hidden, output,
                      Dense(Normal(mean=0.0, sd=W_OUT_SCALE / np.sqrt(NUM_HIDDEN))),
                      synapse())
        return network, input_pop, hidden, output

    def _max_spikes(self, encoded: EncodedInput) -> int | None:
        if encoded.kind != "spikes":
            return None
        from ml_genn.utils.data import calc_max_spikes
        return BATCH_SIZE * calc_max_spikes(encoded.data)

    def _loss_shaping_tau(self) -> float:
        """Decay constant in timesteps. None means the trial length, i.e. exp(-t/T)."""
        tau = self.config.loss_shaping_tau
        return float(self._timesteps if tau is None else tau)

    def _checkpoint_dir(self, create: bool = False) -> Path:
        #L and seed are part of the identity: without them a seed sweep writes every
        #seed to one directory and an L sweep collides on top of it
        cfg = self.config
        #same argument for loss shaping (D42); suffix only when on, so existing
        #unshaped checkpoint paths are unchanged
        shaping = ""
        if cfg.loss_shaping:
            tau = cfg.sequence_length if cfg.loss_shaping_tau is None else cfg.loss_shaping_tau
            shaping = f"_ls{tau:g}"
        #same again for the weight scales, or a drive sweep overwrites itself
        scales = ""
        if (cfg.w_in_scale, cfg.w_rec_scale) != (W_IN_SCALE, W_REC_SCALE):
            scales = f"_wi{cfg.w_in_scale:g}_wr{cfg.w_rec_scale:g}"
        #and again for the two training knobs, or a targeted search overwrites itself
        regularisation = ""
        if cfg.reg_target_duty_cycle != REG_TARGET_DUTY_CYCLE:
            regularisation = f"_nu{cfg.reg_target_duty_cycle:g}"
        ease_in = "" if cfg.lr_ease_in_batches is None else f"_ei{cfg.lr_ease_in_batches}"
        path = (CHECKPOINT_ROOT /
                f"{cfg.horizon}_{cfg.target}_{cfg.iv_leg}_{cfg.encoding}_{cfg.algorithm}"
                f"_L{cfg.sequence_length}_s{cfg.seed}{shaping}{scales}"
                f"{regularisation}{ease_in}")
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _raster_dir(self, create: bool = False) -> Path:
        """Directory holding the per-checkpoint spike rasters."""
        path = self._checkpoint_dir() / "rasters"
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _save_raster(self, raster, epoch: int) -> Path | None:
        """Persist one chunk's sampled spike raster as ``rasters/epoch<N>.npz``.

        ``raster`` is mlGeNN's ``(times, ids)`` pair of per-example lists. Examples
        are drawn from the shuffled order, so a given slot is a different window each
        epoch. That is fine for the pathology this watches for, which is population
        wide, and it means the sample is not always the same four windows.
        """
        times, ids = raster
        if not times:
            return None
        path = self._raster_dir(create=True) / f"epoch{epoch}.npz"
        flat_times = np.concatenate([np.asarray(t) for t in times]) if times else np.empty(0)
        flat_ids = np.concatenate([np.asarray(i) for i in ids]) if ids else np.empty(0)
        example = np.concatenate(
            [np.full(len(t), k, dtype=np.int32) for k, t in enumerate(times)])
        np.savez_compressed(path, times=flat_times, ids=flat_ids, example=example,
                            num_neurons=NUM_HIDDEN, timesteps=self._timesteps)
        logger.debug("[snn] raster for %d examples written to %s", len(times), path)
        return path

    def _check_silent_fraction(self, fraction: float, epoch: int, total: int) -> None:
        """Stop training when too many hidden neurons have gone silent.

        The proposal pauses at 10% and reviews the recovery mechanism before
        continuing, so the default is to raise rather than log and carry on. Both the
        raster and the weights for the breaching checkpoint are already on disk when
        this fires, so the run is diagnosable rather than lost.
        """
        if not fraction > SILENT_NEURON_WARN_THRESHOLD:
            return
        message = (f"[snn] spike deletion: {fraction * 100:.1f}% of hidden neurons "
                   f"silent at epoch {epoch}/{total} "
                   f"(>{SILENT_NEURON_WARN_THRESHOLD * 100:.0f}% threshold). "
                   f"Weights and rasters for review are in "
                   f"{self._checkpoint_dir()}.")
        if self.config.silent_neuron_abort:
            raise RuntimeError(
                message + " Training stopped, as the proposal requires. Review loss "
                "shaping and silent-neuron recovery, then rerun with "
                "--allow-silent-neurons to continue past it.")
        logger.warning("%s Continuing: --allow-silent-neurons is set.", message)

    def _expected_checkpoints(self) -> list[Path]:
        """Weight files save() writes, named ``<epoch>-Conn_<src>_<dst>-g.npy``."""
        epoch = self.config.num_epochs - 1
        directory = self._checkpoint_dir()
        expected = [directory / f"{epoch}-Conn_{src}_{dst}-g.npy"
                    for src, dst in (("input", "hidden"), ("hidden", "hidden"),
                                     ("hidden", "output"))]
        #eprop saves an output bias too
        if self.config.algorithm == "eprop":
            expected.append(directory / f"{epoch}-output-Bias.npy")
        return expected

    def fit(self, data: VrpDataset) -> None:
        from ml_genn.callbacks import BatchProgressBar, SpikeRecorder
        from ml_genn.compilers import EventPropCompiler
        from ml_genn.optimisers import Adam
        from ml_genn.serialisers import Numpy

        SilentNeuronRecovery = _make_silent_neuron_recovery_cls()

        #mlGeNN shuffles with numpy's global RNG, so seed that too
        np.random.seed(self.config.seed)

        #the encoder picks its own representation
        self._encoder = self._get_encoder()
        X, y, train_dates = data.sequences("train", space=self._encoder.input_space)
        scale = data.window_scale if self._encoder.wants_scale else None
        #record what fell outside the training support
        self.pinned_fraction = dict(data.pinned_fraction)
        #mlGeNN wants full batches; drop the trailing partial one
        n_complete = (X.shape[0] // BATCH_SIZE) * BATCH_SIZE
        X, y = X[:n_complete], y[:n_complete]
        if scale is not None:
            scale = scale[:n_complete]
        self.fitted_range = (str(train_dates[0].date()),
                             str(train_dates[n_complete - 1].date()))
        num_epochs = self.config.num_epochs
        self._timesteps = X.shape[1]
        self._y_mean, self._y_std = float(y.mean()), float(y.std() or 1.0)
        y_norm = (y - self._y_mean) / self._y_std

        self._encoder.fit(X, scale)
        encoded = self._encoder.encode(X, scale)
        max_spikes = self._max_spikes(encoded)

        #per-timestep loss needs a target at every timestep
        y_full = np.repeat(y_norm[:, None, None], self._timesteps, axis=1)

        network, input_pop, hidden_pop, output_pop = self._build_network(
            encoded, self._timesteps, max_spikes)
        if self.config.algorithm == "eprop":
            from ml_genn.compilers import EPropCompiler

            #f_target is a duty cycle in Hz clothing, so convert
            f_target = self.config.reg_target_duty_cycle * 1000.0 / DT
            #the 500.0 default never settles inside one trial
            compiler = EPropCompiler(example_timesteps=self._timesteps,
                                     losses="mean_square_error",
                                     optimiser=Adam(self.config.learning_rate),
                                     batch_size=BATCH_SIZE, dt=DT,
                                     c_reg=C_REG_EPROP, f_target=f_target,
                                     tau_reg=float(self._timesteps),
                                     rng_seed=self.config.seed)
        else:
            reg_nu_upper = max(1.0, self.config.reg_target_duty_cycle * self._timesteps)
            kwargs = dict(example_timesteps=self._timesteps,
                          losses="mean_square_error",
                          optimiser=Adam(self.config.learning_rate),
                          batch_size=BATCH_SIZE, dt=DT, per_timestep_loss=True,
                          reg_lambda_upper={hidden_pop: REG_LAMBDA},
                          reg_lambda_lower={hidden_pop: REG_LAMBDA},
                          reg_nu_upper=reg_nu_upper,
                          rng_seed=self.config.seed)
            if self.config.loss_shaping:
                from compilers.loss_shaping import make_loss_shaped_compiler_cls
                compiler = make_loss_shaped_compiler_cls()(
                    **kwargs, loss_shaping_tau=self._loss_shaping_tau())
            else:
                compiler = EventPropCompiler(**kwargs)
        compiled_net = compiler.compile(network)
        compiled_net.num_recording_timesteps = self._timesteps

        #one shared counter for the whole run: see EaseInSchedule on why it is a dict
        ease_in_state = {"batches": 0}
        ease_in_batches = self.config.lr_ease_in_batches
        if ease_in_batches is not None:
            EaseInSchedule = _make_ease_in_schedule_cls()

        self.silent_neuron_fraction: list[float] = []
        with compiled_net:
            epochs_done = 0
            while epochs_done < num_epochs:
                chunk = min(CHECKPOINT_EPOCHS, num_epochs - epochs_done)
                recorder = SpikeRecorder(hidden_pop, key="hidden_spikes",
                                         record_counts=True)
                #a raster for a handful of examples: full spike times for every
                #window would be gigabytes, and the diagnostic only needs a sample
                raster = SpikeRecorder(hidden_pop, key="hidden_raster",
                                       example_filter=list(range(RASTER_EXAMPLES)),
                                       record_counts=False)
                callbacks = [BatchProgressBar(), recorder, raster,
                             SilentNeuronRecovery(hidden_pop)]
                if ease_in_batches is not None:
                    callbacks.append(EaseInSchedule(
                        self.config.learning_rate, ease_in_batches, ease_in_state))
                #callbacks get copied, so read the data from train()'s return
                _, cb_data = compiled_net.train(
                    {input_pop: encoded.data}, {output_pop: y_full},
                    num_epochs=chunk, start_epoch=epochs_done, shuffle=True,
                    callbacks=callbacks)
                epochs_done += chunk

                spike_counts = cb_data["hidden_spikes"]
                silent_frac = _silent_neuron_fraction(spike_counts)
                self.silent_neuron_fraction.append(silent_frac)
                self._save_raster(cb_data["hidden_raster"], epochs_done)
                #checkpoint BEFORE the silent-neuron check, never after. The check
                #raises, and a save placed after it would discard every epoch of GPU
                #work in exactly the case the diagnostic is meant to be inspected.
                #save(), not save_connectivity(): that one does nothing for Dense.
                #The key prefixes the filename, so the final chunk writes the
                #`num_epochs - 1` names _expected_checkpoints and predict() look for.
                compiled_net.save((epochs_done - 1,),
                                  Numpy(self._checkpoint_dir(create=True)))
                logger.info("[snn] epoch %d/%d: %.1f%% of hidden neurons silent, "
                            "weights checkpointed at key %d",
                            epochs_done, num_epochs, silent_frac * 100,
                            epochs_done - 1)
                self._check_silent_fraction(silent_frac, epochs_done, num_epochs)

        missing = [f.name for f in self._expected_checkpoints()
                   if not f.exists()]
        if missing:
            raise RuntimeError(
                f"training did not persist: {missing} absent from "
                f"{self._checkpoint_dir()}, refusing to report success")
        logger.info("[snn] saved %d weight arrays to %s",
                    len(self._expected_checkpoints()), self._checkpoint_dir())

    def predict(self, data: VrpDataset, split: str) -> pd.Series:
        from ml_genn.compilers import InferenceCompiler
        from ml_genn.serialisers import Numpy

        #deserialise_all returns {} on a glob miss, so check exact filenames
        checkpoint_dir = self._checkpoint_dir()
        missing = [f.name for f in self._expected_checkpoints() if not f.exists()]
        if missing:
            raise FileNotFoundError(
                f"{missing} absent from {checkpoint_dir}, run fit() first")

        X, _, dates = data.sequences(split, space=self._encoder.input_space)
        scale = data.window_scale if self._encoder.wants_scale else None
        #mlGeNN wants full batches; pad with repeats, trimmed below
        n = X.shape[0]
        pad = (-n) % BATCH_SIZE
        if pad:
            X = np.concatenate([X, np.repeat(X[-1:], pad, axis=0)], axis=0)
            if scale is not None:
                scale = np.concatenate([scale, np.repeat(scale[-1:], pad, axis=0)], axis=0)

        encoded = self._encoder.encode(X, scale)
        max_spikes = self._max_spikes(encoded)

        network, input_pop, _, output_pop = self._build_network(
            encoded, self._timesteps, max_spikes)
        network.load((self.config.num_epochs - 1,), Numpy(checkpoint_dir))

        #without a seed PoissonInput redraws every predict()
        compiler = InferenceCompiler(evaluate_timesteps=self._timesteps,
                                     batch_size=BATCH_SIZE, dt=DT,
                                     rng_seed=self.config.seed)
        compiled_net = compiler.compile(network)
        with compiled_net:
            y_pred, _ = compiled_net.predict({input_pop: encoded.data}, [output_pop])
        target_hat = y_pred[output_pop][:, 0] * self._y_std + self._y_mean
        return pd.Series(target_hat[:n], index=dates)
