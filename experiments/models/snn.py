"""SNN forecaster on mlGeNN, trained with EventProp.

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

#from the mlGeNN eventprop example
V_THRESH = 0.61
W_IN_SCALE = 2.5
W_REC_SCALE = 1.5
W_OUT_SCALE = 2.0
TAU_MEM_OUT = 20.0   #long output integrator
REG_LAMBDA = 1e-8
REG_TARGET_DUTY_CYCLE = 0.3   #scales reg_nu_upper to the trial length

SILENT_NEURON_DELTA_G = 0.002

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
        if self.config.encoding == "delta":
            return cls(multiplier=self.config.delta_multiplier)
        return cls()

    def _build_network(self, encoded: EncodedInput, timesteps: int, max_spikes: int | None):
        from ml_genn import Connection, Network, Population
        from ml_genn.compilers.event_prop_compiler import default_params
        from ml_genn.connectivity import Dense
        from ml_genn.initializers import Normal
        from ml_genn.neurons import (LeakyIntegrate, LeakyIntegrateFire,
                                     PoissonInput, SpikeInput)
        from ml_genn.synapses import Exponential

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
                LeakyIntegrateFire(v_thresh=V_THRESH,
                                   tau_mem=_sample_heterogeneous_tau_mem(
                                       NUM_HIDDEN, self.config.seed),
                                   tau_refrac=None),
                NUM_HIDDEN, record_spikes=True, name="hidden")
            output = Population(
                LeakyIntegrate(tau_mem=TAU_MEM_OUT, readout="var"), 1, name="output")

            Connection(input_pop, hidden,
                      Dense(Normal(mean=0.0, sd=W_IN_SCALE / np.sqrt(num_neurons))),
                      Exponential(TAU_SYN))
            Connection(hidden, hidden,
                      Dense(Normal(mean=0.0, sd=W_REC_SCALE / np.sqrt(NUM_HIDDEN))),
                      Exponential(TAU_SYN))
            Connection(hidden, output,
                      Dense(Normal(mean=0.0, sd=W_OUT_SCALE / np.sqrt(NUM_HIDDEN))),
                      Exponential(TAU_SYN))
        return network, input_pop, hidden, output

    def _max_spikes(self, encoded: EncodedInput) -> int | None:
        if encoded.kind != "spikes":
            return None
        from ml_genn.utils.data import calc_max_spikes
        return BATCH_SIZE * calc_max_spikes(encoded.data)

    def _checkpoint_dir(self, create: bool = False) -> Path:
        cfg = self.config
        path = CHECKPOINT_ROOT / f"{cfg.horizon}_{cfg.target}_{cfg.iv_leg}_{cfg.encoding}"
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _expected_checkpoints(self) -> list[Path]:
        """Weight files save() writes, named ``<epoch>-Conn_<src>_<dst>-g.npy``."""
        epoch = self.config.num_epochs - 1
        directory = self._checkpoint_dir()
        return [directory / f"{epoch}-Conn_{src}_{dst}-g.npy"
                for src, dst in (("input", "hidden"), ("hidden", "hidden"),
                                 ("hidden", "output"))]

    def fit(self, data: VrpDataset) -> None:
        from ml_genn.callbacks import BatchProgressBar, SpikeRecorder
        from ml_genn.compilers import EventPropCompiler
        from ml_genn.optimisers import Adam
        from ml_genn.serialisers import Numpy

        SilentNeuronRecovery = _make_silent_neuron_recovery_cls()

        #mlGeNN shuffles with numpy's global RNG, so seed that too
        np.random.seed(self.config.seed)

        X, y, train_dates = data.sequences("train")
        #mlGeNN wants full batches; drop the trailing partial one
        n_complete = (X.shape[0] // BATCH_SIZE) * BATCH_SIZE
        X, y = X[:n_complete], y[:n_complete]
        self.fitted_range = (str(train_dates[0].date()),
                             str(train_dates[n_complete - 1].date()))
        num_epochs = self.config.num_epochs
        self._timesteps = X.shape[1]
        self._y_mean, self._y_std = float(y.mean()), float(y.std() or 1.0)
        y_norm = (y - self._y_mean) / self._y_std

        self._encoder = self._get_encoder()
        self._encoder.fit(X)
        encoded = self._encoder.encode(X)
        max_spikes = self._max_spikes(encoded)

        #per-timestep loss needs a target at every timestep
        y_full = np.repeat(y_norm[:, None, None], self._timesteps, axis=1)

        network, input_pop, hidden_pop, output_pop = self._build_network(
            encoded, self._timesteps, max_spikes)
        reg_nu_upper = max(1.0, REG_TARGET_DUTY_CYCLE * self._timesteps)
        compiler = EventPropCompiler(example_timesteps=self._timesteps,
                                     losses="mean_square_error",
                                     optimiser=Adam(self.config.learning_rate),
                                     batch_size=BATCH_SIZE, dt=DT, per_timestep_loss=True,
                                     reg_lambda_upper={hidden_pop: REG_LAMBDA},
                                     reg_lambda_lower={hidden_pop: REG_LAMBDA},
                                     reg_nu_upper=reg_nu_upper,
                                     rng_seed=self.config.seed)
        compiled_net = compiler.compile(network)
        compiled_net.num_recording_timesteps = self._timesteps
        self.silent_neuron_fraction: list[float] = []
        with compiled_net:
            epochs_done = 0
            while epochs_done < num_epochs:
                chunk = min(CHECKPOINT_EPOCHS, num_epochs - epochs_done)
                recorder = SpikeRecorder(hidden_pop, key="hidden_spikes",
                                         record_counts=True)
                #callbacks get copied, so read the data from train()'s return
                _, cb_data = compiled_net.train(
                    {input_pop: encoded.data}, {output_pop: y_full},
                    num_epochs=chunk, start_epoch=epochs_done, shuffle=True,
                    callbacks=[BatchProgressBar(), recorder,
                              SilentNeuronRecovery(hidden_pop)])
                epochs_done += chunk

                spike_counts = cb_data["hidden_spikes"]
                silent_frac = _silent_neuron_fraction(spike_counts)
                self.silent_neuron_fraction.append(silent_frac)
                logger.info("[snn] epoch %d/%d: %.1f%% of hidden neurons silent",
                           epochs_done, num_epochs, silent_frac * 100)
                if silent_frac > SILENT_NEURON_WARN_THRESHOLD:
                    logger.warning(
                        "[snn] spike deletion: %.1f%% of hidden neurons silent at "
                        "epoch %d/%d (>%.0f%% threshold), consider reviewing loss "
                        "shaping / silent-neuron recovery",
                        silent_frac * 100, epochs_done, num_epochs,
                        SILENT_NEURON_WARN_THRESHOLD * 100)

            #save(), not save_connectivity(): that one does nothing for Dense
            compiled_net.save((num_epochs - 1,),
                              Numpy(self._checkpoint_dir(create=True)))

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

        X, _, dates = data.sequences(split)
        #mlGeNN wants full batches; pad with repeats, trimmed below
        n = X.shape[0]
        pad = (-n) % BATCH_SIZE
        if pad:
            X = np.concatenate([X, np.repeat(X[-1:], pad, axis=0)], axis=0)

        encoded = self._encoder.encode(X)
        max_spikes = self._max_spikes(encoded)

        network, input_pop, _, output_pop = self._build_network(
            encoded, self._timesteps, max_spikes)
        network.load((self.config.num_epochs - 1,), Numpy(checkpoint_dir))

        compiler = InferenceCompiler(evaluate_timesteps=self._timesteps,
                                     batch_size=BATCH_SIZE, dt=DT)
        compiled_net = compiler.compile(network)
        with compiled_net:
            y_pred, _ = compiled_net.predict({input_pop: encoded.data}, [output_pop])
        target_hat = y_pred[output_pop][:, 0] * self._y_std + self._y_mean
        return pd.Series(target_hat[:n], index=dates)
