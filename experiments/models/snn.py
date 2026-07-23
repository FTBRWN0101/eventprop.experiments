"""SNN forecaster on mlGeNN, trained with EventProp.

One timestep is one trading day. Needs the mlgenn env, imported lazily.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.dataset import VrpDataset
from encoders.base import ENCODERS, Encoder, EncodedInput
from models.base import Forecaster

NUM_HIDDEN = 256
BATCH_SIZE = 32
NUM_EPOCHS = 50
DT = 1.0        #1 timestep = 1 trading day
TAU_MEM = 3.0   #proposal default: 3 timesteps
TAU_SYN = 1.0   #proposal default: 1 timestep
CHECKPOINT_ROOT = Path(__file__).resolve().parents[1] / ".snn_checkpoints"
_ENCODERS_DIR = Path(__file__).resolve().parents[1] / "encoders"


class SnnForecaster(Forecaster):
    """Recurrent LIF network, pluggable spike encoding, EventProp/MSE training."""

    name = "snn"
    description = "mlGeNN recurrent LIF + EventProp, pluggable encoding (config.encoding)"

    def _get_encoder(self) -> Encoder:
        ENCODERS.discover(_ENCODERS_DIR, "encoders")
        return ENCODERS.get(self.config.encoding)()

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
            input_pop = Population(input_neuron, num_neurons)
            hidden = Population(
                LeakyIntegrateFire(v_thresh=1.0, tau_mem=TAU_MEM, tau_refrac=None),
                NUM_HIDDEN)
            output = Population(
                LeakyIntegrate(tau_mem=TAU_MEM, readout="var"), 1)

            Connection(input_pop, hidden,
                      Dense(Normal(mean=0.0, sd=1.0 / np.sqrt(num_neurons))),
                      Exponential(TAU_SYN))
            Connection(hidden, hidden,
                      Dense(Normal(mean=0.0, sd=0.5 / np.sqrt(NUM_HIDDEN))),
                      Exponential(TAU_SYN))
            Connection(hidden, output,
                      Dense(Normal(mean=0.0, sd=1.0 / np.sqrt(NUM_HIDDEN))),
                      Exponential(TAU_SYN))
        return network, input_pop, output

    def _max_spikes(self, encoded: EncodedInput) -> int | None:
        if encoded.kind != "spikes":
            return None
        from ml_genn.utils.data import calc_max_spikes
        return BATCH_SIZE * calc_max_spikes(encoded.data)

    def _checkpoint_dir(self) -> Path:
        cfg = self.config
        path = CHECKPOINT_ROOT / f"{cfg.horizon}_{cfg.target}_{cfg.iv_leg}_{cfg.encoding}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def fit(self, data: VrpDataset) -> None:
        from ml_genn.compilers import EventPropCompiler
        from ml_genn.optimisers import Adam
        from ml_genn.serialisers import Numpy

        X, y, _ = data.sequences("train")
        self._timesteps = X.shape[1]
        self._y_mean, self._y_std = float(y.mean()), float(y.std() or 1.0)
        y_norm = (y - self._y_mean) / self._y_std

        self._encoder = self._get_encoder()
        self._encoder.fit(X)
        encoded = self._encoder.encode(X)
        max_spikes = self._max_spikes(encoded)

        #per-timestep loss needs a target at every timestep
        y_full = np.repeat(y_norm[:, None, None], self._timesteps, axis=1)

        network, input_pop, output_pop = self._build_network(
            encoded, self._timesteps, max_spikes)
        compiler = EventPropCompiler(example_timesteps=self._timesteps,
                                     losses="mean_square_error", optimiser=Adam(0.001),
                                     batch_size=BATCH_SIZE, dt=DT, per_timestep_loss=True)
        compiled_net = compiler.compile(network)
        with compiled_net:
            compiled_net.train({input_pop: encoded.data}, {output_pop: y_full},
                               num_epochs=NUM_EPOCHS, shuffle=True)
            compiled_net.save_connectivity((NUM_EPOCHS - 1,), Numpy(self._checkpoint_dir()))

    def predict(self, data: VrpDataset, split: str) -> pd.Series:
        from ml_genn.compilers import InferenceCompiler
        from ml_genn.serialisers import Numpy

        X, _, dates = data.sequences(split)
        encoded = self._encoder.encode(X)
        max_spikes = self._max_spikes(encoded)

        network, input_pop, output_pop = self._build_network(
            encoded, self._timesteps, max_spikes)
        network.load((NUM_EPOCHS - 1,), Numpy(self._checkpoint_dir()))

        compiler = InferenceCompiler(evaluate_timesteps=self._timesteps,
                                     batch_size=BATCH_SIZE, dt=DT)
        compiled_net = compiler.compile(network)
        with compiled_net:
            y_pred, _ = compiled_net.predict({input_pop: encoded.data}, [output_pop])
        target_hat = y_pred[output_pop][:, 0] * self._y_std + self._y_mean
        return pd.Series(target_hat, index=dates)
