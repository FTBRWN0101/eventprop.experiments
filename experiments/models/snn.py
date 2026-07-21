"""SNN forecaster on mlGeNN, rate encoding, trained with EventProp.

One timestep is one trading day. Needs the mlgenn env, imported lazily.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.dataset import VrpDataset
from models.base import Forecaster

NUM_HIDDEN = 256
BATCH_SIZE = 32
NUM_EPOCHS = 50
DT = 1.0        #1 timestep = 1 trading day
TAU_MEM = 3.0   #proposal default: 3 timesteps
TAU_SYN = 1.0   #proposal default: 1 timestep
CHECKPOINT_ROOT = Path(__file__).resolve().parents[1] / ".snn_checkpoints"


class SnnForecaster(Forecaster):
    """Recurrent LIF network, Poisson rate-encoded inputs, EventProp/MSE training."""

    name = "snn"
    description = "mlGeNN recurrent LIF + EventProp, rate encoding (proposal Phase 1)"

    def _build_network(self, n_features: int, timesteps: int):
        from ml_genn import Connection, Network, Population
        from ml_genn.compilers.event_prop_compiler import default_params
        from ml_genn.connectivity import Dense
        from ml_genn.initializers import Normal
        from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, PoissonInput
        from ml_genn.synapses import Exponential

        network = Network(default_params)
        with network:
            input_pop = Population(
                PoissonInput(signed_spikes=False, input_frames=timesteps,
                            input_frame_timesteps=1),
                n_features)
            hidden = Population(
                LeakyIntegrateFire(v_thresh=1.0, tau_mem=TAU_MEM, tau_refrac=None),
                NUM_HIDDEN)
            output = Population(
                LeakyIntegrate(tau_mem=TAU_MEM, readout="var"), 1)

            Connection(input_pop, hidden,
                      Dense(Normal(mean=0.0, sd=1.0 / np.sqrt(n_features))),
                      Exponential(TAU_SYN))
            Connection(hidden, hidden,
                      Dense(Normal(mean=0.0, sd=0.5 / np.sqrt(NUM_HIDDEN))),
                      Exponential(TAU_SYN))
            Connection(hidden, output,
                      Dense(Normal(mean=0.0, sd=1.0 / np.sqrt(NUM_HIDDEN))),
                      Exponential(TAU_SYN))
        return network, input_pop, output

    def _rate_scale(self, X: np.ndarray, fit: bool) -> np.ndarray:
        """Rescale features to [0, 1]; PoissonInput needs a non-negative rate."""
        if fit:
            self._x_min = X.min(axis=(0, 1))
            self._x_max = X.max(axis=(0, 1))
        span = self._x_max - self._x_min
        span[span == 0] = 1.0
        return np.clip((X - self._x_min) / span, 0.0, 1.0)

    def _checkpoint_dir(self) -> Path:
        cfg = self.config
        path = CHECKPOINT_ROOT / f"{cfg.horizon}_{cfg.target}_{cfg.iv_leg}"
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
        X = self._rate_scale(X, fit=True)
        #per-timestep loss needs a target at every timestep
        y_full = np.repeat(y_norm[:, None, None], self._timesteps, axis=1)

        network, input_pop, output_pop = self._build_network(X.shape[2], self._timesteps)
        compiler = EventPropCompiler(example_timesteps=self._timesteps,
                                     losses="mean_square_error", optimiser=Adam(0.001),
                                     batch_size=BATCH_SIZE, dt=DT, per_timestep_loss=True)
        compiled_net = compiler.compile(network)
        with compiled_net:
            compiled_net.train({input_pop: X}, {output_pop: y_full},
                               num_epochs=NUM_EPOCHS, shuffle=True)
            compiled_net.save_connectivity((NUM_EPOCHS - 1,), Numpy(self._checkpoint_dir()))

    def predict(self, data: VrpDataset, split: str) -> pd.Series:
        from ml_genn.compilers import InferenceCompiler
        from ml_genn.serialisers import Numpy

        X, _, dates = data.sequences(split)
        X = self._rate_scale(X, fit=False)
        network, input_pop, output_pop = self._build_network(X.shape[2], self._timesteps)
        network.load((NUM_EPOCHS - 1,), Numpy(self._checkpoint_dir()))

        compiler = InferenceCompiler(evaluate_timesteps=self._timesteps,
                                     batch_size=BATCH_SIZE, dt=DT)
        compiled_net = compiler.compile(network)
        with compiled_net:
            y_pred, _ = compiled_net.predict({input_pop: X}, [output_pop])
        target_hat = y_pred[output_pop][:, 0] * self._y_std + self._y_mean
        return pd.Series(target_hat, index=dates)
