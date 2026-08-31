"""Non-spiking recurrent control matched to the EventProp SNN's public settings.

At each timestep mlGeNN's EventProp defaults scale the pre-decay ``Exponential``
synaptic current by ``1 - alpha_syn``, then scale membrane input by
``1 - alpha_mem``.
The generated runner calls synapse updates before neuron updates, so newly emitted
events are delivered for the next timestep.
Only the hidden threshold/reset is replaced: this control applies a smooth activation
to ``V - V_THRESH`` and leaves the membrane intact.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.dataset import VrpDataset
from encoders.base import ENCODERS, Encoder, rasterise
from encoders.learnt import LearntEncoder
from models.base import Forecaster
from models.snn import (BATCH_SIZE, DT, TAU_MEM_OUT, TAU_SYN, V_THRESH,
                        W_OUT_SCALE, _ENCODERS_DIR, _sample_heterogeneous_tau_mem)

logger = logging.getLogger(__name__)
PATIENCE = 10


def build_iso_body(config, num_input: int):
    """Build the torch recurrence, keeping torch out of module import time."""
    import torch
    import torch.nn as nn

    tau_mem = _sample_heterogeneous_tau_mem(config.num_hidden, config.seed)
    alpha_mem = np.exp(-DT / tau_mem).astype(np.float32)
    alpha_syn = float(np.exp(-DT / TAU_SYN))
    alpha_out = float(np.exp(-DT / TAU_MEM_OUT))
    activation = config.iso_activation

    class IsoBody(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w_in = nn.Parameter(torch.empty(num_input, config.num_hidden))
            self.w_rec = nn.Parameter(torch.empty(config.num_hidden, config.num_hidden))
            self.w_out = nn.Parameter(torch.empty(config.num_hidden, 1))
            self.register_buffer("alpha_mem", torch.tensor(alpha_mem))
            self.register_buffer("alpha_syn", torch.tensor(alpha_syn))
            self.register_buffer("alpha_out", torch.tensor(alpha_out))
            self.register_buffer("syn_scale", torch.tensor(1.0 - alpha_syn))
            nn.init.normal_(self.w_in, std=config.w_in_scale / np.sqrt(num_input))
            nn.init.normal_(self.w_rec, std=config.w_rec_scale / np.sqrt(config.num_hidden))
            nn.init.normal_(self.w_out, std=W_OUT_SCALE / np.sqrt(config.num_hidden))

        def _activation(self, membrane):
            shifted = membrane - V_THRESH
            if activation == "sigmoid":
                return torch.sigmoid(shifted)
            if activation == "tanh":
                return torch.tanh(shifted)
            if activation == "relu":
                return torch.relu(shifted)
            return shifted

        def forward(self, x):
            batch, timesteps, _ = x.shape
            hidden = x.new_zeros((batch, config.num_hidden))
            hidden_membrane = x.new_zeros((batch, config.num_hidden))
            input_syn = x.new_zeros((batch, config.num_hidden))
            recurrent_syn = x.new_zeros((batch, config.num_hidden))
            output_membrane = x.new_zeros((batch, 1))
            output_syn = x.new_zeros((batch, 1))
            trace = []
            for timestep in range(timesteps):
                # This mirrors generated EventProp code: scale pre-decay synaptic
                # current, integrate, then decay and deliver emitted events.
                hidden_membrane = self.alpha_mem * hidden_membrane + (
                    (1.0 - self.alpha_mem) * self.syn_scale * (input_syn + recurrent_syn))
                hidden = self._activation(hidden_membrane)
                output_membrane = self.alpha_out * output_membrane + (
                    (1.0 - self.alpha_out) * self.syn_scale * output_syn)
                trace.append(output_membrane.squeeze(-1))
                input_syn = self.alpha_syn * input_syn + x[:, timestep] @ self.w_in
                recurrent_syn = self.alpha_syn * recurrent_syn + hidden @ self.w_rec
                output_syn = self.alpha_syn * output_syn + hidden @ self.w_out
            return torch.stack(trace, dim=1)

    return IsoBody()


class IsoRnnForecaster(Forecaster):
    """Matched non-spiking recurrent control consuming the SNN's encoder inputs."""

    name = "iso_rnn"
    description = "Matched non-spiking leaky recurrent control with no threshold reset"

    @property
    def fit_tolerance_days(self) -> int:
        return self.config.sequence_length + BATCH_SIZE

    def _get_encoder(self) -> Encoder:
        ENCODERS.discover(_ENCODERS_DIR, "encoders")
        cls = ENCODERS.get(self.config.encoding)
        if self.config.encoding.startswith("delta"):
            return cls(multiplier=self.config.delta_multiplier)
        if self.config.encoding.startswith("learnt_"):
            return cls(num_neurons=self.config.learnt_num_neurons, spiking=False)
        return cls()

    def _input_array(self, X: np.ndarray, scale: np.ndarray | None) -> np.ndarray:
        if self._encoder.kind == "spikes":
            events = self._encoder._events(X, scale) if scale is not None else self._encoder._events(X)
            return rasterise(*events, timesteps=X.shape[1])
        encoded = self._encoder.encode(X, scale)
        if encoded.kind not in ("rate", "dense"):
            raise ValueError(f"iso_rnn cannot consume encoded kind {encoded.kind!r}")
        return np.asarray(encoded.data, dtype=np.float32)

    def _mask(self, timesteps: int):
        import torch

        mask = torch.ones(timesteps, dtype=torch.float32)
        if self.config.readout_window:
            start = int(np.floor((1.0 - self.config.readout_window_fraction) * timesteps))
            mask[:start] = 0.0
        return mask

    def _loss(self, trace, target):
        mask = self._mask(trace.shape[1]).to(trace.device)
        return (((trace - target[:, None]) ** 2) * mask).sum() / (len(target) * mask.sum())

    def fit(self, data: VrpDataset) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self._encoder = self._get_encoder()
        X, y, dates = data.sequences("train", space=self._encoder.input_space)
        scale = data.window_scale if self._encoder.wants_scale else None
        n_complete = (len(X) // BATCH_SIZE) * BATCH_SIZE
        if n_complete == 0:
            raise ValueError("iso_rnn needs at least one complete training batch")
        X, y, dates = X[:n_complete], y[:n_complete], dates[:n_complete]
        if scale is not None:
            scale = scale[:n_complete]
        learnt = isinstance(self._encoder, LearntEncoder)
        self._encoder.fit(X, scale)
        encoded = X.astype(np.float32) if learnt else self._input_array(X, scale)
        self.fitted_range = (str(dates[0].date()), str(dates[-1].date()))
        self._y_mean, self._y_std = float(y.mean()), float(y.std() or 1.0)
        y_norm = (y - self._y_mean) / self._y_std
        num_input = self._encoder.num_neurons if learnt else encoded.shape[2]
        self._model = build_iso_body(self.config, num_input)
        parameters = list(self._model.parameters())
        if learnt:
            parameters += list(self._encoder._module.parameters())
        optimiser = torch.optim.Adam(parameters, lr=self.config.learning_rate)
        loader = DataLoader(TensorDataset(torch.tensor(encoded), torch.tensor(y_norm, dtype=torch.float32)),
                            batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        best = (float("inf"), -1, None)
        best_encoder = None
        for epoch in range(self.config.num_epochs):
            self._model.train()
            for xb, yb in loader:
                optimiser.zero_grad()
                body_input = self._encoder.tensor(xb) if learnt else xb
                self._loss(self._model(body_input), yb).backward()
                optimiser.step()
            if self.config.holdout_val:
                value = self._validation_loss(data)
                if value < best[0]:
                    best = (value, epoch, {k: v.detach().clone() for k, v in self._model.state_dict().items()})
                    if learnt:
                        best_encoder = {k: v.detach().clone()
                                        for k, v in self._encoder._module.state_dict().items()}
                if epoch - best[1] >= PATIENCE:
                    break
        if best[2] is not None:
            self._model.load_state_dict(best[2])
            if best_encoder is not None:
                self._encoder._module.load_state_dict(best_encoder)
            self.best_epoch = best[1] + 1
        if learnt:
            self.encoder_artifact = str(self._encoder.save_artifact(self.config, X.shape[2]))

    def _validation_loss(self, data: VrpDataset) -> float:
        import torch

        X, y, _ = data.sequences("val", space=self._encoder.input_space)
        if not len(X):
            raise ValueError("holdout_val is set but the validation split is empty")
        scale = data.window_scale if self._encoder.wants_scale else None
        encoded = self._input_array(X, scale)
        with torch.no_grad():
            return float(self._loss(self._model(torch.tensor(encoded)),
                                    torch.tensor((y - self._y_mean) / self._y_std, dtype=torch.float32)))

    def predict(self, data: VrpDataset, split: str) -> pd.Series:
        import torch

        X, _, dates = data.sequences(split, space=self._encoder.input_space)
        scale = data.window_scale if self._encoder.wants_scale else None
        encoded = self._input_array(X, scale)
        self._model.eval()
        with torch.no_grad():
            pred = self._model(torch.tensor(encoded))[:, -1].numpy()
        return pd.Series(pred * self._y_std + self._y_mean, index=dates)
