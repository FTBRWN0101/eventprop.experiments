"""LSTM baseline, and the control for the SNN's mean-collapse.

Same windows as the SNN but a final-timestep loss. Runs on CPU to keep the
GPU free for GeNN.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.dataset import VrpDataset
from models.base import Forecaster

HIDDEN_SIZE = 64      #modest against ~2250 training windows
NUM_LAYERS = 1
BATCH_SIZE = 32
DROPOUT = 0.0         #single layer, so torch would ignore it anyway
PATIENCE = 10         #epochs without improvement before stopping

logger = logging.getLogger(__name__)


class LstmForecaster(Forecaster):
    """Single-layer LSTM reading the final timestep, trained on final-timestep MSE."""

    name = "lstm"
    description = "LSTM on standardised windows, final-timestep readout (direct target)"

    @property
    def fit_tolerance_days(self) -> int:
        """Windowing only, in trading days (D81).

        The first window ends L - 1 days after the sample starts. Unlike the SNN this
        trainer takes a partial final batch, so nothing is lost at the other end.
        """
        return self.config.sequence_length

    def _build(self, num_features: int):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(num_features, HIDDEN_SIZE, NUM_LAYERS,
                                    batch_first=True, dropout=DROPOUT)
                self.head = nn.Linear(HIDDEN_SIZE, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        return Net()

    def fit(self, data: VrpDataset) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        X, y, dates = data.sequences("train")
        self.fitted_range = (str(dates[0].date()), str(dates[-1].date()))
        self._y_mean, self._y_std = float(y.mean()), float(y.std() or 1.0)
        y_norm = (y - self._y_mean) / self._y_std

        self._model = self._build(X.shape[2])
        optimiser = torch.optim.Adam(self._model.parameters(),
                                     lr=self.config.learning_rate)
        loss_fn = torch.nn.MSELoss()
        loader = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y_norm, dtype=torch.float32)),
            batch_size=BATCH_SIZE, shuffle=True)

        validation = self._validation_set(data)
        best = (float("inf"), -1, None)  #(val loss, epoch, state)
        for epoch in range(self.config.num_epochs):
            self._model.train()
            total = 0.0
            for xb, yb in loader:
                optimiser.zero_grad()
                loss = loss_fn(self._model(xb), yb)
                loss.backward()
                optimiser.step()
                total += float(loss.detach()) * len(xb)
            train_mse = total / len(loader.dataset)

            if validation is None:
                if (epoch + 1) % 10 == 0:
                    logger.info("[lstm] epoch %d/%d: train MSE %.4f (normalised)",
                                epoch + 1, self.config.num_epochs, train_mse)
                continue

            val_mse = self._evaluate(validation, loss_fn)
            if val_mse < best[0]:
                #clone: state_dict hands back live tensors
                best = (val_mse, epoch,
                        {k: v.detach().clone()
                         for k, v in self._model.state_dict().items()})
            if (epoch + 1) % 10 == 0:
                logger.info("[lstm] epoch %d/%d: train MSE %.4f, val MSE %.4f "
                            "(normalised)", epoch + 1, self.config.num_epochs,
                            train_mse, val_mse)
            if epoch - best[1] >= PATIENCE:
                logger.info("[lstm] early stop at epoch %d: no val improvement "
                            "in %d epochs", epoch + 1, PATIENCE)
                break

        if best[2] is not None:
            self._model.load_state_dict(best[2])
            self.best_epoch = best[1] + 1
            logger.info("[lstm] restored epoch %d (val MSE %.4f normalised)",
                        self.best_epoch, best[0])

    def _validation_set(self, data: VrpDataset):
        """Standardised validation windows, or ``None`` when no holdout is configured."""
        import torch

        if not self.config.holdout_val:
            logger.info("[lstm] no validation holdout: training %d epochs without "
                        "early stopping (pass --holdout-val to select on 2017-2019)",
                        self.config.num_epochs)
            return None
        X, y, _ = data.sequences("val")
        if not len(X):
            raise ValueError("holdout_val is set but the validation split is empty")
        y_norm = (y - self._y_mean) / self._y_std
        return (torch.tensor(X, dtype=torch.float32),
                torch.tensor(y_norm, dtype=torch.float32))

    def _evaluate(self, validation, loss_fn) -> float:
        import torch

        X, y = validation
        self._model.eval()
        with torch.no_grad():
            return float(loss_fn(self._model(X), y))

    def predict(self, data: VrpDataset, split: str) -> pd.Series:
        import torch

        X, _, dates = data.sequences(split)
        self._model.eval()
        with torch.no_grad():
            pred = self._model(torch.tensor(X, dtype=torch.float32)).numpy()
        return pd.Series(pred * self._y_std + self._y_mean, index=dates)
