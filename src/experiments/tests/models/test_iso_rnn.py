import numpy as np
import pandas as pd
import torch
from pathlib import Path

from core.config import ExperimentConfig
from models.iso_rnn import IsoRnnForecaster, build_iso_body
from models.snn import BATCH_SIZE, TAU_MEM_OUT, TAU_SYN, V_THRESH, _sample_heterogeneous_tau_mem


class _Data:
    def __init__(self, count=33, width=5):
        self.X = np.linspace(-1.0, 1.0, count * width).reshape(count, width, 1)
        self.y = np.linspace(0.1, 0.8, count)
        self.dates = pd.date_range("2012-01-02", periods=count, freq="B")
        self.window_scale = None

    def sequences(self, split, space="zscore"):
        return self.X.copy(), self.y.copy(), self.dates


def test_iso_body_uses_the_snn_tau_draw_and_output_shape():
    config = ExperimentConfig.load(num_hidden=3, seed=7, iso_activation="identity")
    body = build_iso_body(config, 2)
    assert np.allclose(body.alpha_mem.detach().numpy(),
                       np.exp(-1.0 / _sample_heterogeneous_tau_mem(3, 7)), atol=1e-7)
    assert body(torch.ones((2, 4, 2), dtype=torch.float32)).shape == (2, 4)


def test_iso_body_matches_the_confirmed_mlgenn_update_order():
    """Scaled pre-decay Exp current arrives on the timestep after emission."""
    config = ExperimentConfig.load(num_hidden=1, seed=2, iso_activation="identity")
    body = build_iso_body(config, 1)
    with torch.no_grad():
        body.w_in.fill_(1.0)
        body.w_rec.zero_()
        body.w_out.fill_(1.0)
    x = torch.tensor([[[1.0], [0.0], [0.0], [0.0]]])

    alpha_mem = float(body.alpha_mem.item())
    alpha_syn = float(np.exp(-1.0 / TAU_SYN))
    alpha_out = float(np.exp(-1.0 / TAU_MEM_OUT))
    syn_scale = 1.0 - alpha_syn
    hidden_membrane = input_syn = output_syn = output_membrane = 0.0
    expected = []
    for value in x[0, :, 0].tolist():
        hidden_membrane = alpha_mem * hidden_membrane + (
            (1.0 - alpha_mem) * syn_scale * input_syn)
        hidden = hidden_membrane - V_THRESH
        output_membrane = alpha_out * output_membrane + (
            (1.0 - alpha_out) * syn_scale * output_syn)
        expected.append(output_membrane)
        input_syn = alpha_syn * input_syn + value
        output_syn = alpha_syn * output_syn + hidden

    assert np.allclose(body(x).detach().numpy()[0], expected, atol=1e-6)


def test_iso_drops_the_trailing_partial_batch_and_is_deterministic():
    config = ExperimentConfig.load(num_hidden=2, num_epochs=2, seed=9,
                                   encoding="rate", input_window=5)
    first, second = IsoRnnForecaster(config), IsoRnnForecaster(config)
    data = _Data()
    first.fit(data)
    second.fit(data)
    assert first.fitted_range[1] == str(data.dates[BATCH_SIZE - 1].date())
    assert first.fit_tolerance_days == config.sequence_length + BATCH_SIZE
    assert np.allclose(first.predict(data, "test"), second.predict(data, "test"))


def test_iso_window_masks_only_the_trailing_fraction():
    config = ExperimentConfig.load(readout_window=True, readout_window_fraction=0.4)
    model = IsoRnnForecaster(config)
    assert model._mask(5).tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]


def test_iso_jointly_trains_and_persists_a_learnt_encoder(tmp_path):
    config = ExperimentConfig.load(repo_root=tmp_path, num_hidden=2, num_epochs=1,
                                   encoding="learnt_delta", learnt_num_neurons=3,
                                   input_window=5, seed=3)
    model = IsoRnnForecaster(config)
    model.fit(_Data(count=32))
    assert (tmp_path / "artifacts" / "encoder_checkpoints").exists()
    assert (Path(model.encoder_artifact) / "encoder.pt").is_file()
