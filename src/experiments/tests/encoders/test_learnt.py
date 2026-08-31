import numpy as np
import pytest

from core.config import ExperimentConfig
from encoders.base import rasterise
from encoders.learnt import LearntConvEncoder, LearntDeltaEncoder


torch = pytest.importorskip("torch")


@pytest.mark.parametrize("encoder_cls", [LearntDeltaEncoder, LearntConvEncoder])
def test_learnt_encoder_reaches_parameters_and_has_two_output_modes(encoder_cls):
    encoder = encoder_cls(num_neurons=5, spiking=False)
    encoder.prepare(3)
    x = torch.randn(2, 4, 3, requires_grad=True)
    dense = encoder.tensor(x)
    assert dense.shape == (2, 4, 5)
    assert torch.any((dense > 0.0) & (dense < 1.0))
    dense.sum().backward()
    assert any(parameter.grad is not None and torch.any(parameter.grad != 0)
               for parameter in encoder._module.parameters())

    encoder.spiking, encoder.kind = True, "spikes"
    spikes = encoder.tensor(x.detach())
    assert set(spikes.unique().tolist()) <= {0.0, 1.0}


def test_surrogate_has_nonzero_threshold_gradient():
    encoder = LearntDeltaEncoder(num_neurons=1, spiking=True)
    encoder.prepare(1)
    with torch.no_grad():
        encoder._module.linear.weight.zero_()
        encoder._module.linear.bias.zero_()
        encoder._module.threshold.zero_()
    x = torch.zeros(1, 3, 1, requires_grad=True)
    encoder.tensor(x).sum().backward()
    assert torch.any(encoder._module.threshold.grad != 0)


def test_artifact_round_trips_with_a_content_fingerprint(tmp_path):
    config = ExperimentConfig.load(repo_root=tmp_path, encoding="learnt_delta",
                                   learnt_num_neurons=4, campaign_id="final-20260831")
    encoder = LearntDeltaEncoder(num_neurons=4)
    encoder.prepare(2)
    artifact = encoder.save_artifact(config, 2)
    loaded = LearntDeltaEncoder(num_neurons=4, spiking=True, checkpoint=artifact)
    x = np.array([[[0.0, 1.0], [0.5, 0.0]]], dtype=np.float32)
    encoder.spiking, encoder.kind = True, "spikes"
    before_export = encoder.tensor(torch.tensor(x)).detach().numpy()
    after_export = loaded._numpy_forward(x)
    assert np.array_equal(before_export, after_export)
    assert loaded.fingerprint == encoder.fingerprint
    assert (artifact / "encoder.npz").is_file()
    assert "final-20260831" in artifact.parts


def test_snn_requires_and_names_the_exact_frozen_encoder_artifact(tmp_path):
    from models.snn import SnnForecaster

    base = ExperimentConfig.load(repo_root=tmp_path, encoding="learnt_delta",
                                 learnt_num_neurons=4)
    encoder = LearntDeltaEncoder(num_neurons=4)
    encoder.prepare(2)
    artifact = encoder.save_artifact(base, 2)
    model = SnnForecaster(ExperimentConfig.load(
        repo_root=tmp_path, encoding="learnt_delta", learnt_num_neurons=4,
        encoder_checkpoint=artifact.relative_to(tmp_path)))
    assert model._get_encoder().fingerprint == encoder.fingerprint
    assert f"_enc{encoder.fingerprint[:12]}" in str(model._checkpoint_dir())


def test_rasterise_uses_the_first_reachable_timestep():
    raster = rasterise([np.array([0.0, 1.2])], [np.array([1, 0])], 2, 3)
    assert raster.shape == (1, 3, 2)
    assert raster[0, 0, 1] == 1.0
    assert raster[0, 2, 0] == 1.0
