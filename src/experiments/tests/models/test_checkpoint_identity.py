"""Checkpoint provenance must prevent cross-arm weight reuse."""

import json

import pytest

from core.config import ExperimentConfig


@pytest.fixture
def checkpoint_root(monkeypatch, tmp_path):
    import models.snn as snn

    monkeypatch.setattr(snn, "CHECKPOINT_ROOT", tmp_path)
    return tmp_path


def _directory(**overrides):
    from models.snn import SnnForecaster

    return SnnForecaster(ExperimentConfig(model="snn", **overrides))._checkpoint_dir()


@pytest.mark.parametrize(("field", "value"), [
    ("feature_set", "price-only"),
    ("delta_multiplier", 2.0),
    ("holdout_val", True),
    ("sample_start", "2010-01-01"),
    ("drop_features", ("rv_1",)),
    ("checkpoint_epochs", 1),
    ("silent_neuron_abort", False),
])
def test_every_training_relevant_config_change_gets_a_new_checkpoint_identity(
        checkpoint_root, field, value):
    base = _directory()
    changed = _directory(**{field: value})

    assert changed != base
    assert "_cfg" in changed.name


def test_checkpoint_metadata_round_trips_and_allows_the_exact_same_config(checkpoint_root):
    from models.snn import CHECKPOINT_METADATA_FILENAME, SnnForecaster

    model = SnnForecaster(ExperimentConfig(model="snn", feature_set="price-only"))
    directory = model._checkpoint_dir(create=True)

    metadata = json.loads((directory / CHECKPOINT_METADATA_FILENAME).read_text())
    assert metadata["identity"][:20] in directory.name
    assert metadata["payload"]["config"]["feature_set"] == "price-only"
    assert model._checkpoint_dir() == directory


def test_checkpoint_reuse_refuses_mismatched_metadata(checkpoint_root):
    from models.snn import CHECKPOINT_METADATA_FILENAME, SnnForecaster

    model = SnnForecaster(ExperimentConfig(model="snn"))
    directory = model._checkpoint_dir(create=True)
    metadata_path = directory / CHECKPOINT_METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text())
    metadata["payload"]["config"]["learning_rate"] = 0.5
    metadata_path.write_text(json.dumps(metadata), encoding="utf8")

    with pytest.raises(RuntimeError, match="does not match"):
        model._checkpoint_dir()


def test_checkpoint_reuse_refuses_legacy_directory_without_provenance(checkpoint_root):
    from models.snn import SnnForecaster

    model = SnnForecaster(ExperimentConfig(model="snn"))
    directory = model._checkpoint_dir()
    directory.mkdir(parents=True)
    (directory / "49-Conn_input_hidden-g.npy").write_bytes(b"unverified")

    with pytest.raises(RuntimeError, match="unverifiable provenance"):
        model._checkpoint_dir()


def test_checkpoint_identity_survives_a_repository_move(checkpoint_root, tmp_path):
    old = _directory(repo_root=tmp_path / "old")
    moved = _directory(repo_root=tmp_path / "moved")

    assert old == moved


def test_selecting_a_saved_epoch_reuses_the_training_checkpoint(checkpoint_root):
    trained = _directory(num_epochs=20, checkpoint_epochs=10)
    selected = _directory(num_epochs=20, checkpoint_epochs=10, predict_epoch=9)

    assert selected == trained
