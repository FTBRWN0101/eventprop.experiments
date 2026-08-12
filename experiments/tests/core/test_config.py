import pytest

from core.config import ExperimentConfig


def test_monthly_vix9d_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig.load(horizon="monthly", iv_leg="vix9d")


def test_monthly_vix_accepted():
    assert ExperimentConfig.load(horizon="monthly", iv_leg="vix").horizon_days == 21


def test_seed_zero_rejected():
    """GeNN reads 0 as 'unseeded', so it must never reach a compiler."""
    with pytest.raises(ValueError):
        ExperimentConfig.load(seed=0)


def test_sequence_length_defaults_to_horizon():
    assert ExperimentConfig.load().sequence_length == 5


def test_sequence_length_override():
    assert ExperimentConfig.load(input_window=60).sequence_length == 60
