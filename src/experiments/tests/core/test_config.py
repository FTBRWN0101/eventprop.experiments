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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("horizon", "quarterly"),
        ("iv_leg", "vix3m"),
        ("feature_set", "options-only"),
        ("encoding", "binary"),
        ("test_sampling", "weekly"),
        ("algorithm", "sgd"),
        ("iso_activation", "swish"),
        ("target", "variance"),
    ],
)
def test_unknown_enum_configuration_is_rejected_at_construction(field, value):
    with pytest.raises(ValueError, match=field):
        ExperimentConfig.load(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_window", 0),
        ("num_epochs", 0),
        ("learning_rate", 0.0),
        ("learning_rate", float("inf")),
        ("learning_rate", float("nan")),
        ("delta_multiplier", 0.0),
        ("delta_multiplier", float("inf")),
        ("num_hidden", 0),
        ("checkpoint_epochs", 0),
        ("lstm_hidden_size", 0),
        ("lstm_num_layers", 0),
        ("annualisation", 0),
        ("rvrp_smooth_window", 0),
        ("reg_target_duty_cycle", 1.01),
    ],
)
def test_invalid_numeric_configuration_is_rejected_at_construction(field, value):
    with pytest.raises(ValueError, match=field):
        ExperimentConfig.load(**{field: value})


def test_integer_configuration_does_not_coerce_booleans():
    with pytest.raises(ValueError, match="num_epochs"):
        ExperimentConfig.load(num_epochs=True)


@pytest.mark.parametrize("value", ["2011-01-04", "1990-01-01"])
def test_sample_start_accepts_calendar_iso_dates(value):
    assert ExperimentConfig.load(sample_start=value).sample_start == value


@pytest.mark.parametrize("value", ["2011-1-4", "2011-02-29", "20110104", 20110104])
def test_sample_start_requires_a_calendar_iso_date(value):
    with pytest.raises(ValueError, match="sample_start"):
        ExperimentConfig.load(sample_start=value)


def test_predict_epoch_must_be_an_epoch_produced_by_the_run():
    with pytest.raises(ValueError, match="predict_epoch"):
        ExperimentConfig.load(num_epochs=2, predict_epoch=2)


def test_predict_epoch_must_name_a_written_checkpoint_or_the_final_epoch():
    with pytest.raises(ValueError, match="not written"):
        ExperimentConfig.load(num_epochs=5, checkpoint_epochs=2, predict_epoch=2)
    assert ExperimentConfig.load(
        num_epochs=5, checkpoint_epochs=2, predict_epoch=3).predict_epoch == 3
    assert ExperimentConfig.load(
        num_epochs=5, checkpoint_epochs=2, predict_epoch=4).predict_epoch == 4
