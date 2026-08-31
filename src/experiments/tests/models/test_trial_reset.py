"""D96: clearing synaptic OutPost between trials, which EventProp does not do.

`EventPropCompiler` registers its `ZeroOutPost` custom update but schedules it with
`CustomUpdateOnTimestepBegin` at the *last* timestep of each example
(`event_prop_compiler.py:1503-1509`). `InferenceCompiler` puts the same update in a
reset group fired at batch begin (`inference_compiler.py:445-450`). So during training
the synaptic drive from one trial is still present at the start of the next.

These are the host-side halves: the flag defaults off so every run on disk reproduces,
the checkpoint path separates the two arms, and the sweep passes the flag through. That
the callback actually removes the carry-over is a gpu question and is measured by
`experiments/tools/gradient_check.py`, which fires the same update for the same reason.
"""

import pytest

from core.config import ExperimentConfig


def test_defaults_off_so_existing_results_reproduce():
    """Every arm on disk was trained with the carry-over present. The default has to
    keep reproducing them, or turning the flag on would silently rewrite history.
    """
    assert ExperimentConfig().reset_synapses_between_trials is False


def test_checkpoint_dir_separates_the_reset_arm():
    """D79's argument again: without the suffix the fixed arm overwrites the broken one
    and whichever ran last reports for both.
    """
    from models.snn import SnnForecaster

    plain = SnnForecaster(ExperimentConfig(model="snn"))._checkpoint_dir()
    reset = SnnForecaster(
        ExperimentConfig(model="snn",
                         reset_synapses_between_trials=True))._checkpoint_dir()
    assert plain != reset
    assert str(reset).endswith("_rst")


def test_campaign_namespace_separates_mutable_checkpoints():
    """A final rerun must not overwrite an older campaign's checkpoint directory."""
    from models.snn import SnnForecaster

    old = SnnForecaster(ExperimentConfig(model="snn"))._checkpoint_dir()
    final = SnnForecaster(ExperimentConfig(
        model="snn", campaign_id="final-20260831"))._checkpoint_dir()
    assert old != final
    assert str(final).endswith("_cfinal-20260831")


def test_the_reset_composes_with_the_readout_window():
    """The two are independent knobs and a paired study needs all four combinations to
    land in four different directories.
    """
    from models.snn import SnnForecaster

    def path(**kwargs):
        return SnnForecaster(ExperimentConfig(model="snn", **kwargs))._checkpoint_dir()

    combinations = {
        path(),
        path(readout_window=True),
        path(reset_synapses_between_trials=True),
        path(readout_window=True, reset_synapses_between_trials=True),
    }
    assert len(combinations) == 4


def test_sweep_passes_the_flag_through_and_omits_it_when_off():
    from tools.sweep import Arm, arm_command

    arm = Arm(encoding="delta", algorithm="eventprop", seed=1, input_window=45)
    common = ("PY", "rv_fwd", "vix9d", "weekly", "snn")

    on = arm_command(arm, *common, reset_synapses=True)
    off = arm_command(arm, *common)
    assert "--reset-synapses-between-trials" in on
    #absent rather than passed as a false value, so an old run's argv is unchanged
    assert "--reset-synapses-between-trials" not in off


@pytest.mark.parametrize("flag", [True, False])
def test_the_flag_survives_a_config_round_trip(flag):
    """The sweep reads arms back out of `config.json`, so the field has to persist."""
    config = ExperimentConfig(model="snn", reset_synapses_between_trials=flag)
    assert config.reset_synapses_between_trials is flag


def test_eprop_rejects_the_flag_on_the_host():
    """EPropCompiler registers no ZeroOutPost group, so the callback raises inside the
    library rather than being ignored. All 80 monthly e-prop arms died that way on
    2026-08-24. The same guard already exists for loss shaping and the readout window;
    its absence here was an inconsistency, not a deliberate difference.
    """
    with pytest.raises(ValueError, match="EventProp"):
        ExperimentConfig(reset_synapses_between_trials=True, algorithm="eprop")
    #and the EventProp side is untouched
    assert ExperimentConfig(reset_synapses_between_trials=True,
                            algorithm="eventprop").reset_synapses_between_trials
