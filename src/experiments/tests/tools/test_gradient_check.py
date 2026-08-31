"""The gradient check's own arithmetic, without a GPU or the mlgenn env.

Every conclusion the harness draws rests on two small pure functions: the loss it forms
over a scored set, and the single scale it fits across the sample. If either is wrong the
report is confident and meaningless, which is the exact failure D74 recorded. So they are
tested here, from the plain .venv, and the compiled half is tested in
``test_gradient_check_gpu.py``.
"""
import numpy as np
import pytest

from tools.gradient_check import Arm, fit_one_scale, loss, target_trace


def test_fit_recovers_a_known_scale():
    """The fit is least squares through the origin, so an exactly scaled pair is exact."""
    numerical = np.array([1.0, -2.0, 3.5, 0.25, -0.75])
    fit = fit_one_scale(-4.0 * numerical, numerical)
    assert fit.scale == pytest.approx(-4.0)
    assert fit.correlation == pytest.approx(-1.0)
    assert fit.relative_residual == pytest.approx(0.0, abs=1e-15)
    assert fit.worst_relative == pytest.approx(0.0, abs=1e-15)


def test_the_fit_has_no_intercept_to_hide_behind():
    """A constant offset must land in the residual, not be absorbed.

    This is the reason the fit is through the origin rather than an ordinary regression.
    A window that admits one timestep too many adds a term that is nearly the same for
    every weight, which is exactly what an intercept would soak up. Pearson's r is blind
    to it, so the residual is what the harness fails on.
    """
    numerical = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    offset = fit_one_scale(2.0 * numerical + 3.0, numerical)
    assert offset.correlation == pytest.approx(1.0)
    assert offset.relative_residual > 0.1


def test_a_degenerate_sample_raises_rather_than_reporting():
    with pytest.raises(ValueError, match="says nothing"):
        fit_one_scale(np.array([1.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="no scale is identifiable"):
        fit_one_scale(np.array([1.0, 2.0]), np.zeros(2))
    with pytest.raises(ValueError, match="sample sizes differ"):
        fit_one_scale(np.zeros(3), np.zeros(4))


def test_loss_counts_only_the_scored_timesteps():
    """The negative control of check 2 is this function given a wider set, so the set has
    to be an argument that is actually honoured rather than a description of one.
    """
    voltage = np.zeros(8, dtype=np.float32)
    target = target_trace(8)
    scored = np.array([5, 6, 7])
    assert loss(voltage, scored) == pytest.approx(float(np.sum(target[scored] ** 2)))
    assert loss(voltage, np.arange(8)) == pytest.approx(float(np.sum(target ** 2)))
    assert loss(voltage, np.array([], dtype=int)) == pytest.approx(0.0)


def test_loss_applies_the_shaping_weight_per_timestep():
    voltage = np.zeros(6, dtype=np.float32)
    target = target_trace(6)
    weights = np.arange(1.0, 7.0)
    scored = np.array([1, 3, 5])
    assert loss(voltage, scored, weights) == pytest.approx(
        float(np.sum(weights[scored] * target[scored] ** 2)))


def test_an_arm_scores_what_its_window_admits():
    """The harness and the generated guard have to agree on the scored set, or the
    finite difference is taken over a different loss than the gradient belongs to. The
    arm delegates to ``scored_timesteps``, which is the same function
    ``loss_shaping.windowed_drive_replacement`` builds its guard from.
    """
    from tools.gradient_check import DT, TIMESTEPS

    trial_end = TIMESTEPS * DT
    assert Arm("all").scored.tolist() == list(range(TIMESTEPS))
    #a window ending at the trial end scores the final timestep, which is the whole
    #point of writing the guard on backT rather than copying mlGeNN 3.x
    assert Arm("full", window=(0.0, trial_end)).scored.tolist() == list(range(TIMESTEPS))
    end = Arm("end", window=(0.8 * trial_end, trial_end)).scored
    assert end[0] == int(0.8 * TIMESTEPS) and end[-1] == TIMESTEPS - 1


def test_an_unshaped_arm_has_no_loss_weights():
    assert Arm("plain").loss_weights is None
    weights = Arm("shaped", tau=4.0).loss_weights
    assert weights is not None
    #unit mean over the scored set, so shaping redistributes rather than rescales
    assert float(weights.mean()) == pytest.approx(1.0)
    assert np.all(np.diff(weights) < 0.0)
