"""The gradient check against a real compile. Needs the mlgenn env, run with -m gpu.

``tools/gradient_check.py`` is the report: it prints correlations, residuals and two
sweeps, because the numbers are what a person has to read. This file is the guard: it
runs the same functions on the same fixture and asserts, so a regression in the patch, in
mlGeNN or in the harness fails the suite instead of quietly changing a number nobody
re-reads.

Nothing here re-derives anything. Every assertion is on the arrays the tool itself
produces, and the thresholds are the tool's own constants, so the guard and the report
cannot disagree about what passing means.
"""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("ml_genn", reason="requires the mlgenn conda env")

from tools import gradient_check as gc  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def measured():
    """One fixture, one forward sweep, every arm compiled once.

    Module scoped because compiling is the whole cost: the finite differences are 48
    forward runs of a 24-neuron network and take a fraction of a second, while each arm
    is a GeNN code generation and a native build. Everything downstream is plain numpy on
    the arrays this returns.
    """
    fixture = gc.build_fixture()
    arms = gc.default_arms()
    compiled = gc.compile_arm(fixture, arms[0])
    with compiled:
        baseline = gc.measure(compiled, fixture)
        sample = gc.live_sample(baseline)
        sweep = gc.collect_forward_sweep(compiled, fixture, sample, gc.DEFAULT_EPSILON)
    analytic, fits, voltages = gc.run_arms(fixture, arms, sweep, sample)
    return SimpleNamespace(fixture=fixture, arms=arms, sample=sample, sweep=sweep,
                           baseline=baseline, analytic=analytic, fits=fits,
                           voltages=voltages)


def test_the_fixture_produces_a_gradient_worth_comparing(measured):
    """The precondition every assertion below depends on, asserted on its own.

    D74's finding was a fixture that trained nothing while the tests passed. The
    equivalent failure here is a fixture whose gradients are all zero, or all the same
    number, either of which would let a correlation come out at 1.0 while proving
    nothing. So the sample is required to be large, non-zero and non-degenerate before
    anything is read off it.
    """
    gradient = measured.baseline.gradient[measured.sample]
    assert measured.sample.size >= 20, (
        "too few live weights for a correlation to mean anything")
    assert np.all(gradient != 0.0)
    assert np.unique(gradient).size == gradient.size, (
        "sampled gradients repeat, so the sample carries fewer independent points than "
        "it looks like it does")
    assert measured.baseline.settled, "the measured trial was not at a fixed point"
    assert not measured.baseline.weights_moved


def test_the_forward_pass_is_the_same_in_every_arm(measured):
    """The patches rewrite the backward drive only, so one forward sweep serves them all.

    If this ever fails, every finite difference in the report was taken on the wrong
    trajectory for at least one arm.
    """
    assert gc.check_forward_pass_is_arm_independent(measured.voltages)


def test_every_arm_admits_one_scale(measured):
    """Check 1. One constant relates the analytic gradient to the finite difference.

    The thresholds are the tool's, and the measured residual floor is about 1.2e-3
    against a limit of 2e-2, so this fails on something structural rather than on a
    slightly worse fit.
    """
    for name, fit in measured.fits.items():
        assert fit.correlation >= gc.MIN_CORRELATION, f"{name}: r={fit.correlation}"
        assert fit.relative_residual <= gc.MAX_RELATIVE_RESIDUAL, (
            f"{name}: residual={fit.relative_residual:.3e}")


def test_a_full_trial_window_reproduces_the_unwindowed_drive(measured):
    """Check 3, the cheapest and strongest regression against an off-by-one.

    A window spanning the whole trial must score every timestep the unwindowed line
    scores, including the last one. mlGeNN 3.x's own guard would drop it, which is the
    difference ``loss_shaping.scored_timesteps`` documents at length; if the guard here
    ever acquired that semantics, these gradients would stop matching. The arithmetic is
    otherwise identical, so the match is required to be bit for bit rather than close.
    """
    for left, right in (("stock", "window_full"), ("shaped", "shaped_window_full")):
        assert np.array_equal(measured.analytic[left], measured.analytic[right]), (
            f"{left} and {right} should be the same gradient, max|diff|="
            f"{np.max(np.abs(measured.analytic[left] - measured.analytic[right])):.3e}")


def test_a_windowed_gradient_rejects_a_whole_trial_loss(measured):
    """Check 2 in the form the brief asks for: the narrow arm against the whole trial.

    A guard that did nothing would put the arm's gradient exactly on the whole-trial
    finite difference and give a separation of 1.0.
    """
    for name in ("window_narrow", "window_early", "window_middle",
                 "shaped_window_narrow"):
        arm = next(a for a in measured.arms if a.name == name)
        whole = gc.Arm(name, tau=arm.tau, window=None)
        wrong = gc.fit_one_scale(
            measured.analytic[name],
            gc.finite_difference(measured.sweep, whole.scored, whole.loss_weights))
        separation = (wrong.relative_residual
                      / measured.fits[name].relative_residual)
        assert separation >= gc.MIN_WINDOW_SEPARATION, (
            f"{name} fits a whole-trial loss nearly as well as its own (x{separation:.1f})")


def test_each_arm_is_best_explained_by_its_own_scored_set(measured):
    """Check 2 in full: diagonal dominance of the cross-fit residual matrix.

    Stronger than the whole-trial control alone, and it has no blind spot at the
    production window. A window that ends at the trial end scores the timesteps the
    whole-trial loss is dominated by anyway, so those two losses are nearly proportional
    and a fit with a free scale absorbs most of the difference. Fitting every arm against
    every other arm's scored set does not let that happen: an unguarded drive would make
    each row of this matrix constant.
    """
    matrix = gc.cross_fit(measured.analytic, measured.arms, measured.sweep)
    for row in gc.CROSS_FIT_ARMS:
        diagonal = matrix[(row, row)].relative_residual
        others = {column: matrix[(row, column)].relative_residual
                  for column in gc.CROSS_FIT_ARMS if column != row}
        nearest, value = min(others.items(), key=lambda item: item[1])
        assert value >= gc.MIN_WINDOW_SEPARATION * diagonal, (
            f"{row}'s gradient is explained almost as well by {nearest}'s scored set "
            f"({value:.3e}) as by its own ({diagonal:.3e})")


def test_the_checks_reject_a_deliberately_broken_guard(measured):
    """The test that makes every other test in this file mean something.

    D74 is the record of a suite that was green and proved nothing, so the question
    "would these assertions notice?" has to be answered by running them against a fault
    rather than by reasoning about them. The fault injected is the specific one the guard
    was written to avoid: mlGeNN 3.x's semantics, one timestep early at both ends, built
    into the compiler while the fit is taken against the window that was asked for.
    """
    assert gc.report_negative_control(measured.fixture, measured.sweep, measured.sample,
                                      measured.analytic["stock"]), (
        "a guard shifted one timestep passed every check, so nothing else here is "
        "evidence that the real guard is right")


def test_the_narrow_window_gradient_is_not_the_unwindowed_one(measured):
    """The blunt version of the same thing, kept because it is unmissable.

    ``test_loss_shaping_gpu`` already asserts the patch changes weights after training.
    This asserts the far more specific claim that it changes the raw accumulated
    gradient, at a fixed weight vector, with no optimiser between the two.
    """
    assert not np.allclose(measured.analytic["stock"],
                           measured.analytic["window_narrow"])
    assert not np.allclose(measured.analytic["stock"], measured.analytic["shaped"])
