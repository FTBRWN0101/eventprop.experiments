"""Loss shaping maths and generated C. Pure numpy: runs from the plain project .venv.

The load-bearing test is ``test_weight_lands_on_the_forward_time_it_claims``, which
replays mlGeNN's ring-buffer arithmetic and shows the weight is *not* time reversed.
"""
import re

import numpy as np
import pytest

from compilers.loss_shaping import (drive_replacement, drive_source,
                                    scored_timesteps, shaping_weights,
                                    weight_constants,
                                    windowed_drive_replacement)
from core.config import ExperimentConfig

DT = 1.0
T = 45   #the L=45 trial length D47 calibrated on


def _replay_backward_pass(timesteps: int, dt: float = DT, trials: int = 3):
    """``[(backT, entry)]`` per trial, where ``entry`` is the ``(trial, step)`` whose
    loss term the backward pass reads at that kernel step.

    Mirrors the installed compiler exactly:
      * ``:851``  backT = T*dt - t - dt
      * ``:740``  the read is guarded by ``Trial > 0`` and decrements first
      * ``:750``  the forward write happens after the read, in the same kernel step
      * ``:361``  the between-trial reset does ``read = write``, then wraps write at 2T
    """
    ring: list = [None] * (2 * timesteps)
    write_off = read_off = 0
    out = []
    for trial in range(trials):
        per_trial = []
        for step in range(timesteps):
            back_t = timesteps * dt - step * dt - dt
            entry = None
            if trial > 0:
                read_off -= 1
                entry = ring[read_off]
            per_trial.append((back_t, entry))
            ring[write_off] = (trial, step)
            write_off += 1
        read_off = write_off
        if write_off >= 2 * timesteps:
            write_off = 0
        out.append(per_trial)
    return out


def _emitted_constants(code: str) -> tuple[float, float]:
    """``(scale, rate)`` parsed back out of the generated C."""
    m = re.search(r"drive = \(([-\d.e+]+) \* exp\(-backT \* ([-\d.e+]+)\)\) \* error", code)
    assert m, f"generated drive line does not match the expected shape: {code!r}"
    return float(m.group(1)), float(m.group(2))


def test_ring_replay_reads_the_forward_step_backt_names():
    """Sanity on the emulator itself: at kernel step j the ring hands back the term
    written at forward step T-1-j of the *previous* trial, and backT is its time.
    """
    replay = _replay_backward_pass(T)
    assert all(entry is None for _, entry in replay[0]), "trial 0 has nothing to replay"
    for trial in (1, 2):
        for j, (back_t, entry) in enumerate(replay[trial]):
            written_trial, k = entry
            assert written_trial == trial - 1
            assert k == T - 1 - j
            assert back_t == pytest.approx(k * DT)


def test_weight_lands_on_the_forward_time_it_claims():
    """D29's trap. The weight applied to the loss term from forward step k must be
    w[k]; using the kernel's own ``t`` instead of ``backT`` gives exactly w reversed.
    """
    tau = float(T)
    w = shaping_weights(T, DT, tau)
    scale, rate = _emitted_constants(drive_replacement(DT, T, tau))

    applied = np.empty(T)
    naive = np.empty(T)
    for j, (back_t, (_, k)) in enumerate(_replay_backward_pass(T)[1]):
        applied[k] = scale * np.exp(-rate * back_t)   #what we emit
        naive[k] = scale * np.exp(-rate * (j * DT))   #exp(-t/tau), the trap

    np.testing.assert_allclose(applied, w, rtol=1e-12)
    np.testing.assert_allclose(naive, w[::-1], rtol=1e-12)
    assert not np.allclose(applied, naive), "shape is symmetric, the test proves nothing"
    #and the direction is the proposal's: heaviest at the start of the trial
    assert applied[0] > applied[-1]
    assert applied[0] == applied.max()


def test_weight_peaks_at_the_start_and_decays_monotonically():
    w = shaping_weights(T, DT, float(T))
    assert np.argmax(w) == 0
    assert np.all(np.diff(w) < 0)
    #one e-fold across the trial, by construction of tau = T
    assert w[0] / w[-1] == pytest.approx(np.exp((T - 1) * DT / T))


def test_weight_has_unit_mean_so_shaping_only_redistributes():
    """Total gradient mass is preserved, so a shaped arm is not also an lr change."""
    for tau in (5.0, 21.0, 45.0, 200.0):
        w = shaping_weights(T, DT, tau)
        assert w.mean() == pytest.approx(1.0, rel=1e-12)
        assert (w * DT).sum() == pytest.approx(T * DT, rel=1e-12)


def test_infinite_tau_is_the_unshaped_drive_bit_for_bit():
    """Handover section 7's cheap regression check, proved on the emitted string:
    scale 1.0 and rate 0.0 make the extra factor exactly 1.0 under IEEE.
    """
    scale, rate = _emitted_constants(drive_replacement(DT, T, float("inf")))
    assert (scale, rate) == (1.0, 0.0)
    np.testing.assert_array_equal(shaping_weights(T, DT, float("inf")), np.ones(T))


def test_generated_c_obeys_the_kernel_rules():
    code = drive_replacement(DT, T, float(T))
    assert code.endswith(";"), "a missing semicolon surfaces as an unrelated CUDA error"
    assert "{" not in code and "}" not in code, "no braces, so no f-string escaping trap"
    assert "float" not in code and "double" not in code, "use scalar, never a concrete type"
    scale, rate = _emitted_constants(code)
    #every literal is a float, so no C integer division silently zeroes the weight
    assert scale != 0.0 and rate != 0.0
    #the source it replaces must survive verbatim inside the replacement
    assert "error / (TauM * num_batch * 45.0);" in code
    assert drive_source(DT, T) == "drive = error / (TauM * num_batch * 45.0);"


def test_generated_c_matches_the_numpy_reference():
    for timesteps, dt, tau in ((5, 1.0, 5.0), (45, 1.0, 45.0), (21, 1.0, 3.0)):
        scale, rate = _emitted_constants(drive_replacement(dt, timesteps, tau))
        assert (scale, rate) == weight_constants(timesteps, dt, tau)
        back_t = np.arange(timesteps) * dt   #backT takes exactly these values
        np.testing.assert_allclose(scale * np.exp(-rate * back_t),
                                   shaping_weights(timesteps, dt, tau), rtol=1e-12)


def _replay_sim_code_build(dt: float, timesteps: int) -> str:
    """Sim code as ``build_neuron_model`` assembles it for MSE on a LeakyIntegrate.

    A pinned copy of ``event_prop_compiler.py:684-855`` plus ``utils/model.py:156-168``
    (prepend/append are ``textwrap.dedent`` then concatenate). It exists so the string
    surgery can be checked from the plain .venv; the gpu suite checks this copy is still
    faithful, and the compiler subclass raises at compile time if it is not.
    """
    from textwrap import dedent

    code = "V += 1;\n"   #stand-in for the original leaky-integrator sim code

    def prepend(text):
        nonlocal code
        code = dedent(text) + "\n" + code

    def append(text):
        nonlocal code
        code = code + "\n" + dedent(text) + "\n"

    prepend(f"""
                    LambdaI = drive + ((LambdaI - drive) * Beta) + (A * (LambdaV - drive) * (Alpha - Beta));
                    LambdaV = drive + ((LambdaV - drive) * Alpha);
                    """)
    prepend(f"""
                                const int ringOffset = (batch * num_neurons * {2 * timesteps}) + (id * {2 * timesteps});
                                if (Trial > 0) {{
                                    RingReadOffset--;
                                    const scalar error = RingOutputLossTerm[ringOffset + RingReadOffset];
                                    drive = error / (TauM * num_batch * {dt * timesteps});
                                }}
                                """)
    append(f"""
                                const unsigned int timestep = (int)round(t / dt);
                                RingOutputLossTerm[ringOffset + RingWriteOffset] = YTrue[index] - V;
                                RingWriteOffset++;
                                """)
    prepend(f"""
                    const float backT = {timesteps * dt} - t - dt;

                    // Backward pass
                    scalar drive = 0.0;
                    """)
    return code


def test_replacement_lands_once_and_backt_is_in_scope():
    """dedent does not disturb the match, and the compiler's last prepend puts backT's
    declaration ahead of the drive assignment we rewrite.
    """
    code = _replay_sim_code_build(DT, T)
    assert code.count(drive_source(DT, T)) == 1
    shaped = code.replace(drive_source(DT, T), drive_replacement(DT, T, float(T)))
    assert shaped.index("const float backT") < shaped.index("drive = (")
    #and still inside the Trial > 0 guard, which the backward pass needs on trial 0
    guard = shaped.index("if (Trial > 0)")
    assert guard < shaped.index("drive = (") < shaped.index("}", guard)


@pytest.mark.parametrize("tau", [0.0, -1.0, float("nan")])
def test_bad_tau_raises_on_the_host_not_in_cuda(tau):
    with pytest.raises(ValueError):
        shaping_weights(T, DT, tau)


def test_config_defaults_off_and_validates():
    assert ExperimentConfig().loss_shaping is False
    assert ExperimentConfig().loss_shaping_tau is None
    assert ExperimentConfig(loss_shaping=True, loss_shaping_tau=10.0).loss_shaping
    with pytest.raises(ValueError):
        ExperimentConfig(loss_shaping_tau=0.0)
    with pytest.raises(ValueError, match="EventProp"):
        ExperimentConfig(loss_shaping=True, algorithm="eprop")


def test_checkpoint_dir_separates_shaped_from_unshaped():
    """D42: without this a shaped run overwrites the unshaped one and both report it."""
    from models.snn import SnnForecaster

    plain = SnnForecaster(ExperimentConfig(model="snn"))._checkpoint_dir()
    shaped = SnnForecaster(
        ExperimentConfig(model="snn", loss_shaping=True))._checkpoint_dir()
    retuned = SnnForecaster(
        ExperimentConfig(model="snn", loss_shaping=True,
                         loss_shaping_tau=10.0))._checkpoint_dir()
    assert len({plain, shaped, retuned}) == 3


#---------------------------------------------------------------------------------------
#The readout window, D85. Same file as loss shaping because both rewrite the same line.
#---------------------------------------------------------------------------------------

WINDOW = (36.0, 45.0)   #the final 20% of the L=45 trial, the agreed default


def _emitted_window(code: str) -> tuple[float, float]:
    """``(window_start, window_end)`` parsed back out of the generated guard."""
    m = re.search(r"if \(backT >= ([-\d.e+]+) && backT < ([-\d.e+]+)\)", code)
    assert m, f"generated code carries no window guard: {code!r}"
    return float(m.group(1)), float(m.group(2))


def test_window_guard_admits_exactly_the_scored_timesteps():
    """The load-bearing window test, and the counterpart of the shaping one above.

    It does not ask ``scored_timesteps`` to agree with itself. It replays mlGeNN's ring
    arithmetic, applies the bounds *parsed back out of the generated C* to the backT the
    kernel would actually see, and checks the forward steps that survive are the ones
    the host function names. A guard written on the kernel's own clock instead of backT
    fails here, having passed every smoke test.
    """
    start, end = _emitted_window(windowed_drive_replacement(DT, T, WINDOW))
    replay = _replay_backward_pass(T)

    admitted = set()
    for back_t, entry in replay[1]:
        if start <= back_t < end:
            _, k = entry
            admitted.add(k)

    assert admitted == set(scored_timesteps(T, DT, WINDOW).tolist())


def test_window_scores_the_final_timestep_the_readout_reports():
    """mlGeNN 3.x does not, and that is a bug rather than a convention.

    Its guard reads ``t <= T - window_start && t > T - window_end`` on the kernel clock,
    which admits ``window_start - dt <= k*dt < window_end - dt``: one step early at both
    ends, so the last forward step is never scored even when the window ends at the
    trial end. ``Var.get_readout`` reports exactly that step as the prediction, so
    upstream trains on a set excluding the quantity it predicts. Ours includes it.
    """
    scored = scored_timesteps(T, DT, WINDOW)
    assert T - 1 in scored, "the predicted timestep must receive gradient"

    upstream = [k for k in range(T)
                if WINDOW[0] - DT <= k * DT < WINDOW[1] - DT]
    assert T - 1 not in upstream, "this test is only meaningful if upstream differs"
    assert scored.tolist() != upstream


def test_full_trial_window_is_the_unwindowed_drive():
    """The equivalence the gpu gradient check confirms on real gradients.

    A window spanning the whole trial scores every timestep and keeps the trial-length
    denominator, so it can only differ from the unwindowed line by the guard around it.
    """
    full = windowed_drive_replacement(DT, T, (0.0, T * DT))
    assert (scored_timesteps(T, DT, (0.0, T * DT)).tolist()
            == scored_timesteps(T, DT, None).tolist())
    assert drive_source(DT, T) in full
    assert _emitted_window(full) == (0.0, T * DT)


def test_window_denominator_is_the_window_not_the_trial():
    """Keeping the trial length would cut gradient mass by the window fraction, which
    is a learning-rate change in disguise and would confound the arm.
    """
    code = windowed_drive_replacement(DT, T, WINDOW)
    span = WINDOW[1] - WINDOW[0]
    assert f"(TauM * num_batch * {span})" in code
    assert f"(TauM * num_batch * {DT * T})" not in code


def test_window_and_shaping_compose():
    """Both patches on one line: the guard wraps the weighted assignment, and the
    weight is still read at backT rather than the kernel clock.
    """
    code = windowed_drive_replacement(DT, T, WINDOW, tau=float(T))
    assert _emitted_window(code) == WINDOW
    scale, rate = _emitted_constants(code)
    assert rate == pytest.approx(1.0 / T)
    #mean 1 over the scored steps only, so shaping redistributes inside the window
    #rather than rescaling what the window already selected
    scored = scored_timesteps(T, DT, WINDOW)
    assert shaping_weights(T, DT, float(T), WINDOW)[scored].mean() == pytest.approx(1.0)


def test_shaping_normalisation_follows_the_window():
    """Without this the two features would compound into a silent learning-rate cut."""
    unwindowed = shaping_weights(T, DT, float(T))
    windowed = shaping_weights(T, DT, float(T), WINDOW)
    assert unwindowed.mean() == pytest.approx(1.0)
    assert not np.allclose(unwindowed, windowed)


@pytest.mark.parametrize("window", [
    (45.0, 36.0),      #reversed
    (36.0, 36.0),      #empty
    (-1.0, 45.0),      #starts before the trial
    (36.0, 50.0),      #ends after the trial
    (44.5, 44.9),      #non-empty span that lands between timesteps and scores nothing
])
def test_bad_window_raises_on_the_host_not_in_cuda(window):
    with pytest.raises(ValueError):
        windowed_drive_replacement(DT, T, window)


def test_config_window_defaults_off_and_pins_the_far_edge():
    assert ExperimentConfig().readout_window is False
    assert ExperimentConfig().readout_window_times(DT) is None
    windowed = ExperimentConfig(readout_window=True, input_window=T)
    assert windowed.readout_window_times(DT) == (0.8 * T * DT, T * DT)
    with pytest.raises(ValueError):
        ExperimentConfig(readout_window_fraction=0.0)
    with pytest.raises(ValueError):
        ExperimentConfig(readout_window_fraction=1.5)
    with pytest.raises(ValueError, match="EventProp"):
        ExperimentConfig(readout_window=True, algorithm="eprop")


def test_checkpoint_dir_separates_windowed_from_unwindowed():
    """D79's argument, applied to the window: without it a windowed run overwrites the
    unwindowed one and both report the survivor.
    """
    from models.snn import SnnForecaster

    plain = SnnForecaster(ExperimentConfig(model="snn"))._checkpoint_dir()
    windowed = SnnForecaster(
        ExperimentConfig(model="snn", readout_window=True))._checkpoint_dir()
    narrower = SnnForecaster(
        ExperimentConfig(model="snn", readout_window=True,
                         readout_window_fraction=0.1))._checkpoint_dir()
    assert len({plain, windowed, narrower}) == 3
