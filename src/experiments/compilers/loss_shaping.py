"""Patches to EventProp's MSE adjoint drive: loss shaping, and the readout window.

Both features rewrite the same one line of mlGeNN's generated sim code, so they live
together and compose rather than fighting over it.

LOSS SHAPING, Contribution #6. The proposal's loss is

    L = (1/T) * integral_0^T (V(t) - y)^2 * exp(-t/tau) dt

i.e. the existing per-timestep MSE with a time-dependent weight inside the integral.
Weighting inside the integral (not applying an outer F to it) means the adjoint drive
just gains a per-timestep factor: d/dV(t) picks up exp(-t/tau) at the *forward* time t
of that loss term. No per-trial scalar, no host round trip.

mlGeNN already has this weight, as ``AvgVarExpWeight``, but it is reachable only from
``SparseCategoricalCrossentropy``: ``event_prop_compiler.py:765`` gates the shaped
readouts behind ``if sce_loss``, and ``:844`` raises for MSE. This module makes the same
weight available to ``MeanSquareError`` by rewriting one line of the generated sim code,
so the mlgenn env is never touched (D29).

THE READOUT WINDOW, D85. The per-timestep loss integrates over the *whole* trial, so at
``L = 45`` the network is driven to emit the answer at ``t = 0``, having seen one day of
a forty-five day window, and for 44 of the 45 timesteps it is being asked for something
no forecaster could supply. The window gates the drive to the end of the trial. Outside
it the drive stays at the ``0.0`` mlGeNN initialises it to, so those timesteps
contribute no gradient.

THE TRAP (D29). The backward pass runs time reversed. At kernel time ``t`` the ring
buffer hands back the loss term recorded at forward time

    backT = T*dt - t - dt

So both the weight and the window must be evaluated at ``backT``. Writing ``exp(-t/tau)``
shapes the *end* of the trial instead of the start and still trains, so a smoke test will
not catch it. mlGeNN's own classification branch (``:799``) writes the same thing as
``exp(-(1 - t/T))``, which is the algebraic identity of ``exp(-backT/T)`` up to a constant
``exp(-dt/T)``; using ``backT`` directly is the same shape and says what it means.

BOTH PRECONDITIONS ARE CHECKED IN THE SOURCE, not inferred from the patch working.
``event_prop_compiler.py:849-855`` prepends, and therefore places *before* the MSE block
at ``:738-745``, exactly this:

    const float backT = {example_timesteps * dt} - t - dt;
    // Backward pass
    scalar drive = 0.0;

so ``backT`` is in scope on the line being rewritten, and ``drive`` is already zero when
a guard declines to assign it. ``backT`` is also declared by the shared
``neuron_backward_pass`` template at ``:188`` for spiking populations.

WINDOW SEMANTICS. Forward timestep ``k`` is scored exactly when

    window_start <= k*dt < window_end

Note this deliberately differs from mlGeNN 3.x, whose own guard
(``readouts/readout.py:back_windowed_readout_code``) reads ``t <= T - window_start &&
t > T - window_end`` on the *kernel* clock and therefore admits
``window_start - dt <= k*dt < window_end - dt``. That is one timestep early at both ends,
and in particular it never scores the final forward timestep even when the window ends at
the trial end. The final timestep is the one ``Var.get_readout`` actually reports as the
prediction, so upstream trains on a set that excludes the quantity it predicts. Writing
the guard here on ``backT`` scores the final timestep, which is the intended behaviour.

NORMALISATION. The denominator changes from the trial length to the window length. Left
as the trial length, turning the window on would also divide total gradient mass by the
window fraction, which is a learning-rate cut wearing a disguise, and the arm would be
confounded. The ``TauM`` factor is kept, unlike 3.x which drops it, because keeping it
preserves the gradient-scale convention every run already on disk was trained under.

Shaping weights are normalised to unit mean over the *scored* timesteps for the same
reason: shaping should redistribute gradient mass over the scored part of the trial
rather than rescale it. Without that, shaping alone would cut the effective learning rate
by ``mean(w) ~ 0.64``, and shaping plus a window would compound the two confounds.

Nothing here imports ml_genn at module scope: the maths and the generated C are testable
from the plain project .venv.
"""

from __future__ import annotations

import numpy as np


def _validate(timesteps: int, dt: float, tau: float | None) -> None:
    """Host-side checks. A bad number inside generated CUDA is a wrong answer, not a raise."""
    if timesteps < 1:
        raise ValueError(f"timesteps must be >= 1, got {timesteps}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if tau is not None and not tau > 0.0:   #catches 0, negatives and nan
        raise ValueError(f"loss-shaping tau must be positive, got {tau}")


def scored_timesteps(timesteps: int, dt: float,
                     window: tuple[float, float] | None = None) -> np.ndarray:
    """Forward timestep indices whose loss term actually drives the adjoint.

    Every timestep when there is no window. With one, the generated guard is
    ``backT >= window_start && backT < window_end`` and ``backT`` is the forward time of
    the term being handled, so index ``k`` is admitted exactly when
    ``window_start <= k*dt < window_end``. This function is the host-side statement of
    that same condition, and the tests hold the two to each other.
    """
    every = np.arange(timesteps)
    if window is None:
        return every
    window_start, window_end = window
    forward = every * dt
    return every[(forward >= window_start) & (forward < window_end)]


def _validate_window(timesteps: int, dt: float,
                     window: tuple[float, float] | None) -> None:
    """The window must be a non-empty span that scores at least one timestep.

    An empty or unreachable window generates a guard that is never true, which trains on
    a zero gradient in complete silence. That is the failure this catches on the host.
    """
    if window is None:
        return
    window_start, window_end = window
    if not 0.0 <= window_start < window_end:
        raise ValueError(
            f"readout window must satisfy 0 <= start < end, got {window}")
    if window_end > timesteps * dt:
        raise ValueError(
            f"readout window {window} ends after the trial, which is "
            f"{timesteps * dt} long at dt={dt}")
    if scored_timesteps(timesteps, dt, window).size == 0:
        raise ValueError(
            f"readout window {window} scores no timestep of a {timesteps}-step "
            f"trial at dt={dt}, so the arm would train on a zero gradient")


def weight_constants(timesteps: int, dt: float, tau: float,
                     window: tuple[float, float] | None = None) -> tuple[float, float]:
    """``(scale, rate)`` for ``w(s) = scale * exp(-rate * s)``, s a *forward* time.

    ``scale`` normalises the discrete mean of ``w`` over the *scored* timesteps to 1,
    which is every timestep of the trial unless a window narrows it.
    """
    _validate(timesteps, dt, tau)
    _validate_window(timesteps, dt, window)
    rate = 1.0 / tau
    scored = scored_timesteps(timesteps, dt, window)
    mean = float(np.exp(-rate * dt * scored).mean())
    return 1.0 / mean, rate


def shaping_weights(timesteps: int, dt: float, tau: float,
                    window: tuple[float, float] | None = None) -> np.ndarray:
    """Loss weight per *forward* timestep. Same expression the kernel evaluates.

    One entry per timestep of the trial, including unscored ones: the kernel computes
    the weight from the time alone and the guard separately decides whether that
    timestep contributes. The mean is 1 over the scored entries, so it is 1 over the
    whole array only when there is no window.
    """
    scale, rate = weight_constants(timesteps, dt, tau, window)
    return scale * np.exp(-rate * dt * np.arange(timesteps))


def drive_source(dt: float, timesteps: int) -> str:
    """The unweighted MSE drive line mlGeNN emits, ``event_prop_compiler.py:743``."""
    return f"drive = error / (TauM * num_batch * {dt * timesteps});"


def _drive_body(dt: float, timesteps: int, tau: float | None,
                window: tuple[float, float] | None) -> str:
    """The assignment itself, before any guard is wrapped around it.

    Constants are folded in as C floats: no integer division, no braces to escape, and
    at most one extra multiply and one ``exp`` per timestep with no extra array read.
    """
    span = dt * timesteps if window is None else window[1] - window[0]
    denominator = f"(TauM * num_batch * {span})"
    if tau is None:
        return f"drive = error / {denominator};"
    scale, rate = weight_constants(timesteps, dt, tau, window)
    return f"drive = ({scale} * exp(-backT * {rate})) * error / {denominator};"


def drive_replacement(dt: float, timesteps: int, tau: float) -> str:
    """``drive_source`` with the shaping weight and no window. Unchanged since D42."""
    _validate(timesteps, dt, tau)
    return _drive_body(dt, timesteps, tau, None)


def windowed_drive_replacement(dt: float, timesteps: int,
                               window: tuple[float, float],
                               tau: float | None = None) -> str:
    """``drive_source`` gated to *window*, optionally also shaped.

    ``tau`` of None means the window alone. The guard is on ``backT``, the forward time,
    for the reason the module docstring gives at length.
    """
    _validate(timesteps, dt, tau)
    _validate_window(timesteps, dt, window)
    window_start, window_end = window
    body = _drive_body(dt, timesteps, tau, window)
    #drive is already 0.0 from event_prop_compiler.py:854, so declining to assign is
    #the same as assigning zero, and costs no extra statement
    return f"if (backT >= {window_start} && backT < {window_end}) {{ {body} }}"


def make_drive_patched_compiler_cls():
    """Build DrivePatchedEventPropCompiler.

    Defined inside a function so importing this module does not need the mlgenn env.
    """
    from ml_genn.compilers import EventPropCompiler
    from ml_genn.losses import MeanSquareError

    class DrivePatchedEventPropCompiler(EventPropCompiler):
        """EventProp with a shaped and/or end-of-trial-windowed regression loss."""

        def __init__(self, *args, loss_shaping_tau: float | None = None,
                     readout_window: tuple[float, float] | None = None, **kwargs):
            super().__init__(*args, **kwargs)
            if loss_shaping_tau is None and readout_window is None:
                raise ValueError(
                    "this compiler exists to patch the drive line; with neither loss "
                    "shaping nor a readout window, use EventPropCompiler directly")
            if not self.per_timestep_loss:
                #the trial-loss branch raises for MSE anyway (:844), and the line this
                #rewrites only exists on the per-timestep path
                raise ValueError("the drive patches need per_timestep_loss=True")
            #validate before any of it reaches a string
            _validate(self.example_timesteps, self.dt, loss_shaping_tau)
            _validate_window(self.example_timesteps, self.dt, readout_window)
            self.loss_shaping_tau = (None if loss_shaping_tau is None
                                     else float(loss_shaping_tau))
            self.readout_window = readout_window

        def build_neuron_model(self, pop, model, compile_state):
            model = super().build_neuron_model(pop, model, compile_state)
            #hidden and input populations have no readout and no drive
            if pop.neuron.readout is None:
                return model
            if not isinstance(compile_state.losses[pop], MeanSquareError):
                raise NotImplementedError(
                    "the drive patches are implemented for MeanSquareError outputs "
                    f"only, got {type(compile_state.losses[pop]).__name__}")

            source = drive_source(self.dt, self.example_timesteps)
            found = model.model["sim_code"].count(source)
            if found != 1:
                raise RuntimeError(
                    f"expected exactly one MSE drive line to patch, found {found}. "
                    f"mlGeNN's generated sim code has changed; re-derive the patch "
                    f"against event_prop_compiler.py rather than loosening this check")

            if self.readout_window is None:
                target = drive_replacement(self.dt, self.example_timesteps,
                                           self.loss_shaping_tau)
            else:
                target = windowed_drive_replacement(
                    self.dt, self.example_timesteps, self.readout_window,
                    self.loss_shaping_tau)
            model.replace_sim_code(source, target)
            return model

    return DrivePatchedEventPropCompiler


def make_loss_shaped_compiler_cls():
    """Backwards-compatible name for the shaping-only path.

    Kept because D42's checkpoints, the gpu tests and the 16 loss-shaping runs on disk
    all refer to it. It is the same class; the window simply defaults to off.
    """
    return make_drive_patched_compiler_cls()
