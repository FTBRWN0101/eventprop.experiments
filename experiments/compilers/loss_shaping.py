"""Loss shaping for EventProp regression. Contribution #6.

The proposal's loss is

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

THE TRAP (D29). The backward pass runs time reversed. At kernel time ``t`` the ring
buffer hands back the loss term recorded at forward time

    backT = T*dt - t - dt

(declared at ``event_prop_compiler.py:851``; the ring arithmetic that makes it exact is
at ``:697-757`` for the read/write and ``:361-365`` for the between-trial reset). So the
weight must be evaluated at ``backT``. Writing ``exp(-t/tau)`` shapes the *end* of the
trial instead of the start and still trains, so a smoke test will not catch it. mlGeNN's
own classification branch (``:799``) writes the same thing as ``exp(-(1 - t/T))``, which
is the algebraic identity of ``exp(-backT/T)`` up to a constant ``exp(-dt/T)``; using
``backT`` directly is the same shape and says what it means.

Weights are normalised to unit mean, so shaping *redistributes* gradient mass over the
trial rather than rescaling it. Without that, turning shaping on would also cut the
effective learning rate by ``mean(w) ~ 0.64``, and the arm would be confounded.

Nothing here imports ml_genn at module scope: the maths and the generated C are testable
from the plain project .venv.
"""

from __future__ import annotations

import numpy as np


def _validate(timesteps: int, dt: float, tau: float) -> None:
    """Host-side checks. A bad number inside generated CUDA is a wrong answer, not a raise."""
    if timesteps < 1:
        raise ValueError(f"timesteps must be >= 1, got {timesteps}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if not tau > 0.0:   #catches 0, negatives and nan
        raise ValueError(f"loss-shaping tau must be positive, got {tau}")


def weight_constants(timesteps: int, dt: float, tau: float) -> tuple[float, float]:
    """``(scale, rate)`` for ``w(s) = scale * exp(-rate * s)``, s a *forward* time.

    ``scale`` normalises the discrete mean of ``w`` over the trial's timesteps to 1.
    """
    _validate(timesteps, dt, tau)
    rate = 1.0 / tau
    mean = float(np.exp(-rate * dt * np.arange(timesteps)).mean())
    return 1.0 / mean, rate


def shaping_weights(timesteps: int, dt: float, tau: float) -> np.ndarray:
    """Loss weight per *forward* timestep, mean 1. Same expression the kernel evaluates."""
    scale, rate = weight_constants(timesteps, dt, tau)
    return scale * np.exp(-rate * dt * np.arange(timesteps))


def drive_source(dt: float, timesteps: int) -> str:
    """The unweighted MSE drive line mlGeNN emits, ``event_prop_compiler.py:743``."""
    return f"drive = error / (TauM * num_batch * {dt * timesteps});"


def drive_replacement(dt: float, timesteps: int, tau: float) -> str:
    """``drive_source`` with the shaping weight, evaluated at ``backT``.

    Constants are folded in as C floats: no integer division, no braces to escape, one
    extra multiply and one ``exp`` per timestep and no extra array read.
    """
    scale, rate = weight_constants(timesteps, dt, tau)
    return (f"drive = ({scale} * exp(-backT * {rate})) * error "
            f"/ (TauM * num_batch * {dt * timesteps});")


def make_loss_shaped_compiler_cls():
    """Build LossShapedEventPropCompiler. Defined inside a function so importing this
    module does not need the mlgenn env.
    """
    from ml_genn.compilers import EventPropCompiler
    from ml_genn.losses import MeanSquareError

    class LossShapedEventPropCompiler(EventPropCompiler):
        """EventProp with an exponentially weighted per-timestep regression loss."""

        def __init__(self, *args, loss_shaping_tau: float, **kwargs):
            super().__init__(*args, **kwargs)
            if not self.per_timestep_loss:
                #the trial-loss branch raises for MSE anyway (:844), and the line we
                #rewrite only exists on the per-timestep path
                raise ValueError("loss shaping needs per_timestep_loss=True")
            #validate before any of it reaches a string
            weight_constants(self.example_timesteps, self.dt, loss_shaping_tau)
            self.loss_shaping_tau = float(loss_shaping_tau)

        def build_neuron_model(self, pop, model, compile_state):
            model = super().build_neuron_model(pop, model, compile_state)
            #hidden and input populations have no readout and no drive
            if pop.neuron.readout is None:
                return model
            if not isinstance(compile_state.losses[pop], MeanSquareError):
                raise NotImplementedError(
                    "loss shaping is implemented for MeanSquareError outputs only, got "
                    f"{type(compile_state.losses[pop]).__name__}")

            source = drive_source(self.dt, self.example_timesteps)
            found = model.model["sim_code"].count(source)
            if found != 1:
                raise RuntimeError(
                    f"expected exactly one MSE drive line to shape, found {found}. "
                    f"mlGeNN's generated sim code has changed; re-derive the patch "
                    f"against event_prop_compiler.py rather than loosening this check")
            model.replace_sim_code(
                source,
                drive_replacement(self.dt, self.example_timesteps,
                                  self.loss_shaping_tau))
            return model

    return LossShapedEventPropCompiler
