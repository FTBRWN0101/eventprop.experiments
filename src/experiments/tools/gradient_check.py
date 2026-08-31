"""Numerical gradient check for the patched EventProp adjoint drive. Discharges D74.

`compilers/loss_shaping.py` rewrites one line of mlGeNN's generated sim code, the MSE
adjoint drive. Two features now ride on that rewrite: the exponential loss weight of
Contribution #6, and D85's readout window. Every test the project has of either one
asserts that the patch *changes* the gradient. None asserts that it changes it
*correctly*. D74 says so in as many words, and D88's closing note calls the omission more
overdue than before, because the windowed path moves the same line again.

This tool closes that. It compares the gradient EventProp accumulates against a central
finite difference of the loss, and it is deliberately structured so that a false pass is
hard to produce.

WHY A TOOL AND NOT A TEST. The deliverable is numbers a person has to read: a
correlation, a residual spread, an epsilon sweep, a timebase sweep. pytest swallows
stdout, and a threshold that passes tells you nothing about how much margin it passed
with. So the harness lives here, prints everything, and exits non-zero when a check
fails. `tests/tools/test_gradient_check_gpu.py` then calls the same functions on the
same fixture and asserts against the same constants, so the regression guard and the
report cannot drift apart or disagree about what passing means.

WHAT IS COMPARED, AND WHY IT IS NOT COMPARED DIRECTLY. mlGeNN 2.x has only Adam, so the
usual "one step at a known learning rate recovers the gradient" trick is unavailable:
Adam's first step is approximately `lr * sign(g)` and carries no magnitude. EventProp
accumulates the raw gradient instead, in a per-synapse `Gradient` variable
(`event_prop_compiler.py:102`), and `CompiledNetwork.connection_populations` exposes the
synapse population it lives on, so it can be pulled straight off the device.

The accumulated gradient carries factors of `TauM`, the batch size and the window length.
Matching those by hand would be exactly the kind of unverified claim this tool exists to
avoid, so nothing here reconstructs them. What is tested instead is the relationship that
must hold whatever the global constant is: **one scale factor fits every sampled weight.**
A correct patch gives one constant across the sample. A wrong window breaks the
relationship non-uniformly, because it mixes in timesteps whose errors have a different
sign and magnitude per weight.

Four checks, in ascending order of what they would catch:

1. **Fit.** Every arm admits one scale across the whole sample, with a small residual.
2. **Window.** With a narrow window, the finite difference taken over *only the scored
   timesteps* fits, and the one taken over the *whole trial* does not. That separation is
   the direct evidence that the guard gates the timesteps it claims to. It is then
   generalised: every arm's gradient is fitted against every arm's scored set, and the
   residual matrix has to be diagonally dominant. A whole-trial control on its own is
   weakest at exactly the window D85 ships, because a window ending at the trial end
   scores the timesteps the whole-trial loss is dominated by anyway; the matrix has no
   such blind spot, since an unguarded drive would make every one of its rows constant.
3. **Equivalence.** `readout_window=None` and a full-trial window produce identical
   analytic gradients, and both equal the stock compiler's. Cheapest and strongest
   regression against an off-by-one, and it retro-validates the unwindowed path that
   sixteen runs on disk were trained with.
4. **Composition.** The same comparison with shaping on, with a window on, and with both,
   since all three rewrite the same line.

WHICH WEIGHTS ARE SAMPLED, AND WHY ONLY THOSE. The hidden->output weights. Given the
hidden spike train, the output membrane voltage is *linear* in those weights, so the loss
is quadratic and a central difference is exact rather than merely second-order accurate.
Nothing about the epsilon choice can hide behind truncation error.

The input->hidden and recurrent weights are not sampled, and the reason is a property of
the simulator rather than a convenience. Changing them moves hidden *spike times*, which
the simulation quantises to a timestep. Below the perturbation that tips a spike across a
boundary the loss does not move at all and the difference quotient is exactly zero; at
that perturbation it jumps. There is no epsilon in between. EventProp's spike-time term
is a derivative of the continuous-time system and a discrete-time finite difference
cannot see it, so sampling those weights would produce a disagreement that says nothing
about the patch. `--probe-spiking-weights` measures that staircase and prints it rather
than leaving the claim unevidenced.

TWO FIXTURE DECISIONS THAT ARE NOT ARBITRARY, both established by measurement and both
re-measurable from the switches this tool exposes.

*The synaptic state is zeroed at the start of every trial.* `EventPropCompiler` does not
reset postsynaptic model state between trials. `InferenceCompiler` does, in an explicit
loop over `_psm_reset_vars` at `inference_compiler.py:315-323`; `EventPropCompiler` has
no equivalent. On this fixture the consequence is not a rounding artefact: the first
trial's output voltage runs 0.0, 0.0, 0.044, 0.108 and the second 0.032, 0.055, 0.120,
0.195, a peak difference of 0.55 against a voltage range of 3.6, and the third trial
still differs from the second. The adjoint assumes each trial starts from rest, so a
harness that let the carry-over stand would be finite-differencing a loss the analytic
gradient is not the gradient of. `_make_callbacks` triggers mlGeNN's own `ZeroOutPost`
custom update at every batch begin, after which all four trials are bit identical and the
whole measurement is reproducible across `train()` calls.

*The trial runs at dt/tau_syn = 0.25, not 1.0.* EventProp differentiates the
continuous-time system; the finite difference measures the discrete-time simulation. The
two agree only as dt/tau goes to zero, and the gap is a floor under every residual here.
Measured on this fixture at fixed trial length, relative residual against dt/tau_syn:
1.00 -> 1.34e-2, 0.50 -> 3.37e-3, 0.25 -> 1.22e-3, 0.125 -> 9.2e-4. That is first order
in dt at the coarse end and then flattens onto a floor near 1e-3. The epsilon sweep shows
the floor is not the perturbation size, since it is unchanged from eps 3e-1 down to
3e-4, and this tool does not attribute it further; what matters here is that it is a
floor and that every arm sits on it. At the production timebase it would swamp the effect
being tested, so the fixture uses forty timesteps of 0.25 rather than ten of 1.0.
`--timebase-sweep` re-measures the curve on every run, so the claim is evidence in the
output rather than a comment. It is also a result in its own right, and not a comfortable
one: `models/snn.py` trains at dt = tau_syn = 1.0, where EventProp's gradient and the
gradient of the loss it is descending differ by more than a percent.

THE NEGATIVE CONTROL. A harness that has only ever been seen to pass is not evidence,
which is the whole of what D74 records. `--negative-control` builds the compiler on
mlGeNN 3.x's own window semantics, one timestep early at both ends, while fitting against
the window that was asked for, and requires the checks above to reject both faults. They
do, and it is worth knowing which check does the work: the narrow fault still fits its
claimed scored set at 4.9e-3, inside check 1's limit of 2e-2, so **check 1 alone would
pass an off-by-one**. Check 2 catches it, the separation dropping from x9.2 to x2.6. The
full-trial fault is caught outright by check 3.

THE BATCH LAYOUT. EventProp defers each trial's backward pass into the *next* trial's
forward pass (mlGeNN issue #101, confirmed in D74), so a single batch never accumulates a
gradient at all. Four identical batches are presented and the gradient is read at the
last one, which is the gradient of the second to last trial; with the synaptic state
zeroed those two trials are identical, and the tool asserts that they are. At
`batch_size=1` mlGeNN gives Adam the gradient-*zeroing* variant of its update
(`event_prop_compiler.py:1651-1654`) and `GradientLearn` is a base callback, so it runs
before any callback passed to `train()`. The probe therefore reads on the last
`on_timestep_end` of the batch rather than on `on_batch_end`, which would read zeros.
`Adam(0.0)` keeps the weights still while all this happens, and the tool asserts that too.

    run_gradient_check.bat
    run_gradient_check.bat --all
    run_gradient_check.bat --epsilon-sweep --timebase-sweep --probe-spiking-weights
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from compilers.loss_shaping import (  # noqa: E402
    make_drive_patched_compiler_cls, scored_timesteps, shaping_weights)

logger = logging.getLogger("gradient_check")

#Fixture. Small enough that the whole check is seconds, structured enough that the
#twenty-four sampled gradients are twenty-four distinct numbers rather than a handful of
#repeats: five input neurons on unequal spike patterns, and per-neuron tau_mem, which is
#also what models/snn.py does. Without the heterogeneity the hidden neurons are scalar
#multiples of one another and the sample collapses onto its own spike count.
TIMESTEPS = 40
DT = 0.25
NUM_INPUT = 5
NUM_HIDDEN = 24
V_THRESH = 0.61
TAU_SYN = 1.0
TAU_MEM = 3.0        #per-neuron sampling scales from this, as in snn.py
#The output integrator has to leak inside the trial or check 2 has nothing to separate.
#With tau_mem_out well above the trial length the output is very nearly a running sum,
#every timestep's voltage carries the whole history, and a whole-trial finite difference
#comes out as an almost exact scaled copy of an end-of-trial windowed one -- which a fit
#with a free scale then absorbs. Measured on this fixture, end-of-trial separation
#against tau_mem_out: 20 -> x5.6, 9 -> x9.2, 4.5 -> x10.2, 2.5 -> x9.4. Leaking harder
#than that stops helping and starts raising the early-window arm's own residual, because
#its gradient comes to rest on a handful of spikes: 4.5 -> 1.3e-2, 2.5 -> 2.3e-2 against
#a floor of 1.3e-3. 9.0 is where every arm sits on the floor and the windows still
#separate. It also puts tau_mem_out/trial at 0.9, between the 2.0 of the first fixture
#and the 0.44 that models/snn.py actually trains at.
TAU_MEM_OUT = 9.0
W_IN_MEAN, W_IN_SD = 1.15, 0.35
W_REC_SD = 0.15
W_OUT_SD = 0.6
INPUT_SPIKE_RATE = 0.35   #probability that a given input neuron fires on a given step

#Four batches of one example: one to prime the ring buffer, one whose backward pass is
#read, and two of margin so the reading is demonstrably at a fixed point.
BATCHES = 4

NETWORK_SEED = 7
SPIKE_SEED = 11
GENN_SEED = 1

#The target sweeps from above the trajectory to below it, so the early and late errors
#carry opposite signs. A constant target would leave a full-trial finite difference
#looking like a scaled version of a windowed one, and check 2 would separate the two only
#weakly. This is the same reasoning behind the non-constant target in
#tests/compilers/test_loss_shaping_gpu.py.
TARGET_START, TARGET_END = 1.5, -0.5

#D85's default: the final fifth of the trial. Kept in step with
#core/config.READOUT_WINDOW_FRACTION so the arm under test is the arm that would run.
NARROW_FRACTION = 0.2
#Two windows that D85 would never configure, because D85 pins window_end to the trial
#end. They are here because the production window only ever exercises the guard's LOWER
#bound: with window_end at the trial end, the upper comparison is true for every backT
#there is. An early window tests the upper bound and a middle one tests both, so between
#them the guard is exercised in full rather than half.
EARLY_WINDOW_FRACTION = 0.25
MIDDLE_WINDOW = (0.4, 0.6)
#exp(-t/tau) over a trial's worth of time, which is what config.loss_shaping_tau=None
#means once models/snn.py converts it
SHAPING_TAU = TIMESTEPS * DT

#Central differences on a quadratic are exact, so the useful half of this sweep is the
#small end, where float32 cancellation in the recorded voltage starts to dominate.
EPSILONS: tuple[float, ...] = (3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
DEFAULT_EPSILON = 1e-2

#Trial length is held fixed while dt varies, so every point scores the same span of time
#and only the discretisation changes.
TIMEBASE_SWEEP: tuple[tuple[int, float], ...] = ((10, 1.0), (20, 0.5), (40, 0.25),
                                                 (80, 0.125))

#Pass thresholds. The stock compiler's own residual is printed beside every arm, so
#these are a floor to trip on rather than the standard the arms are judged by: the
#measured floor is ~1e-3, so 2e-2 is more than an order of magnitude of headroom and a
#breach means something structural, not a slightly worse fit.
MIN_CORRELATION = 0.999
MAX_RELATIVE_RESIDUAL = 2e-2
#check 2 asks for a separation, not a threshold on the wrong answer. The null it is
#testing against is a guard that does nothing, which would put the arm's gradient exactly
#on the whole-trial finite difference and give a separation of 1.0. Measured separations
#run from x9.2 for the end-of-trial window up to x36 for the early one; 5.0 is a floor
#that every working arm clears with margin and a dead guard cannot come near.
MIN_WINDOW_SEPARATION = 5.0


@dataclass(frozen=True)
class Arm:
    """One compiled configuration of the drive line.

    ``tau`` None is no loss shaping, ``window`` None is no readout window, and both None
    is the stock ``EventPropCompiler`` rather than the patched subclass -- which is the
    point of that arm, since it is the baseline every other arm is read against.

    ``scored`` and ``loss_weights`` describe the module's own timebase, not an arbitrary
    fixture's. Every arm compared against another has to be on one timebase for the
    comparison to mean anything, so the alternative would be an arm that could silently
    be built against the wrong one. :func:`report_timebase_sweep` varies dt and asks
    ``scored_timesteps`` directly for that reason.
    """

    name: str
    tau: float | None = None
    window: tuple[float, float] | None = None
    stock: bool = False

    @property
    def scored(self) -> np.ndarray:
        return scored_timesteps(TIMESTEPS, DT, self.window)

    @property
    def loss_weights(self) -> np.ndarray | None:
        """Per-timestep loss weight, or None when the loss is unshaped."""
        if self.tau is None:
            return None
        return shaping_weights(TIMESTEPS, DT, self.tau, self.window)


@dataclass
class Fit:
    """One scale factor fitted across the whole sample, and how well it holds."""

    n: int
    scale: float
    correlation: float
    relative_residual: float   #RMS residual over RMS analytic gradient
    worst_relative: float      #largest single residual over RMS analytic gradient

    def line(self) -> str:
        return (f"n={self.n:>3}  r={self.correlation: .9f}  scale={self.scale: .6e}  "
                f"resid_rms={self.relative_residual:.3e}  "
                f"resid_max={self.worst_relative:.3e}")


@dataclass
class ForwardSweep:
    """Everything the finite difference needs, collected once for all arms.

    The patches rewrite the *backward* drive only, so the forward pass is identical in
    every arm and one sweep serves all of them. That is asserted rather than assumed:
    :func:`check_forward_pass_is_arm_independent` compares each arm's recorded voltage
    against this one and requires them bit identical.
    """

    epsilon: float
    voltage: np.ndarray        #(timesteps,) unperturbed output voltage
    plus: np.ndarray           #(sample, timesteps)
    minus: np.ndarray          #(sample, timesteps)
    denominator: np.ndarray    #(sample,) the perturbation actually stored, as float32
    moved_spike_train: int


def target_trace(timesteps: int = TIMESTEPS) -> np.ndarray:
    """Per-timestep regression target. One value per timestep, as the loss demands."""
    return np.linspace(TARGET_START, TARGET_END, timesteps, dtype=np.float32)


def loss(voltage: np.ndarray, scored: np.ndarray,
         weights: np.ndarray | None = None) -> float:
    """Weighted squared error over the scored timesteps, in float64.

    The sum is over the scored set alone, because the loss the analytic gradient belongs
    to is the one the guard admits. Passing a wider set is how check 2 produces its
    negative control, so this function takes the set rather than deriving it.
    """
    target = target_trace(voltage.size)
    error = voltage.astype(np.float64)[scored] - target[scored]
    if weights is None:
        return float(np.sum(error ** 2))
    return float(np.sum(weights[scored] * error ** 2))


def fit_one_scale(analytic: np.ndarray, numerical: np.ndarray) -> Fit:
    """Least squares through the origin, plus the correlation and the residual spread.

    Through the origin on purpose. An intercept would absorb a constant offset between
    the two gradients, and a constant offset is precisely one of the failures worth
    catching: a window that admits one timestep too many adds a term that is nearly the
    same for every weight.
    """
    if analytic.size != numerical.size:
        raise ValueError(f"sample sizes differ, {analytic.size} and {numerical.size}")
    if analytic.size < 2:
        raise ValueError(f"a scale fitted to {analytic.size} point(s) says nothing")
    denominator = float(np.dot(numerical, numerical))
    if denominator == 0.0:
        raise ValueError("every finite difference is zero, so no scale is identifiable")
    scale = float(np.dot(analytic, numerical) / denominator)
    residual = analytic - scale * numerical
    rms = float(np.sqrt(np.mean(analytic ** 2)))
    return Fit(n=int(analytic.size), scale=scale,
               correlation=float(np.corrcoef(analytic, numerical)[0, 1]),
               relative_residual=float(np.sqrt(np.mean(residual ** 2)) / rms),
               worst_relative=float(np.max(np.abs(residual)) / rms))


def _make_callbacks():
    """Build the two probe callbacks. Inside a function so importing this module does
    not need the mlgenn env, following models/snn.py.
    """
    from ml_genn.callbacks import Callback

    class CleanSynapseState(Callback):
        """Zero every postsynaptic current before each trial.

        EventProp's compiled network resets neuron variables between trials but not
        postsynaptic model state, so without this the current left at the end of trial
        n-1 leaks into trial n and the forward pass is no longer the one the adjoint
        differentiates. ``ZeroOutPost`` is mlGeNN's own custom update, registered by
        ``compiler.add_out_post_zero_custom_update``; the base callbacks already fire it
        once per trial, at the beginning of the last timestep, which leaves exactly the
        last timestep's spikes to carry over. Firing it again at batch begin removes
        that remainder using the library's own operation rather than a hand-written one.
        """

        def set_params(self, compiled_network, **kwargs):
            self._compiled_network = compiled_network

        def on_batch_begin(self, batch):
            self._compiled_network.custom_update("ZeroOutPost")

    class GradientProbe(Callback):
        """Pull the accumulated per-synapse gradient off the device.

        The sink is caller owned for the reason ``EaseInSchedule`` documents in
        models/snn.py: mlGeNN shallow copies every callback per ``train()`` call
        (``utils/module.py:50``), so anything written to ``self`` is written to a copy
        the caller never sees.

        Reading on the last ``on_timestep_end`` rather than on ``on_batch_end`` is not a
        stylistic choice. At ``batch_size=1`` mlGeNN builds Adam with its gradient
        zeroing variant, and ``GradientLearn`` is a base callback, so by the time any
        user callback's ``on_batch_end`` runs the gradient is already zero.
        """

        def __init__(self, connections, sink, batch, last_timestep):
            self._connections = connections
            self._sink = sink
            self._batch = batch
            self._last_timestep = last_timestep
            self._current = None

        def set_params(self, compiled_network, **kwargs):
            self._compiled_network = compiled_network

        def on_batch_begin(self, batch):
            self._current = batch

        def on_timestep_end(self, timestep):
            if timestep != self._last_timestep or self._current != self._batch:
                return
            from ml_genn.utils.network import get_underlying_conn
            for name, connection in self._connections.items():
                variable = self._compiled_network.connection_populations[
                    get_underlying_conn(connection)].vars["Gradient"]
                variable.pull_from_device()
                self._sink[(self._current, name)] = variable.view.flatten().copy()

    return CleanSynapseState, GradientProbe


@dataclass
class Fixture:
    """A built network plus the one example and one target it is measured on."""

    network: object
    populations: dict
    connections: dict
    inputs: dict
    targets: dict
    timesteps: int
    dt: float
    tau_syn: float
    hidden_tau_mem: np.ndarray
    input_spikes: tuple[np.ndarray, np.ndarray]


def build_fixture(timesteps: int = TIMESTEPS, dt: float = DT,
                  tau_syn: float = TAU_SYN) -> Fixture:
    """Three-layer recurrent LIF network, one fixed example, one fixed target.

    Shaped after models/snn.py rather than invented: spike input, a recurrent LIF hidden
    layer with heterogeneous ``tau_mem``, and a single non-spiking leaky integrator read
    out through ``Var``. Everything is seeded and nothing in the graph draws at run time.
    """
    from ml_genn import Connection, Network, Population
    from ml_genn.compilers.event_prop_compiler import default_params
    from ml_genn.connectivity import Dense
    from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
    from ml_genn.synapses import Exponential
    from ml_genn.utils.data import preprocess_spikes

    rng = np.random.default_rng(NETWORK_SEED)
    tau_mem = np.clip(rng.gamma(3.0, TAU_MEM / 3.0, NUM_HIDDEN),
                      TAU_MEM, 3.0 * TAU_MEM)
    #the adjoint's A constant is TauM / (TauSyn - TauM) for the spiking populations
    #(event_prop_compiler.py:1132) and TauM / (TauM - TauSyn) for the output
    #(``:672``), so a neuron whose membrane and synaptic constants coincide divides by
    #zero, and the nan then spreads silently through the whole population: the network
    #emits no spikes and every gradient reads 0.0, which looks like a dead fixture
    #rather than a bad constant. Caught here, where the cause is legible.
    coincide = [f"{name} tau_mem" for name, value in
                (("a hidden", float(np.abs(tau_mem - tau_syn).min())),
                 ("the output", abs(TAU_MEM_OUT - tau_syn)))
                if value <= 1e-6]
    if coincide:
        raise ValueError(
            f"{' and '.join(coincide)} coincides with tau_syn={tau_syn}, which makes "
            f"EventProp's A constant infinite")

    network = Network(default_params)
    with network:
        source = Population(SpikeInput(max_spikes=NUM_INPUT * timesteps),
                            NUM_INPUT, name="input")
        hidden = Population(
            LeakyIntegrateFire(v_thresh=V_THRESH, tau_mem=tau_mem, tau_refrac=None),
            NUM_HIDDEN, record_spikes=True, name="hidden")
        output = Population(LeakyIntegrate(tau_mem=TAU_MEM_OUT, readout="var"),
                            1, name="output")
        connections = {
            "input_hidden": Connection(
                source, hidden,
                Dense(rng.normal(W_IN_MEAN, W_IN_SD, (NUM_INPUT, NUM_HIDDEN))),
                Exponential(tau_syn)),
            "recurrent": Connection(
                hidden, hidden,
                Dense(rng.normal(0.0, W_REC_SD, (NUM_HIDDEN, NUM_HIDDEN))),
                Exponential(tau_syn)),
            "hidden_output": Connection(
                hidden, output,
                Dense(rng.normal(0.0, W_OUT_SD, (NUM_HIDDEN, 1))),
                Exponential(tau_syn)),
        }

    spike_rng = np.random.default_rng(SPIKE_SEED)
    times, ids = [], []
    for neuron in range(NUM_INPUT):
        steps = np.flatnonzero(spike_rng.random(timesteps) < INPUT_SPIKE_RATE)
        times.append(steps.astype(np.float32) * dt)
        ids.append(np.full(steps.size, neuron, dtype=np.int32))
    times = np.concatenate(times)
    ids = np.concatenate(ids)
    example = preprocess_spikes(times, ids, NUM_INPUT)

    target = np.tile(target_trace(timesteps).reshape(1, timesteps, 1), (BATCHES, 1, 1))
    return Fixture(
        network=network,
        populations={"input": source, "hidden": hidden, "output": output},
        connections=connections,
        inputs={source: [example] * BATCHES},
        targets={output: target},
        timesteps=timesteps, dt=dt, tau_syn=tau_syn, hidden_tau_mem=tau_mem,
        input_spikes=(times, ids))


def compile_arm(fixture: Fixture, arm: Arm):
    """Compile one configuration of the drive line onto the fixture's network.

    ``Adam(0.0)`` because the gradient has to be read at a known weight vector. The
    optimiser still runs, still zeroes the gradient and still advances its moments; it
    just multiplies its step by zero. :func:`measure` asserts afterwards that the
    weights did not move, so this is checked rather than trusted.
    """
    from ml_genn.compilers import EventPropCompiler
    from ml_genn.optimisers import Adam

    kwargs = dict(example_timesteps=fixture.timesteps, losses="mean_square_error",
                  optimiser=Adam(0.0), batch_size=1, dt=fixture.dt,
                  per_timestep_loss=True, rng_seed=GENN_SEED,
                  backend="single_threaded_cpu")
    if arm.stock:
        compiler = EventPropCompiler(**kwargs)
    else:
        compiler = make_drive_patched_compiler_cls()(
            **kwargs, loss_shaping_tau=arm.tau, readout_window=arm.window)
    #the generated code is cached per directory name, so each arm needs its own or the
    #second arm silently runs the first arm's kernel
    compiled = compiler.compile(fixture.network, name=f"gradcheck_{arm.name}")
    compiled.num_recording_timesteps = fixture.timesteps
    return compiled


@dataclass
class Measurement:
    """One `train()` call's worth of readings at a fixed weight vector."""

    voltage: np.ndarray            #(timesteps,) output V over the measured trial
    gradient: np.ndarray           #(hidden,) hidden->output gradient
    spike_counts: np.ndarray       #(hidden,) hidden spikes on the measured trial
    settled: bool                  #was the measured trial at the fixed point?
    weights_moved: bool = field(default=False)


def measure(compiled, fixture: Fixture) -> Measurement:
    """Run the four batches and read the voltage, the gradient and the spike counts.

    The gradient read at batch ``BATCHES-1`` is the gradient of trial ``BATCHES-2``, so
    the voltage is read one batch earlier to describe the same trial. ``settled`` says
    whether the two trials before the reading were identical, which is the condition
    under which that pairing is exact; it is reported rather than assumed because a
    change to :class:`CleanSynapseState` would break it silently.
    """
    from ml_genn.callbacks import SpikeRecorder, VarRecorder
    from ml_genn.utils.network import get_underlying_conn

    clean_synapse_state, gradient_probe = _make_callbacks()
    sink: dict = {}
    output = fixture.populations["output"]
    hidden = fixture.populations["hidden"]
    connection = fixture.connections["hidden_output"]

    weights = compiled.connection_populations[get_underlying_conn(connection)].vars["g"]
    weights.pull_from_device()
    before = weights.view.flatten().copy()

    _, data = compiled.train(
        fixture.inputs, fixture.targets, num_epochs=1, shuffle=False,
        callbacks=[clean_synapse_state(),
                   gradient_probe({"hidden_output": connection}, sink,
                                  BATCHES - 1, fixture.timesteps - 1),
                   VarRecorder(output, "v", key="voltage"),
                   SpikeRecorder(hidden, key="spikes", record_counts=True)])

    voltages = [np.asarray(batch).reshape(fixture.timesteps) for batch in data["voltage"]]
    weights.pull_from_device()
    return Measurement(
        voltage=voltages[BATCHES - 2],
        gradient=sink[(BATCHES - 1, "hidden_output")],
        spike_counts=np.asarray(data["spikes"])[BATCHES - 2],
        settled=bool(np.array_equal(voltages[BATCHES - 2], voltages[BATCHES - 3])),
        weights_moved=not np.array_equal(before, weights.view.flatten()))


def collect_forward_sweep(compiled, fixture: Fixture, sample: np.ndarray,
                          epsilon: float) -> ForwardSweep:
    """Record the output voltage at ``w +- epsilon`` for every sampled weight.

    The perturbation is read back out of the device-side array rather than assumed to be
    ``2 * epsilon``: GeNN's scalar type is float32, so what is stored is
    ``float32(w + epsilon)``, and at the small end of the epsilon sweep the difference
    between that and ``w + epsilon`` is a measurable fraction of the step. Dividing by
    what was actually stored removes that error instead of attributing it to the patch.
    """
    from ml_genn.utils.network import get_underlying_conn

    weights = compiled.connection_populations[
        get_underlying_conn(fixture.connections["hidden_output"])].vars["g"]
    weights.pull_from_device()
    original = weights.view.flatten().copy()

    baseline = measure(compiled, fixture)
    plus = np.zeros((sample.size, fixture.timesteps))
    minus = np.zeros((sample.size, fixture.timesteps))
    denominator = np.zeros(sample.size)
    moved = 0
    try:
        for row, index in enumerate(sample):
            stored = {}
            for sign in (1, -1):
                weights.view.flat[:] = original
                weights.view.flat[index] = original[index] + sign * epsilon
                weights.push_to_device()
                stored[sign] = float(weights.view.flat[index])
                perturbed = measure(compiled, fixture)
                if not np.array_equal(perturbed.spike_counts, baseline.spike_counts):
                    moved += 1
                (plus if sign == 1 else minus)[row] = perturbed.voltage
            denominator[row] = stored[1] - stored[-1]
    finally:
        weights.view.flat[:] = original
        weights.push_to_device()

    #below about 1e-7 relative, float32 rounds w+epsilon back onto w and the difference
    #quotient divides by zero. That produces infinities that would propagate into every
    #fit as a nan rather than as an error, so it is caught where the cause is legible.
    if not np.all(denominator):
        collapsed = np.flatnonzero(denominator == 0.0)
        raise ValueError(
            f"epsilon {epsilon:g} is below float32 resolution for {collapsed.size} of "
            f"{sample.size} sampled weights, so w+eps and w-eps are the same number "
            f"and no difference quotient exists")

    return ForwardSweep(epsilon=epsilon, voltage=baseline.voltage, plus=plus,
                        minus=minus, denominator=denominator, moved_spike_train=moved)


def finite_difference(sweep: ForwardSweep, scored: np.ndarray,
                      weights: np.ndarray | None = None) -> np.ndarray:
    """`(L(w+eps) - L(w-eps)) / (2*eps)` for every sampled weight, over *scored*.

    The scored set and the loss weights are arguments rather than fixture state so that
    one recorded sweep answers every arm, and so that check 2 can ask the same sweep for
    the windowed loss and the whole-trial loss without re-running anything.
    """
    return np.array([
        (loss(sweep.plus[row], scored, weights) - loss(sweep.minus[row], scored, weights))
        / sweep.denominator[row]
        for row in range(sweep.denominator.size)])


def live_sample(measurement: Measurement) -> np.ndarray:
    """Indices of hidden->output weights whose gradient is not identically zero.

    A hidden neuron that never spikes contributes nothing to the output and nothing to
    the gradient, so both sides of the comparison are exactly zero. Keeping those points
    would inflate the correlation with pairs that agree for a reason that has nothing to
    do with the patch.
    """
    return np.flatnonzero(measurement.gradient != 0.0)


def check_forward_pass_is_arm_independent(voltages: dict) -> bool:
    """Every arm must produce the same forward pass, or one sweep cannot serve all.

    The patches rewrite the backward drive and touch nothing the forward pass reads, so
    this should hold bit for bit. It is asserted because the alternative is a harness
    whose central efficiency is an assumption.
    """
    reference_name, reference = next(iter(voltages.items()))
    identical = True
    for name, voltage in voltages.items():
        if not np.array_equal(voltage, reference):
            worst = float(np.max(np.abs(voltage - reference)))
            logger.error("[gradcheck] arm %s forward pass differs from %s by %.3e",
                         name, reference_name, worst)
            identical = False
    return identical


def default_arms() -> list[Arm]:
    """The eight configurations, ordered so each adds a single thing to the last.

    There is deliberately no unshaped, unwindowed *patched* arm: the compiler refuses to
    be built that way (``loss_shaping.py:221``), on the grounds that a patch which
    patches nothing should be ``EventPropCompiler``. So the unwindowed side of check 3 is
    the stock compiler for the unshaped pair and ``readout_window=None`` for the shaped
    pair, which between them cover both halves of what that check is for.
    """
    trial_end = TIMESTEPS * DT
    full = (0.0, trial_end)
    narrow = ((1.0 - NARROW_FRACTION) * trial_end, trial_end)
    early = (0.0, EARLY_WINDOW_FRACTION * trial_end)
    middle = (MIDDLE_WINDOW[0] * trial_end, MIDDLE_WINDOW[1] * trial_end)
    return [
        Arm("stock", stock=True),
        Arm("window_full", window=full),
        Arm("window_narrow", window=narrow),
        Arm("window_early", window=early),
        Arm("window_middle", window=middle),
        Arm("shaped", tau=SHAPING_TAU),
        Arm("shaped_window_full", tau=SHAPING_TAU, window=full),
        Arm("shaped_window_narrow", tau=SHAPING_TAU, window=narrow),
    ]


#the unshaped arms whose scored sets are cross-fitted against one another in check 2.
#stock leads because it is the null the rest are read against: it is the whole trial, and
#it is the row a guard that did nothing would collapse every other row onto.
CROSS_FIT_ARMS: tuple[str, ...] = ("stock", "window_narrow", "window_early",
                                   "window_middle")


def cross_fit(analytic: dict, arms: list[Arm], sweep: ForwardSweep,
              names: tuple[str, ...] = CROSS_FIT_ARMS) -> dict:
    """Fit every arm's gradient against every arm's scored set.

    The diagonal is each arm read against the timesteps it claims to score. A guard that
    did nothing would make every row constant, because every arm's gradient would be the
    whole-trial gradient and the scored set would be irrelevant to it. So the property
    that matters is not any single number but *diagonal dominance*: each row's minimum
    sits on its own scored set, and sits well below the rest of the row.

    Only the unshaped arms take part. Shaping multiplies the loss by a fixed per-timestep
    weight, so cross-fitting a shaped gradient against an unshaped scored set would fold
    two differences into one number without saying which of them caused it. The shaped
    arms are covered instead by check 1 and by the whole-trial control above.
    """
    by_name = {arm.name: arm for arm in arms}
    numerical = {name: finite_difference(sweep, by_name[name].scored,
                                         by_name[name].loss_weights)
                 for name in names}
    return {(row, column): fit_one_scale(analytic[row], numerical[column])
            for row in names for column in names}


def run_arms(fixture: Fixture, arms: list[Arm], sweep: ForwardSweep,
             sample: np.ndarray) -> tuple[dict, dict, dict]:
    """Compile every arm, read its gradient, and fit it against the sweep.

    Returns the analytic gradients, the fits and the recorded forward voltages, all
    keyed by arm name.
    """
    analytic, fits, voltages = {}, {}, {}
    for arm in arms:
        started = time.time()
        compiled = compile_arm(fixture, arm)
        with compiled:
            measurement = measure(compiled, fixture)
        if measurement.weights_moved:
            raise RuntimeError(
                f"arm {arm.name} moved its weights during the measurement, so the "
                f"gradient and the finite difference are at different weight vectors")
        if not measurement.settled:
            raise RuntimeError(
                f"arm {arm.name} had not reached a fixed point by the measured trial, "
                f"so the gradient read at batch {BATCHES - 1} is not the gradient of "
                f"the trial the voltage was read from")
        analytic[arm.name] = measurement.gradient[sample]
        voltages[arm.name] = measurement.voltage
        #the sweep already holds one row per *sampled* weight, so its finite differences
        #are in sample order and must not be indexed by weight index a second time
        numerical = finite_difference(sweep, arm.scored, arm.loss_weights)
        fits[arm.name] = fit_one_scale(analytic[arm.name], numerical)
        logger.info("[gradcheck] %-22s scored=%2d/%d  %s  (%.1fs)",
                    arm.name, arm.scored.size, TIMESTEPS, fits[arm.name].line(),
                    time.time() - started)
    return analytic, fits, voltages


def report_epsilon_sweep(fixture: Fixture, compiled, arm: Arm,
                         analytic: np.ndarray, sample: np.ndarray,
                         epsilons: tuple[float, ...] = EPSILONS) -> dict:
    """Fit *arm* at each epsilon, so the stable regime is shown rather than asserted.

    On this sample the loss is quadratic in the perturbed weight, so the difference
    quotient is algebraically exact and the large end of the sweep is flat. The sweep
    still earns its place at the small end, where the float32 voltage the difference is
    taken of starts to cancel, and it is the evidence that the working epsilon is not
    sitting on a cliff.
    """
    results = {}
    for epsilon in epsilons:
        sweep = collect_forward_sweep(compiled, fixture, sample, epsilon)
        numerical = finite_difference(sweep, arm.scored, arm.loss_weights)
        fit = fit_one_scale(analytic, numerical)
        results[epsilon] = fit
        logger.info("[gradcheck]   eps=%-8g %s  spike_train_moved=%d",
                    epsilon, fit.line(), sweep.moved_spike_train)
    return results


def report_timebase_sweep(sample_size: int = NUM_HIDDEN,
                          points: tuple[tuple[int, float], ...] = TIMEBASE_SWEEP) -> dict:
    """Re-measure the residual floor against dt/tau_syn, at fixed trial length.

    This is the tool's own control. EventProp's gradient is the gradient of the
    continuous-time system and the finite difference is of the discrete-time simulation,
    so a residual that shrinks with dt is the discretisation and not the patch. If a
    future change makes the residual *stop* shrinking, this is where it shows.

    The last column carries a second, smaller result that belongs beside it. It is the
    same fit with the *final* forward timestep dropped from the finite difference and
    from nothing else, and below dt/tau_syn = 0.5 it comes out consistently better than
    including it: 1.22e-3 -> 7.2e-4 at dt 0.25, 9.2e-4 -> 6.2e-4 at dt 0.125. So the
    analytic gradient behaves as though the last loss term contributes slightly less than
    the simulated loss says it does. The effect is well inside the residual floor, it
    does not grow under refinement, and it is measured here on the **stock** compiler,
    which has no guard and no patch at all, so it is upstream EventProp and not anything
    this project wrote. It is printed rather than left out because it is the one
    disagreement in the whole report that is not accounted for.
    """
    results = {}
    for timesteps, dt in points:
        fixture = build_fixture(timesteps=timesteps, dt=dt)
        arm = Arm(f"timebase_{timesteps}", stock=True)
        compiled = compile_arm(fixture, arm)
        with compiled:
            measurement = measure(compiled, fixture)
            sample = live_sample(measurement)[:sample_size]
            sweep = collect_forward_sweep(compiled, fixture, sample, DEFAULT_EPSILON)
        analytic = measurement.gradient[sample]
        scored = scored_timesteps(timesteps, dt, None)
        fit = fit_one_scale(analytic, finite_difference(sweep, scored))
        without_last = fit_one_scale(analytic, finite_difference(sweep, scored[:-1]))
        results[(timesteps, dt)] = (fit, without_last)
        logger.info("[gradcheck]   dt=%-6g T=%-3d dt/tau_syn=%-5.2f %s  spikes=%-5d "
                    "drop_last=%.3e", dt, timesteps, dt / TAU_SYN, fit.line(),
                    int(measurement.spike_counts.sum()),
                    without_last.relative_residual)
    return results


def report_negative_control(fixture: Fixture, sweep: ForwardSweep,
                            sample: np.ndarray,
                            stock_gradient: np.ndarray) -> bool:
    """Break the guard on purpose and require the checks above to notice.

    D74 is a record of a test suite that was green while proving nothing, so a harness
    that has only ever been seen to pass is not evidence. This injects the specific fault
    the guard was written to avoid: mlGeNN 3.x's own semantics, which read the window on
    the kernel clock rather than on ``backT`` and therefore score one timestep early at
    both ends (``loss_shaping.scored_timesteps`` sets this out in full). The compiler is
    built on the shifted window while the fit is taken against the window that was asked
    for, which is exactly what an off-by-one would look like from outside.

    Both faults have to be rejected, and it is worth knowing *which* check rejects which.
    Measured: the narrow fault still fits its claimed set at 4.9e-3, comfortably inside
    check 1's limit of 2e-2, so check 1 alone would pass it. Check 2 catches it, because
    the separation from a whole-trial loss collapses from x9.2 to x2.6. The full-trial
    fault is caught by check 3 instead, and caught absolutely, because a gradient missing
    one timestep cannot be bit identical to the stock compiler's.
    """
    dt, trial_end = fixture.dt, TIMESTEPS * fixture.dt
    narrow_start = (1.0 - NARROW_FRACTION) * trial_end
    whole = np.arange(TIMESTEPS)
    rejected = True

    for label, claimed_window, built_window in (
            ("narrow", (narrow_start, trial_end), (narrow_start - dt, trial_end - dt)),
            ("full", (0.0, trial_end), (0.0, trial_end - dt))):
        built = Arm(f"negative_{label}", window=built_window)
        compiled = compile_arm(fixture, built)
        with compiled:
            faulty = measure(compiled, fixture)
        analytic = faulty.gradient[sample]
        claimed = Arm(f"negative_{label}_claimed", window=claimed_window)

        as_claimed = fit_one_scale(analytic, finite_difference(sweep, claimed.scored))
        as_built = fit_one_scale(analytic, finite_difference(sweep, built.scored))
        against_whole = fit_one_scale(analytic, finite_difference(sweep, whole))
        separation = (against_whole.relative_residual / as_claimed.relative_residual)
        identical = bool(np.array_equal(analytic, stock_gradient))

        #a fault is only rejected if one of the three checks actually trips on it
        one = (as_claimed.correlation >= MIN_CORRELATION
               and as_claimed.relative_residual <= MAX_RELATIVE_RESIDUAL)
        #check 2 compares an arm against a whole-trial loss, which is a comparison with
        #itself for an arm claiming the whole trial. It would read as a failure at x1.0
        #and mean nothing, so it is None there rather than a spurious catch.
        two = (None if claimed_window == (0.0, trial_end)
               else separation >= MIN_WINDOW_SEPARATION)
        #check 3 only speaks to an arm claiming the whole trial: it is the assertion that
        #such an arm reproduces the stock gradient exactly. For the narrow fault there is
        #nothing for it to say, so it is None rather than a pass.
        three = identical if claimed_window == (0.0, trial_end) else None
        caught = [name for name, passed in
                  (("1", one), ("2", two), ("3", three)) if passed is False]

        logger.info("[gradcheck]   %-7s guard built on %s, fitted as %s",
                    label, tuple(round(value, 4) for value in built_window),
                    tuple(round(value, 4) for value in claimed_window))
        logger.info("[gradcheck]   %-7s as claimed %s", "", as_claimed.line())
        logger.info("[gradcheck]   %-7s as built   %s", "", as_built.line())
        logger.info("[gradcheck]   %-7s separation x%.1f, bit identical to stock: %s, "
                    "rejected by check(s) %s", "", separation, identical,
                    ",".join(caught) if caught else "NONE")
        if not caught:
            rejected = False
            logger.error("[gradcheck] the %s fault passed every check, so a PASS from "
                         "this harness does not mean the guard is right", label)

    return rejected


def probe_spiking_weights(fixture: Fixture, compiled, count: int = 8,
                          epsilons: tuple[float, ...] = EPSILONS) -> dict:
    """Show the staircase that puts input->hidden weights out of reach of this method.

    For each sampled input weight and each epsilon, report how many of the twenty-four
    hidden neurons changed their spike count. Below the step that tips a spike across a
    timestep boundary the answer is zero and the loss does not move at all; above it the
    loss jumps. Printing the staircase is what turns "finite differences cannot see the
    spike-time term" from an assertion into a measurement.
    """
    from ml_genn.utils.network import get_underlying_conn

    weights = compiled.connection_populations[
        get_underlying_conn(fixture.connections["input_hidden"])].vars["g"]
    weights.pull_from_device()
    original = weights.view.flatten().copy()
    baseline = measure(compiled, fixture)
    chosen = np.linspace(0, original.size - 1, count, dtype=int)

    results = {}
    try:
        for epsilon in epsilons:
            moved, delta = 0, 0.0
            for index in chosen:
                traces = {}
                for sign in (1, -1):
                    weights.view.flat[:] = original
                    weights.view.flat[index] = original[index] + sign * epsilon
                    weights.push_to_device()
                    perturbed = measure(compiled, fixture)
                    traces[sign] = perturbed
                    if not np.array_equal(perturbed.spike_counts, baseline.spike_counts):
                        moved += 1
                delta = max(delta, float(np.max(np.abs(
                    traces[1].voltage.astype(np.float64)
                    - traces[-1].voltage.astype(np.float64)))))
            results[epsilon] = (moved, len(chosen) * 2, delta)
            logger.info("[gradcheck]   eps=%-8g spike_train_moved=%d/%d  "
                        "max|dV|=%.3e", epsilon, moved, len(chosen) * 2, delta)
    finally:
        weights.view.flat[:] = original
        weights.push_to_device()
    return results


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                        help="perturbation used for the main comparison")
    parser.add_argument("--epsilon-sweep", action="store_true",
                        help="fit the stock arm at every epsilon in EPSILONS")
    parser.add_argument("--timebase-sweep", action="store_true",
                        help="re-measure the residual floor against dt/tau_syn")
    parser.add_argument("--probe-spiking-weights", action="store_true",
                        help="show why input->hidden weights are not sampled")
    parser.add_argument("--negative-control", action="store_true",
                        help="break the guard on purpose and require the checks to notice")
    parser.add_argument("--all", action="store_true",
                        help="every sweep, probe and control, the full D74 report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    epsilon_sweep = args.epsilon_sweep or args.all
    timebase_sweep = args.timebase_sweep or args.all
    spiking_probe = args.probe_spiking_weights or args.all
    negative_control = args.negative_control or args.all

    arms = default_arms()
    fixture = build_fixture()
    logger.info("[gradcheck] fixture: %d input / %d hidden / 1 output, "
                "%d timesteps of dt=%g (trial %g), tau_syn=%g, "
                "tau_mem in [%.2f, %.2f], tau_mem_out=%g",
                NUM_INPUT, NUM_HIDDEN, TIMESTEPS, DT, TIMESTEPS * DT, TAU_SYN,
                fixture.hidden_tau_mem.min(), fixture.hidden_tau_mem.max(), TAU_MEM_OUT)
    logger.info("[gradcheck] dt/tau_syn = %.2f; %d input spikes; target %g -> %g",
                DT / TAU_SYN, fixture.input_spikes[0].size, TARGET_START, TARGET_END)

    #the forward sweep is collected once, on the stock compiler, because the patches
    #rewrite the backward drive only. check_forward_pass_is_arm_independent below is
    #what entitles the tool to do that.
    reference = arms[0]
    compiled = compile_arm(fixture, reference)
    with compiled:
        baseline = measure(compiled, fixture)
        sample = live_sample(baseline)
        logger.info("[gradcheck] output voltage %.4f -> %.4f, hidden spikes %d, "
                    "%d/%d sampled weights have a non-zero gradient, %d distinct",
                    baseline.voltage[0], baseline.voltage[-1],
                    int(baseline.spike_counts.sum()), sample.size, NUM_HIDDEN,
                    np.unique(baseline.gradient[sample]).size)
        started = time.time()
        sweep = collect_forward_sweep(compiled, fixture, sample, args.epsilon)
        logger.info("[gradcheck] forward sweep at eps=%g: %d runs in %.1fs, "
                    "spike train moved %d times",
                    args.epsilon, 2 * sample.size, time.time() - started,
                    sweep.moved_spike_train)

        if epsilon_sweep:
            logger.info("[gradcheck] epsilon sweep, arm %s", reference.name)
            report_epsilon_sweep(fixture, compiled, reference,
                                 baseline.gradient[sample], sample)
        if spiking_probe:
            logger.info("[gradcheck] input->hidden probe: the spike-time staircase")
            probe_spiking_weights(fixture, compiled)

    logger.info("[gradcheck] one scale fitted across %d weights per arm", sample.size)
    analytic, fits, voltages = run_arms(fixture, arms, sweep, sample)

    failures: list[str] = []

    if not check_forward_pass_is_arm_independent(voltages):
        failures.append("the arms do not share a forward pass, so one sweep cannot "
                        "serve all of them and every fit above is suspect")

    #check 1
    for name, fit in fits.items():
        if fit.correlation < MIN_CORRELATION or \
                fit.relative_residual > MAX_RELATIVE_RESIDUAL:
            failures.append(
                f"arm {name} does not admit one scale: r={fit.correlation:.6f}, "
                f"residual={fit.relative_residual:.3e}")

    #check 2a, the one the brief names: each windowed arm against a whole-trial loss
    logger.info("[gradcheck] window check: each windowed arm against a whole-trial loss")
    for name in ("window_narrow", "window_early", "window_middle",
                 "shaped_window_narrow"):
        arm = next(a for a in arms if a.name == name)
        whole = Arm(name, tau=arm.tau, window=None)
        wrong = fit_one_scale(
            analytic[name], finite_difference(sweep, whole.scored, whole.loss_weights))
        right = fits[name]
        separation = wrong.relative_residual / right.relative_residual
        logger.info("[gradcheck]   %-22s scored %s", name, right.line())
        logger.info("[gradcheck]   %-22s whole  %s", "", wrong.line())
        logger.info("[gradcheck]   %-22s separation x%.1f", "", separation)
        if separation < MIN_WINDOW_SEPARATION:
            failures.append(
                f"arm {name} fits a whole-trial finite difference nearly as well as a "
                f"windowed one (x{separation:.1f}), so nothing here shows the guard "
                f"gating the timesteps it claims to")

    #check 2b: the same evidence in full. A whole-trial control alone is weakest exactly
    #where the production window sits, because a window ending at the trial end scores
    #the timesteps that dominate the whole-trial loss anyway, and a fit with a free scale
    #absorbs most of what is left. Cross-fitting every arm against every scored set does
    #not have that blind spot: an unguarded drive would make each row constant.
    logger.info("[gradcheck] cross fit: relative residual of each arm's gradient "
                "against each arm's scored set, diagonal marked *")
    matrix = cross_fit(analytic, arms, sweep)
    logger.info("[gradcheck]   %-22s %s", "gradient \\ scored set",
                " ".join(f"{name:>16}" for name in CROSS_FIT_ARMS))
    for row in CROSS_FIT_ARMS:
        cells = []
        for column in CROSS_FIT_ARMS:
            mark = "*" if row == column else " "
            cells.append(f"{matrix[(row, column)].relative_residual:>15.3e}{mark}")
        logger.info("[gradcheck]   %-22s %s", row, " ".join(cells))
    for row in CROSS_FIT_ARMS:
        diagonal = matrix[(row, row)].relative_residual
        others = {column: matrix[(row, column)].relative_residual
                  for column in CROSS_FIT_ARMS if column != row}
        nearest, value = min(others.items(), key=lambda item: item[1])
        if value < MIN_WINDOW_SEPARATION * diagonal:
            failures.append(
                f"arm {row}'s gradient is explained almost as well by {nearest}'s "
                f"scored set ({value:.3e}) as by its own ({diagonal:.3e}), so the "
                f"gradient does not identify the timesteps the guard claims to score")

    #check 3
    logger.info("[gradcheck] equivalence check: window=None against a full-trial window")
    for left, right in (("stock", "window_full"),
                        ("shaped", "shaped_window_full")):
        worst = float(np.max(np.abs(analytic[left] - analytic[right])))
        magnitude = float(np.max(np.abs(analytic[left])))
        logger.info("[gradcheck]   %-22s vs %-22s max|diff|=%.3e (%.3e relative), "
                    "bit identical: %s", left, right, worst, worst / magnitude,
                    np.array_equal(analytic[left], analytic[right]))
        if worst > 1e-6 * magnitude:
            failures.append(
                f"{left} and {right} should be the same gradient and differ by "
                f"{worst:.3e} ({worst / magnitude:.3e} relative)")

    if negative_control:
        logger.info("[gradcheck] negative control: mlGeNN 3.x's one-step-early guard, "
                    "injected and required to fail")
        if not report_negative_control(fixture, sweep, sample, analytic["stock"]):
            failures.append(
                "a deliberately broken guard passed every check, so the PASS above is "
                "not evidence of anything")

    if timebase_sweep:
        logger.info("[gradcheck] timebase sweep: the residual floor against dt/tau_syn, "
                    "and the same fit with the final timestep dropped")
        report_timebase_sweep()

    if failures:
        for failure in failures:
            logger.error("[gradcheck] FAIL: %s", failure)
        logger.error("[gradcheck] %d of the four checks failed. The patch, the harness "
                     "or the fixture is wrong, and D74 stays open.", len(failures))
        return 1
    logger.info("[gradcheck] PASS: every arm admits one scale, the window separates "
                "from the whole trial, and the unwindowed and full-window gradients "
                "agree. D74's numerical check is discharged for this fixture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
