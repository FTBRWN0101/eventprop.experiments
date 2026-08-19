"""Loss shaping against a real compile. Needs the mlgenn env, run with -m gpu.

Everything here is the half the pure-numpy suite cannot reach: that the line we rewrite
is still the line mlGeNN emits, that the rewrite survives code generation, and that it
moves the gradient it claims to move.
"""
import numpy as np
import pytest

ml_genn = pytest.importorskip("ml_genn", reason="requires the mlgenn conda env")

pytestmark = pytest.mark.gpu

TIMESTEPS = 8
N_IN, N_HIDDEN = 4, 6
#EventProp defers each batch's backward pass into the NEXT batch's forward pass
#(mlGeNN issue #101), so a fixture with one batch per epoch never applies a
#gradient at all and every weight stays at its initial value however many epochs
#it runs. Measured here: 1 example moves nothing, 2 moves weights but leaves the
#shaped and unshaped arms within allclose tolerance, 4 and 8 separate them
#cleanly. 8 is chosen for margin rather than to sit on the boundary.
N_EXAMPLES = 8


def test_drive_source_still_matches_the_installed_compiler():
    """The whole patch is one string replacement. If upstream reflows that line the
    subclass raises at compile time, but this fails first and says why.
    """
    from inspect import getsource

    from ml_genn.compilers import EventPropCompiler

    from compilers.loss_shaping import drive_source

    source = getsource(EventPropCompiler.build_neuron_model)
    template = "drive = error / (TauM * num_batch * {self.dt * self.example_timesteps});"
    assert source.count(template) == 1
    #and the template renders to what drive_source builds
    assert drive_source(1.0, TIMESTEPS) == template.replace(
        "{self.dt * self.example_timesteps}", str(1.0 * TIMESTEPS))
    #the surrounding lines the pure-numpy suite pins a copy of
    for line in ("const float backT = {self.example_timesteps * self.dt} - t - dt;",
                 "scalar drive = 0.0;",
                 "const scalar error = RingOutputLossTerm[ringOffset + RingReadOffset];"):
        assert line in source, f"upstream reflowed {line!r}, re-derive the patch"


def test_per_trial_loss_is_rejected():
    from compilers.loss_shaping import make_loss_shaped_compiler_cls

    with pytest.raises(ValueError, match="per_timestep_loss"):
        make_loss_shaped_compiler_cls()(
            example_timesteps=TIMESTEPS, losses="mean_square_error",
            per_timestep_loss=False, loss_shaping_tau=float(TIMESTEPS))


def _fixture():
    """Deterministic tiny network plus N_EXAMPLES identical inputs. No RNG in the graph."""
    from ml_genn import Connection, Network, Population
    from ml_genn.compilers.event_prop_compiler import default_params
    from ml_genn.connectivity import Dense
    from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
    from ml_genn.synapses import Exponential
    from ml_genn.utils.data import preprocess_spikes

    rng = np.random.default_rng(0)
    network = Network(default_params)
    with network:
        input_pop = Population(SpikeInput(max_spikes=N_IN * TIMESTEPS), N_IN)
        hidden = Population(
            LeakyIntegrateFire(v_thresh=0.61, tau_mem=3.0, tau_refrac=None), N_HIDDEN)
        output = Population(LeakyIntegrate(tau_mem=20.0, readout="var"), 1)

        conns = [
            Connection(input_pop, hidden, Dense(rng.normal(0, 1.0, (N_IN, N_HIDDEN))),
                       Exponential(1.0)),
            Connection(hidden, hidden, Dense(rng.normal(0, 0.5, (N_HIDDEN, N_HIDDEN))),
                       Exponential(1.0)),
            Connection(hidden, output, Dense(rng.normal(0, 0.5, (N_HIDDEN, 1))),
                       Exponential(1.0)),
        ]

    times = np.tile(np.arange(TIMESTEPS, dtype=np.float32), N_IN)
    ids = np.repeat(np.arange(N_IN, dtype=np.int32), TIMESTEPS)
    #identical examples: the point is to give EventProp enough batches to land a
    #gradient, not to make the task varied
    x = {input_pop: [preprocess_spikes(times.copy(), ids.copy(), N_IN)
                     for _ in range(N_EXAMPLES)]}
    #a non-constant target, so the drive differs timestep to timestep
    y = {output: np.tile(np.linspace(-1.0, 1.0, TIMESTEPS).reshape(1, TIMESTEPS, 1),
                         (N_EXAMPLES, 1, 1))}
    return network, x, y, conns[-1]


def _train_and_read_weights(tau):
    """Train on the CPU backend; return hidden->output weights before and after.

    Returns both so callers can assert training actually moved something. Without
    that guard every assertion here passes vacuously the moment the fixture stops
    training, which is exactly how this file previously went green while proving
    nothing. tau None is the stock compiler.
    """
    from ml_genn.compilers import EventPropCompiler
    from ml_genn.optimisers import Adam
    from ml_genn.utils.network import get_underlying_conn

    from compilers.loss_shaping import make_loss_shaped_compiler_cls

    network, x, y, out_conn = _fixture()
    kwargs = dict(example_timesteps=TIMESTEPS, losses="mean_square_error",
                  optimiser=Adam(0.01), batch_size=1, dt=1.0,
                  per_timestep_loss=True, rng_seed=1,
                  backend="single_threaded_cpu")
    if tau is None:
        compiler = EventPropCompiler(**kwargs)
    else:
        compiler = make_loss_shaped_compiler_cls()(**kwargs, loss_shaping_tau=tau)

    compiled = compiler.compile(network)
    with compiled:
        g = compiled.connection_populations[get_underlying_conn(out_conn)].vars["g"]
        g.pull_from_device()
        before = g.view.flatten().copy()
        compiled.train(x, y, num_epochs=2, shuffle=False)
        g.pull_from_device()
        return before, g.view.flatten().copy()


def test_the_fixture_actually_trains():
    """The precondition both tests below depend on, asserted on its own so a
    regression here reads as a broken fixture rather than a broken patch.
    """
    before, after = _train_and_read_weights(None)
    assert not np.allclose(before, after), (
        "training moved no weights, so nothing downstream of this proves anything. "
        "The usual cause is too few batches: EventProp applies each batch's gradient "
        "during the next batch, so one batch per epoch never updates.")


def test_infinite_tau_reproduces_the_unshaped_compiler():
    """The identity case. scale 1.0 and rate 0.0 make the extra factor exactly 1.0, so
    this should be bit identical; the tolerance only allows for FP contraction.
    """
    before, shaped = _train_and_read_weights(float("inf"))
    _, plain = _train_and_read_weights(None)
    assert not np.allclose(before, plain), "fixture trained nothing, see test above"
    np.testing.assert_allclose(shaped, plain, rtol=1e-12)


def test_shaping_actually_moves_the_gradient():
    """Guards against the failure mode where the rewrite silently does nothing."""
    before, shaped = _train_and_read_weights(float(TIMESTEPS))
    _, plain = _train_and_read_weights(None)
    assert not np.allclose(before, plain), "fixture trained nothing, see test above"
    assert not np.allclose(shaped, plain), "shaped and unshaped weights are identical"
