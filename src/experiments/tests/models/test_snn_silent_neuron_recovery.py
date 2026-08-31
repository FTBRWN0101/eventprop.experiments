"""SilentNeuronRecovery correctness. Needs the mlgenn env, run with -m gpu."""
import numpy as np
import pytest

ml_genn = pytest.importorskip("ml_genn", reason="requires the mlgenn conda env")

pytestmark = pytest.mark.gpu


@pytest.fixture
def silent_neuron_recovery_cls():
    from models.snn import _make_silent_neuron_recovery_cls
    return _make_silent_neuron_recovery_cls()


def test_silent_neuron_gets_delta_g_bump_and_active_neuron_is_untouched(
        silent_neuron_recovery_cls):
    """A silent neuron's incoming weights gain exactly SILENT_NEURON_DELTA_G, and an
    active one is untouched. lr=0 so only the callback moves anything.
    """
    from ml_genn import Connection, Network, Population
    from ml_genn.callbacks import BatchProgressBar
    from ml_genn.compilers import EventPropCompiler
    from ml_genn.compilers.event_prop_compiler import default_params
    from ml_genn.connectivity import Dense
    from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
    from ml_genn.optimisers import Adam
    from ml_genn.synapses import Exponential
    from ml_genn.utils.data import preprocess_spikes
    from ml_genn.utils.network import get_underlying_conn

    from models.snn import SILENT_NEURON_DELTA_G

    n_in, n_hidden, timesteps = 4, 6, 5
    silent_idx, active_idx = 0, 1

    rng = np.random.default_rng(0)
    w_in_hidden = rng.normal(0, 0.05, (n_in, n_hidden))
    w_in_hidden[:, silent_idx] = -10.0   #never crosses threshold
    w_in_hidden[:, active_idx] = 10.0    #fires every timestep
    w_hidden_hidden = np.zeros((n_hidden, n_hidden))  #isolate the input path
    w_hidden_out = rng.normal(0, 0.1, (n_hidden, 1))

    network = Network(default_params)
    with network:
        #total spikes across the batch, not per neuron
        input_pop = Population(SpikeInput(max_spikes=n_in * timesteps), n_in)
        hidden = Population(
            LeakyIntegrateFire(v_thresh=1.0, tau_mem=3.0, tau_refrac=None),
            n_hidden, record_spikes=True)
        output = Population(LeakyIntegrate(tau_mem=3.0, readout="var"), 1)

        Connection(input_pop, hidden, Dense(w_in_hidden), Exponential(1.0))
        Connection(hidden, hidden, Dense(w_hidden_hidden), Exponential(1.0))
        Connection(hidden, output, Dense(w_hidden_out), Exponential(1.0))

    #every input neuron spikes at every timestep
    batch_size = 1  #the cpu backend hard-errors above batch_size=1
    spike_times = np.tile(np.arange(timesteps, dtype=np.float32), n_in)
    spike_ids = np.repeat(np.arange(n_in, dtype=np.int32), timesteps)
    input_data = [preprocess_spikes(spike_times.copy(), spike_ids.copy(), n_in)
                 for _ in range(batch_size)]
    y_full = np.zeros((batch_size, timesteps, 1), dtype=np.float32)

    compiler = EventPropCompiler(
        example_timesteps=timesteps, losses="mean_square_error",
        optimiser=Adam(0.0),  #lr=0 isolates the recovery bump from gradient descent
        batch_size=batch_size, dt=1.0, per_timestep_loss=True,
        backend="single_threaded_cpu")
    compiled_net = compiler.compile(network)
    compiled_net.num_recording_timesteps = timesteps

    SilentNeuronRecovery = silent_neuron_recovery_cls
    with compiled_net:
        in_hidden_conn = get_underlying_conn(hidden.incoming_connections[0]())
        g_var = compiled_net.connection_populations[in_hidden_conn].vars["g"]
        g_var.pull_from_device()
        before = g_var.view.reshape(n_in, n_hidden).copy()

        compiled_net.train(
            {input_pop: input_data}, {output: y_full},
            num_epochs=1, shuffle=False,
            callbacks=[BatchProgressBar(), SilentNeuronRecovery(hidden)])

        g_var.pull_from_device()
        after = g_var.view.reshape(n_in, n_hidden).copy()

    delta = after - before
    np.testing.assert_allclose(delta[:, silent_idx], SILENT_NEURON_DELTA_G, atol=1e-6)
    np.testing.assert_allclose(delta[:, active_idx], 0.0, atol=1e-6)
