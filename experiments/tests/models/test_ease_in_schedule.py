"""The learning-rate ease-in must ramp once, not restart at every chunk.

mlGeNN shallow copies callbacks per ``train()`` call and restarts its own ``batch``
argument every epoch, so the obvious implementations both sawtooth. These tests pin
the behaviour that makes the ramp arrive.
"""

import pytest

pytest.importorskip("ml_genn", reason="needs the mlgenn env")

from models.snn import _make_ease_in_schedule_cls  # noqa: E402

TARGET = 0.001


class _FakeOptimiser:
    def __init__(self):
        self.alpha = TARGET


class _FakeNetwork:
    def __init__(self, optimiser):
        self.optimisers = [(optimiser, None)]


def _schedule(num_batches: int, state: dict):
    cls = _make_ease_in_schedule_cls()
    optimiser = _FakeOptimiser()
    callback = cls(TARGET, num_batches, state)
    callback.set_params(compiled_network=_FakeNetwork(optimiser))
    return callback, optimiser


@pytest.mark.gpu
def test_ramp_starts_low_and_reaches_target_exactly():
    state = {"batches": 0}
    callback, optimiser = _schedule(100, state)

    callback.on_batch_begin(0)
    assert optimiser.alpha == pytest.approx(TARGET / 1000.0)

    for batch in range(1, 100):
        callback.on_batch_begin(batch)
    assert optimiser.alpha == pytest.approx(TARGET, rel=1e-6)


@pytest.mark.gpu
def test_ramp_is_monotonic_and_never_exceeds_target():
    state = {"batches": 0}
    callback, optimiser = _schedule(50, state)
    seen = []
    for batch in range(120):
        callback.on_batch_begin(batch)
        seen.append(optimiser.alpha)
    assert seen == sorted(seen)
    assert max(seen) <= TARGET + 1e-12


@pytest.mark.gpu
def test_counter_survives_the_chunk_boundary():
    """A fresh callback sharing the state dict must resume, not restart."""
    state = {"batches": 0}
    first, optimiser_a = _schedule(100, state)
    for batch in range(60):
        first.on_batch_begin(batch)
    after_first_chunk = optimiser_a.alpha

    #new chunk: mlGeNN copies callbacks, and our loop builds a new one per chunk
    second, optimiser_b = _schedule(100, state)
    second.on_batch_begin(0)
    assert optimiser_b.alpha > after_first_chunk, (
        "the ramp restarted at the chunk boundary")
    assert state["batches"] == 61


@pytest.mark.gpu
def test_shallow_copy_of_the_callback_still_advances_the_shared_counter():
    """mlGeNN does ``copy(callback)``; the counter must not be per-copy state."""
    from copy import copy

    state = {"batches": 0}
    callback, optimiser = _schedule(100, state)
    duplicate = copy(callback)
    duplicate.on_batch_begin(0)
    duplicate.on_batch_begin(1)
    assert state["batches"] == 2, "the shallow copy kept its own counter"


@pytest.mark.gpu
def test_epoch_local_batch_argument_is_ignored():
    """Passing a batch index that restarts must not pull the rate back down."""
    state = {"batches": 0}
    callback, optimiser = _schedule(80, state)
    for _ in range(3):
        for batch in range(20):  #restarts at 0 each epoch, as mlGeNN does
            callback.on_batch_begin(batch)
    #60 real batches of an 80-batch ramp, so it must be well above the floor
    assert optimiser.alpha > 10 * (TARGET / 1000.0)
