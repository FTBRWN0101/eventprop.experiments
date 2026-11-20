"""Safety contracts for the parallel selected-checkpoint materialiser."""

from pathlib import Path

import pytest

from tools import materialize_selection as tool


def _candidates(count: int):
    return [(Path(f"run-{index}"), None) for index in range(count)]


def test_shards_are_disjoint_and_cover_every_candidate_once():
    candidates = _candidates(17)
    shards = [tool._shard_candidates(candidates, index, 2) for index in range(2)]
    written = [candidate for shard in shards for candidate in shard]
    assert set(written) == set(candidates)
    assert len(written) == len(candidates)
    assert set(path for path, _ in shards[0]).isdisjoint(path for path, _ in shards[1])
    assert [path.name for path, _ in shards[0]] == [f"run-{index}" for index in range(0, 17, 2)]
    assert [path.name for path, _ in shards[1]] == [f"run-{index}" for index in range(1, 17, 2)]


@pytest.mark.parametrize(("index", "count"), [(-1, 2), (2, 2), (0, 0)])
def test_shard_bounds_are_rejected_before_any_gpu_work(index, count):
    with pytest.raises(ValueError):
        tool._shard_candidates(_candidates(2), index, count)


def test_shard_names_are_stable_and_non_colliding():
    assert tool._shard_stem(0, 2) == "shard-001-of-002"
    assert tool._shard_stem(1, 2) == "shard-002-of-002"


def test_ensemble_identity_ignores_the_per_run_identifier():
    common = {"horizon": "weekly", "encoding": "delta", "algorithm": "eventprop",
              "input_window": 45, "seed": 1}
    assert tool._ensemble_key({**common, "run_id": "one"}) == tool._ensemble_key(
        {**common, "seed": 2, "run_id": "two"})
