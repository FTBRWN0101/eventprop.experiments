"""The raster inspector must read what the trainer writes, without the mlgenn env."""

import numpy as np
import pytest

from tools.show_raster import cell_rasters, draw, load, main, summarise


def _write(path, times, ids, example, num_neurons=8, timesteps=5):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, times=np.asarray(times, dtype=float),
                        ids=np.asarray(ids, dtype=int),
                        example=np.asarray(example, dtype=int),
                        num_neurons=num_neurons, timesteps=timesteps)
    return path


def test_round_trips_what_the_trainer_writes(tmp_path):
    from core.config import ExperimentConfig
    from models.snn import SnnForecaster

    model = SnnForecaster(ExperimentConfig.load())
    model._timesteps = 5
    model._checkpoint_dir = lambda create=False: tmp_path  #type: ignore[method-assign]
    written = model._save_raster(([np.array([0.0, 3.0])], [np.array([1, 6])]), epoch=10)

    raster = load(written)
    assert np.array_equal(raster["times"], [0.0, 3.0])
    assert np.array_equal(raster["ids"], [1, 6])


def test_summary_counts_silent_neurons(tmp_path):
    #only ids 0 and 1 ever fire, so 6 of 8 are silent
    path = _write(tmp_path / "rasters" / "epoch10.npz",
                  times=[0, 1, 2], ids=[0, 1, 0], example=[0, 0, 0])
    stats = summarise(load(path))
    assert stats["neurons"] == 8
    assert stats["spikes"] == 3
    assert stats["silent"] == 6
    assert stats["silent_fraction"] == pytest.approx(0.75)


def test_summary_rate_is_per_neuron_per_step(tmp_path):
    path = _write(tmp_path / "rasters" / "epoch10.npz",
                  times=[0, 1], ids=[0, 1], example=[0, 0],
                  num_neurons=2, timesteps=5)
    #2 spikes over 2 neurons and 1 example of 5 steps
    assert summarise(load(path))["mean_rate"] == pytest.approx(2 / 2 / 5)


def test_draw_marks_the_step_a_spike_lands_on(tmp_path):
    path = _write(tmp_path / "rasters" / "epoch10.npz",
                  times=[4], ids=[0], example=[0], num_neurons=1, timesteps=5)
    picture = draw(load(path), example=0)
    row = picture.splitlines()[0]
    cells = row.split("|")[1]
    assert cells[4] != " ", "the spike must appear at t=4"
    assert cells[:4].strip() == "", "no spike before t=4"


def test_draw_reports_an_example_with_no_spikes(tmp_path):
    path = _write(tmp_path / "rasters" / "epoch10.npz",
                  times=[0], ids=[0], example=[0])
    assert "no spikes recorded" in draw(load(path), example=3)


def test_cells_are_listed_in_epoch_order(tmp_path):
    cell = tmp_path / "a_cell"
    for epoch in (30, 10, 20):
        _write(cell / "rasters" / f"epoch{epoch}.npz",
               times=[0], ids=[0], example=[0])
    assert [p.stem for p in cell_rasters(cell)] == ["epoch10", "epoch20", "epoch30"]


def test_cell_without_rasters_lists_nothing(tmp_path):
    (tmp_path / "empty_cell").mkdir()
    assert cell_rasters(tmp_path / "empty_cell") == []


def test_main_lists_cells(tmp_path, capsys):
    _write(tmp_path / "a_cell" / "rasters" / "epoch10.npz",
           times=[0], ids=[0], example=[0])
    main(["--list", "--root", str(tmp_path)])
    assert "a_cell" in capsys.readouterr().out


def test_main_rejects_a_missing_epoch(tmp_path):
    _write(tmp_path / "a_cell" / "rasters" / "epoch10.npz",
           times=[0], ids=[0], example=[0])
    with pytest.raises(SystemExit, match="no raster for epoch 99"):
        main(["a_cell", "--epoch", "99", "--root", str(tmp_path)])
