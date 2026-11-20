"""The train panel must exclude every forecast origin whose label crosses the split.

``rv_fwd`` at origin ``t`` is only observable once the return on ``t + horizon`` has
printed, so a training row within one horizon of the split carries test-period returns
inside its own label. The assembler drops those rows. These tests pin down that it drops
exactly them, and that ``label_end``, the metadata it decides with, never reaches the
written panel: ``experiments/core/dataset.py`` recognises targets by name, so a stray
column there would be served to every model as an input feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

SPLIT = pd.Timestamp("2020-01-01")
HORIZON_DAYS = 5


@pytest.fixture
def assembled(data_process_import):
    """A weekly panel assembled from a synthetic random-walk SPX and a flat VIX leg."""
    assembler_module = data_process_import("assembler")
    base = data_process_import("core.base")
    config_module = data_process_import("core.config")
    loaders = data_process_import("core.loaders")

    class _Prices(base.FeatureBuilder):
        """One required feature, so the panel has both a gate and a feature column."""

        name = "test_prices"
        requires = ("spx",)

        def build(self, raw):
            return pd.DataFrame({"spx_level": raw["spx"]})

    #spans the split with room either side, so no row is lost to the window edges
    dates = pd.bdate_range("2019-06-03", periods=200)
    rng = np.random.default_rng(11)
    spx = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(dates)))),
                    index=dates)
    raw = loaders.RawData(frame=pd.DataFrame({"spx": spx, "vix": 20.0}, index=dates))

    config = config_module.ProcessConfig.load(split_date=str(SPLIT.date()))
    assembler = assembler_module.DatasetAssembler(config, [_Prices(config)])
    return assembler.assemble(raw, base.Horizon("weekly", HORIZON_DAYS)), dates


def test_no_training_label_reaches_the_test_period(assembled):
    panel, dates = assembled
    assert len(panel.train) and len(panel.test)
    completion = dates[dates.get_indexer(panel.train.index) + HORIZON_DAYS]
    assert completion.max() < SPLIT


def test_exactly_the_crossing_origins_are_purged(assembled):
    """One horizon of origins, no more: the purge must not eat clean training rows."""
    panel, dates = assembled
    before_split = panel.full.index[panel.full.index < SPLIT]
    crossing = [d for d in before_split
                if dates[dates.get_loc(d) + HORIZON_DAYS] >= SPLIT]

    assert len(crossing) == HORIZON_DAYS
    assert panel.purged_train_rows == len(crossing)
    assert not set(crossing) & set(panel.train.index)
    assert len(panel.train) == len(before_split) - len(crossing)


def test_label_metadata_never_reaches_the_written_panel(assembled):
    panel, _ = assembled
    for name, frame in (("full", panel.full), ("train", panel.train),
                        ("test", panel.test)):
        assert "label_end" not in frame.columns, f"label_end leaked into {name}"
    assert "label_end" not in panel.feature_columns
    assert "label_end" not in panel.target_columns
