"""Per-signal ablation must withhold exactly what it names, and nothing quietly.

A drop that matches no served column would report "this signal does not matter" for
a feature the arm never had, which is the one failure mode that would corrupt
Contribution #3.
"""

import pytest

from core.config import EXCLUDED_FEATURES, ExperimentConfig
from core.dataset import VrpDataset


def _columns(**overrides) -> list[str]:
    config = ExperimentConfig.load(**overrides)
    data = VrpDataset(config)
    return data.feature_columns(data.daily("train"))


def test_default_arm_excludes_skew_and_slope():
    columns = _columns()
    assert "skew" not in columns
    assert "ts_slope" not in columns


def test_restore_readmits_an_excluded_column():
    columns = _columns(restore_features=("skew", "ts_slope"))
    assert "skew" in columns
    assert "ts_slope" in columns


def test_restore_leaves_the_other_exclusions_alone():
    columns = _columns(restore_features=("skew",))
    assert "skew" in columns
    for name in set(EXCLUDED_FEATURES) - {"skew"}:
        assert name not in columns


def test_drop_removes_exactly_one_column():
    full = _columns()
    reduced = _columns(drop_features=("vix",))
    assert set(full) - set(reduced) == {"vix"}


def test_drop_wins_over_restore():
    columns = _columns(restore_features=("skew",), drop_features=("skew",))
    assert "skew" not in columns


def test_dropping_an_unserved_column_raises():
    with pytest.raises(ValueError, match="does not serve"):
        _columns(drop_features=("not_a_feature",))


def test_dropping_an_excluded_column_raises_unless_restored():
    #skew is not served by default, so naming it as a drop is a mistake worth catching
    with pytest.raises(ValueError, match="does not serve"):
        _columns(drop_features=("skew",))


def test_restoring_something_never_excluded_raises():
    with pytest.raises(ValueError, match="not in EXCLUDED_FEATURES"):
        ExperimentConfig.load(restore_features=("vix",))


def test_price_only_arm_still_honours_drop():
    full = _columns(feature_set="price-only")
    assert "rv_5" in full
    reduced = _columns(feature_set="price-only", drop_features=("rv_5",))
    assert set(full) - set(reduced) == {"rv_5"}


def test_ablatable_models_reject_fixed_input_baselines():
    import sys
    from pathlib import Path

    tools = Path(__file__).resolve().parents[2] / "tools"
    if str(tools.parent) not in sys.path:
        sys.path.insert(0, str(tools.parent))
    from tools.ablate_features import check_ablatable

    check_ablatable(["lstm"])
    with pytest.raises(ValueError, match="har_rv is exactly"):
        check_ablatable(["har_rv"])
