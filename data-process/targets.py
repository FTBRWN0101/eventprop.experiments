"""VRP targets: the implied-vol leg minus forward realised vol."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from core.base import Horizon
from core.loaders import RawData
from core.volatility import forward_realised_vol

if TYPE_CHECKING:
    from core.config import ProcessConfig

#shared forward realised-vol label
SHARED_TARGET: str = "rv_fwd"


class TargetBuilder:
    """Builds the VRP target columns for one horizon from the raw panel."""

    def __init__(self, config: "ProcessConfig") -> None:
        self.config = config

    def build(self, raw: RawData, horizon: Horizon) -> pd.DataFrame:
        """Return ``rv_fwd`` plus ``iv_/vrp_/rvrp_`` columns for each configured leg."""
        #not inference-ready: rows with a NaN rv_fwd get dropped
        rv_fwd = forward_realised_vol(raw["spx"], horizon.days, self.config.annualisation)
        columns: dict[str, pd.Series] = {SHARED_TARGET: rv_fwd}

        window = self.config.rvrp_smooth_window
        for leg in self.config.legs_for(horizon):
            if leg not in raw.columns:
                continue
            iv = raw[leg]
            vrp = iv - rv_fwd
            #min_periods=1: tolerate isolated gaps in the leg
            denom = iv.rolling(window, min_periods=1).mean()
            columns[f"iv_{leg}"] = iv
            columns[f"vrp_{leg}"] = vrp
            columns[f"rvrp_{leg}"] = vrp / denom
        return pd.DataFrame(columns)
