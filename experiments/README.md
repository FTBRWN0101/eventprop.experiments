# experiments

Trains and evaluates VRP/rVRP forecasters on the processed panels from
[`data-process`](../data-process) (`data-save/processed/<horizon>/`). Baselines here run in
the pip `.venv`; the mlGeNN SNN runs in a dedicated conda env, added in later steps.

## Running (baselines - `.venv`)

```bash
# Primary weekly task (VIX9D leg), default models har_rv + garch:
python experiments/run_experiment.py

# Sweep the axes:
python experiments/run_experiment.py --horizon monthly --target rvrp --iv-leg vix
python experiments/run_experiment.py --feature-set price-only --test-sampling daily
```

Prints a table of MSE / MAE / directional accuracy / QLIKE / Mincer-Zarnowitz R2, plus a
Diebold-Mariano stat and p-value vs. HAR-RV.

## How targets work

- **RV-based baselines** (HAR-RV, GARCH) forecast forward realised vol `rv_fwd`, then
  convert with the implied-vol leg known at `t`: `vrp_hat = iv_t - rv_fwd_hat`,
  `rvrp_hat = (iv_t - rv_fwd_hat) / denom_t`.
- **Direct models** (LSTM, SNN - later) predict the VRP/rVRP target straight.
- All return predictions in **target space**; evaluation inverts to `rv_fwd_hat` so QLIKE
  is always on the volatility forecast (it is undefined for the signed VRP).

## Conventions

- **Test sampling** defaults to `nonoverlap` (one window per horizon: weekly/monthly
  cadence) so overlapping, autocorrelated targets don't corrupt the Diebold-Mariano test.
- **Feature scaling** (for sequence models) is fit on train only.
- **VIX9D leg** is 2011+; the VIX leg carries the full 1990 sample.

## Adding a model

Drop a module in `models/` with a `Forecaster` subclass: set `name`, implement
`fit(data)` and `predict(data, split)` returning a target-space Series. It self-registers
and is picked up by `run_experiment.py` - no edits to existing files.
