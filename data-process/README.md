# data-process

Turns the raw daily CSVs under [`data-save/<source>/`](../data-save) into aligned,
look-ahead-free **Variance Risk Premium (VRP)** panels - one per forecast horizon -
split into train/test and written to `data-save/processed/<horizon>/`.

The stage mirrors [`data-download`](../data-download): a source-agnostic framework in
`core/`, a registry of drop-in feature builders, and a thin orchestrator (`build_vrp.py`).

## Running

```bash
# Build every horizon (weekly = 5d, monthly = 21d) into data-save/processed/:
python data-process/build_vrp.py

# Override the split or build a subset of horizons:
python data-process/build_vrp.py --split-date 2018-01-01 --horizons weekly
```

Each horizon writes `full.csv`, `train.csv`, `test.csv` (date-indexed) to
`data-save/processed/<horizon>/`.

## What it computes

For a horizon of `N` trading days, at each date `t`:

- `rv_fwd`: annualised realised vol over `(t, t+N]` - the **label**
- `vrp`: `vix_t - rv_fwd_t` (absolute VRP, vol points)
- `rvrp`: `(vix_t - rv_fwd_t) / vix_t` (relative VRP)
- `rvrp_winsor`: `rvrp` clipped at train-only 1st/99th percentiles

Realised vol is the daily-returns estimator: `sqrt(252/N * sum r2) * 100`, on the same
annualised-percentage scale as the VIX. The VIX close at `t` is forward-looking, so
pairing it with **forward** realised vol keeps the construction free of look-ahead.

Features (`vix`, `vix_logret`, `rv_5`, `rv_21`, term-structure ratios, `vvix`, `skew`)
are all known at `t`.

## Conventions

- **Sample policy** - a row is kept only when every *required* feature and the target
  are present. *Optional* features (term structure, VVIX, SKEW) carry NaN before their
  history begins, so the full VIX sample (1990->) is retained while still exposing the
  richer columns where they exist.
- **Split** - rows before `split_date` (default `2020-01-01`) are train; rows from then
  through `test_end` (default `2025-12-31`) are test. The forward target's tail is
  trimmed so no row has an incomplete label.
- **Winsorisation** - `rvrp` bounds are estimated on the **training split only**, then
  applied everywhere, so the test period never informs the clip.

## Adding a feature

Drop a new module in `features/` with a `FeatureBuilder` subclass - set `name`,
`requires` (canonical raw-series names), `optional`, and implement `build(raw)` to
return a daily date-indexed frame. Registration and inclusion are automatic; no
existing file changes. To add a new raw input, add one `RawSeries` row in
`core/loaders.py`.
