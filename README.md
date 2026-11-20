# Diagnosing a spiking neural network on a volatility forecasting benchmark

Recurrent spiking neural networks, trained with EventProp and e-prop in mlGeNN, forecast the S&P 500's realised volatility over the next week.
The code scores them against HAR-RV, GARCH(1,1), an LSTM and naive forecasts on the same held-out dates.
The data panel is leak-free, the train, validation and test periods are fixed in advance, and every checkpoint is chosen on validation data alone.

Built as my MSc project at the University of Sussex (2026).

## What it found

No spiking configuration matched the econometric benchmarks.
The primary configuration, an eight-seed ensemble of delta-coded EventProp networks with a 45-day input window, reached a skill score of 0.2045 against 0.4861 for HAR-RV and 0.5209 for GARCH(1,1), on 302 non-overlapping weekly test dates from 2020 to 2025 (Diebold-Mariano test against HAR-RV: p = 0.032).
The skill score is the out-of-sample R² of the volatility forecast against the training-sample mean, so zero means no better than that constant.
The best of the 20 configurations in the main grid (5 spike encodings, 2 learning rules and 2 input-window lengths, at 8 seeds each) scored 0.2295, below a persistence forecast at 0.3660.
The encoding the April 2026 proposal named as its hypothesis, adaptive delta coding, was among the weakest.
An LSTM given the same features, splits and stopping rule reached 0.4839, so the task admits forecasts at the level of the econometric models.
The spiking forecasts were also too flat: their standard deviation was between 4% and 50% of the target's, against 63% for HAR-RV.
Restricting the loss to the last tenth of each trial helped the two worst encodings, but not the primary configuration.
Choosing each checkpoint on validation data changed the epoch in 116 of 160 runs and removed most catastrophic outcomes, without improving the typical run.

## How the code is laid out

- `src/data_download/`: one fetcher per data source, run together by `fetch_all.py`.
- `src/data_process/`: builds the daily panel and the weekly and monthly forecasting panels (`build_vrp.py`).
- `src/experiments/models/`: the spiking network (`snn.py`) and the baselines: HAR-RV, GARCH, the LSTM and naive forecasts.
- `src/experiments/encoders/`: the spike encodings.
- `src/experiments/compilers/loss_shaping.py`: loss shaping, extended from classification to regression.
- `src/experiments/tools/`: sweeps, checkpoint selection, ensembles, aggregation and the EventProp gradient check.
- `src/experiments/run_experiment.py`: runs one configuration and scores every model on the test split.
- `scripts/windows/`: launchers for the GPU runs.
- [REPLICATE.md](REPLICATE.md): the commands that reproduce the final campaign.

## Run it

The CPU side covers the data, the baselines, the LSTM and the tests.
Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync
uv run python src/data_download/fetch_all.py
uv run python src/data_process/build_vrp.py
uv run python src/experiments/run_experiment.py
```

The first two scripts download the data into `data-save/` and build the forecasting panels.
The last fits HAR-RV and GARCH on the weekly task and prints their test metrics.

### Train the spiking networks

Training needs an NVIDIA GPU, CUDA, the MSVC build tools and a conda env with PyGeNN and mlGeNN.
The launchers in `scripts/windows/` set up the Visual Studio build environment, then run a script in that env.
They point at my machine, so first edit two paths at the top of `scripts/windows/run_in_mlgenn.ps1`:

- `$python`: your env's `python.exe`.
- `$vcvarsAll`: your Visual Studio install's `vcvarsall.bat`.

The `gpu` marker note in `pytest.ini` names the same env path, but only as a comment.

Do NOT upgrade NumPy in that env.
PyGeNN is ABI-locked to the NumPy it was built against.

The runs used Windows 11, an NVIDIA RTX A5000 Laptop GPU (16 GB), CUDA 13.3, Visual Studio 2026, Python 3.12, PyGeNN 5.4.0, mlGeNN 2.3.1 and NumPy 2.5.1.
[REPLICATE.md](REPLICATE.md) gives the command for every run.

## Test it

```bash
uv run pytest
```

Tests that need the data skip until you have downloaded it and built the panels.
Two tests also need the OptionMetrics surface from WRDS, and skip without it.
Tests marked `gpu` need the mlGeNN env and skip without it.

Tested on Linux and Windows with Python 3.12.

## Data

The repo includes no data.
`fetch_all.py` downloads the seven inputs the results use into `data-save/`:

- six Cboe volatility indices (VIX, VIX9D, VIX3M, VIX6M, VVIX and SKEW) from Cboe's website;
- S&P 500 daily prices from Yahoo Finance, through yfinance.

Neither needs a key, and each source's own terms of use apply.

The fetchers for CRSP and OptionMetrics need a WRDS licence, with credentials in `src/data_download/.env` (copy `.env.example`).
They also need the `wrds` package, which the lock leaves out because it pins pandas below 2.3; install it with `uv pip install wrds`.
No reported result uses them, and `fetch_all.py` skips any source it has no credentials for.

The split dates are fixed in code (train to 2016, validation 2017 to 2019, test 2020 to 2025), so a fresh download covers the same test period.

## Write-up

The full write-up isn't published here.
If you'd like to read it, email me at ftbrwn01@gmail.com.

## Acknowledgements

Thanks to my supervisor, Professor Thomas Nowotny, for his guidance throughout the project.

## Licence

The code is under the Apache License 2.0; see [LICENSE](LICENSE).
Copyright 2026 Findlay Brown.

## Citation

To cite this code, use **Cite this repository** on the GitHub page, which reads [CITATION.cff](CITATION.cff).
The work builds on:

- Bellec, G. et al. (2020). A solution to the learning dilemma for recurrent networks of spiking neurons. *Nature Communications* 11: 3625.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics* 31(3): 307-327.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics* 7(2): 174-196.
- Knight, J. C., Komissarov, A. and Nowotny, T. (2021). PyGeNN: a Python library for GPU-enhanced neural networks. *Frontiers in Neuroinformatics* 15: 659005.
- Nowotny, T., Turner, J. P. and Knight, J. C. (2025). Loss shaping enhances exact gradient learning with Eventprop in spiking neural networks. *Neuromorphic Computing and Engineering* 5: 014001.
- Turner, J. P., Knight, J. C., Subramoney, A. and Nowotny, T. (2022). mlGeNN: accelerating SNN inference using GPU-enabled neural networks. *Neuromorphic Computing and Engineering* 2(2): 024002.
- Wunderlich, T. C. and Pehle, C. (2021). Event-based backpropagation can compute exact gradients for spiking neural networks. *Scientific Reports* 11: 12829.
