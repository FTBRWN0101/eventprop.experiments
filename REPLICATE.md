# Replicate the final campaign

These steps reproduce the dissertation's final weekly benchmark.
That is 160 main-grid runs, 40 readout-window runs, the validation-selected ensembles and the baselines they are scored against.
Run every command from the repo root in Windows `cmd`.

## What you need

- **CPU:** Python 3.12 and the dependencies from `uv sync` (see the [README](README.md)).
  This covers the data, the baselines, the LSTM, the tests and the aggregation.
- **GPU:** an NVIDIA GPU, CUDA, the MSVC build tools and a conda env with PyGeNN and mlGeNN.
  This is needed to train the spiking networks.
  Run steps 3 to 5 and the gradient check in that env, from an x64 Native Tools Command Prompt for VS with `DISTUTILS_USE_SDK=1` set (see the [README](README.md)).

Do NOT upgrade NumPy in the mlGeNN env.
PyGeNN is ABI-locked to it.

## Run

1. Download the data:

   ```bat
   uv run python src\data_download\fetch_all.py
   ```

2. Build the panels:

   ```bat
   uv run python src\data_process\build_vrp.py
   ```

3. Run the main grid, 160 runs and about 5 hours on the GPU, as two passes into one manifest:

   ```bat
   python src\experiments\tools\sweep.py --horizon weekly --target rv_fwd --iv-leg vix9d ^
     --campaign-id final_weekly_20260831_v3_holdout ^
     --manifest artifacts\results\runs\final_weekly_20260831_v3_holdout_manifest.json ^
     --algorithms eventprop --reset-synapses-between-trials ^
     --allow-silent-neurons --holdout-val --strict
   python src\experiments\tools\sweep.py --horizon weekly --target rv_fwd --iv-leg vix9d ^
     --campaign-id final_weekly_20260831_v3_holdout ^
     --manifest artifacts\results\runs\final_weekly_20260831_v3_holdout_manifest.json ^
     --algorithms eprop --allow-silent-neurons --holdout-val --strict
   ```

   EventProp resets the synapses between trials, which its exact gradient assumes.
   e-prop runs without the reset, because it does not support it and the config rejects the combination.
   Keep the main grid's own manifest: step 4 runs some of the same encoding, window and seed cells, and a shared manifest would skip them.

4. Run the readout-window replication, 40 runs and about 1.5 hours on the GPU:

   ```bat
   python src\experiments\tools\sweep.py ^
     --campaign-id final_weekly_20260831_v3_window_f010_L45 ^
     --manifest artifacts\results\runs\final_weekly_20260831_v3_window_f010_L45_manifest.json ^
     --encodings "rate,population,latency,delta,delta_adaptive" ^
     --algorithms eventprop --windows 45 --readout-window 0.1 ^
     --reset-synapses-between-trials --allow-silent-neurons --holdout-val
   ```

5. Choose each run's checkpoint on validation data and build the seed ensembles:

   ```bat
   python src\experiments\tools\materialize_selection.py ^
     --campaign-id final_weekly_20260831_v3_holdout ^
     --campaign-id final_weekly_20260831_v3_window_f010_L45 ^
     --selection-metric val_r2 --rate-draws 5
   ```

   This reads the saved checkpoints only.
   It never retrains or changes a training run.

6. Run the LSTM control, eight seeds at each input-window length, on the CPU.
   It needs PyTorch, so run it through `uv`, not in the mlGeNN env:

   ```bat
   for %L in (5 45) do for %S in (1 2 3 4 5 6 7 8) do uv run python src\experiments\run_experiment.py --models lstm,har_rv,garch,constant,persistence --horizon weekly --target rv_fwd --iv-leg vix9d --input-window %L --seed %S --holdout-val
   ```

   In a batch file, write `%%L` and `%%S` instead.

Steps 3 and 4 resume: rerunning skips the runs already in the manifest.
Add `--retry-failed` to pick up any that failed.

## Check it worked

```bat
uv run pytest
python src\experiments\tools\gradient_check.py
```

The first runs the test suite.
The second, in the mlGeNN env, checks the EventProp gradient against a finite difference on the GPU.

On the 302 test dates you should get these skill scores: GARCH(1,1) 0.5209, HAR-RV 0.4861, LSTM 0.4839, persistence 0.3660 and the primary SNN 0.2045.

Step 5 writes the ensembles to `artifacts\results\selected\<campaign>\final\ensembles\`, with `ensemble_index.csv` beside them mapping each ensemble to its configuration.
Each ensemble's `metrics.csv` holds the scores in its `r2_rv` column, with one row for the spiking ensemble and one for each baseline.
The primary SNN is the delta-coded EventProp ensemble of the readout-window campaign.
The LSTM figure is the mean of its eight seeds at an input window of 5; `src\experiments\tools\aggregate.py` reduces seeds to means.

If the scores differ, the panels differ.
Cboe and Yahoo can revise past values, so a fresh download may not match the August 2026 pull exactly.

## Not included

- Data, checkpoints (2.6 GB), GeNN build output and run directories.
  All of them are regenerable, and the code writes them under `data-save/` and `artifacts/`, which git ignores.
- OptionMetrics and CRSP extracts, which are licensed through WRDS.
  No reported result uses them: the panel build reads only the seven Cboe and Yahoo files.
- The figures and results tables, and the script that drew them.
