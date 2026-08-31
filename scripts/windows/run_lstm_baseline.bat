@echo off
REM The LSTM baseline, which the proposal names and no sweep had ever run.
REM
REM Runs from the plain .venv, NOT the mlgenn env. The LSTM needs torch; the mlgenn env
REM has none and must not be given one, because pygenn there is ABI-locked to numpy
REM 2.5.1 and an unpinned install risks the whole environment. Adding lstm to sweep.py's
REM model list on 2026-08-24 made every arm train the SNN for ~180s and then die on
REM ModuleNotFoundError, so the two are kept apart deliberately.
REM
REM No GeNN, so no vcvarsall and no GPU. Safe to run alongside a sweep.
REM
REM Per (horizon, L, seed) rather than per encoding: the LSTM sees the feature matrix,
REM not the spike encoding, so one run covers every encoding arm at that cell. The DM
REM comparison against the SNN is recomputed offline from predictions.csv, which is how
REM D81 already handles cross-run tests.
REM
REM usage: run_lstm_baseline.bat weekly rv_fwd vix9d 5,45
REM        run_lstm_baseline.bat monthly rv_fwd vix 21,45
setlocal enabledelayedexpansion
if "%~4"=="" echo usage: run_lstm_baseline.bat HORIZON TARGET IVLEG "L1,L2" & exit /b 1
call "%~dp0bootstrap.bat"
REM cmd splits arguments on commas as well as spaces, so an unquoted 5,45 arrives as
REM %4=5 and %5=45 and only L=5 ever runs. That happened on 2026-08-27 and cost half a
REM grid silently, so a fifth argument is now an error rather than something ignored.
if not "%~5"=="" (
  echo ERROR: quote the lengths argument, e.g. run_lstm_baseline.bat %~1 %~2 %~3 "%~4,%~5"
  echo        cmd splits on commas, so an unquoted list runs only its first entry.
  exit /b 1
)
cd /d "%REPO%"
REM %~4 is comma separated. Passing "21,45" through PowerShell reached the loop as a
REM single token on 2026-08-25 and only L=21 ran, so the commas are turned into spaces
REM first, which cmd's for always splits on.
set "LENGTHS=%~4"
set "LENGTHS=%LENGTHS:,= %"
for %%L in (%LENGTHS%) do (
  for %%S in (1 2 3 4 5 6 7 8) do (
    echo [lstm] %~1 L=%%L seed=%%S
    "%REPO%\.venv\Scripts\python.exe" "%REPO%\src\experiments\run_experiment.py" ^
      --models lstm,har_rv,garch,constant,persistence ^
      --horizon %~1 --target %~2 --iv-leg %~3 ^
      --input-window %%L --seed %%S --holdout-val > nul 2>&1
    if errorlevel 1 (echo   FAILED) else (echo   ok)
  )
)
echo [lstm] done
