@echo off
REM The monthly programme, in one sequential pass. Roughly 13 GPU-hours.
REM
REM EACH PHASE RUNS IN A FRESH cmd VIA "cmd /c", NOT "call". This keeps each long
REM GeNN phase isolated from the preceding process and preserves the original batch
REM boundary that prevented toolchain-environment accumulation from corrupting phase 3.
REM
REM e-prop runs WITHOUT --reset-synapses-between-trials, and must. EPropCompiler
REM registers no ZeroOutPost custom update group, so the callback raises inside the
REM library. All 80 e-prop arms died that way on 2026-08-24 before config.py grew the
REM guard that now rejects the combination on the host. This matches weekly, where the
REM e-prop arms carry no flag either.
REM
REM EventProp runs WITH the flag, on the correctness argument in D96 and D98.
REM
REM Phase 1a  monthly unwindowed, EventProp, flagged     80 arms   ~3.3 h
REM Phase 1b  monthly unwindowed, e-prop, unflagged      80 arms   ~3.3 h
REM Phase 2   monthly windowed, EventProp, f = 0.10      80 arms   ~3.3 h
REM Phase 3   monthly windowed, EventProp, f = 0.20      80 arms   ~3.3 h
REM
REM Separate manifests throughout: the sweep's arm key is encoding_algorithm_L_seed and
REM carries neither horizon, fraction nor flag. Resumable, so rerunning steps over
REM completed arms. Add --retry-failed to pick up transient error:4551 failures.
call "%~dp0bootstrap.bat"
set "SWEEP=%~dp0run_monthly_sweep.bat"
set "RESULTS=%REPO%\artifacts\results\runs"
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "RUN_ID=%%I"
set "LOGS=%REPO%\logs\monthly\%RUN_ID%"
mkdir "%LOGS%"
cd /d "%REPO%"

echo === phase 1a of 4: monthly unwindowed, EventProp, with the trial reset ===
cmd /c ""%SWEEP%" --reset-synapses-between-trials --algorithms eventprop ^
  --manifest "%RESULTS%\monthly_rst_manifest.json" --strict" > "%LOGS%\monthly_unwindowed_eventprop.log" 2>&1
if errorlevel 1 goto :failed

echo === phase 1b of 4: monthly unwindowed, e-prop, no reset ===
cmd /c ""%SWEEP%" --algorithms eprop ^
  --manifest "%RESULTS%\monthly_eprop_manifest.json" --strict" > "%LOGS%\monthly_unwindowed_eprop.log" 2>&1
if errorlevel 1 goto :failed

echo === phase 2 of 4: monthly windowed, EventProp, fraction 0.10 ===
cmd /c ""%SWEEP%" --reset-synapses-between-trials --algorithms eventprop ^
  --readout-window 0.10 ^
  --manifest "%RESULTS%\monthly_rst_window_0.10_manifest.json" --strict" > "%LOGS%\monthly_window_0.10.log" 2>&1
if errorlevel 1 goto :failed

echo === phase 3 of 4: monthly windowed, EventProp, fraction 0.20 ===
cmd /c ""%SWEEP%" --reset-synapses-between-trials --algorithms eventprop ^
  --readout-window 0.20 ^
  --manifest "%RESULTS%\monthly_rst_window_0.20_manifest.json" --strict" > "%LOGS%\monthly_window_0.20.log" 2>&1
if errorlevel 1 goto :failed

echo === monthly programme complete ===
exit /b 0

:failed
echo === monthly programme stopped: see "%LOGS%" for the failed phase ===
exit /b 1
