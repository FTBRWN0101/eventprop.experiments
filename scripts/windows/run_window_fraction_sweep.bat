@echo off
REM Sweeps the D85 readout-window fraction at L=45, one fraction per invocation.
REM
REM L=45 only, and deliberately. At L=5 the trial is five timesteps, so the fraction is
REM far too coarse to sweep: 0.20, 0.25 and 0.30 all score the same single timestep and
REM anything below 0.20 scores none at all, which the config rejects rather than train
REM on a zero gradient. At L=45 the same fractions give 9, 11 and 13 scored timesteps.
REM
REM Each fraction gets its own manifest, because the sweep's arm key is
REM encoding_algorithm_L_seed and carries no fraction, so one manifest would treat the
REM arms as already recorded. Checkpoints do separate themselves, via the _rw suffix.
REM
REM usage: run_window_fraction_sweep.bat 0.15 [--dry-run] [--retry-failed]
if "%~1"=="" echo usage: run_window_fraction_sweep.bat FRACTION [extra args] & exit /b 1
REM %~dp0 must be captured BEFORE the shift loop: shift moves %0 as well as the
REM numbered arguments, so after collecting, %~dp0 is the last argument's directory and
REM the cd below silently lands somewhere else. That happened on 2026-08-26.
set "HERE=%~dp0"
set FRACTION=%~1
shift
REM collect EVERY remaining argument. "%1 %2 %3" forwarded only three and silently
REM truncated longer invocations mid-flag, which killed a seed extension on 2026-08-26
REM with "argument --seeds: expected one argument" after the first phase had run.
set "REST="
:collect_args
if "%~1"=="" goto args_done
set "REST=%REST% %~1"
shift
goto collect_args
:args_done
call "%HERE%bootstrap.bat"
call "%HERE%run_in_mlgenn.bat" "%REPO%\src\experiments\tools\sweep.py" ^
  --algorithms eventprop ^
  --windows 45 ^
  --readout-window %FRACTION% ^
  --manifest "%REPO%\artifacts\results\runs\window_fraction_%FRACTION%_manifest.json" ^
  %REST%
exit /b %errorlevel%
