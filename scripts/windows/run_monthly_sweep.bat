@echo off
REM The monthly-horizon mirror of the weekly encoding grid, closing the proposal's
REM G5 horizon-degradation gap. Weekly has 382 runs; monthly had 10, all on the old
REM vrp target, so nothing could be compared across horizons.
REM
REM Sequence lengths 21 and 45 mirror weekly's 5 and 45: the first is the horizon
REM itself (HORIZON_DAYS maps monthly to 21 trading days), the second is the long
REM window both horizons share, so L=45 is directly comparable across the two.
REM
REM --iv-leg vix is forced rather than chosen. The monthly panel carries no VIX9D
REM leg and config.py raises on the combination.
REM
REM Target rv_fwd, matching the weekly body of work so the horizon comparison
REM changes one variable rather than two.
REM
REM usage:
REM   run_monthly_sweep.bat                      unwindowed, both algorithms
REM   run_monthly_sweep.bat --readout-window 0.1 --algorithms eventprop
REM   run_monthly_sweep.bat --dry-run
call "%~dp0bootstrap.bat"
call "%~dp0run_in_mlgenn.bat" "%REPO%\src\experiments\tools\sweep.py" ^
  --horizon monthly ^
  --target rv_fwd ^
  --iv-leg vix ^
  --windows 21,45 ^
  --manifest "%REPO%\artifacts\results\runs\monthly_sweep_manifest.json" ^
  %*
exit /b %errorlevel%
