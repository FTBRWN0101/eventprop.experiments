@echo off
REM Materialise validation-selected forecasts, rate-code draw averages and final figures.
REM This reads saved checkpoints only. It never retrains or alters a raw campaign run.
call "%~dp0bootstrap.bat"
set "RUNNER=%~dp0run_in_mlgenn.bat"
set "MAIN=final_weekly_20260831_v3_holdout"
set "WINDOW=final_weekly_20260831_v3_window_f010_L45"

call "%RUNNER%" "%REPO%\src\experiments\tools\materialize_selection.py" ^
  --campaign-id %MAIN% --campaign-id %WINDOW% ^
  --selection-metric val_r2 --rate-draws 5
if errorlevel 1 exit /b %errorlevel%

call "%RUNNER%" "%REPO%\dissertation\make_final_results.py" ^
  --main-campaign %MAIN% --window-campaign %WINDOW%
exit /b %errorlevel%
