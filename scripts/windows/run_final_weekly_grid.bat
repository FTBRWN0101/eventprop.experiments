@echo off
REM Final MSc weekly grid: 5 encodings x 2 algorithms x 2 windows x 8 seeds = 160 arms.
REM
REM Frozen input provenance: data-snapshots/final-weekly-2026-08-31.md.
REM Results and the manifest are deliberately new campaign artefacts. Do not point
REM this at a historic manifest: those runs predate the final code and policy fixes.
call "%~dp0bootstrap.bat"
call "%~dp0run_in_mlgenn.bat" "%REPO%\src\experiments\tools\sweep.py" ^
  --horizon weekly ^
  --target rv_fwd ^
  --iv-leg vix9d ^
  --campaign-id final_weekly_20260831 ^
  --manifest "%REPO%\artifacts\results\runs\final_weekly_20260831_manifest.json" ^
  --strict ^
  %*
exit /b %errorlevel%
