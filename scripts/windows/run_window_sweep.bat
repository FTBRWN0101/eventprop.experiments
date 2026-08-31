@echo off
REM sets up VS for this process only, then runs the D85 readout-window sweep in the
REM mlgenn env. GeNN JIT-compiles CUDA at runtime, so the compiler has to be on PATH.
REM Same pattern as run_drive_sweep.bat.
REM
REM EventProp only: the window patches EventProp's adjoint drive, so an eprop arm would
REM accept the flag and silently ignore it. The config raises rather than allow that,
REM and --algorithms keeps those arms out of the grid in the first place.
REM
REM usage: run_window_sweep.bat [--limit N] [--retry-failed] [--dry-run]
call "%~dp0bootstrap.bat"
call "%~dp0run_in_mlgenn.bat" "%REPO%\src\experiments\tools\sweep.py" ^
  --algorithms eventprop ^
  --readout-window 0.2 ^
  --manifest "%REPO%\artifacts\results\runs\window_sweep_manifest.json" ^
  %*
exit /b %errorlevel%
