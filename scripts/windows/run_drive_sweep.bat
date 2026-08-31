@echo off
REM sets up VS for this process only, then runs the drive-ratio sweep in the
REM mlgenn env. GeNN JIT-compiles CUDA at runtime, so the compiler has to be
REM on PATH. Same pattern as run_mlgenn.bat.
REM usage: run_drive_sweep.bat [--dry-run] [--limit N] [--retry-failed]
call "%~dp0bootstrap.bat"
call "%~dp0run_in_mlgenn.bat" "%REPO%\src\experiments\tools\drive_sweep.py" %*
exit /b %errorlevel%
