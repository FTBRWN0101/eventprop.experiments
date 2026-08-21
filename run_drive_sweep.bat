@echo off
REM sets up VS for this process only, then runs the drive-ratio sweep in the
REM mlgenn env. GeNN JIT-compiles CUDA at runtime, so the compiler has to be
REM on PATH. Same pattern as run_mlgenn.bat.
REM usage: run_drive_sweep.bat [--dry-run] [--limit N] [--retry-failed]
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 exit /b 1
cd /d "%~dp0"
"C:\Users\user\miniconda3\envs\mlgenn\python.exe" experiments\tools\drive_sweep.py %*
