@echo off
REM sets up VS for this process only, then runs run_experiment.py in the
REM mlgenn env. GeNN JIT-compiles CUDA at runtime, so the compiler has to be
REM on PATH.
REM usage: run_mlgenn.bat --models snn --encoding rate [...]
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 exit /b 1
cd /d "%~dp0"
"C:\Users\user\miniconda3\envs\mlgenn\python.exe" experiments\run_experiment.py %*
