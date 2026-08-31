@echo off
REM sets up VS for this process only, then runs run_experiment.py in the
REM mlgenn env. GeNN JIT-compiles CUDA at runtime, so the compiler has to be
REM on PATH.
REM usage: run_mlgenn.bat --models snn --encoding rate [...]
call "%~dp0bootstrap.bat"
call "%~dp0run_in_mlgenn.bat" "%REPO%\src\experiments\run_experiment.py" %*
exit /b %errorlevel%
