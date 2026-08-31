@echo off
REM sets up VS for this process only, then runs the EventProp gradient check in the
REM mlgenn env. GeNN JIT-compiles the model at runtime, so the compiler has to be on
REM PATH. Same pattern as run_drive_sweep.bat.
REM usage: run_gradient_check.bat [--all] [--epsilon-sweep] [--timebase-sweep]
REM                               [--probe-spiking-weights] [--epsilon N]
call "%~dp0bootstrap.bat"
call "%~dp0run_in_mlgenn.bat" "%REPO%\src\experiments\tools\gradient_check.py" %*
exit /b %errorlevel%
