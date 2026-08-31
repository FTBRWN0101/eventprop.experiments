@echo off
REM The 2^4 factorial: readout window x trial reset x drive ratio x encoding, at
REM weekly / rv_fwd / L=45, EventProp, 8 seeds. 128 arms, roughly 5 GPU-hours.
REM
REM Every intervention in this project was measured with the others at their defaults,
REM so no interaction has ever been measured. The two that matter are window x reset,
REM never crossed at weekly, and window x drive, because D84 rejected recurrence
REM dominance using an objective D85 later showed to be misspecified.
REM
REM Needs the mlgenn env and the VS toolchain, like every GeNN sweep here.
REM usage: run_factorial.bat [--dry-run] [--limit N] [--retry-failed]
call "%~dp0bootstrap.bat"
call "%~dp0run_in_mlgenn.bat" "%REPO%\src\experiments\tools\factorial.py" %*
exit /b %errorlevel%
