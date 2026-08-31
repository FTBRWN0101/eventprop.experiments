@echo off
REM Run any script in the mlgenn env with a sanitized Visual Studio build environment.
REM
REM One generic runner instead of a bespoke wrapper per tool. The per-tool wrappers kept
REM going wrong in ways that cost real GPU time on 2026-08-26: run_mlgenn.bat is
REM hardcoded to run_experiment.py and cannot take another script, run_window_sweep.bat
REM hardcodes --readout-window 0.2 so it silently ran the wrong cell, and the fraction
REM wrapper forwarded only three arguments and then, once that was fixed, tripped over
REM `shift` moving %0 so that %~dp0 no longer pointed at the script.
REM
REM Two rules this one follows, both learned from those:
REM   * capture %~dp0 and %* BEFORE anything shifts, because shift moves %0 too
REM   * never hardcode an experiment parameter; the caller passes everything
REM
REM usage: run_in_mlgenn.bat src\experiments\tools\sweep.py --algorithms eventprop ...
setlocal
if "%~1"=="" echo usage: run_in_mlgenn.bat SCRIPT [args] & exit /b 1
set "HERE=%~dp0"
call "%HERE%bootstrap.bat"
set "SCRIPT=%~1"
shift
set "ARGS="
:collect_args
if "%~1"=="" goto args_collected
set "ARGS=%ARGS% "%~1""
shift
goto collect_args
:args_collected
if /I "%SCRIPT:~0,12%"=="experiments\" set "SCRIPT=src\%SCRIPT%"
if /I "%SCRIPT:~0,13%"=="data-process\" set "SCRIPT=src\data_process\%SCRIPT:~13%"
if /I "%SCRIPT:~0,14%"=="data-download\" set "SCRIPT=src\data_download\%SCRIPT:~14%"
if exist "%SCRIPT%" (
    for %%I in ("%SCRIPT%") do set "SCRIPT_PATH=%%~fI"
) else (
    set "SCRIPT_PATH=%REPO%\%SCRIPT%"
)
if not exist "%SCRIPT_PATH%" echo ERROR: script not found: %SCRIPT_PATH% & exit /b 1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%HERE%run_in_mlgenn.ps1" "%SCRIPT_PATH%" %ARGS%
exit /b %errorlevel%
