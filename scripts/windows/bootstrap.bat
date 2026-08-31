@echo off
REM Shared paths for the Windows launchers.
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%\..\..") do set "REPO=%%~fI"
set "GENN_BUILD=%REPO%\artifacts\genn\live"
if not exist "%GENN_BUILD%" mkdir "%GENN_BUILD%"
