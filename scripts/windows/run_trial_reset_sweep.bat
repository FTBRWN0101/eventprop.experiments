@echo off
REM D96: does clearing synaptic OutPost between trials change anything?
REM
REM EventProp schedules its ZeroOutPost update at the beginning of the LAST timestep of
REM each example rather than at trial begin, so synaptic drive carries from one trial
REM into the next. Measured carry-over is 0.55 on the output voltage against a 3.6
REM range. It is upstream, it is unmodified mlGeNN 2.3.1, and it has been in effect for
REM every SNN run this project has produced.
REM
REM Paired against the arms already on disk: same encodings, same seeds, same sequence
REM lengths, same everything, one changed variable. EventProp only, because the flag
REM patches an EventProp scheduling decision.
REM
REM Unwindowed on purpose. The question is whether the carry-over matters at all, and
REM asking it against the plain arms keeps it separable from the readout window.
REM
REM usage: run_trial_reset_sweep.bat [--dry-run] [--limit N] [--retry-failed]
call "%~dp0bootstrap.bat"
call "%~dp0run_in_mlgenn.bat" "%REPO%\src\experiments\tools\sweep.py" ^
  --algorithms eventprop ^
  --reset-synapses-between-trials ^
  --manifest "%REPO%\artifacts\results\runs\trial_reset_manifest.json" ^
  %*
exit /b %errorlevel%
