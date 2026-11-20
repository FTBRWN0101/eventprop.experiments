@echo off
REM Loss shaping at L=45: 5 encodings x EventProp x 8 seeds = 40 arms.
REM Roughly 1.5 GPU-hours, resumable, one manifest.
REM
REM WHY THIS RUNS AT ALL. Loss shaping is an objective of the study and the subject of
REM compilers/loss_shaping.py, but no arm of the final grid enabled it: all 200 configs
REM in final_weekly_20260831_v3_holdout and its window replication carry
REM loss_shaping=false. The extension was implemented and gradient-verified and never
REM exercised. This campaign supplies the missing empirical arm.
REM
REM PAIRED AGAINST THE MAIN GRID, NOT THE WINDOW REPLICATION. Every switch below matches
REM the EventProp phase of run_final_weekly_grid.bat: same seeds, same holdout, same
REM reset policy, same abort policy, no readout window. The only difference is
REM --loss-shaping, so the comparison against the ten EventProp cells already in
REM final_weekly_20260831_v3_holdout is paired within seed and differs in one factor.
REM
REM EVENTPROP ONLY, AND IT HAS TO BE. Loss shaping patches the EventProp adjoint drive.
REM config.py rejects the combination with e-prop on the host before any GPU time is
REM spent, so there is no e-prop phase here and cannot be one.
REM
REM TAU IS LEFT UNSET. snn.py then uses the sequence length, which is the trial length
REM at L=45. Setting it here would add a second swept axis to a one-factor comparison.
REM
REM --allow-silent-neurons per D123, matching the grid this is paired against. Spike
REM deletion is what shaping exists to prevent, so an abort policy that removed the arms
REM where it bites hardest would remove the evidence.
call "%~dp0bootstrap.bat"
set "RUNNER=%~dp0run_in_mlgenn.bat"
set "SWEEP=%REPO%\src\experiments\tools\sweep.py"
set "CAMPAIGN=final_weekly_20260901_shaping_L45"
set "MANIFEST=%REPO%\artifacts\results\runs\%CAMPAIGN%_manifest.json"

echo === loss shaping, EventProp, L=45, 40 arms ===
REM The encoding list is quoted: cmd treats a bare comma as an argument separator, so an
REM unquoted list arrives at argparse as five positional arguments and the sweep aborts.
cmd /c ""%RUNNER%" "%SWEEP%" --horizon weekly --target rv_fwd --iv-leg vix9d --campaign-id %CAMPAIGN% --manifest "%MANIFEST%" --encodings "rate,population,latency,delta,delta_adaptive" --algorithms eventprop --windows 45 --loss-shaping --reset-synapses-between-trials --allow-silent-neurons --holdout-val --strict %*"
if errorlevel 1 (
  echo === shaping campaign finished with failures ===
  echo === rerun this script with --retry-failed to pick up the failed arms ===
  exit /b 1
)
echo === shaping campaign complete: 40 arms in "%MANIFEST%" ===
exit /b 0
