@echo off
REM Final MSc weekly grid: 5 encodings x 2 algorithms x 2 windows x 8 seeds = 160 arms.
REM Roughly 4 GPU-hours, resumable, one manifest.
REM
REM Results and the manifest are deliberately new campaign artefacts. Do not point this
REM at a historic manifest: those runs predate the code and policy fixes in D121-D124.
REM
REM TWO PHASES, SPLIT BY ALGORITHM, AND IT HAS TO BE (D124).
REM EventProp runs WITH --reset-synapses-between-trials, on the correctness argument in
REM D96 and D98: without it the forward initial conditions depend on the weights, so
REM EventProp's derivation does not hold and its gradient is not the gradient of the
REM loss. e-prop runs WITHOUT the flag, and must. EPropCompiler registers no ZeroOutPost
REM custom update group, so the callback raises inside the library, and config.py now
REM rejects the combination on the host before any GPU time is spent. e-prop is an
REM eligibility-trace approximation with no equivalent assumption to break, so the
REM asymmetry follows each rule's own requirements rather than convenience. D98 bounds
REM what it buys, at 58 paired weekly cells: median +0.0301 r2_rv, Wilcoxon p = 0.0276,
REM and the ceiling unmoved.
REM
REM ONE MANIFEST IS SAFE HERE, unlike run_monthly_all.bat, which needs one per phase.
REM The sweep's arm key is encoding_algorithm_L_seed, so it carries the only thing that
REM differs between these two phases. Each arm's own config.json records the reset flag
REM either way, so the protocol stays recoverable per arm.
REM
REM --allow-silent-neurons on BOTH phases, per D123. One abort policy, applied to every
REM arm: an absent cell answers nothing, and the silent fraction is recorded per arm so
REM the caveat is quantitative. aggregate.py already reports it as silent_final.
REM
REM Both phases run even if the first reports failures, so one transient arm cannot cost
REM the second phase its slot. Rerun with --retry-failed to pick stragglers up.
call "%~dp0bootstrap.bat"
set "RUNNER=%~dp0run_in_mlgenn.bat"
set "SWEEP=%REPO%\src\experiments\tools\sweep.py"
set "CAMPAIGN=final_weekly_20260831_v3_holdout"
set "MANIFEST=%REPO%\artifacts\results\runs\%CAMPAIGN%_manifest.json"
set "RC1=0"
set "RC2=0"

echo === phase 1 of 2: EventProp, trial reset on, 80 arms ===
cmd /c ""%RUNNER%" "%SWEEP%" --horizon weekly --target rv_fwd --iv-leg vix9d --campaign-id %CAMPAIGN% --manifest "%MANIFEST%" --algorithms eventprop --reset-synapses-between-trials --allow-silent-neurons --holdout-val --strict %*"
if errorlevel 1 set "RC1=1"

echo === phase 2 of 2: e-prop, no trial reset, 80 arms ===
cmd /c ""%RUNNER%" "%SWEEP%" --horizon weekly --target rv_fwd --iv-leg vix9d --campaign-id %CAMPAIGN% --manifest "%MANIFEST%" --algorithms eprop --allow-silent-neurons --holdout-val --strict %*"
if errorlevel 1 set "RC2=1"

if "%RC1%%RC2%"=="00" (
  echo === final weekly grid complete: 160 arms in "%MANIFEST%" ===
  exit /b 0
)
echo === final weekly grid finished with failures, phase1=%RC1% phase2=%RC2% ===
echo === rerun this script with --retry-failed to pick up the failed arms ===
exit /b 1
